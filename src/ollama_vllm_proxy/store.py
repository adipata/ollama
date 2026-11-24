"""SQLite store.

One writer thread drains a bounded queue so the request path never waits on
disk; if the queue is full the row is dropped and counted. Small management
writes (keys, pod, profiles, secrets) are synchronous. Reads open their own
connection.
"""

from __future__ import annotations

import logging
import os
import queue
import sqlite3
import threading
import time
from contextlib import closing
from typing import Any, Iterable, Sequence

from .observe import requests_ddl

log = logging.getLogger("ovp.store")

_STOP = object()

# Ordered migrations; PRAGMA user_version holds how many have been applied.
MIGRATIONS: list[str] = [
    f"""
CREATE TABLE api_keys (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  key_prefix TEXT NOT NULL,
  key_sha256 TEXT NOT NULL UNIQUE,
  enabled INTEGER NOT NULL DEFAULT 1,
  created_at REAL NOT NULL,
  revoked_at REAL
);
CREATE UNIQUE INDEX api_keys_active_name ON api_keys(name) WHERE revoked_at IS NULL;
{requests_ddl()}
CREATE INDEX requests_ts ON requests(t_arrival_wall);
CREATE INDEX requests_key_ts ON requests(key_id, t_arrival_wall);
CREATE TABLE engine_samples (
  ts REAL NOT NULL,
  deployment_id TEXT,
  name TEXT NOT NULL,
  labels TEXT NOT NULL,
  value REAL
);
CREATE INDEX engine_samples_name_ts ON engine_samples(name, ts);
CREATE INDEX engine_samples_ts ON engine_samples(ts);
CREATE TABLE inventory (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  deployment_id TEXT,
  kind TEXT NOT NULL,
  data TEXT NOT NULL
);
CREATE INDEX inventory_kind_ts ON inventory(kind, ts);
CREATE TABLE pod (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  ssh_command TEXT NOT NULL,
  host TEXT NOT NULL,
  port INTEGER NOT NULL,
  user TEXT NOT NULL,
  key_path TEXT NOT NULL,
  fingerprint TEXT,
  state TEXT,
  updated_at REAL NOT NULL
);
CREATE TABLE profiles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  data TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE deployments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  profile_id INTEGER,
  profile_name TEXT,
  label TEXT,
  snapshot TEXT,
  plan TEXT,
  steps TEXT,
  state TEXT NOT NULL,
  started_at REAL NOT NULL,
  ended_at REAL
);
CREATE TABLE secrets (
  name TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
""",
]


class Store:
    def __init__(self, path: str, maxsize: int = 200_000):
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, mode=0o700, exist_ok=True)
        if not os.path.exists(path):
            # The database holds the vLLM key and the Hugging Face token in clear text.
            os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.failed = 0
        self._lock = threading.Lock()
        self._sync = self.connect()
        self._migrate()
        self._thread = threading.Thread(target=self._loop, name="sqlite-writer", daemon=True)
        self._started = False

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _migrate(self) -> None:
        with self._lock:
            version = self._sync.execute("PRAGMA user_version").fetchone()[0]
            for i in range(version, len(MIGRATIONS)):
                self._sync.executescript("BEGIN;\n" + MIGRATIONS[i] + f"\nPRAGMA user_version = {i + 1};\nCOMMIT;")
                log.info("applied schema migration %d", i + 1)

    # -- queued writes (never block the event loop) -------------------------

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def submit(self, sql: str, params: Sequence[Any] | Iterable[Sequence[Any]] = (), many: bool = False) -> None:
        try:
            self._q.put_nowait((sql, params, many))
        except queue.Full:
            self.dropped += 1

    def insert(self, table: str, row: dict[str, Any], replace: bool = False) -> None:
        cols = list(row)
        verb = "INSERT OR REPLACE" if replace else "INSERT"
        self.submit(
            f"{verb} INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            tuple(row[c] for c in cols),
        )

    @property
    def queued(self) -> int:
        return self._q.qsize()

    def flush(self) -> None:
        """Block until everything submitted so far is committed (call off-loop)."""
        if self._started:
            self._q.join()

    def stop(self) -> None:
        if self._started:
            try:
                self._q.put(_STOP, timeout=5)
            except queue.Full:
                pass
            self._thread.join(timeout=15)
        with self._lock:
            self._sync.close()

    def _loop(self) -> None:
        conn = self.connect()
        try:
            while True:
                batch = [self._q.get()]
                while len(batch) < 1000:
                    try:
                        batch.append(self._q.get_nowait())
                    except queue.Empty:
                        break
                stop = False
                try:
                    conn.execute("BEGIN")
                    for item in batch:
                        if item is _STOP:
                            stop = True
                            continue
                        sql, params, many = item
                        try:
                            if many:
                                conn.executemany(sql, params)
                            else:
                                conn.execute(sql, params)
                        except sqlite3.Error as exc:
                            self.failed += 1
                            log.warning("write failed: %s", exc)
                    conn.execute("COMMIT")
                except sqlite3.Error as exc:
                    self.failed += len(batch)
                    log.error("write batch failed: %s", exc)
                    try:
                        conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                finally:
                    for _ in batch:
                        self._q.task_done()
                if stop:
                    return
        finally:
            conn.close()

    # -- synchronous writes (management data) -------------------------------

    def write(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Execute one statement now; returns lastrowid."""
        with self._lock:
            cur = self._sync.execute(sql, params)
            return cur.lastrowid or 0

    def write_count(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Execute one statement now; returns the number of affected rows."""
        with self._lock:
            return self._sync.execute(sql, params).rowcount

    # -- reads ---------------------------------------------------------------

    def read(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with closing(self.connect()) as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def read_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        rows = self.read(sql, params)
        return rows[0] if rows else None

    # -- small key/value helpers ---------------------------------------------

    def get_state(self, key: str, default: str | None = None) -> str | None:
        row = self.read_one("SELECT value FROM state WHERE key = ?", (key,))
        return row["value"] if row else default

    def set_state(self, key: str, value: str) -> None:
        self.write("INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)", (key, value))

    def get_secret(self, name: str) -> str | None:
        row = self.read_one("SELECT value FROM secrets WHERE name = ?", (name,))
        return row["value"] if row else None

    def set_secret(self, name: str, value: str) -> None:
        self.write(
            "INSERT OR REPLACE INTO secrets (name, value, updated_at) VALUES (?, ?, ?)",
            (name, value, time.time()),
        )

    def delete_secret(self, name: str) -> None:
        self.write("DELETE FROM secrets WHERE name = ?", (name,))

    # -- retention -------------------------------------------------------------

    def prune(self, retention_days: int) -> None:
        if retention_days <= 0:
            return
        cutoff = time.time() - retention_days * 86400
        self.submit("DELETE FROM requests WHERE t_arrival_wall < ?", (cutoff,))
        self.submit("DELETE FROM engine_samples WHERE ts < ?", (cutoff,))
        # Keep the newest inventory row of each kind even when it is old.
        self.submit(
            "DELETE FROM inventory WHERE ts < ? AND id NOT IN (SELECT MAX(id) FROM inventory GROUP BY kind)",
            (cutoff,),
        )
