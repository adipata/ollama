"""Shared by the stub executables of the fake pod. Behaviour comes from files in ../state."""

import json
import os
import sys
import time
from pathlib import Path

STATE = Path(__file__).resolve().parent.parent / "state"


def read(name: str, default: str = "") -> str:
    try:
        return (STATE / name).read_text().strip()
    except OSError:
        return default


def number(name: str, default: float) -> float:
    try:
        return float(read(name, str(default)))
    except ValueError:
        return default


def record(name: str, **data) -> None:
    """Append what this stub was called with, for the tests to inspect."""
    data.setdefault("argv", sys.argv[1:])
    data.setdefault("env", dict(os.environ))
    data["ts"] = time.time()
    with open(STATE / f"{name}.jsonl", "a") as fh:
        fh.write(json.dumps(data) + "\n")
