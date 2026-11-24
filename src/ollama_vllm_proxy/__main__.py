"""Command line: serve (default), keys, admin-token."""

from __future__ import annotations

import argparse
import logging
import socket
import sys
from datetime import datetime

from .config import Settings, load_dotenv
from .keys import KeyError_, KeyStore, load_admin_token
from .store import Store


def _serve(settings: Settings) -> int:
    import uvicorn

    from .app import create_app

    # Fail with a clear message when a real Ollama (or another proxy) holds the port.
    probe = socket.socket(socket.AF_INET6 if ":" in settings.host else socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((settings.host, settings.port))
    except OSError as exc:
        print(
            f"cannot listen on {settings.host}:{settings.port}: {exc.strerror or exc}.\n"
            "If Ollama is running, stop it first (Linux: systemctl disable --now ollama).",
            file=sys.stderr,
        )
        return 1
    finally:
        probe.close()
    if settings.host not in ("127.0.0.1", "localhost", "::1"):
        logging.getLogger("ovp").warning("listening on %s: the proxy is reachable from other machines", settings.host)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=False,
        timeout_graceful_shutdown=10,
    )
    return 0


def _keys(args: argparse.Namespace, settings: Settings) -> int:
    store = Store(settings.db_path)
    keys = KeyStore(store)
    try:
        if args.action == "list":
            rows = keys.list()
            if not rows:
                print("no keys")
            for k in rows:
                created = datetime.fromtimestamp(k["created_at"]).strftime("%Y-%m-%d %H:%M")
                print(f"{k['id']:>4}  {k['name']:<32} {k['prefix']}…  {k['status']:<8} {created}")
        elif args.action == "create":
            key = keys.create(args.name)
            print(f"created key '{key['name']}' (id {key['id']}). Shown once:\n{key['key']}")
        else:
            key = keys.find(args.name)
            if args.action == "revoke":
                keys.revoke(key["id"])
            else:
                keys.update(key["id"], enabled=args.action == "enable")
            print(f"{args.action}d key '{key['name']}'" if args.action != "revoke" else f"revoked key '{key['name']}'")
    except KeyError_ as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.stop()
    return 0


def _admin_token(args: argparse.Namespace, settings: Settings) -> int:
    if settings.admin_token and args.action == "rotate":
        print("ADMIN_TOKEN is set in the environment; change it there.", file=sys.stderr)
        return 1
    token = load_admin_token(settings.admin_token_path, settings.admin_token, rotate=args.action == "rotate")
    print(token)
    if args.action == "rotate":
        print("restart the proxy for the new token to take effect", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="ollama-vllm-proxy", description=__doc__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="run the proxy (default)")
    p_keys = sub.add_parser("keys", help="manage API keys directly in the database")
    p_keys.add_argument("action", choices=["create", "list", "disable", "enable", "revoke"])
    p_keys.add_argument("name", nargs="?", help="key name (or id)")
    p_tok = sub.add_parser("admin-token", help="show or rotate the admin token")
    p_tok.add_argument("action", choices=["show", "rotate"])
    try:
        from .export import add_cli as _add_export_cli  # provided by the monitoring module

        _add_export_cli(sub)
    except ImportError:
        pass
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command in (None, "serve"):
        return _serve(settings)
    if args.command == "keys":
        if args.action != "list" and not args.name:
            parser.error("a key name is required")
        return _keys(args, settings)
    if args.command == "admin-token":
        return _admin_token(args, settings)
    if hasattr(args, "func"):
        return int(args.func(args, settings) or 0)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
