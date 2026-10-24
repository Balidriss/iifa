"""``python -m iifa`` / ``iifa``: run the service."""

from __future__ import annotations

import argparse
import logging
import sys

from iifa import __version__
from iifa.config import ConfigError, load_settings


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "mcp":  # `iifa mcp ...` is `python -m iifa.mcp ...`
        from iifa.mcp.__main__ import main as mcp_main

        return mcp_main(argv[1:])
    parser = argparse.ArgumentParser(
        prog="iifa",
        description="context-budgeted memory service: the HTTP API with MCP at /mcp",
        epilog="`iifa mcp` runs the MCP server over stdio toward a running service "
        "(python -m iifa.mcp --help).",
    )
    parser.add_argument("--version", action="version", version=f"iifa {__version__}")
    parser.add_argument("--config", help="TOML configuration file (or set IIFA_CONFIG)")
    parser.add_argument("--host", help="bind address (default 127.0.0.1; anything else is on you)")
    parser.add_argument("--port", type=int, help="bind port (default 8765)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    try:
        settings = load_settings(config_file=args.config)
    except ConfigError as exc:
        print(f"iifa: {exc}", file=sys.stderr)
        return 2
    if not settings.tokens:
        print(
            "iifa: no client token configured (set IIFA_TOKEN or [[tokens]] in the config)",
            file=sys.stderr,
        )
        return 2

    import uvicorn

    from iifa.api.app import create_app

    host = args.host or settings.host
    port = args.port or settings.port
    if host not in ("127.0.0.1", "localhost", "::1"):
        logging.getLogger("iifa").warning(
            "binding to %s: the service is reachable beyond this machine", host
        )
    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
