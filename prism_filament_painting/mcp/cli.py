"""Start the read-only palette tools over stdio."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, help="An optional CSV or JSON spool list")
    args = parser.parse_args()
    try:
        from .server import create_server
    except ModuleNotFoundError as exc:
        if exc.name == "mcp":
            parser.exit(2, "Install prism-filament-painting[mcp] to use the stdio tools.\n")
        raise
    create_server(args.library).run(transport="stdio")
