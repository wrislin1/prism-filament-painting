"""Generate a multipart plate, extract a colour database, or fit a distance curve."""

import argparse
import hashlib
import json
from pathlib import Path

from prism_filament_painting.core.calibration_board import make_board
from prism_filament_painting.core.calibration_fit import fit_database
from .plate import write_plate


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None, *, install_directory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    board = commands.add_parser("board", help="write four STLs and board.json to a new directory")
    board.add_argument("materials", type=Path)
    board.add_argument("directory", type=Path)
    photo = commands.add_parser("extract", help="sample an sRGB photo using clicked mark centres")
    for key in ("board", "photo", "clicks", "database"):
        photo.add_argument(key, type=Path)
    photo.add_argument("--kind", required=True, choices=("synthetic", "printed"))
    fit = commands.add_parser("fit", help="score a candidate on the reserved stacks")
    fit.add_argument("database", type=Path)
    fit.add_argument("report", type=Path)
    install = commands.add_parser("install", help="fit and publish one dated capture for its four spool IDs")
    install.add_argument("database", type=Path)
    install.add_argument("directory", type=Path, **({"nargs": "?", "default": install_directory} if install_directory else {}))
    install.add_argument("--captured-on", required=True)
    install.add_argument("--valid-through", required=True)
    install.add_argument("--opaque-from-mm", type=float, help="thinnest independently verified opaque backing for this spool, in mm")
    args = parser.parse_args(argv)
    if args.command == "board":
        layout = make_board(read(args.materials))
        args.directory.mkdir()
        write_plate(layout, args.directory)
        save(args.directory / "board.json", layout)
    elif args.command == "extract":
        from PIL import Image
        from .photo import extract

        with args.photo.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            with Image.open(stream) as image:
                result = extract(
                    read(args.board),
                    image,
                    read(args.clicks),
                    source_kind=args.kind,
                    photo_sha256=digest,
                )
        save(args.database, result)
    elif args.command == "install":
        from prism_filament_painting.core.prediction import install_capture

        print(install_capture(args.directory, read(args.database), args.captured_on, args.valid_through,
                              opaque_from_mm=args.opaque_from_mm))
    else:
        save(args.report, fit_database(read(args.database)))


if __name__ == "__main__":
    main()
