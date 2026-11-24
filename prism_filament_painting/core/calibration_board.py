"""Versioned recipes and measurement interchange for the four-spool front-lit board."""

from __future__ import annotations

import math
import re
from itertools import permutations


def make_board(materials: list[dict]) -> dict:
    """Slot zero is the opaque backing. The next three slots are overlay spools."""
    if not isinstance(materials, list) or len(materials) != 4:
        raise ValueError("provide exactly four spool records, backing first")
    ids = set()
    for material in materials:
        spool = material["id"]
        td = material["td_mm"]
        if not isinstance(spool, str) or not spool.strip() or spool in ids:
            raise ValueError("spool IDs must be distinct nonempty strings")
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", material["hex"]):
            raise ValueError("opaque colour must be #RRGGBB")
        if type(td) not in (int, float) or not math.isfinite(td) or td <= 0:
            raise ValueError("TD must be finite and positive")
        ids.add(spool)

    recipes = [("control", [])]
    overlays = [m["id"] for m in materials[1:]]
    for spool in overlays:
        for layers in (1, 2, 4, 8, 3):
            role = "holdout" if layers == 3 else "wedge"
            recipes.append((role, [[spool, layers]]))
    for lower, upper in permutations(overlays, 2):
        recipes.append(("mixed", [[lower, 2], [upper, 1]]))
        recipes.append(("holdout", [[lower, 1], [upper, 3]]))

    patches = []
    for i, (role, bands) in enumerate(recipes):
        row, col = divmod(i, 7)
        patches.append(
            {
                "id": f"r{row + 1}c{col + 1}",
                "role": role,
                "bands": bands,
                "xy_mm": [8 + 12 * col, 8 + 12 * row],
            }
        )
    return {
        "schema": "front-lit-board/v1",
        "materials": materials,
        "size_mm": [98, 62],
        "backing_mm": 1.60,
        "layer_mm": 0.08,
        "patch_mm": 10,
        "mark_centres_mm": [[4, 4], [94, 4], [94, 58], [4, 58]],
        "patches": patches,
    }


def validate_board(board: dict) -> None:
    # v1 is a fixed layout. Prevent a modified/misregistered recipe or split
    # from silently attaching measurements to the wrong plate.
    if board != make_board(board["materials"]):
        raise ValueError("unrecognised board layout, recipe or version")


def validate_database(database: dict) -> None:
    if database["schema"] != "front-lit-colours/v1":
        raise ValueError("unrecognised colour database version")
    validate_board(database["board"])
    capture = database["capture"]
    if capture["kind"] not in ("printed", "synthetic"):
        raise ValueError("capture kind must be printed or synthetic")
    if not re.fullmatch("[0-9a-f]{64}", capture["photo_sha256"]):
        raise ValueError("capture must identify the source photo")
    expected_ids = {patch["id"] for patch in database["board"]["patches"]}
    if set(database["measurements"]) != expected_ids:
        raise ValueError("measurements must cover exactly the board patch IDs")
    for reading in database["measurements"].values():
        rgb = reading["srgb8"]
        if len(rgb) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in rgb):
            raise ValueError("measured colour must be a three-channel sRGB8 value")
