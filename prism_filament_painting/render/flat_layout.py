"""Reconstruct the physical layout of a saved stage-one flat print."""

from itertools import accumulate
from math import isclose, isfinite

import numpy as np

from .flat_print import MAX_CELLS
from .flat_recipes import MAX_DEPTH, MAX_RECIPES


def layout(run):
    if run["schema_version"] != 1:
        raise ValueError("unsupported flat run schema")
    g = run["geometry"]
    rows, columns, depth = g["rows"], g["columns"], g["picture_layers"]
    if (any(type(n) is not int or n < 1 for n in (rows, columns, depth))
            or rows * columns > MAX_CELLS or depth > MAX_DEPTH):
        raise ValueError("invalid flat grid size or picture depth")
    if (g["orientation"], g["row_order"], g["fill_percent"]) != ("face-up", "top-to-bottom", 100):
        raise ValueError("expected a face-up, solid, top-to-bottom flat run")
    backing = g["backing_layer_heights_mm"]
    # Bound the expanded sidecar by the stage-one maximum deposition grid.
    if not backing or (len(backing) + depth) * rows * columns > MAX_CELLS * (2 + MAX_DEPTH):
        raise ValueError("missing backing or too many layer cells for the deposition map")
    heights = [*backing, *([g["picture_layer_height_mm"]] * depth)]
    pitch = g["cell_mm"]
    if any(not isfinite(v) or v <= 0 for v in (pitch, *heights, g["line_width_mm"])):
        raise ValueError("invalid flat print dimensions")
    if not all(isclose(actual, expected) for actual, expected in (
        (g["width_mm"], pitch * columns), (g["height_mm"], pitch * rows),
        (g["total_height_mm"], sum(heights)),
    )):
        raise ValueError("stored dimensions disagree with the cell grid or layer heights")

    slots = run["slots"]
    numbers = [row["slot"] for row in slots]
    if (not 1 <= len(slots) <= 4 or any(type(n) is not int for n in numbers)
            or numbers != list(range(1, len(slots) + 1))):
        raise ValueError("missing or out-of-order loaded slot")
    recipes = run["recipe_table"]
    if not 1 <= len(recipes) <= MAX_RECIPES:
        raise ValueError("invalid recipe table size")
    used_slots = [run["backing_slot"]]
    for recipe in recipes:
        if len(recipe["slots"]) != depth:
            raise ValueError("recipe depth differs from the picture layer count")
        used_slots.extend(recipe["slots"])
    if any(type(n) is not int or n not in numbers for n in used_slots):
        raise ValueError("recipe or backing refers to a missing loaded slot")
    ids = np.asarray(run["cell_recipe_ids"])
    if (ids.shape != (rows, columns) or ids.dtype.kind not in "iu"
            or ids.min() < 0 or ids.max() >= len(recipes)):
        raise ValueError("invalid cell recipe map")
    picture = np.asarray([r["slots"] for r in recipes], dtype=np.uint8)[ids].transpose(2, 0, 1)
    planes = np.concatenate((np.full((len(backing), rows, columns), run["backing_slot"],
                                     dtype=np.uint8), picture))
    return heights, planes


def expected_maps(run, heights, planes, origin):
    return {
        "schema_version": 1, "profile": run["profile"], "geometry": run["geometry"],
        "bed_origin_xy_mm": list(origin), "cell_recipe_ids": run["cell_recipe_ids"],
        "recipes_bottom_to_top": [r["slots"] for r in run["recipe_table"]],
        "slots": [{**s, "gcode_tool": s["slot"] - 1,
                   "has_part": bool(np.any(planes == s["slot"]))} for s in run["slots"]],
        "layers": [{"height_mm": h, "top_z_mm": z, "cell_slots": plane.tolist()}
                   for h, z, plane in zip(heights, accumulate(heights), planes)],
    }
