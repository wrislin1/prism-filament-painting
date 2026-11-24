"""Match a cell grid to flat recipes and render the selected run from its record."""

from dataclasses import asdict
from itertools import combinations
from math import comb

import numpy as np
from PIL import Image

from prism_filament_painting.core.color import hex_to_rgb

from .flat_regions import blurred_error, match_cells, match_regions, narrow_cells, validate_slots
from .flat_passes import count_changes, plan_passes, validate_costs
from prism_filament_painting.core.prediction import calibration_catalog

from .flat_recipes import Setup, recipe_table, setup_calibration

MAX_CELLS = 10_000
MAX_SOURCE_PIXELS = 16_000_000
# At four slots: unrestricted plus six middle-pair and six all-layer-pair plans.
MAX_PLANS = 13
PLATE_BUDGETS = {"u1": (75, 2), "a1": (90, 6)}



def _wire_filament(spool):
    return {("td" if key == "input_distance_mm" else key): value for key, value in asdict(spool).items()}


def _layer_plans(setup):
    slots = tuple(range(1, len(setup.slots) + 1))
    pair_count = comb(len(slots), 2) if len(slots) > 2 else 0
    if 1 + pair_count * (2 if setup.depth == 4 else 1) > MAX_PLANS:
        raise ValueError(f"reduce the number of loaded slots to fit {MAX_PLANS} candidate plans")
    plans = {"full": (slots,) * setup.depth}
    if len(slots) > 2:
        for a, b in combinations(slots, 2):
            pair = (a, b)
            if setup.depth == 4:
                plans[f"middle-{a}-{b}"] = (slots, pair, pair, slots)
            plans[f"all-{a}-{b}"] = (pair,) * setup.depth
    return plans


def make_run(image: Image.Image, setup: Setup, *, grid=None, plan="full",
             calibration=None, costs=None, regions=(), background_slots=None, dither_scope=None) -> dict:
    """Sample the entire image to (columns, rows). By default one pixel is one cell."""
    width, height = image.size if grid is None else grid
    if (type(width) is not int or type(height) is not int or min(width, height) < 1
            or width * height > MAX_CELLS):
        raise ValueError(f"use a coarser grid with 1..{MAX_CELLS} cells")
    if image.width * image.height > MAX_SOURCE_PIXELS:
        raise ValueError(f"resize the source image to at most {MAX_SOURCE_PIXELS} pixels")
    plans = _layer_plans(setup)
    if plan not in plans:
        raise ValueError(f"choose an available plan from {', '.join(plans)}")
    validate_costs(costs, len(setup.slots))

    rgba = image.convert("RGBA")
    if rgba.size != (width, height):
        rgba = rgba.resize((width, height), Image.Resampling.BOX)
    backing = hex_to_rgb(setup.slots[setup.backing_slot - 1].hex)
    target = Image.new("RGB", rgba.size, backing)
    target.paste(rgba, mask=rgba.getchannel("A"))
    calibration = calibration or setup_calibration(setup)
    table = recipe_table(setup, calibration)
    recipes = np.asarray([row.slots for row in table], dtype=np.uint8)
    eligible = {}
    for name, layers in plans.items():
        allowed = np.ones(len(recipes), dtype=bool)
        for z, slots in enumerate(layers):
            allowed &= np.isin(recipes[:, z], slots)
        eligible[name] = np.flatnonzero(allowed)
    extra = {}
    if regions or background_slots is not None or dither_scope is not None:
        if setup.cell_mm < 1:
            raise ValueError("use cells at least 1 mm wide for region or dithered prints")
        background = list(range(1, len(setup.slots) + 1)) if background_slots is None else background_slots
        validate_slots(background)
        if any(s > len(setup.slots) for slots in [background, *(r.slots for r in regions)] for s in slots):
            raise ValueError("region palette refers to an unloaded slot")
        ids, errors, info, direct = match_regions(target, table, recipes, eligible, regions,
                                                   background, dither_scope, plan)
        extra["regions"] = info
        planes = recipes[ids[plan]].reshape(height, width, -1).transpose(2, 0, 1)
        extra["printability"] = narrow_cells(planes)
    else:
        ids, errors = match_cells(target, table, eligible)

    label_errors = calibration.version != "uncalibrated" or bool(calibration_catalog())
    reports = {}
    for name in ids:
        layers = plans[name]
        used = recipes[np.unique(ids[name])]
        picture_usage = [tuple(int(s) for s in np.unique(layer)) for layer in used.T]
        usage = [(setup.backing_slot,)] * len(setup.backing_heights_mm) + picture_usage
        passes, cost = plan_passes(usage, start=setup.backing_slot, costs=costs)
        reports[name] = {
            "allowed_picture_slots": layers,
            "eligible_count": len(eligible[name]),
            "layer_usage": usage,
            "passes": passes,
            "change_count": count_changes(passes, start=setup.backing_slot),
            "pass_cost": cost,
            "mean_delta_e": float(np.mean(errors[name])),
            "p95_delta_e": float(np.percentile(errors[name], 95)),
        }
        if label_errors:
            reports[name]["prediction_sources"] = sorted({table[i].source for i in np.unique(ids[name])})
    if dither_scope is not None:
        occupied = recipes[np.unique(direct)]
        usage = [(setup.backing_slot,)] * len(setup.backing_heights_mm)
        usage += [tuple(map(int, np.unique(layer))) for layer in occupied.T]
        passes, _ = plan_passes(usage, start=setup.backing_slot, costs=costs)
        extra["dithering"] = dict(scope=dither_scope, direct_error=blurred_error(target, table, direct),
                                  dithered_error=blurred_error(target, table, ids[plan]),
                                  direct_changes=count_changes(passes, start=setup.backing_slot),
                                  dithered_changes=reports[plan]["change_count"])
    minutes, grams = PLATE_BUDGETS[setup.profile]
    return {
        **extra,
        "schema_version": 1,
        "geometry": {
            "columns": width, "rows": height, "cell_mm": setup.cell_mm,
            "width_mm": width * setup.cell_mm, "height_mm": height * setup.cell_mm,
            "backing_layer_heights_mm": setup.backing_heights_mm,
            "picture_layers": setup.depth, "picture_layer_height_mm": setup.layer_height_mm,
            "total_height_mm": sum(setup.backing_heights_mm) + setup.depth * setup.layer_height_mm,
            "orientation": "face-up", "nozzle_mm": 0.4, "line_width_mm": 0.45,
            "fill_percent": 100, "row_order": "top-to-bottom",
        },
        "profile": setup.profile, "backing_slot": setup.backing_slot,
        "plate_budget_100mm": {"minutes": minutes, "purge_g": grams},
        "slots": [{"slot": i, **_wire_filament(spool)} for i, spool in enumerate(setup.slots, 1)],
        "model": {"version": setup.model_version, "td_scale": setup.td_scale,
                  "backing_assumption": ("uncalibrated backing spool surface colour" if calibration.version == "uncalibrated" else "opaque backing spool"),
                  "viewing": setup.viewing, "calibration_version": calibration.version},
        "recipe_table": [asdict(row) for row in table],
        "cell_recipe_ids": ids[plan].reshape(height, width).tolist(),
        "selected_plan": plan, "plans": reports,
        "transition_costs": costs,  # None means unit cost per change.
    }


def preview(run: dict) -> Image.Image:
    """Render using just the saved run, including after a JSON round trip."""
    colours = np.asarray([row["rgb"] for row in run["recipe_table"]], dtype=np.uint8)
    return Image.fromarray(colours[np.asarray(run["cell_recipe_ids"])])
