"""Raster previews, masks and saved flat grid validation for the print editor."""

import base64

import numpy as np
from PIL import Image, ImageOps
from prism_filament_painting.core.color import rgb_array_to_lab
from prism_filament_painting.core.finishing import PrintOptions, finished_grid

from .studio_report import pixel_accuracy, studio_stack


def print_grid(image, state, base_factor):
    options = PrintOptions.parse(state.get("print_options", {}))
    grid = finished_grid(image, studio_stack(state), state["mesh"], options, base_factor)
    return {"width": grid.heights.shape[1], "height": grid.heights.shape[0], "pitch": grid.pitch,
            "downsample_factor": grid.factor,
            "cells": base64.b64encode(grid.heights.astype("uint8").tobytes()).decode("ascii")}


def sample_mask(mask_image, columns, rows):
    mask = np.asarray(mask_image.convert("L"))
    ys = (2 * np.arange(rows) + 1) * mask.shape[0] // (2 * rows)
    xs = (2 * np.arange(columns) + 1) * mask.shape[1] // (2 * columns)
    bits = mask[np.ix_(ys, xs)] >= 128
    if not bits.any():
        raise ValueError("No foreground cells found. Paint the region or use the whole plate.")
    return {"mask": ["".join("1" if cell else "0" for cell in row) for row in bits]}


def _validate_region_controls(options, run, rows, cols):
    regions = options.get("regions", [])
    background = options.get("background_slots")
    scope = options.get("dither_scope")
    if not isinstance(regions, list) or len(regions) > 8:
        raise ValueError("invalid saved regions")
    if scope is not None and (type(scope) is not int or not -1 <= scope <= len(regions)):
        raise ValueError("invalid saved dithering region")
    if scope != run.get("dithering", {}).get("scope"):
        raise ValueError("saved dithering differs from the accepted run")
    if not regions and background is None and scope is None:
        if "regions" in run:
            raise ValueError("missing saved region controls")
        return
    palettes = [[1, 2, 3, 4] if background is None else background, *(r["slots"] for r in regions)]
    if any(not isinstance(p, list) or not 1 <= len(p) <= 4 or len(set(p)) != len(p)
           or any(type(i) is not int or not 1 <= i <= 4 for i in p) for p in palettes):
        raise ValueError("invalid region palettes")
    owners = np.zeros((rows, cols), np.uint8)
    for i, region in enumerate(regions, 1):
        mask = region["mask"]
        if (not isinstance(region["name"], str) or not 1 <= len(region["name"]) <= 64
                or not isinstance(mask, list) or not 1 <= len(mask) <= 100 or not 1 <= len(mask[0]) <= 100
                or any(not isinstance(row, str) or len(row) != len(mask[0]) or set(row) - {"0", "1"} for row in mask)):
            raise ValueError("invalid saved region mask")
        ys = (2 * np.arange(rows) + 1) * len(mask) // (2 * rows)
        xs = (2 * np.arange(cols) + 1) * len(mask[0]) // (2 * cols)
        owners[np.array([list(row) for row in mask])[np.ix_(ys, xs)] == "1"] = i
    info = run["regions"]
    if palettes != info["palettes"] or owners.tolist() != info["cell_regions"]:
        raise ValueError("saved region controls disagree with the printed grid")
    for row, ids in zip(owners, run["cell_recipe_ids"]):
        if any(not set(run["recipe_table"][r]["slots"]).issubset(palettes[i]) for i, r in zip(row, ids)):
            raise ValueError("saved recipe uses a slot outside its region")


def validate_snapshot(state, valid_run_id):
    """Bound saved grids and recipe indices before rendering user-supplied project JSON."""
    try:
        result = state["flat"]
        if result["mode"] != "flat" or not valid_run_id(result["run_id"]):
            raise ValueError("invalid flat run id")
        run = result["run"]
        cells = run["cell_recipe_ids"]
        geometry = run["geometry"]
        rows, cols = geometry["rows"], geometry["columns"]
        if any(type(n) is not int or not 1 <= n <= 100 for n in (rows, cols)):
            raise ValueError("invalid flat grid dimensions")
        recipes = run["recipe_table"]
        if not 1 <= len(recipes) <= 256 or len(cells) != rows:
            raise ValueError("invalid flat recipe table or grid")
        for row in cells:
            if len(row) != cols or any(type(i) is not int or not 0 <= i < len(recipes) for i in row):
                raise ValueError("invalid flat cell recipe")
        for recipe in recipes:
            if (len(recipe["rgb"]) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in recipe["rgb"])
                    or len(recipe["slots"]) != 4 or any(v not in (1, 2, 3, 4) for v in recipe["slots"])
                    or not isinstance(recipe["source"], str)):
                raise ValueError("invalid flat recipe")
        allowed = run["plans"][run["selected_plan"]]["allowed_picture_slots"]
        if len(allowed) != 4 or any(not isinstance(layer, list) or not 1 <= len(layer) <= 4
                                    or any(slot not in (1, 2, 3, 4) for slot in layer) for layer in allowed):
            raise ValueError("invalid flat plan")
        _validate_region_controls(state["options"], run, rows, cols)
        if (state["options"]["profile"] != run["profile"] or len(state["palette"]["filaments"]) != 4
                or state["options"].get("plan", "full") != run["selected_plan"]):
            raise ValueError("saved controls disagree with flat run")
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid flat project: {exc}") from exc


def flat_accuracy(image, run):
    image = ImageOps.exif_transpose(image)
    recipes = run["recipe_table"]
    allowed = run["plans"][run["selected_plan"]]["allowed_picture_slots"]
    eligible = [i for i, recipe in enumerate(recipes)
                if all(slot in layer for slot, layer in zip(recipe["slots"], allowed))]
    grid = Image.fromarray(np.asarray(run["cell_recipe_ids"], dtype=np.uint8))
    ids = np.asarray(grid.resize(image.size, Image.Resampling.NEAREST)).ravel()
    labs = rgb_array_to_lab([r["rgb"] for r in recipes])
    sources = np.asarray([r["source"] for r in recipes])
    if "regions" not in run:
        return pixel_accuracy(image, ids, labs, eligible, sources)

    info = run["regions"]
    owner_grid = Image.fromarray(np.asarray(info["cell_regions"], dtype=np.uint8))
    owners = np.asarray(owner_grid.resize(image.size, Image.Resampling.NEAREST)).ravel()
    gamuts = {}
    reachable = set()
    for owner in np.unique(owners):
        local = [i for i in eligible if set(recipes[i]["slots"]).issubset(info["palettes"][owner])]
        gamuts[owner] = np.unique(labs[local], axis=0)
        reachable.update(local)
    return pixel_accuracy(image, ids, labs, sorted(reachable), sources, regional_gamuts=(owners, gamuts))
