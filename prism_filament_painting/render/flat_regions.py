"""Assign cells to region palettes and optionally diffuse linear-light residuals."""

import numpy as np
from prism_filament_painting.core.color import ciede2000_array, rgb_array_to_lab
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


BLOCK_CELLS = 128


class Region(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    slots: list[StrictInt] = Field(min_length=1, max_length=4)
    mask: list[str] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_mask(self):
        width = len(self.mask[0])
        if not 1 <= width <= 100 or any(len(row) != width or set(row) - {"0", "1"} for row in self.mask):
            raise ValueError("region mask must be a rectangular binary grid up to 100 by 100")
        validate_slots(self.slots)
        return self


def validate_slots(slots):
    if not slots or len(set(slots)) != len(slots) or any(s not in (1, 2, 3, 4) for s in slots):
        raise ValueError("choose distinct loaded slots 1 through 4")


def assign_regions(width, height, regions):
    owners = np.zeros((height, width), dtype=np.uint8)
    claims = np.zeros_like(owners)
    for index, region in enumerate(regions, 1):
        mask = np.array([list(row) for row in region.mask]) == "1"
        # Sample cell centres in normalized, EXIF-oriented source space.
        ys = (2 * np.arange(height) + 1) * mask.shape[0] // (2 * height)
        xs = (2 * np.arange(width) + 1) * mask.shape[1] // (2 * width)
        selected = mask[np.ix_(ys, xs)]
        claims += selected
        owners[selected] = index
    return owners, int(np.count_nonzero(claims > 1))


def _linear(rgb):
    srgb = np.asarray(rgb, dtype=float) / 255
    return np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)


def _blur_lab(linear):
    # Separable 1:2:1 filter, one-cell radius, edge replication.
    for axis in (0, 1):
        pads = [(0, 0)] * 3
        pads[axis] = (1, 1)
        padded = np.pad(linear, pads, mode="edge").swapaxes(0, axis)
        linear = ((padded[:-2] + 2 * padded[1:-1] + padded[2:]) / 4).swapaxes(0, axis)
    srgb = np.where(linear <= .0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - .055)
    return rgb_array_to_lab(srgb * 255)


def blurred_error(target, table, ids):
    printed = _linear([r.rgb for r in table])[ids.reshape(target.height, target.width)]
    return float(ciede2000_array(_blur_lab(_linear(target)), _blur_lab(printed)).mean())


def _diffuse(target, colours, owners, scope, eligible, direct):
    work = _linear(target)
    height, width = owners.shape
    result = direct.copy().reshape(height, width)
    palettes = {owner: colours[ids] for owner, ids in eligible.items()}
    for y in range(height):
        for x in range(width):
            owner = owners[y, x]
            if scope != -1 and owner != scope:
                continue
            desired = work[y, x].clip(0, 1)
            delta = palettes[owner] - desired
            best = np.einsum("ij,ij->i", delta, delta).argmin()
            recipe = eligible[owner][best]
            result[y, x] = recipe
            residual = desired - colours[recipe]
            for dx, dy, weight in ((1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)):
                nx, ny = x + dx, y + dy
                if (0 <= nx < width and ny < height and owners[ny, nx] == owner
                        and (not dx or not dy or owners[y, nx] == owner or owners[ny, x] == owner)):
                    work[ny, nx] += residual * (weight / 16)
    return result.ravel()


def match_cells(target, table, eligible):
    """Share each block of cell-by-recipe distances across every eligible subset."""
    labs = rgb_array_to_lab([row.rgb for row in table])
    cells = rgb_array_to_lab(np.asarray(target).reshape(-1, 3))
    ids = {name: np.empty(len(cells), dtype=np.uint8) for name in eligible}
    errors = {name: np.empty(len(cells), dtype=float) for name in eligible}
    for start in range(0, len(cells), BLOCK_CELLS):
        block = slice(start, start + BLOCK_CELLS)
        distances = ciede2000_array(cells[block, None, :], labs[None, :, :])
        indices = np.arange(len(distances))
        for name, candidates in eligible.items():
            # Candidates follow lexicographic slot order; argmin keeps the first tie.
            best = candidates[np.argmin(distances[:, candidates], axis=1)]
            ids[name][block] = best
            errors[name][block] = distances[indices, best]
    return ids, errors


def match_regions(target, table, recipes, eligible, regions, background, scope, selected_plan):
    owners, overlaps = assign_regions(*target.size, regions)
    slots = [background, *(r.slots for r in regions)]
    names = ["background", *(r.name for r in regions)]
    if scope is not None and not -1 <= scope < len(slots):
        raise ValueError("choose an existing dithering region")
    by_plan = {name: {} for name in eligible}
    direct = {name: np.empty(owners.size, np.uint8) for name in eligible}
    errors = {name: np.empty(owners.size) for name in eligible}
    pixels = np.asarray(target).reshape(-1, 3)
    occupied = np.unique(owners)
    for owner in occupied:
        valid = np.isin(recipes, slots[owner]).all(axis=1)
        allowed = {}
        for name, candidates in eligible.items():
            local = candidates[valid[candidates]]
            if local.size:
                by_plan[name][owner] = allowed[name] = local
            elif name == selected_plan:
                raise ValueError(f"{names[owner]} has no printable recipe in plan {name}")
        cells = owners.ravel() == owner
        local_ids, local_errors = match_cells(pixels[cells], table, allowed)
        for name in allowed:
            direct[name][cells] = local_ids[name]
            errors[name][cells] = local_errors[name]
    ids = {name: result for name, result in direct.items() if len(by_plan[name]) == len(occupied)}
    info = dict(cell_regions=owners.tolist(), palettes=slots, overlap_cells=overlaps,
                empty_regions=[names[i] for i in range(1, len(slots)) if not np.any(owners == i)],
                transform="full oriented source, top-left, normalized cell centres")
    baseline = ids[selected_plan]
    if scope is not None:
        colours = [row.rgb for row in table]
        labs = rgb_array_to_lab(colours)
        target_lab = rgb_array_to_lab(pixels)
        linear = _linear(colours)
        for name in ids:
            ids[name] = _diffuse(target, linear, owners, scope, by_plan[name], ids[name])
            errors[name] = ciede2000_array(target_lab, labs[ids[name]])
    return ids, errors, info, baseline


def narrow_cells(planes):
    """Count layer cells with no edge neighbour or opposed exposed sides."""
    padded = np.pad(planes, ((0, 0), (1, 1), (1, 1)), constant_values=0)
    up, down = planes == padded[:, :-2, 1:-1], planes == padded[:, 2:, 1:-1]
    left, right = planes == padded[:, 1:-1, :-2], planes == padded[:, 1:-1, 2:]
    single = ~(left | right | up | down)
    thin = (~(up | down) | ~(left | right)) & ~single
    return dict(isolated_layer_cells=int(single.sum()), thin_layer_cells=int(thin.sum()))
