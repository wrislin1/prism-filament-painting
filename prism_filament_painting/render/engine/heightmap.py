"""Naive luminance-to-height conversion."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from prism_filament_painting.core.color import relative_luminance, relative_luminance_array

from prism_filament_painting.render.errors import QuiltInputError

from .params import ForwardTable, LayerParams, LuminanceParams, MeshParams, validate_luminance_params


def build_height_field(
    image: Image.Image,
    table: ForwardTable,
    *,
    layer_params: LayerParams,
    mesh_params: MeshParams,
    luminance_params: LuminanceParams = LuminanceParams(),
    orientation: str = "auto",
) -> tuple[np.ndarray, int, int, np.ndarray]:
    """Resolve an image into a height field under the triangle cap.

    Returns ``(field, downsample_factor, exact_triangle_count, saddle_clips)``. The downsample
    factor is seeded using a cheap vectorized upper bound on the triangle count, then recovered
    toward the finest exact-counted factor that fits the cap. The exact count and saddle clips
    are threaded out so ``build_mesh`` does not recompute them.
    """
    from .mesh import (
        MeshError,
        _relief_units,
        _saddle_clips,
        mesh_triangle_count_from_clips,
        triangle_count_upper_bound,
    )

    _validate_layer_params(layer_params)
    try:
        luminance_params = validate_luminance_params(luminance_params)
    except ValueError as exc:
        raise QuiltInputError(str(exc)) from exc
    resolved_orientation = _resolve_orientation(orientation, table, layer_params)
    rgba = image if image.mode == "RGBA" else image.convert("RGBA")
    cap = mesh_params.max_triangles
    width0, height0 = rgba.size

    def field_at(factor: int) -> np.ndarray:
        if factor <= 1:
            candidate = rgba
        else:
            candidate = rgba.resize(
                (math.ceil(width0 / factor), math.ceil(height0 / factor)), Image.Resampling.BOX
            )
        return _field_from_rgba(
            candidate,
            layer_params=layer_params,
            mesh_params=mesh_params,
            luminance_params=luminance_params,
            resolved_orientation=resolved_orientation,
        )

    def is_unit(factor: int) -> bool:
        return math.ceil(width0 / factor) <= 1 and math.ceil(height0 / factor) <= 1

    # 1. Pick the downsample factor from the cheap upper bound (vectorized; no full-res exact count).
    field = field_at(1)
    factor = 1
    bound = triangle_count_upper_bound(field)
    if bound > cap:
        factor = max(2, math.ceil(math.sqrt(bound / cap)))  # seed near the boundary (bound ~ area)
        field = field_at(factor)
        while triangle_count_upper_bound(field) > cap:  # refine up (field shrinks each step)
            if is_unit(factor):
                raise MeshError("image cannot be reduced under max_triangles")
            factor += 1
            field = field_at(factor)
        while factor > 1:  # refine down: the bound is conservative, a finer factor may still fit
            finer = field_at(factor - 1)
            if triangle_count_upper_bound(finer) > cap:
                break
            factor -= 1
            field = finer

    # 2. Exact-count the chosen candidate; guaranteed <= upper bound <= cap. Safe exact
    #    candidates have <= cap/4 cells (top+base alone), so cache only those bounded fields.
    relief = _relief_units(1.0)
    exact_cache: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}

    def exact_candidate(candidate_factor: int) -> tuple[np.ndarray, np.ndarray, int]:
        if candidate_factor in exact_cache:
            return exact_cache[candidate_factor]
        candidate_field = field_at(candidate_factor)
        candidate_clips = _saddle_clips(candidate_field)
        candidate_exact = mesh_triangle_count_from_clips(candidate_field, candidate_clips, relief)
        candidate = (candidate_field, candidate_clips, candidate_exact)
        if candidate_exact <= cap:
            exact_cache[candidate_factor] = candidate
        return candidate

    field, clips, exact = exact_candidate(factor)
    while exact > cap:
        if is_unit(factor):
            raise MeshError("image cannot be reduced under max_triangles")
        factor += 1
        field, clips, exact = exact_candidate(factor)

    # 3. Resolution recovery: the bound over-counts non-saddle edges, so it can
    #    over-downsample badly. Probe exact counts by halving toward the finest safe factor,
    #    then binary-search the bracket and repair the adjacent boundary.
    safe_factor = factor
    if safe_factor > 1:
        low = 0
        probe = max(1, safe_factor // 2)
        while True:
            _, _, probe_exact = exact_candidate(probe)
            if probe_exact <= cap:
                safe_factor = probe
                if probe == 1:
                    break
                probe = max(1, probe // 2)
                continue
            low = probe
            break

        if low > 0:
            high = safe_factor
            while high - low > 1:
                mid = (low + high) // 2
                _, _, mid_exact = exact_candidate(mid)
                if mid_exact <= cap:
                    high = mid
                else:
                    low = mid
            safe_factor = high

        while safe_factor > 1:
            _, _, previous_exact = exact_candidate(safe_factor - 1)
            if previous_exact > cap:
                break
            safe_factor -= 1

        factor = safe_factor
        field, clips, exact = exact_candidate(factor)

    return field, factor, exact, clips


def _field_from_rgba(
    rgba: Image.Image,
    *,
    layer_params: LayerParams,
    mesh_params: MeshParams,
    luminance_params: LuminanceParams,
    resolved_orientation: str,
) -> np.ndarray:
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    luminance = relative_luminance_array(rgb)
    span = layer_params.total_layers - layer_params.base_layers
    if not luminance_params.is_identity:
        luminance = np.clip(
            (luminance - luminance_params.black) / (luminance_params.white - luminance_params.black),
            0.0,
            1.0,
        )
        luminance = luminance**luminance_params.gamma
    if resolved_orientation == "dark":
        normalized = 1.0 - luminance
    else:
        normalized = luminance
    field = np.rint(layer_params.base_layers + normalized * span).astype(np.int32)
    field = np.clip(field, layer_params.base_layers, layer_params.total_layers).astype(np.int32)
    if mesh_params.flatten_transparent:
        field[alpha == 0] = layer_params.base_layers
    return field


def _resolve_orientation(orientation: str, table: ForwardTable, layer_params: LayerParams) -> str:
    if orientation not in {"auto", "dark", "bright"}:
        raise QuiltInputError("orientation must be auto, dark, or bright")
    if orientation == "auto":
        return "dark" if _tall_stack_is_dark(table, layer_params) else "bright"
    return orientation


def _tall_stack_is_dark(table: ForwardTable, layer_params: LayerParams) -> bool:
    low = relative_luminance(table.layer_rgb[layer_params.base_layers])
    high = relative_luminance(table.layer_rgb[layer_params.total_layers])
    return high <= low


def triangle_upper_bound_for_size(width: int, height: int, total_layers: int) -> int:
    """Worst-case (all cells at total_layers, maximal saddle clipping) upper bound on the
    triangle count for a width x height field. Table-free, so resource admission (which has no
    forward table or luminance) can size requests conservatively without building the field.
    """
    if width < 1 or height < 1:
        return 0
    layers = int(total_layers)
    cells = width * height
    top_base = 12 * cells  # <= 2*(2+4) per cell
    walls = 8 * layers * cells  # <= 2*4*total_layers per clipped corner, <=4 corners
    interior_edges = (width - 1) * height + width * (height - 1)
    interior = 10 * layers * interior_edges  # <= 2*|a-b| + 4*(a+b) <= 10*total_layers per edge
    boundary = 4 * layers * (width + height)  # <= 2*total_layers per boundary edge
    return int(top_base + walls + interior + boundary)


def choose_downsample_factor_for_size(size: tuple[int, int], *, max_triangles: int, total_layers: int) -> int:
    if max_triangles < 12:
        raise QuiltInputError("max_triangles must be at least 12")
    width, height = size
    bound = triangle_upper_bound_for_size(width, height, total_layers)
    if bound <= max_triangles:
        return 1
    factor = max(1, math.ceil(math.sqrt(bound / max_triangles)))
    while True:
        down_width = max(1, math.ceil(width / factor))
        down_height = max(1, math.ceil(height / factor))
        if triangle_upper_bound_for_size(down_width, down_height, total_layers) <= max_triangles:
            return factor
        if down_width <= 1 and down_height <= 1:
            return factor  # cannot reduce further; admission over-estimates but stays bounded
        factor += 1


def _validate_layer_params(layer_params: LayerParams) -> None:
    if not math.isfinite(layer_params.layer_height_mm) or layer_params.layer_height_mm <= 0:
        raise QuiltInputError("layer_height_mm must be finite and positive")
    if layer_params.base_layers < 1:
        raise QuiltInputError("base_layers must be at least 1")
    if layer_params.total_layers <= layer_params.base_layers:
        raise QuiltInputError("total_layers must be greater than base_layers")
