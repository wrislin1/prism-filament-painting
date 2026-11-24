"""Differentiable color optimizer for renderer height fields."""

from __future__ import annotations

import importlib.util
import itertools
import math
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
from PIL import Image
from prism_filament_painting.core.color import relative_luminance_array
from prism_filament_painting.core.metrics import rgb_mse
from prism_filament_painting.core.prediction import calibration_catalog
from prism_filament_painting.core.model import Filament

from prism_filament_painting.render.errors import QuiltInputError

from .forward import build_forward_table, sources_in_field
from .guidance import Guidance, validate_maps
from .heightmap import build_height_field
from .mesh import MeshError as MeshError, _relief_units, _saddle_clips, mesh_triangle_count_from_clips
from .params import (
    ForwardTable,
    LayerParams,
    LuminanceParams,
    MeshParams,
    OptimizeParams,
    OptimizeResult,
    validate_luminance_params,
)
from .stack_search import baseline_schedule, limited_layers, stack_orders, wide_schedules
from .swaps import derive_swap_schedule


def is_torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def resolve_optimize_device(params: OptimizeParams) -> str | None:
    if not is_torch_available():
        return None
    try:
        return _select_device(_torch(), params).type
    except RuntimeError:
        return None


def differentiable_forward(
    heights: Any,
    filaments: Sequence[Filament],
    swap_schedule: Sequence[int],
    *,
    layer_params: LayerParams,
):
    """Render normalized RGB from continuous layer heights using torch interpolation."""
    forward = build_forward_table(filaments, tuple(swap_schedule), layer_params=layer_params)
    table = _torch().tensor(forward.layer_rgb, device=heights.device, dtype=heights.dtype) / 255.0
    return _interpolate_forward_table(heights, table, layer_params=layer_params)


def _interpolate_forward_table(heights: Any, table: Any, *, layer_params: LayerParams):
    torch = _torch()
    clamped = torch.clamp(heights, 0.0, float(layer_params.total_layers))
    lower = torch.floor(clamped).to(dtype=torch.long)
    upper = torch.clamp(lower + 1, max=layer_params.total_layers)
    fraction = (clamped - lower.to(dtype=clamped.dtype)).unsqueeze(-1)
    return table[lower] * (1.0 - fraction) + table[upper] * fraction


def optimize_height_field(
    image: Image.Image,
    filaments: Sequence[Filament],
    *,
    layer_params: LayerParams,
    mesh_params: MeshParams,
    luminance_params: LuminanceParams = LuminanceParams(),
    swap_schedule: Sequence[int] | None = None,
    seed_swap_schedule: Sequence[int] | None = None,
    orientation: str = "auto",
    params: OptimizeParams | None = None,
) -> OptimizeResult:
    optimize_params = params or OptimizeParams()
    validate_maps(optimize_params, image.size)
    filament_tuple = tuple(filaments)
    if not filament_tuple:
        raise QuiltInputError("filaments must not be empty")
    layer_params = limited_layers(layer_params, optimize_params)
    orders = None
    baseline = filament_tuple
    baseline_indices = tuple(range(len(filament_tuple)))
    if optimize_params.search_stacks:
        orders = list(stack_orders(filament_tuple, layer_params, optimize_params, swap_schedule))
        baseline_indices = orders[0]
        baseline = tuple(filament_tuple[i] for i in baseline_indices)
    input_schedule = baseline_schedule(len(baseline), layer_params, swap_schedule, seed_swap_schedule)
    input_table = build_forward_table(baseline, input_schedule, layer_params=layer_params)
    naive_field, factor, naive_exact, naive_clips = build_height_field(
        image,
        input_table,
        layer_params=layer_params,
        mesh_params=mesh_params,
        luminance_params=luminance_params,
        orientation=orientation,
    )
    transparent = _transparent_mask(image, naive_field.shape)
    loss_mask = transparent if mesh_params.flatten_transparent else None
    luminance_params = validate_luminance_params(luminance_params)
    guidance = Guidance.prepare(optimize_params, naive_field.shape, loss_mask, layer_params)
    naive_loss = preview_loss(
        image,
        naive_field,
        input_table,
        transparent_mask=loss_mask,
        guidance=guidance,
        luminance_params=luminance_params,
    )

    naive_sources = sources_in_field(input_table, naive_field, loss_mask) if calibration_catalog() else None
    if not is_torch_available():
        return _fallback_result(
            naive_field,
            input_table,
            baseline,
            baseline_indices,
            input_schedule,
            factor,
            naive_exact,
            naive_clips,
            naive_loss,
            "optimize-unavailable: torch is not installed",
            prediction_sources=naive_sources,
        )

    try:
        torch = _torch()
        device = _select_device(torch, optimize_params)
    except RuntimeError as exc:
        return _fallback_result(
            naive_field,
            input_table,
            baseline,
            baseline_indices,
            input_schedule,
            factor,
            naive_exact,
            naive_clips,
            naive_loss,
            f"optimize-unavailable: {exc}",
            prediction_sources=naive_sources,
        )

    if naive_field.size > optimize_params.max_pixels:
        return _fallback_result(
            naive_field,
            input_table,
            baseline,
            baseline_indices,
            input_schedule,
            factor,
            naive_exact,
            naive_clips,
            naive_loss,
            f"optimize-unavailable: downsampled field has {naive_field.size} pixels over max {optimize_params.max_pixels}",
            prediction_sources=naive_sources,
        )

    torch.manual_seed(int(optimize_params.seed))
    if loss_mask is not None and not np.any(~loss_mask):
        return _fallback_result(
            naive_field,
            input_table,
            baseline,
            baseline_indices,
            input_schedule,
            factor,
            naive_exact,
            naive_clips,
            naive_loss,
            "optimize-unavailable: image has no opaque pixels to optimize",
            prediction_sources=naive_sources,
        )
    opaque_mask = torch.tensor(~loss_mask, dtype=torch.bool, device=device) if loss_mask is not None else None
    target_rgb = _target_rgb(image, naive_field.shape, luminance_params=luminance_params)
    target = torch.as_tensor(target_rgb, dtype=torch.float32, device=device)
    best: OptimizeResult | None = None
    attempted = 0
    skipped_over_cap = 0

    candidates = _rank_candidate_pairs(
        target_rgb,
        loss_mask,
        filament_tuple,
        layer_params=layer_params,
        explicit_schedule=input_schedule if swap_schedule is not None else None,
        optimize_params=optimize_params,
        seed_schedule=seed_swap_schedule if swap_schedule is None else None,
        stack_indices=orders,
        guidance=guidance,
    )

    for order_indices, order, schedule in candidates:
        attempted += 1
        table = build_forward_table(order, schedule, layer_params=layer_params)
        fields = _candidate_height_fields(
            target,
            naive_field,
            table,
            torch=torch,
            layer_params=layer_params,
            optimize_params=optimize_params,
            device=device,
            opaque_mask=opaque_mask,
            guidance=guidance,
            exact_field=guidance.nearest(target_rgb, table, loss_mask)[0] if guidance else None,
        )
        for field, gradient_loss in fields:
            if loss_mask is not None:
                field[loss_mask] = layer_params.base_layers
            clips = _saddle_clips(field)
            exact = mesh_triangle_count_from_clips(field, clips, _relief_units(1.0))
            if exact <= mesh_params.max_triangles:
                break
        else:
            skipped_over_cap += 1
            continue
        loss = preview_loss(image, field, table, transparent_mask=loss_mask,
                            luminance_params=luminance_params, guidance=guidance)
        report = {
            "mode": "optimize",
            "used_optimizer": True,
            "fallback_reason": None,
            "naive_loss": naive_loss,
            "optimized_loss": loss,
            "gradient_loss": gradient_loss,
            "device": str(device),
            "iterations": 0 if gradient_loss is None else int(optimize_params.iterations),
            "order_indices": order_indices,
            "swap_schedule": tuple(int(value) for value in schedule),
            "attempted_candidates": attempted,
            "skipped_over_cap": skipped_over_cap,
        }
        if guidance:
            report["source_colour_mse"] = preview_loss(image, field, table, transparent_mask=loss_mask,
                                                      luminance_params=luminance_params)
            report["height_strength"] = guidance.strength
        if naive_sources is not None:
            report["naive_prediction_sources"] = naive_sources
            report["optimized_prediction_sources"] = sources_in_field(table, field, loss_mask)
        candidate = OptimizeResult(
            height_field=field,
            forward_table=table,
            order=order,
            order_indices=order_indices,
            swap_schedule=tuple(int(value) for value in schedule),
            downsample_factor=factor,
            exact_triangle_count=exact,
            saddle_clips=clips,
            report=report,
        )
        if best is None or loss < best.report["optimized_loss"]:
            best = candidate
        # Pairs are ranked by their lowest possible pixel loss. Once a nearest-
        # layer field fits the mesh cap, later pairs cannot improve the result.
        if gradient_loss is None:
            break

    if best is None:
        fallback = _fallback_result(
            naive_field,
            input_table,
            baseline,
            baseline_indices,
            input_schedule,
            factor,
            naive_exact,
            naive_clips,
            naive_loss,
            f"optimize-cap-fallback: all {attempted} candidates exceeded max_triangles",
            prediction_sources=naive_sources,
        )
        fallback.report["attempted_candidates"] = attempted
        fallback.report["skipped_over_cap"] = skipped_over_cap
        return fallback
    best.report["attempted_candidates"] = attempted
    best.report["skipped_over_cap"] = skipped_over_cap
    if float(best.report["optimized_loss"]) >= naive_loss:
        fallback = _fallback_result(
            naive_field,
            input_table,
            baseline,
            baseline_indices,
            input_schedule,
            factor,
            naive_exact,
            naive_clips,
            naive_loss,
            (
                "optimize-quality-fallback: best optimized_loss "
                f"{best.report['optimized_loss']:.8g} >= naive_loss {naive_loss:.8g}"
            ),
            prediction_sources=naive_sources,
        )
        fallback.report["attempted_candidates"] = attempted
        fallback.report["skipped_over_cap"] = skipped_over_cap
        return fallback
    return best


def preview_loss(
    image: Image.Image,
    height_field: np.ndarray,
    table: ForwardTable,
    *,
    transparent_mask: np.ndarray | None = None,
    luminance_params: LuminanceParams = LuminanceParams(),
    guidance: Guidance | None = None,
) -> float:
    target = _target_rgb(image, height_field.shape, luminance_params=luminance_params)
    layers = np.asarray(table.layer_rgb, dtype=np.float32) / 255.0
    predicted = layers[np.asarray(height_field, dtype=np.int32)]
    if guidance:
        return float(guidance.field_loss(((predicted - target) ** 2).mean(axis=-1), height_field.astype(np.float32),
                                         None if transparent_mask is None else ~transparent_mask))
    return rgb_mse(predicted, target, transparent_mask=transparent_mask)


def _fallback_result(
    height_field: np.ndarray,
    table: ForwardTable,
    order: tuple[Filament, ...],
    order_indices: tuple[int, ...],
    schedule: tuple[int, ...],
    factor: int,
    exact: int,
    clips: np.ndarray,
    naive_loss: float,
    reason: str,
    *,
    prediction_sources: list[str] | None,
) -> OptimizeResult:
    return OptimizeResult(
        height_field=height_field,
        forward_table=table,
        order=order,
        order_indices=order_indices,
        swap_schedule=schedule,
        downsample_factor=factor,
        exact_triangle_count=exact,
        saddle_clips=clips,
        report={
            "mode": "optimize",
            "used_optimizer": False,
            "fallback_reason": reason,
            "naive_loss": naive_loss,
            "optimized_loss": naive_loss,
            **({"naive_prediction_sources": prediction_sources, "optimized_prediction_sources": prediction_sources}
               if prediction_sources is not None else {}),
            "device": None,
            "iterations": 0,
            "order_indices": order_indices,
            "swap_schedule": schedule,
        },
    )


def _nearest_printable_heights(target, table, base_layers: int, *, torch) -> np.ndarray:
    printable = table[base_layers:]
    # Source RGB loss is separable by pixel. Search printable heights directly,
    # in batches so a large table does not allocate pixels × layers × RGB at once.
    pixels = target.reshape(-1, 3)
    batch_size = max(1, 262_144 // len(printable))
    heights = torch.empty(len(pixels), dtype=torch.int32, device=target.device)
    for start in range(0, len(pixels), batch_size):
        batch = pixels[start : start + batch_size]
        distances = ((batch[:, None, :] - printable) ** 2).sum(dim=-1)
        heights[start : start + batch_size] = distances.argmin(dim=-1) + base_layers
    return heights.reshape(target.shape[:2]).cpu().numpy()


def _candidate_height_fields(
    target,
    naive_field: np.ndarray,
    forward: ForwardTable,
    *,
    torch,
    layer_params: LayerParams,
    optimize_params: OptimizeParams,
    device,
    opaque_mask=None,
    guidance=None,
    exact_field=None,
) -> Iterator[tuple[np.ndarray, float | None]]:
    table = torch.tensor(forward.layer_rgb, device=device, dtype=torch.float32) / 255.0
    if exact_field is None:
        exact_field = _nearest_printable_heights(target, table, layer_params.base_layers, torch=torch)
    yield exact_field, None
    if guidance:
        guidance = guidance.on_device(torch, device)

    def field_loss(rendered, heights):
        if guidance:
            return guidance.field_loss(((rendered - target) ** 2).mean(dim=-1), heights, opaque_mask)
        return _masked_mse(rendered, target, opaque_mask)

    # The nearest field exceeded the triangle cap. The luminance-seeded local
    # search can still produce a feasible relief for this order and schedule.
    height = torch.tensor(naive_field, dtype=torch.float32, device=device).clone().detach()
    height.requires_grad_(True)
    optimizer = torch.optim.Adam([height], lr=float(optimize_params.learning_rate))
    gradient_loss = math.inf
    last_loss = None
    for _ in range(max(0, int(optimize_params.iterations))):
        optimizer.zero_grad(set_to_none=True)
        rendered = _interpolate_forward_table(height, table, layer_params=layer_params)
        loss = field_loss(rendered, height)
        last_loss = loss.detach()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            height.clamp_(float(layer_params.base_layers), float(layer_params.total_layers))
    if last_loss is not None:
        gradient_loss = float(last_loss.cpu().item())

    rounded = torch.round(height.detach()).to(dtype=torch.long)
    rounded.clamp_(layer_params.base_layers, layer_params.total_layers)
    snap_radius = max(0, int(optimize_params.snap_radius))
    offsets = torch.arange(-snap_radius, snap_radius + 1, dtype=torch.long, device=device)
    local_candidates = rounded.unsqueeze(-1) + offsets
    local_candidates.clamp_(layer_params.base_layers, layer_params.total_layers)
    distances = torch.sum((target.unsqueeze(-2) - table[local_candidates]) ** 2, dim=-1)
    if guidance:
        distances = guidance.costs(distances / 3, local_candidates, guidance.weight.unsqueeze(-1),
                                   None if guidance.depth is None else guidance.depth.unsqueeze(-1))
    best_local = torch.argmin(distances, dim=-1, keepdim=True)
    snapped = torch.gather(local_candidates, dim=-1, index=best_local).squeeze(-1).to(dtype=torch.long)

    rounded_loss = field_loss(table[rounded], rounded)
    snapped_loss = field_loss(table[snapped], snapped)
    if float(snapped_loss.detach().cpu().item()) <= float(rounded_loss.detach().cpu().item()):
        field = snapped.detach().cpu().numpy().astype(np.int32)
    else:
        field = rounded.detach().cpu().numpy().astype(np.int32)
    yield field, gradient_loss


def _masked_mse(rendered, target, opaque_mask):
    torch = _torch()
    error = (rendered - target) ** 2
    if opaque_mask is None:
        return torch.mean(error)
    return torch.mean(error[opaque_mask])


def _select_device(torch, params: OptimizeParams):
    requested = params.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        if params.allow_cpu:
            return torch.device("cpu")
        raise RuntimeError("CUDA is unavailable and CPU optimize is disabled")
    if requested == "cpu" and not params.allow_cpu:
        raise RuntimeError("CPU optimize is disabled")
    return torch.device(requested)


def _rank_candidate_pairs(
    target_rgb: np.ndarray,
    transparent_mask: np.ndarray | None,
    filaments: tuple[Filament, ...],
    *,
    layer_params: LayerParams,
    explicit_schedule: tuple[int, ...] | None,
    optimize_params: OptimizeParams,
    seed_schedule: Sequence[int] | None = None,
    stack_indices: Sequence[tuple[int, ...]] | None = None,
    guidance: Guidance | None = None,
) -> list[tuple[tuple[int, ...], tuple[Filament, ...], tuple[int, ...]]]:
    max_candidates = max(0, int(optimize_params.max_candidate_optimizations))
    if max_candidates == 0:
        return []
    if stack_indices is not None:
        orders = [(indices, tuple(filaments[i] for i in indices)) for indices in stack_indices]
    elif optimize_params.pin_order:
        orders = [(tuple(range(len(filaments))), filaments)]
    else:
        orders = _order_candidates(filaments, optimize_params.max_order_permutations)
    max_orders = max(1, int(optimize_params.max_order_candidates))
    if len(orders) > max_orders:
        orders = [
            item
            for _, item in sorted(
                (
                    (
                        _table_nearest_loss(
                            target_rgb,
                            transparent_mask,
                            build_forward_table(
                                order,
                                explicit_schedule
                                if explicit_schedule is not None
                                else derive_swap_schedule(
                                    len(order),
                                    total_layers=layer_params.total_layers,
                                    base_layers=layer_params.base_layers,
                                ),
                                layer_params=layer_params,
                            ),
                            layer_params=layer_params, guidance=guidance,
                        ),
                        (order_indices, order),
                    )
                    for order_indices, order in orders
                ),
                key=lambda value: value[0],
            )[:max_orders]
        ]

    ranked: list[tuple[float, tuple[int, ...], tuple[Filament, ...], tuple[int, ...]]] = []
    for order_indices, order in orders:
        if stack_indices is not None:
            schedules = wide_schedules(len(order), layer_params, explicit_schedule, seed_schedule,
                                       optimize_params.max_schedule_candidates)
        else:
            schedules = _schedule_candidates(
                len(order),
                layer_params=layer_params,
                explicit=explicit_schedule,
                max_candidates=optimize_params.max_schedule_candidates,
                seed=seed_schedule,
            )
        for schedule in schedules:
            table = build_forward_table(order, schedule, layer_params=layer_params)
            loss = _table_nearest_loss(target_rgb, transparent_mask, table, layer_params=layer_params, guidance=guidance)
            ranked.append((loss, order_indices, order, schedule))
    ranked.sort(key=lambda value: value[0])
    return [(order_indices, order, schedule) for _, order_indices, order, schedule in ranked[:max_candidates]]


def _table_nearest_loss(
    target_rgb: np.ndarray,
    transparent_mask: np.ndarray | None,
    table: ForwardTable,
    *,
    layer_params: LayerParams,
    guidance: Guidance | None = None,
) -> float:
    if guidance:
        return guidance.nearest(target_rgb, table, transparent_mask)[1]
    layers = np.asarray(table.layer_rgb, dtype=np.float32) / 255.0
    valid = layers[layer_params.base_layers : layer_params.total_layers + 1]
    distances = np.sum((target_rgb[..., np.newaxis, :] - valid) ** 2, axis=-1)
    nearest = np.min(distances, axis=-1)
    if transparent_mask is not None:
        opaque = ~transparent_mask
        if not np.any(opaque):
            return 0.0
        nearest = nearest[opaque]
    return float(np.mean(nearest))


def _order_candidates(
    filaments: tuple[Filament, ...], max_permutations: int
) -> list[tuple[tuple[int, ...], tuple[Filament, ...]]]:
    size = len(filaments)
    total = math.factorial(size)
    if total > max_permutations:
        return [(tuple(range(size)), filaments)]
    return [(indices, tuple(filaments[index] for index in indices)) for indices in itertools.permutations(range(size))]


def _schedule_candidates(
    palette_size: int,
    *,
    layer_params: LayerParams,
    explicit: tuple[int, ...] | None,
    max_candidates: int,
    seed: Sequence[int] | None = None,
) -> list[tuple[int, ...]]:
    if explicit is not None:
        return [tuple(explicit)]
    even = derive_swap_schedule(
        palette_size, total_layers=layer_params.total_layers, base_layers=layer_params.base_layers
    )
    if palette_size <= 1:
        return [even]
    # A non-pinning ``seed`` (the Prism ranker's tuned schedule) is always kept and
    # explored around, so a far-from-even ranker suggestion is actually considered
    # by the optimizer instead of being missed by the even-centred search.
    seed_tuple: tuple[int, ...] | None = None
    if seed is not None:
        candidate_seed = tuple(int(value) for value in seed)
        if len(candidate_seed) == palette_size - 1 and _valid_schedule(
            candidate_seed, layer_params=layer_params
        ):
            seed_tuple = candidate_seed

    forced: list[tuple[int, ...]] = [even]
    if seed_tuple is not None and seed_tuple != even:
        forced.append(seed_tuple)

    values: set[tuple[int, ...]] = set(forced)
    valid = range(layer_params.base_layers + 1, layer_params.total_layers)
    if palette_size == 2:
        for layer in _sample_layers(valid, max(1, max_candidates)):
            values.add((layer,))
    else:
        offsets = (-2, -1, 1, 2)
        anchors = (even,) if seed_tuple is None else (even, seed_tuple)
        for anchor in anchors:
            for position, current in enumerate(anchor):
                for offset in offsets:
                    candidate = list(anchor)
                    candidate[position] = current + offset
                    if _valid_schedule(candidate, layer_params=layer_params):
                        values.add(tuple(candidate))
    # Forced schedules (even + seed) survive truncation so the seed is never lost.
    extras = [schedule for schedule in sorted(values) if schedule not in forced]
    limit = max(len(forced), max_candidates)
    return (forced + extras)[:limit]


def _sample_layers(values: range, count: int) -> list[int]:
    all_values = list(values)
    if len(all_values) <= count:
        return all_values
    if count == 1:
        return [all_values[len(all_values) // 2]]
    positions = np.linspace(0, len(all_values) - 1, count)
    return [all_values[int(round(position))] for position in positions]


def _valid_schedule(candidate: Sequence[int], *, layer_params: LayerParams) -> bool:
    previous = layer_params.base_layers
    for layer in candidate:
        if layer <= previous or layer >= layer_params.total_layers:
            return False
        previous = int(layer)
    return True


def _target_rgb(
    image: Image.Image,
    shape: tuple[int, int],
    *,
    luminance_params: LuminanceParams = LuminanceParams(),
) -> np.ndarray:
    luminance_params = validate_luminance_params(luminance_params)
    height, width = shape
    rgb = image.convert("RGB")
    if rgb.size != (width, height):
        rgb = rgb.resize((width, height), Image.Resampling.BOX)
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    if luminance_params.is_identity:
        return values
    # Match the 8-bit target used by rgb_mse when ranking and choosing heights.
    adjusted = _apply_luminance_to_rgb(values, luminance_params)
    return np.rint(adjusted * 255.0) / 255.0


def _apply_luminance_to_rgb(rgb: np.ndarray, luminance_params: LuminanceParams) -> np.ndarray:
    luminance = relative_luminance_array(np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
    target_luminance = np.clip(
        (luminance - luminance_params.black) / (luminance_params.white - luminance_params.black),
        0.0,
        1.0,
    )
    target_luminance = target_luminance**luminance_params.gamma
    linear = _srgb_to_linear(rgb)
    scale = np.divide(
        target_luminance,
        luminance,
        out=np.zeros_like(target_luminance, dtype=np.float64),
        where=luminance > 0,
    )
    adjusted = np.clip(linear * scale[..., None], 0.0, 1.0)
    return _linear_to_srgb(adjusted).astype(np.float32)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * np.power(values, 1 / 2.4) - 0.055)


def _transparent_mask(image: Image.Image, shape: tuple[int, int]) -> np.ndarray | None:
    if image.mode != "RGBA":
        return None
    height, width = shape
    alpha = image.getchannel("A")
    if alpha.size != (width, height):
        alpha = alpha.resize((width, height), Image.Resampling.BOX)
    return np.asarray(alpha, dtype=np.uint8) == 0


def _torch():
    import torch

    return torch
