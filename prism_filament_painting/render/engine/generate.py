"""End-to-end Quilt engine orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from PIL import Image, ImageOps
from prism_filament_painting.core.finishing import finished_grid, quantize_detail
from prism_filament_painting.core.forward import RenderConstants, build_forward_stack
from prism_filament_painting.core.prediction import calibration_catalog
from prism_filament_painting.core.model import Filament, PaletteCandidate

from prism_filament_painting.render.errors import QuiltInputError

from .forward import build_forward_table, sources_in_field
from .heightmap import build_height_field
from .mesh import DEFAULT_FULL_VALIDATE_MAX_TRIANGLES, build_mesh, mesh_to_stl_bytes
from .optimize import optimize_height_field, preview_loss, _transparent_mask
from .params import GenerateResult, LayerParams, LuminanceParams, MeshParams, OptimizeParams
from .preview import render_preview
from .stack_search import baseline_schedule, limited_layers, stack_orders
from .swaps import derive_swap_schedule, swap_instructions


def generate(
    image: Image.Image,
    palette: PaletteCandidate | Sequence[Filament],
    *,
    layer_params: LayerParams,
    mesh_params: MeshParams,
    luminance_params: LuminanceParams = LuminanceParams(),
    swap_schedule: Sequence[int] | None = None,
    seed_swap_schedule: Sequence[int] | None = None,
    orientation: str = "auto",
    mode: str = "naive",
    full_validate_max_triangles: int = DEFAULT_FULL_VALIDATE_MAX_TRIANGLES,
    optimize_params: OptimizeParams | None = None,
) -> GenerateResult:
    filaments = _filaments_from_palette(palette)
    if mode not in {"naive", "optimize"}:
        raise QuiltInputError("mode must be naive or optimize")
    optimize_report = None
    if optimize_params is not None:
        layer_params = limited_layers(layer_params, optimize_params)
    finish = mesh_params.print_options
    grid = None
    if finish.active:
        constants = RenderConstants(
            layer_height_mm=layer_params.layer_height_mm, base_layers=layer_params.base_layers,
            total_layers=layer_params.total_layers, black_point=luminance_params.black,
            white_point=luminance_params.white, gamma=luminance_params.gamma, front_lit_scale=layer_params.td_scale,
        )
        # The manual stack fixes the footprint even when guidance changes the printed colours.
        try:
            manual_schedule = swap_schedule if swap_schedule is not None else seed_swap_schedule
            stack = build_forward_stack(filaments, constants=constants, swap_schedule=manual_schedule)
            base_factor = mesh_params.print_base_factor
            if base_factor is None:
                base_factor = build_height_field(
                    ImageOps.exif_transpose(image), stack, layer_params=layer_params, mesh_params=mesh_params,
                    luminance_params=luminance_params, orientation=orientation,
                )[1]
            grid = finished_grid(image, stack, {**vars(mesh_params), "orientation": orientation}, finish, base_factor)
        except ValueError as exc:
            raise QuiltInputError(str(exc)) from exc
    if mode == "optimize":
        optimized = optimize_height_field(
            image,
            filaments,
            layer_params=layer_params,
            mesh_params=mesh_params,
            luminance_params=luminance_params,
            swap_schedule=swap_schedule,
            seed_swap_schedule=seed_swap_schedule,
            orientation=orientation,
            params=optimize_params,
        )
        output_filaments = optimized.order
        schedule = optimized.swap_schedule
        table = optimized.forward_table
        height_field = optimized.height_field
        downsample_factor = optimized.downsample_factor
        expected_count = optimized.exact_triangle_count
        saddle_clips = optimized.saddle_clips
        optimize_report = optimized.report
        if optimize_params and optimize_params.height_strength:
            # Compare the two winning fields, preserving the focus weights, search
            # budget, seed and mesh cap on the zero-strength pass.
            baseline = optimize_height_field(
                image, filaments, layer_params=layer_params, mesh_params=mesh_params,
                luminance_params=luminance_params, swap_schedule=swap_schedule,
                seed_swap_schedule=seed_swap_schedule, orientation=orientation,
                params=replace(optimize_params, height_strength=0, depth_map=None),
            )
            def colour_error(result):
                mask = _transparent_mask(image, result.height_field.shape) if mesh_params.flatten_transparent else None
                return preview_loss(image, result.height_field, result.forward_table,
                                    transparent_mask=mask, luminance_params=luminance_params)

            optimize_report["source_colour_mse"] = colour_error(optimized)
            optimize_report["baseline_colour_mse"] = colour_error(baseline)
        if grid is not None:
            optimize_report = {**optimize_report, "metrics_stage": "before_print_finishing"}
    else:
        # GPU admission can select the naive fallback before entering the
        # optimizer. Use the same feasible baseline on both fallback paths.
        output_filaments = filaments
        if optimize_params and optimize_params.search_stacks:
            indices = next(stack_orders(filaments, layer_params, optimize_params, swap_schedule))
            output_filaments = tuple(filaments[i] for i in indices)
            schedule = baseline_schedule(len(indices), layer_params, swap_schedule, seed_swap_schedule)
        else:
            schedule = derive_swap_schedule(
                len(filaments), total_layers=layer_params.total_layers,
                base_layers=layer_params.base_layers, explicit=swap_schedule,
            )
        table = build_forward_table(output_filaments, schedule, layer_params=layer_params)
        if grid is None:
            height_field, downsample_factor, expected_count, saddle_clips = build_height_field(
                image,
                table,
                layer_params=layer_params,
                mesh_params=mesh_params,
                luminance_params=luminance_params,
                orientation=orientation,
            )
    if grid is not None:
        if mode == "optimize":
            # Sample the optimized heights in the same cropped source coordinates as the manual preview.
            raw = Image.fromarray(height_field.astype(np.float32)).resize(image.size, Image.Resampling.NEAREST)
            raw.info.update(image.info)
            raw = ImageOps.exif_transpose(raw)
            cropped = raw.crop((finish.left, finish.top, raw.width - finish.right, raw.height - finish.bottom))
            sampled = cropped.resize(grid.heights.shape[::-1], Image.Resampling.BOX)
            height_field = quantize_detail(np.asarray(sampled), finish, grid.heights > 0)
        else:
            height_field = grid.heights
        mesh_params = replace(mesh_params, xy_pixel_mm=grid.pitch)
        downsample_factor = grid.factor
        expected_count = saddle_clips = None
    mesh, metadata = build_mesh(
        height_field,
        layer_params=layer_params,
        mesh_params=mesh_params,
        full_validate_max_triangles=full_validate_max_triangles,
        through_cuts=grid is not None,
        precomputed_count=expected_count,
        precomputed_clips=saddle_clips,
    )
    metadata = {**metadata, "downsample_factor": downsample_factor}
    if calibration_catalog():
        metadata["prediction_sources_by_layer"] = list(table.layer_sources)
        metadata["preview_prediction_sources"] = sources_in_field(
            table, height_field, height_field == 0 if grid is not None else None
        )
    if optimize_report is not None:
        metadata["optimize"] = optimize_report
    preview = render_preview(height_field, table)
    if grid is not None and np.any(height_field == 0):
        preview.putalpha(Image.fromarray(((height_field > 0) * 255).astype(np.uint8)))
    return GenerateResult(
        height_field=height_field,
        forward_table=table,
        output_filaments=tuple(output_filaments),
        swap_schedule=schedule,
        layer_params=layer_params,
        swap_text=swap_instructions(output_filaments, schedule, layer_height_mm=layer_params.layer_height_mm),
        preview=preview,
        mesh=mesh,
        mesh_params=mesh_params,
        stl_bytes=mesh_to_stl_bytes(mesh),
        metadata=metadata,
        optimize_report=optimize_report,
    )


def _filaments_from_palette(palette: PaletteCandidate | Sequence[Filament]) -> tuple[Filament, ...]:
    if isinstance(palette, PaletteCandidate):
        filaments = palette.filaments
    else:
        filaments = tuple(palette)
    if not filaments:
        raise QuiltInputError("palette must contain at least one filament")
    return tuple(filaments)
