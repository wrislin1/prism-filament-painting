"""Beer-Lambert forward model for a global filament stack."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from prism_filament_painting.core.forward import predictions_by_layer
from prism_filament_painting.core.model import Filament

from prism_filament_painting.render.errors import QuiltInputError, QuiltInternalError

from .params import ForwardTable, LayerParams


def build_forward_table(
    filaments: Sequence[Filament],
    swap_schedule: Sequence[int],
    *,
    layer_params: LayerParams,
) -> ForwardTable:
    if not filaments:
        raise QuiltInputError("filaments must not be empty")
    if len(swap_schedule) != len(filaments) - 1:
        raise QuiltInputError("swap_schedule length must be one less than filament count")
    if layer_params.total_layers < 0:
        raise QuiltInputError("total_layers must be non-negative")
    layer_rgb, sources = predictions_by_layer(
        filaments,
        swap_schedule,
        layer_height_mm=layer_params.layer_height_mm,
        total_layers=layer_params.total_layers,
        front_lit_scale=layer_params.td_scale,
    )
    if len(layer_rgb) != layer_params.total_layers + 1:
        raise QuiltInternalError("forward table layer count mismatch")
    return ForwardTable(layer_rgb=layer_rgb, layer_sources=sources)


def sources_in_field(table, field, transparent_mask=None):
    visible = field if transparent_mask is None else field[~transparent_mask]
    return sorted({table.layer_sources[i] for i in np.unique(visible)})


def surface_color_at(table: ForwardTable, layer_index: int) -> tuple[int, int, int]:
    if layer_index < 0:
        raise QuiltInputError("layer_index outside forward table")
    try:
        return table.layer_rgb[layer_index]
    except IndexError as exc:
        raise QuiltInputError("layer_index outside forward table") from exc
