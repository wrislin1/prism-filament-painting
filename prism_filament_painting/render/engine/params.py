"""Data containers for the Quilt engine."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image
from prism_filament_painting.core.finishing import PrintOptions
from prism_filament_painting.core.forward import RenderConstants
from prism_filament_painting.core.constraints import (
    LUMINANCE_BLACK_MIN,
    LUMINANCE_GAMMA_MAX,
    LUMINANCE_GAMMA_MIN,
    LUMINANCE_WHITE_MAX,
)
from prism_filament_painting.core.model import Filament

if TYPE_CHECKING:
    from .mesh import QuiltMesh


@dataclass(frozen=True)
class LayerParams:
    layer_height_mm: float = 0.08
    base_layers: int = 1
    total_layers: int = 16
    td_scale: float = RenderConstants().td_scale


@dataclass(frozen=True)
class MeshParams:
    max_dim_mm: float = 300.0
    max_triangles: int = 1_500_000
    xy_pixel_mm: float = 0.4
    flatten_transparent: bool = True
    print_options: PrintOptions = PrintOptions()
    print_base_factor: int | None = None


@dataclass(frozen=True)
class LuminanceParams:
    black: float = 0.0
    white: float = 1.0
    gamma: float = 1.0

    @property
    def is_identity(self) -> bool:
        return self.black == 0.0 and self.white == 1.0 and self.gamma == 1.0


def validate_luminance_params(params: LuminanceParams) -> LuminanceParams:
    if not all(math.isfinite(value) for value in (params.black, params.white, params.gamma)):
        raise ValueError("luminance params must be finite")
    if not LUMINANCE_BLACK_MIN <= params.black < params.white <= LUMINANCE_WHITE_MAX:
        raise ValueError("lum_black must be >= 0, lum_white must be <= 1, and lum_black must be less than lum_white")
    if not LUMINANCE_GAMMA_MIN <= params.gamma <= LUMINANCE_GAMMA_MAX:
        raise ValueError("lum_gamma must be between 0.1 and 5.0")
    return params


@dataclass(frozen=True)
class ForwardTable:
    layer_rgb: tuple[tuple[int, int, int], ...]
    layer_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerateResult:
    height_field: np.ndarray
    forward_table: ForwardTable
    output_filaments: tuple[Filament, ...]
    swap_schedule: tuple[int, ...]
    layer_params: LayerParams
    swap_text: str
    preview: Image.Image
    mesh: QuiltMesh
    mesh_params: MeshParams
    stl_bytes: bytes
    metadata: dict[str, Any]
    optimize_report: dict[str, Any] | None = None


@dataclass(frozen=True)
class OptimizeParams:
    iterations: int = 48
    learning_rate: float = 0.06
    seed: int = 0
    device: str = "auto"
    max_pixels: int = 65_536
    allow_cpu: bool = False
    max_order_permutations: int = 720
    max_order_candidates: int = 8
    max_schedule_candidates: int = 64
    max_candidate_optimizations: int = 32
    snap_radius: int = 1
    pin_order: bool = False
    max_filaments: int | None = None
    max_swaps: int | None = None
    max_layers: int | None = None
    pinned_filaments: tuple[int | None, ...] = ()
    wide_search: bool = False
    focus_mask: Image.Image | None = None
    depth_map: Image.Image | None = None
    height_strength: float = 0.0

    @property
    def search_stacks(self) -> bool:
        return (
            self.wide_search
            or any(i is not None for i in self.pinned_filaments)
            or any(v is not None for v in (self.max_filaments, self.max_swaps, self.max_layers))
        )


@dataclass(frozen=True)
class OptimizeResult:
    height_field: np.ndarray
    forward_table: ForwardTable
    order: tuple[Filament, ...]
    order_indices: tuple[int, ...]
    swap_schedule: tuple[int, ...]
    downsample_factor: int
    exact_triangle_count: int
    saddle_clips: np.ndarray
    report: dict[str, Any]
