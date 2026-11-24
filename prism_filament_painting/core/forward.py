"""forward simulation: luminance -> stack height -> TD-composited color.

Ranking and rendering share the band compositor. The luminance and orientation
functions mirror the renderer's height-field logic. This module raises ``ValueError`` and
imports only numpy + ``prism_filament_painting.core.color``; PIL is used only via the passed
image object (no top-level import).

``luminance_to_layer`` is the opaque luminance core (luminance -> layer index).
``image_to_layer_field`` is the full RGBA-parity entry: it adds the renderer's
``flatten_transparent`` step (``alpha == 0`` pixels forced to ``base_layers``)
and is what reproduces ``_field_from_rgba`` for transparent images. The parity
test locks ``image_to_layer_field`` against the renderer for both flatten settings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .color import (
    beer_lambert_blend,
    hex_to_rgb,
    relative_luminance,
    relative_luminance_array,
)
from .model import Filament
from .prediction import predict_stack, select_calibration


@dataclass(frozen=True)
class RenderConstants:
    """Print geometry and tunable front-lit rendering approximations."""

    layer_height_mm: float = 0.08
    base_layers: int = 2
    total_layers: int = 27
    black_point: float = 0.0  # mirrors renderer LuminanceParams.black
    white_point: float = 1.0  # mirrors renderer LuminanceParams.white
    gamma: float = 1.0
    front_lit_scale: float = 0.1

    @property
    def td_scale(self) -> float:
        """Scale used by earlier editor integrations."""
        return self.front_lit_scale


@dataclass(frozen=True)
class ForwardStack:
    """Per-height surface-color LUT for one ordered stack.

    ``layer_rgb[i]`` is the visible color when the stack is printed up to (and
    including) layer ``i``; index range is ``[0, total_layers]``.
    """

    layer_rgb: np.ndarray  # (total_layers + 1, 3) uint8
    constants: RenderConstants
    layer_sources: tuple[str, ...] = ()


def even_swap_schedule(
    palette_size: int,
    *,
    base_layers: int,
    total_layers: int,
) -> tuple[int, ...]:
    """Even contiguous swap schedule. Matches renderer ``derive_swap_schedule`` (even)."""
    if palette_size < 1:
        raise ValueError("palette_size must be at least 1")
    if total_layers <= base_layers:
        raise ValueError("total_layers must be greater than base_layers")
    expected = palette_size - 1
    if expected == 0:
        return ()
    if total_layers - base_layers - 1 < expected:
        raise ValueError("not enough printable layers for palette size")
    span = total_layers - base_layers
    return tuple(
        round(base_layers + (index * span / palette_size))
        for index in range(1, palette_size)
    )


def _validate_schedule(
    schedule: Sequence[int],
    *,
    expected: int,
    base_layers: int,
    total_layers: int,
) -> None:
    """Strictly validate an explicit swap schedule. Mirrors renderer ``_validate_schedule``.

    Length must be ``expected`` (filament count - 1); each layer must be a plain
    ``int`` (not ``bool``), strictly inside ``(base_layers, total_layers)``, and
    strictly increasing. Raises ``ValueError`` so an out-of-range/non-monotonic
    schedule fails loudly instead of silently producing empty bands.
    """
    if len(schedule) != expected:
        raise ValueError("swap_schedule length must be one less than filament count")
    previous = base_layers
    for layer in schedule:
        if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
            raise ValueError("swap layers must be integers")
        value = int(layer)
        if value <= base_layers or value >= total_layers:
            raise ValueError("swap layer must be inside (base_layers, total_layers)")
        if value <= previous:
            raise ValueError("swap layers must be strictly increasing")
        previous = value


def schedule_in_range(
    schedule: Sequence[int],
    base_layers: int,
    total_layers: int,
) -> bool:
    """Boolean: do all schedule entries lie strictly inside ``(base_layers, total_layers)``?

    Range-only sibling of :func:`_validate_schedule` for boundaries that don't know the
    expected length (e.g. the geometry form, where the palette is separate).
    Single source for the open-interval membership test so it can't drift across callers.
    """
    for layer in schedule:
        if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
            return False
        if not (base_layers < int(layer) < total_layers):
            return False
    return True


def is_valid_schedule(
    schedule: Sequence[int],
    *,
    expected: int,
    base_layers: int,
    total_layers: int,
) -> bool:
    """Boolean wrapper over :func:`_validate_schedule` (range + length + strictly increasing).

    For callers that want a fit predicate rather than a raising validator (e.g. Studio
    boot deciding whether to keep a candidate schedule or derive an even fallback).
    """
    try:
        _validate_schedule(
            schedule, expected=expected, base_layers=base_layers, total_layers=total_layers
        )
        return True
    except ValueError:
        return False


def build_forward_stack(
    filaments: Sequence[Filament],
    *,
    constants: RenderConstants = RenderConstants(),
    swap_schedule: Sequence[int] | None = None,
) -> ForwardStack:
    """Composite an ordered N-filament stack top-down via Beer-Lambert physics.

    An explicit ``swap_schedule`` is strictly validated (range, monotonicity,
    integer type); the auto-derived even schedule is trusted, matching the renderer's
    ``derive_swap_schedule`` (validate explicit, trust derived) so this stays
    bit-for-bit equivalent to ``build_forward_table``.
    """
    if not filaments:
        raise ValueError("filaments must not be empty")
    if constants.total_layers < 0:
        raise ValueError("total_layers must be non-negative")
    if swap_schedule is None:
        swap_schedule = even_swap_schedule(
            len(filaments),
            base_layers=constants.base_layers,
            total_layers=constants.total_layers,
        )
    else:
        _validate_schedule(
            swap_schedule,
            expected=len(filaments) - 1,
            base_layers=constants.base_layers,
            total_layers=constants.total_layers,
        )

    colors, sources = predictions_by_layer(
        filaments,
        swap_schedule,
        layer_height_mm=constants.layer_height_mm,
        total_layers=constants.total_layers,
        front_lit_scale=constants.front_lit_scale,
    )
    if len(colors) != constants.total_layers + 1:
        raise ValueError("forward stack layer count mismatch")
    return ForwardStack(layer_rgb=np.asarray(colors, dtype=np.uint8), constants=constants, layer_sources=sources)


def formula_colors_by_layer(
    filaments: Sequence[Filament],
    swap_schedule: Sequence[int],
    *,
    layer_height_mm: float,
    total_layers: int,
    front_lit_scale: float,
) -> tuple[tuple[int, int, int], ...]:
    """Composite each band over the previous band's final colour, given a checked schedule."""
    filament_rgb = tuple(hex_to_rgb(filament.hex) for filament in filaments)
    base_rgb = filament_rgb[0]
    schedule = tuple(int(value) for value in swap_schedule)

    layer_rgb: list[tuple[int, int, int]] = []
    current = base_rgb
    first_swap = schedule[0] if schedule else total_layers + 1
    layer_rgb.extend(base_rgb for _ in range(min(first_swap, total_layers + 1)))
    for index, start_layer in enumerate(schedule, start=1):
        next_start = schedule[index] if index < len(schedule) else total_layers + 1
        for layer in range(start_layer, min(next_start, total_layers + 1)):
            printed_layers = layer - start_layer + 1
            layer_rgb.append(
                beer_lambert_blend(
                    filament_rgb[index],
                    current,
                    thickness_mm=printed_layers * layer_height_mm,
                    hiding_distance_mm=filaments[index].input_distance_mm * front_lit_scale,
                )
            )
        if layer_rgb:
            current = layer_rgb[-1]
    return tuple(layer_rgb)


def predictions_by_layer(filaments, swap_schedule, *, layer_height_mm, total_layers, front_lit_scale):
    formula = formula_colors_by_layer(filaments, swap_schedule, layer_height_mm=layer_height_mm,
                                total_layers=total_layers, front_lit_scale=front_lit_scale)
    book = select_calibration(filaments, filaments[0], layer_height_mm)
    if book is None:
        predictions = [predict_stack(rgb, None) for rgb in formula]
        return tuple(p.rgb for p in predictions), tuple(p.source for p in predictions)
    colors, sources, bands = [], [], []
    swaps = dict(zip(swap_schedule, filaments[1:]))
    backing_layers = swap_schedule[0] - 1 if len(swap_schedule) else total_layers
    for height, plain in enumerate(formula):
        if height in swaps:
            bands.append((swaps[height].spool_id, 0))
        if bands:
            spool, count = bands[-1]
            bands[-1] = (spool, count + 1)
        backing_mm = min(height, backing_layers) * layer_height_mm
        prediction = predict_stack(plain, book if book.supports_backing(backing_mm) else None, tuple(bands))
        colors.append(prediction.rgb)
        sources.append(prediction.source)
    return tuple(colors), tuple(sources)


def surface_colors_by_layer(filaments, swap_schedule, *, layer_height_mm, total_layers, front_lit_scale):
    colors, _ = predictions_by_layer(filaments, swap_schedule, layer_height_mm=layer_height_mm,
                                     total_layers=total_layers, front_lit_scale=front_lit_scale)
    return colors


def resolve_orientation(stack: ForwardStack, *, orientation: str = "auto") -> str:
    """Resolve "auto" to "dark"/"bright" via the tall-stack-is-dark rule.

    For a dark->light (luminance-ascending) stack the top is brighter than the
    bottom, so this returns "bright": bright image pixels map to tall heights and
    therefore to the bright top filaments.
    """
    if orientation in {"dark", "bright"}:
        return orientation
    if orientation != "auto":
        raise ValueError("orientation must be auto, dark, or bright")
    base = stack.constants.base_layers
    total = stack.constants.total_layers
    low = relative_luminance(tuple(int(c) for c in stack.layer_rgb[base]))
    high = relative_luminance(tuple(int(c) for c in stack.layer_rgb[total]))
    return "dark" if high <= low else "bright"


def luminance_to_layer(
    luminance,
    *,
    constants: RenderConstants,
    orientation: str,
) -> np.ndarray:
    """Map luminance (0..1) to integer layer index. Ports renderer ``_field_from_rgba``.

    ``orientation`` must be "dark" or "bright" (resolve "auto" via
    :func:`resolve_orientation` first).
    """
    heights = continuous_layer_height(luminance, constants=constants, orientation=orientation)
    return np.clip(np.rint(heights).astype(np.int32), constants.base_layers, constants.total_layers).astype(np.int32)


def continuous_layer_height(luminance, *, constants: RenderConstants, orientation: str) -> np.ndarray:
    """Map luminance to layer height before spatial filtering or integer quantization."""
    if orientation not in {"dark", "bright"}:
        raise ValueError("orientation must be dark or bright")
    lum = np.asarray(luminance, dtype=float)
    span = constants.total_layers - constants.base_layers
    black, white, gamma = constants.black_point, constants.white_point, constants.gamma
    is_identity = black == 0.0 and white == 1.0 and gamma == 1.0
    if not is_identity:
        lum = np.clip((lum - black) / (white - black), 0.0, 1.0)
        lum = lum**gamma
    normalized = (1.0 - lum) if orientation == "dark" else lum
    return constants.base_layers + normalized * span


def response_curve(
    stack: ForwardStack,
    *,
    n_levels: int = 256,
    orientation: str = "auto",
) -> np.ndarray:
    """Predicted visible color at ``n_levels`` evenly spaced luminance samples."""
    resolved = resolve_orientation(stack, orientation=orientation)
    levels = np.linspace(0.0, 1.0, n_levels)
    index = luminance_to_layer(levels, constants=stack.constants, orientation=resolved)
    return stack.layer_rgb[index]


def image_to_layer_field(
    image,
    *,
    constants: RenderConstants,
    orientation: str,
    flatten_transparent: bool = True,
) -> np.ndarray:
    """Map an RGBA image to integer layer indices. Full port of renderer ``_field_from_rgba``.

    ``orientation`` must already be resolved ("dark"/"bright"). When
    ``flatten_transparent`` is true, fully-transparent pixels (``alpha == 0``)
    are forced to ``base_layers`` after the luminance mapping — exactly how the renderer
    prints a transparent background as the base filament. PIL is used only via
    the passed image object's ``.mode``/``.convert`` (no import).
    """
    rgba = image if image.mode == "RGBA" else image.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8)
    lum = relative_luminance_array(arr[..., :3])
    field = luminance_to_layer(lum, constants=constants, orientation=orientation)
    if flatten_transparent:
        field = field.copy()
        field[arr[..., 3] == 0] = constants.base_layers
    return field


def preview_sources(image, stack, transparent_mask=None):
    orientation = resolve_orientation(stack)
    field = image_to_layer_field(image, constants=stack.constants, orientation=orientation)
    visible = field if transparent_mask is None else field[~transparent_mask]
    return sorted({stack.layer_sources[i] for i in np.unique(visible)})


def render_preview(
    image,
    stack: ForwardStack,
    *,
    orientation: str = "auto",
    flatten_transparent: bool = True,
):
    """Full predicted preview image for the stack. Imports PIL lazily.

    Models transparency the way the renderer prints it: with ``flatten_transparent``
    (default), ``alpha == 0`` pixels render as the base filament.
    """
    from PIL import Image  # noqa: PLC0415 — keep PIL a soft dependency of palette-core

    resolved = resolve_orientation(stack, orientation=orientation)
    field = image_to_layer_field(
        image,
        constants=stack.constants,
        orientation=resolved,
        flatten_transparent=flatten_transparent,
    )
    return Image.fromarray(stack.layer_rgb[field], mode="RGB")
