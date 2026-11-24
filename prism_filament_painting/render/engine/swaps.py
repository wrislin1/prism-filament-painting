"""Filament swap schedule helpers."""

from __future__ import annotations

from collections.abc import Sequence

from prism_filament_painting.core.model import Filament

from prism_filament_painting.render.errors import QuiltInputError


def derive_swap_schedule(
    palette_size: int,
    *,
    total_layers: int,
    base_layers: int = 1,
    explicit: Sequence[int] | None = None,
    strategy: str = "even",
) -> tuple[int, ...]:
    if palette_size < 1:
        raise QuiltInputError("palette_size must be at least 1")
    if total_layers <= base_layers:
        raise QuiltInputError("total_layers must be greater than base_layers")
    expected = palette_size - 1
    if explicit is not None:
        schedule = _strict_int_tuple(explicit)
        _validate_schedule(schedule, expected=expected, base_layers=base_layers, total_layers=total_layers)
        return schedule
    if strategy != "even":
        raise QuiltInputError(f"unsupported swap strategy: {strategy}")
    if expected == 0:
        return ()
    if total_layers - base_layers - 1 < expected:
        raise QuiltInputError("not enough printable layers for palette size")
    span = total_layers - base_layers
    schedule = tuple(round(base_layers + (index * span / palette_size)) for index in range(1, palette_size))
    _validate_schedule(schedule, expected=expected, base_layers=base_layers, total_layers=total_layers)
    return schedule


def swap_instructions(
    filaments: Sequence[Filament],
    swap_schedule: Sequence[int],
    *,
    layer_height_mm: float,
) -> str:
    if not filaments:
        raise QuiltInputError("filaments must not be empty")
    if len(swap_schedule) != len(filaments) - 1:
        raise QuiltInputError("swap_schedule length must be one less than filament count")
    lines = [
        (
            f"Start with {filaments[0].brand} {filaments[0].name} "
            f"({filaments[0].hex}, TD {filaments[0].td:.2f} mm)"
        )
    ]
    for layer, filament in zip(swap_schedule, filaments[1:], strict=True):
        lines.append(
            f"At layer {layer} (Z = {layer * layer_height_mm:.2f} mm), "
            f"change to {filament.brand} {filament.name} "
            f"({filament.hex}, TD {filament.td:.2f} mm)"
        )
    return "\n".join(lines) + "\n"


def _validate_schedule(
    schedule: tuple[int, ...],
    *,
    expected: int,
    base_layers: int,
    total_layers: int,
) -> None:
    if len(schedule) != expected:
        raise QuiltInputError("swap_schedule length must be palette_size - 1")
    previous = base_layers
    for layer in schedule:
        if layer <= base_layers or layer >= total_layers:
            raise QuiltInputError("swap layer must be inside (base_layers, total_layers)")
        if layer <= previous:
            raise QuiltInputError("swap layers must be strictly increasing")
        previous = layer


def _strict_int_tuple(values: Sequence[int]) -> tuple[int, ...]:
    schedule: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise QuiltInputError("swap_schedule must contain integers")
        schedule.append(value)
    return tuple(schedule)
