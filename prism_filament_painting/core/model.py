"""Core dataclasses for the palette backbone."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from functools import wraps


def _accept_input_alias(cls):
    """Keep the published TD keyword accepted by earlier Python spool readers."""
    init = cls.__init__

    @wraps(init)
    def initialize(self, *args, **kwargs):
        if "td" in kwargs:
            # dataclasses.replace supplies the existing canonical field as well.
            kwargs["input_distance_mm"] = kwargs.pop("td")
        init(self, *args, **kwargs)

    cls.__init__ = initialize
    return cls


@_accept_input_alias
@dataclass(frozen=True)
class Filament:
    brand: str
    name: str
    hex: str
    input_distance_mm: float

    @property
    def td(self) -> float:
        """Published input in mm, as used by existing spool readers."""
        return self.input_distance_mm


@_accept_input_alias
@dataclass(frozen=True)
class IdentifiedFilament(Filament):
    spool_id: str


def physical_filament(brand, name, hex, input_distance_mm, spool_id=None):
    if spool_id:
        return IdentifiedFilament(brand, name, hex, input_distance_mm, spool_id)
    return Filament(brand, name, hex, input_distance_mm)


@dataclass(frozen=True)
class PaletteCandidate:
    """A printable global filament stack; filaments[0] is the bottom band."""

    filaments: tuple[Filament, ...]
    score: float
    notes: str = ""
    swap_schedule: tuple[int, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
