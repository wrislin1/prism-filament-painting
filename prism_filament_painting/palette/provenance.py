"""Origins of the optical inputs used by Prism."""
from dataclasses import dataclass

from prism_filament_painting.core.model import Filament, _accept_input_alias

LEVELS = ("matched", "vendor", "measured", "estimated")


def level(value: str) -> str:
    if value not in LEVELS:
        raise ValueError(f"provenance must be one of {', '.join(LEVELS)}")
    return value


@_accept_input_alias
@dataclass(frozen=True)
class SourcedFilament(Filament):
    opacity_level: str
    color_level: str


@_accept_input_alias
@dataclass(frozen=True)
class IdentifiedSourcedFilament(SourcedFilament):
    spool_id: str


def sourced_filament(brand, name, hex, td, opacity_level, color_level, spool_id=None):
    if spool_id:
        return IdentifiedSourcedFilament(brand, name, hex, td, opacity_level, color_level, spool_id)
    return SourcedFilament(brand, name, hex, td, opacity_level, color_level)
