"""Read JSON, CSV and TSV spool inventories."""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from prism_filament_painting.core.model import Filament
from prism_filament_painting.core.prediction import calibration_catalog

from .materials import MaterialSpool, material_group
from .names import normalize_key as _normalize_key
from .opacity_model import estimate_opacity
from .provenance import SourcedFilament, level, sourced_filament

# A leading "TD1" token: case-insensitive, must be followed by a separator/whitespace or
# end-of-string (the negative lookahead) so a real brand like "TD1Pro" is NOT a false
# positive. Consumes the token plus any immediate separators.
_MARKER_RE = re.compile(r"^\s*td1(?![a-z0-9])[\s:._\-]*", re.IGNORECASE)

_HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")
_KEYS = {
    "brand": ("brand", "manufacturer", "maker", "vendor"),
    "name": ("name", "filament", "colorname", "colourname"),
    "hex": ("hex", "hexcode", "hexcolor", "hexcolour", "colorhex", "colourhex", "color", "colour"),
    # Some spool lists label transmission distance in millimetres as Transmissivity.
    "td": (
        "td",
        "tdmm",
        "inputdistancemm",
        "transmission",
        "transmissiondistance",
        "transmissiondistance(mm)",
        "transmissivity",
        "transmissivity(mm)",
    ),
}


def parse_inventory(path: str | Path | None = None, *, owned_only: bool = False, allow_estimates: bool = True) -> list[Filament]:
    """Parse a JSON, CSV or TSV spool list, or return an empty inventory when no path is supplied."""
    return [s.value for s in inventory_spools(path, owned_only=owned_only, allow_estimates=allow_estimates)]


def spool_from_row(row, *, row_number, allow_estimate=False):
    fields = {_normalize_key(k): v for k, v in row.items()}
    material = next((fields[k] for k in ("material", "type", "filamenttype", "materialtype") if fields.get(k)), "")
    return MaterialSpool(_filament_from_row(row, row_number=row_number, allow_estimate=allow_estimate),
                         material_group(material, fields.get("materialfamily")), owned_is_true(fields.get("owned")))


def inventory_spools(path=None, *, owned_only=False, allow_estimates=True):
    if path is None:
        return []
    source = Path(path)
    rows = inventory_rows(source)
    filaments = []
    for i, row in enumerate(rows):
        if owned_only and any(
            _normalize_key(key) == "owned" and not owned_is_true(value)
            for key, value in row.items()
        ):
            continue
        filaments.append(spool_from_row(row, row_number=i + 1, allow_estimate=allow_estimates))
    return filaments


def owned_is_true(value: Any) -> bool:
    """Explicit truthiness for the ``Owned`` flag (no Python ``bool()`` surprises).

    Real bools pass through; numbers are nonzero-true; strings are true only for an explicit
    affirmative set so a future export serializing ``Owned: "false"`` is not treated as owned.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"true", "1", "yes", "y", "on"}


def iter_inventory_dicts(payload: Any) -> list[Mapping[str, Any]]:
    """Unwrap a parsed JSON inventory payload into a list of raw row mappings.

    Accepts a top-level list, or an object wrapping the list under one of
    ``filaments``/``Filaments``/``library``/``Library``.
    Rows are returned unchanged so callers that need additional fields like
    ``Owned``/``Type`` (the DB importer) can read them; :func:`parse_inventory`
    maps the rows into :class:`Filament` objects and drops those extra fields.
    """
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = next(
            (
                payload[key]
                for key in ("filaments", "Filaments", "library", "Library")
                if isinstance(payload.get(key), list)
            ),
            None,
        )
        if rows is None:
            raise ValueError("JSON inventory must be a list or contain a filament list")
    else:
        raise ValueError("JSON inventory must be an object or list")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("JSON inventory rows must be objects")
    return list(rows)


def inventory_rows(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".json":
        return iter_inventory_dicts(json.loads(path.read_text(encoding="utf-8-sig")))
    if path.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError(f"unsupported inventory extension: {path.suffix}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter="\t" if "\t" in header else ",",
                                skipinitialspace=True, strict=True)
        names = [_normalize_key(k) for k in reader.fieldnames or ()]
        if len(set(names)) != len(names) or not all(
            any(alias in names for alias in _KEYS[key]) for key in ("brand", "name", "hex")
        ):
            raise ValueError("spool list requires distinct brand, name and hex headers")
        rows = []
        for row in reader:
            if None in row:
                raise ValueError(f"too many columns at line {reader.line_num}")
            # These exports describe a personal spool list. Preserve an explicit ownership flag.
            if "owned" not in names:
                row["Owned"] = True
            if not {"type", "material", "filamenttype", "materialtype"}.intersection(names):
                row["Type"] = "Unspecified"
            rows.append(row)
        return rows


def _filament_from_row(row: Mapping[str, Any], *, row_number: int, allow_estimate: bool = False) -> Filament:
    values = {_normalize_key(key): value for key, value in row.items()}
    try:
        brand = str(_pick(values, _KEYS["brand"])).strip()
        name = str(_pick(values, _KEYS["name"])).strip()
        hex_value = _normalize_hex(str(_pick(values, _KEYS["hex"])).strip())
        try:
            raw_distance = _pick(values, _KEYS["td"])
        except KeyError:
            if not allow_estimate:
                raise
            raw_distance = estimate_opacity(hex_value, " ".join(
                str(values.get(k, "")) for k in ("brand", "name", "colorname", "type", "material", "filamenttype", "materialtype", "finish")
            ))
            if raw_distance is None:
                raise ValueError("no opacity estimate available; supply TD or a calibration")
            values["opacitylevel"] = "estimated"
        td = float(raw_distance)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed inventory row {row_number}: {exc}") from exc

    if not brand:
        raise ValueError(f"malformed inventory row {row_number}: missing brand")
    if not name:
        raise ValueError(f"malformed inventory row {row_number}: missing name")
    if not _is_positive_td(td):
        raise ValueError(f"malformed inventory row {row_number}: td must be finite and positive")
    opacity = values.get("opacitylevel")
    if allow_estimate and not opacity:
        opacity = "measured" if strip_measured_marker(brand, name)[2] else "matched"
    opacity = level(str(opacity or "matched"))
    color = level(str(values.get("colorlevel") or "matched"))
    lot = values.get("activelot") or values.get("spoolid")
    if lot and calibration_catalog():
        return sourced_filament(brand, name, hex_value, td, opacity, color, str(lot))
    if (opacity, color) != ("matched", "matched"):
        return SourcedFilament(brand, name, hex_value, td, opacity, color)
    return Filament(brand, name, hex_value, td)


def strip_measured_marker(brand: str, name: str) -> tuple[str, str, bool]:
    """Strip a leading ``TD1`` token from ``brand`` and ``name``; report if found.

    Return trimmed brand and name with a flag indicating a marked measurement.
    """
    cleaned_brand, brand_hit = _strip_one(brand)
    cleaned_name, name_hit = _strip_one(name)
    return cleaned_brand, cleaned_name, (brand_hit or name_hit)


def _strip_one(value: str | None) -> tuple[str, bool]:
    text = (value or "").strip()
    match = _MARKER_RE.match(text)
    if not match:
        return text, False
    return text[match.end():].strip(), True


def _is_positive_td(value: Any) -> bool:
    """The canonical 'usable transmission distance' rule: finite and strictly positive."""
    try:
        td = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(td) and td > 0


def _pick(values: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = values.get(_normalize_key(name))
        if value not in (None, ""):
            return value
    raise KeyError(f"missing required key ({', '.join(names)})")


def _normalize_hex(value: str) -> str:
    if not _HEX_RE.match(value):
        raise ValueError("invalid hex color")
    if not value.startswith("#"):
        value = f"#{value}"
    return value.upper()
