"""Read a user's resolved slicer project and select settings for one print."""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path
from zipfile import BadZipFile, ZipFile, is_zipfile

from ..errors import QuiltInputError

_MEMBER = "Metadata/project_settings.config"
_LIMIT = 4 * 1024 * 1024
_MATERIAL_NUMBERS = ("filament_diameter", "filament_flow_ratio", "filament_density",
                     "nozzle_temperature", "nozzle_temperature_initial_layer")


def read_project_settings(stream):
    """Read a JSON object or one bounded settings member, without extracting a project."""
    if isinstance(stream, Path):
        with stream.open("rb") as source:
            return read_project_settings(source)
    try:
        if stream.seek(0, 2) > 64 * 1024 * 1024:
            raise QuiltInputError("Export the settings as JSON or use a project smaller than 64 MiB.")
        stream.seek(0)
        if is_zipfile(stream):
            stream.seek(0)
            with ZipFile(stream) as archive:
                members = [item for item in archive.infolist() if item.filename == _MEMBER]
                if len(members) != 1 or members[0].file_size > _LIMIT:
                    raise QuiltInputError("Export a project containing one settings file under 4 MiB.")
                data = archive.read(members[0])
        else:
            stream.seek(0)
            data = stream.read(_LIMIT + 1)
            if len(data) > _LIMIT:
                raise QuiltInputError("Export settings as a JSON file smaller than 4 MiB.")
    except (BadZipFile, RuntimeError, EOFError, zlib.error) as exc:
        raise QuiltInputError("Export a readable settings JSON or an unencrypted slicer project.") from exc
    try:
        source = json.loads(data)
    except (ValueError, RecursionError) as exc:
        raise QuiltInputError("Export a readable settings JSON or an unencrypted slicer project.") from exc
    if not isinstance(source, dict):
        raise QuiltInputError("Export the resolved printer, process and filament settings together.")
    return source


def _number(value, key, *, positive=True):
    try:
        if isinstance(value, bool):
            raise ValueError
        number = float(value)
        if not math.isfinite(number) or (positive and number <= 0):
            raise ValueError
    except (ValueError, TypeError, OverflowError) as exc:
        raise QuiltInputError(f"Check {key} in the slicer and export the settings again.") from exc
    return number


def _values(source, key, count):
    values = source.get(key)
    if not isinstance(values, list) or len(values) != count:
        quantity = "one entry" if count == 1 else f"{count} entries"
        raise QuiltInputError(f"Export {quantity} for {key} in the planned order.")
    return values


def _name(value, key):
    if not isinstance(value, str) or not value.strip() or len(value) > 256 or not value.isprintable():
        raise QuiltInputError(f"Choose a named {key} preset and export the resolved project.")
    return value


def _rectangle(source):
    points = _values(source, "printable_area", 4)
    xy = []
    for point in points:
        pair = point.split("x") if isinstance(point, str) else []
        if len(pair) != 2:
            raise QuiltInputError("Export a rectangular printable area in millimetres.")
        xy.append(tuple(_number(v, "printable_area", positive=False) for v in pair))
    xs, ys = zip(*xy)
    left, right, bottom, top = min(xs), max(xs), min(ys), max(ys)
    axis_aligned = all(a[0] == b[0] or a[1] == b[1] for a, b in zip(xy, xy[1:] + xy[:1]))
    if len(set(xy)) != 4 or not axis_aligned:
        raise QuiltInputError("Choose a printer preset with a rectangular printable area.")
    if source.get("bed_exclude_area") not in (None, [], ["0x0"], ["0x0"] * 4):
        raise QuiltInputError("Choose a rectangular print region clear of all excluded bed areas.")
    return left, right, bottom, top


def _material_settings(source, colours, separate_tools):
    count = len(colours)
    loaded = _values(source, "filament_colour", count)
    if [str(c).upper() for c in loaded] != [c.upper() for c in colours]:
        raise QuiltInputError("Load the job colours in slot order, then export the project again.")
    if "filament_is_support" in source and any(str(v) != "0" for v in _values(source, "filament_is_support", count)):
        raise QuiltInputError("Assign print material to every job slot and export the settings again.")
    names = [_name(v, "filament") for v in _values(source, "filament_settings_id", count)]
    mapping = [str(i + 1 if separate_tools else 1) for i in range(count)]
    if "filament_map" in source and [str(v) for v in _values(source, "filament_map", count)] != mapping:
        raise QuiltInputError("Assign the job slots to nozzles " + ", ".join(mapping) + " in order.")
    settings = {
        "printer_settings_id": _name(source.get("printer_settings_id"), "printer"),
        "print_settings_id": _name(source.get("print_settings_id"), "process"),
        "filament_settings_id": names, "filament_colour": list(colours), "filament_map": mapping,
    }
    for key in _MATERIAL_NUMBERS:
        if key in source or key in ("filament_diameter", "filament_flow_ratio"):
            values = _values(source, key, count)
            numbers = [_number(v, key) for v in values]
            if key.startswith("nozzle_temperature") and any(not v.is_integer() for v in numbers):
                raise QuiltInputError(f"Set {key} to a whole number of degrees for each spool.")
            settings[key] = [str(int(v)) if v.is_integer() else str(v) for v in numbers]
    if "flush_volumes_matrix" in source and count > 1 and not separate_tools:
        matrix = [_number(v, "flush_volumes_matrix", positive=False)
                  for v in _values(source, "flush_volumes_matrix", count * count)]
        if any(v < 0 or (i % (count + 1) == 0 and v != 0) for i, v in enumerate(matrix)):
            raise QuiltInputError("Recalculate flush volumes for the loaded colours in the slicer.")
        settings["flush_volumes_matrix"] = [str(v) for v in matrix]
        if "flush_multiplier" in source:
            multiplier = _number(_values(source, "flush_multiplier", 1)[0], "flush_multiplier", positive=False)
            if multiplier < 0:
                raise QuiltInputError("Set a non-negative flush multiplier.")
            settings["flush_multiplier"] = [str(multiplier)]
    return settings


def _check_machine(source, settings, *, count, separate_tools, nozzle, first, regular, heights, top_z):
    diameters = source.get("nozzle_diameter")
    supported_counts = range(count, 5) if separate_tools else (1,)
    if not isinstance(diameters, list) or len(diameters) not in supported_counts:
        raise QuiltInputError("Select a printer with the nozzle count required by the job.")
    for key in ("nozzle_diameter", "min_layer_height", "max_layer_height"):
        settings[key] = [str(_number(v, key)) for v in _values(source, key, len(diameters))]
    if nozzle is not None and any(not math.isclose(float(d), nozzle) for d in settings["nozzle_diameter"]):
        raise QuiltInputError(f"Choose the {nozzle:g} mm nozzle used in the flat print plan.")
    if "extruder_offset" in source:
        offsets = _values(source, "extruder_offset", len(diameters))
        for entry in offsets:
            axes = entry.split("x") if isinstance(entry, str) else []
            if len(axes) != 2:
                raise QuiltInputError("Export one X and Y nozzle offset in millimetres for each tool.")
            for value in axes:
                _number(value, "extruder_offset", positive=False)
        settings["extruder_offset"] = offsets.copy()
    for key in ("single_extruder_multi_material", "use_relative_e_distances"):
        if key in source:
            value = str(source[key])
            if value not in ("0", "1"):
                raise QuiltInputError(f"Set {key} on or off in the printer preset and export again.")
            settings[key] = value
    for key, wanted in (("initial_layer_print_height", first), ("layer_height", regular)):
        if not math.isclose(_number(source.get(key), key), wanted, abs_tol=1e-8):
            raise QuiltInputError(f"Set {key} to {wanted:g} mm to match the plan, then export again.")
        settings[key] = str(wanted)
    low = max(map(float, settings["min_layer_height"]))
    high = min(map(float, settings["max_layer_height"]))
    limit = _number(source.get("printable_height"), "printable_height")
    if min(heights) < low - 1e-8 or max(heights) > high + 1e-8 or top_z > limit + 1e-8:
        raise QuiltInputError("Choose a printer that supports every planned layer height and the full print height.")
    settings["printable_height"] = str(limit)


def _placement(source, settings, size, count):
    left, right, bottom, top = _rectangle(source)
    settings["printable_area"] = list(source["printable_area"])
    tower = str(source.get("enable_prime_tower"))
    if tower not in ("0", "1"):
        raise QuiltInputError("Set the prime tower on or off in the slicer and export again.")
    settings["enable_prime_tower"] = tower
    if tower == "1" and count > 1:
        for axis in "xy":
            key = f"wipe_tower_{axis}"
            settings[key] = [str(_number(_values(source, key, 1)[0], key, positive=False))]
        for key in ("prime_tower_width", "prime_tower_brim_width"):
            value = _number(source.get(key), key, positive=key == "prime_tower_width")
            if value < 0:
                raise QuiltInputError("Set a non-negative prime tower brim width.")
            settings[key] = str(value)
        x = float(settings["wipe_tower_x"][0])
        y = float(settings["wipe_tower_y"][0])
        brim = float(settings["prime_tower_brim_width"])
        tower_right = x + float(settings["prime_tower_width"]) + brim
        if x - brim < left or tower_right > right or y - brim < bottom or y + brim > top:
            raise QuiltInputError("Move the prime tower and its brim inside the printable area.")
        left = tower_right + 2
    width, depth = size
    if width > right - left or depth > top - bottom:
        raise QuiltInputError("Reduce the print size so the model fits on the bed beside any prime tower.")
    return ((left + right - width) / 2, (bottom + top - depth) / 2)


def prepare_project(source, colours, *, first, regular, heights, top_z, size,
                    separate_tools=False, nozzle=None):
    """Validate slot order, layer plan and footprint before writing any project parts."""
    settings = _material_settings(source, colours, separate_tools)
    _check_machine(source, settings, count=len(colours), separate_tools=separate_tools, nozzle=nozzle,
                   first=first, regular=regular, heights=heights, top_z=top_z)
    origin = _placement(source, settings, size, len(colours))
    settings.update(sparse_infill_density="100%", sparse_infill_pattern="rectilinear")
    return settings, origin


def project_variant(settings, *, pause=False):
    """Describe the loaded slots and overrides in a resolved project."""
    result = settings.copy()
    if pause:
        for key in ("filament_colour", "filament_settings_id", "filament_map", *_MATERIAL_NUMBERS):
            if key in result:
                result[key] = result[key][:1]
        result.pop("flush_volumes_matrix", None)
        result["enable_prime_tower"] = "0"
    count = len(result["filament_colour"])
    material = ";".join(key for key in ("filament_colour", *_MATERIAL_NUMBERS) if key in result)
    process = ";".join(key for key in (
        "initial_layer_print_height", "layer_height", "sparse_infill_density", "sparse_infill_pattern",
        "enable_prime_tower", "prime_tower_width", "prime_tower_brim_width",
    ) if key in result)
    result.update(filament_is_support=["0"] * count, filament_ids=[""] * count,
                  filament_self_index=[str(i + 1) for i in range(count)], inherits_group=[""] * (count + 2),
                  different_settings_to_system=[process, *([material] * count), ""])
    return result
