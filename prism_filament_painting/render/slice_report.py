"""Identify the project described by a slicer report."""

import hashlib
import json
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from .flat_passes import count_changes


class SliceReport(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    project_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    slicer: str = Field(min_length=1, max_length=100)
    minutes: float = Field(gt=0)
    flushed_g: float = Field(ge=0)
    tower_g: float = Field(ge=0)
    passes: list[list[StrictInt]] = Field(min_length=1, max_length=6)


def project_key(path):
    """Match the actual geometry, placement, slot assignments and process settings.

    Ignore ZIP timestamps and compression. Hash every member of the project.
    """
    with ZipFile(path) as project:
        members = []
        for name in sorted(project.namelist()):
            with project.open(name) as member:
                if name == "Metadata/project_settings.config":
                    settings = json.load(member)
                    # Studio normalizes spool hex to uppercase. The slicer treats
                    # both spellings as the identical RGB and flush-table input.
                    if "filament_colour" in settings:
                        settings["filament_colour"] = [c.upper() for c in settings["filament_colour"]]
                    payload = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
                    digest = hashlib.sha256(payload)
                else:
                    digest = hashlib.file_digest(member, "sha256")
                members.append((name, digest.hexdigest()))
    return hashlib.sha256(json.dumps(members).encode()).hexdigest()


def geometry_report_key(path, slots, settings):
    """Include loadout inputs that are absent from a geometry-only archive."""
    if settings:
        settings = dict(settings, filament_colour=[color.upper() for color in settings["filament_colour"]])
    payload = {
        "geometry": project_key(path),
        "slot_colours": [slot["hex"].upper() for slot in slots],
        "settings": settings,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def slice_for_project(key, supplied, run):
    if supplied is None or supplied.project_key != key:
        return None
    order = supplied.passes
    usage = run["plans"][run["selected_plan"]]["layer_usage"]
    if len(order) != len(usage) or any(set(p) != set(u) for p, u in zip(order, usage)):
        raise ValueError("sliced passes must match the occupied slots on each layer")
    return dict(**supplied.model_dump(), changes=count_changes(order, start=run["backing_slot"]),
                waste_g=supplied.flushed_g + supplied.tower_g)
