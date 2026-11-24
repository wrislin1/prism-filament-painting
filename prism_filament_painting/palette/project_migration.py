"""Optical-field adapter for classic and flat projects. Distances are in millimetres."""

from copy import deepcopy
import json
import math
from pathlib import Path
import re
from uuid import UUID

from .migration_copy import staged_copy


def _positive(value):
    if isinstance(value, bool):
        raise ValueError("expected a positive finite distance or scale")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("expected a positive finite distance or scale")
    return number


def _source(label):
    if label is not None and not isinstance(label, str):
        raise ValueError("invalid optical provenance")
    if label in ("matched", "measured", "vendor", "estimated", "meter", "wedge", "published"):
        return label
    # Preserve unfamiliar historical labels on the spool, without claiming a source.
    return "unknown"


def _locations(state):
    """Every saved copy of a spool and every scale that governs those spools."""
    spools = list(state["palette"]["filaments"])
    scales = [state]
    if state.get("project_type", "classic") == "flat":
        run = state["flat"]["run"]
        if "slots" in run:
            if len(run["slots"]) != len(spools):
                raise ValueError("saved slot count differs from palette")
            spools.extend(run["slots"])
        if "model" in run:
            scales.append(run["model"])
    elif state.get("project_type", "classic") != "classic":
        raise ValueError("unknown project type")
    return spools, scales


def adapt_project(state, *, validate_v1, default_schedule=None, default_label=None, default_scale=0.1, saved_file=True):
    """Return a validated v2 value. The caller supplies its v1 reader and legacy preset.

    validate_v1 checks the complete saved geometry, options and masks, raising on
    error. Its normalised result is deliberately discarded to preserve saved fields.
    default_schedule(layer, spool_count) is needed only for an implicit v1 schedule.
    """
    result = deepcopy(state)
    version = result["version"]
    if type(version) is not int or version not in (1, 2):
        raise ValueError("unsupported project version")
    # Reject NaN/Infinity even in pass-through values, including on a second load.
    json.dumps(result, allow_nan=False)
    project_id = result["image"].get("project_id")
    if project_id is not None or saved_file:
        if not isinstance(project_id, str) or str(UUID(project_id)) != project_id or UUID(project_id).version != 4:
            raise ValueError("expected a saved project UUID")
    spools, scales = _locations(result)
    key = "td_scale" if version == 1 else "front_lit_scale"
    saved = [_positive(owner[key]) for owner in scales if key in owner]
    if len(set(saved)) > 1 or (version == 2 and len(saved) != len(scales)):
        raise ValueError("inconsistent saved front-lit scale")
    scale = saved[0] if saved else _positive(default_scale)

    if version == 2:
        # Use the same full format validator for both versions via a transient v1 view.
        view = deepcopy(result)
        old_spools, old_scales = _locations(view)
        for spool in old_spools:
            value = _positive(spool["input_distance_mm"])
            if ("td" in spool or not isinstance(spool["input_distance_source"], str)
                    or not spool["input_distance_source"]
                    or not isinstance(spool["color_source"], str) or not spool["color_source"]
                    or not re.fullmatch(r"#[0-9a-fA-F]{6}", spool["hex"])
                    or _positive(spool["hiding_distance_mm"]) != value * scale):
                raise ValueError("invalid migrated spool")
            spool["td"] = value
        for owner in old_scales:
            if "td_scale" in owner:
                raise ValueError("ambiguous saved scale")
            owner["td_scale"] = owner.pop("front_lit_scale")
        view["version"] = 1
        validate_v1(view)
        return result

    if result.get("project_type") != "flat":
        # The classic v1 browser, CPU report and export all used the service
        # scale. Preserve a user-supplied declaration as historical metadata.
        scale = _positive(default_scale)
        if "swap_schedule" not in result:
            if default_schedule is None:
                raise ValueError("implicit swap schedule requires the original schedule resolver")
            result["swap_schedule"] = list(default_schedule(result["layer"], len(spools)))
        if default_label and result.get("schedule_label") == default_label:
            result["schedule_label"] = "Saved four-spool schedule"
    validate_v1(deepcopy(result))
    if saved and saved[0] != scale:
        if "v1_declared_scale" in result:
            raise ValueError("ambiguous historical scale")
        result["v1_declared_scale"] = saved[0]
    for owner in scales:
        if "front_lit_scale" in owner:
            raise ValueError("ambiguous saved scale")
        owner.pop("td_scale", None)
        owner["front_lit_scale"] = scale
    for spool in spools:
        if {"input_distance_mm", "input_distance_source", "hiding_distance_mm", "color_source"} & spool.keys():
            raise ValueError("ambiguous saved spool")
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", spool["hex"]):
            raise ValueError("invalid spool colour")
        value = _positive(spool.pop("td"))
        source = _source(spool.get("td_source", spool.get("opacity_level")))
        color_source = _source(spool.get("color_level"))
        spool.update(input_distance_mm=value, input_distance_source=source, color_source=color_source,
                     hiding_distance_mm=_positive(value * scale))
    result["version"] = 2
    return result


def migrate_project_copy(source, destination, **reader):
    """Keep the v1 file available for rollback. Publish only complete validated JSON."""
    def same(a, b):
        return a.read_bytes() == b.read_bytes()

    with staged_copy(source, destination, equivalent=same) as copy:
        state = adapt_project(json.loads(Path(source).read_bytes()), **reader)
        copy.write_text(json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False),
                        encoding="utf-8")
