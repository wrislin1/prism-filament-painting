"""Generate a printable flat run once, then publish its record and downloads together."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

from .flat_a1 import estimate
from .print_time import calibrated_time, load_calibration
from .slice_report import geometry_report_key, project_key, slice_for_project
from .flat_project import merge_flush_settings, write_flat_project
from .flat_print import PLATE_BUDGETS, preview
from .flat_render import A1Loadout, FlatOptions, generate_flat_run


class FlatResponse(BaseModel):
    mode: Literal["flat"] = "flat"
    run_id: str
    options: FlatOptions
    run: dict
    cost: dict
    preview_url: str
    downloads: dict[str, str | None]
    printer_settings_note: str


def _u1_cost(run, sliced=None):
    g = run["geometry"]
    area = g["width_mm"] * g["height_mm"]
    minutes_limit, grams_limit = PLATE_BUDGETS["u1"]
    calibration = load_calibration("u1")
    choices = {}
    for name, plan in run["plans"].items():
        changes = plan["change_count"]
        timing = calibrated_time(calibration, len(plan["passes"]) * area, changes)
        waste = None
        if calibration and calibration.tower_start_g is not None:
            waste = calibration.tower_start_g + changes * calibration.tower_change_g
        if sliced and name == run["selected_plan"]:
            timing.update(minutes=sliced["minutes"], time_kind="recorded slice", minutes_range=None)
            changes, waste = sliced["changes"], sliced["waste_g"]
        choices[name] = dict(plan=name, change_count=changes, **timing, waste_g=waste,
                             over_time=None if timing["minutes"] is None else timing["minutes"] > minutes_limit,
                             over_waste=None if waste is None else waste > grams_limit,
                             mean_delta_e=plan["mean_delta_e"], p95_delta_e=plan["p95_delta_e"])
    selected = choices[run["selected_plan"]]
    alternative = None
    if calibration and calibration.tower_start_g is not None and (selected["over_time"] or selected["over_waste"]):
        cheaper = (p for p in choices.values() if p["plan"].startswith("all-")
                   and p["change_count"] < selected["change_count"]
                   and p["minutes"] <= selected["minutes"] and p["waste_g"] <= selected["waste_g"])
        alternative = min(cheaper, key=lambda p: (p["over_time"] or p["over_waste"],
                          p["mean_delta_e"], p["p95_delta_e"], p["plan"]), default=None)
    if calibration is None:
        basis = "No time or tower calibration supplied."
    elif calibration.tower_start_g is None:
        basis = "Time is calibrated. No tower mass calibration supplied."
    else:
        basis = "U1 time and tower mass calibrated to desktop slices. Firmware discard is unmeasured."
    return dict(selected=selected, budget_100mm=dict(minutes=minutes_limit, waste_g=grams_limit),
                two_slot_alternative=alternative, basis=basis)


def _studio_cost(run, settings, planning_matrix, sliced):
    if run["profile"] == "u1":
        cost = _u1_cost(run, sliced)
    else:
        cost = estimate(run, settings, sliced=sliced)
        cost["basis"] = ("Planning assumption of 350 mm³ flush per change. Supply recalculated settings."
                         if planning_matrix else "Directional flush matrix from supplied A1 slicer settings.")
        if planning_matrix:
            cost["sources"]["loadout"] = cost["basis"]
        if sliced:
            actual = cost["sliced"]
            forecast = cost["selected"]
            cost["forecast"] = forecast
            cost["selected"] = dict(plan=forecast["plan"], mean_delta_e=forecast["mean_delta_e"],
                                    p95_delta_e=forecast["p95_delta_e"], **actual,
                                    change_count=actual["matrix_for_sliced_order"]["change_count"],
                                    time_kind="recorded slice", minutes_range=None)
    if sliced:
        cost["basis"] = sliced["slicer"] + " slice of the exact configured project."
        if "checked_centres" in sliced:
            cost["basis"] += (f" {sliced['missed_centres']} of {sliced['checked_centres']} layer centres "
                              "lacked the expected bead coverage in G-code.")
    return cost


def _local_export(run, stage, *, slicer_settings, public_geometry, project_settings):
    report = estimate(run, slicer_settings) if slicer_settings is not None else None
    project_settings = merge_flush_settings(project_settings, slicer_settings)
    return write_flat_project(run, stage, project_settings=project_settings, public_geometry=public_geometry,
                              estimate_report=report)


def generate_flat(image, filaments, options, output_dir, *, project_settings=None, slice_lookup=None,
                  export_run=_local_export, url_prefix="/artifacts", installed_note=None):
    loadout = options.slicer_settings
    if options.profile == "a1" and project_settings is not None and "flush_volumes_matrix" in project_settings:
        # The existing estimate uses the single-nozzle plan and its directional flush costs.
        try:
            supplied = A1Loadout.model_validate({**project_settings, "printer_model": "Bambu Lab A1"})
        except ValueError as exc:
            raise ValueError("Recalculate flushing volumes and export the densities and multiplier for all four slots.") from exc
        if loadout is not None and loadout != supplied:
            raise ValueError("Use the same colours, densities and flush volumes for the estimate and the project.")
        loadout = supplied
    render_options = options.model_copy(update={"slicer_settings": loadout})
    run = generate_flat_run(image, filaments, render_options)
    settings = loadout.model_dump() if loadout else None
    if options.profile == "a1" and settings is None:
        settings = dict(printer_model="Bambu Lab A1", filament_colour=[s.hex for s in filaments],
                        filament_density=[1.24] * 4, flush_multiplier=[1],
                        flush_volumes_matrix=[0 if i % 5 == 0 else 350 for i in range(16)])

    run_id = f"quilt_{uuid4().hex[:16]}"
    prefix = f"{url_prefix}/{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as temp:
        stage = Path(temp) / "run"
        export_loadout = settings if project_settings is None or loadout is not None else None
        maps = export_run(run, stage, slicer_settings=export_loadout, public_geometry=True,
                          project_settings=project_settings)
        configured = maps["embedded_printer_settings"]
        if configured:
            key = project_key(stage / "flat.3mf")
        else:
            key = geometry_report_key(stage / "geometry.3mf", run["slots"], settings)
        sliced = slice_for_project(key, options.slice_report, run)
        if sliced is None and configured and project_settings is None and slice_lookup is not None:
            sliced = slice_lookup(key)
        cost = _studio_cost(run, settings, loadout is None, sliced)
        if sliced and not configured:
            cost["basis"] = sliced["slicer"] + " report for the matching geometry. Slicer settings are user-supplied."
        cost["project_key"] = key
        if options.slice_report and options.slice_report.project_key != key:
            cost["basis"] += " The supplied slice report belongs to a different configured project."
        if project_settings is not None:
            note = maps["settings_note"]
        elif configured:
            note = installed_note
            if options.profile == "a1" and options.slicer_settings is None:
                note += " Uses a planning flush matrix. Recalculate flushing volumes in the slicer."
        else:
            note = maps.get("settings_note", "Apply your printer and filament settings in the slicer.")
        if options.profile == "a1":
            (stage / "estimate.json").write_text(json.dumps(cost, allow_nan=False))
        preview(run).save(stage / "preview.png")
        (stage / "run.json").write_text(json.dumps(run, allow_nan=False))
        result = FlatResponse(run_id=run_id, options=options, run=run, cost=cost, preview_url=f"{prefix}/preview.png",
                              downloads=dict(geometry=f"{prefix}/geometry.3mf", slot_map=f"{prefix}/expected.json",
                                             record=f"{prefix}/run.json",
                                             printer_project=f"{prefix}/flat.3mf" if configured else None),
                              printer_settings_note=note)
        stage.rename(output_dir / run_id)
    return result
