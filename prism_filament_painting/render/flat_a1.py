"""Estimate a flat A1 print from its loaded slots and a slicer-saved flush matrix."""

import argparse
import json
from math import isfinite
from pathlib import Path

from .engine.slicer_settings import read_project_settings as read_settings
from .print_time import calibrated_time, load_calibration
from .flat_layout import layout
from .flat_passes import plan_passes, validate_costs
from .flat_print import PLATE_BUDGETS


def _loadout(run, settings):
    colours = [s["hex"].upper() for s in run["slots"]]
    if settings["printer_model"] != "Bambu Lab A1":
        raise ValueError("supply Bambu Lab A1 slicer settings")
    if [c.upper() for c in settings["filament_colour"]] != colours:
        raise ValueError("slicer colours must match the loaded slots in order")
    n = len(colours)
    if not 1 <= n <= 4:
        raise ValueError("load one to four AMS Lite slots")
    # Single-filament projects can retain a larger default table. No switch is possible.
    flat = [0.] if n == 1 else list(map(float, settings["flush_volumes_matrix"]))
    if len(flat) != n * n:
        raise ValueError("supply the slicer's square flush matrix for the loaded slots")
    matrix = [flat[i:i + n] for i in range(0, len(flat), n)]
    validate_costs(matrix, n)
    multiplier, = map(float, settings["flush_multiplier"])
    density = list(map(float, settings["filament_density"]))
    if not isfinite(multiplier) or multiplier < 0:
        raise ValueError("flush multiplier must be finite and non-negative")
    if len(density) != n or any(not isfinite(d) or d <= 0 for d in density):
        raise ValueError("supply a positive filament density for each loaded slot")
    return dict(colours=colours, matrix_mm3=matrix, multiplier=multiplier, density_g_cm3=density)


def transition_totals(passes, start, loadout):
    """Count each directed switch once, carrying the active slot across layer boundaries."""
    changes = []
    volume = mass = 0.
    previous = start
    for layer, order in enumerate(passes, 1):
        for index, slot in enumerate(order):
            if previous != slot:
                v = loadout["matrix_mm3"][previous - 1][slot - 1] * loadout["multiplier"]
                volume += v
                mass += v * loadout["density_g_cm3"][slot - 1] / 1000
                changes.append(dict(layer=layer, from_slot=previous, to_slot=slot,
                                    layer_boundary=index == 0, flush_mm3=v))
            previous = slot
    return dict(passes=passes, changes=changes, change_count=len(changes),
                flush_mm3=volume, flushed_g=mass)


def _plan_report(name, plan, run, loadout, costs, calibration):
    start = run["backing_slot"]
    usage = plan["layer_usage"]
    g = run["geometry"]
    if len(usage) != len(g["backing_layer_heights_mm"]) + g["picture_layers"]:
        raise ValueError("plan layer count differs from the flat run")
    for row in usage:
        if (not 1 <= len(row) <= len(costs)
                or any(type(slot) is not int or not 1 <= slot <= len(costs) for slot in row)
                or len(set(row)) != len(row)):
            raise ValueError("plan must list each occupied loaded slot once per layer")
    passes, _ = plan_passes(usage, start=start, costs=costs)
    result = transition_totals(passes, start, loadout)
    area = g["width_mm"] * g["height_mm"]
    timing = calibrated_time(calibration, len(passes) * area, result["change_count"], result["flush_mm3"])
    startup_g = 250 * loadout["density_g_cm3"][start - 1] / 1000
    prime_g = sum(25 * loadout["density_g_cm3"][edge["to_slot"] - 1] / 1000
                  for edge in result["changes"])
    waste = result["flushed_g"] + startup_g + prime_g
    limit_min, limit_g = PLATE_BUDGETS["a1"]
    return dict(**result, plan=name, mean_delta_e=plan["mean_delta_e"],
                p95_delta_e=plan["p95_delta_e"], **timing,
                startup_tower_g=startup_g, prime_g=prime_g, waste_g=waste,
                over_time=None if timing["minutes"] is None else timing["minutes"] > limit_min, over_waste=waste > limit_g)


def _two_slot_alternative(run, loadout, costs, primary, calibration):
    choices = []
    for name, plan in run["plans"].items():
        if name == primary["plan"] or not name.startswith("all-"):
            continue
        candidate = _plan_report(name, plan, run, loadout, costs, calibration)
        minutes_saved = primary["minutes"] - candidate["minutes"]
        grams_saved = primary["waste_g"] - candidate["waste_g"]
        if minutes_saved >= 0 and grams_saved >= 0 and (minutes_saved > 0 or grams_saved > 0):
            choices.append(candidate)
    return min(choices, key=lambda c: (c["over_time"] or c["over_waste"], c["mean_delta_e"],
                                       c["p95_delta_e"], c["waste_g"], c["plan"]), default=None)


def estimate(run, settings, *, sliced=None):
    """Compare estimated plans and optionally recheck the selected plan from its slice."""
    if run["profile"] != "a1":
        raise ValueError("select an A1 flat run")
    loadout = _loadout(run, settings)
    costs = [[v * loadout["multiplier"] for v in row] for row in loadout["matrix_mm3"]]
    selected = run["selected_plan"]
    plans = run["plans"]
    _, planes = layout(run)
    if [sorted(set(p.ravel())) for p in planes] != [sorted(row) for row in plans[selected]["layer_usage"]]:
        raise ValueError("selected plan layer usage differs from the stored cell recipes")
    calibration = load_calibration("a1")
    primary = _plan_report(selected, plans[selected], run, loadout, costs, calibration)
    alternative = None
    basis = primary
    actual = None
    if sliced is not None:
        # The slicer may reorder the passes. Use its time and material breakdown together.
        order = sliced["passes"]
        if (len(order) != len(planes) or any(set(layer) != set(usage)
                for layer, usage in zip(order, plans[selected]["layer_usage"]))):
            raise ValueError("sliced passes do not match the flat run's occupied slots")
        duration, flushed, tower = (float(sliced[k]) for k in ("minutes", "flushed_g", "tower_g"))
        if any(not isfinite(v) or v < 0 for v in (duration, flushed, tower)):
            raise ValueError("sliced time and masses must be finite and non-negative")
        minutes, grams = PLATE_BUDGETS["a1"]
        actual = dict(matrix_for_sliced_order=transition_totals(order, run["backing_slot"], loadout),
                      minutes=duration, flushed_g=flushed, tower_g=tower, waste_g=flushed + tower,
                      over_time=duration > minutes, over_waste=flushed + tower > grams)
        basis = actual
    if calibration is not None and (basis["over_time"] or basis["over_waste"]):
        alternative = _two_slot_alternative(run, loadout, costs, primary | basis, calibration)
    minutes, grams = PLATE_BUDGETS["a1"]
    return dict(schema_version=1, loadout=loadout, selected=primary, sliced=actual,
                two_slot_alternative=alternative,
                budget_100mm=dict(minutes=minutes, waste_g=grams),
                sources=dict(loadout="Slicer project: row = outgoing slot, column = incoming slot. "
                                     "Recalculate flushing volumes after changing colours.",
                             changes="Minimum directed flush volume over occupied slots, starting with backing loaded. "
                                     "The slicer chooses its own order.",
                             minutes=("Calibrated against A1 desktop slices, including setup, deposition, changes and flush."
                                      if calibration else "No time calibration supplied."),
                             waste="Matrix volume times multiplier and incoming density, plus the design's "
                                   "250 mm³ fixed startup/tower allowance at backing density and 25 mm³ per "
                                   "change for priming. Actual tower paths and firmware purges require verification.",
                             colour="Stage-one rematch: mean and 95th-percentile CIEDE2000 over all image cells."))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("slicer_settings", type=Path, help="saved A1 project (.3mf) or its settings JSON")
    parser.add_argument("--sliced", type=Path, help="JSON with observed passes, minutes, flushed_g and tower_g")
    args = parser.parse_args(argv)
    try:
        sliced = json.loads(args.sliced.read_text()) if args.sliced else None
        result = estimate(json.loads(args.run.read_text()), read_settings(args.slicer_settings), sliced=sliced)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (KeyError, ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
