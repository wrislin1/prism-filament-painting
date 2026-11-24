"""Fit a nonlinear front-lit hiding distance, using a predefined recipe split."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .calibration_board import validate_database
from .color import beer_lambert_blend, ciede2000_array, hex_to_rgb, rgb_array_to_lab


@dataclass(frozen=True)
class DistanceCurve:
    """H in mm = distance_at_td1_mm * (nominal TD / 1 mm) ** exponent."""

    distance_at_td1_mm: float = 0.1
    exponent: float = 1.0

    def __post_init__(self):
        for value in (self.distance_at_td1_mm, self.exponent):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("curve parameters must be finite and positive")


class BoardPredictor:
    def __init__(self, board):
        self.layer_mm = board["layer_mm"]
        self.spools = {
            m["id"]: (np.array(hex_to_rgb(m["hex"])), m["td_mm"]) for m in board["materials"]
        }
        self.base = self.spools[board["materials"][0]["id"]][0]

    def predict(self, bands, curve, *, round_bands=True):
        """Opaque backing, then bottom-to-top bands. Round per band for scoring."""
        under = self.base
        for spool, layers in bands:
            over, td = self.spools[spool]
            distance = curve.distance_at_td1_mm * td**curve.exponent
            thickness = layers * self.layer_mm
            if round_bands:
                under = np.asarray(beer_lambert_blend(over, under, thickness, distance))
            else:
                transmission = 10 ** (-thickness / distance)
                under = over * (1 - transmission) + under * transmission
        return under


class UnidentifiableCurve(ValueError):
    """The capture can supply measured colours but cannot constrain the fit."""


def _wedge_seed(model, wedges):
    # Project the observed RGB onto the line between the known opaque
    # endpoints. Interior transmissions supply log H without an optimizer.
    log_td, log_distance = [], []
    for bands, measured in wedges:
        ((spool, layers),) = bands
        top, td = model.spools[spool]
        contrast = model.base - top
        if not contrast.any():
            continue
        transmission = np.dot(measured - top, contrast) / np.dot(contrast, contrast)
        if 0.02 < transmission < 0.98:
            log_td.append(math.log(td))
            log_distance.append(math.log(-layers * model.layer_mm / math.log10(transmission)))
    design = np.column_stack((np.ones(len(log_td)), log_td))
    if len(log_td) < 2 or np.linalg.matrix_rank(design) != 2:
        raise UnidentifiableCurve("wedges need measurable contrast at two distinct overlay TDs")
    return np.linalg.lstsq(design, log_distance, rcond=None)[0]


def _joint_fit(model, rows, seed):
    """Damped Gauss–Newton on D65 Lab residuals of all training recipes."""
    target = rgb_array_to_lab([rgb for _, rgb in rows])
    lower = np.array([math.log(0.005), 0.05])
    upper = np.array([math.log(2), 3.0])
    params = np.clip(seed, lower, upper)

    def residual(parameters):
        curve = DistanceCurve(math.exp(parameters[0]), parameters[1])
        predicted = [model.predict(bands, curve, round_bands=False) for bands, _ in rows]
        return (rgb_array_to_lab(predicted) - target).ravel()

    for _ in range(40):
        errors = residual(params)
        jac = np.column_stack(
            [(residual(params + step) - errors) / 1e-4 for step in 1e-4 * np.eye(2)]
        )
        delta = np.linalg.lstsq(jac, -errors, rcond=None)[0]
        for backtrack in range(12):
            trial = np.clip(params + delta / 2**backtrack, lower, upper)
            new_errors = residual(trial)
            if new_errors @ new_errors < errors @ errors:
                break
        else:
            break
        change = np.linalg.norm(trial - params)
        params = trial
        if change < 1e-7:
            break
    return DistanceCurve(math.exp(params[0]), float(params[1]))


def _score(model, rows, curve):
    predicted = [model.predict(bands, curve) for bands, _ in rows]
    observed = [rgb for _, rgb in rows]
    errors = ciede2000_array(rgb_array_to_lab(predicted), rgb_array_to_lab(observed))
    return {
        "median_de00": float(np.median(errors)),
        "mean_de00": float(np.mean(errors)),
        "fraction_de00_le_2": float(np.mean(errors <= 2)),
    }


def fit_database(database: dict) -> dict:
    """Expose usable parameters only when both mean and median hold-out ΔE00 fall."""
    validate_database(database)
    board = database["board"]
    model = BoardPredictor(board)
    by_role = {"control": [], "wedge": [], "mixed": [], "holdout": []}
    for patch in board["patches"]:
        rgb = np.array(database["measurements"][patch["id"]]["srgb8"])
        by_role[patch["role"]].append((patch["bands"], rgb))
    training = by_role["wedge"] + by_role["mixed"] + by_role["control"]
    seed = _wedge_seed(model, by_role["wedge"])
    candidate = _joint_fit(model, training, seed)
    holdout = by_role["holdout"]
    baseline = _score(model, holdout, DistanceCurve())
    fitted = _score(model, holdout, candidate)
    accepted = all(fitted[k] < baseline[k] for k in ("median_de00", "mean_de00"))
    return {
        "schema": "front-lit-fit/v1",
        "capture": database["capture"],
        "materials": board["materials"],
        "layer_mm": board["layer_mm"],
        "training_count": len(training),
        "holdout_ids": [p["id"] for p in board["patches"] if p["role"] == "holdout"],
        "wedge_seed": {"log_distance_at_td1": float(seed[0]), "exponent": float(seed[1])},
        "candidate": asdict(candidate),
        "accepted_curve": asdict(candidate) if accepted else None,
        "baseline": baseline,
        "fitted": fitted,
    }
