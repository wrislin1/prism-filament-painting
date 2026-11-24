"""Approximate opacity from sRGB-derived Lab and the stated finish, in mm."""
import json
import math
import os
import re
from pathlib import Path

import numpy as np

from prism_filament_painting.core.color import hex_to_lab


# The embedding application can supply its own default fit.
DEFAULT_CALIBRATION: Path | None = None


def features(color: str, description: str) -> tuple[float, ...]:
    lightness, a, b = hex_to_lab(color)
    words = set(re.findall(r"[a-z]+", description.lower()))
    return (
        1.0, lightness / 100, a / 100, b / 100,
        math.hypot(a, b) / 100,
        float(bool(words & {"matte", "marble"})),
        float(bool(words & {"clear", "translucent", "transparent"})),
    )


def fit_opacity(measurements) -> list[float]:
    """Fit log TD from (hex colour, finish description, measured TD in mm) rows.

    Save the returned weights as JSON and select that file with
    PRISM_OPACITY_CALIBRATION. Ridge regularisation stabilises small samples.
    """
    inputs, targets = [], []
    for color, description, td in measurements:
        if not math.isfinite(td) or td <= 0:
            raise ValueError("measured TD must be finite and positive")
        inputs.append(features(color, description))
        targets.append(math.log(td))
    if not inputs:
        raise ValueError("supply at least one measured spool")
    x = np.asarray(inputs)
    penalty = np.eye(x.shape[1]) * 0.5
    penalty[0, 0] = 0  # leave the intercept unpenalised
    return np.linalg.solve(x.T @ x + penalty, x.T @ targets).tolist()


def estimate_opacity(color: str, description: str) -> float | None:
    """Use the configured log-distance fit, or report that no estimate exists.

    PRISM_OPACITY_CALIBRATION names a JSON array of seven weights in features()
    order. A user can fit these to their measured distances in millimetres.
    """
    path = os.environ.get("PRISM_OPACITY_CALIBRATION", DEFAULT_CALIBRATION)
    if not path:
        return None
    try:
        weights = json.loads(Path(path).read_text(encoding="utf-8"))
        if (not isinstance(weights, list) or len(weights) != 7 or
                any(type(w) not in (int, float) or not math.isfinite(w) for w in weights)):
            raise ValueError("expected seven finite numeric weights")
    except (OSError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid opacity calibration: {exc}") from exc
    log_mm = sum(x * weight for x, weight in zip(features(color, description), weights))
    if not math.isfinite(log_mm):
        raise ValueError("opacity calibration exceeds the numeric range")
    return round(math.exp(max(math.log(0.1), min(math.log(100.0), log_mm))), 2)
