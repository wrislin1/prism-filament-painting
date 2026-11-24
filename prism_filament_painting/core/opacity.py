"""Invert a front-lit patch comparison using the experiment card's sRGB blend.

Published attenuation and inversion: https://kromacut.com/docs/calibration-theory
Distances here are physical millimetres. No published-TD conversion is assumed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite, log10
from statistics import median

import numpy as np

from .color import ciede2000, hex_to_rgb, rgb_array_to_lab

LAYER_MM = 0.08
PATCH_COUNT = 30


def threshold_divisor(top_hex: str, backing_hex: str) -> float:
    top = np.array(hex_to_rgb(top_hex), dtype=float)
    backing = np.array(hex_to_rgb(backing_hex), dtype=float)
    opaque = rgb_array_to_lab(top)

    def difference(fraction):
        return ciede2000(opaque, rgb_array_to_lab(top + fraction * (backing - top)))

    if difference(1) <= 2:
        raise ValueError("Choose a backing more than 2 ΔE00 from the opaque test colour.")
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if difference(mid) <= 2:
            lo = mid
        else:
            hi = mid
    return -log10((lo + hi) / 2)


def measure_crossing(
    divisor: float,
    passes: Sequence[int | None],
    meter_tds: Sequence[float] = (),
    measured_heights: Mapping[int, float] | None = None,
) -> dict:
    """Three first-match readings; None means even patch 30 still differs."""
    if not isfinite(divisor) or divisor <= 0:
        raise ValueError("A must be finite and positive.")
    if len(passes) != 3 or any(
        k is not None and (type(k) is not int or not 1 <= k <= PATCH_COUNT) for k in passes
    ):
        raise ValueError("Enter three readings, each 1–30 or no match.")
    heights = [k * LAYER_MM for k in range(PATCH_COUNT + 1)]
    for k, height in (measured_heights or {}).items():
        if type(k) is not int or not 1 <= k <= PATCH_COUNT or not isfinite(height):
            raise ValueError("Measured heights require a patch number and a finite thickness.")
        heights[k] = height
    if any(a >= b for a, b in zip(heights, heights[1:])):
        raise ValueError("Top thicknesses must increase from the bare backing.")

    brackets = [(heights[-1], None) if k is None else (heights[k - 1], heights[k]) for k in passes]
    lower = min(a for a, _ in brackets)
    upper = None if None in passes else max(b for _, b in brackets)
    h_lo = lower / divisor
    h_hi = None if upper is None else upper / divisor
    result = dict(
        pass_thickness_mm=brackets,
        crossing_mm=(lower, upper),
        hiding_distance_interval_mm=(h_lo, h_hi),
        median_td_mm=None,
        scale=None,
        scale_with_accuracy=None,
        meter_range_mm=None,
    )
    if meter_tds:
        if len(meter_tds) != 5 or any(not isfinite(td) or td <= 0 for td in meter_tds):
            raise ValueError("Supply five positive finite meter readings or omit the scans.")
        m = median(meter_tds)
        result.update(
            meter_range_mm=(min(meter_tds), max(meter_tds)),
            median_td_mm=m,
            scale=(h_lo / m, None if h_hi is None else h_hi / m),
            scale_with_accuracy=(
                h_lo / (max(meter_tds) / 0.925),
                None if h_hi is None else h_hi / (min(meter_tds) / 1.075),
            ),
        )
    return result
