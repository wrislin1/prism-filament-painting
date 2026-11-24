"""Color math for palette ranking and generation."""

from __future__ import annotations

import math

import colour
import numpy as np


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) != 6:
        raise ValueError("hex color must be #RRGGBB")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{_clamp_channel(channel):02X}" for channel in rgb)


def hex_to_lab(value: str) -> np.ndarray:
    return rgb_to_lab(hex_to_rgb(value))


def rgb_to_lab(rgb: tuple[int, int, int]) -> np.ndarray:
    srgb = np.array(rgb, dtype=float) / 255.0
    return np.asarray(colour.XYZ_to_Lab(colour.sRGB_to_XYZ(srgb)), dtype=float)


def rgb_array_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Vectorized sRGB->Lab for an array of shape (..., 3) with 0-255 channels."""
    srgb = np.asarray(rgb, dtype=float) / 255.0
    return np.asarray(colour.XYZ_to_Lab(colour.sRGB_to_XYZ(srgb)), dtype=float)


def ciede2000(first_lab, second_lab) -> float:
    return float(colour.delta_E(first_lab, second_lab, method="CIE 2000"))


def ciede2000_array(first_lab: np.ndarray, second_lab: np.ndarray) -> np.ndarray:
    """Batched CIEDE2000 over matching arrays of Lab triples, shape (..., 3) -> (...)."""
    first = np.asarray(first_lab, dtype=float)
    second = np.asarray(second_lab, dtype=float)
    return np.asarray(colour.delta_E(first, second, method="CIE 2000"), dtype=float)


def beer_lambert_blend(
    over_rgb: tuple[int, int, int],
    under_rgb: tuple[int, int, int],
    thickness_mm: float,
    hiding_distance_mm: float,
) -> tuple[int, int, int]:
    """Blend top/over material over lower material using its effective distance to 10% show-through."""
    if not math.isfinite(thickness_mm) or thickness_mm < 0:
        raise ValueError("thickness_mm must be finite and non-negative")
    if not math.isfinite(hiding_distance_mm) or hiding_distance_mm <= 0:
        raise ValueError("hiding_distance_mm must be finite and positive")
    transmission = 10 ** (-thickness_mm / hiding_distance_mm)
    return tuple(
        _clamp_channel(over * (1 - transmission) + under * transmission)
        for over, under in zip(over_rgb, under_rgb, strict=True)
    )


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    channels = [_linearize(channel / 255.0) for channel in rgb]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def relative_luminance_array(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=float) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    return (0.2126 * linear[..., 0]) + (0.7152 * linear[..., 1]) + (0.0722 * linear[..., 2])


def _linearize(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    return math.pow((value + 0.055) / 1.055, 2.4)


def _clamp_channel(value: float | int) -> int:
    return max(0, min(255, int(round(value))))
