"""Auto-enhance recommender + non-destructive tonal transform for printability."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from prism_filament_painting.core.constraints import (
    LUMINANCE_BLACK_MIN,
    LUMINANCE_GAMMA_MAX,
    LUMINANCE_GAMMA_MIN,
    LUMINANCE_WHITE_MAX,
)

from .analyze import PrintabilityReport
from .color import relative_luminance_array

LOCAL_CONTRAST_MIN = 0.0
LOCAL_CONTRAST_MAX = 1.0
LOCAL_CONTRAST_GUIDE_MAX_DIM = 1024

_RECOMMEND_MIN_SPAN = 0.05


@dataclass(frozen=True)
class EnhanceParams:
    black_point: float = 0.0
    white_point: float = 1.0
    gamma: float = 1.0
    local_contrast: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "EnhanceParams":
        if not isinstance(data, dict):
            raise ValueError("enhance_params must be an object")

        def _num(key: str, default: float) -> float:
            value = data.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
            return value

        def _clamp(value: float, lo: float, hi: float) -> float:
            return min(max(value, lo), hi)

        black = _clamp(_num("black_point", 0.0), LUMINANCE_BLACK_MIN, LUMINANCE_WHITE_MAX)
        white = _clamp(_num("white_point", 1.0), LUMINANCE_BLACK_MIN, LUMINANCE_WHITE_MAX)
        if not black < white:
            raise ValueError("black_point must be less than white_point inside [0, 1]")
        gamma = _clamp(_num("gamma", 1.0), LUMINANCE_GAMMA_MIN, LUMINANCE_GAMMA_MAX)
        local = _clamp(_num("local_contrast", 0.0), LOCAL_CONTRAST_MIN, LOCAL_CONTRAST_MAX)
        return cls(black_point=black, white_point=white, gamma=gamma, local_contrast=local)

    def is_identity(self) -> bool:
        return (
            self.black_point == 0.0
            and self.white_point == 1.0
            and self.gamma == 1.0
            and self.local_contrast == 0.0
        )

    def to_dict(self) -> dict:
        return {
            "black_point": self.black_point,
            "white_point": self.white_point,
            "gamma": self.gamma,
            "local_contrast": self.local_contrast,
        }


def recommend_enhancement(report: PrintabilityReport) -> EnhanceParams:
    """Deterministic auto starting point derived from the Slice 1 report.

    Auto-levels from the report's 32-bin luminance histogram (1st/99th-percentile
    bin edges), a gentle gamma nudge from the histogram centroid, and a default
    local-contrast strength when the image maps to too few printable bands.
    """
    hist = np.asarray(report.histogram, dtype=np.float64)
    bins = hist.size if hist.size else 1
    total = float(hist.sum())

    black, white, gamma = 0.0, 1.0, 1.0
    if total > 0:
        cdf = np.cumsum(hist) / total
        lo = int(min(max(np.searchsorted(cdf, 0.01), 0), bins - 1))
        hi = int(min(max(np.searchsorted(cdf, 0.99), 0), bins - 1))
        cand_black = lo / bins
        cand_white = (hi + 1) / bins
        if cand_black < cand_white:
            # Expand a too-narrow window to a minimum span (centered, clamped to [0, 1])
            # so a tightly-clustered image still gets a real stretch, not identity.
            if (cand_white - cand_black) < _RECOMMEND_MIN_SPAN:
                center = (cand_black + cand_white) / 2.0
                cand_black = center - _RECOMMEND_MIN_SPAN / 2.0
                cand_white = center + _RECOMMEND_MIN_SPAN / 2.0
                if cand_black < 0.0:
                    cand_white = min(1.0, max(cand_white - cand_black, _RECOMMEND_MIN_SPAN))
                    cand_black = 0.0
                if cand_white > 1.0:
                    cand_black = max(0.0, 1.0 - _RECOMMEND_MIN_SPAN)
                    cand_white = 1.0
            black, white = cand_black, cand_white

        centers = (np.arange(bins) + 0.5) / bins
        centroid = float((hist * centers).sum() / total)
        if centroid < 0.4:
            gamma = 0.8  # lift muddy dark midtones
        elif centroid > 0.6:
            gamma = 1.2  # deepen washed-out bright midtones

    if report.band_utilization < 0.5:
        local = 0.6
    elif report.band_utilization < 0.7:
        local = 0.3
    else:
        local = 0.0

    return EnhanceParams.from_dict(
        {"black_point": black, "white_point": white, "gamma": gamma, "local_contrast": local}
    )


def apply_enhancement(image: Image.Image, params: EnhanceParams) -> Image.Image:
    """Non-destructive tonal transform: per-channel levels LUT + optional CLAHE.

    Levels mirror the print model's remap (``(v - black) / (white - black)`` then
    ``v ** gamma``) but applied to image RGB so ranking/analysis see the improved
    image. Local contrast (Task 3) is a CLAHE pass on luminance, re-applied to RGB.
    """
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    arr = np.asarray(rgb, dtype=np.float32) / 255.0  # (H, W, 3) in 0..1

    span = params.white_point - params.black_point  # > 0, validated
    arr = np.clip((arr - params.black_point) / span, 0.0, 1.0)
    if params.gamma != 1.0:
        arr = np.power(arr, params.gamma, dtype=np.float32)

    if params.local_contrast > 0.0:
        arr = _apply_local_contrast(arr, params.local_contrast)

    out = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, "RGB")


def _apply_local_contrast(arr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return arr
    height, width = arr.shape[:2]
    guide = arr
    if max(height, width) > LOCAL_CONTRAST_GUIDE_MAX_DIM:
        scale = LOCAL_CONTRAST_GUIDE_MAX_DIM / max(height, width)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        guide_img = Image.fromarray(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8), "RGB")
        guide = np.asarray(guide_img.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0

    guide_rgb = np.clip(guide * 255.0, 0.0, 255.0).astype(np.uint8)
    lum = np.clip(relative_luminance_array(guide_rgb), 0.0, 1.0).astype(np.float32)
    equalized = _clahe(lum)
    new_lum = np.clip(lum + strength * (equalized - lum), 0.0, 1.0)
    eps = 1e-6
    ratio = np.where(lum > eps, new_lum / np.maximum(lum, eps), 1.0).astype(np.float32)
    if ratio.shape != (height, width):
        ratio_img = Image.fromarray(ratio)
        ratio = np.asarray(ratio_img.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    return np.clip(arr * ratio[..., None], 0.0, 1.0).astype(np.float32)


def _clahe(
    lum: np.ndarray, *, tiles: int = 8, clip_limit: float = 0.01, bins: int = 256
) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization, pure numpy.

    Builds a clipped-CDF mapping per tile, then bilinearly interpolates the four
    surrounding tile mappings per pixel. Memory is O(H*W) -- the per-pixel gather
    indexes a flattened (tiles*tiles*bins) LUT, never an (H, W, bins) tensor.
    """
    height, width = lum.shape
    if lum.size == 0:
        return lum.astype(np.float32)
    if float(np.max(lum) - np.min(lum)) <= 1e-6:
        return np.clip(lum, 0.0, 1.0).astype(np.float32)

    min_tile = 16
    if height < min_tile or width < min_tile:
        return _hist_equalize(lum, clip_limit=clip_limit, bins=bins)

    ty = max(1, min(tiles, height // min_tile))
    tx = max(1, min(tiles, width // min_tile))

    ys = np.linspace(0, height, ty + 1).astype(int)
    xs = np.linspace(0, width, tx + 1).astype(int)
    quant = np.clip(np.round(lum * (bins - 1)).astype(np.int64), 0, bins - 1)

    mappings = np.empty((ty, tx, bins), dtype=np.float32)
    for i in range(ty):
        for j in range(tx):
            tile = quant[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            n = tile.size
            if n == 0:
                mappings[i, j] = np.linspace(0.0, 1.0, bins, dtype=np.float32)
                continue
            hist = np.bincount(tile.ravel(), minlength=bins).astype(np.float32)
            limit = max(1.0, clip_limit * n)
            excess = np.maximum(hist - limit, 0.0).sum()
            hist = np.minimum(hist, limit) + excess / bins
            cdf = np.cumsum(hist)
            cdf /= cdf[-1]
            mappings[i, j] = cdf.astype(np.float32)

    cy = (ys[:-1] + ys[1:]) / 2.0
    cx = (xs[:-1] + xs[1:]) / 2.0
    rows = np.arange(height)
    cols = np.arange(width)

    iy = np.clip(np.searchsorted(cy, rows) - 1, 0, ty - 1)
    iy1 = np.clip(iy + 1, 0, ty - 1)
    dy = cy[iy1] - cy[iy]
    wy = np.where(dy > 0, (rows - cy[iy]) / np.maximum(dy, 1e-6), 0.0)
    wy = np.clip(wy, 0.0, 1.0).astype(np.float32)

    ix = np.clip(np.searchsorted(cx, cols) - 1, 0, tx - 1)
    ix1 = np.clip(ix + 1, 0, tx - 1)
    dx = cx[ix1] - cx[ix]
    wx = np.where(dx > 0, (cols - cx[ix]) / np.maximum(dx, 1e-6), 0.0)
    wx = np.clip(wx, 0.0, 1.0).astype(np.float32)

    flat = mappings.reshape(-1)

    def _lookup(ti: np.ndarray, tj: np.ndarray) -> np.ndarray:
        idx = (ti[:, None] * tx + tj[None, :]) * bins + quant
        return flat[idx]

    v00 = _lookup(iy, ix)
    v01 = _lookup(iy, ix1)
    v10 = _lookup(iy1, ix)
    v11 = _lookup(iy1, ix1)

    wy2 = wy[:, None]
    wx2 = wx[None, :]
    top = v00 * (1.0 - wx2) + v01 * wx2
    bot = v10 * (1.0 - wx2) + v11 * wx2
    out = top * (1.0 - wy2) + bot * wy2
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _hist_equalize(
    lum: np.ndarray, *, clip_limit: float = 0.01, bins: int = 256
) -> np.ndarray:
    quant = np.clip(np.round(lum * (bins - 1)).astype(np.int64), 0, bins - 1)
    n = quant.size
    if n == 0:
        return lum.astype(np.float32)
    if float(np.max(lum) - np.min(lum)) <= 1e-6:
        return np.clip(lum, 0.0, 1.0).astype(np.float32)

    hist = np.bincount(quant.ravel(), minlength=bins).astype(np.float32)
    limit = max(1.0, clip_limit * n)
    excess = np.maximum(hist - limit, 0.0).sum()
    hist = np.minimum(hist, limit) + excess / bins
    cdf = np.cumsum(hist)
    cdf /= cdf[-1]
    return np.clip(cdf[quant], 0.0, 1.0).astype(np.float32)
