"""Shared preview metrics for Prism palette ranking and Quilt previews."""

from __future__ import annotations

import numpy as np

from .color import ciede2000_array, relative_luminance_array, rgb_array_to_lab


def _rgb_image_array(value) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("RGB image array must have shape (height, width, 3)")
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
        return np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _opaque_mask(shape: tuple[int, int], transparent_mask: np.ndarray | None) -> np.ndarray:
    if transparent_mask is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(transparent_mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError("transparent_mask shape must match image height/width")
    return ~mask


def _masked_rgb(value, transparent_mask: np.ndarray | None = None) -> np.ndarray:
    rgb = _rgb_image_array(value)
    opaque = _opaque_mask(rgb.shape[:2], transparent_mask)
    if not np.any(opaque):
        return np.empty((0, 3), dtype=np.uint8)
    return rgb[opaque].reshape(-1, 3)


def rgb_mse(predicted, target, *, transparent_mask: np.ndarray | None = None) -> float:
    """Mean squared RGB error on normalized [0, 1] channels."""

    pred = _rgb_image_array(predicted)
    tgt = _rgb_image_array(target)
    if pred.shape != tgt.shape:
        raise ValueError("predicted and target arrays must have the same shape")
    opaque = _opaque_mask(pred.shape[:2], transparent_mask)
    if not np.any(opaque):
        return 0.0
    diff = pred.astype(np.float32) / 255.0 - tgt.astype(np.float32) / 255.0
    error = np.mean(diff * diff, axis=-1)
    return float(np.mean(error[opaque]))


def delta_e_stats(predicted, target, *, transparent_mask: np.ndarray | None = None) -> dict[str, float]:
    """CIEDE2000 summary stats for predicted preview vs target."""

    pred = _rgb_image_array(predicted)
    tgt = _rgb_image_array(target)
    if pred.shape != tgt.shape:
        raise ValueError("predicted and target arrays must have the same shape")
    opaque = _opaque_mask(pred.shape[:2], transparent_mask)
    if not np.any(opaque):
        return {"mean_delta_e": 0.0, "p95_delta_e": 0.0, "max_delta_e": 0.0}
    pred_lab = rgb_array_to_lab(pred[opaque].reshape(-1, 3))
    tgt_lab = rgb_array_to_lab(tgt[opaque].reshape(-1, 3))
    delta = np.asarray(ciede2000_array(pred_lab, tgt_lab), dtype=float)
    return {
        "mean_delta_e": float(np.mean(delta)),
        "p95_delta_e": float(np.percentile(delta, 95)),
        "max_delta_e": float(np.max(delta)),
    }


def tonal_coverage(rgb, *, transparent_mask: np.ndarray | None = None, prefix: str = "") -> dict[str, float]:
    """Luminance min/max/span for opaque pixels."""

    pixels = _masked_rgb(rgb, transparent_mask)
    if pixels.size == 0:
        return {
            f"{prefix}tonal_min": 0.0,
            f"{prefix}tonal_max": 0.0,
            f"{prefix}tonal_span": 0.0,
        }
    lum = relative_luminance_array(pixels)
    return {
        f"{prefix}tonal_min": float(np.min(lum)),
        f"{prefix}tonal_max": float(np.max(lum)),
        f"{prefix}tonal_span": float(np.max(lum) - np.min(lum)),
    }


def clip_fractions(
    rgb,
    *,
    transparent_mask: np.ndarray | None = None,
    shadow_threshold: float = 0.02,
    highlight_threshold: float = 0.98,
) -> dict[str, float]:
    """Fractions of opaque pixels clipped near black or white."""

    pixels = _masked_rgb(rgb, transparent_mask)
    if pixels.size == 0:
        return {"shadow_clip_fraction": 0.0, "highlight_clip_fraction": 0.0}
    lum = relative_luminance_array(pixels)
    return {
        "shadow_clip_fraction": float(np.mean(lum <= shadow_threshold)),
        "highlight_clip_fraction": float(np.mean(lum >= highlight_threshold)),
    }


def banding_risk(layer_rgb) -> dict[str, float]:
    """Heuristic banding risk from adjacent visible layer color spacing."""

    layers = _rgb_image_array(np.asarray(layer_rgb).reshape(-1, 1, 3)).reshape(-1, 3)
    if layers.shape[0] < 2:
        return {
            "banding_risk": 0.0,
            "min_adjacent_delta_e": 99.0,
            "repeated_layer_fraction": 0.0,
        }
    labs = rgb_array_to_lab(layers)
    adjacent = np.asarray(ciede2000_array(labs[:-1], labs[1:]), dtype=float)
    repeated = adjacent < 1.0
    low_contrast = adjacent < 3.0
    return {
        "banding_risk": float((np.mean(repeated) * 0.7) + (np.mean(low_contrast) * 0.3)),
        "min_adjacent_delta_e": float(np.min(adjacent)),
        "repeated_layer_fraction": float(np.mean(repeated)),
    }


def preview_metrics(
    predicted,
    target,
    *,
    transparent_mask: np.ndarray | None = None,
    layer_rgb=None,
) -> dict[str, float]:
    """Combined metrics used by Prism diagnostics and Quilt reports."""

    pred = _rgb_image_array(predicted)
    tgt = _rgb_image_array(target)
    metrics: dict[str, float] = {
        "rgb_mse": rgb_mse(pred, tgt, transparent_mask=transparent_mask),
        **delta_e_stats(pred, tgt, transparent_mask=transparent_mask),
        **tonal_coverage(tgt, transparent_mask=transparent_mask, prefix="target_"),
        **tonal_coverage(pred, transparent_mask=transparent_mask, prefix="predicted_"),
        **clip_fractions(tgt, transparent_mask=transparent_mask),
    }
    if layer_rgb is not None:
        metrics.update(banding_risk(layer_rgb))
    return metrics


__all__ = [
    "banding_risk",
    "clip_fractions",
    "delta_e_stats",
    "preview_metrics",
    "rgb_mse",
    "tonal_coverage",
]
