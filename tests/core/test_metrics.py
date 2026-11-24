"""Shared preview metric tests for Prism and Quilt."""

from __future__ import annotations

import numpy as np
import pytest

from prism_filament_painting.core.metrics import (
    banding_risk,
    clip_fractions,
    delta_e_stats,
    preview_metrics,
    rgb_mse,
    tonal_coverage,
)


def test_rgb_mse_respects_transparent_mask() -> None:
    predicted = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    target = np.array([[[0, 0, 0], [0, 0, 0]]], dtype=np.uint8)
    transparent = np.array([[False, True]])

    assert rgb_mse(predicted, target, transparent_mask=transparent) == 0.0
    assert rgb_mse(predicted, target) == pytest.approx(0.5)


def test_delta_e_stats_reports_mean_percentile_and_max() -> None:
    predicted = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    target = np.array([[[0, 0, 0], [250, 250, 250]]], dtype=np.uint8)

    stats = delta_e_stats(predicted, target)

    assert stats["mean_delta_e"] > 0
    assert stats["p95_delta_e"] >= stats["mean_delta_e"]
    assert stats["max_delta_e"] >= stats["p95_delta_e"]


def test_clip_fractions_and_tonal_coverage_ignore_transparent_pixels() -> None:
    rgb = np.array([[[0, 0, 0], [128, 128, 128], [255, 255, 255]]], dtype=np.uint8)
    transparent = np.array([[False, False, True]])

    clips = clip_fractions(rgb, transparent_mask=transparent)
    coverage = tonal_coverage(rgb, transparent_mask=transparent)

    assert clips["shadow_clip_fraction"] == pytest.approx(0.5)
    assert clips["highlight_clip_fraction"] == pytest.approx(0.0)
    assert 0.0 < coverage["tonal_span"] < 1.0


def test_banding_risk_increases_when_layer_colors_repeat() -> None:
    smooth = np.array(
        [[0, 0, 0], [64, 64, 64], [128, 128, 128], [192, 192, 192], [255, 255, 255]],
        dtype=np.uint8,
    )
    repeated = np.array(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255]],
        dtype=np.uint8,
    )

    assert banding_risk(repeated)["banding_risk"] > banding_risk(smooth)["banding_risk"]


def test_preview_metrics_combines_shared_fields() -> None:
    predicted = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    target = np.array([[[0, 0, 0], [250, 250, 250]]], dtype=np.uint8)

    metrics = preview_metrics(predicted, target)

    assert set(metrics) >= {
        "rgb_mse",
        "mean_delta_e",
        "p95_delta_e",
        "max_delta_e",
        "target_tonal_span",
        "predicted_tonal_span",
        "shadow_clip_fraction",
        "highlight_clip_fraction",
    }
