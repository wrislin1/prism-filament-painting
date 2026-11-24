"""Prism Beta layer-target scoring primitives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from PIL import Image

from prism_filament_painting.core.forward import RenderConstants, luminance_to_layer

from .color import ciede2000, ciede2000_array, relative_luminance_array, rgb_array_to_lab

_TARGET_MAX_DIM = 128
_MAX_LAYER_SAMPLES = 16


@dataclass(frozen=True)
class LayerTarget:
    layer: int
    weight: float
    mean_lab: np.ndarray
    samples_lab: np.ndarray
    sample_weights: np.ndarray
    chroma_variance: float


@dataclass(frozen=True)
class LayerTargetModel:
    by_orientation: Mapping[str, tuple[LayerTarget, ...]]
    opaque_pixel_count: int
    layer_count: int


def build_layer_target_model(
    image: Image.Image,
    *,
    constants: RenderConstants,
    max_dim: int = _TARGET_MAX_DIM,
    max_samples_per_layer: int = _MAX_LAYER_SAMPLES,
) -> LayerTargetModel:
    """Map source pixels onto filament painting layers for both print orientations."""

    rgba = image if image.mode == "RGBA" else image.convert("RGBA")
    if max(rgba.size) > max_dim:
        rgba = rgba.copy()
        rgba.thumbnail((max_dim, max_dim), Image.Resampling.NEAREST)
    arr = np.asarray(rgba, dtype=np.uint8).reshape(-1, 4)
    opaque = arr[:, 3] != 0
    if opaque.any():
        rgb = arr[opaque, :3]
    else:
        rgb = arr[:, :3]
    if rgb.shape[0] == 0:
        raise ValueError("image has no pixels")

    lum = relative_luminance_array(rgb)
    lab = rgb_array_to_lab(rgb)
    by_orientation = {
        orientation: _targets_for_layers(
            lab,
            luminance_to_layer(lum, constants=constants, orientation=orientation),
            total_pixels=rgb.shape[0],
            max_samples_per_layer=max_samples_per_layer,
        )
        for orientation in ("bright", "dark")
    }
    return LayerTargetModel(
        by_orientation=by_orientation,
        opaque_pixel_count=int(rgb.shape[0]),
        layer_count=int(constants.total_layers) + 1,
    )


def layer_cost(predicted_lab: np.ndarray, target: LayerTarget) -> float:
    """Weighted CIEDE2000 from one predicted layer color to a target distribution."""

    pred = np.asarray(predicted_lab, dtype=float)
    samples = np.asarray(target.samples_lab, dtype=float)
    if samples.size == 0:
        return ciede2000(pred, target.mean_lab)
    repeated = np.repeat(pred.reshape(1, 3), samples.shape[0], axis=0)
    delta = np.asarray(ciede2000_array(repeated, samples), dtype=float)
    weights = np.asarray(target.sample_weights, dtype=float)
    if weights.shape != delta.shape or float(np.sum(weights)) <= 0:
        weights = np.full(delta.shape, 1.0 / max(1, delta.shape[0]), dtype=float)
    return float(np.sum(delta * weights))


def _targets_for_layers(
    lab: np.ndarray,
    layers: np.ndarray,
    *,
    total_pixels: int,
    max_samples_per_layer: int,
) -> tuple[LayerTarget, ...]:
    targets: list[LayerTarget] = []
    for layer in sorted(int(value) for value in np.unique(layers)):
        mask = layers == layer
        layer_lab = lab[mask]
        if layer_lab.size == 0:
            continue
        samples = _representative_samples(layer_lab, max_samples=max_samples_per_layer)
        weights = np.full(samples.shape[0], 1.0 / samples.shape[0], dtype=float)
        chroma = np.hypot(layer_lab[:, 1], layer_lab[:, 2])
        targets.append(
            LayerTarget(
                layer=layer,
                weight=float(layer_lab.shape[0] / total_pixels),
                mean_lab=layer_lab.mean(axis=0),
                samples_lab=samples,
                sample_weights=weights,
                chroma_variance=float(np.var(chroma)) if chroma.size else 0.0,
            )
        )
    return tuple(targets)


def _representative_samples(lab: np.ndarray, *, max_samples: int) -> np.ndarray:
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1")
    if lab.shape[0] <= max_samples:
        return np.asarray(lab, dtype=float)
    order = np.lexsort((lab[:, 2], lab[:, 1], lab[:, 0]))
    sorted_lab = lab[order]
    indices = np.linspace(0, sorted_lab.shape[0] - 1, max_samples)
    return sorted_lab[np.rint(indices).astype(int)]


__all__ = ["LayerTarget", "LayerTargetModel", "build_layer_target_model", "layer_cost"]
