"""Deterministic filament painting suitability analysis for source images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from prism_filament_painting.core.forward import (
    RenderConstants,
    build_forward_stack,
    image_to_layer_field,
    luminance_to_layer,
    resolve_orientation,
)

from .color import relative_luminance_array


MIN_BAND_UTILIZATION = 0.5


@dataclass(frozen=True)
class PrintabilityReport:
    score: int
    dynamic_range: float
    shadow_clip_pct: float
    highlight_clip_pct: float
    distinct_layers: int
    layer_budget: int
    band_utilization: float
    histogram: list[int]
    issues: list[dict]

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "dynamic_range": self.dynamic_range,
            "shadow_clip_pct": self.shadow_clip_pct,
            "highlight_clip_pct": self.highlight_clip_pct,
            "distinct_layers": self.distinct_layers,
            "layer_budget": self.layer_budget,
            "band_utilization": self.band_utilization,
            "histogram": list(self.histogram),
            "issues": [dict(issue) for issue in self.issues],
        }


def analyze_printability(image: Image.Image, *, total_layers: int) -> PrintabilityReport:
    base_layers = RenderConstants().base_layers
    if (
        isinstance(total_layers, bool)
        or not isinstance(total_layers, int)
        or total_layers <= base_layers
    ):
        raise ValueError(f"total_layers must be an integer greater than {base_layers}")
    if image.width * image.height == 0:
        raise ValueError("image has no pixels")

    sample = _prepare_image(image)
    rgb = np.asarray(sample, dtype=np.uint8).reshape(-1, 3)
    lum = np.clip(relative_luminance_array(rgb), 0.0, 1.0)

    p1, p99 = np.percentile(lum, [1, 99])
    dynamic_range = float(np.clip(p99 - p1, 0.0, 1.0))
    shadow_clip_pct = float(np.mean(lum <= 0.02))
    highlight_clip_pct = float(np.mean(lum >= 0.98))

    constants = RenderConstants(total_layers=total_layers)
    layers = luminance_to_layer(lum, constants=constants, orientation="bright")
    distinct_layers = int(np.unique(layers).size)
    layer_budget = total_layers - constants.base_layers + 1
    band_utilization = float(distinct_layers / layer_budget)
    histogram = np.histogram(lum, bins=32, range=(0.0, 1.0))[0].astype(int).tolist()

    issues = _issues(
        dynamic_range=dynamic_range,
        band_utilization=band_utilization,
        shadow_clip_pct=shadow_clip_pct,
        highlight_clip_pct=highlight_clip_pct,
    )
    score = _score(
        dynamic_range=dynamic_range,
        band_utilization=band_utilization,
        shadow_clip_pct=shadow_clip_pct,
        highlight_clip_pct=highlight_clip_pct,
    )

    return PrintabilityReport(
        score=score,
        dynamic_range=dynamic_range,
        shadow_clip_pct=shadow_clip_pct,
        highlight_clip_pct=highlight_clip_pct,
        distinct_layers=distinct_layers,
        layer_budget=layer_budget,
        band_utilization=band_utilization,
        histogram=histogram,
        issues=issues,
    )


def resolved_bands(
    image: Image.Image,
    filaments,
    swap_schedule,
    *,
    total_layers: int,
    constants: RenderConstants | None = None,
    layer_field_cache: dict | None = None,
) -> dict:
    """Count the distinct colors a print actually resolves for one palette.

    Palette-aware counterpart to ``analyze_printability``'s palette-agnostic
    ``band_utilization``: builds the real Beer-Lambert forward stack for
    ``filaments`` + ``swap_schedule`` and counts the *distinct resolved colors*
    the print produces across the layers this image reaches. Adjacent layers can
    resolve to the same color, so this is <= the luminance-band count.

    The per-pixel layer field depends on (image, ``constants``, resolved
    orientation) — not on filament colors — so a caller ranking many candidates
    for one image may pass a shared ``layer_field_cache`` dict to compute the field
    at most once per (orientation, constants). The cache key carries ``constants``
    (a frozen, hashable dataclass), so reuse stays correct even when the cache is
    shared across calls with different geometry on the same image.

    Takes ``palette_core`` types only (no Prism dependency). Returns a dict of
    native ints: ``{"resolved", "budget"}``.
    """
    base_layers = RenderConstants().base_layers
    if (
        isinstance(total_layers, bool)
        or not isinstance(total_layers, int)
        or total_layers <= base_layers
    ):
        raise ValueError(f"total_layers must be an integer greater than {base_layers}")
    if image.width * image.height == 0:
        raise ValueError("image has no pixels")

    if constants is None:
        constants = RenderConstants(total_layers=total_layers)
    stack = build_forward_stack(
        filaments, constants=constants, swap_schedule=tuple(swap_schedule) or None
    )
    orientation = resolve_orientation(stack, orientation="auto")
    # The field depends on (image, constants, orientation). Key the cache on
    # (orientation, constants) so a shared cache can't return a wrong reuse when
    # geometry differs; the cache is still scoped to a single image per caller.
    cache_key = (orientation, constants)
    if layer_field_cache is not None and cache_key in layer_field_cache:
        field = layer_field_cache[cache_key]
    else:
        field = image_to_layer_field(
            image, constants=constants, orientation=orientation, flatten_transparent=True
        )
        if layer_field_cache is not None:
            layer_field_cache[cache_key] = field

    reached = np.unique(field)
    resolved = int(np.unique(stack.layer_rgb[reached], axis=0).shape[0])
    budget = constants.total_layers - constants.base_layers + 1
    return {"resolved": resolved, "budget": budget}


def _prepare_image(image: Image.Image) -> Image.Image:
    sample = image.convert("RGB") if image.mode != "RGB" else image
    max_dim = max(sample.size)
    if max_dim <= 256:
        return sample

    scale = 256 / max_dim
    size = (max(1, round(sample.width * scale)), max(1, round(sample.height * scale)))
    return sample.resize(size, Image.Resampling.NEAREST)


def _issues(
    *,
    dynamic_range: float,
    band_utilization: float,
    shadow_clip_pct: float,
    highlight_clip_pct: float,
) -> list[dict]:
    issues: list[dict] = []
    if dynamic_range < 0.6:
        issues.append(
            {
                "code": "low_dynamic_range",
                "severity": "warn",
                "message": "Image luminance uses a narrow tonal range.",
            }
        )
    if band_utilization < MIN_BAND_UTILIZATION:
        issues.append(
            {
                "code": "few_bands",
                "severity": "warn",
                "message": "Image maps to too few printable layer bands.",
            }
        )
    if shadow_clip_pct > 0.02:
        issues.append(
            {
                "code": "shadow_clipping",
                "severity": "warn",
                "message": "More than 2% of pixels are near black.",
            }
        )
    if highlight_clip_pct > 0.02:
        issues.append(
            {
                "code": "highlight_clipping",
                "severity": "warn",
                "message": "More than 2% of pixels are near white.",
            }
        )
    return issues


def _score(
    *,
    dynamic_range: float,
    band_utilization: float,
    shadow_clip_pct: float,
    highlight_clip_pct: float,
) -> int:
    clipping_penalty = min(0.4, shadow_clip_pct + highlight_clip_pct)
    raw = (0.5 * dynamic_range) + (0.5 * band_utilization) - clipping_penalty
    return int(round(float(np.clip(raw, 0.0, 1.0)) * 100))
