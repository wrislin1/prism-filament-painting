"""Deterministic dominant-color extraction."""

from __future__ import annotations

import numpy as np
from PIL import Image


def extract_dominant(image: Image.Image, k: int) -> list[tuple[tuple[int, int, int], float]]:
    if k < 1:
        raise ValueError("k must be at least 1")

    if "A" in image.getbands():
        # Drop fully-transparent pixels before quantization so background alpha=0
        # pixels (e.g. from a Patchwork foreground mask) don't skew the palette.
        # Mirrors the alpha==0-as-background convention in rank._bin_target_by_luminance
        # and scoring.build_layer_target_model.
        rgba = image if image.mode == "RGBA" else image.convert("RGBA")
        arr = np.asarray(rgba, dtype=np.uint8).reshape(-1, 4)
        opaque = arr[:, 3] != 0
        if opaque.any():
            rgb_pixels = arr[opaque, :3]  # shape (N, 3)
        else:
            # Fully-transparent image: fall back to all pixels (mirrors rank.py guard).
            rgb_pixels = arr[:, :3]
        # Reconstruct a minimal (N, 1) RGB image so quantize() can operate on it.
        n = rgb_pixels.shape[0]
        rgb = Image.fromarray(rgb_pixels.reshape(n, 1, 3), mode="RGB")
    else:
        rgb = image.convert("RGB")

    quantized = rgb.quantize(colors=k, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette() or []
    counts = quantized.getcolors(maxcolors=rgb.width * rgb.height)
    if counts is None:
        raise ValueError("could not extract color counts")
    total = sum(count for count, _index in counts)
    weighted = []
    for count, index in counts:
        offset = index * 3
        color = tuple(palette[offset : offset + 3])
        weighted.append((color, count / total))
    return sorted(weighted, key=lambda item: (-item[1], item[0]))
