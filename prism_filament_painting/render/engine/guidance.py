"""Colour and relief costs for optional source-aligned optimizer maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from prism_filament_painting.render.errors import QuiltInputError

from .params import LayerParams, OptimizeParams

if TYPE_CHECKING:
    from torch import Tensor


def validate_maps(params: OptimizeParams, image_size: tuple[int, int]) -> None:
    if not 0 <= params.height_strength <= 1:
        raise QuiltInputError("height_strength must be between 0 and 1")
    if params.height_strength and params.depth_map is None:
        raise QuiltInputError("height_strength requires a depth_map")
    for name, image in (("focus_mask", params.focus_mask), ("depth_map", params.depth_map)):
        if image is not None and image.size != image_size:
            raise QuiltInputError(f"{name} must match the source image dimensions")


@dataclass(frozen=True)
class Guidance:
    # Arrays have one entry per printable pixel. The same expression is used
    # with NumPy while screening stacks and with torch for local refinement.
    weight: np.ndarray | Tensor
    depth: np.ndarray | Tensor | None
    strength: float
    base: int
    span: int

    @classmethod
    def prepare(cls, params: OptimizeParams, shape: tuple[int, int],
                transparent: np.ndarray | None, layers: LayerParams) -> Guidance | None:
        def sample(image):
            return np.asarray(image.convert("L").resize(shape[::-1], Image.Resampling.BOX),
                              dtype=np.float32) / 255

        if params.focus_mask is None and not params.height_strength:
            return None
        weight = np.ones(shape, np.float32)
        if params.focus_mask is not None:
            weight += 99 * sample(params.focus_mask)
        visible = weight if transparent is None else weight[~transparent]
        if not visible.size or (not params.height_strength and visible.min() == visible.max()):
            return None
        weight /= visible.mean()
        depth = sample(params.depth_map) if params.height_strength else None
        return cls(weight, depth, params.height_strength, layers.base_layers,
                   max(1, layers.total_layers - layers.base_layers))

    def costs(self, colour_error, heights, weights, depth):
        cost = colour_error * weights
        if depth is not None:
            cost = cost + self.strength * (((heights - self.base) / self.span) - depth) ** 2
        return cost

    def field_loss(self, colour_error, heights, visible):
        costs = self.costs(colour_error, heights, self.weight, self.depth)
        return costs.mean() if visible is None else costs[visible].mean()

    def nearest(self, target, table, transparent):
        """Return exact printable minimizers and their mean objective, in bounded batches."""
        colours = np.asarray(table.layer_rgb[self.base:], dtype=np.float32) / 255
        levels = np.arange(self.base, self.base + len(colours), dtype=np.float32)
        pixels = target.reshape(-1, 3)
        weights = self.weight.reshape(-1, 1)
        depth = None if self.depth is None else self.depth.reshape(-1, 1)
        field = np.empty(len(pixels), dtype=np.int32)
        minima = np.empty(len(pixels), dtype=np.float32)
        batch = max(1, 262_144 // len(levels))
        for start in range(0, len(pixels), batch):
            rows = slice(start, start + batch)
            errors = ((pixels[rows, None] - colours) ** 2).mean(axis=-1)
            costs = self.costs(errors, levels, weights[rows], None if depth is None else depth[rows])
            chosen = costs.argmin(axis=1)
            field[rows] = chosen + self.base
            minima[rows] = costs[np.arange(len(chosen)), chosen]
        if transparent is not None:
            minima = minima[~transparent.ravel()]
        return field.reshape(target.shape[:2]), float(minima.mean())

    def on_device(self, torch, device):
        return Guidance(torch.as_tensor(self.weight, device=device),
                        None if self.depth is None else torch.as_tensor(self.depth, device=device),
                        self.strength, self.base, self.span)
