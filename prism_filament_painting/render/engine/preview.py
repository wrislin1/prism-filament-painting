"""Preview rendering for Quilt height fields."""

from __future__ import annotations

import numpy as np
from PIL import Image

from prism_filament_painting.render.errors import QuiltInternalError

from .params import ForwardTable


def render_preview(height_field: np.ndarray, table: ForwardTable) -> Image.Image:
    field = np.asarray(height_field, dtype=np.int32)
    lut = np.asarray(table.layer_rgb, dtype=np.uint8)
    if field.min() < 0 or field.max() >= len(lut):
        raise QuiltInternalError("height_field contains layer outside forward table")
    rgb = lut[field]
    return Image.fromarray(rgb, "RGB")
