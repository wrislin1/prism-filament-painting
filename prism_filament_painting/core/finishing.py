"""Raster finishing on square print cells. Zero height denotes a through-cut."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageOps

from .color import relative_luminance_array
from .forward import continuous_layer_height, resolve_orientation


@dataclass(frozen=True)
class PrintOptions:
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0
    radius: float = 0
    hole_x: float = 0
    hole_y: float = 0
    diameter: float = 0
    smoothing: float = 0
    dither: bool = False

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or data.keys() - cls.__dataclass_fields__.keys():
            raise ValueError("Unknown print option")
        options = cls(**data)
        for name in ("left", "top", "right", "bottom"):
            value = getattr(options, name)
            if type(value) is not int or not 0 <= value <= 2**31 - 1:
                raise ValueError("Crop margins must be nonnegative whole pixels")
        for name in ("radius", "diameter", "hole_x", "hole_y", "smoothing"):
            value = getattr(options, name)
            if type(value) not in (float, int) or not 0 <= value <= 1e9:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if options.smoothing > 1 or type(options.dither) is not bool:
            raise ValueError("Use smoothing from 0 to 1 and a boolean dither switch")
        return options

    @property
    def active(self):
        return any((self.left, self.top, self.right, self.bottom, self.radius,
                    self.diameter, self.smoothing, self.dither))


@dataclass(frozen=True)
class PrintGrid:
    source: Image.Image
    heights: np.ndarray
    pitch: float
    factor: int


def _outline(size, pitch, options):
    w, h = size
    if not options.radius and not options.diameter:
        return np.ones((h, w), dtype=bool)
    width, height = w * pitch, h * pitch
    radius = options.radius
    if 2 * radius > min(width, height):
        raise ValueError("Corner radius exceeds half the short side")
    y, x = np.ogrid[:h, :w]
    x, y = (x + 0.5) * pitch, (y + 0.5) * pitch

    def edge_distance(x, y):
        dx = np.abs(x - width / 2) - (width / 2 - radius)
        dy = np.abs(y - height / 2) - (height / 2 - radius)
        return radius - np.hypot(np.maximum(dx, 0), np.maximum(dy, 0)) - np.minimum(np.maximum(dx, dy), 0)

    keep = edge_distance(x, y) >= 0
    if radius and keep.all():
        raise ValueError("Corner radius is too small for this print grid")
    if options.diameter:
        hole_radius = options.diameter / 2
        # Each snapped boundary can move by at most half a cell diagonal.
        clearance = edge_distance(options.hole_x, options.hole_y) - hole_radius
        if clearance < 1.2 + math.sqrt(2) * pitch:
            raise ValueError("The hanging hole must leave a 1.2 mm wall after snapping to the print grid")
        hole = np.hypot(x - options.hole_x, y - options.hole_y) <= hole_radius
        if not hole.any():
            raise ValueError("Hole diameter is too small for this print grid")
        keep &= ~hole
    return keep


def _triangle_count(field):
    """Top and bottom quads, unit-height walls, and clipped saddle corners."""
    solid = field.astype(np.int64, copy=False)
    edges = np.pad(solid, 1)
    walls = np.abs(np.diff(edges, axis=0)).sum() + np.abs(np.diff(edges, axis=1)).sum()
    a, b, c, d = solid[:-1, :-1], solid[:-1, 1:], solid[1:, :-1], solid[1:, 1:]
    saddle = (np.minimum(a, d) > np.maximum(b, c)) | (np.minimum(b, c) > np.maximum(a, d))
    corners = sum(((cell + (cell > 0)) * saddle).sum() for cell in (a, b, c, d))
    return int(4 * np.count_nonzero(field) + 2 * (walls + corners))


def finished_grid(image, stack, mesh, options, base_factor):
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    # Preserve the physical scale of the original, triangle-capped plate when
    # a detail edit needs a coarser grid or a crop removes its longest edge.
    if type(base_factor) is not int or base_factor < 1:
        raise ValueError("The original downsample factor must be a positive integer")
    base_factor = min(base_factor, max(image.size))
    original_cells = math.ceil(max(image.size) / base_factor)
    original_mm = min(original_cells * mesh["xy_pixel_mm"], mesh["max_dim_mm"])
    pitch = original_mm / max(image.size)
    box = (options.left, options.top, image.width - options.right, image.height - options.bottom)
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("Crop must leave at least one source pixel on each axis")
    source = image if box == (0, 0, *image.size) else image.crop(box)
    cap = mesh["max_triangles"]
    factor = max(base_factor, math.ceil(math.sqrt(source.width * source.height * 4 / cap)))
    orientation = resolve_orientation(stack, orientation=mesh.get("orientation", "auto"))
    while True:
        size = tuple(math.ceil(side / factor) for side in source.size)
        cell_mm = pitch * max(source.size) / max(size)
        keep = _outline(size, cell_mm, options)
        grid = source.resize(size, Image.Resampling.BOX) if size != source.size else source
        rgba = np.asarray(grid)
        constants = stack.constants
        lum = relative_luminance_array(rgba[..., :3])
        heights = continuous_layer_height(lum, constants=constants, orientation=orientation)
        if mesh["flatten_transparent"]:
            heights[rgba[..., 3] == 0] = constants.base_layers
        field = quantize_detail(heights, options, keep)
        if mesh["flatten_transparent"]:
            field[keep & (rgba[..., 3] == 0)] = constants.base_layers
        count = _triangle_count(field)
        if count <= cap:
            return PrintGrid(source, field, cell_mm, factor)
        if size == (1, 1):
            raise ValueError("Plate exceeds the triangle cap")
        factor = max(factor + 1, math.ceil(factor * math.sqrt(count / cap)))


def quantize_detail(heights, options, keep):
    """Filter continuous heights inside the plate and snap to printable layers."""
    if options.smoothing:
        # Cross-shaped local mean, excluding cutouts, with edge replication.
        padded = np.pad(heights * keep, 1, mode="edge")
        weights = np.pad(keep.astype(int), 1, mode="edge")

        def cross(a):
            return a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]

        mean = cross(padded) / np.maximum(1, cross(weights))
        heights = heights + options.smoothing * (mean - heights)
    if options.dither:
        # Ordered thresholds use the two neighbouring printable layers in proportion.
        order = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
        y, x = np.ogrid[:heights.shape[0], :heights.shape[1]]
        lower = np.floor(heights)
        result = lower + ((heights - lower) > (order[y % 4, x % 4] + 0.5) / 16)
    else:
        result = np.rint(heights)
    return np.where(keep, result, 0).astype(np.int32)
