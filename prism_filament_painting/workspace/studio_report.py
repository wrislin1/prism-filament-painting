"""Compare a Studio print with the source using CIEDE2000 under neutral lighting."""

import base64
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
from prism_filament_painting.core.color import ciede2000_array, rgb_array_to_lab
from prism_filament_painting.core.forward import (
    ForwardStack,
    RenderConstants,
    build_forward_stack,
    image_to_layer_field,
    resolve_orientation,
)
from prism_filament_painting.core.finishing import PrintOptions, finished_grid
from prism_filament_painting.core.model import physical_filament


# Prism’s calibration wedge treats colours within 2 ΔE00 as indistinguishable.
REACH_LEVELS = (("Indistinguishable", 2), ("Noticeable", 5), ("Clearly different", 10))
_PIXELS_PER_BLOCK = 65_536


def _indexed_png(indices: np.ndarray, palette: list[int]) -> str:
    image = Image.frombytes("P", (indices.shape[1], indices.shape[0]), indices.tobytes())
    image.putpalette(palette)
    data = BytesIO()
    image.save(data, format="PNG", transparency=0)
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode("ascii")


def _print_layers(image: Image.Image, stack: ForwardStack, mesh: dict, factor: int) -> np.ndarray:
    # Studio supplies the factor from its triangle-cap recovery. Use the renderer's BOX
    # grid, then expand its discrete printed cells to compare every source pixel.
    size = tuple((side + factor - 1) // factor for side in image.size)
    grid = image.resize(size, Image.Resampling.BOX) if size != image.size else image
    pixels = np.asarray(grid).reshape(-1, 4)
    field = np.empty(len(pixels), dtype=np.uint8)  # Studio caps stacks at 128 layers.
    orientation = resolve_orientation(stack, orientation=mesh.get("orientation", "auto"))
    for start in range(0, len(field), _PIXELS_PER_BLOCK):
        block = slice(start, start + _PIXELS_PER_BLOCK)
        field[block] = image_to_layer_field(
            Image.fromarray(pixels[block][None, :, :]),
            constants=stack.constants,
            orientation=orientation,
            flatten_transparent=mesh["flatten_transparent"],
        ).ravel()
    layers = Image.fromarray(field.reshape(grid.height, grid.width))
    return np.asarray(layers.resize(image.size, Image.Resampling.NEAREST)).ravel()


def _reach_class(lab: np.ndarray, printable: np.ndarray) -> np.ndarray:
    """Number of tolerances exceeded by the nearest printable colour."""
    limits = [threshold for _, threshold in REACH_LEVELS]
    nearest = np.full(len(lab), np.inf)
    for color in printable:
        # For sRGB lightness (0–100), the CIEDE2000 lightness weight is below 2.
        # Once the strictest tolerance is met, further colours cannot change the class.
        candidates = np.flatnonzero((nearest > limits[0]) & (np.abs(lab[:, 0] - color[0]) <= 2 * limits[-1]))
        if candidates.size:
            nearest[candidates] = np.minimum(nearest[candidates], ciede2000_array(color, lab[candidates]))
    return np.searchsorted(limits, nearest, side="left")


def _layer_sources(stack: ForwardStack) -> np.ndarray:
    return np.asarray(stack.layer_sources)


def studio_stack(state: dict) -> ForwardStack:
    layer = state["layer"]
    luminance = state["luminance"]
    constants = RenderConstants(
        front_lit_scale=state.get("front_lit_scale", RenderConstants().front_lit_scale),
        layer_height_mm=layer["layer_height_mm"],
        base_layers=layer["base_layers"],
        total_layers=layer["total_layers"],
        black_point=luminance["black_point"],
        white_point=luminance["white_point"],
        gamma=luminance["gamma"],
    )
    filaments = [
        physical_filament(item.get("brand", ""), item.get("name", ""), item["hex"],
                          item["input_distance_mm"] if "input_distance_mm" in item else item["td"], item.get("spool_id"))
        for item in state["palette"]["filaments"]
    ]
    return build_forward_stack(filaments, constants=constants, swap_schedule=state["swap_schedule"])


def predicted_accuracy(image: Image.Image, state: dict, downsample_factor: int) -> dict:
    stack = studio_stack(state)
    constants = stack.constants
    options = PrintOptions.parse(state.get("print_options", {}))
    if options.active:
        grid = finished_grid(image, stack, state["mesh"], options, downsample_factor)
        image = grid.source
        layers = np.asarray(Image.fromarray(grid.heights).resize(image.size, Image.Resampling.NEAREST)).ravel()
    else:
        image = ImageOps.exif_transpose(image)
        layers = _print_layers(image, stack, state["mesh"], downsample_factor)
    layer_lab = rgb_array_to_lab(stack.layer_rgb)
    return pixel_accuracy(image, layers, layer_lab, slice(constants.base_layers, None), _layer_sources(stack),
                          occupied=layers > 0 if options.active else None)


def pixel_accuracy(image, layers, layer_lab, eligible, sources, *, occupied=None, regional_gamuts=None):
    """Score visible pixels, optionally using region IDs and their printable Lab gamuts."""
    printable = np.unique(layer_lab[eligible], axis=0)
    pixels = np.asarray(image).reshape(-1, 4)
    visible = pixels[:, 3] != 0
    if occupied is not None:
        visible &= occupied
    count = int(visible.sum())
    errors = np.zeros(len(pixels), dtype=float)
    # Zero marks transparency. Visible codes are one plus the number of exceeded thresholds.
    flags = np.zeros(len(pixels), dtype=np.uint8)
    for start in range(0, len(pixels), _PIXELS_PER_BLOCK):
        indices = start + np.flatnonzero(visible[start:start + _PIXELS_PER_BLOCK])
        if not len(indices):
            continue
        target = rgb_array_to_lab(pixels[indices, :3])
        errors[indices] = ciede2000_array(layer_lab[layers[indices]], target)
        if regional_gamuts is None:
            flags[indices] = 1 + _reach_class(target, printable)
        else:
            region_ids, gamuts = regional_gamuts
            owners = region_ids[indices]
            for owner in np.unique(owners):
                local = owners == owner
                flags[indices[local]] = 1 + _reach_class(target[local], gamuts[owner])

    # Reduce the visible float64 distances in source order, as delta_e_stats does.
    mean = float(errors[visible].mean()) if count else None
    peak = float(errors.max())
    heat = np.rint(errors * (254 / peak)).astype(np.uint8) if peak else np.zeros_like(flags)
    heat += visible
    shape = (image.height, image.width)
    # Index 0 is transparent. Visible colours run from black to amber.
    heat_palette = [0, 0, 0]
    for level in range(255):
        heat_palette.extend((level, round(level * 0.75), round(level * 0.2)))
    levels = []
    for index, (label, threshold) in enumerate(REACH_LEVELS):
        missing = flags > index + 1
        mask = visible.astype(np.uint8) + missing
        levels.append({
            "label": label, "threshold": threshold, "pixels": int(missing.sum()),
            "map": _indexed_png(mask.reshape(shape), [0, 0, 0, 42, 45, 52, 255, 63, 162]),
        })
    used_heights = np.unique(layers[visible])
    return {
        "mean_delta_e00": mean,
        "peak_delta_e00": peak if count else None,
        "visible_pixels": count,
        "unreachable_pixels": levels[0]["pixels"],
        "unreachable_threshold": levels[0]["threshold"],
        "reach_levels": levels,
        "prediction_sources": {
            "difference": np.unique(sources[used_heights]).tolist(),
            "reachability": np.unique(sources[eligible]).tolist(),
        },
        "heat_map": _indexed_png(heat.reshape(shape), heat_palette),
        "unreachable_map": levels[0]["map"],
    }
