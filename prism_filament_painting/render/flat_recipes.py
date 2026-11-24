"""Bounded recipe enumeration and the calibration boundary for flat prints."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from math import isfinite

from prism_filament_painting.core.color import hex_to_rgb
from prism_filament_painting.core.forward import RenderConstants, formula_colors_by_layer
from prism_filament_painting.core.prediction import BoardCalibration, predict_stack, select_calibration
from prism_filament_painting.core.model import Filament

RGB = tuple[int, int, int]
MAX_RECIPES = 256
MAX_DEPTH = 4
MODEL_VERSION = "palette-core-contiguous-bands-v1"


@dataclass(frozen=True)
class Setup:
    """Physical slots are numbered from 1. Picture layers run bottom to top."""

    slots: tuple[Filament, ...]
    backing_slot: int = 1
    depth: int = 4
    layer_height_mm: float = 0.12
    cell_mm: float = 1.0
    profile: str = "u1"
    td_scale: float = RenderConstants().td_scale
    viewing: str = "front-lit over white"
    model_version: str = field(default=MODEL_VERSION, init=False)
    backing_heights_mm: tuple[float, ...] = field(default=(0.20, 0.12), init=False)

    def __post_init__(self):
        n = len(self.slots)
        if not 1 <= n <= 4:
            raise ValueError("load between one and four spools")
        if type(self.depth) is not int or not 1 <= self.depth <= MAX_DEPTH:
            raise ValueError(f"reduce picture depth to 1..{MAX_DEPTH} (at most {MAX_RECIPES} recipes)")
        if n ** self.depth > MAX_RECIPES:
            raise ValueError(f"reduce slots or depth to fit {MAX_RECIPES} recipes")
        if type(self.backing_slot) is not int or not 1 <= self.backing_slot <= n:
            raise ValueError("backing slot must be loaded")
        if self.profile not in ("u1", "a1"):
            raise ValueError("profile must be u1 or a1")
        for value in (self.cell_mm, self.layer_height_mm, self.td_scale):
            if not isfinite(value) or value <= 0:
                raise ValueError("cell pitch, picture layer height and TD scale must be positive and finite")
        for spool in self.slots:
            if not all(isinstance(s, str) and s.strip() for s in (spool.brand, spool.name, spool.hex)):
                raise ValueError("each loaded spool needs a brand, name and hex colour")
            hex_to_rgb(spool.hex)
            if not isfinite(spool.td) or spool.td <= 0:
                raise ValueError("each loaded spool needs positive finite TD in mm")


Lookup = Callable[[Setup, tuple[int, ...]], RGB | None]


@dataclass(frozen=True)
class Calibration:
    """Providers return None on a miss. Change version when any measurement changes."""

    version: str = "uncalibrated"
    measured: Lookup | None = None
    interpolated: Lookup | None = None
    fitted: Lookup | None = None


@dataclass(frozen=True)
class Recipe:
    slots: tuple[int, ...]
    rgb: RGB
    source: str


def _predict(setup: Setup, slots: tuple[int, ...], calibration: Calibration | BoardCalibration) -> Recipe:
    # The shared compositor starts at the backing colour. Picture layer 1 is
    # the first deposition over that surface. Collapse only adjacent repeats.
    last = setup.backing_slot
    bands = [setup.slots[last - 1]]
    starts = []
    for layer, slot in enumerate(slots, start=1):
        if slot != last:
            bands.append(setup.slots[slot - 1])
            starts.append(layer)
            last = slot
    rgb = formula_colors_by_layer(
        bands, starts, layer_height_mm=setup.layer_height_mm, total_layers=setup.depth, front_lit_scale=setup.td_scale,
    )[-1]
    if isinstance(calibration, BoardCalibration):
        ends = [*starts[1:], setup.depth + 1]
        recipe = (tuple((spool.spool_id, end - start) for spool, start, end in zip(bands[1:], starts, ends)),)
    else:
        recipe = (setup, slots)
    predicted = predict_stack(rgb, calibration, *recipe)
    return Recipe(slots, predicted.rgb, predicted.source)


def setup_calibration(setup):
    if setup.viewing != "front-lit over white":
        return Calibration()
    book = select_calibration(setup.slots, setup.slots[setup.backing_slot - 1], setup.layer_height_mm)
    if book is not None and book.supports_backing(sum(setup.backing_heights_mm)):
        return book
    return Calibration()


def recipe_table(setup: Setup, calibration=None) -> tuple[Recipe, ...]:
    return _recipe_table(setup, calibration or setup_calibration(setup))


@lru_cache(maxsize=8)
def _recipe_table(setup, calibration):
    """Cache immutable tables, including optically tied but physically distinct recipes."""
    return tuple(_predict(setup, slots, calibration)
                 for slots in product(range(1, len(setup.slots) + 1), repeat=setup.depth))
