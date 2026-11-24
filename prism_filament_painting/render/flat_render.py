"""Flat print controls and CPU generation from a supplied image and four slots."""

from typing import Literal

from PIL import ImageOps
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from .flat_print import make_run
from .flat_recipes import Setup
from .flat_regions import Region
from .slice_report import SliceReport


class A1Loadout(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    printer_model: Literal["Bambu Lab A1"]
    filament_colour: list[str] = Field(min_length=4, max_length=4)
    filament_density: list[float] = Field(min_length=4, max_length=4)
    flush_multiplier: list[float] = Field(min_length=1, max_length=1)
    flush_volumes_matrix: list[float] = Field(min_length=16, max_length=16)


class FlatOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    profile: Literal["u1", "a1"]
    long_edge_cells: int = Field(default=100, ge=1, le=100)
    cell_mm: float = Field(default=1, ge=1, le=2)
    plan: str = "full"
    regions: list[Region] = Field(default_factory=list, max_length=8)
    background_slots: list[StrictInt] | None = Field(default=None, min_length=1, max_length=4)
    dither_scope: StrictInt | None = Field(default=None, ge=-1, le=8)
    slicer_settings: A1Loadout | None = None
    slice_report: SliceReport | None = None


def generate_flat_run(image, filaments, options: FlatOptions) -> dict:
    """Match the oriented source to printable recipes for the selected loadout."""
    if len(filaments) != 4:
        raise ValueError("select four loaded slots in physical order")
    image = ImageOps.exif_transpose(image)
    setup = Setup(slots=tuple(filaments), profile=options.profile, cell_mm=options.cell_mm)
    grid = tuple(max(1, round(n * options.long_edge_cells / max(image.size))) for n in image.size)
    costs = None
    if options.slicer_settings is not None:
        if options.profile != "a1":
            raise ValueError("flush settings apply only to the A1")
        matrix = options.slicer_settings.flush_volumes_matrix
        multiplier, = options.slicer_settings.flush_multiplier
        costs = [[v * multiplier for v in matrix[i:i + 4]] for i in range(0, 16, 4)]
    return make_run(
        image, setup, grid=grid, plan=options.plan, costs=costs, regions=options.regions,
        background_slots=options.background_slots, dither_scope=options.dither_scope,
    )
