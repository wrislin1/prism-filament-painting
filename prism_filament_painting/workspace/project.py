"""Validate public Studio documents before saving, reporting or rendering."""

import re

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal

from prism_filament_painting.core.finishing import PrintOptions
from prism_filament_painting.core.forward import is_valid_schedule
from prism_filament_painting.palette.project_migration import adapt_project
from .studio_render import validate_snapshot


class NumericControls(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class Layers(NumericControls):
    layer_height_mm: float = Field(gt=0, le=1)
    base_layers: int = Field(strict=True, ge=1)
    total_layers: int = Field(strict=True, ge=12, le=128)


class Mesh(NumericControls):
    max_dim_mm: float = Field(gt=0, le=2000)
    xy_pixel_mm: float = Field(gt=0, le=10)
    max_triangles: int = Field(strict=True, ge=12, le=1500000)
    flatten_transparent: bool
    orientation: Literal["auto", "bright", "dark"] = "auto"


class Luminance(NumericControls):
    black_point: float = Field(ge=0, lt=1)
    white_point: float = Field(gt=0, le=1)
    gamma: float = Field(ge=0.1, le=5)

    @model_validator(mode="after")
    def ordered(self):
        if self.black_point >= self.white_point:
            raise ValueError("Black point must be below white point.")
        return self


def validate_controls(state):
    if state.get("project_type") == "flat":
        validate_snapshot(state, re.compile(r"quilt_[a-f0-9]{16}").fullmatch)
        return
    layer = Layers.model_validate(state["layer"])
    Mesh.model_validate(state["mesh"])
    Luminance.model_validate(state["luminance"])
    PrintOptions.parse(state.get("print_options", {}))
    count = len(state["palette"]["filaments"])
    if not (1 <= count <= 8 and layer.base_layers < layer.total_layers and is_valid_schedule(
        state["swap_schedule"], expected=count - 1, base_layers=layer.base_layers, total_layers=layer.total_layers
    )):
        raise ValueError("Choose an increasing swap schedule inside the printable layers.")


def canonical(state):
    try:
        return adapt_project(state, validate_v1=validate_controls, saved_file=False)
    except (TypeError, KeyError, AttributeError, IndexError) as exc:
        raise ValueError("Invalid project document.") from exc
