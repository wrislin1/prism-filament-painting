"""Export the material volumes of one classic global swap schedule."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from prism_filament_painting.core.color import beer_lambert_blend, hex_to_rgb
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from prism_filament_painting.render.errors import QuiltInputError
from .mesh import (
    QuiltMesh, _XY_UNITS, _relief_units, _saddle_clips, _stepped_vertices_and_faces,
    _xy_scale, mesh_triangle_count_from_clips,
)
from .params import GenerateResult, MeshParams


class ClearFilament(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", allow_inf_nan=False)
    name: str = Field(min_length=1)
    hex: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    td: float = Field(gt=0)
    td_measured: bool = False


@dataclass(frozen=True)
class BandExport:
    face_down: bool
    clear_layers: int
    clear: ClearFilament | None


def band_export_options(enabled: bool, mesh: MeshParams, face_down: bool, clear_layers: int,
                        clear_filament: str) -> BandExport | None:
    if not enabled:
        if face_down or clear_layers or clear_filament:
            raise QuiltInputError("band options require include_classic_band_pieces")
        return None
    if face_down and not clear_layers:
        raise QuiltInputError("band_face_down requires a clear cap to form a flat viewing face")
    if bool(clear_layers) != bool(clear_filament) or clear_layers < 0:
        raise QuiltInputError("a clear cap requires a positive clear_layers count and a clear_filament")
    try:
        clear = ClearFilament.model_validate_json(clear_filament) if clear_filament else None
    except ValidationError as exc:
        raise QuiltInputError("clear_filament requires a name, hex colour and positive finite TD in mm") from exc
    if clear and (mesh.print_options.radius or mesh.print_options.diameter):
        raise QuiltInputError("A clear cap covers the entire rectangle. Omit the cap to use rounded corners or a hanging hole.")
    return BandExport(face_down, clear_layers, clear)


def _cap_preview(result: GenerateResult, clear: ClearFilament, ceiling: int, path: Path) -> dict:
    params = result.layer_params
    heights, inverse = np.unique(result.height_field, return_inverse=True)
    classic = result.forward_table.layer_rgb
    td_mm = clear.td * params.td_scale
    if td_mm <= 0:
        raise QuiltInputError("effective clear filament TD must be positive")
    over = hex_to_rgb(clear.hex)
    tinted = [
        beer_lambert_blend(over, classic[h], (ceiling - int(h)) * params.layer_height_mm, td_mm)
        for h in heights
    ]
    rgb = np.asarray(tinted, dtype=np.uint8)[inverse.reshape(result.height_field.shape)]
    Image.fromarray(rgb).save(path)
    # Mean absolute channel change per pixel in 8-bit sRGB, then summarize across pixels.
    difference = np.abs(rgb.astype(np.int16) - np.asarray(result.preview)).mean(axis=2)
    return {
        "filament": clear.model_dump(), "td_scale": params.td_scale,
        "prediction": "Model prediction with caller-reported measured TD" if clear.td_measured else
                      "Estimated preview using supplied unmeasured TD",
        "mean_rgb_mae": float(difference.mean()), "worst_rgb_mae": float(difference.max()),
    }


def _piece_lattice(thickness: np.ndarray, clips: np.ndarray, relief: int, *,
                   fill_depth: int, max_triangles: int) -> tuple[np.ndarray, np.ndarray]:
    # A vertical wall needs vertices only at actual height transitions. Using
    # their sorted ranks keeps even a very thick cap from expanding per layer.
    levels = np.union1d(thickness, [0, fill_depth] if fill_depth else [0])
    ranks = np.searchsorted(levels, thickness)
    fill = len(levels) - 1 if fill_depth else 0
    if fill_depth:
        count = mesh_triangle_count_from_clips(ranks, clips, relief)
        # Each clipped corner gains two horizontal triangles. Its diagonal wall
        # now runs from the cell roof to the filled centre instead of from zero.
        count += int((clips.sum(axis=2) * (2 + 2 * (fill - 2 * ranks))).sum())
        if count > max_triangles:
            raise QuiltInputError("clear cap exceeds max_triangles")
    xyz, faces = _stepped_vertices_and_faces(ranks, relief_units=relief, clips=clips, fill_clips_to=fill)
    xyz[:, 2] = levels[xyz[:, 2]]
    return xyz, faces


def write_band_pieces(directory: Path, result: GenerateResult, options: BandExport, run_id: str,
                      *, url_prefix="/outputs") -> dict:
    field = result.height_field
    clips = _saddle_clips(field)  # Keep the classic saddle footprint at every cut.
    rows, columns = field.shape
    pitch = _xy_scale(columns, rows, result.mesh_params)
    relief = _relief_units(pitch)
    peak = int(field.max())
    ceiling = peak + options.clear_layers
    height = result.layer_params.layer_height_mm
    prefix = f"{url_prefix}/{run_id}"
    pieces = []

    def emit(name: str, thickness: np.ndarray | None, base: int, *, cap: bool = False) -> None:
        entry = {"kind": "clear_cap" if cap else "colour_band", "filament": name, "stl_url": None}
        pieces.append(entry)
        if thickness is None:  # A scheduled colour can be entirely above the actual surface.
            return
        xyz, faces = _piece_lattice(
            thickness, clips, relief, fill_depth=ceiling if cap else 0, max_triangles=result.mesh_params.max_triangles,
        )
        vertices = xyz.astype(float)
        vertices[:, :2] = vertices[:, :2] / _XY_UNITS * pitch
        depth = xyz[:, 2]
        if cap:
            z = base - depth  # Built from the flat face towards the coloured surface.
            faces = faces[:, ::-1]
        else:
            z = base + depth
        entry.update(bottom_mm=float(z.min()) * height, top_mm=float(z.max()) * height)
        if options.face_down:
            vertices[:, 1] = rows * pitch - vertices[:, 1]
            z = ceiling - z
        vertices[:, 2] = z * height
        stored = vertices.astype(np.float32)
        if not np.isfinite(stored).all() or len(np.unique(stored, axis=0)) != len(stored):
            raise QuiltInputError("piece dimensions exceed STL coordinate precision")
        mesh = QuiltMesh(stored, faces)
        mesh.validate(full=True)
        filename = f"{len(pieces)}.stl"
        (directory / filename).write_bytes(mesh.to_stl_bytes())
        entry["stl_url"] = f"{prefix}/{filename}"

    cuts = (0, *(layer - 1 for layer in result.swap_schedule), result.layer_params.total_layers)
    for i, filament in enumerate(result.output_filaments):
        low, high = cuts[i:i + 2]
        thickness = np.clip(field - low, 0, high - low) if low < peak else None
        emit(f"{filament.brand} {filament.name}", thickness, low)

    report = {"face_down": options.face_down, "y_max_mm": rows * pitch, "top_mm": ceiling * height,
              "clear_layers": options.clear_layers, "pieces": pieces}
    if options.clear is not None:
        emit(options.clear.name, ceiling - field, ceiling, cap=True)
        report["cap_preview"] = {
            **_cap_preview(result, options.clear, ceiling, directory / "1.png"),
            "url": f"{prefix}/1.png",
        }
        report["print_note"] = (
            "Print the clear piece alone with its flat viewing face on the bed. Its stepped underside "
            "needs supports when printed alone face-up over uneven relief. For a combined job, keep all "
            "parts aligned. For separate colours, place each flat band base on the bed."
        )
    return report
