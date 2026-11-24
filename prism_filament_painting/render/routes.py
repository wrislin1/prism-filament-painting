"""HTTP adapter for local classic and flat generation."""

import base64
from io import BytesIO
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

from prism_filament_painting.core.finishing import PrintOptions
from prism_filament_painting.core.model import physical_filament
from prism_filament_painting.palette.materials import MaterialSpool, choose_material, material_group
from prism_filament_painting.workspace.storage import MAX_UPLOAD, read_upload, decoded_image
from prism_filament_painting.workspace.project import canonical
from .engine.classic_band_pieces import ClearFilament, band_export_options, write_band_pieces
from .engine.classic_project import write_classic_project
from .engine.generate import generate
from .engine.optimize import is_torch_available
from .engine.params import LayerParams, LuminanceParams, MeshParams, OptimizeParams
from .engine.slicer_settings import read_project_settings
from .flat_generation import generate_flat
from .flat_render import FlatOptions


class ExportRequest(BaseModel):
    state: dict
    mode: Literal["naive", "optimize"] = "naive"
    print_base_factor: int | None = Field(default=None, ge=1, le=4096)
    project_settings: dict | None = None
    include_band_pieces: bool = False
    face_down: bool = False
    clear_layers: int = Field(default=0, ge=0, le=128)
    clear_filament: ClearFilament | None = None
    max_swaps: int | None = Field(default=None, ge=0, le=7)
    pinned_filaments: list[int | None] = Field(default_factory=list, max_length=8)


def filament(row):
    try:
        return physical_filament(row.get("brand", ""), row["name"], row["hex"],
                                 row.get("input_distance_mm", row.get("td")), row.get("spool_id"))
    except KeyError as exc:
        raise ValueError("Each spool needs a name, colour and input distance.") from exc


def guidance_map(data, size):
    if not data:
        return None
    prefix = "data:image/png;base64,"
    if not isinstance(data, str) or not data.startswith(prefix) or len(data) > 2 * MAX_UPLOAD:
        raise ValueError("Choose a source-sized PNG map.")
    with Image.open(BytesIO(base64.b64decode(data[len(prefix):], validate=True)), formats=("PNG",)) as image:
        if image.size != size:
            raise ValueError("Map dimensions must match the source.")
        return image.convert("L")


def export_classic(local, payload):
    state = canonical(payload.state)
    if state.get("project_type") == "flat":
        raise ValueError("Use the flat print downloads.")
    image = local.image(state["image"])
    layer = LayerParams(**state["layer"], td_scale=state["front_lit_scale"])
    fields = state["mesh"].copy()
    orientation = fields.pop("orientation", "auto")
    mesh = MeshParams(**fields, print_options=PrintOptions.parse(state.get("print_options", {})),
                      print_base_factor=payload.print_base_factor)
    lum = state["luminance"]
    strength = float(state.get("height_strength", 0))
    if not 0 <= strength <= 1 or strength and not state.get("depth_map"):
        raise ValueError("Height guidance requires a source-sized map and a strength from zero to one.")
    mode = "optimize" if strength or state.get("focus_mask") else payload.mode
    if mode == "optimize" and not is_torch_available():
        raise ValueError("Install the optimize extra to use CPU optimisation.")
    params = OptimizeParams(device="cpu", allow_cpu=True, depth_map=guidance_map(state.get("depth_map"), image.size),
                            focus_mask=guidance_map(state.get("focus_mask"), image.size), height_strength=strength,
                            max_swaps=payload.max_swaps, pinned_filaments=tuple(payload.pinned_filaments))
    bands = band_export_options(payload.include_band_pieces, mesh, payload.face_down, payload.clear_layers,
                                payload.clear_filament.model_dump_json() if payload.clear_filament else "")
    result = generate(image, [filament(s) for s in state["palette"]["filaments"]], layer_params=layer,
                      luminance_params=LuminanceParams(lum["black_point"], lum["white_point"], lum["gamma"]),
                      mesh_params=mesh, orientation=orientation, mode=mode, optimize_params=params,
                      swap_schedule=state["swap_schedule"])
    run_id = uuid4().hex
    prefix = f"/artifacts/{run_id}"
    with TemporaryDirectory(dir=local.artifacts) as temp:
        stage = Path(temp) / "print"
        stage.mkdir()
        (stage / "0.stl").write_bytes(result.stl_bytes)
        (stage / "0.swaps.txt").write_text(result.swap_text, encoding="utf-8")
        result.preview.save(stage / "0.png")
        projects = write_classic_project(stage, result, project_settings=payload.project_settings)
        pieces = write_band_pieces(stage, result, bands, run_id, url_prefix="/artifacts") if bands else None
        stage.rename(local.artifacts / run_id)
    return {"run_id": run_id, "stl_url": f"{prefix}/0.stl", "output_url": f"{prefix}/0.png",
            "swap_url": f"{prefix}/0.swaps.txt", "swap_schedule": result.swap_schedule,
            "metadata": result.metadata, "bands": pieces,
            **{key: f"{prefix}/{name}" for key, name in projects.items()}}


def render_router(local):
    router = APIRouter()

    @router.post("/api/print-settings")
    def settings(file: UploadFile = File()):
        return read_project_settings(file.file)

    @router.post("/studio/generate")
    def classic(payload: ExportRequest):
        return export_classic(local, payload)

    @router.post("/studio/flat/run")
    def flat(palette: str = Form(), options: str = Form(), source: str = Form("{}"),
             material_group_choice: str = Form("auto", alias="material_group"),
             image: UploadFile | None = File(None), project_settings: UploadFile | None = File(None)):
        options = FlatOptions.model_validate_json(options)
        payload = json.loads(palette)
        spools = payload.get("filaments") if isinstance(payload, dict) else None
        if not isinstance(spools, list) or len(spools) != 4 or not all(isinstance(s, dict) for s in spools):
            raise ValueError("Choose four loaded spools in physical order.")
        entries = [MaterialSpool(s, material_group(s.get("material"), s.get("material_family"))) for s in spools]
        choice = choose_material(entries, material_group_choice, loaded=entries)
        if image:
            pixels = decoded_image(read_upload(image))
            ref = local.add_image(pixels, image.filename)
        else:
            ref = json.loads(source)
            pixels = local.image(ref)
        settings = read_project_settings(project_settings.file) if project_settings else None
        result = generate_flat(pixels, [filament(s) for s in spools], options, local.artifacts,
                               project_settings=settings).model_dump(mode="json")
        return canonical({"version": 1, "project_type": "flat", "image": ref, "palette": {"filaments": spools},
                          **choice.metadata(), "options": result.pop("options"), "flat": result})

    @router.get("/artifacts/{run_id}/{name}")
    def artifact(run_id: str, name: str):
        if not re.fullmatch(r"(?:[a-f0-9]{32}|quilt_[a-f0-9]{16})", run_id) or not re.fullmatch(
            r"[A-Za-z0-9_-]+(?:\.[a-z0-9]+)+", name
        ):
            raise HTTPException(404, "Artifact not found.")
        path = local.artifacts / run_id / name
        if not path.is_file():
            raise HTTPException(404, "Artifact not found.")
        return FileResponse(path, filename=name)
    return router
