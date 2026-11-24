"""Local spool library, palette ranking and Studio project routes."""

import csv
from importlib.metadata import distribution
from io import BytesIO, StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from prism_filament_painting.core.constraints import studio_constraints
from prism_filament_painting.core.forward import RenderConstants
from prism_filament_painting.core.prediction import calibration_catalog
from prism_filament_painting.palette import rank_palettes, store
from prism_filament_painting.palette.catalogues import SOURCES
from prism_filament_painting.palette.enhance import EnhanceParams, apply_enhancement
from prism_filament_painting.palette.inventory import inventory_rows
from prism_filament_painting.palette.materials import choose_material
from prism_filament_painting.palette.sample_inputs import sample_image, spool_list
from prism_filament_painting.palette.meter import save_readings
from prism_filament_painting.render.engine.optimize import is_torch_available
from .storage import checked_id, classic_state, decoded_image, read_upload, write_json
from .project import canonical
from .studio_presets import resolved_presets
from .studio_render import flat_accuracy, print_grid
from .studio_report import predicted_accuracy, studio_stack
from .suggestion_routes import suggestion_router
from .wedges import wedge_router


CATALOGUE_NAMES = {"openprinttag": "OpenPrintTag material database", "ofd": "Open Filament Database"}


def catalogue_notice(source):
    repo, revision = SOURCES[source]
    filename = repo.split("/")[-1] + "-MIT.txt"
    text = distribution("prism-filament-painting").read_text("licenses/licenses/catalogues/" + filename)
    if text is None:
        raise RuntimeError("Missing installed catalogue notice")
    return {"name": CATALOGUE_NAMES[source], "revision": revision, "license": "MIT",
            "scope": "Data from this catalogue at the specified revision.",
            "url": f"https://github.com/{repo}/blob/{revision}/LICENSE", "text": text}


class SpoolInput(BaseModel):
    brand: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=150)
    material: str = Field(min_length=1, max_length=100)
    hex: str = Field(pattern=r"^#[a-fA-F0-9]{6}$")
    td: float = Field(gt=0, allow_inf_nan=False)


def inventory_view(local):
    with local.connect() as conn:
        rows = store.list_filaments(conn, available_only=True)
        effective = store.measured_filaments(conn, rows)
    entries = store.material_spools(rows)
    spools = []
    for row in rows:
        spool = effective.get(row.id, row)
        spools.append({"id": row.id, "brand": spool.brand, "name": spool.name, "hex": spool.hex, "td": spool.td,
                       "material": row.material, "material_family": row.material_family,
                       "spool_id": row.active_lot or row.spool_id})
    return spools, entries, choose_material(entries)


def workspace_router(local):
    router = APIRouter()
    router.include_router(wedge_router(local.connect, local.templates))
    router.include_router(suggestion_router(local))
    page = local.templates.TemplateResponse

    @router.get("/")
    def home(request: Request):
        with local.connect() as conn:
            pool, choice = store.ranking_pool(conn)
            empty = not pool and not store.list_filaments(conn)
        return page(request, "home.html", {"empty": empty, "slots": min(8, len(pool)), "group": choice.group})

    @router.post("/sample")
    def start_sample():
        with local.connect() as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            if store.list_filaments(conn):
                return RedirectResponse("/", status_code=303)
            store.import_spool_list(conn, csv.DictReader(StringIO(spool_list())))
        return RedirectResponse("/?picture=sample", status_code=303)

    @router.get("/sample/picture.png")
    def example_picture():
        output = BytesIO()
        sample_image().save(output, format="PNG")
        return Response(output.getvalue(), media_type="image/png")

    @router.get("/filaments")
    def library(request: Request):
        return page(request, "library.html", {"catalogues": CATALOGUE_NAMES})

    @router.get("/filaments/licenses/{source}", response_class=PlainTextResponse)
    def catalogue_license(source: str):
        if source not in CATALOGUE_NAMES:
            raise HTTPException(404, "Unknown catalogue.")
        return catalogue_notice(source)["text"]

    @router.get("/filaments/export")
    def export_spools(response: Response):
        with local.connect() as conn:
            rows = store.list_filaments(conn)
        # Generic spool lists may omit the external IDs. Carry both scoped notices.
        notices = {key: catalogue_notice(key) for key in CATALOGUE_NAMES}
        response.headers["Content-Disposition"] = 'attachment; filename="prism-spools.json"'
        return {"filaments": rows, "catalogue_notices": notices}

    @router.get("/api/filaments")
    def list_spools():
        with local.connect() as conn:
            return store.list_filaments(conn)

    @router.post("/api/filaments", status_code=201)
    def add_spool(data: SpoolInput):
        with local.connect() as conn:
            return store.create_filament(conn, source=store.SOURCE_PERSONAL, **data.model_dump())

    @router.post("/api/filaments/import")
    def import_spools(file: UploadFile = File()):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".csv", ".tsv", ".json"}:
            raise ValueError("Choose a CSV, TSV or JSON spool list.")
        with TemporaryDirectory(dir=local.directory) as tmp, local.connect() as conn:
            path = Path(tmp) / ("spools" + suffix)
            path.write_bytes(read_upload(file))
            return store.import_spool_list(conn, inventory_rows(path), only_owned=False)

    @router.post("/api/filaments/readings")
    def readings(file: UploadFile = File()):
        with local.connect() as conn, conn:
            return {"saved": save_readings(conn, read_upload(file).decode("utf-8-sig"))}

    @router.post("/api/rank")
    def rank(image: UploadFile = File(), slots: int = Form(2, ge=1, le=8),
             material_group: str = Form("auto"), local_contrast: float = Form(0, ge=0, le=1)):
        pixels = decoded_image(read_upload(image))
        with local.connect() as conn:
            spools, choice = store.ranking_pool(conn, material_group, provenance=True)
        if slots > len(spools):
            raise ValueError(f"Choose at most {len(spools)} available spools from this material.")
        if local_contrast:
            pixels = apply_enhancement(pixels, EnhanceParams(local_contrast=local_contrast))
        candidates = rank_palettes(pixels, spools, palette_size=slots, max_candidates=5)
        ref = local.add_image(pixels, image.filename)
        states = [classic_state(ref, candidate, choice) for candidate in candidates]
        key = ref["upload_token"]
        write_json(local.images / (key + ".json"), states)
        return {"candidates": [{"state": state, "studio_url": f"/studio?selection={key}&index={i}"}
                               for i, state in enumerate(states)]}

    @router.get("/studio")
    @router.get("/studio/flat")
    def editor(request: Request, project_id: str | None = None, selection: str | None = None,
               index: int = Query(0, ge=0, lt=5)):
        state = None
        missing = False
        try:
            if project_id:
                state = local.project(project_id)
            elif selection:
                path = local.images / (checked_id(selection) + ".json")
                try:
                    state = json.loads(path.read_bytes())[index]
                except (FileNotFoundError, IndexError):
                    raise HTTPException(404, "Ranked palette not found.") from None
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            missing = True
        spools, entries, choice = inventory_view(local)
        flat = state.get("project_type") == "flat" if state else request.url.path == "/studio/flat"
        defaults = RenderConstants()
        layers = {"min": 12, "max": 128, "default": defaults.total_layers, "base_layers": defaults.base_layers,
                  "layer_height_mm": defaults.layer_height_mm, "td_scale": defaults.front_lit_scale}
        boot = {"state": state, "inventory": spools, "constraints": studio_constraints(), "layer_constraints": layers,
                "spool_materials": {s.value.id: s.group for s in entries}, "default_material": choice.group,
                "calibration_enabled": bool(calibration_catalog())}
        return page(request, "studio_flat.html" if flat else "studio.html", {
            "boot_json": json.dumps(boot, allow_nan=False).replace("<", "\\u003c"), "choice": choice,
            "missing": missing, "inventory_count": len(spools), "local_material": choice, "has_state": bool(state), "calibration_enabled": boot["calibration_enabled"],
            "layer_constraints": layers, "optimizer_enabled": is_torch_available()}, status_code=404 if missing else 200)

    @router.get("/studio/image/{token}")
    def image(token: str):
        return FileResponse(local.image_path({"upload_token": token}), media_type="image/png")

    @router.get("/studio/projects/{key}/image")
    def project_image(key: str):
        return FileResponse(local.image_path({"project_id": key}), media_type="image/png")

    @router.get("/studio/projects")
    def projects():
        return {"projects": [local.summary(json.loads(path.read_bytes())) for path in sorted(local.projects.glob("*.json"))]}

    @router.post("/studio/projects")
    def save(payload: dict):
        return local.save(payload.get("state"))

    @router.get("/studio/projects/{key}")
    def reopen(key: str):
        return {"state": local.project(key)}

    @router.post("/studio/projects/{key}")
    def update(key: str, payload: dict):
        return local.save(payload.get("state"), key)

    @router.delete("/studio/projects/{key}")
    def delete(key: str):
        try:
            (local.projects / (checked_id(key) + ".json")).unlink()
        except FileNotFoundError:
            raise HTTPException(404, "Project not found.") from None
        return {"ok": True}

    @router.get("/studio/presets")
    def presets(material_group: str = "auto"):
        with local.connect() as conn:
            return resolved_presets(conn, material_group)

    @router.post("/studio/prediction-table")
    def predictions(payload: dict):
        stack = studio_stack(canonical(payload))
        return {"layer_rgb": stack.layer_rgb.tolist(), "layer_sources": stack.layer_sources}

    @router.post("/studio/print-grid")
    def grid(payload: dict, base_factor: int = Query(1, ge=1, le=4096)):
        state = canonical(payload)
        return print_grid(local.image(state["image"]), state, base_factor)

    @router.post("/studio/report")
    def report(payload: dict, downsample_factor: int = Query(1, ge=1, le=4096)):
        state = canonical(payload)
        pixels = local.image(state["image"])
        if state.get("project_type") == "flat":
            return flat_accuracy(pixels, state["flat"]["run"])
        return predicted_accuracy(pixels, state, downsample_factor)

    return router
