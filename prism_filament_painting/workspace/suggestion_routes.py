"""Prism's deterministic spool suggestion API and HTML contact sheet."""

import base64
from io import BytesIO
import csv
import sqlite3
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from prism_filament_painting.palette.inventory import inventory_spools
from prism_filament_painting.palette.materials import choose_material
from prism_filament_painting.palette.store import SOURCE_PERSONAL, filaments_for_ranking, list_filaments, material_spools, measured_filaments
from prism_filament_painting.palette.suggestions import LIBRARY_CAP, MAX_SLOTS, PREVIEW_EDGE, suggest

from .storage import decoded_image, read_upload
from prism_filament_painting.core.forward import RenderConstants



Mode = Literal["next", "loadouts", "full-set"]
MODES = {"next": "Add one spool", "loadouts": "Use my spools", "full-set": "Search all spools"}


class SuggestionForm:
    def __init__(
        self,
        image: UploadFile | None = File(None),
        inventory: UploadFile | None = File(None),
        filament_ids: list[int] = Form(default=[]),
        slots: int = Form(4),
        metric: Literal["mean_delta_e", "p95_delta_e"] = Form("mean_delta_e"),
        limit: int = Form(5),
        material_group: str = Form("auto"),
    ):
        self.image = image
        self.inventory = inventory if inventory and inventory.filename else None
        self.ids = filament_ids
        self.slots = slots
        self.metric = metric
        self.limit = limit
        self.material_group = material_group


def _fields(spool):
    return dict(brand=spool.brand, name=spool.name, hex=spool.hex, td=spool.td,
                opacity_level=getattr(spool, "opacity_level", "matched"),
                color_level=getattr(spool, "color_level", "matched"))


def _preview(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _pools(local, mode, form, root):
    if form.ids and form.inventory:
        raise ValueError("choose database filaments or upload an inventory")
    if mode == "full-set" and (form.ids or form.inventory):
        raise ValueError("full-set searches the whole active database")

    selected = []
    if form.inventory:
        path = root / ("inventory" + Path(form.inventory.filename).suffix.lower())
        path.write_bytes(read_upload(form.inventory))
        selected = inventory_spools(path, owned_only=mode == "loadouts")
        if mode == "loadouts":
            choice = choose_material(selected, form.material_group)
            return choice.values(selected), [], choice

    if mode == "next" and len(form.ids) >= MAX_SLOTS:
        raise ValueError("too many filaments selected")
    if any(not 1 <= value < 2**63 for value in form.ids):
        raise ValueError("invalid filament ID")
    with local.connect() as conn:
        if form.ids:
            if len(form.ids) > LIBRARY_CAP:
                ids = set(form.ids)
                chosen = [row for row in list_filaments(conn) if row.id in ids]
            else:
                chosen = list_filaments(conn, ids=form.ids)
            if len(chosen) != len(form.ids):
                raise ValueError("choose distinct, active filament IDs")
            selected = material_spools(chosen)
            choice = choose_material(selected, form.material_group, loaded=selected if mode == "next" else ())
            chosen_values = filaments_for_ranking(choice.values(selected), conn=conn, provenance=True)
            if mode == "next" and len(chosen_values) != len(chosen):
                raise ValueError("selection contains duplicate filament variants")
            if mode == "loadouts":
                return chosen_values, [], choice
        personal = mode == "loadouts"
        rows = list_filaments(
            conn, source=SOURCE_PERSONAL if personal else "", available_only=personal
        )
        pool = material_spools(rows)
        choice = choose_material(pool, form.material_group, loaded=selected)
        loaded = chosen_values if form.ids else choice.values(selected)
        return filaments_for_ranking(choice.values(pool), conn=conn, provenance=True), loaded, choice


def _response(local, mode, form):
    try:
        with tempfile.TemporaryDirectory(dir=local.directory) as directory:
            root = Path(directory)
            if not form.image:
                raise ValueError("Choose an image.")
            pool, loaded, choice = _pools(local, mode, form, root)
            image = decoded_image(read_upload(form.image))
        result = suggest(
            image,
            pool,
            mode=mode,
            loaded=loaded,
            slots=form.slots,
            metric=form.metric,
            limit=form.limit,
        )
    except (ValueError, csv.Error) as exc:
        raise HTTPException(400, str(exc)) from exc
    except sqlite3.OperationalError as exc:
        raise HTTPException(503, "database is temporarily unavailable") from exc

    candidates = []
    for item in result.candidates:
        preview_error = item.errors[form.metric]
        candidates.append(
            {
                "filaments": [_fields(f) for f in item.filaments],
                "swap_schedule": item.schedule,
                **item.errors,
                **({"prediction_sources": item.prediction_sources, "score_sources": item.score_sources}
                   if item.prediction_sources is not None else {}),
                "score": item.score,
                "preview": _preview(item.preview),
                "addition": _fields(item.addition) if item.addition else None,
                "reduction": result.baseline[form.metric] - preview_error if result.baseline else None,
            }
        )
    exhaustive = result.evaluated == result.possible
    scope = "Exhaustive search" if exhaustive else "This search was not exhaustive"
    unit = "possible additions" if mode == "next" else "possible sets"
    summary = f"{scope}. Scored {result.evaluated} of {result.possible} {unit}."
    response = {
        **choice.metadata(),
        "mode": mode,
        "metric": form.metric,
        "total_layers": RenderConstants().total_layers,
        "baseline": result.baseline,
        **({"baseline_sources": result.baseline_sources} if result.baseline_sources is not None else {}),
        "candidates": candidates,
        "search": {
            "library_size": result.library_size,
            "possible": result.possible,
            "evaluated": result.evaluated,
            "exhaustive": exhaustive,
            "summary": summary,
            "partial_sets_evaluated": result.partial_evaluated,
            "library_limit": LIBRARY_CAP,
            "method": "exhaustive" if exhaustive else "beam_and_swap",
            "preview_max_edge": PREVIEW_EDGE,
            "score_profile": "alpha",
            "search_profile": "alpha",
        },
    }

    if loaded:
        response["loaded_filaments"] = [_fields(f) for f in loaded]
    return response


def suggestion_router(local):
    router = APIRouter()

    @router.post("/palette/suggest/{mode}")
    def palette_suggest(
        mode: Mode, form: Annotated[SuggestionForm, Depends()],
    ):
        """Use repeated filament_ids or a CSV/JSON inventory for loaded or owned spools."""
        return _response(local, mode, form)


    @router.get("/suggestions", response_class=HTMLResponse)
    def suggestion_form(request: Request, mode: Mode = "next"):
        try:
            with local.connect() as conn:
                rows = list_filaments(conn)
                effective = measured_filaments(conn, rows)
        except sqlite3.OperationalError:
            raise HTTPException(503, "Database is temporarily unavailable.") from None
        spools = material_spools(rows)
        return local.templates.TemplateResponse(request, "suggestions.html", {
            "modes": MODES, "mode": mode, "rows": rows, "effective": effective, "library_limit": LIBRARY_CAP,
            "catalogue_report": choose_material(spools).report,
            "owned_report": choose_material([s for s in spools if s.owned]).report,
        })


    @router.post("/suggestions/{mode}", response_class=HTMLResponse)
    def suggestion_sheet(
        request: Request, mode: Mode, form: Annotated[SuggestionForm, Depends()],
    ):
        try:
            result = _response(local, mode, form)
        except HTTPException as exc:
            if request.headers.get("X-Requested-With") == "fetch":
                return PlainTextResponse(str(exc.detail), status_code=exc.status_code)
            raise
        return local.templates.TemplateResponse(
            request, "suggestions.html", {"modes": MODES, "mode": mode, "result": result}
        )

    return router
