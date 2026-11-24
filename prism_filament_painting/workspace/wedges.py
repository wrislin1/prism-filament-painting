"""Generate an identified wedge, then retain the raw observations on that print."""

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from statistics import median
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from prism_filament_painting.palette.store import get_filament, list_filaments
from prism_filament_painting.core.opacity import LAYER_MM, PATCH_COUNT, measure_crossing, threshold_divisor
from pydantic import BaseModel, Field, FiniteFloat, PositiveFloat, StrictInt, StringConstraints, ValidationError

from .wedge_mesh import DEPTH, RAIL, WIDTH, backing_plan, wedge_archive

Lot = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Hex = Annotated[str, StringConstraints(pattern=r"^#?[a-fA-F0-9]{6}$")]


class Scan(BaseModel):
    td_mm: Annotated[PositiveFloat, Field(allow_inf_nan=False)]
    hex: Hex


Scans = Annotated[list[Scan], Field(min_length=5, max_length=5)]


class Spool(BaseModel):
    filament_id: int
    spool_id: Lot
    scans: Scans | None = None


class Pair(BaseModel):
    top: Spool
    backing: Spool


class Reading(BaseModel):
    passes: Annotated[list[StrictInt | None], Field(min_length=3, max_length=3)]
    top_scans: Scans | None = None
    measured_heights_mm: dict[int, FiniteFloat] = Field(default_factory=dict, max_length=30)
    conditions: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    # A failed physical control is retained as a limit and excluded from ranking.
    controls_passed: bool


def _median_hex(scans: list[Scan]) -> str:
    td = median(s.td_mm for s in scans)
    return "#" + next(s.hex for s in scans if s.td_mm == td).lstrip("#").upper()


def _spool(conn: sqlite3.Connection, choice: Spool) -> dict:
    filament = get_filament(conn, choice.filament_id)
    if not filament.active:
        raise ValueError("Choose an active filament.")
    return {
        **choice.model_dump(),
        "name": f"{filament.brand} {filament.name} {filament.material}",
        "library_td_mm": filament.td,
        "library_hex": filament.hex,
        "opacity_level": filament.opacity_level,
        "library_color_level": filament.color_level,
        "color_level": "measured" if choice.scans else filament.color_level,
        "hex": _median_hex(choice.scans) if choice.scans else filament.hex.upper(),
    }


def _record(conn: sqlite3.Connection, wedge_id: str) -> dict:
    row = conn.execute("SELECT * FROM opacity_wedges WHERE id=?", (wedge_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Wedge not found.")
    measurement = json.loads(row["observation"]) if row["observation"] else None
    if measurement is not None:
        # The existing HTTP client expects the original wire field names.
        for new, old in (("hiding_distance_interval_mm", "hiding_mm"),
                         ("hiding_distance_mm", "ranking_hiding_mm")):
            if new in measurement:
                measurement[old] = measurement.pop(new)
    return {
        **json.loads(row["spec"]),
        "measured_at": row["measured_at"],
        "measurement": measurement,
    }


def _form_spool(filament_id: int, lot: str, scans: str) -> Spool:
    pairs = [re.split(r"[\s,]+", line.strip()) for line in scans.strip().splitlines()]
    if any(len(pair) != 2 for pair in pairs):
        raise HTTPException(422, "Enter five TD and hex pairs, one per line.")
    try:
        return Spool(
            filament_id=filament_id,
            spool_id=lot,
            scans=[Scan(td_mm=td, hex=color) for td, color in pairs] or None,
        )
    except ValidationError as exc:
        raise HTTPException(422, "Check the spool label and optional five TD and hex pairs.") from exc


def wedge_router(connect, templates, check_auth=None):
    router = APIRouter()

    @contextmanager
    def _connection():
        try:
            with connect() as conn, conn:
                yield conn
        except sqlite3.OperationalError as exc:
            raise HTTPException(503, "Filament database temporarily unavailable.") from exc


    @router.get("/opacity-wedges", response_class=HTMLResponse)
    def page(request: Request):
        with _connection() as conn:
            filaments = list_filaments(conn)
            history = [
                json.loads(row[0])
                for row in conn.execute("SELECT spec FROM opacity_wedges ORDER BY rowid DESC LIMIT 100")
            ]
        return templates.TemplateResponse(
            request, "opacity_wedges.html", {"filaments": filaments, "history": history}
        )


    @router.post("/opacity-wedges", response_class=RedirectResponse, status_code=303)
    def generate_from_form(
        top: Annotated[int, Form()],
        backing: Annotated[int, Form()],
        top_lot: Annotated[str, Form()],
        backing_lot: Annotated[str, Form()],
        top_scans: Annotated[str, Form()] = "",
        backing_scans: Annotated[str, Form()] = "",
        authorization: str | None = Header(None),
    ):
        pair = Pair(
            top=_form_spool(top, top_lot, top_scans),
            backing=_form_spool(backing, backing_lot, backing_scans),
        )
        wedge = generate(pair, authorization)
        return RedirectResponse(f"/api/opacity-wedges/{wedge['id']}/parts.zip", status_code=303)


    @router.post("/api/opacity-wedges", status_code=201)
    def generate(pair: Pair, authorization: str | None = Header(None)):
        if check_auth is not None:
            check_auth(authorization)
        with _connection() as conn:
            try:
                top = _spool(conn, pair.top)
                backing = _spool(conn, pair.backing)
                if (
                    top["filament_id"] == backing["filament_id"]
                    and top["spool_id"] == backing["spool_id"]
                ):
                    raise ValueError("Choose distinct spools.")
                divisor = threshold_divisor(top["hex"], backing["hex"])
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            wedge_id = uuid4().hex
            scans = backing["scans"]
            td = median(s["td_mm"] for s in scans) if scans else backing["library_td_mm"]
            backing_mm, capped = backing_plan(td)
            spec = dict(
                spec_version=2,
                backing_design_td_mm=td,
                backing_target_capped=capped,
                id=wedge_id,
                page_url=f"/opacity-wedges?wedge={wedge_id}",
                top=top,
                backing=backing,
                divisor=divisor,
                backing_fraction=10**-divisor,
                layer_mm=LAYER_MM,
                top_patches=PATCH_COUNT,
                bare_patch=0,
                swap_z_mm=backing_mm,
                rail_mm=RAIL,
                size_mm=[WIDTH, DEPTH, round(backing_mm + RAIL, 2)],
            )
            conn.execute(
                "INSERT INTO opacity_wedges(id,top_id,spool_id,spec) VALUES(?,?,?,?)",
                (wedge_id, top["filament_id"], top["spool_id"], json.dumps(spec)),
            )
            return spec


    @router.get("/api/opacity-wedges/{wedge_id}")
    def detail(wedge_id: str):
        with _connection() as conn:
            return _record(conn, wedge_id)


    @router.get("/api/opacity-wedges/{wedge_id}/parts.zip")
    def parts(wedge_id: str):
        with _connection() as conn:
            spec = _record(conn, wedge_id)
        return Response(
            wedge_archive(spec),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="opacity-wedge-{spec["id"]}.zip"'},
        )


    @router.put("/api/opacity-wedges/{wedge_id}/reading")
    def record_reading(wedge_id: str, reading: Reading, authorization: str | None = Header(None)):
        if check_auth is not None:
            check_auth(authorization)
        raw = reading.model_dump(mode="json")
        with _connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            spec = _record(conn, wedge_id)
            if spec["measurement"]:
                if raw == spec["measurement"]["raw"]:
                    return spec
                raise HTTPException(409, "This wedge already has a saved reading.")
            top = spec["top"]
            if reading.top_scans and top["scans"] and raw["top_scans"] != top["scans"]:
                raise HTTPException(422, "Use the five scans retained when this wedge was generated.")
            if reading.top_scans and _median_hex(reading.top_scans) != top["hex"]:
                raise HTTPException(
                    422,
                    "The median-scan colour differs. Generate a wedge with these scans and read it.",
                )
            scans = raw["top_scans"] or top["scans"] or []
            try:
                result = measure_crossing(
                    spec["divisor"],
                    reading.passes,
                    [s["td_mm"] for s in scans],
                    reading.measured_heights_mm,
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            lo, hi = result["hiding_distance_interval_mm"]
            # A two-sided envelope spanning at most two patch steps can inform ranking.
            passes = reading.passes
            resolved = (
                reading.controls_passed and lo > 0 and hi is not None and max(passes) - min(passes) <= 1
            )
            estimate = (lo + hi) / 2 if resolved else None
            observation = {
                "method": "opacity wedge",
                "raw": raw,
                **result,
                "hiding_distance_mm": estimate,
            }
            stamp = datetime.now(UTC).isoformat()
            conn.execute(
                "UPDATE opacity_wedges SET observation=?, measured_at=?, hiding_distance_mm=? WHERE id=?",
                (json.dumps(observation), stamp, estimate, wedge_id),
            )
            conn.execute(
                """UPDATE filaments SET meter_distance_mm=CASE WHEN ? IS NOT NULL THEN ?
                            WHEN spool_id IS NULL OR spool_id=? THEN meter_distance_mm ELSE NULL END,
                            spool_id=?, active_lot=NULL, measured_at=?, updated_at=? WHERE id=?""",
                (
                    result["median_td_mm"],
                    result["median_td_mm"],
                    top["spool_id"],
                    top["spool_id"],
                    stamp,
                    stamp,
                    top["filament_id"],
                ),
            )
            return _record(conn, wedge_id)

    return router
