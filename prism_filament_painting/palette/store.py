"""Persistent local filament store for Prism."""

from __future__ import annotations

import math
import re
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prism_filament_painting.core.prediction import calibration_catalog
from prism_filament_painting.core.forward import RenderConstants

from .dedup import MEASURED_METHOD, measured_precedence, strip_measured_marker
from .inventory import _KEYS, _filament_from_row, _normalize_key, owned_is_true as _owned_is_true
from .library_migration import _upgrade
from .meter import active_readings
from .library_schema import DB_USER_VERSION
from .materials import MaterialSpool, choose_material, material_group
from .model import Filament
from .provenance import SourcedFilament, sourced_filament, level

SOURCE_PERSONAL = "personal"
SOURCE_PUBLIC = "public_curated"
_VALID_SOURCES = {SOURCE_PERSONAL, SOURCE_PUBLIC}

# Token-aware material→family map. Keys are the first material token (upper-cased, with a
# trailing '+'/whitespace/'-' stripped) so PLA+/PLA Matte/PLA Silk collapse to PLA, while
# look-alikes that share a prefix stay distinct: PCTG is NOT PC, COPE/CPE are their own
# families, PVB is not PETG. Unknown materials pass through unchanged (display preserved).
_MATERIAL_FAMILIES = {
    "PLA": "PLA",
    "PETG": "PETG",
    "PCTG": "PCTG",
    "ABS": "ABS",
    "ASA": "ASA",
    "TPU": "TPU",
    "TPE": "TPE",
    "PC": "PC",
    "PVA": "PVA",
    "PVB": "PVB",
    "HIPS": "HIPS",
    "CPE": "CPE",
    "COPE": "COPE",
    "PA": "PA",
    "NYLON": "PA",
}


@dataclass(frozen=True)
class FilamentRecord:
    id: int
    source: str
    brand: str
    name: str
    material: str
    material_family: str
    hex: str
    td: float
    td1: float | None
    notes: str
    method: str
    spool_id: str | None
    active_lot: str | None
    measured_lab_l: float | None
    measured_lab_a: float | None
    measured_lab_b: float | None
    td_confidence: float | None
    measured_at: str | None
    external_id: str | None
    external_slug: str | None
    source_url: str | None
    available: bool
    active: bool
    created_at: str
    updated_at: str
    deleted_at: str | None
    norm_key: str
    opacity_level: str
    color_level: str


@dataclass(frozen=True)
class PublicFilamentUpsert:
    record: FilamentRecord
    inserted: bool


@dataclass(frozen=True)
class PersonalFilamentUpsert:
    record: FilamentRecord
    inserted: bool


@dataclass(frozen=True)
class ImportStats:
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    errors: tuple[tuple[int, str], ...] = field(default_factory=tuple)
    replaced_ids: tuple[int, ...] = ()


def connect_db(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    if conn.execute("PRAGMA user_version").fetchone()[0] != DB_USER_VERSION:
        _upgrade(conn)


def create_filament(
    conn: sqlite3.Connection,
    *,
    source: str,
    brand: str,
    name: str,
    material: str,
    hex: str,
    td: float,
    opacity_level: str | None = None,
    color_level: str = "matched",
    material_family: str = "",
    td1: float | None = None,
    notes: str = "",
    method: str = "",
    spool_id: str | None = None,
    measured_lab_l: float | None = None,
    measured_lab_a: float | None = None,
    measured_lab_b: float | None = None,
    td_confidence: float | None = None,
    measured_at: str | None = None,
    external_id: str | None = None,
    external_slug: str | None = None,
    source_url: str | None = None,
    available: bool = True,
    replaced_ids: list[int] | None = None,
    commit: bool = True,
) -> FilamentRecord:
    if source == SOURCE_PERSONAL:
        brand, name, measured = strip_measured_marker(brand, name)
        if measured:
            method = MEASURED_METHOD
    values = _validated_values(
        source=source,
        brand=brand,
        name=name,
        material=material,
        material_family=material_family,
        hex=hex,
        td=td,
        opacity_level=opacity_level,
        color_level=color_level,
        td1=td1,
        notes=notes,
        method=method,
        spool_id=spool_id,
        measured_lab_l=measured_lab_l,
        measured_lab_a=measured_lab_a,
        measured_lab_b=measured_lab_b,
        td_confidence=td_confidence,
        measured_at=measured_at,
        external_id=external_id,
        external_slug=external_slug,
        source_url=source_url,
        available=available,
    )
    if source == SOURCE_PERSONAL:
        _resolve_personal_collision(
            conn,
            norm_key_value=values["norm_key"],
            replaced_ids=replaced_ids,
            incoming_measured=values["method"] == MEASURED_METHOD,
        )
    now = _now()
    try:
        cursor = conn.execute(
            """
            INSERT INTO filaments (
                source, brand, name, material, material_family, hex, input_distance_mm, meter_distance_mm, opacity_level, color_level,
                input_distance_source, color_source,
                notes, method, spool_id, measured_lab_l, measured_lab_a,
                measured_lab_b, td_confidence, measured_at,
                external_id, external_slug, source_url,
                available, active, created_at, updated_at, norm_key
            )
            VALUES (
                :source, :brand, :name, :material, :material_family, :hex, :td, :td1, :opacity_level, :color_level,
                :opacity_level, :color_level,
                :notes, :method, :spool_id, :measured_lab_l, :measured_lab_a,
                :measured_lab_b, :td_confidence, :measured_at,
                :external_id, :external_slug, :source_url,
                :available, 1, :created_at, :updated_at, :norm_key
            )
            """,
            {**values, "created_at": now, "updated_at": now},
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError(_integrity_message(exc)) from exc
    if commit:
        conn.commit()
    return _require_record(conn, cursor.lastrowid)


def upsert_public_filament(
    conn: sqlite3.Connection,
    *,
    external_id: str,
    brand: str,
    name: str,
    material: str,
    hex: str,
    td: float,
    opacity_level: str | None = None,
    color_level: str = "matched",
    material_family: str = "",
    td1: float | None = None,
    notes: str = "",
    method: str = "",
    external_slug: str | None = None,
    source_url: str | None = None,
    available: bool = True,
    commit: bool = True,
) -> FilamentRecord:
    return upsert_public_filament_with_status(
        conn,
        external_id=external_id,
        brand=brand,
        name=name,
        material=material,
        material_family=material_family,
        hex=hex,
        td=td,
        opacity_level=opacity_level,
        color_level=color_level,
        td1=td1,
        notes=notes,
        method=method,
        external_slug=external_slug,
        source_url=source_url,
        available=available,
        commit=commit,
    ).record


def upsert_public_filament_with_status(
    conn: sqlite3.Connection,
    *,
    external_id: str,
    brand: str,
    name: str,
    material: str,
    hex: str,
    td: float,
    opacity_level: str | None = None,
    color_level: str = "matched",
    material_family: str = "",
    td1: float | None = None,
    notes: str = "",
    method: str = "",
    external_slug: str | None = None,
    source_url: str | None = None,
    available: bool = True,
    commit: bool = True,
) -> PublicFilamentUpsert:
    if not str(external_id or "").strip():
        raise ValueError("public filament external_id is required")
    existing = conn.execute(
        """SELECT id, CASE WHEN measured_at IS NOT NULL THEN meter_distance_mm END AS measured_td1
           FROM filaments WHERE source = ? AND external_id = ?""",
        (SOURCE_PUBLIC, str(external_id).strip()),
    ).fetchone()
    if existing is None:
        record = create_filament(
            conn,
            source=SOURCE_PUBLIC,
            brand=brand,
            name=name,
            material=material,
            material_family=material_family,
            hex=hex,
            td=td,
            opacity_level=opacity_level,
            color_level=color_level,
            td1=td1,
            notes=notes,
            method=method,
            external_id=str(external_id).strip(),
            external_slug=external_slug,
            source_url=source_url,
            available=available,
            commit=commit,
        )
        return PublicFilamentUpsert(record=record, inserted=True)
    record = update_filament(
        conn,
        existing["id"],
        commit=commit,
        source=SOURCE_PUBLIC,
        brand=brand,
        name=name,
        material=material,
        material_family=material_family,
        hex=hex,
        td=td,
        opacity_level=opacity_level,
        color_level=color_level,
        td1=td1 if td1 is not None else existing["measured_td1"],
        notes=notes,
        method=method,
        external_id=str(external_id).strip(),
        external_slug=external_slug,
        source_url=source_url,
        available=available,
        active=True,
        deleted_at=None,
    )
    return PublicFilamentUpsert(record=record, inserted=False)


def upsert_personal_filament(
    conn: sqlite3.Connection,
    *,
    brand: str,
    name: str,
    material: str,
    hex: str,
    td: float,
    opacity_level: str | None = None,
    color_level: str = "matched",
    material_family: str = "",
    td1: float | None = None,
    notes: str = "",
    method: str = "",
    available: bool = True,
    replaced_ids: list[int] | None = None,
    commit: bool = False,
) -> PersonalFilamentUpsert:
    """Insert or update an active ``personal`` row keyed by ``norm_key(brand,name,material)``.

    Mirrors :func:`upsert_public_filament_with_status` but keys on the personal unique
    index (``ux_filaments_personal_active_norm``) so re-importing the same filament painting
    library updates rows in place instead of failing the duplicate constraint.
    """
    brand, name, measured = strip_measured_marker(brand, name)
    if measured:
        method = MEASURED_METHOD
    key = norm_key(brand, name, material)
    existing = conn.execute(
        """SELECT id,input_distance_mm AS td,hex,method,opacity_level,color_level, CASE WHEN measured_at IS NOT NULL THEN meter_distance_mm END AS measured_td1
           FROM filaments WHERE source = ? AND active = 1 AND norm_key = ?""",
        (SOURCE_PERSONAL, key),
    ).fetchone()
    if existing is None:
        record = create_filament(
            conn,
            source=SOURCE_PERSONAL,
            brand=brand,
            name=name,
            material=material,
            material_family=material_family,
            hex=hex,
            td=td,
            opacity_level=opacity_level,
            color_level=color_level,
            td1=td1,
            notes=notes,
            method=method,
            available=available,
            replaced_ids=replaced_ids,
            commit=commit,
        )
        return PersonalFilamentUpsert(record=record, inserted=True)
    # A repeated spool-list import can omit a distance or carry the unmarked
    # catalogue twin of a legacy scan. Neither should erase the stronger value.
    estimate_only = td in (None, "") or opacity_level == "estimated"
    keep_known = estimate_only and existing["opacity_level"] != "estimated"
    incoming_measured = method == MEASURED_METHOD or opacity_level == "measured"
    if keep_known or (existing["opacity_level"] == "measured" and not incoming_measured):
        td, opacity_level, method = existing["td"], existing["opacity_level"], existing["method"]
    if existing["color_level"] == "measured" and color_level != "measured":
        hex, color_level = existing["hex"], "measured"
    record = update_filament(
        conn,
        existing["id"],
        replaced_ids=replaced_ids,
        commit=commit,
        source=SOURCE_PERSONAL,
        brand=brand,
        name=name,
        material=material,
        material_family=material_family,
        hex=hex,
        td=td,
        opacity_level=opacity_level,
        color_level=color_level,
        td1=td1 if td1 is not None else existing["measured_td1"],
        notes=notes,
        method=method,
        available=available,
        active=True,
        deleted_at=None,
    )
    return PersonalFilamentUpsert(record=record, inserted=False)


def import_spool_list(
    conn: sqlite3.Connection,
    rows: Iterable[Mapping[str, Any]],
    *,
    only_owned: bool = True,
) -> ImportStats:
    """Bulk-import filament painting filament dicts into the local DB as ``personal`` rows.

    Maps each filament painting row (``Brand``/``Name``/``Color``/``Transmissivity``/``Type``) onto a
    personal filament and upserts it by norm_key. With ``only_owned`` (the default), rows whose
    ``Owned`` flag is not truthy are skipped. A row that fails validation (bad hex/td, missing
    brand/name/material) is counted in ``skipped``/``errors`` and the batch continues — one bad
    row never aborts the import. The whole batch runs in a single transaction (mirroring
    :func:`sync_public_library`): a transient ``sqlite3.OperationalError`` rolls back and
    propagates raw so the route returns 503, rather than being swallowed as a skipped row.
    """
    inserted = updated = skipped = 0
    errors: list[tuple[int, str]] = []
    replaced_ids: list[int] = []
    savepoint_name: str | None = None
    transaction_started = False
    try:
        if conn.in_transaction:
            savepoint_name = f"prism_hf_import_{id(conn)}"
            conn.execute(f"SAVEPOINT {savepoint_name}")
        else:
            conn.execute("BEGIN IMMEDIATE")
        transaction_started = True

        for index, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                skipped += 1
                errors.append((index, "row is not an object"))
                continue
            if only_owned and not _owned_is_true(_spool_list_field(row, "owned")):
                skipped += 1
                continue
            try:
                mapped = _map_spool_list_row(row)
                result = upsert_personal_filament(
                    conn,
                    **mapped,
                    replaced_ids=replaced_ids,
                    commit=False,
                )
            except (ValueError, TypeError) as exc:
                skipped += 1
                errors.append((index, str(exc)))
                continue
            if result.inserted:
                inserted += 1
            else:
                updated += 1

        if savepoint_name is None:
            conn.commit()
        else:
            conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
    except sqlite3.OperationalError:
        # Transient DB faults (locked, disk full) roll back and propagate raw so the
        # route returns 503, matching sync_public_library — not a silent skipped row.
        if transaction_started:
            _rollback_import_transaction(conn, savepoint_name=savepoint_name)
        raise
    except sqlite3.Error:
        if transaction_started:
            _rollback_import_transaction(conn, savepoint_name=savepoint_name)
        raise
    return ImportStats(inserted, updated, skipped, tuple(errors), tuple(replaced_ids))


def _rollback_import_transaction(conn: sqlite3.Connection, *, savepoint_name: str | None) -> None:
    try:
        if savepoint_name is None:
            conn.rollback()
        else:
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
    except sqlite3.Error:
        pass


def update_filament(
    conn: sqlite3.Connection, filament_id: int, *, commit: bool = True,
    replaced_ids: list[int] | None = None, **changes: Any,
) -> FilamentRecord:
    current = _require_record(conn, filament_id)
    merged = {
        "source": current.source,
        "brand": current.brand,
        "name": current.name,
        "material": current.material,
        "material_family": current.material_family,
        "hex": current.hex,
        "td": current.td,
        "opacity_level": current.opacity_level,
        "color_level": current.color_level,
        "td1": current.td1,
        "notes": current.notes,
        "method": current.method,
        "external_id": current.external_id,
        "external_slug": current.external_slug,
        "source_url": current.source_url,
        "available": current.available,
    }
    merged.update({key: value for key, value in changes.items() if key in merged})
    if merged["source"] == SOURCE_PERSONAL:
        merged["brand"], merged["name"], measured = strip_measured_marker(
            merged["brand"], merged["name"]
        )
        if measured:
            merged["method"] = MEASURED_METHOD
    if merged["method"] != current.method and "opacity_level" not in changes:
        merged["opacity_level"] = None
    values = _validated_values(**merged)
    values["active_lot"] = str(changes.get("active_lot", current.active_lot) or "").strip() or None
    active = bool(changes.get("active", current.active))
    deleted_at = changes.get("deleted_at", current.deleted_at)
    if merged["source"] == SOURCE_PERSONAL and active:
        _resolve_personal_collision(
            conn,
            norm_key_value=values["norm_key"],
            replaced_ids=replaced_ids,
            incoming_measured=values["method"] == MEASURED_METHOD,
            exclude_id=filament_id,
        )
    try:
        conn.execute(
            """
            UPDATE filaments
            SET source = :source,
                brand = :brand,
                name = :name,
                material = :material,
                material_family = :material_family,
                hex = :hex,
                input_distance_mm = :td,
                input_distance_source = CASE WHEN :input_changed THEN :opacity_level ELSE input_distance_source END,
                color_source = CASE WHEN :color_changed THEN :color_level ELSE color_source END,
                opacity_level = :opacity_level,
                color_level = :color_level,
                meter_distance_mm = :td1,
                notes = :notes,
                active_lot = :active_lot,
                method = :method,
                external_id = :external_id,
                external_slug = :external_slug,
                source_url = :source_url,
                available = :available,
                active = :active,
                updated_at = :updated_at,
                deleted_at = :deleted_at,
                norm_key = :norm_key
            WHERE id = :id
            """,
            {
                **values,
                "active": 1 if active else 0,
                "input_changed": (values["td"], values["opacity_level"]) != (current.td, current.opacity_level),
                "color_changed": (values["hex"], values["color_level"]) != (current.hex, current.color_level),
                # A save in the creation second must still be distinguishable from an untouched row.
                "updated_at": datetime.now(UTC).isoformat(timespec="microseconds"),
                "deleted_at": deleted_at,
                "id": filament_id,
            },
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError(_integrity_message(exc)) from exc
    if commit:
        conn.commit()
    return _require_record(conn, filament_id)


def soft_delete_filament(
    conn: sqlite3.Connection, filament_id: int, *, commit: bool = True
) -> FilamentRecord:
    return update_filament(conn, filament_id, active=False, deleted_at=_now(), commit=commit)


def archive_filament(conn: sqlite3.Connection, filament_id: int, *, commit: bool = True) -> FilamentRecord:
    # Archiving changes membership only. Revalidating a legacy row here could
    # normalize its name or provenance and make exact restoration impossible.
    conn.execute(
        "UPDATE filaments SET active=0, deleted_at=? WHERE id=? AND active=1",
        (datetime.now(UTC).isoformat(timespec="microseconds"), filament_id),
    )
    record = _require_record(conn, filament_id)
    if commit:
        conn.commit()
    return record


def restore_archived_filaments(conn: sqlite3.Connection, archives: list[str]) -> int:
    """Restore exactly the archive operations shown on the page, all or none."""
    if not archives:
        raise ValueError("Choose an archived spool to restore.")
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        for archive in archives:
            spool, _, stamp = archive.partition("|")
            try:
                restored = conn.execute(
                    "UPDATE filaments SET active=1, deleted_at=NULL "
                    "WHERE id=? AND active=0 AND deleted_at IS NULLIF(?, '')",
                    (spool, stamp),
                ).rowcount
            except sqlite3.IntegrityError as exc:
                raise ValueError("An active spool already uses this identity. Review the archived spools.") from exc
            if not restored:
                raise ValueError("The archive has changed. Check the spool list and try again.")
    return len(archives)


def _resolve_personal_collision(
    conn: sqlite3.Connection,
    *,
    norm_key_value: str,
    incoming_measured: bool,
    replaced_ids: list[int] | None,
    exclude_id: int | None = None,
) -> None:
    """Enforce "measured wins" on the active-personal norm_key slot before a write.

    If an active personal row already holds ``norm_key_value`` and the incoming row is the
    measured one, archive the existing loser (freeing the partial-unique slot) so the caller
    can take it. If the incoming row loses or ties, raise the existing duplicate ``ValueError``
    — preserving today's create UX (the import path swallows it as a skipped row). No active
    twin → no-op. The caller must run inside one transaction (``commit=False`` writes).
    """
    query = (
        "SELECT id, method FROM filaments "
        "WHERE source = ? AND active = 1 AND norm_key = ?"
    )
    params: list[Any] = [SOURCE_PERSONAL, norm_key_value]
    if exclude_id is not None:
        query += " AND id <> ?"
        params.append(exclude_id)
    existing = conn.execute(query, params).fetchone()
    if existing is None:
        return
    existing_measured = existing["method"] == MEASURED_METHOD
    if measured_precedence(incoming_measured, existing_measured) == "incoming":
        archived = archive_filament(conn, existing["id"], commit=False)
        if replaced_ids is not None:
            replaced_ids.append(archived.id)
        return
    raise ValueError("duplicate active personal filament")


def merge_personal_filaments(
    conn: sqlite3.Connection, winner_id: int, loser_ids: Iterable[int]
) -> int:
    """Archive ``loser_ids`` so only ``winner_id`` survives. Returns the loser count.

    The survivor is left untouched (it already carries the values the user chose to keep);
    losers are soft-deleted, freeing their norm_key slots. All rows must be active personal
    filaments — cross-source or archived ids are rejected. One transaction.
    """
    winner = _require_record(conn, winner_id)
    if winner.source != SOURCE_PERSONAL or not winner.active:
        raise ValueError("merge winner must be an active personal filament")
    losers: list[int] = []
    seen = {winner_id}
    for loser_id in loser_ids:
        if loser_id in seen:
            continue
        seen.add(loser_id)
        record = _require_record(conn, loser_id)
        if record.source != SOURCE_PERSONAL or not record.active:
            raise ValueError("merge losers must be active personal filaments")
        losers.append(loser_id)
    if not losers:
        return 0
    started = False
    try:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
            started = True
        for loser_id in losers:
            archive_filament(conn, loser_id, commit=False)
        if started:
            conn.commit()
    except Exception:
        if started:
            conn.rollback()
        raise
    return len(losers)


def get_filament(conn: sqlite3.Connection, filament_id: int) -> FilamentRecord:
    return _require_record(conn, filament_id)


def list_filaments(
    conn: sqlite3.Connection,
    *,
    include_deleted: bool = False,
    q: str = "",
    source: str = "",
    available_only: bool = False,
    ids: list[int] | None = None,
) -> list[FilamentRecord]:
    clauses: list[str] = []
    params: list[Any] = []
    if not include_deleted:
        clauses.append("active = 1")
    if source:
        clauses.append("source = ?")
        params.append(source)
    if available_only:
        clauses.append("available = 1")
    if ids is not None:
        if not ids:
            return []
        clauses.append(f"id IN ({','.join('?' for _ in ids)})")
        params.extend(ids)
    if q:
        pattern = f"%{q.strip().lower()}%"
        clauses.append("(lower(brand) LIKE ? OR lower(name) LIKE ? OR lower(material) LIKE ?)")
        params.extend([pattern, pattern, pattern])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT * FROM filaments
        {where}
        ORDER BY lower(brand), lower(name), lower(material), id
        """,
        params,
    ).fetchall()
    return [_record_from_row(row) for row in rows]


def material_spools(rows):
    return [MaterialSpool(row, material_group(row.material, row.material_family),
                          row.available and row.source == SOURCE_PERSONAL) for row in rows]


def ranking_pool(conn, requested="auto", *, provenance=False):
    pool = material_spools(list_filaments(conn, available_only=True))
    choice = choose_material(pool, requested)
    return filaments_for_ranking(choice.values(pool), conn=conn, provenance=provenance), choice


def ranking_filaments(conn: sqlite3.Connection, *, provenance: bool = False) -> list[Filament]:
    return ranking_pool(conn, provenance=provenance)[0]


def _wedge_measurements(conn: sqlite3.Connection) -> dict[int, tuple[str, float]]:
    # Keep physical hiding distance in the measurement. Adapt only at the
    # ranking boundary, where the existing compositor expects a conventional TD.
    measurements = conn.execute(
        """SELECT w.top_id, w.measured_at, w.hiding_distance_mm
           FROM opacity_wedges w JOIN filaments f ON f.id = w.top_id AND coalesce(f.active_lot, f.spool_id) = w.spool_id
           WHERE w.hiding_distance_mm IS NOT NULL
           ORDER BY w.measured_at, w.rowid"""
    )
    td_scale = RenderConstants().td_scale
    return {row["top_id"]: (row["measured_at"], row["hiding_distance_mm"] / td_scale) for row in measurements}


def measured_filaments(conn, rows, *, readings=None):
    """Resolve each active lot independently of family selection or availability."""
    wedges = _wedge_measurements(conn)
    if readings is None:
        readings = active_readings(conn)
    result = {}
    for row in rows:
        distance, stamp = None, ""
        color, color_level = row.hex, row.color_level
        if row.td1 and row.measured_at and (row.active_lot is None or row.active_lot == row.spool_id):
            distance, stamp = row.td1, row.measured_at
        wedge = wedges.get(row.id)
        if wedge:
            stamp, distance = wedge
        reading = readings.get(row.id)
        if reading:
            color, color_level = reading["hex"], "measured"
            # Hand-entered dates have day precision. Prefer a timed wedge on a tie.
            day = reading["measured_on"]
            if day > stamp[:10] or (day == stamp[:10] and not wedge):
                distance = reading["distance_mm"]
        if distance is not None:
            result[row.id] = SourcedFilament(row.brand, row.name, color, distance, "measured", color_level)
    return result


def filaments_for_ranking(
    records: Iterable[FilamentRecord], *, conn: sqlite3.Connection | None = None, provenance: bool = False
) -> list[Filament]:
    """Apply ranking precedence within a selected pool, using active-lot wedges from the store."""
    rows = list(records)
    overrides = measured_filaments(conn, rows) if conn is not None else {}
    # A physical-spool measurement wins the local family slot, including when
    # the operator chose a synced library entry as the spool's starting point.
    overridden_families = {_family_norm_key(row) for row in rows if row.source == SOURCE_PERSONAL}
    by_key: dict[str, FilamentRecord] = {}
    for row in rows:
        family = _family_norm_key(row)
        if row.source == SOURCE_PUBLIC and family in overridden_families and row.id not in overrides:
            continue
        # Public-only variants retain their distinct full material keys.
        key = family if family in overridden_families else row.norm_key
        if key not in by_key or (row.id in overrides and by_key[key].id not in overrides):
            by_key[key] = row
    result = []
    for row in sorted(by_key.values(), key=lambda r: (r.brand.lower(), r.name.lower(), r.material.lower())):
        filament = overrides.get(row.id) or SourcedFilament(
            row.brand, row.name, row.hex, row.td, row.opacity_level, row.color_level
        )
        lot = row.active_lot or row.spool_id
        if lot and calibration_catalog():
            result.append(sourced_filament(
                filament.brand, filament.name, filament.hex, filament.td, filament.opacity_level, filament.color_level, lot
            ))
        elif provenance or "estimated" in (filament.opacity_level, filament.color_level):
            result.append(filament)
        else:
            result.append(Filament(filament.brand, filament.name, filament.hex, filament.td))
    return result


def _validated_values(
    *,
    source: str,
    brand: str,
    name: str,
    material: str,
    hex: str,
    td: float,
    opacity_level: str | None = None,
    color_level: str = "matched",
    material_family: str = "",
    td1: float | None = None,
    notes: str = "",
    method: str = "",
    spool_id: str | None = None,
    measured_lab_l: float | None = None,
    measured_lab_a: float | None = None,
    measured_lab_b: float | None = None,
    td_confidence: float | None = None,
    measured_at: str | None = None,
    external_id: str | None = None,
    external_slug: str | None = None,
    source_url: str | None = None,
    available: bool = True,
) -> dict[str, Any]:
    if source not in _VALID_SOURCES:
        raise ValueError("source must be public_curated or personal")
    row = _filament_from_row(
        {"brand": brand, "name": name, "hex": hex, "td": td, "material": material},
        row_number=1, allow_estimate=True,
    )
    if td in (None, ""):
        opacity_level = "estimated"
    material = str(material or "").strip()
    if not material:
        raise ValueError("material is required")
    td1_value = None if td1 in (None, "") else float(td1)
    if td1_value is not None and (not math.isfinite(td1_value) or td1_value <= 0):
        raise ValueError("td1 must be finite and positive")
    measured_values = {
        "measured_lab_l": _optional_float(measured_lab_l, "measured_lab_l"),
        "measured_lab_a": _optional_float(measured_lab_a, "measured_lab_a"),
        "measured_lab_b": _optional_float(measured_lab_b, "measured_lab_b"),
        "td_confidence": _optional_float(td_confidence, "td_confidence"),
    }
    if measured_values["td_confidence"] is not None and not (
        0.0 <= measured_values["td_confidence"] <= 1.0
    ):
        raise ValueError("td_confidence must be between 0 and 1")
    if opacity_level is None:
        legacy_meter = source == SOURCE_PERSONAL and method == MEASURED_METHOD
        opacity_level = "measured" if legacy_meter else "matched"
    return {
        "source": source,
        "brand": row.brand,
        "name": row.name,
        "material": material,
        "material_family": str(material_family or "").strip(),
        "hex": row.hex,
        "td": row.td,
        "opacity_level": level(opacity_level),
        "color_level": level(color_level),
        "td1": td1_value,
        "notes": str(notes or "").strip(),
        "method": str(method or "").strip(),
        "spool_id": None if spool_id in (None, "") else str(spool_id).strip(),
        **measured_values,
        "measured_at": None if measured_at in (None, "") else str(measured_at).strip(),
        "external_id": None if external_id in (None, "") else str(external_id).strip(),
        "external_slug": None if external_slug in (None, "") else str(external_slug).strip(),
        "source_url": None if source_url in (None, "") else str(source_url).strip(),
        "available": 1 if available else 0,
        "norm_key": norm_key(row.brand, row.name, material),
    }


def _optional_float(value: Any, field: str) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def norm_key(brand: str, name: str, material: str) -> str:
    return "|".join(_normalize_key(value) for value in (brand, name, material))


def _material_family(material: str) -> str:
    """Map a material string to its family token (PLA+/PLA Matte → PLA), token-aware.

    Splits off the first token (stripping a trailing ``+`` and whitespace/hyphens) and looks it
    up in an exact family table so look-alikes never collapse onto a shorter prefix (``PCTG`` is
    not ``PC``; ``COPE``/``CPE``/``PVB`` keep their own families). Unknown materials pass through
    unchanged so display is preserved.
    """
    text = str(material or "").strip()
    if not text:
        return ""
    token = re.split(r"[\s\-+]", text, maxsplit=1)[0].upper()
    return _MATERIAL_FAMILIES.get(token, text)


def _spool_list_field(row: Mapping[str, Any], *names: str) -> Any:
    """Pick the first non-empty value from a filament painting row, matching keys case/format-insensitively."""
    normalized = {_normalize_key(key): value for key, value in row.items()}
    for name in names:
        value = normalized.get(_normalize_key(name))
        if value not in (None, ""):
            return value
    return None


def _map_spool_list_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Map one raw filament painting filament dict onto Prism personal-row fields.

    Color→hex, Transmissivity→td, Type→material (+ derived material_family). Tolerant of
    the input's capitalized keys. Returns raw values; validation (hex/td/brand/name/material)
    happens downstream in :func:`upsert_personal_filament`/`_validated_values`.
    """
    material = str(_spool_list_field(row, "type", "material", "filamenttype", "materialtype") or "").strip()
    fields = {
        **{key: _spool_list_field(row, *aliases) for key, aliases in _KEYS.items()},
        "material": material,
        "material_family": _material_family(material),
        "opacity_level": _spool_list_field(row, "opacity_level"),
        "color_level": _spool_list_field(row, "color_level") or "matched",
    }
    if fields["td"] is None:
        estimate = _filament_from_row(row, row_number=1, allow_estimate=True)
        fields.update(td=estimate.td, opacity_level="estimated")
    return fields


def _family_norm_key(row: FilamentRecord) -> str:
    return norm_key(row.brand, row.name, row.material_family or row.material)


def _record_from_row(row: sqlite3.Row) -> FilamentRecord:
    return FilamentRecord(
        id=row["id"],
        source=row["source"],
        brand=row["brand"],
        name=row["name"],
        material=row["material"],
        material_family=row["material_family"],
        hex=row["hex"],
        td=row["input_distance_mm"],
        opacity_level=row["opacity_level"],
        color_level=row["color_level"],
        td1=row["meter_distance_mm"],
        notes=row["notes"],
        method=row["method"],
        spool_id=row["spool_id"],
        active_lot=row["active_lot"],
        measured_lab_l=row["measured_lab_l"],
        measured_lab_a=row["measured_lab_a"],
        measured_lab_b=row["measured_lab_b"],
        td_confidence=row["td_confidence"],
        measured_at=row["measured_at"],
        external_id=row["external_id"],
        external_slug=row["external_slug"],
        source_url=row["source_url"],
        available=bool(row["available"]),
        active=bool(row["active"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        deleted_at=row["deleted_at"],
        norm_key=row["norm_key"],
    )


def _require_record(conn: sqlite3.Connection, filament_id: int) -> FilamentRecord:
    row = conn.execute("SELECT * FROM filaments WHERE id = ?", (filament_id,)).fetchone()
    if row is None:
        raise ValueError(f"filament {filament_id} not found")
    return _record_from_row(row)


def _integrity_message(exc: sqlite3.IntegrityError) -> str:
    message = str(exc).lower()
    if "ux_filaments_personal_active_norm" in message or "filaments.norm_key" in message:
        return "duplicate active personal filament"
    if "ux_filaments_public_external" in message or "filaments.source, filaments.external_id" in message:
        return "duplicate public filament external_id"
    return str(exc)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
