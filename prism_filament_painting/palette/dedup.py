"""Filament de-duplication: the "TD1" measured-marker normalizer + duplicate review.

Users mark a hand-measured filament by typing a ``TD1`` prefix into the brand (e.g.
``brand="TD1 Sunlu", name="Black"``). That defeats the natural identity key
(``norm_key`` is brand+name+material) so a measured row and its stale twin both survive
as duplicates. This module:

* :func:`strip_measured_marker` — turn that typed prefix into a first-class
  ``method="TD1"`` marker (pure, idempotent, used on every personal write).
* :func:`measured_precedence` — "a measured row beats an unmeasured one" comparator.
* :func:`normalize_measured_markers` — one-shot migration that strips existing markers
  and auto-merges the *exact* twin (measured wins). Idempotent.
* :func:`find_duplicate_candidates` — conservative grouping of the *drifted* near-dupes
  (``Grey``/``Gray``; brand ``BambuLab Matte`` vs ``BambuLab``) for a manual review UI.
  Never auto-merges.

Module-top imports stay free of :mod:`palette.store` (which imports the pure helpers
here) — the DB functions import ``store`` lazily inside their bodies.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .inventory import _normalize_key, strip_measured_marker

#: The canonical "this is my measured value" marker, stored in ``filaments.method``.
MEASURED_METHOD = "TD1"

# Review-UI tuning. Detection feeds a manual confirm step, so it errs toward surfacing:
# the brand gate is strict (one collapsed brand is a prefix of the other) but the
# name/colour signals are generous enough to catch re-measured drift.
_COLOR_DISTANCE_MAX = 30.0  # RGB Euclidean; re-measured same colour ~13-20, different >50
_NAME_EDIT_MAX = 2  # "grey" vs "gray" == 1
_MIN_BRAND_PREFIX = 3


def measured_precedence(incoming_measured: bool, existing_measured: bool) -> str:
    """Return ``'incoming'`` | ``'existing'`` | ``'equal'`` — a measured row beats an
    unmeasured one; same-class is ``'equal'`` (caller keeps its current behaviour)."""
    if incoming_measured and not existing_measured:
        return "incoming"
    if existing_measured and not incoming_measured:
        return "existing"
    return "equal"


@dataclass(frozen=True)
class NormalizationReport:
    scanned: int = 0
    renamed_only: int = 0  # marker stripped, no twin
    merged: int = 0  # measured won an exact norm_key collision; loser archived
    skipped_no_marker: int = 0  # no marker (idempotent re-run lands here)
    survivors: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DuplicateGroup:
    rows: tuple  # tuple[FilamentRecord, ...]
    reason: str


def normalize_measured_markers(conn) -> NormalizationReport:
    """Strip existing ``TD1`` markers from active personal rows and auto-merge exact twins.

    For each active personal row with a marker: recompute the stripped ``norm_key`` and look
    for the stale active twin. No twin → rename in place + ``method='TD1'``. Exact twin →
    archive the twin **first** (frees the partial-unique slot), then re-key this measured row.
    One transaction; never lets two active personal rows share a norm_key. Idempotent: a
    second run finds no markers and changes nothing.
    """
    from .store import (
        SOURCE_PERSONAL,
        list_filaments,
        norm_key,
        soft_delete_filament,
        update_filament,
    )

    scanned = renamed = merged = skipped = 0
    survivors: list[int] = []
    started = False
    try:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
            started = True

        for record in list_filaments(conn, source=SOURCE_PERSONAL):
            scanned += 1
            brand, name, measured = strip_measured_marker(record.brand, record.name)
            if not measured:
                skipped += 1
                continue
            target_key = norm_key(brand, name, record.material)
            twin = conn.execute(
                "SELECT id, method FROM filaments "
                "WHERE source = ? AND active = 1 AND norm_key = ? AND id <> ?",
                (SOURCE_PERSONAL, target_key, record.id),
            ).fetchone()
            if twin is None:
                update_filament(
                    conn, record.id, brand=brand, name=name,
                    method=MEASURED_METHOD, commit=False,
                )
                renamed += 1
                survivors.append(record.id)
                continue
            if twin["method"] == MEASURED_METHOD:
                # An already-normalized measured row owns the slot; this is a redundant
                # measured duplicate — archive it and keep the existing survivor.
                soft_delete_filament(conn, record.id, commit=False)
                survivors.append(twin["id"])
            else:
                # Stale unmeasured twin loses: archive it first, then re-key this row.
                soft_delete_filament(conn, twin["id"], commit=False)
                update_filament(
                    conn, record.id, brand=brand, name=name,
                    method=MEASURED_METHOD, commit=False,
                )
                survivors.append(record.id)
            merged += 1

        if started:
            conn.commit()
    except Exception:
        if started:
            conn.rollback()
        raise

    return NormalizationReport(
        scanned=scanned,
        renamed_only=renamed,
        merged=merged,
        skipped_no_marker=skipped,
        survivors=tuple(survivors),
    )


def find_duplicate_candidates(conn) -> list[DuplicateGroup]:
    """Group active personal rows that look like the same filament under different names.

    Exact norm_key dupes cannot exist (the partial unique index forbids them), so this
    targets the naming drift the migration cannot safely auto-merge. Two rows link only
    when they share a material family AND one collapsed brand is a prefix of the other AND
    (their names match/subset/are within one or two edits, OR their colours are close).
    Connected rows form a group; groups feed the review UI — nothing is auto-merged.
    """
    from .store import SOURCE_PERSONAL, list_filaments

    rows = list_filaments(conn, source=SOURCE_PERSONAL)
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    for x in rows:
        parent.setdefault(x.id, x.id)
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if _are_candidates(a, b):
                union(a.id, b.id)

    clusters: dict[int, list] = {}
    for record in rows:
        clusters.setdefault(find(record.id), []).append(record)

    groups: list[DuplicateGroup] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        members = sorted(
            members,
            key=lambda r: (r.method != MEASURED_METHOD, r.brand.lower(), r.name.lower()),
        )
        family = members[0].material_family or members[0].material or "filament"
        groups.append(
            DuplicateGroup(
                rows=tuple(members),
                reason=f"Same {family} family — brand/name look like the same filament.",
            )
        )
    groups.sort(key=lambda g: (g.rows[0].brand.lower(), g.rows[0].name.lower()))
    return groups


def _are_candidates(a, b) -> bool:
    if _normalize_key(a.material_family or a.material) != _normalize_key(
        b.material_family or b.material
    ):
        return False
    if not _brand_prefix_match(a.brand, b.brand):
        return False
    if _name_match(a.name, b.name):
        return True
    return _rgb_distance(a.hex, b.hex) <= _COLOR_DISTANCE_MAX


def _brand_prefix_match(brand_a: str, brand_b: str) -> bool:
    ca, cb = _normalize_key(brand_a), _normalize_key(brand_b)
    if not ca or not cb or min(len(ca), len(cb)) < _MIN_BRAND_PREFIX:
        return False
    return ca.startswith(cb) or cb.startswith(ca)


def _name_match(name_a: str, name_b: str) -> bool:
    na, nb = _normalize_key(name_a), _normalize_key(name_b)
    if na == nb:
        return True
    ta, tb = _tokens(name_a), _tokens(name_b)
    if ta and tb and (ta <= tb or tb <= ta):
        return True
    return _edit_distance(na, nb) <= _NAME_EDIT_MAX


def _tokens(text: str) -> set[str]:
    return {tok for tok in (_normalize_key(p) for p in re.split(r"\s+", (text or "").strip())) if tok}


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > _NAME_EDIT_MAX:
        return _NAME_EDIT_MAX + 1  # early-out; we only care about the <= threshold
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[len(b)]


def _rgb_distance(hex_a: str, hex_b: str) -> float:
    try:
        return math.dist(_hex_to_rgb(hex_a), _hex_to_rgb(hex_b))
    except (ValueError, IndexError):
        return math.inf


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = (value or "").lstrip("#")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
