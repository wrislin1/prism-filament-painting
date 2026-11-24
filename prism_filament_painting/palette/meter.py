"""Dated meter readings for an inventory row and physical lot."""
import csv
from collections import Counter
from datetime import date
from io import StringIO
import sqlite3

from .inventory import _is_positive_td, _normalize_hex


def save_readings(conn: sqlite3.Connection, text: str) -> int:
    """Validate and append the whole batch atomically. Identical replays are harmless."""
    if not text.strip() or len(text) > 256_000:
        raise ValueError("Enter a batch of up to 256,000 characters.")
    reader = csv.DictReader(StringIO(text.strip()), strict=True)
    required = {"spool", "distance_mm", "hex", "date"}
    names = set(reader.fieldnames or ())
    if (not required <= names or names - required - {"lot", "sample"}
            or len(names) != len(reader.fieldnames or ())):
        raise ValueError("Use spool,distance_mm,hex,date with optional lot and sample columns.")
    occurrences = Counter()
    inserted = 0
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        for line, row in enumerate(reader, 2):
            try:
                if None in row:
                    raise ValueError("too many columns")
                spool = int(row["spool"])
                if not 0 < spool < 2**63:
                    raise ValueError("invalid spool number")
                current = conn.execute("SELECT coalesce(active_lot,spool_id) FROM filaments WHERE id=? AND active=1", (spool,)).fetchone()
                if current is None:
                    raise ValueError("unknown or archived spool")
                distance = row["distance_mm"]
                if not _is_positive_td(distance):
                    raise ValueError("distance must be finite and positive")
                color = _normalize_hex((row["hex"] or "").strip())
                measured_on = date.fromisoformat((row["date"] or "").strip()).isoformat()
                lot = (row.get("lot") or "").strip() or current[0] or ""
                identity = (spool, lot, measured_on, float(distance), color)
                occurrences[identity] += 1
                capture = (row.get("sample") or "").strip()
                sample = f"id:{capture}" if capture else f"row:{occurrences[identity]}"
                inserted += conn.execute(
                    """INSERT INTO meter_readings(filament_id,lot,measured_on,distance_mm,hex,sample)
                       VALUES(?,?,?,?,?,?) ON CONFLICT DO NOTHING""",
                    (*identity, sample),
                ).rowcount
                if lot and not current[0]:
                    conn.execute("UPDATE filaments SET active_lot=? WHERE id=?", (lot, spool))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Line {line}: {exc}") from exc
        if reader.line_num < 2:
            raise ValueError("Enter at least one reading below the header.")
    return inserted


def active_readings(conn):
    records = conn.execute(
        """SELECT r.* FROM meter_readings r JOIN filaments f ON f.id=r.filament_id
           WHERE r.lot=coalesce(f.active_lot,f.spool_id,'') ORDER BY r.measured_on,r.id"""
    )
    return {row["filament_id"]: row for row in records}
