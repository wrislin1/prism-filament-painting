"""Create a schema-5 copy of a selected local spool library."""

from contextlib import closing
import json
from itertools import zip_longest
import math
from pathlib import Path
import sqlite3

from .library_schema import _migrate
from .migration_copy import staged_copy


def _readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def _version(db):
    return db.execute("PRAGMA user_version").fetchone()[0]


def _equivalent(left, right):
    # An existing destination can have committed or pending WAL transactions.
    # Compare consistent logical snapshots, including schema, indexes and IDs.
    with closing(_readonly(left)) as a, closing(_readonly(right)) as b:
        a.execute("BEGIN")
        b.execute("BEGIN")
        return _version(a) == _version(b) and all(
            x == y for x, y in zip_longest(a.iterdump(), b.iterdump())
        )


def _rank_inputs(db, *, renamed):
    # All inputs to lot/date selection, family precedence and ordering. Keep the
    # rowid tie-break on wedges with equal timestamps. Raw observations are archival.
    td, meter, wedge = (("input_distance_mm", "meter_distance_mm", "hiding_distance_mm")
                        if renamed else ("td", "td1", "ranking_hiding_mm"))
    groups = (
        db.execute(f"""SELECT id, brand, name, material, material_family, norm_key,
            active, available, source, hex, {td}, {meter}, opacity_level, color_level,
            active_lot, spool_id, measured_at FROM filaments ORDER BY id""").fetchall(),
        db.execute(f"""SELECT id, top_id, spool_id, measured_at, {wedge}
            FROM opacity_wedges ORDER BY measured_at, rowid""").fetchall(),
        db.execute("SELECT * FROM meter_readings ORDER BY measured_on, id").fetchall(),
    )
    return tuple([tuple(row) for row in group] for group in groups)


def _check_integrity(db):
    if db.execute("PRAGMA foreign_key_check").fetchone() or [row[0] for row in db.execute("PRAGMA integrity_check")] != ["ok"]:
        raise ValueError("invalid spool library")


def _upgrade(db, *, before_commit=None):
    db.execute("PRAGMA foreign_keys=ON")
    with db:
        db.execute("BEGIN IMMEDIATE")
        version = _version(db)
        if version == 5:
            _rank_inputs(db, renamed=True)
            db.execute("SELECT input_distance_source, color_source FROM filaments LIMIT 0")
            _check_integrity(db)
            return
        if version not in range(5) or (version == 0 and db.execute(
                "SELECT 1 FROM sqlite_schema LIMIT 1").fetchone()):
            raise ValueError(f"unsupported library version {version}")

        _migrate(db, version)
        before = _rank_inputs(db, renamed=False)
        db.execute("ALTER TABLE filaments RENAME COLUMN td TO input_distance_mm")
        db.execute("ALTER TABLE filaments RENAME COLUMN td1 TO meter_distance_mm")
        db.execute("ALTER TABLE opacity_wedges RENAME COLUMN ranking_hiding_mm TO hiding_distance_mm")
        db.execute("ALTER TABLE filaments ADD COLUMN input_distance_source TEXT NOT NULL DEFAULT 'unknown'")
        db.execute("ALTER TABLE filaments ADD COLUMN color_source TEXT NOT NULL DEFAULT 'unknown'")
        # Versions before 4 had no independent provenance fields. The v4
        # compatibility defaults cannot establish the origin of a raw input.
        if version == 4:
            db.execute("UPDATE filaments SET input_distance_source=opacity_level, color_source=color_level")
        else:
            db.execute("UPDATE filaments SET input_distance_source='measured' WHERE opacity_level='measured'")
        for row_id, text in db.execute("SELECT id, observation FROM opacity_wedges WHERE observation IS NOT NULL"):
            observation = json.loads(text)
            if not isinstance(observation, dict):
                raise ValueError("invalid wedge observation")
            for old, new in (("hiding_mm", "hiding_distance_interval_mm"),
                             ("ranking_hiding_mm", "hiding_distance_mm")):
                if old in observation:
                    if new in observation:
                        raise ValueError("ambiguous wedge observation")
                    observation[new] = observation.pop(old)
            db.execute("UPDATE opacity_wedges SET observation=? WHERE id=?",
                       (json.dumps(observation, allow_nan=False), row_id))

        for (value,) in db.execute("""SELECT input_distance_mm FROM filaments
                UNION ALL SELECT meter_distance_mm FROM filaments WHERE meter_distance_mm IS NOT NULL
                UNION ALL SELECT distance_mm FROM meter_readings
                UNION ALL SELECT hiding_distance_mm FROM opacity_wedges WHERE hiding_distance_mm IS NOT NULL"""):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid stored optical distance")
        if _rank_inputs(db, renamed=True) != before:
            raise ValueError("stored ranking inputs changed during migration")
        _check_integrity(db)
        if before_commit is not None:
            before_commit(db)
        db.execute("PRAGMA user_version=5")


def migrate_library_copy(source, destination, *, before_commit=None):
    """Back up committed SQLite state (including WAL), then upgrade only the copy.

    The source is the previous release's rollback file. Repeating a migration
    accepts an equivalent destination and refuses to overwrite a divergent one.
    """
    with staged_copy(source, destination, equivalent=_equivalent) as copy:
        with closing(_readonly(source)) as original, closing(sqlite3.connect(copy)) as db:
            original.backup(db)
            db.execute("PRAGMA journal_mode=DELETE")
            _upgrade(db, before_commit=before_commit)
