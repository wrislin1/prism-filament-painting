"""Invented spool histories in the four historical library schemas."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import json
import multiprocessing
import os
import sqlite3

import pytest

from prism_filament_painting.palette.library_migration import _upgrade, migrate_library_copy


V1 = """
CREATE TABLE IF NOT EXISTS filaments (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL CHECK(source IN ('public_curated', 'personal')),
    brand TEXT NOT NULL,
    name TEXT NOT NULL,
    material TEXT NOT NULL,
    material_family TEXT NOT NULL DEFAULT '',
    hex TEXT NOT NULL,
    td REAL NOT NULL,
    td1 REAL,
    notes TEXT NOT NULL DEFAULT '',
    method TEXT NOT NULL DEFAULT '',
    external_id TEXT,
    external_slug TEXT,
    source_url TEXT,
    available INTEGER NOT NULL DEFAULT 1,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    norm_key TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_filaments_personal_active_norm
    ON filaments(norm_key)
    WHERE source = 'personal' AND active = 1;
CREATE UNIQUE INDEX IF NOT EXISTS ux_filaments_public_external
    ON filaments(source, external_id)
    WHERE source = 'public_curated' AND external_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_filaments_norm_key ON filaments(norm_key);
CREATE INDEX IF NOT EXISTS ix_filaments_source ON filaments(source);
CREATE INDEX IF NOT EXISTS ix_filaments_brand ON filaments(brand);
CREATE INDEX IF NOT EXISTS ix_filaments_material ON filaments(material);
CREATE INDEX IF NOT EXISTS ix_filaments_active_available ON filaments(active, available);
PRAGMA user_version = 1;
"""


def library(path, version):
    """Build each old layout independently of the migration being tested."""
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript(V1)
        for i, name, active, available in [(18, "Flint", 1, 1), (35, "Brume", 1, 1),
                                           (64, "Flint", 0, 0), (81, "Malt", 1, 0)]:
            db.execute("""INSERT INTO filaments(id,source,brand,name,material,material_family,
                hex,td,td1,created_at,updated_at,norm_key,active,available,external_id)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                       (i, "public_curated" if i == 81 else "personal", "Kite", name, "PLA", "PLA",
                        "#362b58", 6.75, 3.5 if i == 18 else None, "2001-03-01", "2001-03-02",
                        name.lower(), active, available, "catalog-8" if i == 81 else None))
        if version >= 2:
            for column in ("spool_id TEXT", "measured_lab_l REAL", "measured_lab_a REAL",
                           "measured_lab_b REAL", "td_confidence REAL", "measured_at TEXT"):
                db.execute(f"ALTER TABLE filaments ADD COLUMN {column}")
            db.execute("""UPDATE filaments SET spool_id='coil-alpha', method='TD1',
                measured_at='2001-03-02', measured_lab_l=35, measured_lab_a=-5,
                measured_lab_b=14, td_confidence=0.75 WHERE id=18""")
        if version >= 3:
            db.executescript("""
                CREATE TABLE opacity_wedges(id TEXT PRIMARY KEY, top_id INTEGER NOT NULL REFERENCES
                    filaments(id), spool_id TEXT NOT NULL, spec TEXT NOT NULL, observation TEXT,
                    measured_at TEXT, ranking_hiding_mm REAL);
                CREATE INDEX ix_wedges_spool ON opacity_wedges(top_id,spool_id,measured_at);
            """)
            spec = {"id": "invented-patches", "top": {"filament_id": 18, "spool_id": "coil-alpha",
                    "library_td_mm": 6.75}, "layer_mm": 0.08, "divisor": 1.5}
            for lot, distance in [("coil-alpha", 0.375), ("coil-spare", None)]:
                observation = {"ranking_hiding_mm": distance, "raw": {"passes": [7, 7, 8]},
                               "hiding_mm": [0.25, 0.5 if distance else None], "crossing_mm": [0.4, 0.6]}
                db.execute("INSERT INTO opacity_wedges VALUES(?,?,?,?,?,?,?)",
                           (lot, 18, lot, json.dumps(spec), json.dumps(observation), "2001-03-05", distance))
        if version >= 4:
            for col in ("opacity_level", "color_level"):
                db.execute(f"ALTER TABLE filaments ADD COLUMN {col} TEXT NOT NULL DEFAULT 'matched' "
                           f"CHECK ({col} IN ('measured','matched','vendor','estimated'))")
            db.execute("ALTER TABLE filaments ADD COLUMN active_lot TEXT")
            db.execute("UPDATE filaments SET opacity_level='vendor',color_level='estimated',active_lot=spool_id")
            db.execute("""CREATE TABLE meter_readings(id INTEGER PRIMARY KEY, filament_id INTEGER NOT NULL
                REFERENCES filaments(id), lot TEXT NOT NULL, measured_on TEXT NOT NULL, distance_mm REAL
                NOT NULL, hex TEXT NOT NULL, sample TEXT NOT NULL,
                UNIQUE(filament_id,lot,measured_on,distance_mm,hex,sample))""")
            db.executemany("INSERT INTO meter_readings VALUES(?,18,?,?,?,?,?)", [
                (59, "coil-alpha", "2001-03-05", 4.25, "#314762", "tip"),
                (73, "coil-spare", "2001-03-06", 9, "#a47a23", "middle")])
        db.execute(f"PRAGMA user_version={version}")


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_library_golden(tmp_path, version):
    source, dest = tmp_path / "source.db", tmp_path / "converted.db"
    library(source, version)
    original = source.read_bytes()
    migrate_library_copy(source, dest)
    with closing(sqlite3.connect(dest)) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (5,)
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert db.execute("SELECT id,active,available FROM filaments ORDER BY id").fetchall() == [
            (18, 1, 1), (35, 1, 1), (64, 0, 0), (81, 1, 0)]
        assert db.execute("""SELECT input_distance_mm,meter_distance_mm,input_distance_source,
            color_source,spool_id FROM filaments WHERE id=18""").fetchone() == (
                6.75, 3.5, {1: "unknown", 2: "measured", 3: "measured", 4: "vendor"}[version],
                "estimated" if version == 4 else "unknown", "coil-alpha" if version >= 2 else None)
        if version >= 3:
            rows = db.execute("SELECT observation,hiding_distance_mm FROM opacity_wedges ORDER BY id").fetchall()
            assert [r[1] for r in rows] == [0.375, None]
            assert [json.loads(r[0])["hiding_distance_interval_mm"] for r in rows] == [[0.25, 0.5], [0.25, None]]
            assert json.loads(rows[0][0])["crossing_mm"] == [0.4, 0.6]
        if version == 4:
            assert db.execute("SELECT * FROM meter_readings ORDER BY id").fetchall() == [
                (59, 18, "coil-alpha", "2001-03-05", 4.25, "#314762", "tip"),
                (73, 18, "coil-spare", "2001-03-06", 9., "#a47a23", "middle")]
            assert db.execute("SELECT active_lot FROM filaments WHERE id=18").fetchone() == ("coil-alpha",)
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE filaments SET active=1 WHERE id=64")
    first = dest.read_bytes()
    migrate_library_copy(source, dest)
    assert dest.read_bytes() == first
    migrate_library_copy(dest, tmp_path / "second.db")
    assert source.read_bytes() == original


def test_empty_and_unsupported(tmp_path):
    source, dest = tmp_path / "empty", tmp_path / "result"
    with closing(sqlite3.connect(source)):
        pass
    migrate_library_copy(source, dest)
    with closing(sqlite3.connect(dest)) as db:
        assert db.execute("SELECT count(*) FROM filaments").fetchone() == (0,)
    with closing(sqlite3.connect(source)) as db:
        db.execute("PRAGMA user_version=9")
    with pytest.raises(ValueError, match="version"):
        migrate_library_copy(source, tmp_path / "unknown")
    with pytest.raises(ValueError, match="separate"):
        migrate_library_copy(source, source)
    assert not (tmp_path / "unknown").exists()


def test_failure_rolls_back_ddl_and_rows_then_retry(tmp_path):
    source, dest = tmp_path / "old", tmp_path / "new"
    library(source, 4)
    with closing(sqlite3.connect(source)) as db, db:
        db.execute("UPDATE opacity_wedges SET observation='invalid' WHERE id='coil-spare'")
    original = source.read_bytes()
    with pytest.raises(ValueError):
        migrate_library_copy(source, dest)
    assert source.read_bytes() == original and not dest.exists()
    # Inspect the failed transaction itself as well as the outer copy guarantee.
    with closing(sqlite3.connect(":memory:")) as copy, closing(sqlite3.connect(source)) as db:
        db.backup(copy)
        with pytest.raises(ValueError):
            _upgrade(copy)
        assert copy.execute("PRAGMA user_version").fetchone() == (4,)
        assert copy.execute("SELECT td FROM filaments WHERE id=18").fetchone() == (6.75,)
        observation, = copy.execute("SELECT observation FROM opacity_wedges WHERE id='coil-alpha'").fetchone()
        assert "ranking_hiding_mm" in json.loads(observation)
        copy.execute("UPDATE opacity_wedges SET observation=NULL WHERE id='coil-spare'")
        copy.commit()
        _upgrade(copy)
        assert copy.execute("PRAGMA user_version").fetchone() == (5,)


def _race(source, dest, ready):
    ready.wait(timeout=30)
    migrate_library_copy(source, dest)


def test_two_processes_and_pending_writer(tmp_path):
    source, dest = tmp_path / "wal.db", tmp_path / "v5.db"
    library(source, 4)
    with closing(sqlite3.connect(source)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("UPDATE filaments SET notes='committed in WAL' WHERE id=18")
        writer.commit()
        writer.execute("UPDATE filaments SET notes='uncommitted' WHERE id=18")
        ctx = multiprocessing.get_context("spawn")
        with ctx.Manager() as manager, ProcessPoolExecutor(2, mp_context=ctx) as pool:
            barrier = manager.Barrier(2)
            jobs = [pool.submit(_race, source, dest, barrier) for _ in range(2)]
            for job in jobs:
                job.result(timeout=60)
        writer.rollback()
        assert writer.execute("PRAGMA user_version").fetchone() == (4,)
    with closing(sqlite3.connect(dest)) as db, db:
        assert db.execute("SELECT notes FROM filaments WHERE id=18").fetchone() == ("committed in WAL",)
        db.execute("UPDATE filaments SET notes='new local edit' WHERE id=18")
    with pytest.raises(FileExistsError):
        migrate_library_copy(source, dest)


def _upgrade_shared_copy(copy, ready):
    with closing(sqlite3.connect(copy)) as db:
        ready.wait(timeout=30)
        _upgrade(db)


def test_concurrent_writers_recheck_version_inside_copy_transaction(tmp_path):
    original, copy = tmp_path / "original", tmp_path / "shared-copy"
    library(original, 1)
    with closing(sqlite3.connect(original)) as a, closing(sqlite3.connect(copy)) as b:
        a.backup(b)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Manager() as manager, ProcessPoolExecutor(2, mp_context=ctx) as pool:
        barrier = manager.Barrier(2)
        jobs = [pool.submit(_upgrade_shared_copy, copy, barrier) for _ in range(2)]
        for job in jobs:
            job.result(timeout=60)
    with closing(sqlite3.connect(original)) as a, closing(sqlite3.connect(copy)) as b:
        assert a.execute("PRAGMA user_version").fetchone() == (1,)
        assert b.execute("PRAGMA user_version").fetchone() == (5,)
        assert b.execute("SELECT count(*) FROM filaments").fetchone() == (4,)


def _interrupt_copy(copy):
    def stop_after_filament_ddl(statement):
        if statement.startswith("ALTER TABLE opacity_wedges RENAME"):
            os._exit(17)

    with closing(sqlite3.connect(copy)) as db:
        db.set_trace_callback(stop_after_filament_ddl)
        _upgrade(db)


def test_process_interruption_and_retry(tmp_path):
    source, copy = tmp_path / "historical", tmp_path / "working"
    library(source, 3)
    with closing(sqlite3.connect(source)) as a, closing(sqlite3.connect(copy)) as b:
        a.backup(b)
    proc = multiprocessing.get_context("spawn").Process(target=_interrupt_copy, args=(copy,))
    proc.start()
    proc.join(30)
    try:
        assert proc.exitcode == 17
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join()
    with closing(sqlite3.connect(copy)) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (3,)
        assert db.execute("SELECT td,td1 FROM filaments WHERE id=18").fetchone() == (6.75, 3.5)
        _upgrade(db)
        assert db.execute("PRAGMA user_version").fetchone() == (5,)
