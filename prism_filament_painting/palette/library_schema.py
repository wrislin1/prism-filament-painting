"""Current library version and historical migrations through schema 4."""

import sqlite3


DB_USER_VERSION = 5


def _migrate(conn: sqlite3.Connection, version: int) -> None:
    if version < 1:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS filaments (
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
                spool_id TEXT,
                measured_lab_l REAL,
                measured_lab_a REAL,
                measured_lab_b REAL,
                td_confidence REAL,
                measured_at TEXT,
                external_id TEXT,
                external_slug TEXT,
                source_url TEXT,
                available INTEGER NOT NULL DEFAULT 1,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT,
                norm_key TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS ux_filaments_personal_active_norm
                ON filaments(norm_key)
                WHERE source = 'personal' AND active = 1"""
        )
        conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS ux_filaments_public_external
                ON filaments(source, external_id)
                WHERE source = 'public_curated' AND external_id IS NOT NULL"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS ix_filaments_norm_key ON filaments(norm_key)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_filaments_source ON filaments(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_filaments_brand ON filaments(brand)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_filaments_material ON filaments(material)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_filaments_active_available ON filaments(active, available)")
        version = 2
    if version < 2:
        existing = {
            row["name"] if isinstance(row, sqlite3.Row) else row[1]
            for row in conn.execute("PRAGMA table_info(filaments)").fetchall()
        }
        migrations = {
            "spool_id": "ALTER TABLE filaments ADD COLUMN spool_id TEXT",
            "measured_lab_l": "ALTER TABLE filaments ADD COLUMN measured_lab_l REAL",
            "measured_lab_a": "ALTER TABLE filaments ADD COLUMN measured_lab_a REAL",
            "measured_lab_b": "ALTER TABLE filaments ADD COLUMN measured_lab_b REAL",
            "td_confidence": "ALTER TABLE filaments ADD COLUMN td_confidence REAL",
            "measured_at": "ALTER TABLE filaments ADD COLUMN measured_at TEXT",
        }
        for column, sql in migrations.items():
            if column not in existing:
                conn.execute(sql)

    if version < 3:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS opacity_wedges (
                id TEXT PRIMARY KEY,
                top_id INTEGER NOT NULL REFERENCES filaments(id),
                spool_id TEXT NOT NULL,
                spec TEXT NOT NULL,
                observation TEXT,
                measured_at TEXT,
                ranking_hiding_mm REAL
            )"""
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS ix_wedges_spool
                ON opacity_wedges(top_id, spool_id, measured_at)"""
        )

    if version < 4:
        for column in ("opacity_level", "color_level"):
            conn.execute(
                f"ALTER TABLE filaments ADD COLUMN {column} TEXT NOT NULL DEFAULT 'matched' "
                f"CHECK ({column} IN ('measured','matched','vendor','estimated'))"
            )
        conn.execute("ALTER TABLE filaments ADD COLUMN active_lot TEXT")
        conn.execute(
            "UPDATE filaments SET opacity_level='measured' WHERE source='personal' AND method='TD1'"
        )
        conn.execute(
            """CREATE TABLE meter_readings (
                id INTEGER PRIMARY KEY, filament_id INTEGER NOT NULL REFERENCES filaments(id),
                lot TEXT NOT NULL, measured_on TEXT NOT NULL, distance_mm REAL NOT NULL,
                hex TEXT NOT NULL, sample TEXT NOT NULL,
                UNIQUE(filament_id, lot, measured_on, distance_mm, hex, sample)
            )"""
        )
        conn.execute("PRAGMA user_version = 4")
