"""Read filament data files from two reviewed MIT catalogue revisions, offline."""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

from .inventory import _is_positive_td
from .store import connect_db, init_db, upsert_public_filament_with_status

SOURCES = {
    "openprinttag": ("OpenPrintTag/openprinttag-database", "07787de895eab8d223b480e567c1d058397e59f8"),
    "ofd": ("OpenFilamentCollective/open-filament-database", "d1b75e8b7739a8991e0e3bfb600acc161c340776"),
}


def _openprinttag(data, brand):
    # Optional offline importer dependency. yaml.safe_load accepts data types only.
    from yaml import safe_load

    @lru_cache(maxsize=256)
    def brand_name(slug):
        return safe_load((data / "brands" / f"{slug}.yaml").read_text())["name"]

    for path in sorted((data / "materials").glob("*/*.yaml")):
        slug = path.parent.name
        if brand and slug != brand:
            continue
        row = safe_load(path.read_text())
        if row.get("class") != "FFF" or row.get("secondary_colors"):
            continue
        color = (row.get("primary_color") or {}).get("color_rgba", "")
        if not isinstance(color, str) or not re.fullmatch(r"#[a-fA-F0-9]{6}(?:[a-fA-F0-9]{2})?", color):
            continue
        yield path, row["uuid"], dict(brand=brand_name(slug), name=row["name"], material=row.get("type") or row.get("abbreviation") or "Unspecified",
                                      hex=color[:7], td=row.get("transmission_distance"))


def _open_filament(data, brand):
    @lru_cache(maxsize=256)
    def name(path):
        return json.loads(path.read_text())["name"]

    for path in sorted(data.glob("*/*/*/*/variant.json")):
        slug, material, *_ = path.relative_to(data).parts
        if brand and slug != brand:
            continue
        row = json.loads(path.read_text())
        color = row.get("color_hex")
        if isinstance(color, list):
            if len(color) != 1:
                continue
            color = color[0]
        if not isinstance(color, str) or not re.fullmatch(r"#?[a-fA-F0-9]{6}", color):
            continue
        product = name(path.parents[1] / "filament.json")
        yield path, row["uuid"], dict(brand=name(data / slug / "brand.json"),
                                      name=f"{product} {row['name']}", material=material,
                                      hex=color, td=row.get("transmission_distance"))


def import_catalogue(conn, source, checkout, *, brand=None, include_estimates=False):
    repo, revision = SOURCES[source]
    reader = _openprinttag if source == "openprinttag" else _open_filament
    counts = Counter()
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        for path, uid, fields in reader(checkout / "data", brand):
            if fields["td"] is not None and not _is_positive_td(fields["td"]):
                counts["invalid_distance"] += 1
                continue
            if fields["td"] is None and not include_estimates:
                counts["without_distance"] += 1
                continue
            try:
                result = upsert_public_filament_with_status(
                    conn, external_id=f"{source}:{uid}",
                    source_url=f"https://github.com/{repo}/blob/{revision}/{quote(path.relative_to(checkout).as_posix())}",
                    commit=False, **fields,
                )
            except ValueError as exc:
                raise ValueError(f"{path.relative_to(checkout)}: {exc}") from exc
            counts["inserted" if result.inserted else "updated"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=SOURCES)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("--brand", help="limit to a catalogue brand directory")
    parser.add_argument("--include-estimates", action="store_true")
    args = parser.parse_args()
    git = ["git", "-C", str(args.checkout)]
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCES[args.source][1] or subprocess.check_output(
        [*git, "status", "--porcelain", "--untracked-files=all", "--", "data", "LICENSE"]
    ):
        parser.error("Use a clean data directory at the reviewed catalogue revision.")
    with connect_db(args.database) as conn:
        init_db(conn)
        counts = import_catalogue(conn, args.source, args.checkout, brand=args.brand, include_estimates=args.include_estimates)
    print(dict(counts))


if __name__ == "__main__":
    main()
