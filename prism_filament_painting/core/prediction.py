"""Resolve the source of a front-lit stack on a known opaque backing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .calibration_board import validate_database
from .calibration_fit import (
    BoardPredictor,
    DistanceCurve,
    UnidentifiableCurve,
    fit_database,
)


@dataclass(frozen=True)
class Prediction:
    rgb: tuple[int, int, int]
    source: str


def predict_stack(formula_rgb, calibration, *recipe):
    """Try the three evidence providers in order. The formula is independent of them."""
    if calibration is not None:
        for source in ("measured", "interpolated", "fitted"):
            lookup = getattr(calibration, source)
            if lookup is not None:
                rgb = lookup(*recipe)
                if rgb is not None:
                    if len(rgb) != 3 or any(
                        type(v) is not int or not 0 <= v <= 255 for v in rgb
                    ):
                        raise ValueError(
                            f"{source} prediction must be an 8-bit RGB triple"
                        )
                    return Prediction(tuple(rgb), source)
    return Prediction(formula_rgb, "formula")


def _set_key(ids, backing):
    return sha256(
        json.dumps([backing, sorted(ids)], ensure_ascii=True).encode()
    ).hexdigest()


def _opaque_threshold(board, verified):
    if verified is None:
        return board["backing_mm"]
    if type(verified) not in (float, int) or not 0 < verified <= board["backing_mm"]:
        raise ValueError(
            "verified opaque thickness must be positive and at most the board backing thickness"
        )
    return verified


def install_capture(
    directory, database, captured_on, valid_through, *, opaque_from_mm=None
):
    """Publish the database and its own holdout result as one immutable dated file."""
    validate_database(database)
    start, end = date.fromisoformat(captured_on), date.fromisoformat(valid_through)
    if start > datetime.now(timezone.utc).date() or end < start:
        raise ValueError("invalid capture or validity date")
    materials = database["board"]["materials"]
    scope = _set_key([m["id"] for m in materials], materials[0]["id"])
    root = Path(directory) / database["capture"]["kind"]
    root.mkdir(parents=True, exist_ok=True)
    bundle = {
        "database": database,
        "opaque_from_mm": _opaque_threshold(database["board"], opaque_from_mm),
        "captured_on": start.isoformat(),
        "valid_through": end.isoformat(),
    }
    try:
        bundle["fit"] = fit_database(database)
    except UnidentifiableCurve as exc:
        bundle.update(fit=None, fit_error=str(exc))
    dest = root / f"{scope}.{start.isoformat()}.json"
    with NamedTemporaryFile(mode="w", dir=root, encoding="utf-8") as tmp:
        json.dump(bundle, tmp, allow_nan=False)
        tmp.flush()
        os.link(tmp.name, dest)
    return dest


def _accepted_curve(fit, database):
    if fit is None:
        return None
    board = database["board"]
    if (
        fit["schema"] != "front-lit-fit/v1"
        or fit["materials"] != board["materials"]
        or fit["capture"] != database["capture"]
        or fit["layer_mm"] != board["layer_mm"]
    ):
        raise ValueError("fit does not belong to this colour database")
    curve = fit["accepted_curve"]
    if curve is None:
        return None
    if curve != fit["candidate"] or not all(
        fit["fitted"][key] < fit["baseline"][key]
        for key in ("mean_de00", "median_de00")
    ):
        raise ValueError("curve does not match the hold-out verdict")
    return DistanceCurve(**curve)


class BoardCalibration:
    def __init__(self, bundle):
        db = bundle["database"]
        validate_database(db)
        board = db["board"]
        self.opaque_from_mm = _opaque_threshold(board, bundle.get("opaque_from_mm"))
        self.predictor = BoardPredictor(board)
        self.materials = {
            m["id"]: (m["hex"].upper(), m["td_mm"]) for m in board["materials"]
        }
        self.backing = board["materials"][0]["id"]
        self.layer_mm = board["layer_mm"]
        self.kind = db["capture"]["kind"]
        self.captured_on = date.fromisoformat(bundle["captured_on"])
        self.valid_through = date.fromisoformat(bundle["valid_through"])
        self.version = f'{self.captured_on}/{db["capture"]["photo_sha256"]}'
        self.curve = _accepted_curve(bundle["fit"], db)
        self.samples = {
            tuple((s, n) for s, n in patch["bands"]): tuple(
                db["measurements"][patch["id"]]["srgb8"]
            )
            for patch in board["patches"]
        }
        self.neighbours = {}
        for bands, rgb in self.samples.items():
            if bands:
                lower, (top, n) = bands[:-1], bands[-1]
                self.neighbours.setdefault((lower, top), []).append((n, rgb))
        for points in self.neighbours.values():
            points.sort()

    def supports_backing(self, thickness_mm):
        return thickness_mm >= self.opaque_from_mm or math.isclose(
            thickness_mm, self.opaque_from_mm, abs_tol=1e-9
        )

    def measured(self, bands):
        return self.samples.get(bands)

    def interpolated(self, bands):
        if bands:
            lower, (top, n) = bands[:-1], bands[-1]
            points = self.neighbours.get((lower, top), ())
            for (a, first), (b, second) in zip(points, points[1:]):
                if a < n < b:
                    return tuple(
                        round(x + (y - x) * (n - a) / (b - a))
                        for x, y in zip(first, second)
                    )
        return None

    def fitted(self, bands):
        if self.curve is not None:
            return tuple(map(int, self.predictor.predict(bands, self.curve)))
        return None


@lru_cache(maxsize=32)
def _read_capture(path, inode, mtime_ns):
    return BoardCalibration(json.loads(path.read_text(encoding="utf-8")))


def calibration_catalog():
    path = os.environ.get("PRISM_CALIBRATION_DIR")
    if path:
        kind = (
            "synthetic"
            if os.environ.get("PRISM_SYNTHETIC_CALIBRATION") == "1"
            else "printed"
        )
        root = Path(path) / kind
        try:
            stat = root.stat()
            return _catalog(root, stat.st_ino, stat.st_mtime_ns)
        except OSError:
            pass
    return {}


@lru_cache(maxsize=8)
def _catalog(root, inode, mtime_ns):
    latest = {}
    for path in sorted(root.glob("*.????-??-??.json")):
        key, _, _ = path.name.rsplit(".", 2)
        latest[key] = path
    return latest


def select_calibration(filaments, backing, layer_height_mm):
    """An unidentified lot or a mismatch is a formula-only model, never a near match."""
    catalog = calibration_catalog()
    if not catalog:
        return None
    ids = [getattr(f, "spool_id", None) for f in filaments]
    if (
        len(ids) != 4
        or not all(isinstance(s, str) and s for s in ids)
        or len(set(ids)) != 4
    ):
        return None
    backing_id = getattr(backing, "spool_id", None)
    path = catalog.get(_set_key(ids, backing_id))
    if path is None:
        return None
    try:
        stat = path.stat()
        book = _read_capture(path, stat.st_ino, stat.st_mtime_ns)
        today = datetime.now(timezone.utc).date()
        if (
            book.kind == path.parent.name
            and book.backing == backing_id
            and book.layer_mm == layer_height_mm
            and book.captured_on.isoformat() == path.stem.rsplit(".", 1)[1]
            and book.captured_on <= today <= book.valid_through
            and all(
                book.materials.get(s) == (f.hex.upper(), f.input_distance_mm)
                for s, f in zip(ids, filaments)
            )
        ):
            return book
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return None
