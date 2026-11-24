"""End-to-end acceptance on an independently projected synthetic plate photograph."""

from collections import Counter
from hashlib import sha256
from io import BytesIO
import copy
import json
import struct

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from prism_filament_painting.core.calibration_board import make_board, validate_database
from prism_filament_painting.core.calibration_fit import BoardPredictor, DistanceCurve, fit_database
from prism_filament_painting.core.color import beer_lambert_blend, hex_to_rgb
from prism_filament_painting.calibration.__main__ import main
from prism_filament_painting.calibration.photo import extract


MATERIALS = [
    {"id": "base", "hex": "#242833", "td_mm": 0.5},
    {"id": "yellow", "hex": "#DDBF48", "td_mm": 1.2},
    {"id": "red", "hex": "#C64554", "td_mm": 3.5},
    {"id": "blue", "hex": "#558DDA", "td_mm": 7.0},
]
TRUTH = DistanceCurve(0.16, 0.68)


def project(x, y):
    # Analytic camera projection independent of the corner solver.
    return [
        (5.3 * x + 0.48 * y + 24) / (1 + 0.0017 * x - 0.0008 * y),
        (0.26 * x + 4.7 * y + 19) / (1 + 0.0017 * x - 0.0008 * y),
    ]


def synthetic_photo(board, truth):
    image = Image.new("RGB", (680, 360), (70, 70, 70))
    draw = ImageDraw.Draw(image)

    def rectangle(x, y, w, h, rgb):
        draw.polygon(
            [tuple(project(u, v)) for u, v in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))],
            fill=rgb,
        )

    rectangle(0, 0, *board["size_mm"], hex_to_rgb(MATERIALS[0]["hex"]))
    for x, y in board["mark_centres_mm"]:
        rectangle(x - 1.6, y - 1.6, 3.2, 3.2, hex_to_rgb(MATERIALS[1]["hex"]))
    rectangle(7, 3, 1.6, 2, hex_to_rgb(MATERIALS[1]["hex"]))
    expected = {}
    for patch in board["patches"]:
        rgb = hex_to_rgb(MATERIALS[0]["hex"])
        for spool, count in patch["bands"]:
            material = next(m for m in MATERIALS if m["id"] == spool)
            distance = truth.distance_at_td1_mm * material["td_mm"] ** truth.exponent
            rgb = beer_lambert_blend(hex_to_rgb(material["hex"]), rgb, count * 0.08, distance)
        x, y = patch["xy_mm"]
        rectangle(x, y, 10, 10, rgb)
        # Small specular defect inside the measurement window.
        rectangle(x + 4.5, y + 4, 1.1, 1.1, (250, 250, 250))
        expected[patch["id"]] = rgb
    clicks = {
        "mark_centres_px": [project(x, y) for x, y in board["mark_centres_mm"]],
        "checker": [],
    }
    for i, colour in enumerate(
        ("#D7D5D1", "#252A29", "#A64B5B", "#427BA8", "#62AB6D", "#C9A34C", "#9762A1", "#688F97")
    ):
        x, y = 548 + 54 * (i % 2), 25 + 65 * (i // 2)
        draw.rectangle((x - 3, y - 3, x + 32, y + 32), fill=colour)
        clicks["checker"].append(
            {
                "reference_hex": colour,
                "corners_px": [[x, y], [x + 29, y], [x + 29, y + 29], [x, y + 29]],
            }
        )

    # Camera colour cast in linear light plus mild optical blur and read noise.
    encoded = np.asarray(image.filter(ImageFilter.GaussianBlur(1.0))) / 255.0
    linear = np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)
    tinted = linear * [1.16, 0.87, 0.76] + [0.005, 0.002, 0.001]
    signal = 255 * np.where(
        tinted <= 0.0031308, 12.92 * tinted, 1.055 * tinted ** (1 / 2.4) - 0.055
    )
    signal += np.random.default_rng(616).normal(0, 1.7, signal.shape)
    return Image.fromarray(np.rint(signal).clip(0, 255).astype("uint8")), clicks, expected


def test_cli_photo_and_recovery(tmp_path):
    def put(name, data):
        (tmp_path / name).write_text(json.dumps(data))

    def cli(*args):
        main(list(map(str, args)))

    put("materials.json", MATERIALS)
    plate = tmp_path / "plate"
    cli("board", tmp_path / "materials.json", plate)
    manifest = plate / "board.json"
    board = json.loads(manifest.read_text())
    photo, clicks, expected = synthetic_photo(board, TRUTH)
    photo.save(tmp_path / "photo.png")
    put("clicks.json", clicks)
    dbpath = tmp_path / "colours.json"
    cli(
        "extract",
        manifest,
        tmp_path / "photo.png",
        tmp_path / "clicks.json",
        dbpath,
        "--kind",
        "synthetic",
    )
    database = json.loads(dbpath.read_text())
    assert database["capture"]["kind"] == "synthetic"
    raw_errors, corrected_errors = [], []
    for key, row in database["measurements"].items():
        raw_errors.append(np.abs(np.array(row["raw_srgb8"]) - expected[key]).max())
        corrected_errors.append(np.abs(np.array(row["srgb8"]) - expected[key]).max())
        assert row["n_pixels"] > 300
    assert max(corrected_errors) <= 3
    assert np.median(raw_errors) >= 6
    cli("fit", dbpath, tmp_path / "fit.json")
    report = json.loads((tmp_path / "fit.json").read_text())
    curve = DistanceCurve(**report["accepted_curve"])
    assert curve.distance_at_td1_mm == pytest.approx(TRUTH.distance_at_td1_mm, abs=0.01)
    assert curve.exponent == pytest.approx(TRUTH.exponent, abs=0.03)
    assert report["training_count"] == 19 and len(report["holdout_ids"]) == 9
    assert report["fitted"]["median_de00"] < report["baseline"]["median_de00"] / 4
    print(
        json.dumps(
            {
                "truth": vars(TRUTH),
                "fit": vars(curve),
                "baseline": report["baseline"],
                "fitted": report["fitted"],
                "max_channel_error": int(max(corrected_errors)),
            }
        )
    )

    # Replacing all hold-out observations cannot change the candidate. Set them
    # to the baseline's exact predictions, making acceptance strictly impossible.
    predictor = BoardPredictor(board)
    for patch in board["patches"]:
        if patch["id"] in report["holdout_ids"]:
            rgb = predictor.predict(patch["bands"], DistanceCurve())
            database["measurements"][patch["id"]]["srgb8"] = rgb.astype(int).tolist()
    rejected = fit_database(database)
    assert rejected["candidate"] == report["candidate"]
    assert rejected["baseline"]["mean_de00"] == 0
    assert rejected["accepted_curve"] is None


def test_registered_print_geometry(tmp_path):
    from prism_filament_painting.calibration.plate import write_plate

    board = make_board(MATERIALS)
    write_plate(board, tmp_path)
    solids = []
    for idx in range(4):
        data = (tmp_path / f"slot-{idx}.stl").read_bytes()
        (count,) = struct.unpack_from("<I", data, 80)
        assert len(data) == 84 + 50 * count
        boxes = []
        for start in range(84, len(data), 12 * 50):
            edges = Counter()
            vertices = []
            for facet in struct.iter_unpack("<12fH", data[start : start + 12 * 50]):
                pts = np.reshape(facet[3:12], (3, 3))
                vertices.extend(pts)
                assert np.dot(np.cross(pts[1] - pts[0], pts[2] - pts[0]), facet[:3]) > 0
                for i in range(3):
                    edges[(tuple(pts[i]), tuple(pts[(i + 1) % 3]))] += 1
            assert all(n == 1 and edges[(v, u)] == 1 for (u, v), n in edges.items())
            boxes.append((np.min(vertices, axis=0), np.max(vertices, axis=0)))
        solids.append(boxes)
    assert len(solids[0]) == 1
    np.testing.assert_allclose(solids[0][0], [[0, 0, 0], [98, 62, 1.6]])
    assert len({str(p["bands"]) for p in board["patches"]}) == 28
    for x, y in board["mark_centres_mm"]:
        assert any(
            np.allclose(lo, [x - 1.6, 62 - y - 1.6, 1.6])
            and np.allclose(hi, [x + 1.6, 62 - y + 1.6, 1.92])
            for lo, hi in solids[1]
        )
    for patch in board["patches"]:
        x, y = patch["xy_mm"]
        z = 1.6
        for spool, layers in patch["bands"]:
            slot = next(i for i, m in enumerate(MATERIALS) if m["id"] == spool)
            h = layers * 0.08
            assert any(
                np.allclose(lo, [x, 62 - y - 10, z]) and np.allclose(hi, [x + 10, 62 - y, z + h])
                for lo, hi in solids[slot]
            )
            z += h


def test_without_checker_and_invalid_inputs():
    board = make_board(MATERIALS)
    photo, clicks, _ = synthetic_photo(board, TRUTH)

    def read(points):
        return extract(board, photo, points, source_kind="synthetic", photo_sha256="b" * 64)

    plain = read({"mark_centres_px": clicks["mark_centres_px"]})
    assert plain["capture"]["linear_rgb_affine"] is None
    for row in plain["measurements"].values():
        assert row["srgb8"] == np.rint(row["raw_srgb8"]).tolist()
    with pytest.raises(ValueError, match="clockwise"):
        read({"mark_centres_px": list(reversed(clicks["mark_centres_px"]))})
    with pytest.raises(ValueError, match="independently"):
        read({**clicks, "checker": [clicks["checker"][0]] * 6})
    tampered = copy.deepcopy(plain)
    tampered["board"]["patches"][-1]["role"] = "mixed"
    with pytest.raises(ValueError, match="layout"):
        fit_database(tampered)
    del plain["measurements"]["r1c1"]
    with pytest.raises(ValueError, match="cover exactly"):
        validate_database(plain)


def test_second_nonlinear_board_with_rounded_clicks():
    board = make_board(MATERIALS)
    truth = DistanceCurve(0.075, 1.38)
    image, clicks, _ = synthetic_photo(board, truth)
    clicks["mark_centres_px"] = np.rint(clicks["mark_centres_px"]).astype(int).tolist()
    data = extract(board, image, clicks, source_kind="synthetic", photo_sha256="a" * 64)
    report = fit_database(data)
    recovered = report["accepted_curve"]
    assert recovered["distance_at_td1_mm"] == pytest.approx(truth.distance_at_td1_mm, abs=0.006)
    assert recovered["exponent"] == pytest.approx(truth.exponent, abs=0.06)
    print(
        json.dumps(
            {
                "truth": vars(truth),
                "fit": recovered,
                "baseline": report["baseline"],
                "fitted": report["fitted"],
            }
        )
    )


def test_unidentifiable_wedges():
    # Three different colours with the same nominal TD cannot identify a power law.
    board = make_board([{**m, "td_mm": 3} for m in MATERIALS])
    model = BoardPredictor(board)
    data = {
        "schema": "front-lit-colours/v1",
        "board": board,
        "capture": {"kind": "synthetic", "photo_sha256": "0" * 64},
        "measurements": {
            p["id"]: {"srgb8": model.predict(p["bands"], TRUTH).astype(int).tolist()}
            for p in board["patches"]
        },
    }
    with pytest.raises(ValueError, match="distinct overlay TDs"):
        fit_database(data)


def synthetic_database(board):
    photo, clicks, _ = synthetic_photo(board, TRUTH)
    data = BytesIO()
    photo.save(data, format="PNG")
    return extract(board, photo, clicks, source_kind="synthetic", photo_sha256=sha256(data.getvalue()).hexdigest())


def test_capture_selection_and_prediction_sources(tmp_path, monkeypatch):
    from dataclasses import replace
    from datetime import datetime, timedelta, timezone

    from prism_filament_painting.core.forward import RenderConstants, build_forward_stack
    from prism_filament_painting.core.model import physical_filament
    from prism_filament_painting.core.prediction import install_capture

    board = make_board(MATERIALS)
    db = synthetic_database(board)
    spools = [physical_filament("Fixture", m["id"], m["hex"], m["td_mm"], m["id"]) for m in MATERIALS]
    settings = RenderConstants(total_layers=20)
    swaps = (4, 13, 17)

    def stack(slots=spools, geometry=settings):
        return build_forward_stack(slots, constants=geometry, swap_schedule=swaps)

    monkeypatch.setenv("PRISM_CALIBRATION_DIR", str(tmp_path))
    monkeypatch.delenv("PRISM_SYNTHETIC_CALIBRATION", raising=False)
    today = datetime.now(timezone.utc).date()
    yesterday = (today - timedelta(days=1)).isoformat()
    today = today.isoformat()
    source = tmp_path / "database.json"
    source.write_text(json.dumps(db))
    main(["install", str(source), str(tmp_path), "--captured-on", yesterday, "--valid-through", today, "--opaque-from-mm", "0.08"])
    plain = stack()
    assert set(plain.layer_sources) == {"formula"}  # printed mode refuses synthetic boards
    monkeypatch.setenv("PRISM_SYNTHETIC_CALIBRATION", "1")
    calibrated = stack()
    assert calibrated.layer_sources[4] == "measured"
    assert calibrated.layer_rgb[4].tolist() == db["measurements"]["r1c2"]["srgb8"]
    assert calibrated.layer_sources[8] == "interpolated"  # five yellow layers, bracketed by four and eight
    a = np.array(db["measurements"]["r1c4"]["srgb8"])
    b = np.array(db["measurements"]["r1c5"]["srgb8"])
    assert calibrated.layer_rgb[8].tolist() == np.rint(a + (b - a) / 4).tolist()
    assert calibrated.layer_sources[12] == "fitted"  # nine yellow layers: no extrapolation
    fit = fit_database(db)
    predictor = BoardPredictor(board)
    expected = predictor.predict([("yellow", 9)], DistanceCurve(**fit["accepted_curve"]))
    np.testing.assert_array_equal(calibrated.layer_rgb[12], expected)
    assert calibrated.layer_sources[13] == "fitted"  # an unrelated wedge cannot bracket a mixed stack
    for slots, geometry in [
        ([replace(spools[0], spool_id="new lot"), *spools[1:]], settings),
        ([spools[1], spools[0], *spools[2:]], settings),
        (spools, replace(settings, layer_height_mm=.12)),
        ([replace(spools[0], input_distance_mm=1.23), *spools[1:]], settings),
        ([replace(spools[0], hex="#999999"), *spools[1:]], settings),
    ]:
        assert set(stack(slots, geometry).layer_sources) == {"formula"}
    assert "measured" in stack([spools[0], *reversed(spools[1:])]).layer_sources

    for patch in board["patches"]:
        if patch["role"] == "holdout":
            rgb = predictor.predict(patch["bands"], DistanceCurve())
            db["measurements"][patch["id"]]["srgb8"] = rgb.tolist()
    latest = install_capture(tmp_path, db, today, today, opaque_from_mm=.08)
    assert json.loads(latest.read_text())["fit"]["accepted_curve"] is None
    rejected = stack()
    assert rejected.layer_sources[4] == "measured" and rejected.layer_sources[8] == "interpolated"
    assert rejected.layer_sources[12] == "formula"
    np.testing.assert_array_equal(rejected.layer_rgb[12], plain.layer_rgb[12])
    assert rejected.layer_rgb[13].tolist() != calibrated.layer_rgb[13].tolist()
    np.testing.assert_array_equal(rejected.layer_rgb[13], plain.layer_rgb[13])
    for h, result in [(4, calibrated), (8, calibrated), (13, calibrated), (13, rejected)]:
        print(h, result.layer_sources[h], result.layer_rgb[h].tolist())

    with pytest.raises(FileExistsError):
        install_capture(tmp_path, db, today, today, opaque_from_mm=.08)
    # Corrupt newest records fail closed, including a fit copied from a different capture.
    bundle = json.loads(latest.read_text())
    bundle["fit"]["capture"]["photo_sha256"] = "f" * 64
    latest.write_text(json.dumps(bundle))
    assert set(stack().layer_sources) == {"formula"}
    # Identical solid readings provide no measurable wedge slope. Exact and
    # interpolated readings still work without a fitted report.
    flat = copy.deepcopy(db)
    for reading in flat["measurements"].values():
        reading["srgb8"] = list(hex_to_rgb(MATERIALS[0]["hex"]))
    weak_root = tmp_path / "weak"
    weak_path = install_capture(weak_root, flat, today, today, opaque_from_mm=.08)
    assert json.loads(weak_path.read_text())["fit"] is None
    monkeypatch.setenv("PRISM_CALIBRATION_DIR", str(weak_root))
    assert stack().layer_sources[4:14:4] == ("measured", "interpolated", "formula")
    standard = tmp_path / "standard-backing"
    install_capture(standard, db, today, today)
    monkeypatch.setenv("PRISM_CALIBRATION_DIR", str(standard))
    assert set(stack().layer_sources) == {"formula"}  # 0.24 mm backing under the overlays
    thick = build_forward_stack(spools, constants=replace(settings, total_layers=30), swap_schedule=(21, 26, 28))
    assert thick.layer_sources[19:22] == ("formula", "measured", "measured")
    expired = tmp_path / "expired"
    install_capture(expired, db, yesterday, yesterday)
    monkeypatch.setenv("PRISM_CALIBRATION_DIR", str(expired))
    assert set(stack().layer_sources) == {"formula"}


def install_synthetic_photo(directory, monkeypatch):
    """A fresh offline board shared by the service integration tests."""
    from datetime import datetime, timezone
    from prism_filament_painting.core.prediction import install_capture

    board = make_board(MATERIALS)
    db = synthetic_database(board)
    day = datetime.now(timezone.utc).date().isoformat()
    install_capture(directory, db, day, day, opaque_from_mm=.08)
    monkeypatch.setenv("PRISM_CALIBRATION_DIR", str(directory))
    monkeypatch.setenv("PRISM_SYNTHETIC_CALIBRATION", "1")
    return [{"brand": "Board", "name": m["id"], "hex": m["hex"], "td": m["td_mm"], "spool_id": m["id"]}
            for m in MATERIALS]
