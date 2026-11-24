"""Rebuild the public inputs from source and use them with the inventory and ranker."""
import hashlib
import json
from importlib.resources import files

import pytest
from PIL import Image

from prism_filament_painting.palette.enhance import EnhanceParams, apply_enhancement
from prism_filament_painting.palette.inventory import parse_inventory
from prism_filament_painting.palette.opacity_model import estimate_opacity, fit_opacity
from prism_filament_painting.palette.ranking import rank_palettes, score_palette
from prism_filament_painting.palette.sample_inputs import sample_image, spool_list
from prism_filament_painting.core.forward import build_forward_stack, render_preview


def test_example_inventory_and_image(tmp_path):
    path = tmp_path / "spools.csv"
    path.write_bytes(spool_list().encode("ascii"))
    assert path.read_bytes() == files("prism_filament_painting.palette").joinpath("data/example_spools.csv").read_bytes()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "8ca9914c59fc2a07b7bacccef310696a44c3fd04678f6ee3b27b47aafa44f101"
    image = sample_image()
    assert hashlib.sha256(image.tobytes()).hexdigest() == "028f446240171548796d740f45aa940becd7d29bbd3a5dd4fc7e12248e8ac41a"
    png = tmp_path / "ramp.png"
    image.save(png)
    with Image.open(png) as decoded:
        assert decoded.size == (31, 19) and not decoded.info
        assert decoded.tobytes() == image.tobytes()
        assert parse_inventory() == []
        with pytest.raises(ValueError, match="inventory must not be empty"):
            rank_palettes(decoded, [], palette_size=2, max_candidates=3)
        spools = parse_inventory(path, owned_only=True)
        assert len(spools) == 3
        best = rank_palettes(decoded, spools, palette_size=3, max_candidates=2)[0]
        rescored = score_palette(decoded, best.filaments)
        assert (best.score, best.swap_schedule) == (rescored.score, rescored.swap_schedule)
        stack = build_forward_stack(best.filaments, swap_schedule=best.swap_schedule)
        preview = render_preview(decoded, stack)
        assert hashlib.sha256(preview.tobytes()).hexdigest() == "7c8841412cf4dcf391b2575e2a0ffdebeae73f0c47ba87d7023b216ca2ac9a83"


def test_black_point_clips_generated_shadows():
    image = Image.frombytes("L", (256, 1), bytes(range(256))).convert("RGB")
    adjusted = apply_enhancement(image, EnhanceParams(black_point=0.3))
    assert image.convert("L").histogram()[0] == 1
    assert adjusted.convert("L").histogram()[0] == 78


def test_missing_and_user_fitted_opacity(monkeypatch, tmp_path):
    monkeypatch.setenv("PRISM_OPACITY_CALIBRATION", "")
    path = tmp_path / "one.csv"
    path.write_text("brand,name,hex,td\nPaper Orbit,Ochre Dial,#CA9137,\n")
    assert estimate_opacity("#CA9137", "matte") is None
    with pytest.raises(ValueError, match="no opacity estimate available"):
        parse_inventory(path)

    fit = tmp_path / "fit.json"
    measurements = [("#CA9137", "matte", 2.75), ("#6979A2", "", 2.75)]
    fit.write_text(json.dumps(fit_opacity(measurements)))
    monkeypatch.setenv("PRISM_OPACITY_CALIBRATION", str(fit))
    spool, = parse_inventory(path)
    assert (spool.td, spool.opacity_level) == (2.75, "estimated")
    fit.write_text("[1,2]")
    with pytest.raises(ValueError, match="seven finite"):
        parse_inventory(path)
