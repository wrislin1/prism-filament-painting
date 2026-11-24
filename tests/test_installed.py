"""Run against an installed wheel from a temporary directory outside the source tree."""

import asyncio
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, distribution
from io import BytesIO
import json
from pathlib import Path
import re
import struct
import subprocess
import sysconfig
import time
from zipfile import ZipFile

import pytest

try:
    distribution("prism-filament-painting")
except PackageNotFoundError:
    pytest.skip("This suite exercises the installed distribution.", allow_module_level=True)

import httpx
from PIL import Image

from prism_filament_painting.palette.sample_inputs import sample_image, spool_list


def command(name):
    # Resolve from this interpreter's environment, independent of PATH.
    import shutil
    path = shutil.which(name, path=sysconfig.get_path("scripts"))
    assert path, f"Install the wheel to provide {name}"
    return path


@contextmanager
def running(directory):
    log = directory / "server.log"
    with log.open("w") as output:
        server = subprocess.Popen([command("prism-filament-painting"), "--port", "0", "--data-dir", "state"],
                                  cwd=directory, stdout=output, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            lines = log.read_text()
            address = re.search(r"Open Prism at (http://127\.0\.0\.1:\d+)", lines)
            if address:
                break
            assert server.poll() is None, lines
            time.sleep(0.1)
        else:
            pytest.fail("Launcher failed to bind its loopback port.\n" + lines)
        with httpx.Client(base_url=address[1], timeout=60, trust_env=False) as client:
            yield client
    finally:
        server.terminate()
        server.wait(timeout=10)


def test_first_screen_and_sample(tmp_path):
    with running(tmp_path) as client:
        assert (tmp_path / "server.log").read_text().splitlines() == [f"Open Prism at {str(client.base_url).rstrip('/')}"]
        home = client.get("/").raise_for_status().text
        assert "Your library is empty" in home and 'href="/filaments#spool"' in home
        assert 'action="/sample"' in home and 'id="rank"' not in home
        response = client.post("/sample")
        assert response.status_code == 303
        home = client.get(response.headers["location"]).raise_for_status().text
        assert 'id="rank"' in home and 'id="use-sample" checked' in home
        assert 'action="/sample"' not in home
        rows = client.get("/api/filaments").json()
        assert len(rows) == 3 and all(row["brand"] == "Paper Orbit" for row in rows)
        client.post("/sample")
        assert client.get("/api/filaments").json() == rows
        stale = client.get("/studio?project_id=missing")
        assert stale.status_code == 404 and "This print could not be opened" in stale.text
        picture = client.get("/sample/picture.png").raise_for_status()
        assert Image.open(BytesIO(picture.content)).tobytes() == sample_image().tobytes()


def test_first_use(tmp_path):
    from prism_filament_painting import launcher
    installed = distribution("prism-filament-painting")
    assert Path(launcher.__file__).resolve() == installed.locate_file("prism_filament_painting/launcher.py").resolve()
    assert set(installed.metadata.get_all("Provides-Extra")) == {"optimize", "catalogues", "mcp"}
    buf = BytesIO()
    ramp = sample_image()
    ramp.save(buf, format="PNG")
    with running(tmp_path) as client:
        for url in ("/", "/filaments", "/studio", "/studio/flat", "/opacity-wedges", "/suggestions"):
            page = client.get(url).raise_for_status().text
            assert "/static/prism.css" in page
            assert "mask-foreground" not in page and "depth-estimate" not in page
        assert "system-ui" in client.get("/static/prism.css").raise_for_status().text
        scripts = Path(launcher.__file__).with_name("workspace") / "static"
        for path in scripts.rglob("*.js"):
            asset = client.get("/static/" + path.relative_to(scripts).as_posix()).raise_for_status().text
            assert all(route not in asset for route in ("/studio/depth/", "/studio/flat/mask", "/model3d"))
        assert client.get("/api/filaments").json() == []
        for i, (name, color) in enumerate([("Night Reed", "#293745"), ("Lemon Thread", "#E4C86C")]):
            client.post("/api/filaments", json={"brand": "Invented", "name": name, "material": "PLA",
                                               "hex": color, "td": 1.25 + i}).raise_for_status()
            if i == 0:
                home = client.get("/").raise_for_status().text
                assert 'max="1" value="1"' in home and 'action="/sample"' not in home
                client.post("/sample")
                assert len(client.get("/api/filaments").json()) == 1
        assert len(client.get("/api/filaments").json()) == 2
        ranked = client.post("/api/rank", files={"image": ("ramp.png", buf.getvalue(), "image/png")},
                             data={"slots": 2}).raise_for_status().json()["candidates"][0]
        assert "/static/studio/main.js" in client.get(ranked["studio_url"]).raise_for_status().text
        saved = client.post("/studio/projects", json={"state": ranked["state"]}).raise_for_status().json()
    # Restart from the same user directory to rule out in-memory persistence.
    with running(tmp_path) as client:
        state = client.get("/studio/projects/" + saved["project"]["id"]).raise_for_status().json()["state"]
        assert state == saved["state"] and state["version"] == 2
        image = client.get("/studio/projects/" + saved["project"]["id"] + "/image").raise_for_status()
        assert Image.open(BytesIO(image.content)).convert("RGB").tobytes() == ramp.tobytes()
        result = client.post("/studio/generate", json={"state": state}).raise_for_status().json()
        mesh = client.get(result["stl_url"]).raise_for_status().content
        triangles, = struct.unpack_from("<I", mesh, 80)
        assert triangles > 0 and len(mesh) == 84 + 50 * triangles
        swaps = client.get(result["swap_url"]).raise_for_status().text
        schedule = state["swap_schedule"]
        assert result["swap_schedule"] == schedule and len(schedule) == 1
        layer, = schedule
        z = layer * state["layer"]["layer_height_mm"]
        assert f"At layer {layer} (Z = {z:.2f} mm)" in swaps
        assert Image.open(BytesIO(client.get(result["output_url"]).raise_for_status().content)).size == ramp.size
        print(f"2 spools, {ramp.size} ramp, reopened v2 project, {triangles} triangles, swap {layer} at {z:.2f} mm")


def test_four_slot_geometry(tmp_path):
    spools = [{"brand": "Invented", "name": name, "material": "PLA", "hex": color, "td": 1.5 + i}
              for i, (name, color) in enumerate([("Shell", "#E2E2D8"), ("Bay", "#283953"),
                                                 ("Ochre", "#C68D35"), ("Leaf", "#729A68")])]
    buf = BytesIO()
    sample_image().save(buf, format="PNG")
    with running(tmp_path) as client:
        state = client.post("/studio/flat/run", files={"image": ("ramp.png", buf.getvalue())}, data={
            "palette": json.dumps({"filaments": spools}), "options": json.dumps({"profile": "u1", "long_edge_cells": 5}),
        }).raise_for_status().json()
        saved = client.post("/studio/projects", json={"state": state}).raise_for_status().json()
        assert client.get("/studio/projects/" + saved["project"]["id"]).json()["state"] == saved["state"]
        flat = state["flat"]
        assert flat["downloads"]["printer_project"] is None and flat["cost"]["selected"]["minutes"] is None
        archive = client.get(flat["downloads"]["geometry"]).raise_for_status()
        with ZipFile(BytesIO(archive.content)) as output:
            assert "3D/3dmodel.model" in output.namelist()
            assert "Metadata/project_settings.config" not in output.namelist()


        # An independently supplied flush matrix must reach the configured project and estimate.
        from render.test_slicer_settings import preset
        settings = preset(flat=True)
        settings.update(filament_colour=[s["hex"] for s in spools], filament_map=["1"] * 4,
                        filament_settings_id=[s["name"] for s in spools], filament_diameter=["1.75"] * 4,
                        filament_flow_ratio=["1"] * 4)
        loadout = dict(printer_model="Bambu Lab A1", filament_colour=settings["filament_colour"],
                       filament_density=[1.2] * 4, flush_multiplier=[0.8],
                       flush_volumes_matrix=[0 if i % 5 == 0 else 60 + i for i in range(16)])
        data = dict(palette=json.dumps({"filaments": spools}),
                    options=json.dumps(dict(profile="a1", long_edge_cells=5, slicer_settings=loadout)))
        geometry = flat["run"]["geometry"]
        settings.update(initial_layer_print_height=str(geometry["backing_layer_heights_mm"][0]),
                        layer_height=str(geometry["picture_layer_height_mm"]))

        def configured():
            return client.post("/studio/flat/run", data=data, files={
                "image": ("ramp.png", buf.getvalue()), "project_settings": ("settings.json", json.dumps(settings))})

        result = configured().raise_for_status().json()["flat"]
        with ZipFile(BytesIO(client.get(result["downloads"]["printer_project"]).raise_for_status().content)) as project:
            embedded = json.loads(project.read("Metadata/project_settings.config"))
            for key in ("filament_density", "flush_multiplier", "flush_volumes_matrix"):
                assert list(map(float, embedded[key])) == loadout[key]
        settings["filament_density"] = [1.6] * 4
        assert configured().status_code == 422

def test_calibration_entry_point(tmp_path):
    from calibration.test_calibration import MATERIALS, TRUTH, synthetic_photo

    def cli(*args):
        subprocess.run([command("prism-calibrate"), *args], cwd=tmp_path, check=True, timeout=60)

    (tmp_path / "materials.json").write_text(json.dumps(MATERIALS))
    cli("board", "materials.json", "plate")
    board = json.loads((tmp_path / "plate/board.json").read_text())
    assert len(list((tmp_path / "plate").glob("*.stl"))) == 4
    photo, marks, _ = synthetic_photo(board, TRUTH)
    photo.save(tmp_path / "capture.png")
    (tmp_path / "marks.json").write_text(json.dumps(marks))
    cli("extract", "plate/board.json", "capture.png", "marks.json", "measurements.json", "--kind", "synthetic")
    cli("fit", "measurements.json", "fit.json")
    report = json.loads((tmp_path / "fit.json").read_text())
    assert report["accepted_curve"] and report["training_count"] == 19 and len(report["holdout_ids"]) == 9
    cli("install", "measurements.json", "calibration", "--captured-on", "2025-01-01", "--valid-through", "2025-12-31")
    assert list((tmp_path / "calibration/synthetic").glob("*.json"))


def test_stdio_entry_point(tmp_path):
    pytest.importorskip("mcp", reason="The stdio example requires the mcp extra.")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    library = tmp_path / "spools.csv"
    image = tmp_path / "ramp.png"
    library.write_text(spool_list())
    sample_image().save(image)

    async def exercise():
        params = StdioServerParameters(command=command("prism-palette-mcp"), args=["--library", str(library)], cwd=tmp_path)
        async with stdio_client(params) as (reader, writer), ClientSession(reader, writer) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            assert {t.name for t in tools} == {"list_filaments", "rank_candidate_palettes", "render_palette_preview"}
            assert all(t.annotations.readOnlyHint for t in tools)
            inventory = await session.call_tool("list_filaments", {})
            assert json.loads(inventory.content[0].text)["total"] == 3
            ranked = await session.call_tool("rank_candidate_palettes", {"image_path": str(image), "candidates": 1})
            assert not ranked.isError
            best = json.loads(next(item.text for item in ranked.content if item.type == "text" and "swap_layers" in item.text))
            preview = await session.call_tool("render_palette_preview", {"image_path": str(image),
                "filament_ids": [f["id"] for f in best["filaments"]], "swap_layers": best["swap_layers"]})
            assert not preview.isError and preview.content[-1].data == ranked.content[-1].data

    asyncio.run(asyncio.wait_for(exercise(), timeout=60))


def test_catalogue_notices_travel_with_export(tmp_path):
    from prism_filament_painting.palette import store
    from prism_filament_painting.palette.catalogues import SOURCES
    from prism_filament_painting.workspace.storage import LocalState

    state = LocalState(tmp_path / "state")
    with running(tmp_path) as client:
        empty = client.get("/filaments/export").raise_for_status().json()
        assert empty["filaments"] == []
        assert set(empty["catalogue_notices"]) == set(SOURCES)
        page = client.get("/filaments").raise_for_status().text
        with state.connect() as conn:
            for key in SOURCES:
                store.upsert_public_filament_with_status(
                    conn, external_id=f"{key}:invented", brand="Invented", name=key,
                    material="PLA", hex="#456789", td=2)
        data = client.get("/filaments/export").raise_for_status().json()
        assert len(data["filaments"]) == len(SOURCES)
        for key, (repo, revision) in SOURCES.items():
            link = f"/filaments/licenses/{key}"
            assert link in page
            notice = data["catalogue_notices"][key]
            assert notice["url"] == f"https://github.com/{repo}/blob/{revision}/LICENSE"
            assert notice["text"] == client.get(link).raise_for_status().text
            assert "Copyright" in notice["text"]


def test_cpu_optimisation():
    torch = pytest.importorskip("torch", reason="CPU optimisation requires the optimize extra.")
    from prism_filament_painting.core.model import Filament
    from prism_filament_painting.render.engine.optimize import optimize_height_field
    from prism_filament_painting.render.engine.params import LayerParams, MeshParams, OptimizeParams

    torch.set_num_threads(1)
    spools = [Filament("Invented", "Ink", "#282433", 1.5), Filament("Invented", "Pear", "#DACA65", 2.5)]
    result = optimize_height_field(
        sample_image().resize((8, 6)), spools, layer_params=LayerParams(total_layers=6),
        mesh_params=MeshParams(), params=OptimizeParams(device="cpu", allow_cpu=True, iterations=2),
    )
    assert result.report["used_optimizer"] and result.report["device"] == "cpu"
    assert result.height_field.shape == (6, 8)
