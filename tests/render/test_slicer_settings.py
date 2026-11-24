"""User-owned project settings, independent of any installed printer profiles."""

from io import BytesIO
import json
from types import SimpleNamespace
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pytest

from prism_filament_painting.render.engine.classic_project import write_classic_project
from prism_filament_painting.render.engine.slicer_settings import read_project_settings
from prism_filament_painting.render.flat_project import write_flat_project


COLOURS = ["#4B786D", "#BE956D"]


def preset(*, separate=False, flat=False):
    nozzles = 2 if separate else 1
    return dict(printer_settings_id="My printer", print_settings_id="My layers",
                filament_settings_id=["My first spool", "My second spool"], filament_colour=COLOURS,
                filament_map=["1", "2" if separate else "1"], filament_diameter=["1.75"] * 2,
                filament_flow_ratio=["0.98", "1.02"], nozzle_diameter=["0.4"] * nozzles,
                min_layer_height=["0.05"] * nozzles, max_layer_height=["0.28"] * nozzles,
                printable_area=["10x20", "150x20", "150x160", "10x160"], printable_height="120",
                enable_prime_tower="0", initial_layer_print_height=".2" if flat else ".08",
                layer_height=".12" if flat else ".08", machine_start_gcode="unused local text")


def classic():
    # A tetrahedron with its top at layer 12 and one change at layer 5.
    vertices = np.array([[0, 0, 0], [8, 0, 0], [0, 8, 0], [0, 0, .96]])
    mesh = SimpleNamespace(vertices=vertices, bounds=np.array([vertices.min(axis=0), vertices.max(axis=0)]),
                           faces=[(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)])
    return SimpleNamespace(mesh=mesh, swap_schedule=(5,), layer_params=SimpleNamespace(
        layer_height_mm=.08, total_layers=12),
        output_filaments=[SimpleNamespace(brand="Sample", name=str(i), hex=c) for i, c in enumerate(COLOURS)])


def flat_run(separate):
    return dict(schema_version=1, profile="u1" if separate else "a1", backing_slot=1,
                geometry=dict(rows=1, columns=2, picture_layers=2, width_mm=2., height_mm=1.,
                              total_height_mm=.44, cell_mm=1., line_width_mm=.45, nozzle_mm=.4,
                              orientation="face-up", row_order="top-to-bottom", fill_percent=100,
                              backing_layer_heights_mm=[.2], picture_layer_height_mm=.12),
                slots=[dict(slot=i, hex=c, brand="Sample", name=str(i)) for i, c in enumerate(COLOURS, 1)],
                recipe_table=[dict(slots=[1, 2]), dict(slots=[2, 1])], cell_recipe_ids=[[0, 1]])


def test_json_and_project_read_the_same_settings():
    payload = json.dumps(preset()).encode()
    project = BytesIO()
    with ZipFile(project, "w") as archive:
        archive.writestr("Metadata/project_settings.config", payload)
        archive.writestr("ignored.txt", "unrelated")
    assert read_project_settings(project) == read_project_settings(BytesIO(payload))


@pytest.mark.parametrize("data", [b"[]", b"not JSON", b" " * (4 * 1024 * 1024 + 1)])
def test_unusable_settings_file(data):
    with pytest.raises(ValueError, match="Export"):
        read_project_settings(BytesIO(data))


def test_missing_project_member():
    project = BytesIO()
    with ZipFile(project, "w") as archive:
        archive.writestr("3D/3dmodel.model", "")
    with pytest.raises(ValueError, match="one settings file"):
        read_project_settings(project)


@pytest.mark.parametrize(("changes", "message"), [
    (dict(filament_colour=COLOURS[::-1]), "slot order"), (dict(filament_colour=COLOURS[:1]), "one entry for filament_settings_id"),
    (dict(filament_settings_id=[""]), "2 entries"), (dict(filament_map=["2", "1"]), "Assign"),
    (dict(filament_flow_ratio=["1", "0"]), "filament_flow_ratio"),
    (dict(filament_flow_ratio=None), "filament_flow_ratio"),
    (dict(nozzle_diameter=["0.4", "0.4"]), "nozzle count"),
    (dict(print_settings_id=None), "process"), (dict(initial_layer_print_height=".2"), "0.08 mm"),
    (dict(layer_height=True), "layer_height"), (dict(max_layer_height=["NaN"]), "max_layer_height"),
    (dict(min_layer_height=[".1"]), "every planned layer"), (dict(printable_height=".1"), "full print height"),
    (dict(printable_area=["0x0", "1x1", "2x2", "3x3"]), "rectangular"),
    (dict(flush_volumes_matrix=["0", "1", "2"]), "4 entries"),
])
def test_reject_before_opening_a_project(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        write_classic_project(tmp_path, classic(), project_settings=preset() | changes)
    assert list(tmp_path.iterdir()) == []


def test_classic_colour_changes_pause_and_explicit_fallback(tmp_path):
    result = classic()
    assert write_classic_project(tmp_path, result) == {}
    files = write_classic_project(tmp_path, result, project_settings=preset())
    assert set(files) == {"project_url", "pause_project_url"}
    for kind, filename in files.items():
        pause = kind == "pause_project_url"
        with ZipFile(tmp_path / filename) as archive:
            config = json.loads(archive.read("Metadata/project_settings.config"))
            layer, = ET.fromstring(archive.read("Metadata/custom_gcode_per_layer.xml")).findall("plate/layer")
        assert config["filament_colour"] == (COLOURS[:1] if pause else COLOURS)
        assert list(map(float, config["filament_flow_ratio"])) == ([.98] if pause else [.98, 1.02])
        assert config["printer_settings_id"] == "My printer"
        assert "machine_start_gcode" not in config
        assert (layer.get("top_z"), layer.get("extruder"), layer.get("type")) == (
            "0.4", "1" if pause else "2", "1" if pause else "2")


def test_single_loaded_spool_offers_manual_changes(tmp_path):
    source = preset()
    for key in ("filament_colour", "filament_map", "filament_settings_id", "filament_flow_ratio", "filament_diameter"):
        source[key] = source[key][:1]
    assert write_classic_project(tmp_path, classic(), project_settings=source) == {"pause_project_url": "0.pause.3mf"}


@pytest.mark.parametrize("separate", [False, True])
def test_flat_slot_maps_heights_and_geometry_only_fallback(tmp_path, separate):
    run = flat_run(separate)
    blank = tmp_path / "blank"
    maps = write_flat_project(run, blank, public_geometry=True)
    assert maps["settings_note"] and not maps["embedded_printer_settings"]
    with ZipFile(blank / "geometry.3mf") as archive:
        assert "Metadata/project_settings.config" not in archive.namelist()
    dest = tmp_path / "configured"
    configured = write_flat_project(run, dest, project_settings=preset(flat=True, separate=separate))
    assert configured["embedded_printer_settings"]
    assert configured["layers"] == maps["layers"]
    with ZipFile(dest / "flat.3mf") as archive:
        model = ET.fromstring(archive.read("Metadata/model_settings.config"))
        mapping = model.find("plate/metadata[@key='filament_maps']").get("value")
        assert mapping == ("1 2" if separate else "1 1")
        settings = json.loads(archive.read("Metadata/project_settings.config"))
        assert settings["filament_map"] == mapping.split()
        ranges = ET.fromstring(archive.read("Metadata/layer_config_ranges.xml"))
        assert [float(r.get("max_z")) for r in ranges.findall("object/range")] == pytest.approx(
            [layer["top_z_mm"] for layer in maps["layers"]])
    with pytest.raises(ValueError, match="0.4 mm nozzle"):
        write_flat_project(run, tmp_path / "invalid", project_settings=preset(flat=True, separate=separate) |
                           {"nozzle_diameter": ["0.6"] * (2 if separate else 1)})
    assert not (tmp_path / "invalid").exists()


def test_tower_placement_uses_the_exported_bed_origin(tmp_path):
    source = preset() | dict(enable_prime_tower="1", wipe_tower_x=["14"], wipe_tower_y=["25"],
                             prime_tower_brim_width="2", prime_tower_width="35")
    run = classic()
    write_classic_project(tmp_path, run, project_settings=source)
    with ZipFile(tmp_path / "0.3mf") as archive:
        item = ET.fromstring(archive.read("3D/3dmodel.model")).find("{*}build/{*}item")
    dx, dy, _ = map(float, item.get("transform").split()[-3:])
    assert run.mesh.bounds[0, 0] + dx >= 53
    assert run.mesh.bounds[1, 0] + dx <= 150
    assert run.mesh.bounds[0, 1] + dy >= 20
    with pytest.raises(ValueError, match="Move the prime tower"):
        write_classic_project(tmp_path, run, project_settings=source | {"wipe_tower_y": ["200"]})


def test_change_above_the_model_is_rejected(tmp_path):
    run = classic()
    run.swap_schedule = (50,)
    with pytest.raises(ValueError, match="height of the generated model"):
        write_classic_project(tmp_path, run, project_settings=preset())
    assert list(tmp_path.iterdir()) == []
