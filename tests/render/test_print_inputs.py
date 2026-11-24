"""An independent plate calibration and user slice report require only local inputs."""
import json
from zipfile import ZipFile

import pytest

from prism_filament_painting.render.print_time import calibrated_time, load_calibration
from prism_filament_painting.render.slice_report import SliceReport, project_key, slice_for_project


def test_time_from_supplied_rates(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_TIME_CALIBRATION", "")
    assert calibrated_time(load_calibration("trial"), 60, 3)["minutes"] is None
    path = tmp_path / "rates.json"
    rates = dict(start_minutes=2, area_rate=0.5, reference_layers=3, change_minutes=2,
                 reference_changes=1, flush_rate=0.25)
    path.write_text(json.dumps({"trial": rates}))
    monkeypatch.setenv("PRISM_TIME_CALIBRATION", str(path))
    result = calibrated_time(load_calibration("trial"), 60, 3, 4)
    assert result["minutes"] == 19 and result["deposition_min"] == 10
    assert result["minutes_range"] is None
    rates.update(min_margin_minutes=1, relative_margin=0.1)
    path.write_text(json.dumps({"trial": rates}))
    assert calibrated_time(load_calibration("trial"), 60, 3, 4)["minutes_range"] == pytest.approx([17.1, 20.9])
    assert load_calibration("other") is None
    rates["reference_changes"] = 0
    path.write_text(json.dumps({"trial": rates}))
    with pytest.raises(ValueError, match="reference counts"):
        load_calibration("trial")


def test_only_a_matching_user_report_can_supply_an_observation(tmp_path):
    archive = tmp_path / "plate.3mf"
    with ZipFile(archive, "w") as output:
        output.writestr("3D/3dmodel.model", "<model/>")
    key = project_key(archive)
    report = SliceReport(project_key=key, slicer="Example", minutes=8.5, flushed_g=0.25,
                         tower_g=0.5, passes=[[1], [2, 1]])
    run = dict(selected_plan="full", plans={"full": {"layer_usage": [[1], [1, 2]]}}, backing_slot=1)
    assert slice_for_project(key, None, run) is None
    assert slice_for_project("0" * 64, report, run) is None
    observed = slice_for_project(key, report, run)
    assert (observed["minutes"], observed["changes"], observed["waste_g"]) == (8.5, 2, 0.75)
    with pytest.raises(ValueError, match="occupied slots"):
        slice_for_project(key, report.model_copy(update={"passes": [[2]]}), run)
