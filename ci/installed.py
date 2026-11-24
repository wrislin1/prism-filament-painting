"""Check required suite coverage and installed command and route names."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from xml.etree import ElementTree

from naming import check_names


def verify(xml, target, tests):
    cases = ElementTree.parse(xml).findall(".//testcase")
    passed = {case.attrib["name"] for case in cases}
    required = {"test_first_use", "test_four_slot_geometry", "test_calibration_entry_point",
                "test_stdio_entry_point", "test_cpu_optimisation"}
    if not required <= passed or any(
        case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")
    ):
        raise ValueError("incomplete installed suite")

    sys.path.insert(0, str(tests))
    from test_installed import command, running

    for cli in ("prism-filament-painting", "prism-calibrate", "prism-palette-mcp"):
        check_names(cli, subprocess.check_output([command(cli), "--help"]))
    with tempfile.TemporaryDirectory() as state, running(Path(state)) as client:
        for route in ("/openapi.json", "/", "/filaments", "/studio", "/studio/flat",
                      "/opacity-wedges", "/suggestions"):
            check_names(route, client.get(route).raise_for_status().content)
    report = {"installed_from": target, "python": sys.version.split()[0],
              "passed": len(cases), "skipped": 0, "features": sorted(required)}
    (xml.parent / "installation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    verify(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
