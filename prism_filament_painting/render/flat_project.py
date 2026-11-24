"""Export a stored flat run as a multipart 3MF and an offline deposition map."""

from io import BufferedWriter
import json
from shutil import copyfileobj
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET
from xml.sax.saxutils import quoteattr
from zipfile import ZIP_DEFLATED, ZipFile

from .engine.classic_project import _CONTENT_TYPES, _RELATIONSHIPS
from .engine.slicer_settings import prepare_project, project_variant
from .flat_layout import expected_maps, layout
from .flat_mesh import compact_grid, slot_mesh

ASSEMBLY_ID = 5  # IDs 1 through 4 identify the physical toolhead parts.


def _flat_process(g, heights):
    # Two outer beads should meet at the centre of a minimum-size island.
    # Keep the nominal run width for larger cells and cap the width at 125% of nozzle diameter.
    outer_width = g["line_width_mm"]
    if outer_width <= g["cell_mm"] / 2 <= 1.25 * g["nozzle_mm"]:
        outer_width = g["cell_mm"] / 2
    return dict(
        initial_layer_print_height=str(heights[0]), layer_height=str(g["picture_layer_height_mm"]),
        sparse_infill_density="100%", sparse_infill_pattern="rectilinear",
        line_width=str(g["line_width_mm"]), wall_generator="classic",
        outer_wall_line_width=str(outer_width), inner_wall_line_width=str(g["line_width_mm"]),
        enable_support="0", brim_type="no_brim", wipe_tower_wall_type="rectangle",
        wipe_tower_cone_angle="0",
    )


def _name(spool):
    return f"Slot {spool['slot']} - {spool['brand']} {spool['name']}"


def _write_geometry(archive, grid, axes, spools, origin, *, orca_format):
    with archive.open("3D/3dmodel.model", "w") as raw, BufferedWriter(raw) as out:
        def xml(value):
            out.write(value.encode("utf-8"))

        application = ('<metadata name="Application">BambuStudio-01.10.01.50</metadata>'
                       '<metadata name="OrcaSlicer">2.4.2</metadata>' if orca_format else
                       '<metadata name="Application">Snapmaker_Orca-2.4.0</metadata>')
        xml('<model unit="millimeter" xml:lang="en-US" '
            'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
            'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">'
            f'{application}'
            '<metadata name="BambuStudio:3mfVersion">1</metadata><resources>')
        for spool in spools:
            slot = spool["slot"]
            vertices, faces = slot_mesh(grid, axes, slot)
            xml(f'<object id="{slot}" type="model" name={quoteattr(_name(spool))}><mesh><vertices>')
            for x, y, z in vertices:
                xml(f'<vertex x="{x:.12g}" y="{y:.12g}" z="{z:.12g}"/>\n')
            xml('</vertices><triangles>')
            for a, b, c in faces:
                xml(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>\n')
            xml('</triangles></mesh></object>')
        xml(f'<object id="{ASSEMBLY_ID}" type="model" name="Flat print"><components>')
        for spool in spools:
            xml(f'<component objectid="{spool["slot"]}"/>')
        xml('</components></object></resources><build>'
            f'<item objectid="{ASSEMBLY_ID}" transform="1 0 0 0 1 0 0 0 1 '
            f'{origin[0]:.12g} {origin[1]:.12g} 0"/></build></model>')


def _model_settings(spools, slots, profile):
    root = ET.Element("config")

    def metadata(parent, key, value):
        ET.SubElement(parent, "metadata", key=key, value=str(value))

    obj = ET.SubElement(root, "object", id=str(ASSEMBLY_ID))
    metadata(obj, "name", "Flat print")
    for spool in spools:
        part = ET.SubElement(obj, "part", id=str(spool["slot"]), subtype="normal_part")
        metadata(part, "name", _name(spool))
        metadata(part, "extruder", spool["slot"])
    plate = ET.SubElement(root, "plate")
    metadata(plate, "plater_id", 1)
    # Part extruder selects the filament. The plate map assigns filaments to nozzles.
    nozzle_map = ["1" if profile == "a1" else str(s["slot"]) for s in slots]
    metadata(plate, "filament_maps", " ".join(nozzle_map))
    instance = ET.SubElement(plate, "model_instance")
    metadata(instance, "object_id", ASSEMBLY_ID)
    metadata(instance, "instance_id", 0)
    metadata(instance, "identify_id", 1)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _height_ranges(heights):
    root = ET.Element("objects")
    obj = ET.SubElement(root, "object", id="1")  # One-based build item index.
    z = 0.
    for height in heights:
        span = ET.SubElement(obj, "range", min_z=f"{z:.12g}", max_z=f"{z + height:.12g}")
        ET.SubElement(span, "option", opt_key="layer_height").text = str(height)
        z += height
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def merge_flush_settings(project_settings, slicer_settings):
    """Apply the same directional costs to the print project and its estimate."""
    if project_settings is None or slicer_settings is None:
        return project_settings
    project_settings = project_settings.copy()
    for key in ("flush_volumes_matrix", "flush_multiplier", "filament_density"):
        if key not in slicer_settings:
            continue
        if key in project_settings:
            try:
                values = [float(v) for v in project_settings[key]]
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Export numeric {key} values for the loaded slots.") from exc
            if values != list(map(float, slicer_settings[key])):
                raise ValueError(f"Use matching {key} values in the project and flush settings.")
        project_settings[key] = slicer_settings[key]
    return project_settings


def write_flat_project(run, destination, *, project_settings=None, public_geometry=False,
                       settings_provider=None, estimate_report=None):
    """Write a geometry-only project by default. Optional settings must match the saved run."""
    heights, planes = layout(run)
    g = run["geometry"]
    if run["profile"] not in ("a1", "u1"):
        raise ValueError("Choose a single-nozzle or separate-tool flat print profile.")
    if project_settings is not None:
        settings, origin = prepare_project(
            project_settings, [s["hex"] for s in run["slots"]], first=heights[0],
            regular=g["picture_layer_height_mm"], heights=heights, top_z=sum(heights),
            size=(g["width_mm"], g["height_mm"]), separate_tools=run["profile"] == "u1",
            nozzle=g["nozzle_mm"])
        settings = project_variant(settings)
        process = _flat_process(g, heights)
        settings.update(process)
        overrides = settings["different_settings_to_system"]
        overrides[0] = ";".join(dict.fromkeys([*overrides[0].split(";"), *process]))
        settings_note = "Apply your matching printer and filament presets before slicing the 3MF."
    elif settings_provider is not None:
        settings, origin, settings_note = settings_provider(run, heights)
    else:
        settings, origin = None, (0., 0.)
        settings_note = "Open the multipart geometry and apply your printer, slot map and layer heights."
    maps = expected_maps(run, heights, planes, origin)
    maps["embedded_printer_settings"] = settings is not None
    if settings_note:
        maps["settings_note"] = settings_note
    spools = [s for s in maps["slots"] if s["has_part"]]
    grid, axes = compact_grid(planes, run["geometry"]["cell_mm"], heights)
    destination = Path(destination)
    if destination.exists():
        raise ValueError("choose a new export directory")
    with TemporaryDirectory(dir=destination.parent) as tmp:
        stage = Path(tmp) / "export"
        stage.mkdir()
        with ZipFile(stage / "flat.3mf", "w", ZIP_DEFLATED, compresslevel=1) as archive:
            types = _CONTENT_TYPES
            if settings is None:
                # OPC overrides can only name parts present in this package.
                types = "\n".join(line for line in types.splitlines()
                                  if 'PartName="/Metadata/project_settings.config"' not in line)
            archive.writestr("[Content_Types].xml", types)
            archive.writestr("_rels/.rels", _RELATIONSHIPS)
            _write_geometry(archive, grid, axes, spools, origin,
                            orca_format=project_settings is not None or run["profile"] == "a1")
            archive.writestr("Metadata/model_settings.config", _model_settings(spools, run["slots"], run["profile"]))
            archive.writestr("Metadata/layer_config_ranges.xml", _height_ranges(heights))
            if settings is not None:
                archive.writestr("Metadata/project_settings.config", json.dumps(settings))
        if public_geometry:
            if settings is None:
                (stage / "flat.3mf").rename(stage / "geometry.3mf")
            else:
                # Keep the identical bed placement, model and part-to-slot mapping.
                with ZipFile(stage / "flat.3mf") as src, ZipFile(
                        stage / "geometry.3mf", "w", ZIP_DEFLATED, compresslevel=1) as dst:
                    for name in src.namelist():
                        if name == "Metadata/project_settings.config":
                            continue
                        if name == "[Content_Types].xml":
                            dst.writestr(name, "\n".join(line for line in types.splitlines()
                                         if 'PartName="/Metadata/project_settings.config"' not in line))
                        else:
                            with src.open(name) as source, dst.open(name, "w") as target:
                                copyfileobj(source, target)
        (stage / "expected.json").write_text(json.dumps(maps, allow_nan=False) + "\n")
        if estimate_report is not None:
            (stage / "estimate.json").write_text(json.dumps(estimate_report, allow_nan=False) + "\n")
        stage.rename(destination)
    return maps
