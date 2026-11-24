"""Write editable Bambu/Orca projects from the final renderer mesh and schedule.

The layer-event encoding is adapted from Akoozie/layertone (MIT), commit
836090ede4c0679cd05e56b49793f6562e9200a0. See licenses/layertone-MIT.txt.
"""
from __future__ import annotations

import json
import shutil
from io import BufferedWriter
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

from ..errors import QuiltInputError
from .slicer_settings import prepare_project, project_variant

if TYPE_CHECKING:
    from .params import GenerateResult


_CONTENT_TYPES = """<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/Metadata/model_settings.config" ContentType="application/xml"/>
<Override PartName="/Metadata/project_settings.config" ContentType="application/json"/>
</Types>"""
_RELATIONSHIPS = """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="model" Target="/3D/3dmodel.model" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""
_MODEL_SETTINGS = """<config><object id="1">
<metadata key="name" value="Prism"/><metadata key="extruder" value="1"/>
<part id="1" subtype="normal_part"><metadata key="name" value="Prism"/></part>
</object><plate><metadata key="plater_id" value="1"/>
<metadata key="filament_map_mode" value="Auto For Flush"/>
<metadata key="filament_maps" value="{filament_maps}"/><model_instance>
<metadata key="object_id" value="1"/><metadata key="instance_id" value="0"/>
<metadata key="identify_id" value="1"/></model_instance></plate></config>"""


def _write_model(archive: ZipFile, result: GenerateResult, transform: str) -> None:
    mesh = result.mesh
    with archive.open("3D/3dmodel.model", "w") as raw, BufferedWriter(raw) as model:
        model.write(b'<model unit="millimeter" xml:lang="en-US" '
                    b'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
                    b'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">'
                    b'<metadata name="Application">BambuStudio-01.10.01.50</metadata>'
                    b'<metadata name="OrcaSlicer">2.4.2</metadata>'
                    b'<metadata name="BambuStudio:3mfVersion">1</metadata>'
                    b'<resources><object id="1" type="model" name="Prism"><mesh><vertices>')
        for x, y, z in mesh.vertices:
            model.write(f'<vertex x="{x:.12g}" y="{y:.12g}" z="{z:.12g}"/>\n'.encode())
        model.write(b'</vertices><triangles>')
        for a, b, c in mesh.faces:
            model.write(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>\n'.encode())
        model.write(f'</triangles></mesh></object></resources><build><item objectid="1" '
                    f'transform="{transform}"/></build></model>'.encode())


def _layer_events(result: GenerateResult, slots: list[int], pause: bool) -> bytes:
    root = ET.Element("custom_gcodes_per_layer")
    plate = ET.SubElement(root, "plate")
    ET.SubElement(plate, "plate_info", id="1")
    for i, layer in enumerate(result.swap_schedule, start=1):
        if slots[i] == slots[i - 1]:
            continue
        filament = result.output_filaments[i]
        ET.SubElement(
            plate, "layer", top_z=f"{layer * result.layer_params.layer_height_mm:.12g}",
            type="1" if pause else "2", extruder="1" if pause else str(slots[i]),
            color=filament.hex, extra=f"Load {filament.hex}" if pause else "",
            gcode="M400 U1" if pause else "tool_change",
        )
    ET.SubElement(plate, "mode", value="MultiAsSingle")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def write_classic_project(directory: Path, result: GenerateResult, *, project_settings=None,
                          settings_provider=None) -> dict[str, str]:
    """With no supplied settings, keep the STL, preview and manual swap list."""
    if project_settings is None and settings_provider is None:
        return {}
    colors = []
    spool_slots = {}
    slots = []
    for filament in result.output_filaments:
        spool = (filament.brand.casefold(), filament.name.casefold(), filament.hex.upper())
        if spool not in spool_slots:
            colors.append(filament.hex.upper())
            spool_slots[spool] = len(colors)
        slots.append(spool_slots[spool])

    height = result.layer_params.layer_height_mm
    if project_settings is None:
        transform, manual_settings, configured = settings_provider(result, colors, height)
    else:
        # A single loaded spool also supports manual swaps in a multi-colour job.
        loaded = project_settings.get("filament_colour")
        pause_only = len(colors) > 1 and isinstance(loaded, list) and len(loaded) == 1
        lower, upper = result.mesh.bounds
        if any(layer * height > upper[2] + 1e-8 for layer in result.swap_schedule):
            raise QuiltInputError("Move the filament changes within the height of the generated model.")
        settings, origin = prepare_project(
            project_settings, colors[:1] if pause_only else colors, first=height, regular=height,
            heights=[height], top_z=max(upper[2], result.layer_params.total_layers * height),
            size=upper[:2] - lower[:2])
        dx, dy = origin - lower[:2]
        transform = f"1 0 0 0 1 0 0 0 1 {dx:.12g} {dy:.12g} 0"
        configured = project_variant(settings) if len(colors) <= 16 and not pause_only else None
        manual_settings = project_variant(settings, pause=True)
    project = directory / "0.3mf"
    manual = directory / "0.pause.3mf"
    with ZipFile(manual, "w", ZIP_DEFLATED, compresslevel=1) as archive:
        archive.writestr("[Content_Types].xml", _CONTENT_TYPES)
        archive.writestr("_rels/.rels", _RELATIONSHIPS)
        _write_model(archive, result, transform)
    # Reuse the compressed mesh for the second download, even at the triangle cap.
    variants = [(manual, True, manual_settings)]
    links = {"pause_project_url": manual.name}
    if configured is not None:
        shutil.copyfile(manual, project)
        variants.append((project, False, configured))
        links["project_url"] = project.name
    for path, pause, settings in variants:
        active_colors = settings["filament_colour"]
        with ZipFile(path, "a", ZIP_DEFLATED) as archive:
            archive.writestr("Metadata/model_settings.config", _MODEL_SETTINGS.format(
                filament_maps=" ".join(["1"] * len(active_colors))))
            archive.writestr("Metadata/project_settings.config", json.dumps(settings))
            archive.writestr("Metadata/custom_gcode_per_layer.xml", _layer_events(result, slots, pause))
    return links
