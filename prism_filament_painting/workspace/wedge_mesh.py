"""Two closed-shell binary STL parts in one coordinate system (millimetres)."""

from __future__ import annotations

import json
import struct
from decimal import ROUND_CEILING, Decimal
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from prism_filament_painting.core.forward import RenderConstants
from prism_filament_painting.core.opacity import LAYER_MM, PATCH_COUNT

PITCH = 6.4
WIDTH = round((PATCH_COUNT + 1) * PITCH, 2)
DEPTH = 22
MIN_BACKING_MM = Decimal("0.64")
MAX_BACKING_MM = Decimal("4.00")
RAIL = 4.0
# A box consists of six outward-facing quads.
FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def backing_plan(td_mm: float) -> tuple[float, bool]:
    """Three front-lit hiding distances leave 0.1% predicted show-through."""
    hiding = Decimal(str(td_mm)) * Decimal(str(RenderConstants().td_scale))
    target = 3 * hiding
    step = Decimal(str(LAYER_MM))
    bounded = min(max(target, MIN_BACKING_MM), MAX_BACKING_MM)
    layers = (bounded / step).to_integral_value(rounding=ROUND_CEILING)
    return float(step * layers), target > MAX_BACKING_MM


def _boxes_stl(boxes):
    triangles = bytearray()
    for x, y, z, width, depth, height in boxes:
        vertices = [
            (x + i * width, y + j * depth, z + k * height)
            for k in (0, 1)
            for i, j in ((0, 0), (1, 0), (1, 1), (0, 1))
        ]
        for face, normal in zip(
            FACES,
            ((0, 0, -1), (0, 0, 1), (0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0)),
            strict=True,
        ):
            a, b, c, d = (vertices[i] for i in face)
            for points in ((a, b, c), (a, c, d)):
                triangles.extend(
                    struct.pack("<12fH", *normal, *points[0], *points[1], *points[2], 0)
                )
    return bytes(80) + struct.pack("<I", len(triangles) // 50) + triangles


def _label(k, backing_mm):
    # Raised seven-segment numerals, separated strokes, 0.64 mm wide and four layers tall.
    segments = (
        (0.46, 4.40, 1.48, 0.64),
        (1.76, 2.62, 0.64, 1.56),
        (1.76, 0.86, 0.64, 1.56),
        (0.46, 0.00, 1.48, 0.64),
        (0.00, 0.86, 0.64, 1.56),
        (0.00, 2.62, 0.64, 1.56),
        (0.70, 2.20, 1.00, 0.64),
    )
    digits = (
        "012345",
        "12",
        "01643",
        "01236",
        "5612",
        "05623",
        "056432",
        "012",
        "0123456",
        "012356",
    )
    label = str(k)
    x = (k + 0.5) * PITCH - (len(label) * 3.0 - 0.6) / 2
    for char in label:
        for i in digits[int(char)]:
            dx, dy, width, depth = segments[int(i)]
            yield x + dx, 0.8 + dy, backing_mm, width, depth, 4 * LAYER_MM
        x += 3.0


def wedge_parts(backing_mm=4.0, rail_mm=RAIL):
    top = [(0, 17, backing_mm, WIDTH, 5, rail_mm)]
    for k in range(PATCH_COUNT + 1):
        if k:
            top.append((k * PITCH + 0.3, 7, backing_mm, PITCH - 0.6, 9, k * LAYER_MM))
        top.extend(_label(k, backing_mm))
    return {
        "backing.stl": _boxes_stl([(0, 0, 0, WIDTH, DEPTH, backing_mm)]),
        "test-spool.stl": _boxes_stl(top),
    }


def wedge_archive(spec):
    backing_mm = spec["swap_z_mm"]
    first_test_layer = round(backing_mm / spec["layer_mm"]) + 1
    cap_note = ""
    if spec.get("backing_target_capped"):
        cap_note = (
            "Backing reached the thickness ceiling before its opacity target. "
            "Choose a different backing if the thicker control differs.\n"
        )
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as out:
        for name, data in wedge_parts(backing_mm, spec["rail_mm"]).items():
            out.writestr(name, data)
        out.writestr("wedge.json", json.dumps(spec, indent=2))
        out.writestr(
            "print.txt",
            "Import both STLs together as parts of one object, keeping their coordinates.\n"
            "Use solid (100%) fill, 0.08 mm first and regular layers, no adaptive layers.\n"
            f"Load backing first. Insert one manual filament change after Z={backing_mm:.2f} mm "
            f"(before the first test-spool extrusion, layer {first_test_layer} "
            f"at Z={backing_mm + spec['layer_mm']:.2f} mm).\n"
            f"Verify the change in preview. Compare the {backing_mm:.2f} mm backing and "
            f"the {spec['rail_mm']:.2f} mm rail with thicker controls of each spool "
            "before reading. Compare patch centres in diffuse front light.\n"
            + cap_note,
        )
    return buffer.getvalue()
