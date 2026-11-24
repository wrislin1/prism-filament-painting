"""Export registered, watertight cuboids as one binary STL per material slot."""

import struct
from pathlib import Path

from prism_filament_painting.core.calibration_board import validate_board


def triangles(x, y, z, width, depth, height):
    points = [
        (x + i * width, y + j * depth, z + k * height)
        for k in (0, 1)
        for j in (0, 1)
        for i in (0, 1)
    ]
    faces = [
        ((0, 0, -1), (0, 2, 3, 1)),
        ((0, 0, 1), (4, 5, 7, 6)),
        ((0, -1, 0), (0, 1, 5, 4)),
        ((0, 1, 0), (2, 6, 7, 3)),
        ((-1, 0, 0), (0, 4, 6, 2)),
        ((1, 0, 0), (1, 3, 7, 5)),
    ]
    for normal, (a, b, c, d) in faces:
        for face in ((a, b, c), (a, c, d)):
            yield struct.pack(
                "<12fH", *normal, *points[face[0]], *points[face[1]], *points[face[2]], 0
            )


def write_plate(board: dict, directory: Path) -> None:
    validate_board(board)
    width, depth = board["size_mm"]
    base, layer, side = board["backing_mm"], board["layer_mm"], board["patch_mm"]
    slots = {m["id"]: [] for m in board["materials"]}
    backing, marks, *_ = slots.values()
    backing.append((0, 0, 0, width, depth, base))
    # Coordinates in the manifest have their origin at the top left of a top
    # view. STL +Y points away from the viewer towards the far edge.
    for x, y in board["mark_centres_mm"]:
        marks.append((x - 1.6, depth - y - 1.6, base, 3.2, 3.2, 4 * layer))
    marks.append((7, depth - 5, base, 1.6, 2, 4 * layer))  # top-left key
    for patch in board["patches"]:
        x, y = patch["xy_mm"]
        z = base
        for spool, count in patch["bands"]:
            height = count * layer
            slots[spool].append((x, depth - y - side, z, side, side, height))
            z += height
    for index, boxes in enumerate(slots.values()):
        with (directory / f"slot-{index}.stl").open("wb") as stream:
            stream.write(struct.pack("<80sI", b"Front-lit calibration board (mm)", 12 * len(boxes)))
            for box in boxes:
                stream.writelines(triangles(*box))
