"""Height-field mesh generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from prism_filament_painting.render.errors import QuiltInputError

from .params import LayerParams, MeshParams


class MeshError(QuiltInputError):
    """Raised when Quilt cannot produce a slicer-valid mesh."""


DEFAULT_FULL_VALIDATE_MAX_TRIANGLES = 500_000
_XY_UNITS = 1_000_000
_DEFAULT_SADDLE_RELIEF_MM = 0.01
_NW = 0
_NE = 1
_SE = 2
_SW = 3
_NORTH = 0
_EAST = 1
_SOUTH = 2
_WEST = 3


@dataclass
class QuiltMesh:
    vertices: np.ndarray
    faces: np.ndarray
    is_watertight: bool = False
    is_winding_consistent: bool = False

    @property
    def bounds(self) -> np.ndarray:
        return np.vstack((self.vertices.min(axis=0), self.vertices.max(axis=0)))

    def metadata(self) -> dict:
        bounds = self.bounds
        dims = bounds[1] - bounds[0]
        return {
            "dims_mm": tuple(float(value) for value in dims),
            "triangle_count": int(len(self.faces)),
            "vertex_count": int(len(self.vertices)),
            "watertight": bool(self.is_watertight),
            "winding_consistent": bool(self.is_winding_consistent),
            "bounds": tuple(tuple(float(value) for value in row) for row in bounds),
        }

    def to_stl_bytes(self, *, compute_normals: bool = True) -> bytes:
        triangles = self.vertices[self.faces].astype(np.float32)
        if compute_normals:
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0).astype(np.float32)
        else:
            normals = np.zeros((len(self.faces), 3), dtype=np.float32)
        record_type = np.dtype(
            [
                ("normal", "<f4", 3),
                ("v0", "<f4", 3),
                ("v1", "<f4", 3),
                ("v2", "<f4", 3),
                ("attr", "<u2"),
            ]
        )
        records = np.zeros(len(self.faces), dtype=record_type)
        if records.itemsize != 50:
            raise MeshError(f"binary STL record size must be 50 bytes, got {records.itemsize}")
        records["normal"] = normals
        records["v0"] = triangles[:, 0]
        records["v1"] = triangles[:, 1]
        records["v2"] = triangles[:, 2]
        header = b"Quilt stepped STL (numpy writer)".ljust(80, b"\0")
        return header + np.array([len(self.faces)], dtype="<u4").tobytes() + records.tobytes()

    def validate(self, *, full: bool) -> None:
        if full:
            # Validate the float32 precision that to_stl_bytes() actually writes, so a
            # "watertight" report cannot disagree with the served STL (saddle chamfers can
            # collapse below float32 precision at extreme xy_pixel_mm/max_dim_mm).
            check_vertices = self.vertices.astype(np.float32).astype(np.float64)
            mesh = trimesh.Trimesh(vertices=check_vertices, faces=self.faces, process=False)
            if not mesh.is_watertight or not mesh.is_winding_consistent:
                self.is_watertight = bool(mesh.is_watertight)
                self.is_winding_consistent = bool(mesh.is_winding_consistent)
                raise MeshError("mesh is not watertight with consistent winding")
            self.is_watertight = True
            self.is_winding_consistent = True
            return
        edges = np.concatenate(
            (
                self.faces[:, [0, 1]],
                self.faces[:, [1, 2]],
                self.faces[:, [2, 0]],
            ),
            axis=0,
        )
        undirected = np.sort(edges, axis=1)
        _, counts = np.unique(undirected, axis=0, return_counts=True)
        if not np.all(counts == 2):
            raise MeshError("mesh is not a closed 2-manifold")
        _, directed_counts = np.unique(edges, axis=0, return_counts=True)
        if not np.all(directed_counts == 1):
            # Closed 2-manifold (passed the undirected check) but winding is inconsistent;
            # record both flags honestly rather than leaving winding at its default.
            self.is_watertight = True
            self.is_winding_consistent = False
            raise MeshError("mesh winding is inconsistent")
        self.is_watertight = True
        self.is_winding_consistent = True


def build_mesh(
    height_field: np.ndarray,
    *,
    layer_params: LayerParams,
    mesh_params: MeshParams,
    full_validate_max_triangles: int = DEFAULT_FULL_VALIDATE_MAX_TRIANGLES,
    through_cuts: bool = False,
    precomputed_count: int | None = None,
    precomputed_clips: np.ndarray | None = None,
) -> tuple[QuiltMesh, dict]:
    field = np.asarray(height_field, dtype=np.int32)
    if field.ndim != 2 or field.shape[0] < 1 or field.shape[1] < 1:
        raise MeshError("height_field must be non-empty 2-D")
    below_base = field < layer_params.base_layers
    if through_cuts:
        below_base &= field != 0
    if below_base.any() or field.max() > layer_params.total_layers or not field.any():
        raise MeshError("height_field values outside layer bounds")
    # Saddle clips are relief-independent, so the single pass computed for the preflight
    # count is reused by the geometry builder (avoids recomputing _saddle_clips per request).
    clips = precomputed_clips if precomputed_clips is not None else _saddle_clips(field)
    if precomputed_count is not None:
        expected_triangles = precomputed_count
    else:
        expected_triangles = mesh_triangle_count_from_clips(field, clips, _relief_units(1.0))
    if expected_triangles > mesh_params.max_triangles:
        raise MeshError("mesh exceeds max_triangles")
    vertices, faces = _vertices_and_faces(field, layer_params=layer_params, mesh_params=mesh_params, clips=clips)
    if len(faces) != expected_triangles:
        raise MeshError("mesh triangle preflight did not match generated mesh")
    mesh = QuiltMesh(vertices=vertices, faces=faces)
    full_validation = len(faces) <= full_validate_max_triangles
    mesh.validate(full=full_validation)
    metadata = mesh.metadata()
    return mesh, metadata


def mesh_to_stl_bytes(mesh: QuiltMesh) -> bytes:
    return mesh.to_stl_bytes()


def mesh_triangle_count(height_field: np.ndarray) -> int:
    field = np.asarray(height_field, dtype=np.int32)
    if field.ndim != 2 or field.shape[0] < 1 or field.shape[1] < 1:
        return 0
    # The exact count uses unit-scale relief; the geometry (see _vertices_and_faces) uses
    # xy-scale relief. This is safe because the triangle count is relief-independent: the
    # _XY_UNITS//4 relief clamp keeps every edge interval non-empty regardless of magnitude.
    return mesh_triangle_count_from_clips(field, _saddle_clips(field), _relief_units(1.0))


def mesh_triangle_count_from_clips(field: np.ndarray, clips: np.ndarray, relief_units: int) -> int:
    height, width = field.shape
    total = 0
    for row in range(height):
        for col in range(width):
            if field[row, col] == 0:
                continue
            corners = int(clips[row, col].sum())
            polygon_vertices = 4 + corners
            total += 2 * (polygon_vertices - 2)  # top + base
            total += 2 * corners * int(field[row, col])  # diagonal relief walls
    total += _grid_side_triangle_count(field, clips, relief_units)
    return int(total)


def triangle_count_upper_bound(field: np.ndarray) -> int:
    """A cheap, fully vectorized upper bound on mesh_triangle_count(field).

    Used to choose the downsample factor without exact-counting at full resolution.
    The top/base and relief-wall terms are EXACT (derived from the same saddle stencil as
    the exact counter); only the grid-side term is over-estimated, so any field that passes
    this bound is guaranteed to pass the exact count.
    """
    field = np.asarray(field, dtype=np.int64)
    if field.ndim != 2 or field.shape[0] < 1 or field.shape[1] < 1:
        return 0
    height, width = field.shape
    # Per-cell clipped-corner count, vectorizing _saddle_clips: a corner is clipped when its
    # interior lattice point is a saddle, scattered into each of the four touching cells.
    corners = np.zeros((height, width), dtype=np.int64)
    if height > 1 and width > 1:
        nw = field[:-1, :-1]
        ne = field[:-1, 1:]
        sw = field[1:, :-1]
        se = field[1:, 1:]
        saddle = (np.minimum(nw, se) > np.maximum(ne, sw)) | (np.minimum(ne, sw) > np.maximum(nw, se))
        s = saddle.astype(np.int64)
        corners[:-1, :-1] += s  # NW cell -> SE corner
        corners[:-1, 1:] += s  # NE cell -> SW corner
        corners[1:, :-1] += s  # SW cell -> NE corner
        corners[1:, 1:] += s  # SE cell -> NW corner
    top_base = int((2 * (2 + corners)).sum())  # exact: 2*(polygon_vertices-2)
    walls = int((2 * corners * field).sum())  # exact: 2*corners*field
    grid = 0
    if width > 1:  # interior vertical edges
        a = field[:, :-1]
        b = field[:, 1:]
        grid += int((2 * np.abs(a - b) + 4 * (a + b)).sum())
    if height > 1:  # interior horizontal edges
        a = field[:-1, :]
        b = field[1:, :]
        grid += int((2 * np.abs(a - b) + 4 * (a + b)).sum())
    # Boundary edges (cell present on only one side): exactly 2*height each.
    grid += int(2 * (field[:, 0].sum() + field[:, -1].sum() + field[0, :].sum() + field[-1, :].sum()))
    return top_base + walls + grid


def _vertices_and_faces(
    field: np.ndarray,
    *,
    layer_params: LayerParams,
    mesh_params: MeshParams,
    clips: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = field.shape
    xy_scale = _xy_scale(width, height, mesh_params)
    # Geometry uses xy-scale relief; the preflight count uses unit-scale relief. Safe because
    # the triangle count is relief-independent (see mesh_triangle_count).
    relief_units = _relief_units(xy_scale)
    vertices_units, faces = _stepped_vertices_and_faces(field, relief_units=relief_units, clips=clips)
    vertices = vertices_units.astype(float)
    vertices[:, 0] = vertices[:, 0] / _XY_UNITS * xy_scale
    vertices[:, 1] = vertices[:, 1] / _XY_UNITS * xy_scale
    vertices[:, 2] *= layer_params.layer_height_mm
    return vertices, faces


def _stepped_vertices_and_faces(
    field: np.ndarray, *, relief_units: int, clips: np.ndarray | None = None, fill_clips_to: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    if clips is None:
        clips = _saddle_clips(field)
    height, width = field.shape
    vertex_index: dict[tuple[int, int, int], int] = {}
    vertices: list[tuple[int, int, int]] = []
    faces: list[tuple[int, int, int]] = []

    def vertex(point: tuple[int, int], z: int) -> int:
        key = (int(point[0]), int(point[1]), int(z))
        found = vertex_index.get(key)
        if found is not None:
            return found
        vertex_index[key] = len(vertices)
        vertices.append(key)
        return len(vertices) - 1

    def add_triangle(a: int, b: int, c: int) -> None:
        if a != b and b != c and a != c:
            faces.append((a, b, c))

    def add_side(p0: tuple[int, int], p1: tuple[int, int], z0: int, z1: int) -> None:
        if z1 <= z0 or p0 == p1:
            return
        for lower in range(z0, z1):
            upper = lower + 1
            v0 = vertex(p0, lower)
            v1 = vertex(p1, lower)
            v2 = vertex(p1, upper)
            v3 = vertex(p0, upper)
            add_triangle(v0, v1, v2)
            add_triangle(v0, v2, v3)

    for row in range(height):
        for col in range(width):
            if field[row, col] == 0:  # A through-cut or an empty cell in this band.
                continue
            polygon = _cell_polygon(col, row, clips[row, col], relief_units)
            top = [vertex(point, int(field[row, col])) for point in polygon]
            bottom = [vertex(point, 0) for point in polygon]
            for index in range(1, len(polygon) - 1):
                add_triangle(top[0], top[index], top[index + 1])
                add_triangle(bottom[0], bottom[index + 1], bottom[index])
            for corner in np.flatnonzero(clips[row, col]):
                p0, p1 = _corner_clip_edge(col, row, int(corner), relief_units)
                if fill_clips_to:
                    # The inverted clear cap fills the classic corner relief all the
                    # way to the bottom of the slab. Each cell owns 1/4 of the diamond.
                    dx, dy = ((0, 0), (1, 0), (1, 1), (0, 1))[corner]
                    centre = ((col + dx) * _XY_UNITS, (row + dy) * _XY_UNITS)
                    add_triangle(vertex(p1, fill_clips_to), vertex(p0, fill_clips_to), vertex(centre, fill_clips_to))
                    add_triangle(vertex(p0, 0), vertex(p1, 0), vertex(centre, 0))
                    add_side(p1, p0, int(field[row, col]), fill_clips_to)
                else:
                    add_side(p0, p1, 0, int(field[row, col]))

    _add_grid_side_faces(field, clips, relief_units, add_side)
    return np.asarray(vertices, dtype=np.int64), np.asarray(faces, dtype=np.int64)


def _saddle_clips(field: np.ndarray) -> np.ndarray:
    height, width = field.shape
    clips = np.zeros((height, width, 4), dtype=bool)
    for row in range(1, height):
        for col in range(1, width):
            nw = int(field[row - 1, col - 1])
            ne = int(field[row - 1, col])
            sw = int(field[row, col - 1])
            se = int(field[row, col])
            if min(nw, se) > max(ne, sw) or min(ne, sw) > max(nw, se):
                clips[row - 1, col - 1, _SE] = True
                clips[row - 1, col, _SW] = True
                clips[row, col - 1, _NE] = True
                clips[row, col, _NW] = True
    return clips


def _cell_polygon(col: int, row: int, clipped: np.ndarray, relief: int) -> list[tuple[int, int]]:
    x0 = col * _XY_UNITS
    x1 = (col + 1) * _XY_UNITS
    y0 = row * _XY_UNITS
    y1 = (row + 1) * _XY_UNITS
    e = relief
    points: list[tuple[int, int]] = []
    if clipped[_NW]:
        points.extend(((x0, y0 + e), (x0 + e, y0)))
    else:
        points.append((x0, y0))
    if clipped[_NE]:
        points.extend(((x1 - e, y0), (x1, y0 + e)))
    else:
        points.append((x1, y0))
    if clipped[_SE]:
        points.extend(((x1, y1 - e), (x1 - e, y1)))
    else:
        points.append((x1, y1))
    if clipped[_SW]:
        points.extend(((x0 + e, y1), (x0, y1 - e)))
    else:
        points.append((x0, y1))
    return points


def _corner_clip_edge(col: int, row: int, corner: int, relief: int) -> tuple[tuple[int, int], tuple[int, int]]:
    x0 = col * _XY_UNITS
    x1 = (col + 1) * _XY_UNITS
    y0 = row * _XY_UNITS
    y1 = (row + 1) * _XY_UNITS
    e = relief
    if corner == _NW:
        return (x0, y0 + e), (x0 + e, y0)
    if corner == _NE:
        return (x1 - e, y0), (x1, y0 + e)
    if corner == _SE:
        return (x1, y1 - e), (x1 - e, y1)
    if corner == _SW:
        return (x0 + e, y1), (x0, y1 - e)
    raise MeshError("unknown saddle relief corner")


def _grid_side_triangle_count(field: np.ndarray, clips: np.ndarray, relief_units: int) -> int:
    height, width = field.shape
    total = 0
    for row in range(height):
        for edge_col in range(width + 1):
            left = _side_interval(clips[row, edge_col - 1], _EAST, relief_units) if edge_col > 0 else None
            right = _side_interval(clips[row, edge_col], _WEST, relief_units) if edge_col < width else None
            left_height = int(field[row, edge_col - 1]) if edge_col > 0 else 0
            right_height = int(field[row, edge_col]) if edge_col < width else 0
            total += _shared_edge_triangle_count(left, left_height, right, right_height)
    for edge_row in range(height + 1):
        for col in range(width):
            upper = _side_interval(clips[edge_row - 1, col], _SOUTH, relief_units) if edge_row > 0 else None
            lower = _side_interval(clips[edge_row, col], _NORTH, relief_units) if edge_row < height else None
            upper_height = int(field[edge_row - 1, col]) if edge_row > 0 else 0
            lower_height = int(field[edge_row, col]) if edge_row < height else 0
            total += _shared_edge_triangle_count(upper, upper_height, lower, lower_height)
    return total


def _shared_edge_triangle_count(
    first: tuple[int, int] | None,
    first_height: int,
    second: tuple[int, int] | None,
    second_height: int,
) -> int:
    total = 0
    if first is not None and second is not None:
        overlap = _intersect_interval(first, second)
        if overlap is not None and first_height != second_height:
            total += 2 * abs(first_height - second_height)
        total += 2 * first_height * len(_subtract_interval(first, second))
        total += 2 * second_height * len(_subtract_interval(second, first))
        return total
    if first is not None:
        return 2 * first_height
    if second is not None:
        return 2 * second_height
    return 0


def _add_grid_side_faces(
    field: np.ndarray,
    clips: np.ndarray,
    relief_units: int,
    add_side,
) -> None:
    height, width = field.shape
    for row in range(height):
        for edge_col in range(width + 1):
            left = _side_interval(clips[row, edge_col - 1], _EAST, relief_units) if edge_col > 0 else None
            right = _side_interval(clips[row, edge_col], _WEST, relief_units) if edge_col < width else None
            left_height = int(field[row, edge_col - 1]) if edge_col > 0 else 0
            right_height = int(field[row, edge_col]) if edge_col < width else 0
            x = edge_col * _XY_UNITS
            y = row * _XY_UNITS
            if left is not None and right is not None:
                overlap = _intersect_interval(left, right)
                if overlap is not None:
                    if left_height > right_height:
                        add_side(*_vertical_points(x, y, overlap, owner="left"), right_height, left_height)
                    elif right_height > left_height:
                        add_side(*_vertical_points(x, y, overlap, owner="right"), left_height, right_height)
                for interval in _subtract_interval(left, right):
                    add_side(*_vertical_points(x, y, interval, owner="left"), 0, left_height)
                for interval in _subtract_interval(right, left):
                    add_side(*_vertical_points(x, y, interval, owner="right"), 0, right_height)
            elif left is not None:
                add_side(*_vertical_points(x, y, left, owner="left"), 0, left_height)
            elif right is not None:
                add_side(*_vertical_points(x, y, right, owner="right"), 0, right_height)

    for edge_row in range(height + 1):
        for col in range(width):
            upper = _side_interval(clips[edge_row - 1, col], _SOUTH, relief_units) if edge_row > 0 else None
            lower = _side_interval(clips[edge_row, col], _NORTH, relief_units) if edge_row < height else None
            upper_height = int(field[edge_row - 1, col]) if edge_row > 0 else 0
            lower_height = int(field[edge_row, col]) if edge_row < height else 0
            x = col * _XY_UNITS
            y = edge_row * _XY_UNITS
            if upper is not None and lower is not None:
                overlap = _intersect_interval(upper, lower)
                if overlap is not None:
                    if upper_height > lower_height:
                        add_side(*_horizontal_points(x, y, overlap, owner="upper"), lower_height, upper_height)
                    elif lower_height > upper_height:
                        add_side(*_horizontal_points(x, y, overlap, owner="lower"), upper_height, lower_height)
                for interval in _subtract_interval(upper, lower):
                    add_side(*_horizontal_points(x, y, interval, owner="upper"), 0, upper_height)
                for interval in _subtract_interval(lower, upper):
                    add_side(*_horizontal_points(x, y, interval, owner="lower"), 0, lower_height)
            elif upper is not None:
                add_side(*_horizontal_points(x, y, upper, owner="upper"), 0, upper_height)
            elif lower is not None:
                add_side(*_horizontal_points(x, y, lower, owner="lower"), 0, lower_height)


def _side_interval(clipped: np.ndarray, side: int, relief_units: int) -> tuple[int, int] | None:
    if side == _NORTH:
        start = _relief_start(clipped[_NW], relief_units)
        end = _relief_end(clipped[_NE], relief_units)
    elif side == _EAST:
        start = _relief_start(clipped[_NE], relief_units)
        end = _relief_end(clipped[_SE], relief_units)
    elif side == _SOUTH:
        start = _relief_start(clipped[_SW], relief_units)
        end = _relief_end(clipped[_SE], relief_units)
    elif side == _WEST:
        start = _relief_start(clipped[_NW], relief_units)
        end = _relief_end(clipped[_SW], relief_units)
    else:
        raise MeshError("unknown side")
    if start >= end:
        return None
    return (start, end)


def _relief_start(clipped: bool, relief_units: int) -> int:
    return relief_units if clipped else 0


def _relief_end(clipped: bool, relief_units: int) -> int:
    return _XY_UNITS - relief_units if clipped else _XY_UNITS


def _intersect_interval(first: tuple[int, int], second: tuple[int, int]) -> tuple[int, int] | None:
    start = max(first[0], second[0])
    end = min(first[1], second[1])
    if start >= end:
        return None
    return (start, end)


def _subtract_interval(first: tuple[int, int], second: tuple[int, int]) -> list[tuple[int, int]]:
    overlap = _intersect_interval(first, second)
    if overlap is None:
        return [first]
    pieces: list[tuple[int, int]] = []
    if first[0] < overlap[0]:
        pieces.append((first[0], overlap[0]))
    if overlap[1] < first[1]:
        pieces.append((overlap[1], first[1]))
    return pieces


def _vertical_points(
    x: int,
    y: int,
    interval: tuple[int, int],
    *,
    owner: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    start, end = interval
    if owner == "left":
        return (x, y + start), (x, y + end)
    return (x, y + end), (x, y + start)


def _horizontal_points(
    x: int,
    y: int,
    interval: tuple[int, int],
    *,
    owner: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    start, end = interval
    if owner == "upper":
        return (x + end, y), (x + start, y)
    return (x + start, y), (x + end, y)


def _relief_units(xy_scale: float) -> int:
    units = int(round(_DEFAULT_SADDLE_RELIEF_MM / xy_scale * _XY_UNITS))
    return min(_XY_UNITS // 4, max(1, units))


def _xy_scale(width: int, height: int, mesh_params: MeshParams) -> float:
    if (
        not np.isfinite(mesh_params.xy_pixel_mm)
        or not np.isfinite(mesh_params.max_dim_mm)
        or mesh_params.xy_pixel_mm <= 0
        or mesh_params.max_dim_mm <= 0
    ):
        raise MeshError("mesh dimensions must be positive")
    current_max = max(width, height) * mesh_params.xy_pixel_mm
    if current_max <= mesh_params.max_dim_mm:
        return mesh_params.xy_pixel_mm
    return mesh_params.max_dim_mm / max(width, height)
