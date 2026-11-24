"""Build a closed, outward-wound boundary for each material on a rectilinear grid."""

import numpy as np


def compact_grid(planes, pitch, heights):
    """Coalesce identical consecutive slices. Coordinates remain in millimetres."""
    # Mesh axes are x, y, z (image row zero is at the upper side of the plate).
    grid = planes[:, ::-1, :].transpose(2, 1, 0)
    axes = []
    for axis, scale in enumerate((pitch, pitch, None)):
        other_axes = tuple(i for i in range(3) if i != axis)
        changes = np.any(np.diff(grid, axis=axis) != 0, axis=other_axes)
        boundaries = np.r_[0, np.flatnonzero(changes) + 1, grid.shape[axis]]
        coords = (np.arange(grid.shape[axis] + 1) * scale if scale is not None
                  else np.r_[0, np.cumsum(heights)])
        axes.append(coords[boundaries])
        grid = np.take(grid, boundaries[:-1], axis=axis)
    return grid, axes


def _boundary(mask):
    """List outward-oriented quads as integer grid-vertex IDs and their owning cells."""
    padded = np.pad(mask, 1)
    shape = np.array(mask.shape) + 1
    quads, owners = [], []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for side in (0, 1):
            neighbour = [slice(1, -1)] * 3
            neighbour[axis] = slice(2, None) if side else slice(None, -2)
            cells = np.argwhere(mask & ~padded[tuple(neighbour)])
            owners.append(np.ravel_multi_index(cells.T, mask.shape))
            offsets = np.zeros((4, 3), dtype=int)
            offsets[:, axis] = side
            offsets[:, u] = (0, 1, 1, 0)
            offsets[:, v] = (0, 0, 1, 1)
            corners = cells[:, None, :] + (offsets if side else offsets[::-1])
            quads.append(np.ravel_multi_index(corners.T, shape).T)
    return np.concatenate(quads), np.concatenate(owners)


def slot_mesh(grid, axes, slot):
    """Weld corners connected along boundary edges, preserving distinct touching shells."""
    quads, owners = _boundary(grid == slot)
    if not len(quads):
        raise ValueError(f"loaded slot {slot} has no printable volume")
    # Sort half-edges by undirected edge, then owner. At a checkerboard edge
    # four faces meet. Pair the two faces of each occupied cell independently.
    ids = np.arange(quads.size).reshape(-1, 4)
    starts = quads.ravel()
    ends = np.roll(quads, -1, axis=1).ravel()
    lo, hi = np.minimum(starts, ends), np.maximum(starts, ends)
    order = np.lexsort((np.repeat(owners, 4), hi, lo))
    a, b = order.reshape(-1, 2).T
    following = np.roll(ids, -1, axis=1).ravel()
    # Each connected fan around a lattice point becomes one output vertex.
    left = np.r_[a, following[a]]
    right = np.r_[following[b], b]
    labels = ids.ravel().copy()
    while True:
        merged = labels.copy()
        np.minimum.at(merged, left, labels[right])
        np.minimum.at(merged, right, labels[left])
        merged = merged[merged]
        if np.array_equal(labels, merged):
            break
        labels = merged
    roots, indices = np.unique(labels, return_inverse=True)
    corners = indices.reshape(-1, 4)
    lattice = np.unravel_index(starts[roots], np.array(grid.shape) + 1)
    vertices = np.column_stack([coords[index] for coords, index in zip(axes, lattice)])

    # A saddle edge can have the same endpoint fans on both sheets. Give
    # each pair its own midpoint so even then no indexed edge has four faces.
    saddle = (lo[a][1:] == lo[a][:-1]) & (hi[a][1:] == hi[a][:-1])
    pairs = np.flatnonzero(np.r_[False, saddle] | np.r_[saddle, False])
    mid = np.full(quads.size, -1)
    mid[a[pairs]] = mid[b[pairs]] = len(vertices) + np.arange(len(pairs))
    mids = mid.reshape(-1, 4)
    split = np.any(mids >= 0, axis=1)
    regular = corners[~split]
    triangles = [regular[:, [0, 1, 2]], regular[:, [0, 2, 3]]]
    if len(pairs):
        end_ids = indices[following[a[pairs]]]
        midpoint = (vertices[indices[a[pairs]]] + vertices[end_ids]) / 2
        center = np.arange(split.sum()) + len(vertices) + len(midpoint)
        vertices = np.vstack([vertices, midpoint, vertices[corners[split]].mean(axis=1)])
        for side in range(4):
            first, last = corners[split, side], corners[split, (side + 1) % 4]
            middle = mids[split, side]
            divided = middle >= 0
            triangles.append(np.column_stack((center, first, np.where(divided, middle, last))))
            triangles.append(np.column_stack((center[divided], middle[divided], last[divided])))
    return vertices, np.vstack(triangles)
