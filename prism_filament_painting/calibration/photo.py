"""Sample patch interiors directly in the EXIF-oriented photo pixel frame."""

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from prism_filament_painting.core.calibration_board import validate_board, validate_database
from prism_filament_painting.core.color import hex_to_rgb


def _quad(points, size):
    pts = np.asarray(points, dtype=float)
    if (
        pts.shape != (4, 2)
        or not np.isfinite(pts).all()
        or (pts < 0).any()
        or (pts > np.array(size) - 1).any()
    ):
        raise ValueError("four finite pixel centres inside the image are required")
    edges = np.roll(pts, -1, axis=0) - pts
    after = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * after[:, 1] - edges[:, 1] * after[:, 0]
    if not (cross > 0).all():
        raise ValueError("click convex corners clockwise from the keyed mark")
    return pts


def _homography(source, destination):
    rows, rhs = [], []
    for (x, y), (u, v) in zip(source, destination, strict=True):
        rows.extend(([x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]))
        rhs.extend((u, v))
    return np.append(np.linalg.solve(rows, rhs), 1).reshape(3, 3)


def _sample(image, quad):
    quad = _quad(quad, image.size)
    origin = np.floor(quad.min(axis=0)).astype(int)
    end = np.ceil(quad.max(axis=0)).astype(int) + 1
    crop = image.crop((*origin, *end))
    mask = Image.new("1", crop.size)
    ImageDraw.Draw(mask).polygon([tuple(point) for point in quad - origin], fill=1)
    pixels = np.asarray(crop)[np.asarray(mask, dtype=bool)]
    if len(pixels) < 25 or np.min(end - origin) < 5:
        raise ValueError("sample too small: photograph the board at higher resolution")
    centre = np.median(pixels, axis=0)
    return centre, {
        "n_pixels": len(pixels),
        "mad_srgb8": np.median(np.abs(pixels - centre), axis=0).tolist(),
    }


def _linear(rgb):
    encoded = np.asarray(rgb) / 255.0
    return np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)


def _correct(value, matrix):
    linear = np.clip(np.append(_linear(value), 1) @ matrix, 0, 1)
    return 255 * np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


def _checker_matrix(image, references):
    if len(references) < 6:
        raise ValueError("supply at least six independent ColorChecker patches")
    observed, expected = [], []
    for patch in references:
        measured, _ = _sample(image, patch["corners_px"])
        observed.append(_linear(measured))
        expected.append(_linear(hex_to_rgb(patch["reference_hex"])))
    design = np.column_stack((observed, np.ones(len(observed))))
    matrix, _, rank, singular = np.linalg.lstsq(design, expected, rcond=None)
    if rank < 4 or singular[-1] < singular[0] * 1e-4:
        raise ValueError("ColorChecker samples need independently varying colours")
    return matrix


def extract(board, photo, clicks, *, source_kind, photo_sha256):
    validate_board(board)
    image = ImageOps.exif_transpose(photo).convert("RGB")
    corners = _quad(clicks["mark_centres_px"], image.size)
    transform = _homography(board["mark_centres_mm"], corners)
    references = clicks.get("checker")
    matrix = None if references is None else _checker_matrix(image, references)
    measurements = {}
    for patch in board["patches"]:
        # Use the middle half on both axes to reject seams, neighbouring
        # stacks, blurred edges and small registration errors.
        interior = np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
        points = patch["xy_mm"] + board["patch_mm"] * interior
        homogeneous = np.column_stack((points, np.ones(4))) @ transform.T
        quad = homogeneous[:, :2] / homogeneous[:, 2:]
        raw, stats = _sample(image, quad)
        corrected = raw if matrix is None else _correct(raw, matrix)
        measurements[patch["id"]] = {
            **stats,
            "raw_srgb8": raw.tolist(),
            "srgb8": np.rint(corrected).astype(int).tolist(),
        }
    database = {
        "schema": "front-lit-colours/v1",
        "board": board,
        "measurements": measurements,
        "capture": {
            "kind": source_kind,
            "photo_sha256": photo_sha256,
            "oriented_size_px": list(image.size),
            "clicks": clicks,
            "linear_rgb_affine": None if matrix is None else matrix.tolist(),
        },
    }
    validate_database(database)
    return database
