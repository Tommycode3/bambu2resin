"""Mesh-plane intersection: from triangles to contour segments.

Closed polygons are not reconstructed. Half of real-world models are not
watertight and some have edges shared by more than two faces: requiring
closed contours would mean failing exactly on the files one wants to print.
Loose segments are enough, because the subsequent fill works by winding
number and does not need to know which segment follows which.
"""

from __future__ import annotations

import numpy as np


def plane_segments(vertices: np.ndarray, faces: np.ndarray, z: float) -> np.ndarray:
    """Segments of the intersection with the horizontal plane at height `z`.

    Returns an (M, 2, 2) array: for each segment the start and end point in
    XY. The orientation is consistent with that of the faces, so the outer
    contour runs counterclockwise and holes run clockwise.
    """
    if len(faces) == 0:
        return np.zeros((0, 2, 2), dtype=np.float64)

    tri = vertices[faces]  # (F, 3, 3)
    height = tri[:, :, 2] - z

    # A vertex exactly on the plane counts as "above": this way a fully
    # coplanar triangle is never considered crossed.
    above = height >= 0.0
    count = above.sum(axis=1)
    crossing = (count == 1) | (count == 2)
    if not crossing.any():
        return np.zeros((0, 2, 2), dtype=np.float64)

    tri = tri[crossing]
    above = above[crossing]
    count = count[crossing]
    crossing_faces = faces[crossing]

    # The "lone" vertex is the one on the opposite side from the other two: the
    # two edges crossing the plane both start from it.
    lone = np.where(count == 1, above.argmax(axis=1), (~above).argmax(axis=1))
    rows = np.arange(len(tri))
    first = (lone + 1) % 3
    second = (lone + 2) % 3

    # Each edge must be computed in canonical order, not in the order the
    # triangle lists it. The two triangles sharing an edge traverse it in
    # opposite directions and, when interpolating, get points that differ in
    # the last bit: the contour is left with a microscopic gap, and if a
    # sampling row falls into it that row disappears.
    p1 = _canonical_crossing(vertices, crossing_faces[rows, lone], crossing_faces[rows, first], z)
    p2 = _canonical_crossing(vertices, crossing_faces[rows, lone], crossing_faces[rows, second], z)

    # With the lone vertex above the plane, going apex->first->second keeps
    # the solid on the left; when it is below, the direction is reversed.
    flip = count == 2
    start = np.where(flip[:, None], p2, p1)
    end = np.where(flip[:, None], p1, p2)

    segments = np.stack([start, end], axis=1)

    # Zero-length segments come from vertices lying on the plane and carry no
    # information.
    length = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    return segments[length > 0.0]


def _canonical_crossing(
    vertices: np.ndarray, index_a: np.ndarray, index_b: np.ndarray, z: float
) -> np.ndarray:
    """Crossing of an edge, independent of the direction it is traversed in.

    The endpoints are sorted by index before interpolating: two triangles
    sharing the edge thus perform exactly the same computation and get the
    same point down to the last bit.
    """
    swap = index_b < index_a
    low = np.where(swap, index_b, index_a)
    high = np.where(swap, index_a, index_b)
    return _edge_crossing(vertices[low], vertices[high], z)


def _edge_crossing(a: np.ndarray, b: np.ndarray, z: float) -> np.ndarray:
    """XY point where edge a-b crosses the plane at height z."""
    dz = b[:, 2] - a[:, 2]
    # zero dz means a horizontal edge: it does not cross, the value is not used
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dz != 0.0, (z - a[:, 2]) / dz, 0.0)
    t = np.clip(t, 0.0, 1.0)[:, None]
    return a[:, :2] + t * (b[:, :2] - a[:, :2])


def signed_area(segments: np.ndarray) -> float:
    """Signed area enclosed by the segments.

    The shoelace formula is applied to each segment: for a closed contour the
    contributions add up to the area, and for a contour with holes the holes
    (traversed the other way) subtract themselves. It is also the strictest
    check of the orientation: if the segments were reversed the area would be
    negative.
    """
    if len(segments) == 0:
        return 0.0
    x0, y0 = segments[:, 0, 0], segments[:, 0, 1]
    x1, y1 = segments[:, 1, 0], segments[:, 1, 1]
    return float(0.5 * np.sum(x0 * y1 - x1 * y0))
