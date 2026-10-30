"""Automatic choice of the print pose.

In MSLA orientation decides almost everything. Each layer is peeled from the
vat film, and the peel force grows with the cross-section area: a flat surface
perpendicular to the build direction produces a huge layer that pulls on the
part all at once. That is why the rule of thumb in resin is to tilt everything,
never lay it flat.

The score combines three quantities computed from the face normals, without
slicing:

  support       area projected downwards, i.e. how much surface actually
                needs supporting (and where the marks will end up)
  horizontality area of faces nearly perpendicular to the build direction,
                the proxy for peel forces
  height        number of layers, i.e. print time
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError

# How many directions to evaluate at a time. The normals x directions product
# is a dense matrix: in blocks it stays under 200 MB even on meshes with half a
# million triangles.
_BATCH_BYTES = 200_000_000

# Exponent that isolates faces nearly perpendicular to the build direction.
_HORIZONTAL_SHARPNESS = 6


@dataclass
class Weights:
    cross_section: float = 1.5
    support: float = 1.0
    horizontal: float = 0.5
    height: float = 0.4


PRESETS: dict[str, Weights] = {
    # The default favours print success: small cross-section, little to
    # support, no flat faces.
    "reliability": Weights(cross_section=1.5, support=1.0, horizontal=0.5, height=0.4),
    # Accepts larger cross-sections in exchange for printing low, hence fast.
    "speed": Weights(cross_section=0.8, support=0.6, horizontal=0.4, height=2.0),
    # Minimizes the surface touched by supports, i.e. the marks to finish.
    "finish": Weights(cross_section=1.0, support=2.5, horizontal=0.5, height=0.3),
}

# Italian names accepted by earlier versions of the CLI.
PRESET_ALIASES: dict[str, str] = {
    "affidabilita": "reliability",
    "velocita": "speed",
    "finitura": "finish",
}


@dataclass
class Orientation:
    up: np.ndarray
    """Direction of the object that will end up facing upwards."""

    rotation: np.ndarray
    """4x4 matrix that maps `up` onto the Z axis."""

    height_mm: float
    cross_section_mm2: float
    support_area_mm2: float
    horizontal_area_mm2: float
    footprint_mm2: float
    fits: bool
    score: float
    tilt_deg: float

    def describe(self) -> str:
        return (
            f"height {self.height_mm:6.1f} mm   "
            f"max section ~{self.cross_section_mm2:8.1f} mm2   "
            f"to support {self.support_area_mm2:8.1f} mm2   "
            f"flat faces {self.horizontal_area_mm2:8.1f} mm2   "
            f"tilt {self.tilt_deg:5.1f} deg"
            + ("" if self.fits else "   DOES NOT FIT")
        )


def fibonacci_directions(count: int) -> np.ndarray:
    """Directions distributed uniformly on the sphere (Fibonacci spiral)."""
    indices = np.arange(count, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / count)
    theta = np.pi * (1.0 + 5.0**0.5) * indices
    return np.column_stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    )


def candidate_directions(mesh: trimesh.Trimesh, samples: int = 256, facets: int = 24) -> np.ndarray:
    """Directions to try: uniform sampling plus the large flat faces.

    Sampling alone risks narrowly missing the "large face resting on the
    plate" pose, which is often the right one or the one to discard
    deliberately. The normals of the largest faces are therefore added
    explicitly, in both directions.
    """
    directions = [fibonacci_directions(samples)]

    if len(mesh.facets):
        areas = np.asarray(mesh.facets_area)
        order = np.argsort(areas)[::-1][:facets]
        normals = np.asarray(mesh.facets_normal)[order]
        directions.append(normals)
        directions.append(-normals)

    stacked = np.vstack(directions)
    norms = np.linalg.norm(stacked, axis=1)
    stacked = stacked[norms > 1e-9] / norms[norms > 1e-9, None]

    # deduplicate: two directions within ~1 degree are the same pose
    keep: list[np.ndarray] = []
    for direction in stacked:
        if not any(float(np.dot(direction, k)) > 0.9998 for k in keep):
            keep.append(direction)
    return np.array(keep)


def hull_shadow_areas(mesh: trimesh.Trimesh, directions: np.ndarray) -> np.ndarray:
    """Shadow area of the convex hull along each build direction.

    A layer's peel force is proportional to its area, so the quantity that
    really matters is the maximum cross-section. Computing it exactly would mean
    slicing for every candidate pose; the convex hull shadow is an upper bound
    that is computed in an instant and ranks the poses the same way: a bar
    lying down projects its whole length, the same bar standing up projects
    only its cross-section.
    """
    try:
        hull_points = np.asarray(mesh.convex_hull.vertices, dtype=np.float64)
    except Exception:
        hull_points = np.asarray(mesh.vertices, dtype=np.float64)

    areas = np.empty(len(directions), dtype=np.float64)
    for i, up in enumerate(directions):
        u, v = _plane_basis(up)
        projected = hull_points @ np.column_stack([u, v])
        try:
            # in 2D scipy calls the enclosed area "volume"
            areas[i] = ConvexHull(projected).volume
        except (QhullError, ValueError):
            extent = projected.max(axis=0) - projected.min(axis=0)
            areas[i] = float(extent[0] * extent[1])
    return areas


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _batched_dot(matrix: np.ndarray, directions: np.ndarray):
    """Yield (direction_block, products) while keeping memory under control."""
    per_batch = max(1, int(_BATCH_BYTES / (matrix.shape[0] * 8)))
    for start in range(0, len(directions), per_batch):
        block = directions[start : start + per_batch]
        yield block, matrix @ block.T


def evaluate_directions(
    mesh: trimesh.Trimesh,
    directions: np.ndarray,
    build_volume: tuple[float, float, float] | None = None,
    weights: Weights | None = None,
) -> list[Orientation]:
    """Evaluate each candidate direction and return the poses sorted by score."""
    weights = weights or Weights()

    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    total_area = float(areas.sum())
    if total_area <= 0:
        return []

    diagonal = float(np.linalg.norm(mesh.extents)) or 1.0
    original_up = np.array([0.0, 0.0, 1.0])

    support_all: list[np.ndarray] = []
    horizontal_all: list[np.ndarray] = []
    for _block, dots in _batched_dot(normals, directions):
        # -n.u > 0 means a downward-facing face: its projected area is
        # exactly what needs supporting.
        downward = np.clip(-dots, 0.0, None)
        support_all.append(areas @ downward)
        horizontal_all.append(areas @ (np.abs(dots) ** _HORIZONTAL_SHARPNESS))

    support = np.concatenate(support_all)
    horizontal = np.concatenate(horizontal_all)

    heights_all: list[np.ndarray] = []
    for _block, projected in _batched_dot(vertices, directions):
        heights_all.append(projected.max(axis=0) - projected.min(axis=0))
    heights = np.concatenate(heights_all)

    cross = hull_shadow_areas(mesh, directions)
    cross_reference = float(cross.max()) or 1.0

    scores = (
        weights.cross_section * (cross / cross_reference)
        + weights.support * (support / total_area)
        + weights.horizontal * (horizontal / total_area)
        + weights.height * (heights / diagonal)
    )

    results: list[Orientation] = []
    for i, direction in enumerate(directions):
        fits = True
        if build_volume is not None:
            fits = heights[i] <= build_volume[2]
        tilt = float(np.degrees(np.arccos(np.clip(abs(np.dot(direction, original_up)), -1.0, 1.0))))
        results.append(
            Orientation(
                up=direction,
                rotation=trimesh.geometry.align_vectors(direction, np.array([0.0, 0.0, 1.0])),
                height_mm=float(heights[i]),
                cross_section_mm2=float(cross[i]),
                support_area_mm2=float(support[i]),
                horizontal_area_mm2=float(horizontal[i]),
                footprint_mm2=0.0,
                fits=fits,
                score=float(scores[i]),
                tilt_deg=tilt,
            )
        )

    results.sort(key=lambda o: (not o.fits, o.score))
    return results


def auto_orient(
    mesh: trimesh.Trimesh,
    build_volume: tuple[float, float, float] | None = None,
    weights: Weights | None = None,
    samples: int = 256,
) -> list[Orientation]:
    directions = candidate_directions(mesh, samples=samples)
    return evaluate_directions(mesh, directions, build_volume=build_volume, weights=weights)


def apply_orientation(mesh: trimesh.Trimesh, orientation: Orientation) -> trimesh.Trimesh:
    """Rotate the mesh into the chosen pose and rest it on the plate (minimum z at 0)."""
    out = mesh.copy()
    out.apply_transform(orientation.rotation)
    lowest = float(out.vertices[:, 2].min())
    out.apply_translation([0.0, 0.0, -lowest])
    return out
