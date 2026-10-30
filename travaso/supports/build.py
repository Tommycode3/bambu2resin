"""Support and raft generation for resin printing.

A support has three parts: a thin tip that touches the part and must break
off cleanly, a pillar that carries the load, and the raft that keeps
everything attached to the build plate. The tip is the compromise: too thin
and it does not hold, too thick and it leaves a crater to sand down.

Primitives are not merged with boolean operations. The slicer fill works by
winding number, so interpenetrating bodies merge on their own: each primitive
just has to be oriented outward, otherwise it would subtract instead of add.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .islands import Island


@dataclass
class SupportSettings:
    lift_mm: float = 5.0
    """How far to lift the part off the build plate to make room for the supports."""

    tip_diameter_mm: float = 0.4
    tip_length_mm: float = 1.5
    pillar_diameter_mm: float = 1.0
    penetration_mm: float = 0.15
    """How far the tip enters the part: without it, the contact is a single face."""

    raft_thickness_mm: float = 1.2
    raft_margin_mm: float = 2.5

    overhang_angle_deg: float = 45.0
    """Below this inclination from horizontal the surface needs support."""

    spacing_mm: float = 4.0
    sections: int = 10

    def __post_init__(self) -> None:
        if self.tip_diameter_mm >= self.pillar_diameter_mm:
            raise ValueError("the tip must be thinner than the pillar")
        if self.lift_mm <= self.raft_thickness_mm:
            raise ValueError("the lift must exceed the raft thickness")


@dataclass
class SupportReport:
    islands: int = 0
    overhang_points: int = 0
    pillars: int = 0
    landed_on_model: int = 0
    raft_area_mm2: float = 0.0
    added_volume_mm3: float = 0.0
    notes: list[str] = field(default_factory=list)


def _frustum(r_bottom: float, r_top: float, height: float, sections: int) -> trimesh.Trimesh:
    """Truncated cone with its base at z=0, oriented outward."""
    angle = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    bottom = np.column_stack([r_bottom * np.cos(angle), r_bottom * np.sin(angle), np.zeros(sections)])
    top = np.column_stack([r_top * np.cos(angle), r_top * np.sin(angle), np.full(sections, height)])
    vertices = np.vstack([bottom, top, [[0.0, 0.0, 0.0]], [[0.0, 0.0, height]]])

    centre_bottom = 2 * sections
    centre_top = 2 * sections + 1
    faces = []
    for i in range(sections):
        j = (i + 1) % sections
        faces.append([i, j, sections + j])
        faces.append([i, sections + j, sections + i])
        faces.append([centre_bottom, j, i])
        faces.append([centre_top, sections + i, sections + j])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int64), process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def overhang_points(mesh: trimesh.Trimesh, settings: SupportSettings) -> np.ndarray:
    """Points to support on surfaces that are not steep enough.

    A vertical wall holds by itself; a nearly horizontal downward-facing
    surface does not. The threshold is expressed as an angle from the plane of
    the build plate, and is equivalent to a threshold on the vertical
    component of the normal.
    """
    normals = np.asarray(mesh.face_normals)
    areas = np.asarray(mesh.area_faces)
    limit = -np.cos(np.radians(settings.overhang_angle_deg))
    downward = np.flatnonzero(normals[:, 2] <= limit)
    if len(downward) == 0:
        return np.zeros((0, 3))

    # One point per spacing^2 of projected surface, randomly distributed but
    # reproducibly.
    projected = areas[downward] * np.abs(normals[downward, 2])
    target = max(1, int(projected.sum() / (settings.spacing_mm**2)))
    rng = np.random.default_rng(12345)
    chosen = rng.choice(downward, size=min(target * 4, len(downward) * 4), p=projected / projected.sum())

    triangles = mesh.triangles[chosen]
    weights = rng.dirichlet(np.ones(3), size=len(chosen))
    points = np.einsum("ij,ijk->ik", weights, triangles)

    return thin_points(points, settings.spacing_mm)


def thin_points(points: np.ndarray, spacing: float) -> np.ndarray:
    """Thin out the points keeping one per `spacing`, lowest first."""
    if len(points) == 0:
        return points
    order = np.argsort(points[:, 2])  # lower points take precedence
    kept: list[int] = []
    tree_points: list[np.ndarray] = []
    for index in order:
        candidate = points[index]
        if tree_points:
            distances = np.linalg.norm(np.asarray(tree_points) - candidate, axis=1)
            if distances.min() < spacing:
                continue
        kept.append(int(index))
        tree_points.append(candidate)
    return points[kept]


def _landing_z(
    mesh: trimesh.Trimesh, point: np.ndarray, raft_top: float, settings: SupportSettings
) -> tuple[float, bool]:
    """Find where the pillar lands: on the part below or on the raft."""
    origin = point + np.array([0.0, 0.0, -settings.tip_length_mm - 1e-3])
    try:
        locations, _index_ray, _index_tri = mesh.ray.intersects_location(
            ray_origins=origin[None, :], ray_directions=np.array([[0.0, 0.0, -1.0]])
        )
    except Exception:
        return raft_top, False

    if len(locations):
        below = locations[locations[:, 2] < origin[2]]
        if len(below):
            highest = float(below[:, 2].max())
            if highest > raft_top:
                return highest - settings.penetration_mm, True
    return raft_top, False


def build_supports(
    mesh: trimesh.Trimesh, points: np.ndarray, settings: SupportSettings
) -> tuple[list[trimesh.Trimesh], np.ndarray, int]:
    """Build tips and pillars for each point to support."""
    raft_top = settings.raft_thickness_mm
    pieces: list[trimesh.Trimesh] = []
    feet: list[np.ndarray] = []
    landed = 0

    tip_r = settings.tip_diameter_mm / 2.0
    pillar_r = settings.pillar_diameter_mm / 2.0

    for point in points:
        contact_z = float(point[2]) + settings.penetration_mm
        tip_base_z = contact_z - settings.tip_length_mm
        bottom_z, on_model = _landing_z(mesh, point, raft_top, settings)
        if on_model:
            landed += 1
        if tip_base_z <= bottom_z:
            # Not enough room for the tip: only a small pillar is placed
            tip_base_z = bottom_z

        tip = _frustum(pillar_r, tip_r, max(contact_z - tip_base_z, 1e-3), settings.sections)
        tip.apply_translation([point[0], point[1], tip_base_z])
        pieces.append(tip)

        height = tip_base_z - bottom_z
        if height > 1e-3:
            pillar = trimesh.creation.cylinder(
                radius=pillar_r, height=height, sections=settings.sections
            )
            pillar.apply_translation([point[0], point[1], bottom_z + height / 2.0])
            pieces.append(pillar)

        feet.append(np.array([point[0], point[1]]))

    return pieces, (np.array(feet) if feet else np.zeros((0, 2))), landed


def build_raft(
    footprint_xy: np.ndarray, settings: SupportSettings
) -> tuple[trimesh.Trimesh | None, float]:
    """Base plate that keeps everything attached to the build plate."""
    from shapely.geometry import MultiPoint

    if len(footprint_xy) == 0:
        return None, 0.0

    shape = MultiPoint([tuple(p) for p in footprint_xy]).convex_hull.buffer(settings.raft_margin_mm)
    if shape.is_empty or shape.area <= 0:
        return None, 0.0

    raft = trimesh.creation.extrude_polygon(shape, height=settings.raft_thickness_mm)
    if raft.volume < 0:
        raft.invert()
    return raft, float(shape.area)


def generate(
    mesh: trimesh.Trimesh,
    islands: list[Island],
    settings: SupportSettings | None = None,
) -> tuple[trimesh.Trimesh, SupportReport]:
    """Add supports and raft to a mesh already oriented and resting at z=0.

    The part is lifted by `lift_mm`: that is the space the supports fill.
    """
    settings = settings or SupportSettings()
    report = SupportReport(islands=len(islands))

    lifted = mesh.copy()
    lifted.apply_translation([0.0, 0.0, settings.lift_mm - float(mesh.bounds[0][2])])

    # Islands come from the layer stack of the NON-lifted part: their height
    # must be converted to the lifted part's frame.
    island_points = np.array(
        [[i.centroid_mm[0], i.centroid_mm[1], i.z_mm + settings.lift_mm] for i in islands]
    ).reshape(-1, 3)

    hangs = overhang_points(lifted, settings)
    report.overhang_points = len(hangs)

    if len(island_points) and len(hangs):
        # Islands are never thinned out: each one must get its own support.
        keep = cKDTree(island_points).query(hangs)[0] > settings.spacing_mm / 2.0
        hangs = hangs[keep]
    points = np.vstack([island_points, hangs]) if len(hangs) else island_points

    if len(points) == 0:
        report.notes.append("no points to support: the part always grows from below")
        return lifted, report

    pieces, feet, landed = build_supports(lifted, points, settings)
    report.pillars = len(points)
    report.landed_on_model = landed

    model_footprint = lifted.vertices[:, :2] if float(lifted.bounds[0][2]) <= settings.raft_thickness_mm else None
    raft_points = feet if model_footprint is None else np.vstack([feet, model_footprint])
    raft, area = build_raft(raft_points, settings)
    report.raft_area_mm2 = area
    if raft is not None:
        pieces.append(raft)

    report.added_volume_mm3 = float(sum(max(p.volume, 0.0) for p in pieces))
    combined = trimesh.util.concatenate([lifted, *pieces])
    return combined, report
