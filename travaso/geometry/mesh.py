"""Mesh preparation and repair before resin printing.

In FDM an imperfect mesh often gets through: the slicer closes the contours and
the eye does not notice. In MSLA each layer is an image derived from a
plane-mesh intersection, and a hole in the shell produces an open contour,
i.e. a wrong layer. So here the geometry must be closed before, not after.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from trimesh import grouping

from ..bambu.model import ModelObject
from .holes import FillReport, count_nonmanifold_edges, fill_planar_holes


@dataclass
class MeshReport:
    """What was found and what was fixed."""

    name: str
    faces_before: int = 0
    faces_after: int = 0
    vertices_merged: int = 0
    degenerate_removed: int = 0
    duplicate_removed: int = 0
    holes_filled: int = 0
    boundary_edges_before: int = 0
    boundary_edges_left: int = 0
    nonmanifold_edges: int = 0
    fill: FillReport | None = None
    winding_fixed: bool = False
    normals_inverted: bool = False
    watertight: bool = False
    volume_mm3: float = 0.0
    area_mm2: float = 0.0
    bodies: int = 1
    problems: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Safe to slice: closed shell and positive volume."""
        return self.watertight and self.volume_mm3 > 0

    @property
    def repaired(self) -> bool:
        return bool(
            self.vertices_merged
            or self.degenerate_removed
            or self.duplicate_removed
            or self.holes_filled
            or self.winding_fixed
            or self.normals_inverted
        )

    def summary(self) -> str:
        if self.usable:
            base = f"closed, {self.volume_mm3 / 1000.0:.2f} cm3"
        else:
            base = "NOT CLOSED"
        if self.bodies > 1:
            base += f", {self.bodies} separate shells"
        return base


def to_trimesh(obj: ModelObject) -> trimesh.Trimesh:
    """Convert a project object into an already-positioned trimesh mesh."""
    return trimesh.Trimesh(
        vertices=obj.transformed_vertices(),
        faces=obj.mesh.faces,
        process=False,
    )


def _boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    """Edges belonging to a single face: they are the borders of holes."""
    if len(mesh.faces) == 0:
        return 0
    return len(grouping.group_rows(mesh.edges_sorted, require_count=1))


def prepare(mesh: trimesh.Trimesh, name: str = "", fill: bool = True) -> tuple[trimesh.Trimesh, MeshReport]:
    """Clean up and close a mesh, reporting every change made.

    Order matters. Duplicate vertices must be removed first: without that,
    adjacent triangles are not connected and every edge looks like a border.
    Winding must be made consistent before looking for boundary loops.
    And hole filling comes last, once the topology is clean.
    """
    report = MeshReport(name=name, faces_before=len(mesh.faces))

    mesh.remove_infinite_values()

    v_before = len(mesh.vertices)
    mesh.merge_vertices()
    report.vertices_merged = v_before - len(mesh.vertices)

    f_before = len(mesh.faces)
    mesh.update_faces(mesh.nondegenerate_faces())
    report.degenerate_removed = f_before - len(mesh.faces)

    f_before = len(mesh.faces)
    mesh.update_faces(mesh.unique_faces())
    report.duplicate_removed = f_before - len(mesh.faces)

    mesh.remove_unreferenced_vertices()

    if not mesh.is_winding_consistent:
        trimesh.repair.fix_winding(mesh)
        report.winding_fixed = True

    report.boundary_edges_before = _boundary_edge_count(mesh)

    if fill and not mesh.is_watertight:
        # First the tiny holes, which trimesh closes well and at no cost,
        # then the large planar faces left behind by CAD exports.
        trimesh.repair.fill_holes(mesh)
        mesh, fill_report = fill_planar_holes(mesh)
        report.fill = fill_report
        report.holes_filled = fill_report.loops_filled

    # A negative volume means inward-facing normals: the slicer would fill
    # the air instead of the part.
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()
        report.normals_inverted = True

    report.faces_after = len(mesh.faces)
    report.boundary_edges_left = _boundary_edge_count(mesh)
    report.nonmanifold_edges = count_nonmanifold_edges(mesh)
    report.watertight = bool(mesh.is_watertight)
    report.area_mm2 = float(mesh.area)
    report.volume_mm3 = float(mesh.volume) if mesh.is_watertight else 0.0
    report.bodies = int(mesh.body_count)

    if not report.watertight:
        report.problems.append(
            f"open shell: {report.boundary_edges_left} unclosed boundary edges"
        )
    if report.nonmanifold_edges:
        report.problems.append(
            f"{report.nonmanifold_edges} non-manifold edges (shared by more than two faces): "
            "structural defect, must be fixed in CAD"
        )
    if report.bodies > 1:
        report.problems.append(
            f"{report.bodies} separate shells: they are distinct parts and must be handled individually"
        )
    if len(mesh.faces) == 0:
        report.problems.append("empty mesh")

    return mesh, report


def scale_for_shrinkage(
    mesh: trimesh.Trimesh, shrink_xy_pct: float, shrink_z_pct: float
) -> trimesh.Trimesh:
    """Compensate resin shrinkage by enlarging the model.

    If the resin shrinks by 1%, the part must be printed 1% larger so that it
    returns to size after shrinking: the factor is 1/(1-r), not 1+r. At 1% the
    difference is small, but on long parts and resins that shrink 3% it becomes
    a visible error.
    """
    for value, label in ((shrink_xy_pct, "XY"), (shrink_z_pct, "Z")):
        if not -50.0 < value < 50.0:
            raise ValueError(f"{label} shrinkage of {value}% is outside the plausible range")

    fx = 1.0 / (1.0 - shrink_xy_pct / 100.0)
    fz = 1.0 / (1.0 - shrink_z_pct / 100.0)
    matrix = np.diag([fx, fx, fz, 1.0])
    out = mesh.copy()
    out.apply_transform(matrix)
    return out
