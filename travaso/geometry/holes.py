"""Closing the planar holes left by CAD exports.

trimesh's `fill_holes` only closes holes with three or four edges. What is
actually found in CAD-exported models is different: a few dozen very large
boundary loops (up to a few hundred vertices) that are practically flat, i.e.
planar faces lost along the way during conversion.

Those holes close well by projecting the loop onto its best-fit plane and
triangulating the resulting polygon. Two coplanar, nested loops must not be
closed separately, though: they are the outer and inner boundary of the same
holed face, and closing both would also plug the hole.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Polygon
from trimesh import grouping

# A loop is considered flat if its relative thickness (third singular value
# divided by the first) is below this threshold.
PLANARITY_TOL = 0.08
# Two planes are the same plane if their normals are nearly parallel and the
# offset is below this distance.
COPLANAR_ANGLE_COS = 0.999
COPLANAR_DIST_MM = 0.05


@dataclass
class BoundaryLoop:
    indices: list[int]
    points: np.ndarray
    origin: np.ndarray
    normal: np.ndarray
    planarity: float
    closed: bool = True

    @property
    def size(self) -> int:
        return len(self.indices)

    @property
    def fillable(self) -> bool:
        """Only a closed, flat loop bounds a face that can be rebuilt."""
        return self.closed and self.planarity <= PLANARITY_TOL


@dataclass
class FillReport:
    loops_found: int = 0
    loops_filled: int = 0
    loops_skipped_open: int = 0
    loops_skipped_nonplanar: int = 0
    loops_skipped_invalid: int = 0
    loops_treated_as_holes: int = 0
    faces_added: int = 0
    faces_rejected_duplicate: int = 0
    faces_rejected_nonmanifold: int = 0
    reverted: bool = False
    skipped_detail: list[str] = field(default_factory=list)


def find_boundary_loops(mesh: trimesh.Trimesh) -> list[BoundaryLoop]:
    """Boundary loops, walked over UNORIENTED adjacency.

    Following half-edges in face order only works if the winding is consistent
    across the whole mesh. When it is not (and in real files that happens
    often), boundary half-edges do not form cycles, the walk breaks after one
    step and all edges already traversed are lost. Unoriented adjacency does
    not have this problem; the orientation of the new faces is fixed later, on
    the closed mesh.
    """
    if len(mesh.faces) == 0:
        return []

    idx = grouping.group_rows(mesh.edges_sorted, require_count=1)
    if len(idx) == 0:
        return []

    edges = mesh.edges_sorted[idx]
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for k, (a, b) in enumerate(edges):
        adjacency[int(a)].append((int(b), k))
        adjacency[int(b)].append((int(a), k))

    used = np.zeros(len(edges), dtype=bool)

    # Vertices with a single boundary edge are endpoints of open chains: by
    # starting there the chain is consumed entirely, and the true cycles are
    # left with only degree-two vertices.
    order = sorted(adjacency, key=lambda v: len(adjacency[v]))

    loops: list[BoundaryLoop] = []
    for start in order:
        while True:
            first_step = [(o, k) for o, k in adjacency[start] if not used[k]]
            if not first_step:
                break

            chain = [start]
            current = start
            closed = False
            while True:
                options = [(o, k) for o, k in adjacency[current] if not used[k]]
                if not options:
                    break
                nxt, edge_id = options[0]
                used[edge_id] = True
                if nxt == start:
                    closed = True
                    break
                chain.append(nxt)
                current = nxt

            if len(chain) < 3:
                continue

            pts = mesh.vertices[chain]
            origin = pts.mean(axis=0)
            _u, sigma, vt = np.linalg.svd(pts - origin, full_matrices=False)
            planarity = float(sigma[2] / sigma[0]) if sigma[0] > 0 else 1.0
            loops.append(
                BoundaryLoop(
                    indices=chain,
                    points=pts,
                    origin=origin,
                    normal=vt[2] / np.linalg.norm(vt[2]),
                    planarity=planarity,
                    closed=closed,
                )
            )

    return loops


def _as_polygon(geom) -> Polygon | None:
    """Reduce the result of a shapely cleanup to a simple polygon.

    `buffer(0)` on a loop that self-intersects once projected can return
    several pieces: the largest, which is the real face, is kept and the
    fragments are discarded.
    """
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    if not isinstance(geom, Polygon) or geom.area <= 0:
        return None
    return geom


def count_nonmanifold_edges(mesh: trimesh.Trimesh) -> int:
    """Edges shared by more than two faces.

    They are structural defects: no hole filling fixes them, and a repair that
    increases them has made the mesh worse instead of better.
    """
    if len(mesh.faces) == 0:
        return 0
    return sum(1 for row in grouping.group_rows(mesh.edges_sorted, require_count=None) if len(row) > 2)


def _drop_faces_causing_nonmanifold(
    base_faces: np.ndarray, new_faces: np.ndarray
) -> tuple[np.ndarray, int]:
    """Remove from the new faces those that create non-manifold edges.

    Reverting the whole repair because two triangles out of five hundred are
    wrong throws away the good work. Instead, the edges that end up shared by
    more than two faces after closing are found and only the new faces touching
    them are removed, repeating until the result stabilizes.
    """
    dropped = 0
    for _ in range(4):
        if len(new_faces) == 0:
            return new_faces, dropped

        combined = np.vstack([base_faces, new_faces])
        edges = np.sort(combined[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        groups = grouping.group_rows(edges, require_count=None)

        offset = len(base_faces)
        guilty: set[int] = set()
        for row in groups:
            if len(row) <= 2:
                continue
            faces_on_edge = [int(r) // 3 for r in row]
            # only new faces are at fault: the original ones stay where they were
            for face_index in faces_on_edge:
                if face_index >= offset:
                    guilty.add(face_index - offset)

        if not guilty:
            return new_faces, dropped

        keep = np.ones(len(new_faces), dtype=bool)
        keep[list(guilty)] = False
        dropped += int((~keep).sum())
        new_faces = new_faces[keep]

    return new_faces, dropped


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _group_coplanar(loops: list[BoundaryLoop]) -> list[list[int]]:
    """Group the loops that lie on the same plane."""
    groups: list[list[int]] = []
    for i, loop in enumerate(loops):
        for group in groups:
            ref = loops[group[0]]
            if abs(float(np.dot(ref.normal, loop.normal))) < COPLANAR_ANGLE_COS:
                continue
            if abs(float(np.dot(loop.origin - ref.origin, ref.normal))) > COPLANAR_DIST_MM:
                continue
            group.append(i)
            break
        else:
            groups.append([i])
    return groups


def fill_planar_holes(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, FillReport]:
    """Close the planar holes and return the mesh with the report."""
    report = FillReport()
    loops = find_boundary_loops(mesh)
    report.loops_found = len(loops)
    if not loops:
        return mesh, report

    vertices = [mesh.vertices]
    new_faces: list[np.ndarray] = []
    next_index = len(mesh.vertices)

    for group in _group_coplanar(loops):
        planar = [i for i in group if loops[i].fillable]
        for i in group:
            if loops[i].fillable:
                continue
            if not loops[i].closed:
                report.loops_skipped_open += 1
                report.skipped_detail.append(
                    f"open boundary chain of {loops[i].size} vertices: it does not bound "
                    "a face and cannot be closed reliably"
                )
            else:
                report.loops_skipped_nonplanar += 1
                report.skipped_detail.append(
                    f"non-planar loop of {loops[i].size} vertices (relative thickness "
                    f"{loops[i].planarity:.3f}): left open"
                )
        if not planar:
            continue

        ref = loops[planar[0]]
        u, v = _plane_basis(ref.normal)
        basis = np.column_stack([u, v])

        # Project all coplanar loops onto the same 2D plane
        polys: list[tuple[int, Polygon, np.ndarray]] = []
        for i in planar:
            pts2d = (loops[i].points - ref.origin) @ basis
            poly = Polygon(pts2d)
            if not poly.is_valid:
                poly = _as_polygon(poly.buffer(0))
            if poly is None or not poly.is_valid or poly.area <= 0:
                report.loops_skipped_invalid += 1
                report.skipped_detail.append(
                    f"loop of {loops[i].size} vertices with degenerate projection: left open"
                )
                continue
            polys.append((i, poly, pts2d))

        if not polys:
            continue

        # The largest is the outer contour, the ones it contains are holes
        polys.sort(key=lambda item: item[1].area, reverse=True)
        consumed: set[int] = set()

        for pos, (i, outer, _pts) in enumerate(polys):
            if i in consumed:
                continue
            consumed.add(i)
            holes = []
            hole_loop_ids: list[int] = []
            for j, inner, inner_pts in polys[pos + 1 :]:
                if j in consumed:
                    continue
                if outer.contains(inner):
                    holes.append(inner_pts)
                    hole_loop_ids.append(j)
                    consumed.add(j)
                    report.loops_treated_as_holes += 1

            shell = np.asarray(outer.exterior.coords)[:-1]
            try:
                target = Polygon(shell, list(holes)) if holes else Polygon(shell)
                if not target.is_valid:
                    target = _as_polygon(target.buffer(0))
                    if target is None:
                        raise ValueError("projection cannot be reduced to a simple polygon")
                verts2d, faces2d = trimesh.creation.triangulate_polygon(target)
            except Exception as exc:  # triangulation failed: better open than wrong
                report.loops_skipped_invalid += 1
                report.skipped_detail.append(f"triangulation failed ({exc}): loop left open")
                continue

            # Generated vertices must be mapped back to existing ones, otherwise
            # the mesh stays badly stitched and does not become watertight.
            source_idx: list[int] = []
            source_2d: list[np.ndarray] = []
            for j in [i, *hole_loop_ids]:
                source_idx.extend(loops[j].indices)
                source_2d.append((loops[j].points - ref.origin) @ basis)
            source_pts = np.vstack(source_2d)
            tree = cKDTree(source_pts)

            distances, matches = tree.query(verts2d)
            remap = np.empty(len(verts2d), dtype=np.int64)
            extra: list[np.ndarray] = []
            for k, (dist, match) in enumerate(zip(distances, matches)):
                if dist < 1e-6:
                    remap[k] = source_idx[int(match)]
                else:
                    remap[k] = next_index + len(extra)
                    extra.append(ref.origin + verts2d[k][0] * u + verts2d[k][1] * v)

            if extra:
                vertices.append(np.asarray(extra))
                next_index += len(extra)

            new_faces.append(remap[faces2d])
            report.loops_filled += 1
            report.faces_added += len(faces2d)

    if not new_faces:
        return mesh, report

    # A face that already exists closes nothing: it creates an edge shared by
    # three faces, a defect worse than the hole it was meant to plug.
    candidate = np.vstack(new_faces)
    existing = {tuple(sorted(face)) for face in mesh.faces}
    keep = np.array([tuple(sorted(face)) not in existing for face in candidate], dtype=bool)
    report.faces_rejected_duplicate = int((~keep).sum())
    candidate = candidate[keep]
    report.faces_added = int(len(candidate))
    if len(candidate) == 0:
        return mesh, report

    candidate, dropped = _drop_faces_causing_nonmanifold(mesh.faces, candidate)
    report.faces_rejected_nonmanifold = dropped
    report.faces_added = int(len(candidate))
    if len(candidate) == 0:
        report.loops_filled = 0
        return mesh, report

    filled = trimesh.Trimesh(
        vertices=np.vstack(vertices),
        faces=np.vstack([mesh.faces, candidate]),
        process=False,
    )
    filled.merge_vertices()
    filled.remove_unreferenced_vertices()

    # Last line of defence: a repair must never leave the mesh worse than it
    # found it.
    if count_nonmanifold_edges(filled) > count_nonmanifold_edges(mesh):
        report.reverted = True
        report.loops_filled = 0
        report.faces_added = 0
        report.skipped_detail.append("filling reverted: it would have increased the non-manifold edges")
        return mesh, report

    return filled, report
