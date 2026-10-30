import numpy as np
import pytest
import trimesh

from travaso.geometry.holes import (
    count_nonmanifold_edges,
    fill_planar_holes,
    find_boundary_loops,
)


def _remove_top_cap(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove the top face, as a CAD export that loses planar faces does."""
    top = mesh.vertices[:, 2].max()
    normals = mesh.face_normals
    centers = mesh.triangles_center
    cap = (normals[:, 2] > 0.99) & (centers[:, 2] > top - 1e-6)
    assert cap.any(), "no top face found"
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[~cap], process=False)


def test_closes_a_missing_planar_face():
    box = trimesh.creation.box(extents=(10.0, 20.0, 5.0))
    opened = _remove_top_cap(box)
    assert not opened.is_watertight

    closed, report = fill_planar_holes(opened)
    trimesh.repair.fix_normals(closed)

    assert closed.is_watertight
    assert report.loops_filled == 1
    assert closed.volume == pytest.approx(box.volume, rel=1e-6)


def test_holed_face_is_not_plugged_in_the_centre():
    """Two coplanar, nested loops are the outer border and hole of the same
    face: closing them separately would also fill the hole."""
    ring = trimesh.creation.annulus(r_min=3.0, r_max=8.0, height=4.0)
    opened = _remove_top_cap(ring)

    loops = find_boundary_loops(opened)
    assert len(loops) == 2, "expected the outer border and the hole border"

    closed, report = fill_planar_holes(opened)
    trimesh.repair.fix_normals(closed)

    assert report.loops_treated_as_holes == 1
    assert closed.is_watertight
    # if the hole had been plugged the volume would grow by pi*r^2*h
    assert closed.volume == pytest.approx(ring.volume, rel=1e-6)


def test_never_worsens_an_already_broken_mesh():
    """The repair is forbidden from increasing non-manifold edges."""
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    opened = _remove_top_cap(box)
    # add a stray face that shares an existing edge
    broken = trimesh.Trimesh(
        vertices=np.vstack([opened.vertices, [[0.0, 0.0, 40.0]]]),
        faces=np.vstack([opened.faces, [[0, 1, len(opened.vertices)]]]),
        process=False,
    )
    before = count_nonmanifold_edges(broken)
    repaired, _ = fill_planar_holes(broken)
    assert count_nonmanifold_edges(repaired) <= before


def test_already_closed_mesh_stays_intact():
    box = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    out, report = fill_planar_holes(box)
    assert report.loops_found == 0
    assert report.faces_added == 0
    assert len(out.faces) == len(box.faces)
