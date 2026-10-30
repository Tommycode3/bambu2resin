import numpy as np
import pytest
import trimesh

from travaso.printers import PHOTON_M3
from travaso.slicer.raster import Grid, rasterize
from travaso.slicer.section import plane_segments, signed_area
from travaso.slicer.stack import SliceSettings, place_on_plate, slice_mesh, summarize

GRID = Grid.from_printer(PHOTON_M3)


def _section(mesh, z=0.0):
    return plane_segments(np.asarray(mesh.vertices), np.asarray(mesh.faces), z)


def _raster_area(mesh, z=0.0, subsamples=4, grid=GRID):
    img = rasterize(_section(mesh, z), grid, subsamples=subsamples)
    return float(img.astype(np.float64).sum()) / 255.0 * grid.area_mm2


def test_contours_are_oriented_the_right_way():
    """If the segments ran the other way the area would be negative."""
    cyl = trimesh.creation.cylinder(radius=10.0, height=20.0, sections=256)
    assert signed_area(_section(cyl)) > 0


def test_the_hole_of_a_ring_stays_empty():
    ring = trimesh.creation.annulus(r_min=4.0, r_max=10.0, height=20.0, sections=512)
    ring.apply_translation([60.0, 40.0, 0.0])
    expected = np.pi * (10.0**2 - 4.0**2)
    assert _raster_area(ring) == pytest.approx(expected, rel=2e-3)


def test_overlapping_bodies_merge_instead_of_cancelling():
    """With the even-odd rule the shared region would be emptied: two 10 mm
    cubes overlapping by half would give 100 mm2 instead of 150."""
    a = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    a.apply_translation([60.0, 40.0, 0.0])
    b = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    b.apply_translation([65.0, 40.0, 0.0])
    union = trimesh.util.concatenate([a, b])
    assert _raster_area(union) == pytest.approx(150.0, rel=1e-4)


def test_mirroring_actually_flips_the_image():
    box = trimesh.creation.box(extents=(20.0, 20.0, 10.0))
    box.apply_translation([20.0, 40.0, 0.0])  # off-centre, so the flip is visible

    straight = Grid(GRID.width, GRID.height, GRID.pixel_w_mm, GRID.pixel_h_mm, mirror_x=False)
    mirrored = Grid(GRID.width, GRID.height, GRID.pixel_w_mm, GRID.pixel_h_mm, mirror_x=True)

    a = rasterize(_section(box), straight, 1)
    b = rasterize(_section(box), mirrored, 1)
    assert not np.array_equal(a, b)
    np.testing.assert_array_equal(a[:, ::-1], b)


def test_sliced_volume_of_a_cylinder():
    cyl = place_on_plate(
        trimesh.creation.cylinder(radius=10.0, height=20.0, sections=512), PHOTON_M3
    )
    stats, _ = summarize(slice_mesh(cyl, PHOTON_M3, SliceSettings(0.05)), 0.05)
    assert stats.layers == 400
    assert stats.volume_mm3 == pytest.approx(np.pi * 100.0 * 20.0, rel=2e-3)
    assert stats.max_area_mm2 == pytest.approx(np.pi * 100.0, rel=2e-3)


def test_layers_are_sampled_at_half_thickness():
    box = place_on_plate(trimesh.creation.box(extents=(10.0, 10.0, 1.0)), PHOTON_M3)
    layers = list(slice_mesh(box, PHOTON_M3, SliceSettings(0.1)))
    assert layers[0].z_mm == pytest.approx(0.05)
    assert layers[1].z_mm == pytest.approx(0.15)


def test_a_cone_has_its_largest_section_at_the_bottom():
    cone = trimesh.creation.cone(radius=8.0, height=16.0, sections=256)
    cone = place_on_plate(cone, PHOTON_M3)
    stats, areas = summarize(slice_mesh(cone, PHOTON_M3, SliceSettings(0.1)), 0.1)
    assert stats.max_area_layer < len(areas) // 10
    assert areas[0] > areas[-1]


def test_implausible_layer_height_rejected():
    with pytest.raises(ValueError):
        SliceSettings(layer_height_mm=2.0)


def test_empty_mesh_produces_no_layers():
    empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)
    assert list(slice_mesh(empty, PHOTON_M3)) == []


def test_an_inverted_primitive_subtracts_instead_of_adding():
    """This is how cavities and holes are made without boolean operations: the
    part counts +1, the inverted primitive -1, and where they overlap 0 remains."""
    block = trimesh.creation.box(extents=(20.0, 20.0, 10.0))
    block.apply_translation([60.0, 40.0, 0.0])

    notch = trimesh.creation.box(extents=(6.0, 6.0, 10.0))
    notch.apply_translation([70.0, 40.0, 0.0])  # half inside, half outside
    notch.invert()

    combined = trimesh.util.concatenate([block, notch])
    area = _raster_area(combined)

    # 400 mm2 minus the half of the notch that lies inside (3x6 = 18)
    assert area == pytest.approx(400.0 - 18.0, rel=1e-3)


def test_the_inverted_part_outside_the_body_does_not_become_material():
    """With the 'non-zero winding' rule that region would be filled."""
    notch = trimesh.creation.box(extents=(6.0, 6.0, 10.0))
    notch.apply_translation([60.0, 40.0, 0.0])
    notch.invert()
    assert _raster_area(notch) == pytest.approx(0.0, abs=1e-6)


def test_an_inner_cavity_stays_empty():
    """Outer shell plus inverted inner shell: that is hollowing."""
    outer = trimesh.creation.box(extents=(20.0, 20.0, 10.0))
    outer.apply_translation([60.0, 40.0, 0.0])
    inner = trimesh.creation.box(extents=(16.0, 16.0, 6.0))
    inner.apply_translation([60.0, 40.0, 0.0])
    inner.invert()

    shell = trimesh.util.concatenate([outer, inner])
    assert _raster_area(shell) == pytest.approx(400.0 - 256.0, rel=1e-3)


def test_no_row_disappears_when_a_vertex_falls_on_the_sampled_row():
    """Regression, and it applies to every print, not only to hollowing.

    The two triangles sharing an edge traverse it in opposite directions:
    interpolating the crossing gave points differing in the last bit, the
    contour was left with a microscopic gap, and the sampling row falling
    into it disappeared entirely. On a cube, whose vertices are aligned with
    the grid, this happened on sixteen rows.
    """
    step = 0.3
    cube = trimesh.creation.box(extents=(35.0, 35.0, 35.0))
    origin = cube.bounds[0] - 2 * step
    counts = np.ceil((cube.bounds[1] + 2 * step - origin) / step).astype(int) + 1

    grid = Grid(
        width=int(counts[0]), height=int(counts[1]), pixel_w_mm=step, pixel_h_mm=step
    )
    vertices = np.asarray(cube.vertices) - origin
    faces = np.asarray(cube.faces)

    lost = 0
    for k in range(int(counts[2])):
        image = rasterize(plane_segments(vertices, faces, (k + 0.5) * step), grid, subsamples=1)
        if not image.any():
            continue
        filled = image.any(axis=1)
        occupied = np.flatnonzero(filled)
        lost += sum(1 for r in range(occupied[0], occupied[-1] + 1) if not filled[r])

    assert lost == 0, f"{lost} rows disappeared inside the part"


def test_an_edge_shared_by_two_triangles_gives_the_same_point():
    """The crossing point must not depend on which triangle asks for it."""
    cube = trimesh.creation.box(extents=(35.0, 35.0, 35.0))
    segments = plane_segments(np.asarray(cube.vertices), np.asarray(cube.faces), 0.0)

    # endpoints shared between adjacent segments must match exactly
    endpoints = np.vstack([segments[:, 0], segments[:, 1]])
    unique = np.unique(endpoints, axis=0)
    # each endpoint appears twice (end of one segment, start of the next)
    assert len(unique) == len(endpoints) // 2, (
        "some shared endpoints differ, the contour has microscopic gaps"
    )
