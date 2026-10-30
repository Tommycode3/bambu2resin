import numpy as np
import pytest
import trimesh

from travaso.geometry.hollow import HollowSettings, hollow
from travaso.printers import PHOTON_M3
from travaso.slicer.stack import SliceSettings, place_on_plate, slice_mesh


def _cube(side: float = 20.0) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(side, side, side))


def test_cavity_has_dimensions_of_a_plausible_wall():
    """Regression: without a frame of empty voxels the distance transform had
    no background to measure from and the ENTIRE solid became cavity.

    Exact equality is not required: the cavity is built on a grid, so the wall
    is known to within about one step. We check that the cavity side lies in
    the range compatible with the requested wall.
    """
    side, wall, step = 20.0, 2.0, 0.4
    _, report = hollow(_cube(side), HollowSettings(wall_mm=wall, voxel_mm=step))

    assert report.hollowed
    cavity_side = report.cavity_volume_mm3 ** (1.0 / 3.0)
    minimum = side - 2.0 * (wall + 2.0 * step)
    maximum = side - 2.0 * max(wall - 2.0 * step, 0.0)
    assert minimum <= cavity_side <= maximum, (
        f"cavity side {cavity_side:.2f} outside [{minimum:.2f}, {maximum:.2f}]"
    )
    # the defect this test guards against: a cavity as large as the whole part
    assert report.saved_fraction < 0.75


def test_achieved_thickness_approaches_requested_one():
    for step in (0.5, 0.3, 0.25):
        _, report = hollow(_cube(), HollowSettings(wall_mm=2.0, voxel_mm=step))
        assert abs(report.achieved_wall_mm - 2.0) <= step * 1.5


def test_bounding_box_unchanged_after_hollowing():
    """Drain cylinders must not stick out: they would skew placement on the
    plate and produce empty layers above and below."""
    cube = _cube()
    assembly, report = hollow(cube, HollowSettings(wall_mm=2.0, voxel_mm=0.4))
    assert report.hollowed
    np.testing.assert_allclose(assembly.extents, cube.extents, atol=1e-6)


def test_sliced_part_is_actually_hollow():
    cube = _cube()
    solid = place_on_plate(cube, PHOTON_M3)
    solid_volume = sum(l.area_mm2 for l in slice_mesh(solid, PHOTON_M3, SliceSettings(0.2))) * 0.2

    hollowed, report = hollow(cube, HollowSettings(wall_mm=2.0, voxel_mm=0.4))
    placed = place_on_plate(hollowed, PHOTON_M3)
    hollow_volume = sum(l.area_mm2 for l in slice_mesh(placed, PHOTON_M3, SliceSettings(0.2))) * 0.2

    assert solid_volume == pytest.approx(8000.0, rel=0.01)
    assert hollow_volume < solid_volume * 0.6


def test_without_holes_cavity_is_trapped():
    """A closed cavity stays full of resin and acts as a suction cup: worse than solid."""
    _, with_holes = hollow(_cube(), HollowSettings(wall_mm=2.0, voxel_mm=0.4))
    _, without = hollow(_cube(), HollowSettings(wall_mm=2.0, voxel_mm=0.4, drain_holes=0))

    assert with_holes.drained
    assert not without.drained
    assert any("suction cup" in n for n in without.notes)


def test_saving_threshold_tells_thin_parts_apart():
    """On a thin part the cavity is tiny but the holes stay as large as needed
    for the resin to drain: the part would be drilled to recover a few cubic
    millimetres. The threshold must still discriminate, not reject everything."""
    threshold = HollowSettings(wall_mm=2.0, voxel_mm=0.3, min_saving=0.30)

    slab = trimesh.creation.box(extents=(60.0, 60.0, 4.5))
    out, thin = hollow(slab, threshold)
    assert not thin.hollowed
    assert out is slab
    assert thin.notes

    # the same criterion on a bulky part must hollow it instead
    _, bulky = hollow(_cube(30.0), threshold)
    assert bulky.hollowed
    assert bulky.saved_fraction > 0.30


def test_subtractive_pieces_travel_separately():
    """Inner shell and holes are needed for the assembly but not for overhang
    detection: they would place supports inside the cavity."""
    _, report = hollow(_cube(), HollowSettings(wall_mm=2.0, voxel_mm=0.4))
    assert len(report.subtractive) == 1 + report.drains
    for piece in report.subtractive:
        assert piece.volume < 0, "subtractive bodies must be inverted"


def test_inconsistent_parameters_rejected():
    with pytest.raises(ValueError):
        HollowSettings(wall_mm=0.0)
    with pytest.raises(ValueError):
        # step too coarse: fewer than two voxels of wall
        HollowSettings(wall_mm=1.0, voxel_mm=0.8)
