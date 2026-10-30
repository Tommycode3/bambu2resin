import numpy as np
import pytest
import trimesh

from travaso.geometry.orient import (
    PRESETS,
    apply_orientation,
    auto_orient,
    fibonacci_directions,
    hull_shadow_areas,
)

VOLUME = (163.84, 102.4, 180.0)


def test_rotation_maps_chosen_direction_onto_z_axis():
    box = trimesh.creation.box(extents=(10.0, 20.0, 5.0))
    for pose in auto_orient(box, VOLUME)[:5]:
        mapped = pose.rotation[:3, :3] @ pose.up
        np.testing.assert_allclose(mapped, [0.0, 0.0, 1.0], atol=1e-9)


def test_a_plate_is_stood_upright():
    """Printed flat, the plate has the largest possible cross-section: in resin
    that is the pose that generates the most peel force."""
    plate = trimesh.creation.box(extents=(40.0, 30.0, 3.0))
    poses = auto_orient(plate, VOLUME)
    best = poses[0]
    flat = next(p for p in poses if p.tilt_deg < 2.0)

    assert best.tilt_deg > 60.0
    assert best.cross_section_mm2 < flat.cross_section_mm2 / 5.0


def test_hull_shadow_of_a_bar():
    bar = trimesh.creation.box(extents=(100.0, 9.0, 12.0))
    areas = hull_shadow_areas(bar, np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))
    assert areas[0] == pytest.approx(100.0 * 9.0, rel=1e-6)  # lying down
    assert areas[1] == pytest.approx(9.0 * 12.0, rel=1e-6)  # standing up


def test_apply_orientation_rests_on_plate():
    mesh = trimesh.creation.icosphere(radius=7.0)
    mesh.apply_translation([12.0, -4.0, 33.0])
    pose = auto_orient(mesh, VOLUME)[0]
    placed = apply_orientation(mesh, pose)

    assert placed.vertices[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    assert placed.extents[2] == pytest.approx(pose.height_mm, rel=1e-9)


def test_poses_too_tall_end_up_last():
    low_volume = (163.84, 102.4, 20.0)
    bar = trimesh.creation.box(extents=(100.0, 9.0, 12.0))
    poses = auto_orient(bar, low_volume)

    assert poses[0].fits, "the first proposed pose must fit in the volume"
    assert all(p.fits for p in poses[: sum(1 for x in poses if x.fits)])
    assert any(not p.fits for p in poses), "a 100 mm bar does not fit upright in 20 mm"


def test_speed_preset_prints_lower():
    bar = trimesh.creation.box(extents=(60.0, 20.0, 8.0))
    reliable = auto_orient(bar, VOLUME, weights=PRESETS["reliability"])[0]
    fast = auto_orient(bar, VOLUME, weights=PRESETS["speed"])[0]
    assert fast.height_mm < reliable.height_mm


def test_fibonacci_directions_are_unit_and_spread():
    d = fibonacci_directions(200)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)
    assert d[:, 2].min() < -0.9 and d[:, 2].max() > 0.9
