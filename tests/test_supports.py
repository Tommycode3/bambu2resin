import numpy as np
import pytest
import trimesh

from travaso.printers import PHOTON_M3
from travaso.slicer.raster import Grid
from travaso.slicer.stack import SliceSettings, place_on_plate, slice_mesh
from travaso.supports.build import (
    SupportSettings,
    _frustum,
    build_raft,
    generate,
    overhang_points,
    thin_points,
)
from travaso.supports.islands import find_islands

GRID = Grid.from_printer(PHOTON_M3)
STEP = SliceSettings(0.2)  # thick layers: topology matters here, not fineness


def _post_with_side_sphere():
    """The bottom of the sphere appears detached from the post: a textbook island."""
    post = trimesh.creation.box(extents=(6.0, 6.0, 24.0))
    post.apply_translation([0.0, 0.0, 12.0])
    sphere = trimesh.creation.icosphere(radius=5.0, subdivisions=3)
    sphere.apply_translation([6.0, 0.0, 18.0])
    return place_on_plate(trimesh.util.concatenate([post, sphere]), PHOTON_M3)


def test_pixel_to_plate_conversion_undoes_mirroring():
    """Anyone reading a position from the pixels must undo the mirror, otherwise
    they get the point reflected about the centre of the plate."""
    straight = Grid(100, 50, 0.04, 0.04, mirror_x=False)
    mirrored = Grid(100, 50, 0.04, 0.04, mirror_x=True)

    assert straight.to_plate_mm(0, 0) == pytest.approx((0.02, 0.02))
    assert mirrored.to_plate_mm(0, 0) == pytest.approx((99.5 * 0.04, 0.02))
    # reflection about the centre
    x_straight = straight.to_plate_mm(30, 10)[0]
    x_mirrored = mirrored.to_plate_mm(30, 10)[0]
    assert x_straight + x_mirrored == pytest.approx(100 * 0.04)


def test_island_detected_at_the_right_spot():
    """Regression: the centroid was reported in image coordinates, i.e.
    reflected, and the support ended up on the wrong side of the part."""
    model = _post_with_side_sphere()
    islands = list(find_islands(slice_mesh(model, PHOTON_M3, STEP), GRID, min_area_mm2=0.02))

    assert len(islands) == 1
    # the sphere's centre is 2 mm past the centre of the model's bounding box
    expected_x = PHOTON_M3.display_width_mm / 2.0 + 2.0
    assert islands[0].centroid_mm[0] == pytest.approx(expected_x, abs=0.3)
    assert islands[0].centroid_mm[1] == pytest.approx(PHOTON_M3.display_height_mm / 2.0, abs=0.3)


def test_a_shape_growing_from_below_has_no_islands():
    for solid in (
        trimesh.creation.cylinder(radius=6.0, height=12.0, sections=64),
        trimesh.creation.icosphere(radius=6.0, subdivisions=3),
        trimesh.creation.cone(radius=6.0, height=12.0, sections=64),
    ):
        placed = place_on_plate(solid, PHOTON_M3)
        islands = list(find_islands(slice_mesh(placed, PHOTON_M3, STEP), GRID, min_area_mm2=0.05))
        assert islands == [], f"{solid.__class__.__name__}: unexpected islands {islands}"


def test_supports_remove_the_island():
    model = _post_with_side_sphere()
    islands = list(find_islands(slice_mesh(model, PHOTON_M3, STEP), GRID, min_area_mm2=0.02))
    assert islands

    supported, report = generate(model, islands, SupportSettings())
    assert report.pillars >= len(islands)

    placed = place_on_plate(supported, PHOTON_M3, lift_mm=0.0)
    remaining = list(find_islands(slice_mesh(placed, PHOTON_M3, STEP), GRID, min_area_mm2=0.02))
    assert remaining == [], f"islands left after supports: {remaining}"


def test_the_raft_widens_the_first_layer():
    model = _post_with_side_sphere()
    islands = list(find_islands(slice_mesh(model, PHOTON_M3, STEP), GRID, min_area_mm2=0.02))
    supported, _ = generate(model, islands, SupportSettings())
    placed = place_on_plate(supported, PHOTON_M3, lift_mm=0.0)

    first = next(iter(slice_mesh(placed, PHOTON_M3, STEP))).area_mm2
    assert first > 30.0, "without a wide raft the part detaches from the plate"


def test_the_frustum_is_oriented_outward():
    """With winding-number filling an inverted primitive would subtract
    material instead of adding it."""
    sections, r_base, r_tip, height = 12, 0.5, 0.2, 1.5
    f = _frustum(r_base, r_tip, height, sections)
    assert f.is_watertight
    assert f.volume > 0

    # Exact volume of an inscribed n-sided pyramid frustum:
    # (n/2) sin(2pi/n) * h/3 * (r1^2 + r1 r2 + r2^2)
    expected = (
        (sections / 2.0)
        * np.sin(2.0 * np.pi / sections)
        * height
        / 3.0
        * (r_base**2 + r_base * r_tip + r_tip**2)
    )
    assert f.volume == pytest.approx(expected, rel=1e-9)


def test_the_raft_is_solid_and_oriented():
    raft, area = build_raft(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]), SupportSettings())
    assert raft is not None and raft.is_watertight and raft.volume > 0
    assert area > 100.0  # the margin enlarges the starting square


def test_thinning_respects_the_minimum_distance():
    points = np.random.default_rng(4).random((400, 3)) * 10.0
    kept = thin_points(points, 2.0)
    from scipy.spatial.distance import pdist

    assert pdist(kept).min() >= 2.0 - 1e-9


def test_overhang_on_a_horizontal_downward_face():
    """A cube resting flat has its bottom face to support; a cone does not."""
    cube = trimesh.creation.box(extents=(20.0, 20.0, 5.0))
    points = overhang_points(cube, SupportSettings(spacing_mm=4.0))
    assert len(points) > 5
    assert points[:, 2].max() < cube.bounds[0][2] + 1e-6, "points must lie on the bottom face"


def test_inconsistent_parameters_rejected():
    with pytest.raises(ValueError):
        SupportSettings(tip_diameter_mm=2.0, pillar_diameter_mm=1.0)
    with pytest.raises(ValueError):
        SupportSettings(lift_mm=1.0, raft_thickness_mm=1.2)


def test_the_iterative_solver_ends_without_islands():
    """One round is not enough: lifting the part creates islands that were not
    there before, because the bottom used to rest on the plate."""
    from travaso.supports.solve import support_until_clean

    model = _post_with_side_sphere()
    rounds = []
    assembly, report = support_until_clean(
        model,
        PHOTON_M3,
        STEP,
        SupportSettings(),
        on_round=lambda g, p, i: rounds.append((g, p, i)),
    )

    assert report.clean, f"islands left: {report.remaining}"
    assert report.rounds >= 1
    assert rounds[-1][2] == 0
    # the part has been lifted and the raft is underneath
    assert assembly.bounds[0][2] == pytest.approx(0.0, abs=1e-6)


def test_the_part_is_lifted_off_the_plate():
    from travaso.supports.solve import lowest_points, support_until_clean

    settings = SupportSettings(lift_mm=5.0)
    model = _post_with_side_sphere()
    assembly, _ = support_until_clean(model, PHOTON_M3, STEP, settings)

    # the original model must no longer touch the plate: below it there is only
    # raft and pillars, which are the added bodies
    assert assembly.bounds[1][2] > model.bounds[1][2], "the assembly must be taller than the part"


def test_lowest_points_lie_on_the_bottom():
    from travaso.supports.solve import lowest_points

    cone = trimesh.creation.cone(radius=5.0, height=10.0, sections=32)
    points = lowest_points(cone, SupportSettings())
    assert len(points) > 0
    assert points[:, 2].max() <= cone.bounds[0][2] + 1.0
