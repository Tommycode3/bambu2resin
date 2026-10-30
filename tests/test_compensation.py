import numpy as np
import pytest
import trimesh
from scipy import ndimage

from travaso.geometry.mesh import scale_for_shrinkage
from travaso.printers import PHOTON_M3
from travaso.slicer.raster import Grid, rasterize
from travaso.slicer.section import plane_segments
from travaso.slicer.stack import SliceSettings, place_on_plate, slice_mesh

GRID = Grid(width=1400, height=1400, pixel_w_mm=0.04, pixel_h_mm=0.04)


def _radius(mesh, offset_mm=0.0, z=0.0):
    segments = plane_segments(np.asarray(mesh.vertices), np.asarray(mesh.faces), z)
    image = rasterize(segments, GRID, 4, offset_mm=offset_mm)
    area = float(image.astype(np.float64).sum()) / 255.0 * GRID.area_mm2
    return np.sqrt(area / np.pi)


def test_edge_moves_by_requested_amount():
    cylinder = trimesh.creation.cylinder(radius=10.0, height=6.0, sections=1024)
    cylinder.apply_translation([24.0, 24.0, 0.0])

    for offset in (0.0, -0.05, -0.10, 0.10):
        expected = 10.0 + offset
        # the distance transform is discrete: the error stays below a tenth of a pixel
        assert _radius(cylinder, offset) == pytest.approx(expected, abs=0.012)


def test_shrinking_the_part_widens_the_holes():
    """This is why compensation is needed: light bleeding past the edge
    narrows holes and fattens pins, so fits designed for FDM no longer go
    together."""
    plate = trimesh.creation.annulus(r_min=2.5, r_max=12.0, height=6.0, sections=1024)
    plate.apply_translation([28.0, 28.0, 0.0])
    segments = plane_segments(np.asarray(plate.vertices), np.asarray(plate.faces), 0.0)

    def hole_diameter(offset):
        image = rasterize(segments, GRID, 4, offset_mm=offset)
        solid = image > 127
        labels, count = ndimage.label(~solid)
        border = set(
            np.unique(
                np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
            ).tolist()
        )
        hole = np.isin(labels, [j for j in range(1, count + 1) if j not in border])
        return 2.0 * np.sqrt(hole.sum() * GRID.area_mm2 / np.pi)

    assert hole_diameter(0.0) == pytest.approx(5.0, abs=0.01)
    assert hole_diameter(-0.05) == pytest.approx(5.1, abs=0.01)
    assert hole_diameter(-0.10) == pytest.approx(5.2, abs=0.01)


def test_elephant_foot_fades_instead_of_vanishing_abruptly():
    settings = SliceSettings(0.05, bleed_mm=0.02, elephant_foot_mm=0.2, elephant_foot_layers=10)
    offsets = [settings.offset_for_layer(i) for i in range(12)]

    assert offsets[0] == pytest.approx(-0.22)  # bleed plus full elephant foot
    assert offsets[5] == pytest.approx(-0.02 - 0.1)  # halfway through the fade
    assert offsets[10] == pytest.approx(-0.02)  # done: only the bleed remains
    assert offsets[11] == pytest.approx(-0.02)
    assert all(a <= b for a, b in zip(offsets, offsets[1:])), "must increase monotonically"


def test_shrinkage_uses_reciprocal_not_sum():
    """If the resin shrinks by 1% the part must be made 1/(1-0.01) larger,
    not 1.01: at 1% it changes little, at 3% and on a long part it shows."""
    cube = trimesh.creation.box(extents=(100.0, 100.0, 100.0))
    large = scale_for_shrinkage(cube, 3.0, 3.0)

    expected = 100.0 / (1.0 - 0.03)
    assert large.extents[0] == pytest.approx(expected, rel=1e-9)
    assert large.extents[0] != pytest.approx(103.0, rel=1e-6)


def test_shrinkage_can_differ_between_xy_and_z():
    cube = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    out = scale_for_shrinkage(cube, 2.0, 0.0)
    assert out.extents[0] == pytest.approx(10.0 / 0.98, rel=1e-9)
    assert out.extents[2] == pytest.approx(10.0, rel=1e-9)


def test_compensations_reach_the_layers():
    cylinder = place_on_plate(
        trimesh.creation.cylinder(radius=8.0, height=3.0, sections=512), PHOTON_M3
    )
    without = list(slice_mesh(cylinder, PHOTON_M3, SliceSettings(0.05)))
    with_comp = list(
        slice_mesh(
            cylinder,
            PHOTON_M3,
            SliceSettings(0.05, bleed_mm=0.05, elephant_foot_mm=0.2, elephant_foot_layers=8),
        )
    )

    r = lambda layer: np.sqrt(layer.area_mm2 / np.pi)
    assert r(without[0]) == pytest.approx(8.0, abs=0.01)
    assert r(with_comp[0]) == pytest.approx(8.0 - 0.25, abs=0.02)  # bleed + elephant foot
    assert r(with_comp[30]) == pytest.approx(8.0 - 0.05, abs=0.02)  # bleed only


def test_implausible_parameters_rejected():
    with pytest.raises(ValueError):
        SliceSettings(0.05, bleed_mm=5.0)
    with pytest.raises(ValueError):
        SliceSettings(0.05, elephant_foot_mm=-0.1)
    with pytest.raises(ValueError):
        scale_for_shrinkage(trimesh.creation.box(), 80.0, 0.0)
