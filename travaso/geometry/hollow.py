"""Hollowing with drain holes.

A solid resin part is expensive: a solid 40 mm cube takes 64 ml of resin,
almost all of it wasted, and a solid mass shrinks more as it cures. But
hollowing without holes is worse than not hollowing: the cavity stays full of
liquid resin, the part weighs as much as before, and during peeling it acts as
a suction cup against the vat film.

The cavity is derived from a distance field: the part is discretized, each
voxel's distance from the surface is measured, and everything farther than the
wall thickness is taken as interior. The surface of that region, inverted,
subtracts from the solid: with positive-winding fill no boolean operation is
needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
import trimesh.voxel.ops as voxel_ops
from scipy import ndimage

from ..slicer.raster import Grid, rasterize
from ..slicer.section import plane_segments


@dataclass
class HollowSettings:
    wall_mm: float = 2.0
    voxel_mm: float = 0.35
    drain_diameter_mm: float = 3.0
    drain_holes: int = 2
    min_cavity_mm3: float = 500.0
    """Cavities smaller than this are not worth the holes they would need.

    Two 3 mm holes through a 2 mm wall already remove about thirty cubic
    millimetres: a twenty cubic millimetre cavity would be a net loss, on top of
    leaving two holes in the part.
    """

    min_saving: float = 0.10
    """Minimum fraction of resin saved below which the part is not hollowed.

    On a thin part the possible cavity is tiny, but the drain holes stay as
    large as needed for the resin to flow out: you would end up drilling the
    part in several places to recover a few cubic millimetres. In that case it
    is better to leave it solid.
    """

    max_voxels: int = 12_000_000
    """Cap on the number of voxels.

    The distance transform and the labelling allocate several arrays as large
    as the grid: twelve million voxels cost a few hundred MB, sixty million cost
    several GB and take the machine down.
    """

    def __post_init__(self) -> None:
        if self.wall_mm <= 0:
            raise ValueError("wall thickness must be positive")
        if self.voxel_mm <= 0:
            raise ValueError("voxel pitch must be positive")
        if self.voxel_mm > self.wall_mm / 2.0:
            raise ValueError(
                f"pitch {self.voxel_mm} mm too coarse for a {self.wall_mm} mm wall: "
                "at least two voxels of thickness are needed"
            )


@dataclass
class HollowReport:
    hollowed: bool = False
    voxel_mm: float = 0.0
    wall_mm: float = 0.0
    achieved_wall_mm: float = 0.0
    """Achieved thickness, estimated on the grid.

    This is an estimate, not an exact measurement: the distance transform is
    Euclidean in three dimensions, so near edges and corners the minimum falls
    diagonally rather than along an axis. The error stays within about one grid
    step, which is also the resolution at which the cavity was built.
    """

    solid_volume_mm3: float = 0.0
    cavity_volume_mm3: float = 0.0
    cavities: int = 0
    drains: int = 0
    drain_points: list = field(default_factory=list)
    drained: bool = True
    subtractive: list = field(default_factory=list)
    """Inner shell and holes, inverted. They are part of the assembly to slice
    but not of the outer surface: overhang detection must ignore them."""

    notes: list[str] = field(default_factory=list)

    @property
    def saved_fraction(self) -> float:
        if self.solid_volume_mm3 <= 0:
            return 0.0
        return self.cavity_volume_mm3 / self.solid_volume_mm3


def _voxelize(
    mesh: trimesh.Trimesh, settings: HollowSettings
) -> tuple[np.ndarray, np.ndarray, float]:
    """Discretize the part by slicing it, not with trimesh voxelization.

    The default one subdivides triangles until they are smaller than the
    pitch: on a 185 mm part with large triangles it generates millions of
    vertices and reaches nine gigabytes to produce a five-million-cell matrix.
    A voxel grid, however, is nothing more than a stack of layers, and the
    rasterizer already exists: it costs one image at a time, fills interiors on
    its own with the same rule used for printing, and needs no morphological
    hole filling.

    Returns the matrix, the centre of voxel (0,0,0) and the effective pitch.
    """
    pitch = settings.voxel_mm
    size = mesh.extents
    margin = 2

    while True:
        estimate = float(np.prod(np.ceil(size / pitch) + 2 * margin + 1))
        if estimate <= settings.max_voxels:
            break
        if pitch >= settings.wall_mm / 2.0:
            raise MemoryError(
                f"the part would need {estimate / 1e6:.0f} million voxels even at the maximum "
                f"pitch of {pitch:.2f} mm (cap {settings.max_voxels / 1e6:.0f}): "
                "increase --wall or skip hollowing"
            )
        pitch = min(pitch * 1.5, settings.wall_mm / 2.0)

    lower = mesh.bounds[0] - margin * pitch
    counts = np.ceil((mesh.bounds[1] + margin * pitch - lower) / pitch).astype(int) + 1

    grid = Grid(
        width=int(counts[0]),
        height=int(counts[1]),
        pixel_w_mm=pitch,
        pixel_h_mm=pitch,
    )
    shifted = np.asarray(mesh.vertices, dtype=np.float64) - lower
    faces = np.asarray(mesh.faces, dtype=np.int64)

    matrix = np.zeros((int(counts[0]), int(counts[1]), int(counts[2])), dtype=bool)
    for k in range(int(counts[2])):
        z = (k + 0.5) * pitch
        segments = plane_segments(shifted, faces, z)
        if len(segments) == 0:
            continue
        matrix[:, :, k] = (rasterize(segments, grid, subsamples=1) > 127).T

    # The centre of voxel (0,0,0) sits half a step inside the grid corner
    origin = lower + 0.5 * pitch
    return matrix, origin, pitch


def hollow(
    mesh: trimesh.Trimesh, settings: HollowSettings | None = None, drain_axis: int = 2
) -> tuple[trimesh.Trimesh, HollowReport]:
    """Hollow the part and open drain holes in it.

    `drain_axis` is the axis along which the holes are opened: for printing it
    is Z, because the resin must be able to flow downwards.
    """
    settings = settings or HollowSettings()
    report = HollowReport(wall_mm=settings.wall_mm)

    if not mesh.is_watertight:
        report.notes.append(
            "the mesh is not watertight: the cavity is derived from the grid "
            "fill and may be inaccurate"
        )

    try:
        matrix, origin, pitch = _voxelize(mesh, settings)
    except MemoryError as error:
        report.notes.append(str(error))
        return mesh, report

    report.voxel_mm = pitch
    if pitch != settings.voxel_mm:
        report.notes.append(f"pitch raised to {pitch:.2f} mm to avoid running out of memory")

    voxel_volume = pitch**3
    report.solid_volume_mm3 = float(matrix.sum()) * voxel_volume

    # Distance of each solid voxel from the nearest empty one, i.e. from the surface
    distance = ndimage.distance_transform_edt(matrix, sampling=pitch)

    # The transform returns the distance between voxel centres, i.e. one step
    # more than the distance between cells; marching cubes then places the
    # cavity surface half a step further out. The net result is that the
    # achieved wall equals (minimum distance - one step), and to hit the
    # requested one the threshold must be raised by half a step.
    threshold = settings.wall_mm + pitch / 2.0
    inside = distance > threshold

    if not inside.any():
        report.notes.append(
            f"part too thin for a {settings.wall_mm} mm wall: left solid"
        )
        return mesh, report

    labels, count = ndimage.label(inside)
    sizes = np.bincount(labels.ravel())
    keep = [
        index
        for index in range(1, count + 1)
        if sizes[index] * voxel_volume >= settings.min_cavity_mm3
    ]
    if not keep:
        report.notes.append("the possible cavities are too small: part left solid")
        return mesh, report

    cavity = np.isin(labels, keep)
    report.cavities = len(keep)
    report.cavity_volume_mm3 = float(cavity.sum()) * voxel_volume

    if report.saved_fraction < settings.min_saving:
        report.cavity_volume_mm3 = 0.0
        report.cavities = 0
        report.notes.append(
            f"hollowing would save only {100 * float(cavity.sum()) * voxel_volume / report.solid_volume_mm3:.0f}% "
            f"but would need {settings.drain_diameter_mm} mm holes: part too "
            "thin, left solid"
        )
        return mesh, report
    report.achieved_wall_mm = float(distance[cavity].min()) - pitch
    if abs(report.achieved_wall_mm - settings.wall_mm) > pitch:
        report.notes.append(
            f"achieved wall {report.achieved_wall_mm:.2f} mm instead of "
            f"{settings.wall_mm:.2f}: reduce the pitch to get closer"
        )

    shell = voxel_ops.matrix_to_marching_cubes(cavity, pitch=pitch)
    # Marching cubes already returns millimetres relative to the grid origin:
    # it only needs translating, not rescaling.
    shell.apply_translation(origin)
    shell.invert()  # inverted, it subtracts from the solid

    pieces = [mesh, shell]
    drains, drained = _drain_holes(
        mesh, matrix, cavity, labels, keep, origin, pitch, settings, drain_axis
    )
    pieces.extend(drains)
    report.drains = len(drains)
    report.drained = drained
    report.drain_points = [tuple(np.round(d.centroid, 2)) for d in drains]
    if not drained:
        report.notes.append(
            "a cavity has no outlet to the outside: it would stay full of resin "
            "and act as a suction cup"
        )

    report.hollowed = True
    report.subtractive = pieces[1:]  # everything except the outer surface
    return trimesh.util.concatenate(pieces), report


def _check_drainage(
    solid: np.ndarray,
    cavity: np.ndarray,
    drill: np.ndarray,
) -> bool:
    """Can the resin actually get out?

    It is not enough that the cylinder was created: we must check that air
    reaches from the outside into the cavity through the hole. The empty space
    is flooded starting from the grid border and we check whether the cavity is
    reached. An isolated cavity would stay full of resin and act as a suction
    cup against the vat film: worse than leaving the part solid.
    """
    air = (~solid) | cavity | drill
    labels, count = ndimage.label(air)
    if count == 0:
        return False

    on_border = np.unique(
        np.concatenate(
            [
                labels[0].ravel(),
                labels[-1].ravel(),
                labels[:, 0].ravel(),
                labels[:, -1].ravel(),
                labels[:, :, 0].ravel(),
                labels[:, :, -1].ravel(),
            ]
        )
    )
    on_border = on_border[on_border != 0]
    if len(on_border) == 0:
        return False

    reachable = np.isin(labels, on_border)
    return bool(reachable[cavity].all())


def _drill_mask(
    shape: tuple[int, ...],
    points: list[np.ndarray],
    origin: np.ndarray,
    pitch: float,
    radius: float,
    drain_axis: int,
) -> np.ndarray:
    """Voxels occupied by the drain cylinders, for the outlet check."""
    drill = np.zeros(shape, dtype=bool)
    if not points:
        return drill

    def axis_ramp(axis: int) -> np.ndarray:
        """Indices along one axis, shaped for broadcasting.

        `np.indices` would materialize three integer arrays as large as the
        whole grid: on a large part that is gigabytes just to draw circles.
        Two broadcast vectors are enough.
        """
        form = [1, 1, 1]
        form[axis] = shape[axis]
        return np.arange(shape[axis], dtype=np.float32).reshape(form)

    other_axes = [a for a in range(3) if a != drain_axis]
    limit = (radius / pitch) ** 2
    for point in points:
        squared = sum(
            (axis_ramp(a) - np.float32((point[a] - origin[a]) / pitch)) ** 2 for a in other_axes
        )
        drill |= squared <= limit
    return drill


def _drain_holes(
    mesh: trimesh.Trimesh,
    solid: np.ndarray,
    cavity: np.ndarray,
    labels: np.ndarray,
    keep: list[int],
    origin: np.ndarray,
    pitch: float,
    settings: HollowSettings,
    drain_axis: int,
) -> tuple[list[trimesh.Trimesh], bool]:
    """Open the holes: one at the bottom for the resin to drain, one at the top for air."""
    holes: list[trimesh.Trimesh] = []
    centres: list[np.ndarray] = []
    radius = settings.drain_diameter_mm / 2.0
    lower, upper = mesh.bounds

    for index in keep:
        coords = np.argwhere(labels == index)
        world = coords * pitch + origin

        order = np.argsort(world[:, drain_axis])
        wanted = [world[order[0]]]  # the lowest point: resin drains from there
        if settings.drain_holes >= 2:
            wanted.append(world[order[-1]])  # the highest: air enters from there

        for point in wanted[: settings.drain_holes]:
            # The cylinder spans exactly the part's extent along the drain axis:
            # this way it surely exits the surface without sticking out of the
            # bounding box, which would otherwise skew placement on the plate
            # and produce empty layers above and below.
            height = float(upper[drain_axis] - lower[drain_axis])
            hole = trimesh.creation.cylinder(radius=radius, height=height, sections=16)
            centre = np.array(point, dtype=float)
            centre[drain_axis] = float(lower[drain_axis] + upper[drain_axis]) / 2.0
            hole.apply_translation(centre)
            hole.invert()
            holes.append(hole)
            centres.append(np.array(point, dtype=float))

    drill = _drill_mask(solid.shape, centres, origin, pitch, radius, drain_axis)
    return holes, _check_drainage(solid, cavity, drill)
