"""Full slicing: from the placed mesh to the stack of images.

At each height only the triangles crossing that plane matter. Finding them by
re-scanning the whole mesh layer by layer costs the number of triangles times
the number of layers, which for half a million triangles and two thousand
layers becomes a billion comparisons. Here the triangles are sorted by minimum
height and an active list is kept that grows and shrinks as the plane rises.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import trimesh

from .raster import Grid, rasterize
from .section import plane_segments, signed_area


@dataclass
class SliceSettings:
    layer_height_mm: float = 0.05
    subsamples: int = 4

    bleed_mm: float = 0.0
    """How much the printer grows the part on each side.

    Light bleeds past the lit pixel and cures a shell around the part: pins
    come out thicker and holes tighter than designed. The value is measured
    by printing a test piece and compensated by shrinking by the same amount.
    Typically between 0.02 and 0.08 mm depending on resin and exposure.
    """

    elephant_foot_mm: float = 0.0
    """Growth of the first layers, caused by their longer exposure."""

    elephant_foot_layers: int = 0
    """Number of layers over which the elephant foot compensation fades out."""

    def __post_init__(self) -> None:
        if not 0.001 <= self.layer_height_mm <= 0.5:
            raise ValueError(f"layer height {self.layer_height_mm} mm outside the plausible range")
        if abs(self.bleed_mm) > 1.0:
            raise ValueError(f"bleed of {self.bleed_mm} mm is implausible")
        if self.elephant_foot_mm < 0:
            raise ValueError("elephant foot is a growth, it cannot be negative")

    def offset_for_layer(self, index: int) -> float:
        """Edge offset for a given layer, in millimetres.

        Bleed applies everywhere; elephant foot only at the bottom, and it
        fades out instead of stopping abruptly, otherwise it leaves a visible
        step right where the part rests.
        """
        offset = -self.bleed_mm
        if self.elephant_foot_mm > 0 and index < self.elephant_foot_layers:
            fraction = 1.0 - index / max(self.elephant_foot_layers, 1)
            offset -= self.elephant_foot_mm * fraction
        return offset


@dataclass
class Layer:
    index: int
    z_mm: float
    image: np.ndarray
    area_mm2: float

    @property
    def empty(self) -> bool:
        return self.area_mm2 <= 0.0


@dataclass
class SliceStats:
    layers: int = 0
    empty_layers: int = 0
    max_area_mm2: float = 0.0
    max_area_layer: int = 0
    volume_mm3: float = 0.0
    height_mm: float = 0.0
    first_layer_area_mm2: float = 0.0


def place_on_plate(
    mesh: trimesh.Trimesh, printer, lift_mm: float = 0.0
) -> trimesh.Trimesh:
    """Centre the part on the build plate and rest it on (or lift it along) the Z axis."""
    out = mesh.copy()
    lower, upper = out.bounds
    centre = (lower + upper) / 2.0
    out.apply_translation(
        [
            printer.display_width_mm / 2.0 - centre[0],
            printer.display_height_mm / 2.0 - centre[1],
            lift_mm - lower[2],
        ]
    )
    return out


def slice_mesh(
    mesh: trimesh.Trimesh, printer, settings: SliceSettings | None = None
) -> Iterator[Layer]:
    """Generate the layers from bottom to top.

    Each layer is sampled at half its thickness: that is the section that best
    represents the material actually cured in that slice.
    """
    settings = settings or SliceSettings()
    grid = Grid.from_printer(printer)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return

    z_face = vertices[faces][:, :, 2]
    z_min_face = z_face.min(axis=1)
    z_max_face = z_face.max(axis=1)

    order = np.argsort(z_min_face)
    z_min_sorted = z_min_face[order]

    bottom = float(vertices[:, 2].min())
    top = float(vertices[:, 2].max())
    height = settings.layer_height_mm
    count = max(1, int(np.ceil((top - bottom) / height)))

    active = np.zeros(0, dtype=np.int64)
    cursor = 0

    for index in range(count):
        z = bottom + (index + 0.5) * height

        # triangles that start below this height enter...
        new_cursor = int(np.searchsorted(z_min_sorted, z, side="right"))
        if new_cursor > cursor:
            active = np.concatenate([active, order[cursor:new_cursor]])
            cursor = new_cursor

        # ...and those that have already ended below it leave
        if len(active):
            active = active[z_max_face[active] >= z]

        if len(active) == 0:
            yield Layer(index=index, z_mm=z, image=np.zeros((grid.height, grid.width), np.uint8), area_mm2=0.0)
            continue

        segments = plane_segments(vertices, faces[active], z)
        image = rasterize(
            segments,
            grid,
            subsamples=settings.subsamples,
            offset_mm=settings.offset_for_layer(index),
        )
        area = float(image.astype(np.float64).sum()) / 255.0 * grid.area_mm2
        yield Layer(index=index, z_mm=z, image=image, area_mm2=area)


def summarize(layers: Iterator[Layer], layer_height_mm: float) -> tuple[SliceStats, list[float]]:
    """Consume the layers collecting statistics, without keeping their images."""
    stats = SliceStats()
    areas: list[float] = []
    for layer in layers:
        stats.layers += 1
        areas.append(layer.area_mm2)
        if layer.empty:
            stats.empty_layers += 1
        if layer.area_mm2 > stats.max_area_mm2:
            stats.max_area_mm2 = layer.area_mm2
            stats.max_area_layer = layer.index
        if stats.layers == 1:
            stats.first_layer_area_mm2 = layer.area_mm2
    stats.volume_mm3 = float(sum(areas) * layer_height_mm)
    stats.height_mm = stats.layers * layer_height_mm
    return stats, areas
