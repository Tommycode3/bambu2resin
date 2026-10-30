"""Island detection in the layer stack.

An island is material that appears in a layer with nothing below to hold it.
In FDM the problem does not exist: the nozzle always deposits onto something.
In resin the island cures detached from the part, floats in the vat and ends
up stuck wherever, often ruining the print and sometimes the vat film.

Islands are better found on the already rasterized layers than on the
geometry: a sloped overhang may produce no island at all, and a seemingly
harmless part may produce dozens. Here we look at exactly what the printer
will cure.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Island:
    layer: int
    z_mm: float
    area_mm2: float
    pixels: int
    centroid_mm: tuple[float, float]
    """Centroid in millimetres, in the build plate frame."""

    def __str__(self) -> str:
        return (
            f"layer {self.layer} (z={self.z_mm:.2f} mm): {self.area_mm2:.3f} mm2 "
            f"at ({self.centroid_mm[0]:.1f}, {self.centroid_mm[1]:.1f})"
        )


def _content_window(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Rectangle containing the material, to avoid labelling the whole panel."""
    rows = np.any(mask, axis=1)
    if not rows.any():
        return None
    cols = np.any(mask, axis=0)
    r = np.flatnonzero(rows)
    c = np.flatnonzero(cols)
    return int(r[0]), int(r[-1]) + 1, int(c[0]), int(c[-1]) + 1


class IslandWatcher:
    """Watch a stack of layers as it goes by.

    It is fed one layer at a time instead of receiving the whole sequence: this
    way the same pass that writes the file can also check that no islands are
    left, without slicing a second time.
    """

    def __init__(
        self,
        grid,
        threshold: int = 128,
        tolerance_px: int = 1,
        min_area_mm2: float = 0.0,
    ):
        self.grid = grid
        self.threshold = threshold
        self.tolerance_px = tolerance_px
        self.min_area_mm2 = min_area_mm2
        self.islands: list[Island] = []
        self._previous: np.ndarray | None = None

    def feed(self, layer) -> list[Island]:
        """Feed a layer and return the islands that appear in it."""
        current = layer.image > self.threshold

        if self._previous is None:
            self._previous = current
            return []

        window = _content_window(current)
        if window is None:
            self._previous = current
            return []

        r0, r1, c0, c1 = window
        patch = current[r0:r1, c0:c1]
        below = self._previous[r0:r1, c0:c1]
        if self.tolerance_px > 0:
            below = ndimage.binary_dilation(below, iterations=self.tolerance_px)

        found: list[Island] = []
        labels, count = ndimage.label(patch)
        if count:
            # A label that touches not even one supported pixel is an island
            supported = set(np.unique(labels[below & patch]).tolist())
            sizes = np.bincount(labels.ravel(), minlength=count + 1)
            for index in range(1, count + 1):
                if index in supported:
                    continue
                area = float(sizes[index]) * self.grid.area_mm2
                if area < self.min_area_mm2:
                    continue
                ys, xs = np.nonzero(labels == index)
                found.append(
                    Island(
                        layer=layer.index,
                        z_mm=layer.z_mm,
                        area_mm2=area,
                        pixels=int(sizes[index]),
                        centroid_mm=self.grid.to_plate_mm(float(xs.mean()) + c0, float(ys.mean()) + r0),
                    )
                )

        self._previous = current
        self.islands.extend(found)
        return found


def find_islands(
    layers: Iterable,
    grid,
    threshold: int = 128,
    tolerance_px: int = 1,
    min_area_mm2: float = 0.0,
) -> Iterator[Island]:
    """Walk the layers from the bottom and yield the islands found.

    The first layer produces none: it rests on the build plate or the raft.
    """
    watcher = IslandWatcher(grid, threshold, tolerance_px, min_area_mm2)
    for layer in layers:
        yield from watcher.feed(layer)
