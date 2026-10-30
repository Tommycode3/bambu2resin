"""From contour segments to the layer image.

Filling uses the POSITIVE winding number rule, not even-odd and not
"non-zero winding" either. The three rules behave differently and the
difference is substantial:

  even-odd      two interpenetrating bodies hollow each other out in the shared
                region, instead of merging
  winding != 0  merging works, but an inverted primitive (winding -1) is also
                filled where it sticks out of the part, instead of subtracting
  winding > 0   merges solids AND allows subtracting with inverted primitives,
                which is how cavities and holes are made without booleans

With the segment direction produced by `section`, the inside of an outer
contour has winding +1.

Antialiasing is deliberately asymmetric: horizontally the coverage of each
pixel is computed exactly, because the ends of each span are known as real
numbers; vertically several sub-rows per pixel are sampled. This way the cost
grows with the number of sub-rows rather than with its square.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Grid:
    """Pixel grid of the panel, in build plate coordinates."""

    width: int
    height: int
    pixel_w_mm: float
    pixel_h_mm: float
    mirror_x: bool = False
    mirror_y: bool = False

    @property
    def area_mm2(self) -> float:
        return self.pixel_w_mm * self.pixel_h_mm

    def to_plate_mm(self, col: float, row: float) -> tuple[float, float]:
        """From image coordinates to millimetres on the build plate.

        The image is mirrored for the panel, so the column does not map
        directly to X: anyone reading a position from the pixels must undo the
        mirroring, otherwise they get the point reflected about the centre of
        the plate.
        """
        c = (self.width - 1 - col) if self.mirror_x else col
        r = (self.height - 1 - row) if self.mirror_y else row
        return ((c + 0.5) * self.pixel_w_mm, (r + 0.5) * self.pixel_h_mm)

    @classmethod
    def from_printer(cls, printer) -> "Grid":
        return cls(
            width=printer.resolution_x,
            height=printer.resolution_y,
            pixel_w_mm=printer.display_width_mm / printer.resolution_x,
            pixel_h_mm=printer.display_height_mm / printer.resolution_y,
            mirror_x=printer.mirror_x,
            mirror_y=printer.mirror_y,
        )


def _offset_edge(coverage: np.ndarray, offset_px: float) -> np.ndarray:
    """Move the part's edge by `offset_px`, preserving the antialiasing.

    A signed distance from the edge is reconstructed: in solid and empty
    regions it comes from the distance transform, on the edge from the
    fractional coverage already computed, which is more precise than a count
    of whole pixels. A shift and a clamp are then enough to get the moved edge
    with the same softness as before.
    """
    solid = coverage >= 0.5
    if not solid.any():
        return coverage if offset_px <= 0 else coverage

    inside = ndimage.distance_transform_edt(solid)
    outside = ndimage.distance_transform_edt(~solid)
    signed = np.where(solid, inside - 0.5, -(outside - 0.5)).astype(np.float32)

    edge = (coverage > 0.0) & (coverage < 1.0)
    signed = np.where(edge, coverage - 0.5, signed)

    return np.clip(signed + np.float32(offset_px) + 0.5, 0.0, 1.0).astype(np.float32)


def rasterize(
    segments: np.ndarray, grid: Grid, subsamples: int = 4, offset_mm: float = 0.0
) -> np.ndarray:
    """Coverage of each pixel, from 0 to 255.

    `segments` is (M, 2, 2) in millimetres, with the origin at the corner of
    the build plate.

    `offset_mm` moves the part's edge: negative shrinks it, positive grows it.
    It compensates for light bleeding past the lit pixel and curing a shell
    of resin around the part, which therefore comes out larger than the
    design and with tighter holes.
    """
    if subsamples < 1:
        raise ValueError("subsamples must be at least 1")

    out = np.zeros((grid.height, grid.width), dtype=np.uint8)
    if len(segments) == 0:
        return out

    # In pixel coordinates: the centre of pixel i is at i + 0.5, so the pixel
    # covers the interval [i, i+1).
    x = segments[:, :, 0] / grid.pixel_w_mm
    y = segments[:, :, 1] / grid.pixel_h_mm

    # A layer almost always covers a fraction of the panel. Working only on the
    # touched rectangle avoids allocating and scanning ten million pixels for
    # every slice.
    # If the edge is to be grown, the window must make room for the new material
    pad = int(np.ceil(abs(offset_mm) / grid.pixel_w_mm)) + 2 if offset_mm else 0
    col0 = int(np.clip(np.floor(x.min()) - pad, 0, grid.width))
    col1 = int(np.clip(np.ceil(x.max()) + 1 + pad, 0, grid.width))
    row0 = int(np.clip(np.floor(y.min()) - pad, 0, grid.height))
    row1 = int(np.clip(np.ceil(y.max()) + 1 + pad, 0, grid.height))
    if col1 <= col0 or row1 <= row0:
        return out

    window_w = col1 - col0
    window_h = row1 - row0
    coverage = np.zeros((window_h, window_w), dtype=np.float32)

    x = x - col0
    y = (y - row0) * subsamples

    x0, x1 = x[:, 0], x[:, 1]
    y0, y1 = y[:, 0], y[:, 1]

    rows_total = window_h * subsamples
    low = np.minimum(y0, y1)
    high = np.maximum(y0, y1)

    # Sampling rows crossed: sample i is at i + 0.5
    start = np.clip(np.ceil(low - 0.5), 0, rows_total).astype(np.int64)
    stop = np.clip(np.ceil(high - 0.5), 0, rows_total).astype(np.int64)
    counts = np.maximum(stop - start, 0)
    total = int(counts.sum())
    if total == 0:
        return out

    # "Ragged" expansion: one element per crossing
    segment_of = np.repeat(np.arange(len(segments)), counts)
    begins = np.zeros(len(segments) + 1, dtype=np.int64)
    np.cumsum(counts, out=begins[1:])
    sample_row = np.repeat(start, counts) + (np.arange(total) - begins[segment_of])

    dy = y1 - y0
    t = (sample_row + 0.5 - y0[segment_of]) / dy[segment_of]
    crossing_x = x0[segment_of] + t * (x1[segment_of] - x0[segment_of])
    # Sign chosen so that the inside of an outer contour is +1: this way an
    # inverted primitive is -1 and subtracts, and whatever stays negative
    # outside the part is not filled.
    winding = np.where(dy[segment_of] > 0, -1, 1).astype(np.int32)

    order = np.lexsort((crossing_x, sample_row))
    sample_row = sample_row[order]
    crossing_x = crossing_x[order]
    winding = winding[order]

    # Accumulated winding number, reset at the start of each row. A global
    # cumulative sum would only work if every row closed exactly at zero, and
    # on an imperfect mesh that is not guaranteed.
    running = np.cumsum(winding)
    row_change = np.flatnonzero(np.diff(sample_row)) + 1
    row_start = np.zeros(len(sample_row), dtype=np.int64)
    row_start[row_change] = row_change
    np.maximum.accumulate(row_start, out=row_start)
    baseline = np.where(row_start > 0, running[row_start - 1], 0)
    inside = (running - baseline) > 0

    # A filled span goes from one crossing to the next, on the same row
    same_row = sample_row[:-1] == sample_row[1:]
    span = inside[:-1] & same_row
    if not span.any():
        return out

    left = crossing_x[:-1][span]
    right = crossing_x[1:][span]
    out_row = (sample_row[:-1][span] // subsamples).astype(np.int64)

    np.clip(left, 0.0, float(window_w), out=left)
    np.clip(right, 0.0, float(window_w), out=right)
    keep = right > left
    left, right, out_row = left[keep], right[keep], out_row[keep]
    if len(left) == 0:
        return out

    weight = np.float32(1.0 / subsamples)
    first = np.floor(left).astype(np.int64)
    last = np.floor(right).astype(np.int64)
    np.clip(first, 0, window_w - 1, out=first)
    np.clip(last, 0, window_w - 1, out=last)

    single = first == last
    # Span contained in a single pixel: the coverage is its length
    if single.any():
        np.add.at(
            coverage,
            (out_row[single], first[single]),
            ((right[single] - left[single]) * weight).astype(np.float32),
        )

    multi = ~single
    if multi.any():
        fr, ls = first[multi], last[multi]
        rw = out_row[multi]
        np.add.at(coverage, (rw, fr), ((fr + 1 - left[multi]) * weight).astype(np.float32))
        np.add.at(coverage, (rw, ls), ((right[multi] - ls) * weight).astype(np.float32))

        # Fully covered pixels between the two ends are filled with a
        # difference array, avoiding a loop per span.
        diff = np.zeros((window_h, window_w + 1), dtype=np.float32)
        np.add.at(diff, (rw, fr + 1), weight)
        np.add.at(diff, (rw, ls), -weight)
        coverage += np.cumsum(diff[:, :-1], axis=1)

    np.clip(coverage, 0.0, 1.0, out=coverage)
    if offset_mm:
        coverage = _offset_edge(coverage, offset_mm / grid.pixel_w_mm)
    out[row0:row1, col0:col1] = np.rint(coverage * 255.0).astype(np.uint8)

    # The panel is seen from below through the vat: without mirroring the
    # part comes out flipped and snap-fits do not fit.
    if grid.mirror_x:
        out = out[:, ::-1]
    if grid.mirror_y:
        out = out[::-1, :]
    return out
