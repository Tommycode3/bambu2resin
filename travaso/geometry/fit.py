"""Bounding-box check against the build volume."""

from __future__ import annotations

from itertools import permutations

import numpy as np

AXIS_NAMES = ("X", "Y", "Z")


def fit_in_volume(
    size: np.ndarray, volume: tuple[float, float, float], margin_mm: float = 0.0
) -> tuple[bool, tuple[int, int, int] | None]:
    """Does the part fit in the volume, possibly rotated by 90 degrees about the axes?

    Comparing the axes in the given order is wrong: a wide, low part may fit
    perfectly well once turned. So every assignment of the three dimensions to
    the three machine axes is tried and the one that fits is returned.

    Returns (fits, permutation) where permutation[i] is the machine axis that
    dimension i of the part maps to.
    """
    needed = np.asarray(size, dtype=float) + 2.0 * margin_mm

    # If the part fits in its current pose, that is the answer: rotating to
    # gain margin nobody asked for would be pointless noise.
    if all(needed[i] <= volume[i] for i in range(3)):
        return True, (0, 1, 2)

    best: tuple[int, int, int] | None = None
    best_slack = -np.inf

    for perm in permutations(range(3)):
        slacks = [volume[perm[i]] - needed[i] for i in range(3)]
        if min(slacks) < 0:
            continue
        # Among poses that fit, prefer the one with the most remaining margin
        # on the tightest axis: it leaves room for supports.
        if min(slacks) > best_slack:
            best_slack = min(slacks)
            best = perm

    return best is not None, best


def describe_fit(size: np.ndarray, volume: tuple[float, float, float]) -> str:
    fits, perm = fit_in_volume(size, volume)
    if not fits:
        over = [
            f"{AXIS_NAMES[i]} {size[i]:.1f} > {max(volume):.1f}"
            for i in range(3)
            if size[i] > max(volume)
        ]
        return "DOES NOT FIT" + (f" ({', '.join(over)})" if over else "")
    if perm == (0, 1, 2):
        return ""
    mapping = ", ".join(f"{AXIS_NAMES[i]}->{AXIS_NAMES[perm[i]]}" for i in range(3))
    return f"fits rotated ({mapping})"
