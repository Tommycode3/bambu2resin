"""Conversions between the 3MF convention and numpy 4x4 matrices."""

from __future__ import annotations

import numpy as np

IDENTITY = np.eye(4)


def parse_3mf_transform(text: str | None) -> np.ndarray:
    """Convert the 3MF `transform` attribute into a 4x4 matrix.

    3MF writes 12 floats in row-vector convention:
        m00 m01 m02  m10 m11 m12  m20 m21 m22  tx ty tz
    where the transformed point is `p_row @ M3 + T`. numpy works with column
    vectors, so the 3x3 must be transposed.
    """
    if not text:
        return IDENTITY.copy()
    values = [float(x) for x in text.split()]
    if len(values) != 12:
        raise ValueError(f"3MF transform has {len(values)} values, expected 12")
    m = np.eye(4)
    m[:3, :3] = np.array(values[:9], dtype=np.float64).reshape(3, 3).T
    m[:3, 3] = values[9:]
    return m


def parse_bambu_matrix(text: str | None) -> np.ndarray:
    """Convert the `matrix` attribute of model_settings.config (16 floats, row-major)."""
    if not text:
        return IDENTITY.copy()
    values = [float(x) for x in text.split()]
    if len(values) != 16:
        raise ValueError(f"Bambu matrix has {len(values)} values, expected 16")
    return np.array(values, dtype=np.float64).reshape(4, 4).T


def decompose_scale(m: np.ndarray) -> np.ndarray:
    """Scale factors along the three axes (norms of the 3x3 columns)."""
    return np.linalg.norm(m[:3, :3], axis=0)


def is_mirroring(m: np.ndarray) -> bool:
    """True if the transform flips orientation (negative determinant)."""
    return bool(np.linalg.det(m[:3, :3]) < 0)
