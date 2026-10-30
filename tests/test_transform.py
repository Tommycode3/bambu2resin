import numpy as np
import pytest

from travaso.geometry.fit import fit_in_volume
from travaso.geometry.transform import decompose_scale, is_mirroring, parse_3mf_transform


def test_transform_is_row_vector():
    """3MF uses the row-vector convention: the 3x3 must be transposed."""
    # 90 degree rotation about Z, then translation
    m = parse_3mf_transform("0 1 0 -1 0 0 0 0 1 10 20 30")
    p = m[:3, :3] @ np.array([1.0, 0.0, 0.0]) + m[:3, 3]
    np.testing.assert_allclose(p, [10.0, 21.0, 30.0], atol=1e-12)


def test_empty_transform_is_identity():
    np.testing.assert_allclose(parse_3mf_transform(None), np.eye(4))


def test_transform_wrong_length():
    with pytest.raises(ValueError):
        parse_3mf_transform("1 2 3")


def test_scale_and_mirroring():
    m = parse_3mf_transform("2 0 0 0 3 0 0 0 4 0 0 0")
    np.testing.assert_allclose(decompose_scale(m), [2, 3, 4])
    assert not is_mirroring(m)
    assert is_mirroring(parse_3mf_transform("-1 0 0 0 1 0 0 0 1 0 0 0"))


def test_fit_considers_rotation():
    """A wide, low part fits when turned: comparing the axes in order is wrong."""
    volume = (163.84, 102.4, 180.0)
    assert fit_in_volume(np.array([161.8, 161.8, 27.0]), volume)[0]
    assert not fit_in_volume(np.array([18.0, 18.0, 214.9]), volume)[0]


def test_fit_with_margin():
    volume = (10.0, 10.0, 10.0)
    assert fit_in_volume(np.array([9.0, 9.0, 9.0]), volume)[0]
    assert not fit_in_volume(np.array([9.0, 9.0, 9.0]), volume, margin_mm=1.0)[0]
