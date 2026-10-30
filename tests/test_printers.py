from travaso.printers import PHOTON_M3, get_printer

import pytest


def test_photon_m3_has_square_40um_pixels():
    """163.84/4096 and 102.4/2560 both give 40 um: this checks that the
    display numbers are mutually consistent and not taken from marketing."""
    assert PHOTON_M3.pixel_width_um == pytest.approx(40.0)
    assert PHOTON_M3.pixel_height_um == pytest.approx(40.0)


def test_photon_m3_mirrors_only_x():
    assert PHOTON_M3.mirror_x and not PHOTON_M3.mirror_y


def test_unknown_printer():
    with pytest.raises(KeyError):
        get_printer("photon_inesistente")
