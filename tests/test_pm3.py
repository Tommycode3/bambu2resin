import numpy as np
import pytest
import trimesh

from travaso.formats.pm3 import (
    EXTRA_TABLE_LENGTH,
    HEADER_TABLE_LENGTH,
    MACHINE_TABLE_LENGTH,
    PrintParameters,
    estimate_print_time_s,
    read_pm3,
    write_pm3,
)
from travaso.formats.pw0 import decode_pw0, encode_pw0
from travaso.printers import PHOTON_M3
from travaso.slicer.stack import SliceSettings, place_on_plate, slice_mesh


def _grey_map(image: np.ndarray) -> np.ndarray:
    """How the decoder remaps the 4 bits: 0 -> 0, 15 -> 255, otherwise (c<<4)|c."""
    code = image >> 4
    return np.where(code == 0, 0, np.where(code == 15, 255, (code << 4) | code)).astype(np.uint8)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((32, 32), np.uint8),
        np.full((32, 32), 255, np.uint8),
        np.tile(np.arange(32, dtype=np.uint8) * 8, (32, 1)),
        np.random.default_rng(3).integers(0, 256, (48, 40), dtype=np.uint8),
    ],
)
def test_pw0_round_trip(image):
    rebuilt = decode_pw0(encode_pw0(image), image.shape[1], image.shape[0])
    np.testing.assert_array_equal(rebuilt, _grey_map(image))


def test_pw0_splits_runs_longer_than_the_counter():
    """The black counter goes up to 4095: a longer run must be split."""
    image = np.zeros((100, 100), np.uint8)  # 10000 pixels > 4095
    data = encode_pw0(image)
    assert len(data) == 6  # three two-byte pieces
    np.testing.assert_array_equal(decode_pw0(data, 100, 100), image)


def test_tables_have_the_format_lengths():
    assert (HEADER_TABLE_LENGTH, MACHINE_TABLE_LENGTH, EXTRA_TABLE_LENGTH) == (84, 156, 56)


def test_write_and_read_back(tmp_path):
    cylinder = place_on_plate(
        trimesh.creation.cylinder(radius=6.0, height=2.0, sections=128), PHOTON_M3
    )
    layers = list(slice_mesh(cylinder, PHOTON_M3, SliceSettings(0.05)))
    volume = sum(layer.area_mm2 for layer in layers) * 0.05

    destination = str(tmp_path / "test.pm3")
    written = write_pm3(
        destination,
        (layer.image for layer in layers),
        PHOTON_M3,
        PrintParameters(layer_height_mm=0.05),
        volume_mm3=volume,
    )
    read_back = read_pm3(destination)

    assert read_back.valid, read_back.problems
    assert read_back.version == 516
    assert read_back.machine_name == "Photon M3"
    assert read_back.layer_image_format == "pw0Img"
    assert read_back.resolution == (4096, 2560)
    assert read_back.layer_count == written.layers == len(layers)
    assert read_back.print_time_s == written.print_time_s
    assert read_back.volume_ml == pytest.approx(written.volume_ml, rel=1e-5)
    assert read_back.layer_height_mm == pytest.approx(0.05, rel=1e-5)


def test_table_addresses_are_in_order(tmp_path):
    box = place_on_plate(trimesh.creation.box(extents=(5.0, 5.0, 0.5)), PHOTON_M3)
    layers = list(slice_mesh(box, PHOTON_M3, SliceSettings(0.05)))
    destination = str(tmp_path / "order.pm3")
    write_pm3(destination, (l.image for l in layers), PHOTON_M3)

    t = read_pm3(destination).tables
    order = [t["header"], t["preview"], t["colour_table"], t["layerdef"], t["extra"], t["machine"], t["layer_image"]]
    assert order == sorted(order), "tables must follow one another without overlapping"
    assert t["software"] == 0, "the SOFTWARE table only exists from version 517"


def test_implausible_parameters_rejected():
    with pytest.raises(ValueError):
        PrintParameters(bottom_layers=0)
    with pytest.raises(ValueError):
        PrintParameters(exposure_s=0.0)
    with pytest.raises(ValueError):
        PrintParameters(antialiasing=5)


def test_print_time_estimate():
    p = PrintParameters(exposure_s=2.0, bottom_exposure_s=30.0, bottom_layers=5, wait_before_cure_s=1.0)
    # 5 bottom layers at 30s + 95 normal at 2s + 100 waits of 1s + the travel
    assert estimate_print_time_s(100, p) > 5 * 30 + 95 * 2 + 100


def test_layer_with_wrong_size_rejected(tmp_path):
    with pytest.raises(ValueError, match="size"):
        write_pm3(str(tmp_path / "x.pm3"), [np.zeros((10, 10), np.uint8)], PHOTON_M3)
