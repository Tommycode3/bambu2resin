import struct

import numpy as np
import pytest
from scipy import ndimage

from travaso.calibration import (
    FESSURE_MM,
    piastra_tolleranza,
    tempi_geometrici,
    torre_esposizione,
)
from travaso.formats.pm3 import PrintParameters, read_pm3, write_pm3
from travaso.printers import PHOTON_M3
from travaso.slicer.stack import SliceSettings, place_on_plate, slice_mesh


def test_times_grow_geometrically():
    tempi = tempi_geometrici(1.5, 6.0, 8)
    assert tempi[0] == pytest.approx(1.5)
    assert tempi[-1] == pytest.approx(6.0)
    assert all(a < b for a, b in zip(tempi, tempi[1:]))
    # geometric: the ratios are constant, not the differences
    rapporti = [b / a for a, b in zip(tempi, tempi[1:])]
    assert max(rapporti) - min(rapporti) < 0.02


def test_inconsistent_times_rejected():
    with pytest.raises(ValueError):
        tempi_geometrici(2.0, 1.0, 5)
    with pytest.raises(ValueError):
        tempi_geometrici(1.0, 5.0, 1)


def test_subtractive_geometry_does_not_inflate_bounding_box():
    """Regression: slots and holes protruding from the part skew placement
    on the plate and add empty layers above and below."""
    tempi = tempi_geometrici(1.5, 6.0, 8)
    torre = torre_esposizione(tempi, altezza_fascia_mm=2.5, base_mm=1.5)
    assert torre.mesh.extents[2] == pytest.approx(1.5 + 8 * 2.5, abs=1e-6)
    assert torre.mesh.extents[0] == pytest.approx(26.0, abs=1e-6)

    piastra = piastra_tolleranza(spessore_mm=3.0)
    # 3 mm of plate plus 3 mm of pins
    assert piastra.mesh.extents[2] == pytest.approx(6.0, abs=1e-6)


def test_every_tower_height_has_an_exposure():
    """No uncovered band: a layer without a band would use the fallback time
    and skew the reading of the test piece."""
    tempi = tempi_geometrici(1.5, 6.0, 6)
    torre = torre_esposizione(tempi)
    altezza = torre.mesh.extents[2]

    ripiego = -1.0
    for z in np.arange(0.02, altezza, 0.05):
        assert torre.exposure_for_z(float(z), ripiego) != ripiego, f"height {z:.2f} not covered"


def test_slots_actually_cut_the_front():
    tempi = tempi_geometrici(2.0, 4.0, 3)
    torre = place_on_plate(torre_esposizione(tempi).mesh, PHOTON_M3)
    # a layer inside the first band, which lies between 1.5 and 4.0 mm
    layer = next(l for l in slice_mesh(torre, PHOTON_M3, SliceSettings(0.05)) if 2.4 < l.z_mm < 2.6)

    solido = layer.image > 127
    righe = np.flatnonzero(solido.any(axis=1))
    # the outermost row on the slot side must be interrupted
    for riga in righe[:6]:
        transizioni = np.count_nonzero(np.diff(solido[riga].astype(np.int8)))
        if transizioni > 2:
            break
    else:
        pytest.fail("no slot found on the front of the band")


def test_plate_holes_have_nominal_diameter():
    piastra = place_on_plate(piastra_tolleranza().mesh, PHOTON_M3)
    layer = next(l for l in slice_mesh(piastra, PHOTON_M3, SliceSettings(0.05)) if l.z_mm > 1.0)

    solido = layer.image > 127
    etichette, quante = ndimage.label(~solido)
    bordo = set(
        np.unique(
            np.concatenate([etichette[0], etichette[-1], etichette[:, 0], etichette[:, -1]])
        ).tolist()
    )
    interni = [j for j in range(1, quante + 1) if j not in bordo]
    area_pixel = PHOTON_M3.display_width_mm / PHOTON_M3.resolution_x
    diametri = sorted(
        2.0 * np.sqrt(np.count_nonzero(etichette == j) * area_pixel**2 / np.pi) for j in interni
    )
    assert len(diametri) == 5
    for misurato, nominale in zip(diametri, (2.0, 3.0, 4.0, 5.0, 6.0)):
        assert misurato == pytest.approx(nominale, abs=0.02)


def test_per_layer_exposure_ends_up_in_file(tmp_path):
    tempi = tempi_geometrici(2.0, 5.0, 4)
    torre = torre_esposizione(tempi)
    posata = place_on_plate(torre.mesh, PHOTON_M3)
    params = PrintParameters(layer_height_mm=0.1, bottom_layers=4)

    layers = list(slice_mesh(posata, PHOTON_M3, SliceSettings(0.1)))

    def esposizione(index: int) -> float:
        if index < params.bottom_layers:
            return params.bottom_exposure_s
        return torre.exposure_for_z((index + 0.5) * 0.1, params.exposure_s)

    destinazione = str(tmp_path / "torre.pm3")
    write_pm3(
        destinazione,
        (l.image for l in layers),
        PHOTON_M3,
        params,
        exposure_for_layer=esposizione,
    )

    dati = open(destinazione, "rb").read()
    indirizzi = struct.unpack_from("<10I", dati, 12)
    header, layerdef = indirizzi[2] + 16, indirizzi[6] + 16

    # regression on field order: weight and price come BEFORE the
    # per-layer settings flag, not after
    per_layer = struct.unpack_from("<I", dati, header + 64)[0]
    assert per_layer == 1, "without this flag the printer ignores per-layer times"

    quanti = struct.unpack_from("<I", dati, layerdef)[0]
    tempi_file = [
        struct.unpack_from("<f", dati, layerdef + 4 + i * 32 + 16)[0] for i in range(quanti)
    ]
    assert set(round(t, 2) for t in tempi_file) >= set(tempi)
    assert tempi_file[0] == pytest.approx(params.bottom_exposure_s)
    assert read_pm3(destinazione).valid


def test_without_per_layer_exposure_flag_stays_off(tmp_path):
    import trimesh

    box = place_on_plate(trimesh.creation.box(extents=(5.0, 5.0, 0.5)), PHOTON_M3)
    layers = list(slice_mesh(box, PHOTON_M3, SliceSettings(0.05)))
    destinazione = str(tmp_path / "semplice.pm3")
    write_pm3(destinazione, (l.image for l in layers), PHOTON_M3)

    dati = open(destinazione, "rb").read()
    header = struct.unpack_from("<10I", dati, 12)[2] + 16
    assert struct.unpack_from("<I", dati, header + 64)[0] == 0
