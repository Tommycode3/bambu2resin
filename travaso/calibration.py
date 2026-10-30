"""Calibration test pieces.

Exposure times and light bleed cannot be derived: they depend on the resin,
its color, temperature and how much the panel has aged. They must be
measured, and the way to measure them is to print a purpose-made test piece.

There are two, answering two different questions:

  exposure     how long to keep the light on. A tower is printed in which each
               horizontal band has a different time, and you look at which band
               came out well. The format stores the exposure in every layer
               record, so a single print and half an hour are enough.

  tolerance    how much the light bleeds. A plate with holes and pins of known
               diameter is printed, measured with calipers and compared with
               the nominal values: holes come out narrower and pins thicker,
               by the same amount.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

# Slot widths that stress the exposure: below half a millimeter an overly
# long time closes them completely.
FESSURE_MM = (0.2, 0.3, 0.4, 0.6, 0.8)
# Diameters that are easy to measure with calipers.
DIAMETRI_MISURABILI_MM = (2.0, 3.0, 4.0, 5.0, 6.0)


@dataclass
class Provino:
    mesh: trimesh.Trimesh
    nome: str
    istruzioni: list[str] = field(default_factory=list)
    fasce: list[tuple[float, float, float]] = field(default_factory=list)
    """(lower z, upper z, exposure) for the exposure test piece."""

    def exposure_for_z(self, z: float, default: float) -> float:
        for basso, alto, tempo in self.fasce:
            if basso <= z < alto:
                return tempo
        return default


def _foro(x: float, y: float, diametro: float, altezza: float, z0: float) -> trimesh.Trimesh:
    """Inverted cylinder: with positive-winding fill it removes material.

    It does not protrude beyond the thickness it has to cross: subtractive
    geometry sticking out of the bounding box would skew placement on the plate
    and add empty layers above and below the part.
    """
    cilindro = trimesh.creation.cylinder(radius=diametro / 2.0, height=altezza, sections=48)
    cilindro.apply_translation([x, y, z0 + altezza / 2.0])
    cilindro.invert()
    return cilindro


def _fessura(
    x: float, larghezza: float, altezza: float, z0: float, profondita: float, y_faccia: float
) -> trimesh.Trimesh:
    """Slot cut into the front face, open to the outside.

    Better than a vertical hole: a hole in a band would be capped by the band
    above, so invisible and full of resin. A slot on the front stays open,
    drains by itself and can be read with the part in hand.
    """
    intaglio = trimesh.creation.box(extents=(larghezza, profondita, altezza))
    intaglio.apply_translation([x, y_faccia + profondita / 2.0, z0 + altezza / 2.0])
    intaglio.invert()
    return intaglio


def torre_esposizione(
    tempi: list[float],
    larghezza_mm: float = 26.0,
    profondita_mm: float = 14.0,
    altezza_fascia_mm: float = 2.5,
    base_mm: float = 1.5,
    profondita_fessura_mm: float = 1.2,
) -> Provino:
    """Stepped tower, one band for each exposure time to test.

    Each band is slightly narrower than the one below, so they can be counted
    by eye and no step juts outward creating an overhang. On the front of each
    band there is a series of progressively narrower slots: if the time is too
    long the light closes them, and that is the most readable sign of
    overexposure.
    """
    if not tempi:
        raise ValueError("at least one time to test is required")

    pezzi: list[trimesh.Trimesh] = []
    base = trimesh.creation.box(extents=(larghezza_mm, profondita_mm, base_mm))
    base.apply_translation([0.0, 0.0, base_mm / 2.0])
    pezzi.append(base)

    fasce: list[tuple[float, float, float]] = [(0.0, base_mm, tempi[0])]
    z = base_mm
    y_faccia = -profondita_mm / 2.0

    for indice, tempo in enumerate(tempi):
        larghezza = larghezza_mm - indice * 1.2
        if larghezza < 10.0:
            raise ValueError("too many times for the chosen width: the tower gets too thin")

        fascia = trimesh.creation.box(extents=(larghezza, profondita_mm, altezza_fascia_mm))
        fascia.apply_translation([0.0, 0.0, z + altezza_fascia_mm / 2.0])
        pezzi.append(fascia)

        passo = larghezza / (len(FESSURE_MM) + 1)
        for k, apertura in enumerate(FESSURE_MM):
            x = -larghezza / 2.0 + passo * (k + 1)
            pezzi.append(
                _fessura(
                    x,
                    apertura,
                    altezza_fascia_mm - 0.6,
                    z + 0.3,
                    profondita_fessura_mm,
                    y_faccia,
                )
            )

        fasce.append((z, z + altezza_fascia_mm, tempo))
        z += altezza_fascia_mm

    # The last band reaches exactly the top: no layer is left uncovered
    fasce[-1] = (fasce[-1][0], z + 1e-6, fasce[-1][2])

    istruzioni = [
        "Print the test piece flat on the build plate, without supports.",
        f"Bands are counted from the bottom: {len(tempi)} steps, each narrower than the previous one.",
        "On the front of each band there are five slots, from widest to narrowest.",
        "",
        "Find the band with the LONGEST time in which all slots are still open",
        "and the edges stay sharp: that is the right time.",
        "If the slots are already closed in the first band, repeat the test with shorter times.",
        "If they are still open in the last one, repeat the test with longer times.",
        "",
    ]
    for indice, tempo in enumerate(tempi, start=1):
        istruzioni.append(f"    band {indice}: {tempo:.2f} s")
    istruzioni.append(f"    slots per band (mm): {', '.join(str(f) for f in FESSURE_MM)}")

    return Provino(
        mesh=trimesh.util.concatenate(pezzi),
        nome="exposure tower",
        istruzioni=istruzioni,
        fasce=fasce,
    )


def piastra_tolleranza(
    larghezza_mm: float = 44.0, profondita_mm: float = 26.0, spessore_mm: float = 3.0
) -> Provino:
    """Plate with holes and pins of known diameter, to measure bleed.

    Holes and pins together are not redundant: bleeding light shifts them in
    opposite directions, so measuring both separates bleed from a possible
    scale error, which would instead shift them in the same direction.
    """
    pezzi: list[trimesh.Trimesh] = []
    piastra = trimesh.creation.box(extents=(larghezza_mm, profondita_mm, spessore_mm))
    piastra.apply_translation([0.0, 0.0, spessore_mm / 2.0])
    pezzi.append(piastra)

    passo = larghezza_mm / (len(DIAMETRI_MISURABILI_MM) + 1)
    y_fori = profondita_mm / 4.0
    y_colonnini = -profondita_mm / 4.0

    for k, diametro in enumerate(DIAMETRI_MISURABILI_MM):
        x = -larghezza_mm / 2.0 + passo * (k + 1)
        pezzi.append(_foro(x, y_fori, diametro, spessore_mm, 0.0))

        colonnino = trimesh.creation.cylinder(radius=diametro / 2.0, height=3.0, sections=64)
        colonnino.apply_translation([x, y_colonnini, spessore_mm + 1.5])
        pezzi.append(colonnino)

    istruzioni = [
        "Print the test plate flat on the build plate, without supports,",
        "using the exposure time found with the tower.",
        "Measure with calipers and compare with the nominal values:",
        f"    holes (top row):        {', '.join(f'{d:.1f}' for d in DIAMETRI_MISURABILI_MM)} mm",
        f"    pins (bottom row):      {', '.join(f'{d:.1f}' for d in DIAMETRI_MISURABILI_MM)} mm",
        f"    long side of the plate: {larghezza_mm:.1f} mm",
        "",
        "Holes come out narrower and pins thicker, by the same amount:",
        "that amount divided by two is the bleed, to pass to build as --bleed.",
        "If instead they shift in the same direction it is not bleed but scale:",
        "in that case adjust --shrink-xy.",
    ]

    return Provino(
        mesh=trimesh.util.concatenate(pezzi),
        nome="tolerance plate",
        istruzioni=istruzioni,
    )


def tempi_geometrici(minimo: float, massimo: float, passi: int) -> list[float]:
    """Times distributed in a geometric progression.

    The resin's response to exposure is not linear: a given increase in time
    matters more between one and two seconds than between nine and ten. A
    geometric scale puts the steps where the difference shows.
    """
    if passi < 2:
        raise ValueError("at least two steps are required")
    if minimo <= 0 or massimo <= minimo:
        raise ValueError("invalid time range")
    return [round(float(t), 2) for t in np.geomspace(minimo, massimo, passi)]
