"""travaso - ports Bambu Studio projects to Anycubic resin printers."""

from __future__ import annotations

import argparse
import os
import sys

from .loader import load_project
from .geometry.fit import describe_fit
from .calibration import piastra_tolleranza, tempi_geometrici, torre_esposizione
from .geometry.hollow import HollowSettings, hollow
from .geometry.mesh import prepare, scale_for_shrinkage, to_trimesh
from .geometry.orient import PRESET_ALIASES, PRESETS, apply_orientation, auto_orient
from .formats.pm3 import PrintParameters, patch_preview, read_pm3, write_pm3
from .slicer.raster import Grid
from .slicer.stack import SliceSettings, place_on_plate, slice_mesh
from .supports.build import SupportSettings
from .supports.islands import IslandWatcher
from .supports.solve import support_until_clean
from .printers import REGISTRY, get_printer

LEVEL_ORDER = {"perdita": 0, "attenzione": 1, "info": 2}
LEVEL_LABEL = {"perdita": "LOST", "attenzione": "WARNING", "info": "info"}


def _priority(value: str) -> str:
    """Accept the English preset names and the legacy Italian ones."""
    return PRESET_ALIASES.get(value, value)


def cmd_inspect(args: argparse.Namespace) -> int:
    project = load_project(args.file)
    printer = get_printer(args.printer)

    print(f"\n  {args.file.split('/')[-1]}")
    print(f"  source: {project.application or 'unknown (non-Bambu 3MF)'}")
    print(f"  target: {printer.vendor} {printer.model}\n")

    for plate in project.plates:
        objects = project.objects_on_plate(plate.index)
        if not objects:
            continue
        label = f" ({plate.name})" if plate.name else ""
        print(f"  plate {plate.index}{label} - {len(objects)} objects")
        for obj in objects:
            size = obj.size()
            note = describe_fit(size, printer.build_volume_mm)
            flag = f"   {note}" if note else ""
            print(
                f"    {obj.name:<26} {obj.mesh.n_faces:>8} tri   "
                f"{size[0]:6.1f} x {size[1]:6.1f} x {size[2]:6.1f} mm{flag}"
            )
        print()

    if project.notes:
        print("  what changes when moving to resin:")
        for note in sorted(project.notes, key=lambda n: LEVEL_ORDER.get(n.level, 9)):
            print(f"    {LEVEL_LABEL.get(note.level, note.level):<11} {note.message}")
        print()
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    project = load_project(args.file)
    print(f"\n  {args.file.split('/')[-1]}  -  mesh health\n")

    bloccanti = 0
    for obj in project.objects:
        mesh, report = prepare(to_trimesh(obj), name=obj.name, fill=not args.no_fill)
        stato = "ok  " if report.usable else "BLOC"
        if not report.usable:
            bloccanti += 1
        print(f"  {stato} {obj.name:<26} {report.faces_after:>8} tri   {report.summary()}")

        interventi = []
        if report.vertices_merged:
            interventi.append(f"{report.vertices_merged} duplicate vertices merged")
        if report.degenerate_removed:
            interventi.append(f"{report.degenerate_removed} degenerate faces removed")
        if report.duplicate_removed:
            interventi.append(f"{report.duplicate_removed} duplicate faces removed")
        if report.holes_filled:
            chiusi = report.boundary_edges_before - report.boundary_edges_left
            interventi.append(
                f"{report.holes_filled} planar faces rebuilt "
                f"({chiusi} of {report.boundary_edges_before} boundary edges closed)"
            )
        if report.winding_fixed:
            interventi.append("face winding fixed")
        if report.normals_inverted:
            interventi.append("normals flipped to point outward")
        for i in interventi:
            print(f"         repaired: {i}")
        for problema in report.problems:
            print(f"         PROBLEM: {problema}")

    print()
    if bloccanti:
        print(f"  {bloccanti} objects cannot be sliced as they are.\n")
        return 1
    print("  all meshes are ready to be sliced.\n")
    return 0


def cmd_orient(args: argparse.Namespace) -> int:
    project = load_project(args.file)
    printer = get_printer(args.printer)
    weights = PRESETS[args.priority]

    print(f"\n  {args.file.split('/')[-1]}  ->  {printer.vendor} {printer.model}")
    print(f"  priority: {args.priority}\n")

    for obj in project.objects:
        mesh, _ = prepare(to_trimesh(obj), name=obj.name)
        poses = auto_orient(mesh, printer.build_volume_mm, weights=weights)
        if not poses:
            print(f"  {obj.name}: empty geometry, skipped")
            continue

        best = poses[0]
        naturale = next((p for p in poses if p.tilt_deg < 2.0), None)

        print(f"  {obj.name}")
        print(f"    recommended  {best.describe()}")
        if naturale is not None and naturale is not best:
            guadagno = naturale.cross_section_mm2 / max(best.cross_section_mm2, 1e-9)
            print(f"    as it is now {naturale.describe()}")
            print(f"    -> max cross-section reduced {guadagno:.1f}x, "
                  f"height from {naturale.height_mm:.0f} to {best.height_mm:.0f} mm")
        for alt in poses[1 : 1 + args.alternatives]:
            print(f"    alternative  {alt.describe()}")
        print()
    return 0


def _profilo(aree: list[float], righe: int = 14) -> list[str]:
    """Cross-section area profile along the height, to see where the load is."""
    if not aree:
        return []
    massimo = max(aree) or 1.0
    passo = max(1, len(aree) // righe)
    out = []
    for start in range(0, len(aree), passo):
        blocco = aree[start : start + passo]
        media = sum(blocco) / len(blocco)
        barre = int(round(media / massimo * 46))
        out.append(f"    {start:>6} |{'#' * barre}{' ' * (46 - barre)}| {media:8.1f} mm2")
    return out


def cmd_slice(args: argparse.Namespace) -> int:
    import trimesh

    project = load_project(args.file)
    printer = get_printer(args.printer)
    settings = SliceSettings(layer_height_mm=args.layer_height, subsamples=args.antialias)

    oggetti = project.objects
    if args.object is not None:
        if not 0 <= args.object < len(oggetti):
            print(f"  object {args.object} does not exist (there are {len(oggetti)})")
            return 1
        oggetti = [oggetti[args.object]]

    preparate = []
    for obj in oggetti:
        mesh, report = prepare(to_trimesh(obj), name=obj.name)
        if not report.watertight:
            print(f"  note: '{obj.name}' is not watertight ({report.boundary_edges_left} open edges); "
                  "winding-number fill still works, but the cross-section may be inaccurate")
        if args.orient:
            posa = auto_orient(mesh, printer.build_volume_mm, weights=PRESETS[args.priority])[0]
            mesh = apply_orientation(mesh, posa)
            print(f"  '{obj.name}' oriented: tilt {posa.tilt_deg:.0f} degrees, "
                  f"height {posa.height_mm:.1f} mm, max cross-section ~{posa.cross_section_mm2:.0f} mm2")
        preparate.append(mesh)

    if len(preparate) > 1 and args.orient:
        print("  warning: with several objects oriented individually the poses may "
              "overlap; automatic plate arrangement is not implemented yet")

    scena = trimesh.util.concatenate(preparate) if len(preparate) > 1 else preparate[0]
    scena = place_on_plate(scena, printer, lift_mm=args.lift)

    dimensioni = scena.extents
    volume = printer.build_volume_mm
    if any(dimensioni[i] > volume[i] for i in range(3)):
        print(f"\n  DOES NOT FIT: footprint {dimensioni[0]:.1f} x {dimensioni[1]:.1f} x {dimensioni[2]:.1f} mm "
              f"vs {volume[0]} x {volume[1]} x {volume[2]} mm\n")
        return 1

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    print(f"\n  slicing at {settings.layer_height_mm} mm, antialiasing {settings.subsamples}x\n")
    aree: list[float] = []
    massima = (0.0, 0)
    vuoti = 0
    salvati = 0
    da_salvare: set[int] = set()

    layers = list(slice_mesh(scena, printer, settings))
    if args.out and layers:
        passo = max(1, len(layers) // max(1, args.preview))
        da_salvare = {i for i in range(0, len(layers), passo)}

    for layer in layers:
        aree.append(layer.area_mm2)
        if layer.empty:
            vuoti += 1
        if layer.area_mm2 > massima[0]:
            massima = (layer.area_mm2, layer.index)
        if layer.index in da_salvare:
            from PIL import Image

            Image.fromarray(layer.image).save(os.path.join(args.out, f"layer_{layer.index:05d}.png"))
            salvati += 1

    volume_mm3 = sum(aree) * settings.layer_height_mm
    print(f"  layers          {len(aree)} ({vuoti} empty)")
    print(f"  height          {len(aree) * settings.layer_height_mm:.2f} mm")
    print(f"  resin           {volume_mm3 / 1000.0:.2f} cm3")
    print(f"  max section     {massima[0]:.1f} mm2 at layer {massima[1]} "
          f"(z {massima[1] * settings.layer_height_mm:.1f} mm)")
    print(f"  first layer     {aree[0]:.1f} mm2\n")
    print("  cross-section profile (from the bottom):")
    for riga in _profilo(aree):
        print(riga)
    if salvati:
        print(f"\n  {salvati} PNG previews in {args.out}")
    print()
    return 0


def _anteprima_frontale(larghezza: int = 224, altezza: int = 168):
    """Build a front elevation view by stacking the layer projections.

    A top-down projection would only be the part's shadow; stacking one row per
    layer instead gives the silhouette seen from the front, which is
    recognizable on the printer's screen.
    """
    import numpy as np

    lungo_x: list = []
    lungo_y: list = []

    def aggiungi(image) -> None:
        # Two orthogonal elevations: from one direction a part may shrink to a
        # sliver, from the other it shows in full.
        lungo_x.append(image[::8].max(axis=0))
        lungo_y.append(image[:, ::8].max(axis=1))

    def componi():
        from PIL import Image

        if not lungo_x:
            return None

        migliore = None
        for righe in (lungo_x, lungo_y):
            # bottom-to-top becomes top-to-bottom
            candidata = Image.fromarray(np.flipud(np.array(righe, dtype=np.uint8)))
            riquadro = candidata.getbbox()
            if riquadro is None:
                continue
            ritagliata = candidata.crop(riquadro)
            # prefer the less squashed elevation, which shows more
            proporzione = min(ritagliata.width, ritagliata.height) / max(ritagliata.width, ritagliata.height)
            if migliore is None or proporzione > migliore[0]:
                migliore = (proporzione, ritagliata)

        if migliore is None:
            return None
        sorgente = migliore[1]

        # Scale preserving the aspect ratio: stretching the silhouette to the
        # thumbnail format would make it unrecognizable.
        fattore = min(larghezza / sorgente.width, altezza / sorgente.height)
        nuova = (max(1, int(sorgente.width * fattore)), max(1, int(sorgente.height * fattore)))
        tela = Image.new("L", (larghezza, altezza), 0)
        tela.paste(
            sorgente.resize(nuova, Image.BILINEAR),
            ((larghezza - nuova[0]) // 2, (altezza - nuova[1]) // 2),
        )
        grigio = np.array(tela)
        return np.repeat(grigio[:, :, None], 3, axis=2)

    return aggiungi, componi


def cmd_build(args: argparse.Namespace) -> int:
    import trimesh

    project = load_project(args.file)
    printer = get_printer(args.printer)
    grid = Grid.from_printer(printer)
    settings = SliceSettings(
        layer_height_mm=args.layer_height,
        subsamples=args.antialias,
        bleed_mm=args.bleed,
        elephant_foot_mm=(
            printer.elephant_foot_mm if args.elephant_foot is None else args.elephant_foot
        ),
        elephant_foot_layers=args.bottom_layers * 2,
    )
    params = PrintParameters(
        layer_height_mm=args.layer_height,
        exposure_s=args.exposure,
        bottom_exposure_s=args.bottom_exposure,
        bottom_layers=args.bottom_layers,
    )

    oggetti = project.objects
    if args.object is not None:
        if not 0 <= args.object < len(oggetti):
            print(f"  object {args.object} does not exist (there are {len(oggetti)})")
            return 1
        oggetti = [oggetti[args.object]]

    preparate = []
    for obj in oggetti:
        mesh, report = prepare(to_trimesh(obj), name=obj.name)
        if not report.watertight:
            print(f"  note: '{obj.name}' is not watertight ({report.boundary_edges_left} open edges)")
        if args.orient:
            posa = auto_orient(mesh, printer.build_volume_mm, weights=PRESETS[args.priority])[0]
            mesh = apply_orientation(mesh, posa)
            print(f"  '{obj.name}': tilted {posa.tilt_deg:.0f} degrees, "
                  f"height {posa.height_mm:.1f} mm, max cross-section ~{posa.cross_section_mm2:.0f} mm2")
        preparate.append(mesh)

    scena = trimesh.util.concatenate(preparate) if len(preparate) > 1 else preparate[0]

    if args.shrink_xy or args.shrink_z:
        # Must be applied before hollowing and supporting, so everything else
        # is generated at the final scale.
        prima = scena.extents.copy()
        scena = scale_for_shrinkage(scena, args.shrink_xy, args.shrink_z)
        print(f"  shrinkage compensated: {prima[0]:.2f} x {prima[1]:.2f} x {prima[2]:.2f} -> "
              f"{scena.extents[0]:.2f} x {scena.extents[1]:.2f} x {scena.extents[2]:.2f} mm")

    scena = place_on_plate(scena, printer)

    pezzi_cavita: list = []
    if args.hollow:
        _assieme, cavo = hollow(
            scena,
            HollowSettings(
                wall_mm=args.wall,
                voxel_mm=args.voxel,
                drain_diameter_mm=args.drain_diameter,
            ),
        )
        # `scena` stays the outer surface only: inner shell and holes travel
        # separately, because supports must rest only on the outside.
        pezzi_cavita = cavo.subtractive
        if cavo.hollowed:
            print(f"  hollowed: wall {cavo.achieved_wall_mm:.2f} mm, "
                  f"cavity {cavo.cavity_volume_mm3 / 1000:.2f} cm3 "
                  f"(-{100 * cavo.saved_fraction:.0f}% resin), {cavo.drains} holes, "
                  f"drainage {'ok' if cavo.drained else 'BLOCKED'}")
        for nota in cavo.notes:
            print(f"  {nota}")

    supporti = None
    if not args.no_supports:
        support_settings = SupportSettings(
            lift_mm=args.lift,
            spacing_mm=args.spacing,
            overhang_angle_deg=args.overhang_angle,
            tip_diameter_mm=args.tip_diameter,
        )
        scena, supporti = support_until_clean(
            scena,
            printer,
            settings,
            support_settings,
            max_rounds=args.rounds,
            min_island_mm2=args.min_island,
            extra=pezzi_cavita,
            on_round=lambda giro, punti, isole: print(
                f"  round {giro}: {punti} pillars -> {isole} islands left"
            ),
        )
        print(f"  {supporti.pillars} pillars ({supporti.landed_on_model} resting on the part), "
              f"raft {supporti.raft_area_mm2:.0f} mm2, "
              f"support resin {supporti.added_volume_mm3 / 1000.0:.2f} cm3")
        for nota in supporti.notes:
            print(f"  {nota}")

    if args.no_supports and pezzi_cavita:
        scena = trimesh.util.concatenate([scena, *pezzi_cavita])

    ingombro = scena.extents
    volume = printer.build_volume_mm
    if any(ingombro[i] > volume[i] for i in range(3)):
        print(f"\n  DOES NOT FIT: {ingombro[0]:.1f} x {ingombro[1]:.1f} x {ingombro[2]:.1f} mm "
              f"vs {volume[0]} x {volume[1]} x {volume[2]} mm\n")
        return 1

    aggiungi, componi = _anteprima_frontale()
    guardia = IslandWatcher(grid, min_area_mm2=args.min_island)
    volume_mm3 = 0.0
    primo = None

    def flusso():
        nonlocal volume_mm3, primo
        for layer in slice_mesh(scena, printer, settings):
            volume_mm3 += layer.area_mm2 * settings.layer_height_mm
            if primo is None:
                primo = layer.area_mm2
            guardia.feed(layer)
            aggiungi(layer.image)
            yield layer.image

    print(f"  slicing at {settings.layer_height_mm} mm and writing {args.out}")
    risultato = write_pm3(args.out, flusso(), printer, params, volume_mm3=lambda: volume_mm3)

    miniatura = componi()
    if miniatura is not None:
        patch_preview(args.out, miniatura)

    riletto = read_pm3(args.out)
    print(f"\n  {risultato.layers} layers, {risultato.bytes_written / 1e6:.2f} MB")
    print(f"  resin           {risultato.volume_ml:.2f} ml (~{risultato.volume_ml * 1.1:.1f} g)")
    print(f"  estimated time  {risultato.print_time_s // 3600}h {(risultato.print_time_s % 3600) // 60}m")
    print(f"  first layer     {primo:.1f} mm2")
    compensazioni = []
    if settings.bleed_mm:
        compensazioni.append(f"bleed {settings.bleed_mm:.3f} mm")
    if settings.elephant_foot_mm:
        compensazioni.append(
            f"elephant foot {settings.elephant_foot_mm:.2f} mm over {settings.elephant_foot_layers} layers"
        )
    if args.shrink_xy or args.shrink_z:
        compensazioni.append(f"shrinkage {args.shrink_xy:g}%/{args.shrink_z:g}%")
    print(f"  compensations   {', '.join(compensazioni) if compensazioni else 'none'}")
    print(f"  read-back       {'consistent' if riletto.valid else riletto.problems}")

    print()
    if guardia.islands:
        print(f"  WARNING: {len(guardia.islands)} unsupported islands left. "
              "They will float in the vat.")
        for isola in guardia.islands[:5]:
            print(f"    {isola}")
        if len(guardia.islands) > 5:
            print(f"    ... and {len(guardia.islands) - 5} more")
    elif not args.no_supports:
        print("  no islands left: every part of the model grows from something solid")

    if primo is not None and primo < 20.0:
        print(f"  WARNING: the first layer touches the plate over just {primo:.1f} mm2: "
              "the part may detach")
    for nota in risultato.assumptions:
        print(f"  assumption: {nota}")
    print()
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    printer = get_printer(args.printer)
    settings = SliceSettings(layer_height_mm=args.layer_height, subsamples=args.antialias)

    if args.kind == "exposure":
        tempi = tempi_geometrici(args.t_from, args.t_to, args.steps)
        provino = torre_esposizione(tempi)
        params = PrintParameters(
            layer_height_mm=args.layer_height,
            exposure_s=tempi[len(tempi) // 2],
            bottom_exposure_s=args.bottom_exposure,
            bottom_layers=args.bottom_layers,
        )
    else:
        provino = piastra_tolleranza()
        params = PrintParameters(
            layer_height_mm=args.layer_height,
            exposure_s=args.exposure,
            bottom_exposure_s=args.bottom_exposure,
            bottom_layers=args.bottom_layers,
        )

    scena = place_on_plate(provino.mesh, printer)
    ingombro = scena.extents
    print(f"\n  {provino.nome}: {ingombro[0]:.1f} x {ingombro[1]:.1f} x {ingombro[2]:.1f} mm")

    def esposizione(index: int) -> float:
        # The first layers keep the bottom exposure: without them the test piece
        # does not stick to the plate, and the test would be wasted.
        if index < params.bottom_layers:
            return params.bottom_exposure_s
        return provino.exposure_for_z((index + 0.5) * settings.layer_height_mm, params.exposure_s)

    per_layer = esposizione if provino.fasce else None

    aggiungi, componi = _anteprima_frontale()
    volume_mm3 = 0.0

    def flusso():
        nonlocal volume_mm3
        for layer in slice_mesh(scena, printer, settings):
            volume_mm3 += layer.area_mm2 * settings.layer_height_mm
            aggiungi(layer.image)
            yield layer.image

    risultato = write_pm3(
        args.out,
        flusso(),
        printer,
        params,
        volume_mm3=lambda: volume_mm3,
        exposure_for_layer=per_layer,
    )
    miniatura = componi()
    if miniatura is not None:
        patch_preview(args.out, miniatura)

    riletto = read_pm3(args.out)
    print(f"  {risultato.layers} layers, {risultato.bytes_written / 1e6:.2f} MB, "
          f"{risultato.volume_ml:.2f} ml, {risultato.print_time_s // 60} minutes")
    print(f"  read-back {'consistent' if riletto.valid else riletto.problems}\n")

    istruzioni = "\n".join(provino.istruzioni)
    print(istruzioni + "\n")
    guida = args.out.rsplit(".", 1)[0] + "_instructions.txt"
    with open(guida, "w") as handle:
        handle.write(f"{provino.nome}\n{'=' * len(provino.nome)}\n\n{istruzioni}\n")
    print(f"  instructions saved to {guida}\n")
    return 0


def cmd_printers(args: argparse.Namespace) -> int:
    for printer in REGISTRY.values():
        print()
        print(printer.describe())
        print(f"  source        {printer.source}")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="travaso", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="analyze a .3mf and report what the conversion implies")
    p_inspect.add_argument("file")
    p_inspect.add_argument("--printer", default="photon_m3")
    p_inspect.set_defaults(func=cmd_inspect)

    p_check = sub.add_parser("check", help="check and repair meshes before printing")
    p_check.add_argument("file")
    p_check.add_argument("--no-fill", action="store_true", help="do not fill holes, only report them")
    p_check.set_defaults(func=cmd_check)

    p_orient = sub.add_parser("orient", help="compute the best print orientation")
    p_orient.add_argument("file")
    p_orient.add_argument("--printer", default="photon_m3")
    p_orient.add_argument("--priority", "--priorita", type=_priority, default="reliability",
                            choices=sorted(PRESETS), help="orientation weighting preset")
    p_orient.add_argument("--alternatives", "--alternative", dest="alternatives", type=int, default=0, help="how many alternative poses to show")
    p_orient.set_defaults(func=cmd_orient)

    p_slice = sub.add_parser("slice", help="slice and report print statistics")
    p_slice.add_argument("file")
    p_slice.add_argument("--printer", default="photon_m3")
    p_slice.add_argument("--layer-height", type=float, default=0.05)
    p_slice.add_argument("--antialias", type=int, default=4, help="sub-rows per pixel")
    p_slice.add_argument("--object", type=int, default=None, help="slice a single object")
    p_slice.add_argument("--orient", action="store_true", help="apply the recommended pose")
    p_slice.add_argument("--priority", "--priorita", type=_priority, default="reliability",
                           choices=sorted(PRESETS), help="orientation weighting preset")
    p_slice.add_argument("--lift", type=float, default=0.0, help="lift height above the plate")
    p_slice.add_argument("--out", default=None, help="folder in which to save PNG previews")
    p_slice.add_argument("--preview", type=int, default=8, help="how many previews to save")
    p_slice.set_defaults(func=cmd_slice)

    p_build = sub.add_parser("build", help="full pipeline: from 3mf to .pm3 file")
    p_build.add_argument("file")
    p_build.add_argument("--out", required=True, help=".pm3 file to write")
    p_build.add_argument("--printer", default="photon_m3")
    p_build.add_argument("--layer-height", type=float, default=0.05)
    p_build.add_argument("--antialias", type=int, default=4)
    p_build.add_argument("--object", type=int, default=None)
    p_build.add_argument("--orient", action="store_true")
    p_build.add_argument("--priority", "--priorita", type=_priority, default="reliability",
                           choices=sorted(PRESETS), help="orientation weighting preset")
    p_build.add_argument("--lift", type=float, default=5.0, help="lift above the plate")
    p_build.add_argument("--no-supports", action="store_true", help="do not generate supports and raft")
    p_build.add_argument("--spacing", type=float, default=4.0, help="distance between supports")
    p_build.add_argument("--overhang-angle", type=float, default=45.0)
    p_build.add_argument("--tip-diameter", type=float, default=0.4)
    p_build.add_argument("--min-island", type=float, default=0.02, help="minimum island area in mm2")
    p_build.add_argument("--rounds", type=int, default=4, help="maximum support rounds")
    p_build.add_argument("--hollow", action="store_true", help="hollow the part")
    p_build.add_argument("--wall", type=float, default=2.0, help="wall thickness in mm")
    p_build.add_argument("--voxel", type=float, default=0.3, help="discretization step")
    p_build.add_argument("--drain-diameter", type=float, default=3.0)
    p_build.add_argument("--bleed", type=float, default=0.0,
                         help="how much the printer enlarges the part per side, in mm")
    p_build.add_argument("--elephant-foot", type=float, default=None,
                         help="enlargement of the first layers (default: the printer's)")
    p_build.add_argument("--shrink-xy", type=float, default=0.0,
                         help="resin shrinkage in XY, in percent")
    p_build.add_argument("--shrink-z", type=float, default=0.0,
                         help="resin shrinkage in Z, in percent")
    p_build.add_argument("--exposure", type=float, default=2.5, help="exposure in seconds")
    p_build.add_argument("--bottom-exposure", type=float, default=30.0)
    p_build.add_argument("--bottom-layers", type=int, default=6)
    p_build.set_defaults(func=cmd_build)

    p_cal = sub.add_parser("calibrate", help="generate a test piece to measure exposure and bleed")
    p_cal.add_argument("kind", choices=["exposure", "tolerance"])
    p_cal.add_argument("--out", required=True)
    p_cal.add_argument("--printer", default="photon_m3")
    p_cal.add_argument("--layer-height", type=float, default=0.05)
    p_cal.add_argument("--antialias", type=int, default=4)
    p_cal.add_argument("--from", "--da", dest="t_from", type=float, default=1.5, help="minimum time to test")
    p_cal.add_argument("--to", "--a", dest="t_to", type=float, default=6.0, help="maximum time to test")
    p_cal.add_argument("--steps", "--passi", dest="steps", type=int, default=8, help="how many bands")
    p_cal.add_argument("--exposure", type=float, default=2.5, help="exposure time for the plate")
    p_cal.add_argument("--bottom-exposure", type=float, default=30.0)
    p_cal.add_argument("--bottom-layers", type=int, default=6)
    p_cal.set_defaults(func=cmd_calibrate)

    p_printers = sub.add_parser("printers", help="list supported printers")
    p_printers.set_defaults(func=cmd_printers)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
