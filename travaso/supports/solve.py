"""Support until no islands are left.

Placing supports in one shot is not enough, for a structural reason: islands
show up when slicing, but what gets sliced depends on the supports already
placed. Lifting the part off the build plate creates new ones (the bottom,
which used to rest on the plate, is now suspended), and a pillar can create
some in turn.

So we proceed in rounds: build, slice, see what is left unsupported, add,
retry. The stopping criterion is not a number of attempts but a check: no
islands on the layers the printer will receive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from ..slicer.raster import Grid
from ..slicer.stack import SliceSettings, slice_mesh
from .build import SupportSettings, build_raft, build_supports, overhang_points, thin_points
from .islands import Island, find_islands


@dataclass
class SolveReport:
    rounds: int = 0
    seed_points: int = 0
    overhang_points: int = 0
    islands_found: int = 0
    islands_remaining: int = 0
    pillars: int = 0
    landed_on_model: int = 0
    raft_area_mm2: float = 0.0
    added_volume_mm3: float = 0.0
    remaining: list[Island] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.islands_remaining == 0


def lowest_points(mesh: trimesh.Trimesh, settings: SupportSettings) -> np.ndarray:
    """The lowest points of the part, which are left hanging once it is lifted.

    They are mandatory supports: whatever lies above grows from there.
    """
    z = mesh.vertices[:, 2]
    floor = float(z.min())
    # a thin band above the minimum, to also catch nearby tips
    band = mesh.vertices[z <= floor + max(settings.tip_length_mm * 0.5, 0.2)]
    if len(band) == 0:
        return np.zeros((0, 3))
    return thin_points(band, settings.spacing_mm)


def support_until_clean(
    mesh: trimesh.Trimesh,
    printer,
    slice_settings: SliceSettings | None = None,
    settings: SupportSettings | None = None,
    max_rounds: int = 4,
    min_island_mm2: float = 0.02,
    on_round=None,
    extra: list[trimesh.Trimesh] | None = None,
) -> tuple[trimesh.Trimesh, SolveReport]:
    """Lift the part, support it and verify by slicing, in rounds.

    `extra` are bodies that belong to the assembly to slice but not to the
    surface the supports rest on: typically the inner shell and the holes of
    a hollowing. Sampling overhangs on those too would place pillars inside
    the cavity, where they are not needed and cannot be removed.
    """
    settings = settings or SupportSettings()
    slice_settings = slice_settings or SliceSettings()
    grid = Grid.from_printer(printer)
    report = SolveReport()

    rise = settings.lift_mm - float(mesh.bounds[0][2])
    lifted = mesh.copy()
    lifted.apply_translation([0.0, 0.0, rise])

    others: list[trimesh.Trimesh] = []
    for piece in extra or []:
        moved = piece.copy()
        moved.apply_translation([0.0, 0.0, rise])
        others.append(moved)

    seeds = lowest_points(lifted, settings)
    hangs = overhang_points(lifted, settings)
    report.seed_points = len(seeds)
    report.overhang_points = len(hangs)

    points = np.vstack([seeds, hangs]) if len(hangs) else seeds
    if len(points) == 0:
        report.notes.append("no points to support")
        return trimesh.util.concatenate([lifted, *others]) if others else lifted, report

    assembly = lifted
    for round_index in range(1, max_rounds + 1):
        report.rounds = round_index
        pieces, feet, landed = build_supports(lifted, points, settings)
        report.pillars = len(points)
        report.landed_on_model = landed

        raft, area = build_raft(feet, settings)
        report.raft_area_mm2 = area
        if raft is not None:
            pieces.append(raft)
        report.added_volume_mm3 = float(sum(max(p.volume, 0.0) for p in pieces))

        assembly = trimesh.util.concatenate([lifted, *others, *pieces])

        remaining = list(
            find_islands(
                slice_mesh(assembly, printer, slice_settings), grid, min_area_mm2=min_island_mm2
            )
        )
        if on_round is not None:
            on_round(round_index, len(points), len(remaining))

        if not remaining:
            report.islands_remaining = 0
            report.remaining = []
            return assembly, report

        if round_index == 1:
            report.islands_found = len(remaining)

        # Remaining islands become new points to support. The heights are
        # already in the assembly frame, because the raft starts at z=0.
        fresh = np.array([[i.centroid_mm[0], i.centroid_mm[1], i.z_mm] for i in remaining])
        before = len(points)
        points = np.vstack([points, fresh])
        if len(points) == before:
            report.notes.append("no new points to add: stopping")
            break

    report.islands_remaining = len(remaining)
    report.remaining = remaining
    report.notes.append(
        f"after {report.rounds} rounds {len(remaining)} islands remain: "
        "try reducing --spacing or changing the orientation"
    )
    return assembly, report
