"""Checks that apply to any geometry source."""

from __future__ import annotations

from .bambu.model import Note, Project


def size_notes(project: Project) -> list[Note]:
    """Implausible sizes: almost always a model in normalized units."""
    notes: list[Note] = []
    for obj in project.objects:
        largest = float(obj.size().max())
        # A part genuinely shorter than 5 mm exists, but it is rare: far more
        # often the model arrives in normalized units (side ~1) from a CAD
        # export or a generator, and must be scaled before printing.
        if largest < 5.0:
            notes.append(
                Note(
                    "attenzione",
                    f"'{obj.name}' measures {largest:.3f} mm on its longest side: check the scale, "
                    "the model seems to be in normalized units rather than millimeters",
                )
            )
        elif largest > 1000.0:
            notes.append(
                Note("attenzione", f"'{obj.name}' measures {largest:.0f} mm: implausible size")
            )
    return notes
