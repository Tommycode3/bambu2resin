"""Single entry point to load a project, whatever its format.

The extension cannot be trusted: binary STLs renamed to .3mf turn up
regularly. The type must be detected from the content.
"""

from __future__ import annotations

import os
import zipfile

import numpy as np

from .bambu.model import Mesh, ModelObject, Note, Plate, Project
from .bambu.reader import read_project
from .diagnostics import size_notes

# signatures recognized at the start of the file
_MAGIC = {
    b"solid": "stl-ascii",
    b"ply\n": "ply",
    b"ply\r": "ply",
}


def sniff_format(path: str) -> str:
    """Detect the real format by reading the start of the file."""
    if zipfile.is_zipfile(path):
        return "3mf"

    with open(path, "rb") as fp:
        head = fp.read(84)

    for magic, kind in _MAGIC.items():
        if head.startswith(magic):
            return kind

    # Binary STL: 80 bytes of free-form header, then the triangle count.
    # Check that the count matches the file size, so that an arbitrary file
    # is not mistaken for an STL.
    if len(head) == 84:
        count = int.from_bytes(head[80:84], "little")
        if os.path.getsize(path) == 84 + count * 50:
            return "stl-binary"

    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return ext or "unknown"


def load_project(path: str) -> Project:
    kind = sniff_format(path)
    if kind == "3mf":
        return read_project(path)

    if kind in ("stl-binary", "stl-ascii", "ply", "obj"):
        return _load_single_mesh(path, kind)

    raise ValueError(
        f"unrecognized format for {os.path.basename(path)} (detected: {kind}). "
        "Supported: .3mf, .stl, .obj, .ply"
    )


def _load_single_mesh(path: str, kind: str) -> Project:
    import trimesh

    notes: list[Note] = []
    declared = os.path.splitext(path)[1].lower().lstrip(".")
    if declared and not kind.startswith(declared):
        notes.append(
            Note(
                "attenzione",
                f"the file is named .{declared} but its content is {kind}: "
                "read according to the actual content",
            )
        )

    # trimesh picks the loader from the extension: on a renamed file it would
    # go the wrong way, so it is given the actually detected type.
    file_type = {"stl-binary": "stl", "stl-ascii": "stl"}.get(kind, kind)
    with open(path, "rb") as fp:
        loaded = trimesh.load(fp, file_type=file_type, force="mesh")
    mesh = Mesh(
        vertices=np.asarray(loaded.vertices, dtype=np.float64),
        faces=np.asarray(loaded.faces, dtype=np.int32),
    )
    name = os.path.splitext(os.path.basename(path))[0]

    obj = ModelObject(
        object_id=1,
        part_index=0,
        name=name,
        mesh=mesh,
        transform=np.eye(4),
        plate=1,
    )
    notes.append(
        Note("info", f"single mesh ({kind}): no project data, geometry only")
    )
    project = Project(path=path, application=kind, objects=[obj], plates=[Plate(index=1)])
    project.notes = notes + size_notes(project)
    return project
