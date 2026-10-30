"""Data model of a Bambu Studio project read from .3mf."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Per-triangle attributes Bambu uses for painting. They are all FDM concepts:
# resin has neither multi-material nor painted supports, so they are lost.
# Not a bug, but the user must be told.
FDM_FACE_ATTRS = {
    "paint_color": "multi-material/color painting (AMS)",
    "mmu_segmentation": "multi-material segmentation",
    "face_property": "per-face painting (color/supports/seam)",
    "paint_supports": "hand-painted supports",
    "paint_seam": "painted seam position",
    "paint_fuzzy_skin": "painted fuzzy skin",
}

# Subtypes of 'part' inside a Bambu object. Only normal_part is real geometry
# to print; the rest modulates the FDM slicer's behavior.
PART_SUBTYPES_GEOMETRY = {"normal_part"}
PART_SUBTYPES_FDM_ONLY = {
    "modifier_part": "parameter modifier",
    "negative_part": "negative part (subtraction)",
    "support_blocker": "support blocker",
    "support_enforcer": "support enforcer",
}


@dataclass
class Mesh:
    """Triangle mesh in the object's local coordinates."""

    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray  # (M, 3) int32
    face_attrs: set[str] = field(default_factory=set)

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


@dataclass
class ModelObject:
    """An object placed on the plate, with its mesh already transformed."""

    object_id: int
    part_index: int
    name: str
    mesh: Mesh
    transform: np.ndarray  # 4x4, local -> global
    extruder: int | None = None
    plate: int | None = None
    printable: bool = True
    subtype: str = "normal_part"
    settings: dict[str, str] = field(default_factory=dict)

    def transformed_vertices(self) -> np.ndarray:
        v = self.mesh.vertices
        return v @ self.transform[:3, :3].T + self.transform[:3, 3]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        v = self.transformed_vertices()
        return v.min(axis=0), v.max(axis=0)

    def size(self) -> np.ndarray:
        lo, hi = self.bounds()
        return hi - lo


@dataclass
class Plate:
    index: int
    name: str = ""
    object_ids: list[int] = field(default_factory=list)


@dataclass
class Note:
    """Something the user needs to know about the conversion."""

    level: str  # "info" | "attenzione" (warning) | "perdita" (loss)
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.message}"


@dataclass
class Project:
    path: str
    application: str = ""
    objects: list[ModelObject] = field(default_factory=list)
    plates: list[Plate] = field(default_factory=list)
    print_settings: dict = field(default_factory=dict)
    notes: list[Note] = field(default_factory=list)

    def objects_on_plate(self, index: int) -> list[ModelObject]:
        return [o for o in self.objects if o.plate == index]
