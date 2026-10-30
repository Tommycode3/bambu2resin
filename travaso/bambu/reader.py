"""Reader for Bambu Studio projects (.3mf).

A Bambu 3MF is a ZIP archive with:
  3D/3dmodel.model            root document: resources + placement (`build`)
  3D/Objects/object_N.model   meshes referenced via the "production" extension
                              (p:path attribute); sometimes, though, the meshes
                              live directly in the root document
  Metadata/model_settings.config   names, extruder, parts, plates
  Metadata/project_settings.config FDM print parameters (JSON)

Placement is read through the standard 3MF path (component transform composed
with the build item transform), not from Bambu's internal matrices: the two do
not match and only the former is normative.

Each .model document is traversed ONCE with expat, in streaming mode. A root
document can hold 40 MB of geometry: building its DOM tree costs hundreds of
MB for a few MB of useful mesh.
"""

from __future__ import annotations

import json
import xml.parsers.expat
import zipfile
from array import array
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

import numpy as np

from ..diagnostics import size_notes
from ..geometry.transform import parse_3mf_transform
from .model import (
    FDM_FACE_ATTRS,
    PART_SUBTYPES_FDM_ONLY,
    Mesh,
    ModelObject,
    Note,
    Plate,
    Project,
)

ROOT_MODEL = "3D/3dmodel.model"
MODEL_SETTINGS = "Metadata/model_settings.config"
PROJECT_SETTINGS = "Metadata/project_settings.config"

NS_PRODUCTION = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
# expat with namespace_separator=" " returns names as "uri local"
ATTR_PATH_SAX = f"{NS_PRODUCTION} path"

# The 3MF declares the unit in the <model> tag. Bambu always writes millimeter,
# but a .3mf exported by other tools may be in meters or inches: without
# conversion the part comes out a thousand times smaller.
UNIT_TO_MM = {
    "micron": 0.001,
    "millimeter": 1.0,
    "centimeter": 10.0,
    "inch": 25.4,
    "foot": 304.8,
    "meter": 1000.0,
}


@dataclass
class _ObjectDef:
    """Raw definition of an `<object>` inside a .model document."""

    mesh: Mesh | None = None
    components: list[tuple[str | None, int, np.ndarray]] = field(default_factory=list)


@dataclass
class _BuildItem:
    object_id: int
    transform: np.ndarray
    printable: bool = True


@dataclass
class _Doc:
    """Everything needed from a .model document, collected in one pass."""

    objects: dict[int, _ObjectDef] = field(default_factory=dict)
    build_items: list[_BuildItem] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    unit: str = "millimeter"


def _parse_doc_streaming(fp) -> _Doc:
    """Traverse a .model document with expat without building trees."""
    doc = _Doc()

    obj_id: int | None = None
    verts = array("d")
    faces = array("i")
    attrs_seen: set[str] = set()
    comps: list[tuple[str | None, int, np.ndarray]] = []
    has_mesh = False

    meta_key: str | None = None
    meta_buf: list[str] = []

    def start(name: str, attrs: dict) -> None:
        nonlocal obj_id, verts, faces, attrs_seen, comps, has_mesh, meta_key, meta_buf
        tag = name.rsplit(" ", 1)[-1]

        if tag == "vertex":
            if obj_id is None:
                return
            verts.append(float(attrs["x"]))
            verts.append(float(attrs["y"]))
            verts.append(float(attrs["z"]))

        elif tag == "triangle":
            if obj_id is None:
                return
            faces.append(int(attrs["v1"]))
            faces.append(int(attrs["v2"]))
            faces.append(int(attrs["v3"]))
            if len(attrs) > 3:
                for key in attrs:
                    local = key.rsplit(" ", 1)[-1]
                    if local not in ("v1", "v2", "v3"):
                        attrs_seen.add(local)

        elif tag == "object":
            obj_id = int(attrs.get("id", 0))
            verts, faces = array("d"), array("i")
            attrs_seen, comps = set(), []
            has_mesh = False

        elif tag == "mesh":
            has_mesh = True

        elif tag == "component":
            comps.append(
                (
                    attrs.get(ATTR_PATH_SAX) or attrs.get("p:path"),
                    int(attrs.get("objectid", 0)),
                    parse_3mf_transform(attrs.get("transform")),
                )
            )

        elif tag == "item":
            doc.build_items.append(
                _BuildItem(
                    object_id=int(attrs.get("objectid", 0)),
                    transform=parse_3mf_transform(attrs.get("transform")),
                    printable=attrs.get("printable", "1") not in ("0", "false"),
                )
            )

        elif tag == "model":
            doc.unit = (attrs.get("unit") or "millimeter").lower()

        elif tag == "metadata":
            meta_key = attrs.get("name")
            meta_buf = []

    def chardata(data: str) -> None:
        if meta_key is not None:
            meta_buf.append(data)

    def end(name: str) -> None:
        nonlocal obj_id, verts, faces, attrs_seen, comps, meta_key
        tag = name.rsplit(" ", 1)[-1]

        if tag == "metadata" and meta_key is not None:
            doc.metadata[meta_key] = "".join(meta_buf).strip()
            meta_key = None
            return

        if tag != "object" or obj_id is None:
            return

        mesh = None
        if has_mesh and len(faces):
            mesh = Mesh(
                vertices=np.frombuffer(verts, dtype=np.float64).reshape(-1, 3).copy(),
                faces=np.frombuffer(faces, dtype=np.int32).reshape(-1, 3).copy(),
                face_attrs=set(attrs_seen),
            )
        doc.objects[obj_id] = _ObjectDef(mesh=mesh, components=list(comps))
        obj_id = None
        verts, faces = array("d"), array("i")
        attrs_seen, comps = set(), []

    parser = xml.parsers.expat.ParserCreate(namespace_separator=" ")
    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = chardata
    parser.ParseFile(fp)
    return doc


class BambuProjectReader:
    def __init__(self, path: str):
        self.path = path
        self._zip = zipfile.ZipFile(path)
        self._docs: dict[str, _Doc] = {}
        self.notes: list[Note] = []

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> "BambuProjectReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _doc(self, path: str) -> _Doc:
        path = path.lstrip("/")
        if path not in self._docs:
            with self._zip.open(path) as fp:
                self._docs[path] = _parse_doc_streaming(fp)
        return self._docs[path]

    # -------------------------------------------------------------- resolution

    def _resolve(
        self, path: str | None, object_id: int, depth: int = 0
    ) -> list[tuple[Mesh, np.ndarray]]:
        """Expand an object into its leaf meshes with the composed transform."""
        if depth > 16:
            raise RecursionError("3MF component hierarchy too deep")

        doc_path = (path or ROOT_MODEL).lstrip("/")
        definition = self._doc(doc_path).objects.get(object_id)
        if definition is None:
            self.notes.append(
                Note("attenzione", f"object {object_id} not found in {doc_path}, skipped")
            )
            return []

        if definition.mesh is not None:
            return [(definition.mesh, np.eye(4))]

        out: list[tuple[Mesh, np.ndarray]] = []
        for comp_path, comp_id, comp_tf in definition.components:
            for mesh, child_tf in self._resolve(comp_path or doc_path, comp_id, depth + 1):
                out.append((mesh, comp_tf @ child_tf))
        return out

    # --------------------------------------------------------------- metadata

    def _read_model_settings(self) -> tuple[dict[int, dict], list[Plate]]:
        settings: dict[int, dict] = {}
        plates: list[Plate] = []
        if MODEL_SETTINGS not in self._zip.namelist():
            return settings, plates

        root = ET.fromstring(self._zip.read(MODEL_SETTINGS))
        for node in root:
            tag = node.tag.rsplit("}", 1)[-1]
            if tag == "object":
                oid = int(node.get("id", "0"))
                meta = {m.get("key"): m.get("value") for m in node.findall("metadata") if m.get("key")}
                parts = [
                    {
                        "id": int(part.get("id", "0")),
                        "subtype": part.get("subtype", "normal_part"),
                        "name": next(
                            (m.get("value") for m in part.findall("metadata") if m.get("key") == "name"),
                            "",
                        ),
                    }
                    for part in node.findall("part")
                ]
                settings[oid] = {"meta": meta, "parts": parts}
            elif tag == "plate":
                meta = {m.get("key"): m.get("value") for m in node.findall("metadata") if m.get("key")}
                plate = Plate(
                    index=int(meta.get("plater_id", len(plates) + 1)),
                    name=meta.get("plater_name", "") or "",
                )
                for inst in node.findall("model_instance"):
                    imeta = {
                        m.get("key"): m.get("value") for m in inst.findall("metadata") if m.get("key")
                    }
                    if "object_id" in imeta:
                        plate.object_ids.append(int(imeta["object_id"]))
                plates.append(plate)
        return settings, plates

    def _read_project_settings(self) -> dict:
        if PROJECT_SETTINGS not in self._zip.namelist():
            return {}
        try:
            return json.loads(self._zip.read(PROJECT_SETTINGS))
        except json.JSONDecodeError:
            self.notes.append(Note("attenzione", "project_settings.config unreadable"))
            return {}

    # ------------------------------------------------------------------ reading

    def read(self) -> Project:
        root_doc = self._doc(ROOT_MODEL)

        unit_scale = UNIT_TO_MM.get(root_doc.unit)
        if unit_scale is None:
            self.notes.append(
                Note("attenzione", f"unknown 3MF unit '{root_doc.unit}', assuming millimeters")
            )
            unit_scale = 1.0
        elif unit_scale != 1.0:
            self.notes.append(
                Note("info", f"file unit: {root_doc.unit} (converted to mm, x{unit_scale:g})")
            )

        obj_settings, plates = self._read_model_settings()
        plate_of = {oid: plate.index for plate in plates for oid in plate.object_ids}

        objects: list[ModelObject] = []
        for item in root_doc.build_items:
            oid = item.object_id
            info = obj_settings.get(oid, {})
            meta = info.get("meta", {})
            parts = info.get("parts", [])
            name = meta.get("name") or f"object {oid}"
            extruder = int(meta["extruder"]) if str(meta.get("extruder", "")).isdigit() else None

            resolved = self._resolve(None, oid)
            if not resolved:
                continue

            item_tf = item.transform
            if unit_scale != 1.0:
                item_tf = item_tf.copy()
                item_tf[:3, :] *= unit_scale

            for index, (mesh, local_tf) in enumerate(resolved):
                part = parts[index] if index < len(parts) else {}
                objects.append(
                    ModelObject(
                        object_id=oid,
                        part_index=index,
                        name=part.get("name") or name,
                        mesh=mesh,
                        transform=item_tf @ local_tf,
                        extruder=extruder,
                        plate=plate_of.get(oid),
                        printable=item.printable,
                        subtype=part.get("subtype", "normal_part"),
                        settings=meta,
                    )
                )

        project = Project(
            path=self.path,
            application=root_doc.metadata.get("Application", ""),
            objects=objects,
            plates=plates or [Plate(index=1)],
            print_settings=self._read_project_settings(),
        )
        if not plates:
            for obj in project.objects:
                obj.plate = 1

        project.notes = self.notes + self._diagnose(project)
        return project

    # ------------------------------------------------------------ diagnostics

    def _diagnose(self, project: Project) -> list[Note]:
        """Report what in the FDM project does not survive the move to resin."""
        notes: list[Note] = []

        attrs_found: set[str] = set()
        for obj in project.objects:
            attrs_found |= obj.mesh.face_attrs
        for attr in sorted(attrs_found):
            label = FDM_FACE_ATTRS.get(attr, attr)
            notes.append(Note("perdita", f"per-face painting '{attr}' ({label}): does not exist in resin"))

        extruders = {o.extruder for o in project.objects if o.extruder}
        if len(extruders) > 1:
            notes.append(
                Note(
                    "perdita",
                    f"the project uses {len(extruders)} extruders/AMS colors ({sorted(extruders)}): "
                    "resin is single-material, the objects must be printed separately and assembled",
                )
            )

        for obj in project.objects:
            if obj.subtype in PART_SUBTYPES_FDM_ONLY:
                notes.append(
                    Note(
                        "perdita",
                        f"'{obj.name}' is a {PART_SUBTYPES_FDM_ONLY[obj.subtype]}: "
                        "FDM-only concept, ignored",
                    )
                )
            if obj.settings.get("enable_support") == "1":
                notes.append(
                    Note("info", f"'{obj.name}' had FDM supports enabled: for resin they must be regenerated from scratch")
                )

        notes.extend(size_notes(project))

        if len(project.plates) > 1:
            notes.append(
                Note("info", f"the project has {len(project.plates)} plates: they must be converted one by one")
            )

        settings = project.print_settings
        if settings.get("layer_height"):
            notes.append(Note("info", f"source FDM layer height: {settings['layer_height']} mm"))
        if settings.get("nozzle_diameter"):
            nozzle = settings["nozzle_diameter"]
            nozzle = nozzle[0] if isinstance(nozzle, list) and nozzle else nozzle
            notes.append(Note("info", f"source nozzle: {nozzle} mm"))
        return notes


def read_project(path: str) -> Project:
    with BambuProjectReader(path) as reader:
        return reader.read()
