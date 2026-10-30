"""Writer for the Anycubic .pm3 container (Photon M3, version 516).

The format is not documented by Anycubic. The layout implemented here comes
from the UVtools code (`reference/AnycubicFile.cs`), which derived it through
reverse engineering and is the only reliable public source.

File layout:

    0    FileMark: signature, version, and the addresses of all tables
    52   HEADER    print parameters
         PREVIEW   RGB565 thumbnail
         (grey table, without a header)
         LAYERDEF  one 32-byte record per layer, written last
         EXTRA     two-stage lift speeds
         MACHINE   machine identity
         ...       the layers' RLE data, one after another

The addresses are only known along the way, so FileMark and LAYERDEF are
rewritten at the end by seeking back in the file.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from .pw0 import decode_pw0, encode_pw0

VERSION_516 = 516
MARK_SIZE = 12
TABLE_BASE_LENGTH = MARK_SIZE + 4

HEADER_TABLE_LENGTH = 84  # v516; 80 in v515, 92 in v517, 96 in v518
MACHINE_TABLE_LENGTH = 156  # v516; 224 in v518
EXTRA_TABLE_LENGTH = 56
LAYER_RECORD_SIZE = 32
NUMBER_OF_TABLES = 8  # v516

PREVIEW_WIDTH = 224
PREVIEW_HEIGHT = 168

# Typical density of a photopolymer resin, used to estimate the weight
RESIN_DENSITY_G_ML = 1.1


@dataclass
class PrintParameters:
    """Print parameters.

    Exposure times depend on the resin and must be calibrated: the ones here
    are starting values for a 4K machine at 50 um, not a verified profile.
    """

    layer_height_mm: float = 0.05
    exposure_s: float = 2.5
    bottom_exposure_s: float = 30.0
    bottom_layers: int = 6
    wait_before_cure_s: float = 2.5

    lift_height_mm: float = 8.0
    lift_speed_mm_s: float = 2.0  # 120 mm/min
    retract_speed_mm_s: float = 2.0
    bottom_lift_height_mm: float = 8.0
    bottom_lift_speed_mm_s: float = 2.0
    bottom_retract_speed_mm_s: float = 2.0

    antialiasing: int = 16
    resin_density_g_ml: float = RESIN_DENSITY_G_ML

    def __post_init__(self) -> None:
        if self.bottom_layers < 1:
            raise ValueError("at least one bottom layer is needed to stick to the build plate")
        if self.exposure_s <= 0 or self.bottom_exposure_s <= 0:
            raise ValueError("exposure times must be positive")
        if self.antialiasing not in (1, 2, 4, 8, 16):
            raise ValueError("allowed antialiasing: 1, 2, 4, 8, 16")


@dataclass
class WriteResult:
    path: str
    layers: int
    bytes_written: int
    rle_bytes: int
    volume_ml: float
    print_time_s: int
    assumptions: list[str] = field(default_factory=list)


def _terminated_string(text: str, length: int) -> bytes:
    raw = text.encode("ascii", errors="replace")[: length - 1]
    return raw + b"\x00" * (length - len(raw))


def _named_table(name: str, payload: bytes, include_base: bool) -> bytes:
    """Table with a header: 12-byte name plus the length.

    Some tables declare the length of their fields only, others include the
    header as well. Getting this distinction wrong shifts every subsequent
    address.
    """
    length = len(payload) + (TABLE_BASE_LENGTH if include_base else 0)
    return _terminated_string(name, MARK_SIZE) + struct.pack("<I", length) + payload


def preview_rgb565(image: np.ndarray) -> bytes:
    """Convert an RGB thumbnail to little-endian RGB565."""
    rgb = np.ascontiguousarray(image, dtype=np.uint8)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    r = rgb[:, :, 0].astype(np.uint16)
    g = rgb[:, :, 1].astype(np.uint16)
    b = rgb[:, :, 2].astype(np.uint16)
    packed = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    return packed.astype("<u2").tobytes()


def estimate_print_time_s(layer_count: int, params: PrintParameters) -> int:
    """Estimate the print time: exposure, wait and lift travel."""
    bottom = min(params.bottom_layers, layer_count)
    normal = max(0, layer_count - bottom)

    exposure = bottom * params.bottom_exposure_s + normal * params.exposure_s
    waiting = layer_count * params.wait_before_cure_s
    travel = bottom * (
        params.bottom_lift_height_mm / max(params.bottom_lift_speed_mm_s, 1e-6)
        + params.bottom_lift_height_mm / max(params.bottom_retract_speed_mm_s, 1e-6)
    ) + normal * (
        params.lift_height_mm / max(params.lift_speed_mm_s, 1e-6)
        + params.lift_height_mm / max(params.retract_speed_mm_s, 1e-6)
    )
    return int(round(exposure + waiting + travel))


def write_pm3(
    path: str,
    layers: Iterable[np.ndarray],
    printer,
    params: PrintParameters | None = None,
    *,
    volume_mm3: float | Callable[[], float] = 0.0,
    preview: np.ndarray | None = None,
    exposure_for_layer: Callable[[int], float] | None = None,
) -> WriteResult:
    """Write a .pm3 file from the layer images.

    `volume_mm3` also accepts a zero-argument function: the volume is only
    known after going through all the layers, and a caller passing a
    generator cannot know it at call time.

    `exposure_for_layer` allows the exposure to vary along the height: the
    format stores it in every layer record, and that is how a test print with
    several different times is made in a single run. When it is given, the
    file must be flagged with the per-layer settings indicator, otherwise the
    printer uses the global value and ignores the individual records.
    """
    params = params or PrintParameters()
    if printer.file_extension != "pm3" or printer.file_version != VERSION_516:
        raise ValueError(
            f"{printer.model} usa .{printer.file_extension} v{printer.file_version}, "
            "this writer produces .pm3 v516"
        )

    # Layers are compressed as they come, but the LAYERDEF table must be sized
    # before writing the data: so the already-compressed blocks are kept, which
    # weigh as much as the final file rather than as much as the images.
    blobs: list[bytes] = []
    nonzero: list[int] = []
    for image in layers:
        if image.shape != (printer.resolution_y, printer.resolution_x):
            raise ValueError(
                f"layer {len(blobs)} has size {image.shape}, expected "
                f"({printer.resolution_y}, {printer.resolution_x})"
            )
        blobs.append(encode_pw0(image))
        nonzero.append(int(np.count_nonzero(image)))

    count = len(blobs)
    if count == 0:
        raise ValueError("no layers to write")

    resolved_volume = volume_mm3() if callable(volume_mm3) else volume_mm3
    volume_ml = resolved_volume / 1000.0
    print_time = estimate_print_time_s(count, params)

    header = _named_table(
        "HEADER",
        struct.pack(
            # 10 float, 3 uint, 2 float, the currency symbol, 5 uint.
            # The order is the format's: weight and price come BEFORE the
            # per-layer settings indicator, not after.
            "<10f3I2f4s5I",
            max(
                printer.display_width_mm / printer.resolution_x,
                printer.display_height_mm / printer.resolution_y,
            )
            * 1000.0,  # PixelSizeUm
            params.layer_height_mm,
            params.exposure_s,
            params.wait_before_cure_s,
            params.bottom_exposure_s,
            float(params.bottom_layers),
            params.lift_height_mm,
            params.lift_speed_mm_s,
            params.retract_speed_mm_s,
            volume_ml,
            params.antialiasing,
            printer.resolution_x,
            printer.resolution_y,
            volume_ml * params.resin_density_g_ml,  # WeightG
            0.0,  # Price
            b"\x00" * 4,  # PriceCurrencySymbol
            1 if exposure_for_layer is not None else 0,  # PerLayerSettings
            print_time,
            0,  # TransitionLayerCount
            0,  # TransitionLayerType
            0,  # AdvancedMode (present because TableLength >= 84)
        ),
        include_base=False,
    )
    assert len(header) == TABLE_BASE_LENGTH + HEADER_TABLE_LENGTH, len(header)

    if preview is None:
        preview_pixels = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
    else:
        preview_pixels = preview
    preview_data = preview_rgb565(preview_pixels)
    preview_table = (
        _terminated_string("PREVIEW", MARK_SIZE)
        + struct.pack("<I", TABLE_BASE_LENGTH + 12 + len(preview_data))
        + struct.pack("<I", PREVIEW_WIDTH)
        + _terminated_string("x", 4)
        + struct.pack("<I", PREVIEW_HEIGHT)
    )

    # Grey table: unlike the others, it has no header
    step = 255.0 / max(1, params.antialiasing)
    grey = bytes(min(int((i + 1) * step), 255) for i in range(16))
    colour_table = struct.pack("<II", 0, 16) + grey + struct.pack("<I", 0)

    extra = _named_table(
        "EXTRA",
        struct.pack(
            "<I6fI6f",
            2,  # BottomLiftCount
            params.bottom_lift_height_mm,
            params.bottom_lift_speed_mm_s,
            params.bottom_retract_speed_mm_s,
            0.0,  # BottomLiftHeight2
            params.bottom_lift_speed_mm_s,
            params.bottom_retract_speed_mm_s,
            2,  # NormalLiftCount
            params.lift_height_mm,
            params.lift_speed_mm_s,
            params.retract_speed_mm_s,
            0.0,  # LiftHeight2
            params.lift_speed_mm_s,
            params.retract_speed_mm_s,
        ),
        include_base=False,
    )
    assert len(extra) == TABLE_BASE_LENGTH + EXTRA_TABLE_LENGTH, len(extra)

    machine = _named_table(
        "MACHINE",
        _terminated_string(printer.model, 96)
        + _terminated_string("pw0Img", 16)
        + struct.pack(
            "<IIfffII",
            16,  # MaxAntialiasingLevel
            1,  # PropertyFields (1 up to v516)
            printer.display_width_mm,
            printer.display_height_mm,
            printer.max_height_mm,
            VERSION_516,  # MaxFileVersion for .pm3
            6506241,  # MachineBackground
        ),
        include_base=True,
    )
    assert len(machine) == MACHINE_TABLE_LENGTH, len(machine)

    layerdef_table_length = 4 + LAYER_RECORD_SIZE * count

    with open(path, "wb") as handle:
        header_address = 52  # size of the FileMark in v516
        handle.seek(header_address)
        handle.write(header)

        preview_address = handle.tell()
        handle.write(preview_table)
        preview_data_offset = handle.tell()
        handle.write(preview_data)

        colour_table_address = handle.tell()
        handle.write(colour_table)

        layerdef_address = handle.tell()
        # The EXTRA and MACHINE tables sit right after the space reserved for
        # LAYERDEF, which is filled in at the end.
        handle.seek(layerdef_address + TABLE_BASE_LENGTH + layerdef_table_length)

        extra_address = handle.tell()
        handle.write(extra)
        machine_address = handle.tell()
        handle.write(machine)

        layer_image_address = handle.tell()
        records = []
        rle_bytes = 0
        for index, blob in enumerate(blobs):
            address = handle.tell()
            handle.write(blob)
            rle_bytes += len(blob)
            bottom = index < params.bottom_layers
            records.append(
                struct.pack(
                    "<IIffffII",
                    address,
                    len(blob),
                    params.bottom_lift_height_mm if bottom else params.lift_height_mm,
                    params.bottom_lift_speed_mm_s if bottom else params.lift_speed_mm_s,
                    (
                        exposure_for_layer(index)
                        if exposure_for_layer is not None
                        else (params.bottom_exposure_s if bottom else params.exposure_s)
                    ),
                    params.layer_height_mm,
                    nonzero[index],
                    0,
                )
            )

        handle.seek(layerdef_address)
        handle.write(
            _named_table(
                "LAYERDEF",
                struct.pack("<I", count) + b"".join(records),
                include_base=False,
            )
        )

        handle.seek(preview_data_offset)
        handle.write(preview_data)

        handle.seek(0)
        handle.write(
            _terminated_string("ANYCUBIC", MARK_SIZE)
            + struct.pack(
                "<10I",
                VERSION_516,
                NUMBER_OF_TABLES,
                header_address,
                0,  # SoftwareAddress: only from v517
                preview_address,
                colour_table_address,
                layerdef_address,
                extra_address,
                machine_address,
                layer_image_address,
            )
        )
        handle.seek(0, 2)
        size = handle.tell()

    return WriteResult(
        path=path,
        layers=count,
        bytes_written=size,
        rle_bytes=rle_bytes,
        volume_ml=volume_ml,
        print_time_s=print_time,
        assumptions=[
            "layout derived from the UVtools code, not from Anycubic documentation",
            "exposure times not calibrated: they depend on the resin",
        ],
    )


@dataclass
class Pm3Summary:
    version: int
    tables: dict[str, int]
    machine_name: str
    layer_image_format: str
    resolution: tuple[int, int]
    display_mm: tuple[float, float, float]
    layer_height_mm: float
    exposure_s: float
    bottom_exposure_s: float
    bottom_layers: int
    antialiasing: int
    layer_count: int
    volume_ml: float
    print_time_s: int
    total_rle_bytes: int
    problems: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.problems


def patch_preview(path: str, image: np.ndarray) -> bool:
    """Replace the thumbnail in an already written file.

    The preview sits at the start of the file but can only be composed after
    seeing all the layers, so an empty slot is written first and filled here.
    """
    with open(path, "r+b") as handle:
        handle.seek(MARK_SIZE)
        fields = struct.unpack("<10I", handle.read(40))
        preview_address = fields[4]
        if preview_address == 0:
            return False
        data = preview_rgb565(image)
        if len(data) != PREVIEW_WIDTH * PREVIEW_HEIGHT * 2:
            raise ValueError(f"the thumbnail must be {PREVIEW_WIDTH}x{PREVIEW_HEIGHT}")
        handle.seek(preview_address + TABLE_BASE_LENGTH + 12)
        handle.write(data)
    return True


def read_pm3(path: str) -> Pm3Summary:
    """Read back a .pm3 and check its internal consistency.

    Used to check what was just written, and to inspect files produced by
    Photon Workshop for comparison.
    """
    with open(path, "rb") as handle:
        data = handle.read()

    problems: list[str] = []
    if data[:8] != b"ANYCUBIC":
        raise ValueError(f"{path} does not start with the ANYCUBIC signature")

    fields = struct.unpack_from("<10I", data, MARK_SIZE)
    version = fields[0]
    if version != VERSION_516:
        problems.append(f"version {version}, this reader knows 516")

    names = (
        "header",
        "software",
        "preview",
        "colour_table",
        "layerdef",
        "extra",
        "machine",
        "layer_image",
    )
    tables = dict(zip(names, fields[2:]))

    def table_name_at(address: int) -> str:
        return data[address : address + MARK_SIZE].split(b"\x00")[0].decode("ascii", "replace")

    for expected, key in (("HEADER", "header"), ("LAYERDEF", "layerdef"), ("EXTRA", "extra"), ("MACHINE", "machine")):
        address = tables[key]
        if address == 0:
            problems.append(f"{expected} table address is null")
            continue
        found = table_name_at(address)
        if found != expected:
            problems.append(f"found '{found}' at the {expected} address instead")

    h = tables["header"] + TABLE_BASE_LENGTH
    (
        _pixel_um,
        layer_height,
        exposure,
        _wait,
        bottom_exposure,
        bottom_layers,
    ) = struct.unpack_from("<6f", data, h)
    antialiasing, res_x, res_y = struct.unpack_from("<3I", data, h + 40)
    volume_ml = struct.unpack_from("<f", data, h + 36)[0]
    # offset 64 is PerLayerSettings; the print time comes right after
    print_time = struct.unpack_from("<I", data, h + 68)[0]

    m = tables["machine"] + TABLE_BASE_LENGTH
    machine_name = data[m : m + 96].split(b"\x00")[0].decode("ascii", "replace")
    image_format = data[m + 96 : m + 112].split(b"\x00")[0].decode("ascii", "replace")
    display_w, display_h, machine_z = struct.unpack_from("<3f", data, m + 120)

    ld = tables["layerdef"] + TABLE_BASE_LENGTH
    layer_count = struct.unpack_from("<I", data, ld)[0]

    total_rle = 0
    for index in range(layer_count):
        address, length = struct.unpack_from("<II", data, ld + 4 + index * LAYER_RECORD_SIZE)
        total_rle += length
        if address + length > len(data):
            problems.append(f"layer {index} points past the end of the file")
            break
        if address < tables["layer_image"]:
            problems.append(f"layer {index} points before the data area")
            break

    if layer_count:
        address, length = struct.unpack_from("<II", data, ld + 4)
        decoded = decode_pw0(data[address : address + length], res_x, res_y)
        if decoded.shape != (res_y, res_x):
            problems.append("the first layer does not decode at the declared resolution")

    return Pm3Summary(
        version=version,
        tables=tables,
        machine_name=machine_name,
        layer_image_format=image_format,
        resolution=(res_x, res_y),
        display_mm=(display_w, display_h, machine_z),
        layer_height_mm=layer_height,
        exposure_s=exposure,
        bottom_exposure_s=bottom_exposure,
        bottom_layers=int(bottom_layers),
        antialiasing=antialiasing,
        layer_count=layer_count,
        volume_ml=volume_ml,
        print_time_s=print_time,
        total_rle_bytes=total_rle,
        problems=problems,
    )
