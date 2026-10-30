"""Registry of target printers.

The values are NOT taken from marketing material: they come from the UVtools
machine profiles (`reference/photon_m3.ini`), which are the most reliable
publicly available reference for Anycubic printers. Every field that ends up in
the print file must be traceable to a verifiable source, which is why the
registry carries its provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Printer:
    key: str
    vendor: str
    model: str

    # LCD
    resolution_x: int
    resolution_y: int
    display_width_mm: float
    display_height_mm: float

    # Volume
    max_height_mm: float

    # The panel is seen from below through the vat: without the correct
    # mirroring the part comes out mirrored and joints no longer fit.
    mirror_x: bool
    mirror_y: bool

    # Print file container
    file_extension: str
    file_version: int

    # Process compensations
    elephant_foot_mm: float

    # Default machine parameters (lift speeds, PWM, wait times)
    machine_defaults: dict = field(default_factory=dict)

    source: str = ""

    @property
    def pixel_width_um(self) -> float:
        return self.display_width_mm / self.resolution_x * 1000.0

    @property
    def pixel_height_um(self) -> float:
        return self.display_height_mm / self.resolution_y * 1000.0

    @property
    def build_volume_mm(self) -> tuple[float, float, float]:
        return (self.display_width_mm, self.display_height_mm, self.max_height_mm)

    def describe(self) -> str:
        return (
            f"{self.vendor} {self.model}\n"
            f"  resolution    {self.resolution_x} x {self.resolution_y}\n"
            f"  display       {self.display_width_mm} x {self.display_height_mm} mm\n"
            f"  pixel         {self.pixel_width_um:.2f} x {self.pixel_height_um:.2f} um\n"
            f"  volume        {self.display_width_mm} x {self.display_height_mm} x {self.max_height_mm} mm\n"
            f"  mirror        X={'yes' if self.mirror_x else 'no'} Y={'yes' if self.mirror_y else 'no'}\n"
            f"  file          .{self.file_extension} (version {self.file_version})"
        )


PHOTON_M3 = Printer(
    key="photon_m3",
    vendor="Anycubic",
    model="Photon M3",
    resolution_x=4096,
    resolution_y=2560,
    display_width_mm=163.84,
    display_height_mm=102.4,
    max_height_mm=180.0,
    mirror_x=True,
    mirror_y=False,
    file_extension="pm3",
    file_version=516,
    elephant_foot_mm=0.2,
    machine_defaults={
        "wait_time_before_cure_s": 2.5,
        "bottom_lift_height_mm": 8.0,
        "lift_height_mm": 8.0,
        "bottom_lift_speed_mm_min": 120.0,
        "lift_speed_mm_min": 120.0,
        "retract_speed_mm_min": 120.0,
        "bottom_light_pwm": 255,
        "light_pwm": 255,
    },
    source="UVtools PrusaSlicer/printer/Anycubic Photon M3.ini (FILEFORMAT_PM3, FILEVERSION_516)",
)


REGISTRY: dict[str, Printer] = {p.key: p for p in (PHOTON_M3,)}


def get_printer(key: str) -> Printer:
    try:
        return REGISTRY[key]
    except KeyError:
        disponibili = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown printer '{key}'; available: {disponibili}") from None
