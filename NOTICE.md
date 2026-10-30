# Provenance and credits

## The Anycubic .pm3 format

Anycubic does not document the format of its print files. The binary layout
implemented in `travaso/formats/pm3.py` and `travaso/formats/pw0.py` was derived
from the source code of **[UVtools](https://github.com/sn4k3/UVtools)** by
Tiago Conceição, who reverse-engineered it and is the only reliable public source
on the subject. A copy of the file that was consulted is kept in
`reference/AnycubicFile.cs`, unmodified, under its original license
(GNU AGPL-3.0).

The RGB565 preview encoding comes from `UVtools.Core/FileFormats/FileFormat.cs`
in the same repository. Only a few lines of it were needed, so it is not
included here, to avoid redistributing three hundred kilobytes of someone
else's code for a single function.

UVtools is distributed under the **GNU AGPL-3.0**. travaso adopts the same
license (AGPL-3.0-or-later): it keeps the provenance explicit and makes
redistribution unambiguous.

If you use travaso, the credit for decoding the format belongs to UVtools.

## Photon M3 machine data

Resolution, display size, mirroring and format version come from a printer
profile (`reference/photon_m3.ini`: the PrusaSlicer printer profile distributed
with UVtools, whose `printer_notes` carry UVtools' `FILEFORMAT_PM3` /
`FILEVERSION_516` keywords), not from product sheets:
163.84 x 102.4 mm over 4096 x 2560 pixels gives exactly square 40 µm pixels,
whereas the "23 µm" figure quoted on some sites is inconsistent with the panel
geometry.

## Dependencies

All permissively licensed and compatible with AGPL-3.0:

| library | license | used for |
|---|---|---|
| numpy | BSD-3-Clause | all numerical computation |
| scipy | BSD-3-Clause | distance transforms, connected-component labelling, KD-tree, convex hulls |
| trimesh | MIT | geometric primitives, mesh repair, ray casting |
| shapely | BSD-3-Clause | polygons for hole closing and the raft |
| scikit-image | BSD-3-Clause | marching cubes for the cavity surface |
| Pillow | MIT-CMU | PNG previews |
| manifold3d | Apache-2.0 | declared dependency, not used yet |
