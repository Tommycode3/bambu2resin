# bambu2resin

Takes a Bambu Studio `.3mf` project and produces a `.pm3` file for Anycubic MSLA resin printers: orientation, hollowing with drainage checks, supports, light-bleed and shrinkage compensation, slicing, and a binary encoder that re-reads its own output. FDM and resin are different processes, so most of the work is re-orienting, supporting and hollowing the part, and the tool reports what a Bambu project carries (painted supports, modifiers, multi-plate layouts) that can't survive into resin. The CLI is called `travaso`, Italian for decanting.

Status: the `.pm3` files are structurally consistent and round-trip through the reader, but none has been printed on real hardware yet. Read [Before printing](#before-printing) before a long print.

## Usage

Python 3.11+.

```bash
git clone https://github.com/Tommycode3/bambu2resin.git && cd bambu2resin
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
travaso printers                      # lists the supported printer
```

```bash
travaso inspect project.3mf           # objects, plates, what will not survive in resin
travaso check project.3mf             # mesh health + repair (non-manifold edges reported)
travaso orient project.3mf --priority finish --alternatives 3
travaso calibrate exposure  --out tower.pm3      # once per resin
travaso calibrate tolerance --out plate.pm3
travaso build project.3mf --orient --out part.pm3
```

With everything on:

```bash
travaso build project.3mf \
  --orient --priority reliability \
  --hollow --wall 2.0 \
  --bleed 0.045 \
  --exposure 2.7 --bottom-exposure 30 --bottom-layers 6 \
  --layer-height 0.05 \
  --out part.pm3
```

`inspect`, `check`, `orient`, `slice` (stats and PNG previews) and `build` also take `.stl`, `.obj` and `.ply`, and detect the format from the file content, since binary STLs renamed `.3mf` are common. Every command has `--help`.

| option | default | effect |
|---|---|---|
| `--layer-height` | 0.05 | layer thickness, mm |
| `--orient` | off | apply the recommended pose |
| `--hollow --wall N` | off, 2.0 | hollow with an N mm wall and open drain holes |
| `--bleed N` | 0.0 | per-side growth from light bleed (from the tolerance plate) |
| `--shrink-xy`, `--shrink-z` | 0.0 | resin shrinkage, percent |
| `--exposure`, `--bottom-exposure`, `--bottom-layers` | 2.5, 30, 6 | exposure times (from the tower) |
| `--no-supports` | supports on | no supports and raft |
| `--spacing`, `--tip-diameter` | 4.0, 0.4 | support density and contact size, mm |

Tests: `pip install -e ".[dev]" && pytest -q`. They cover the parts where mistakes are invisible by eye (normal orientation, fill rules, edge offset, binary layout, island detection, hollowing, drainage), and several are regression tests for real defects, named in the docstring.

## How it works

The pipeline is: load and read the Bambu project, repair the mesh, orient, pre-scale for shrinkage, place on the plate, hollow, solve supports, slice, run-length encode, write `.pm3`, and read it back.

### Orientation

Each MSLA layer is peeled off the vat film with a force that grows with its area. Candidate up-vectors $\mathbf{u}$ are a 256-point Fibonacci sphere plus $\pm$ the normals of the 24 largest planar facets, deduplicated within about 1°, and each is scored from face normals $\mathbf{n}_f$ and areas $A_f$ without slicing:

$$
S(\mathbf{u}) = w_c \frac{X(\mathbf{u})}{\max_{\mathbf{u}'} X}
+ w_s \frac{\sum_f A_f \max(0, -\mathbf{n}_f\cdot\mathbf{u})}{\sum_f A_f}
+ w_h \frac{\sum_f A_f\, |\mathbf{n}_f\cdot\mathbf{u}|^{6}}{\sum_f A_f}
+ w_z \frac{H(\mathbf{u})}{\lVert \text{extents} \rVert}
$$

$X(\mathbf{u})$ is the area of the convex hull's shadow, an upper bound on the largest cross-section that orders poses the same way and scores hundreds of them in milliseconds (concave parts are over-estimated). The second term is the area that needs support, the third isolates near-horizontal faces as a peel-force proxy, and $H$ is print height. Poses that don't fit sort last and the lowest score wins. Presets $(w_c, w_s, w_h, w_z)$: `reliability` (1.5, 1.0, 0.5, 0.4, default), `speed` (0.8, 0.6, 0.4, 2.0), `finish` (1.0, 2.5, 0.5, 0.3). The normals-by-directions product is computed in blocks capped at 200 MB.

### Slicing

Each layer is a plane-mesh intersection rasterised on the panel grid (Photon M3: 4096 x 2560 pixels on 163.84 x 102.4 mm, exactly 40 µm square, X mirrored because the panel is seen from below). The fill rule is positive winding:

$$
\text{pixel is solid} \iff w(\mathbf{p}) = \sum_{\text{contours}} \operatorname{wind}(\mathbf{p}) > 0
$$

Even-odd would empty the overlap of two interpenetrating bodies, and non-zero would fill an inverted primitive ($w=-1$) where it sticks out. Positive winding unites solids and subtracts inverted primitives, so cavities, drain holes and supports compose with no mesh booleans, provided every primitive is oriented correctly. It also tolerates locally non-watertight meshes. Horizontal coverage is exact from the span endpoints and each row is sampled with `--antialias` sub-rows (default 4), so cost grows linearly with sub-sampling. Only each layer's bounding window is rasterised.

### Edge compensation

Light bleeds past lit pixels, so outer dimensions grow and holes shrink by $b$ per side. A signed distance $d$ in pixels is rebuilt from inside and outside EDTs, replaced on edge pixels by the fractional coverage $c-\tfrac12$, and offset per layer:

$$
c'_{ij} = \operatorname{clip}\!\left(d_{ij} + \frac{\delta_k}{p} + \tfrac12,\ 0,\ 1\right),
\qquad
\delta_k = -b - e\left(1 - \frac{k}{K}\right)\,[k < K]
$$

with pixel pitch $p$ and elephant-foot growth $e$ tapered over the first $K$ layers. Shrinkage $r$ is handled by pre-scaling by $1/(1-r)$, not $1+r$, since the part has to come back to nominal after shrinking; on a 3 % resin the difference shows on long parts.

### Hollowing

The part is voxelised (0.3 mm in `build`, capped at 12 M voxels) and the cavity is $\Omega = \{\mathbf{x} : D(\mathbf{x}) > t + h/2\}$, with $D$ the EDT distance to the surface, $t$ the wall and $h$ the pitch. The $h/2$ cancels two half-pitch biases (EDT between voxel centres, marching cubes half a pitch outward), and the achieved wall $\min_\Omega D - h$ is reported; accuracy is about one voxel pitch. Components under 500 mm³ are dropped, and if hollowing saves less than 10 % of the volume the part stays solid. The cavity is extracted with marching cubes and inverted so the winding rule subtracts it. Each cavity gets two 3 mm drain holes, at its lowest point for resin and its highest for air, and the air region is then flood-filled from the bounding box; any cavity it doesn't reach is reported, because trapped resin and vat suction are worse than a solid part.

### Supports

An island is material in layer $k$ with nothing under it, found on the raster layers by keeping components of layer $k$ that don't touch layer $k-1$ dilated by one pixel. That measures what the printer will cure, which geometry alone doesn't tell you. Supports (tip 0.4 mm penetrating 0.15 mm, pillar 1.0 mm, raft 1.2 mm) are seeded at the lowest points and on surfaces with $n_z \le -\cos 45^\circ$, about one per $s^2$ of projected area and no two closer than $s$; pillars land on the raft, or on the model when a downward ray hits it. Lifting the part turns its bottom into an overhang, so the solver loops (build, slice, find islands, support their centroids) up to `--rounds` (default 4) until no islands remain. Each round re-slices the whole part, tens of seconds on large parts.

### .pm3 and PW0

The version 516 layout comes from UVtools' reverse-engineered implementation ([NOTICE.md](NOTICE.md)): FileMark with table addresses, header, RGB565 preview, grey table, a 32-byte record per layer, extra/machine tables, then layer data, with the FileMark and layer table patched at the end once addresses are known. Pixels are quantised to 4 bits; runs of pure black or white carry a 12-bit count in two bytes, greys a 4-bit count in one byte, and code $c$ decodes to $(c \ll 4) \mid c$ so 0 and 255 are exact. `read_pm3` then checks signatures, addresses and records, and can also inspect files from Anycubic Photon Workshop. Machine data comes from UVtools/PrusaSlicer profiles because spec-sheet pixel sizes don't match panel geometry.

## Before printing

Exposure and light bleed can't be derived: they depend on resin, colour, temperature and LCD ageing. The defaults are plausible starting values. Two test prints measure them in an afternoon.

`travaso calibrate exposure --out tower.pm3 --from 1.5 --to 6.0 --steps 8` prints 8 stacked bands at different exposures in one print, since the format stores exposure per layer (about 1 h 40 min, 6.6 ml by default). Each band is a little narrower than the one below so they can be counted, and has five progressively narrower slots. Take the longest time at which all slots are open and edges sharp, and pass it as `--exposure`.

`travaso calibrate tolerance --out plate.pm3 --exposure 2.7` prints holes and pins of known diameter. If holes come out smaller and pins larger by the same amount, half that amount is the bleed (`--bleed`). If both shift the same way it's a scale error: use `--shrink-xy`. Full instructions are written next to the `.pm3` as a text file.

## Limitations

- No file printed yet; output is verified only against its own reader.
- Default exposure times are not calibrated.
- No automatic arrangement: several individually oriented objects can overlap.
- A residual lit area of 0.28 % of cavity area, two spots of about 1 mm, was measured inside cavities. Not enough to ruin a print, not fully explained.
- One printer (Anycubic Photon M3). Profiles live in `reference/`.

## License

AGPL-3.0-or-later. The `.pm3` format was derived from [UVtools](https://github.com/sn4k3/UVtools), which is AGPL-licensed; the credit for decoding the format is theirs. See [NOTICE.md](NOTICE.md).
