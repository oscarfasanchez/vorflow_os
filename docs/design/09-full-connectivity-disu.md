# Milestone 9 — Fully unstructured DISU with full connectivity

**Status:** done (unreleased) · **Risk:** medium · **Behavior change:** new opt-in module; the MFUSG export keeps its output
**Back to** [design notes](README.md)

## Outcome

`vorflow.disu` builds fully unstructured MODFLOW grids (MF6 DISU and
MODFLOW-USG DISU) from vorflow 2D grids and FloPy-style layer arrays:

```python
from vorflow import disu

d = disu.DisuGrid(grid, top, botm, idomain).build()
print(d.report())
flopy.mf6.ModflowGwfdisu(gwf, **d.get_gridprops_mf6_disu())
```

- **Full connectivity.** Cells in different layers whose shared faces
  overlap vertically are connected (IHC=2), as proposed by Provost et al.
  (2025).
- **Pinch-outs and pass-through.** Cells with idomain -1 or below
  `min_thickness` are removed without leaving gaps in the column.
- **Different plan grids per layer.** For example, a pit refined only at
  depth with `disu.refill`, which shares the outline's vertices with the
  base grid, or `disu.splice` for grids made separately. Contacts where the
  grid changes are gap-free; `disu.split_contacts` keeps dipping bottoms
  there as given.
- **Sublayers**, everywhere or in some columns, with `disu.subdivide`.
- **Node tracking:** `nodemap`, `cellmap`, `to_nodes`/`to_layers` and
  `locate`/`column`; `idomain_layers`, `layer_view` and `vertex_grid` for 2D
  maps with FloPy.
- **Existing DISV models** go in through `DisuGrid.from_disv`.

vorflow stays a mesher. It takes numbers in FloPy's conventions and reads
no rasters. `get_gridprops_mf6_disu()` and `get_gridprops_mfusg_disu()`
return dictionaries with the keys of `flopy.utils.gridgen`'s
`get_gridprops_disu6/5`.

## Why

Layers that follow dipping or undulating surfaces leave neighbouring cells
vertically offset. With DIS/DISV connectivity, a cell only connects
sideways to the same layer index, so flow along a dipping unit is forced
through the wrong neighbours. Bardot et al. (2023) showed that even XT3D
cannot recover from this. Provost et al. (2025) showed that the cause is the
connectivity, and that XT3D becomes accurate once every vertically
overlapping neighbour is connected. That needs DISU. FloPy can build DISU
for quadtree grids (gridgen), but it has no generator for this "full"
connectivity, and it cannot connect layers that use different plan grids.

## Credits

- **The full-connectivity algorithm** follows C. D. Langevin's USGS
  `disv2disu.py`, published with:
  - Provost, A.M., Bardot, K., Langevin, C.D. and McCallum, J.L. (2025).
    *Accurate simulation of flow through dipping aquifers with MODFLOW 6
    using enhanced cell connectivity.* Groundwater 63(3), 399-408.
    <https://doi.org/10.1111/gwat.13459>.
  - Data and code: <https://doi.org/10.5066/P13BNARA>.
- **The problem** was set out by Bardot, K., Lesueur, M., Siade, A.J. and
  McCallum, J.L. (2023). *Revisiting MODFLOW's capability to model flow
  through sedimentary structures.* Groundwater 61, 663-673.
  <https://doi.org/10.1111/gwat.13273>.
- **Pass-through of pinched cells** follows LoopFlopy
  (<https://github.com/kerrybardot/loopflopy>, J. McCallum's version of
  `disv2disu.py`), part of the workflow of Bardot et al. (2025),
  *Environmental Modelling & Software* 192, 106557.

`vorflow.disu` is an independent, vectorised implementation. On offset
layers, its connection arrays equal those of the USGS reference entry for
entry (see Validation).

## How it works

**Columns and pass-through** (`_columns.py`):
- idomain 0 is no-flow and never passes flow. idomain -1, or thickness at or
  below `min_thickness`, means pass-through.
- `collapse="split"` (the default) gives a pass-through run's interval half
  to the active cell above and half to the cell below (`"up"`/`"down"` give
  it to one side). Each column stays contiguous, so thickness is conserved
  and the vertical connection lengths are exact.
- `collapse=None` keeps the gap, which is what MF6 does with DISV idomain -1
  and what LoopFlopy does.

**Interface merging** (`_connect3d.staggered_links`):
- Across each plan face, the interface elevations of both columns are merged
  in one sorted pass for all faces.
- Every piece of the shared wall that lies inside an active cell on both
  sides becomes one connection: IHC=2, HWVA = face width (MF6 computes the
  overlap itself), CL = perpendicular plan distances.
- With aligned layers this reduces exactly to DISV connectivity.

**Every piece is kept (cell-by-cell area balance):**
- Each cell's faces must add up to its own geometry, because MODFLOW
  balances flow cell by cell. Every wall piece and every plan overlap is
  therefore kept, however small; MODFLOW handles tiny connections fine.
- Merging is optional (`sliver_abs_tol`, off by default): a wall piece
  thinner than `sliver_abs_tol`, or a plan overlap smaller than
  `sliver_abs_tol**2`, goes to its neighbours. That moves face area between
  cell pairs, so `report()` warns that the result is no longer exact.
- Measured on the Voronoi models below: merging removed 3-14 % of the
  connections with no measurable gain in build or MF6 run time, and gave
  errors of 2e-4 to 6e-4 m on dipping single grids and 0.028 m at a change
  of grid (uniform gradient, XT3D).
- A consistent alternative, not built: snap near-equal layer elevations
  before building, so the cells really are aligned.

**Different grids per layer** (`DisuGrid(grids=[...])`):
- Cells with the same generator and footprint in several grids form one
  column (`splice` keeps the base cells unchanged for this), so full
  connectivity also works across a change of grid.
- Cells that end above a change of grid are joined to the cells that start
  below it by plan-overlap vertical links, which conserve area.
- Cells of different grids that touch along an edge get faces too.
- **Gap-free contacts.** Cells that overlap across a change of grid form
  groups (connected components of the overlaps). Each group shares one
  elevation, the area-weighted mean of the bottoms above (exactly those
  bottoms when they agree, e.g. nested refinement). The cells above absorb
  the change in their thickness; given tops must meet the bottoms above.
  `report()` lists the groups and the largest shift. Sampling each new top
  from the cell above its centre instead left overlaps of up to 1.22 m on
  dipping layers (0.005 m error).
- Fine cells may straddle several coarse cells: XT3D treats the vertical
  connections like hanging nodes, and the result is exact. Nesting the
  fine cells is not needed.
- **Why the contact is flat under a refinement.** Each cell has one top and
  one bottom, so a dipping layer is a staircase on each grid. Fine cells
  straddle the steps of the coarse cells above, so the groups chain
  together and the whole contact becomes one level. The coarse bottoms
  then move by about dip x half the refined width: 2.57 m at a 3 % dip on
  the notebook's 200 m pit.

**`split_contacts(grids, top, botm, idomain, wedge, min_thickness, layers)`:**
- Cuts each group that needs it flat and adds one wedge layer between the
  cut and the given contact, so the given bottoms stay where they are.
  Groups whose bottoms already agree (nested refinement) are left alone.
- `wedge="upper"`: on the grid above, from each upper bottom down to the
  lowest one; it belongs to the layer below. On the notebook pit it adds
  39 cells.
- `wedge="lower"`: on the grid below, from the highest contact down to the
  contact at each lower cell (the `top` given for that layer, else the
  bottom above the cell's centre); it belongs to the layer above and
  follows the contact on the finer grid. On the notebook pit it adds 202
  cells.
- Wedge pieces up to `min_thickness` are not made; the bottom next to them
  moves to the cut instead.
- It returns the new layer lists and `parent`, the input layer of each new
  layer; after `subdivide`, `subdivide_parent[parent]` gives the original
  layers. Both wedges reproduce a uniform gradient to 4.5e-8 m in MF6
  (XT3D), as does the flat contact.
- **When to change the grid at depth.** It suits near-vertical features,
  such as a pit or a fault zone. For subhorizontal features (dipping less
  than about 45 degrees), use the same plan grid in every layer, refined
  where needed, as in normal meshing.

**`refill(base, polygon, resolution, growth_factor, fields, outline_split, priority)`:**
- vorflow's exact barriers make every outline vertex a node on the barrier
  line, spaced by that mesh's size field. A separately made fine mesh
  spaces them differently (11.8 m against 28.6 m on the notebook pit, 4 of
  68 shared), so `splice` leaves hanging vertices. They are harmless for
  DISU, but break face orthogonality (up to 42.5 degrees).
- `refill` keeps the base cells outside and re-meshes only the inside. Each
  outline segment gets a generator pair mirrored across it, on circles at
  its end nodes (radius 0.6 x the longer adjacent segment); with no other
  generator inside a circle, the segment is exactly the face between the
  pair (Berge et al. 2019, as for the barriers). Gmsh nodes outside the
  circles fill the rest, graded from the outline spacing to `resolution`
  by `growth_factor` or by `fields`, as in `add_polygon`. The outside generator of each pair only shapes the face.
- The result shares every base outline vertex and every face is orthogonal.
- `outline_split=k` splits the outline faces on the inside: finer border
  cells, with the base vertices still shared. The extra vertices hang on
  the outside cells' faces, as in a quadtree grid, with up to 20 degrees of
  orthogonality error along the outline. XT3D handles them exactly.
- The core of `resolution` cells is inset from the outline by the distance
  at which the fields reach the outline spacing, at most half the polygon's
  inradius. If they cannot grow that far, `priority="resolution"` (the
  default) keeps the core size and warns, and `priority="growth"` raises it
  by bisection just enough. On the notebook pit with `growth_factor=1.1`,
  `"resolution"` reaches 17 m at a 28.6 m outline (neighbour area ratio
  3.0) and `"growth"` uses 23 m core cells (ratio 1.9);
  `outline_split=2` with the default growth gives a ratio of 1.8.
  `attrs["refill"]` records `core_size` and `size_at_outline`.
- Without XT3D a change of grid is not exact even with shared outline
  vertices, because the vertical connections join cells whose centres are
  offset in plan (0.066 m with `outline_split=1` and 0.10 m with 2, flat
  pit, horizontal gradient).

**Diagnostics** (`report()`):
- the offset ratio r = offset / thinner cell, which is the paper's quantity:
  same-layer neighbours stop touching at r = 1;
- a sublayer suggestion where r >= 1;
- merged-piece counts, when merging is switched on, with a warning;
- the contact groups at a change of grid and, when the contact moved
  bottoms, the largest move with a pointer to `split_contacts`;
- the XT3D advice below.

## What MODFLOW does with the geometry

Checked in the MF6 and USG-T sources on 2026-10-29.

**MF6, two-point flow:**
`C = W K1 h1 K2 h2 / (K1 h1 CL2 + K2 h2 CL1)`, where h is the overlap for
IHC=2.

**MF6, XT3D:**
- It uses the 3D vector between cell centres, ANGLDEGX for the face normal,
  and HWVA × overlap for the area. CL1/CL2 enter only as a ratio, so the plan
  perpendicular distances are exact for XT3D.
- **It ignores the vertical part of that vector unless NPF has ANGLE2**
  (`Xt3dInterface.f90`, `nozee`). Give `angle2=0.0, angle3=0.0` (MF6 needs
  both) whenever layers are offset. Without them, XT3D gave the same result
  as two-point flow in our tests.

**MF6, DISU input:**
- IDOMAIN accepts only 0 and 1 (`Disu.f90`), so it cannot carry layer or
  zone tags. The layer number goes on the IHC diagonal, which MF6 does not
  read, and the rest is in `cellmap`.

**MODFLOW-USG:**
- FAHL is an area, which USG divides by the mean thickness.
- With `ivsd` 0 or -1, USG labels every connection to a lower layer as
  vertical. `get_gridprops_mfusg_disu` therefore writes `ivsd=1` with an explicit
  `ivc`, otherwise staggered connections would take their conductance from
  VKA.
- USG also resets heads below a cell bottom, even in confined layers.

## Validation

The MODFLOW runs below are local benchmark scripts, outside the test suite
(`benchmarks/disu_validation/`, not in the repository). The tests check the
geometry those runs depend on: each cell's connections add up to its own
faces, contacts meet, and the connection arrays equal the reference.

**Reference implementation:**
- On random offset layers (up to 30×30×8 cells), ja, ihc, cl12, hwva,
  angldegx and iac equal the USGS `disv2disu.py` (`staggered=True`) entry for
  entry (every piece kept, the default).
- A test compares the connections with a brute-force overlap search,
  including pass-through and no-flow cells.

**DISV equivalence:** with `connectivity="layered"`, MF6 heads equal DISV
heads to 1e-8, including pass-through cells.

**Provost et al. dipping aquifer** (3-layer aquifer of K 1 between 3 + 3
layers of low K; analytic head on the perimeter):

| dip | layered (any formulation) | full, two-point | full + XT3D (ANGLE2/3) |
|---|---|---|---|
| 10° | flux angle ≈ 0° | 9.2°, \|q\| 0.95 | **10.00°, \|q\| 1.000** |
| 30° | flux angle ≈ 0° | 24.1° | **30.00°, \|q\| 1.000** |
| 45° | flux angle ≈ 0° | exact | **exact** |
| 60° (r = 1.7) | flux angle ≈ 0° | 49° | 66°: degraded, as the paper predicts for r > 1 |

- In MODFLOW-USG (no XT3D) at 30°, the maximum aquifer head error is 0.49 m
  with layered connectivity and 0.09 m with full connectivity.
- **Pit refined at depth** (a 4×4 grid refined 2× inside a centre square
  below layer 3): a linear head field is reproduced to 1e-11 with XT3D, for
  both flat and dipping layers. This covers hanging-node faces on the ring,
  overlay links and cross-grid faces.
- **Voronoi pit at depth** (the notebook model: 60 m cells, 12 m inside a
  200 m pit below layer 3; six layers, flat and 3 % dip; horizontal,
  vertical and combined gradients; MF6, XT3D): max head error < 1e-6 m
  (about 4e-8 m) with an independent fine grid joined by `splice`, with
  `refill` (`outline_split` 1 and 2) and with `split_contacts` (both
  wedges). Before the fixes in this note the
  same model gave up to 0.029 m:

  | setup | max error |
  |---|---|
  | merging on (old default), new tops sampled from above | 0.029 m |
  | merging off, new tops sampled (contact overlaps up to 1.22 m) | 0.0048 m |
  | merging off, gap-free contact (current) | 4e-8 m |
  | fine cells cut to nest under the coarse ones, sampled tops | 0.0045 m |
  | one grid in every layer, merging on (old default) | 6e-4 m |
  | one grid in every layer, merging off (current) | 4e-8 m |

## Limitations

- A gap-free contact moves the bottoms above a non-nested change of grid
  to one elevation per group (on the notebook pit, by up to 2.6 m with a
  3 % dip). `split_contacts` keeps them, at the cost of one wedge layer.
- Plan cells of different grids that overlap get only vertical connections.
  Where a dip is steep, give adjacent layers the same grid in that area.
- Cells match across grids only when their generator and footprint are
  identical (as after `splice`). Independently generated grids are joined
  only by plan overlaps.
- `splice` does not force nodes to match along the outline, so it leaves
  hanging nodes (exact for DISU, but not orthogonal); use `refill` when the
  fine cells can be made from the base grid.
- `refill` takes one polygon without holes per call, whose outline is made
  of base cell faces. The cells along the outline keep the base spacing
  (or a fraction of it, with `outline_split`).
- FloPy's `PlotCrossSection` misdrew a DISU model whose layers have
  different cell counts after `subdivide`. The review notebook draws its
  section from `cellmap` instead.

## Files

- New: `src/vorflow/disu.py`, `src/vorflow/_columns.py`,
  `src/vorflow/_connect3d.py`, `src/vorflow/_disu_export.py`,
  `src/vorflow/_refill.py`.
- Examples: `examples/disu_grids.ipynb`.
- Tests: `tests/test_disu_*.py`.
- Solver checks: local scripts in `benchmarks/disu_validation/` (git-ignored).
- Changed: `src/vorflow/_mfusg.py`. The plan faces are shared (`_planar_faces`
  now also returns the face normal), and the USG arrays and GSF come from the
  DISU core, with unchanged output.
