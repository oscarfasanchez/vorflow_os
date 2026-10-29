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
flopy.mf6.ModflowGwfdisu(gwf, **d.get_gridprops_disu6())
```

- **Full connectivity.** Cells in different layers whose shared faces
  overlap vertically are connected (IHC=2), as proposed by Provost et al.
  (2025).
- **Pinch-outs and pass-through.** Cells with idomain -1 or below
  `min_thickness` are removed without leaving gaps in the column.
- **Different plan grids per layer.** For example, a pit refined only at
  depth with `disu.splice`.
- **Sublayers**, everywhere or in some columns, with `disu.subdivide`.
- **Node tracking:** `nodemap`, `cellmap`, `to_nodes`/`to_layers` and
  `locate`/`column`.
- **Existing DISV models** go in through `DisuGrid.from_disv`.

vorflow stays a mesher. It takes numbers in FloPy's conventions, returns the
`gridprops` dictionaries of `flopy.utils.gridgen` (`get_gridprops_disu6/5`),
and reads no rasters.

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

**Slivers:**
- An offset is noise, and is snapped, only when it is below `sliver_tol`
  (5 %) of the thinner cell and also flatter than `sliver_dip` (1 degree)
  between the two centres.
- A run of slivers goes half to the piece above and half to the piece below.
  Face area is kept, and HWVA is scaled because MF6 recomputes the overlap
  from the cell elevations.
- Both conditions matter. The thickness rule alone merged real 11-degree
  offsets between 0.5 m cells and broke XT3D exactness.

**Different grids per layer** (`DisuGrid(grids=[...])`):
- Cells with the same generator and footprint in several grids form one
  column (`splice` keeps the base cells unchanged for this), so full
  connectivity also works across a change of grid.
- Cells that end above a change of grid are joined to the cells that start
  below it by plan-overlap vertical links, which conserve area.
- Cells of different grids that touch along an edge get faces too.
- Tops of new cells come from the bottom of the cell above them. Any
  remaining elevation overlap sets MF6 `vertical_offset_tolerance`.

**Diagnostics** (`report()`):
- the offset ratio r = offset / thinner cell, which is the paper's quantity:
  same-layer neighbours stop touching at r = 1;
- a sublayer suggestion where r >= 1;
- sliver counts;
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
  vertical. `get_gridprops_disu5` therefore writes `ivsd=1` with an explicit
  `ivc`, otherwise staggered connections would take their conductance from
  VKA.
- USG also resets heads below a cell bottom, even in confined layers.

## Validation

**Reference implementation:**
- On random offset layers (up to 30×30×8 cells), ja, ihc, cl12, hwva,
  angldegx and iac equal the USGS `disv2disu.py` (`staggered=True`) entry for
  entry, with slivers off.
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

## Limitations

- Where fine cells straddle the edges of the coarse cells above a change of
  grid (any non-nested Voronoi refinement), the interface cannot be flat for
  both grids. Each fine top comes from the coarse cell above its centre, and
  the mismatch is reported (and accepted through MF6
  `vertical_offset_tolerance`).
  - On the notebook's Voronoi pit model (3 % dip), the mismatch is up to
    1.2 m on 10-20 m layers.
  - A linear head field is then reproduced to 0.005 m over a ~10 m range,
    against 1e-11 for nested grids.
- Plan cells of different grids that overlap get only vertical connections.
  Where a dip is steep, give adjacent layers the same grid in that area.
- Cells match across grids only when their generator and footprint are
  identical (as after `splice`). Independently generated grids are joined
  only by plan overlaps.
- `splice` does not force nodes to match along the outline, so it leaves
  hanging nodes. A node-matched refill built on the hybrid subregion code is
  a possible follow-up, if the ring accuracy calls for it.

## Files

- New: `src/vorflow/disu.py`, `src/vorflow/_columns.py`,
  `src/vorflow/_connect3d.py`, `src/vorflow/_disu_export.py`.
- Tests: `tests/test_disu_*.py`.
- Changed: `src/vorflow/_mfusg.py`. The plan faces are shared (`_planar_faces`
  now also returns the face normal), and the USG arrays and GSF come from the
  DISU core, with unchanged output.
