# User guide

```{include} ../README.md
:start-after: <!-- docs:installation-start -->
:end-before: <!-- docs:usage-end -->
```

## Checking inputs before meshing

`ConceptualMesh.check()` looks for input geometry that tends to produce badly
shaped or very small Voronoi cells, and reports it before you mesh. It needs
no Gmsh and takes a fraction of a second, so run it whenever the inputs
change:

```python
findings = blueprint.check(background_lc=50.0)
print(findings[["feature_id", "other_id", "kind", "severity", "ratio"]])
findings.to_file("input_findings.gpkg")  # inspect them in a GIS
```

Pass the `background_lc` you will give `MeshGenerator`. Without it, the
coarsest feature `resolution` stands in, so defects in coarse areas far from
any feature may go unreported.

Each defect is measured against the local mesh size `h` at its location: the
size the mesh will aim for there, from `background_lc` and each feature's
`resolution` and `growth_factor`. `ratio = distance / h` sets the severity.
An `error` very likely gives bad cells; a `warning` may. The result is a
GeoDataFrame with one row per defect site, errors first. `feature_id` and
`other_id` are the `zone_id`, `line_id` or `point_id` involved, `x`/`y` mark
the narrowest point, and `geometry` shows the segment, gap or polygon part.
It is empty when nothing is found.

| kind | what it means | error if ratio < | warning if ratio < | what to do |
|---|---|---|---|---|
| `short_segment` | two consecutive vertices of a polygon or line are very close | 0.1 | 0.25 | Remove the vertex, or set `simplify_tolerance` on the feature. |
| `close_vertices` | two mesh points are very close: vertices of different features, a point, or a line crossing a polygon edge just beside a vertex | 0.1 | 0.5 | Move the crossing or the vertex apart, or make them coincide. |
| `line_end_near_miss` | a line end touches nothing but lies close to another feature | 0.5 | 1.0 (inclusive) | Extend the line onto the feature, pull it well back, or raise `connectivity_tolerance` so it snaps. |
| `point_near_miss` | a point lies close to, but not on, another feature | 0.5 | 1.0 (inclusive) | Move the point onto the feature or away from it. |
| `sliver_polygon` | a polygon part, or a strip of one, is narrower than `h`; often a leftover of overlap resolution | 0.5 | 1.0 | Remove the strip, change `z_order`, or snap the polygon outlines together. Below 0.5 the strip may also take the wrong zone. |
| `narrow_gap` | a line runs close to another feature, or to itself, away from where they meet | never | 1.0 (inclusive) | Only one or two cells fit across. Accept it, refine the area, or move the line. |

Snapped features are not defects: a line end or point within
`connectivity_tolerance` of another feature is joined to it by `generate()`
and is not reported.

One defect often trips several checks; a thin strip, for example, also has
short edges. Findings within `2 * h` of each other are treated as one site,
and only the most severe is kept. Its `site_kinds` column lists every kind
found there, so a site may need more than one fix. Run `check()` again after
editing.

What is checked:

- The features as `generate()` cleans them (simplification, overlap
  resolution, snapping, clipping and line junctions), but before
  densification. The vertices densification adds are vorflow's own, so they
  are not reported.
- Embedded features only. Field-only (`embed=False`) features add no mesh
  points or edges, so they are not checked; they still set `h`.
- Barrier, straddle and quad-buffer features as the plain lines and
  polygons they are drawn from. The extra nodes and quad strips the mesher
  builds around them are not modelled. One exception: a barrier end that
  runs alongside another barrier, or along the domain outline, is not a near
  miss. The mesher gives that gap one row of cells, so it is reported as a
  narrow gap.

`check()` changes nothing: the registered features and any earlier
`generate()` result are left as they were. It does not catch every bad cell.
Many come from grading and from clipping at the domain boundary, not from the
inputs; use `vorflow.utils.calculate_mesh_quality` on the finished grid for
those.

## One-ring ghost-node corrections for DISU

After building a `vorflow.disu.DisuGrid`, request one collection of corrections:

```python
corrections = d.estimate_gnc()  # geometric, one-ring, 3D connections
print(corrections.corrected_connections, len(corrections.unsupported))
```

The geometric estimator uses only cell geometry. Both available methods search
immediate connected neighbours; they do not add numerical cells or layers.
Vertical connections and lateral elevation offsets are selected by default.
Use `connections="all"` to include in-plane offsets, for example at a splice.
`min_offset` skips offsets smaller than a specified fraction of the perpendicular
centre distance. `blocked_pairs` excludes zero-based node pairs from estimation
and interpolation support; it does not create a model hydraulic barrier (see
*Barriers and faults* below).

`records` gives each host, neighbour, ghost target, contributing nodes and
weights. `unsupported` gives selected connections where weights could not be
built. Those connections receive no GNC record and retain standard conductance;
`strict=True` raises instead. `record_count` counts ghost records, while
`corrected_connections` counts physical connections: a connection can use two
ghost records. Inspect coverage before applying the package.

For MODFLOW 6, keep XT3D disabled when applying GNC:

```python
options = corrections.get_gridprops_mf6_gnc()
if options is not None:
    flopy.mf6.ModflowGwfgnc(gwf, **options)
```

For MODFLOW-USG, explicit correction is the default export mode. FloPy stores
USG fractions as single precision; the native writer preserves double precision:

```python
options = corrections.get_gridprops_mfusg_gnc()
if options is not None:
    package = flopy.mfusg.MfUsgGnc(model, **options)
    model.write_input()
    corrections.write_mfusg_gnc(package.fn_path)
```

Empty collections return `None` from either package-export helper. Existing
DISU exports are unchanged; estimating a collection attaches no model package.

### Barriers and faults

`d.barrier_faces(trace)` finds the connections a vertical barrier cuts: every
horizontal connection whose two cell centres lie on opposite sides of the
trace. Give one line for every layer, or one per layer with `None` where a
layer has no barrier. On a grid meshed with `add_line(..., is_barrier=True)`
the cut faces lie on the line (`on_trace` is True); on other grids they are
the faces nearest it, a zig-zag that still separates the two sides. A cell
centre on the trace counts as left of it, in the direction the line is drawn.

```python
faces = d.barrier_faces(fault)                     # LineString, all layers
flopy.mf6.ModflowGwfhfb(gwf, **faces.get_gridprops_mf6_hfb(hydchr))
corrections = d.estimate_gnc(barriers=faces.pairs)  # no ghost interpolates across
```

`hydchr` is the barrier conductivity divided by its width, one value or one per
connection; `get_gridprops_mfusg_hfb` gives the arguments for
`flopy.modflow.ModflowHfb` in a USG model. With `barriers`, each ghost head is
interpolated from cells on its own side; the barrier connections themselves are
still corrected. Without it, a ghost next to the barrier can average heads
across the head jump. A connection whose only support lies across the barrier
is reported in `unsupported` and keeps standard conductance (and its HFB, if
it is a barrier connection). Next to a dipping fault this is common: in the
faults notebook `barriers=` leaves 366 connections without a ghost instead of
13, among them the thin treads along the traces, whose ghost points sit at the
fault edge of their cells.

For a dipping fault, mesh each layer at its own trace and give one trace per
layer. The fault becomes a staircase: walls on each trace, and treads where a
vertical connection lies between the traces of two consecutive layers (the
polygon joining them, closed at the fault tips). `faces.vertical` marks the
treads. HFB is the hydraulic flow barrier: MODFLOW 6 (6.7 or later),
MODFLOW-USG and USG-Transport accept it on vertical connections. USG multiplies a
vertical barrier's hydraulic characteristic by the mean thickness of the two
cells, so `get_gridprops_mfusg_hfb` divides it by that thickness (exact for
confined layers; convertible layers use the saturated thickness).

```python
traces = [fault_at(z) for z in layer_mid_depths]   # one LineString per layer, None where absent
faces = d.barrier_faces(traces)
```

`faces.factor` does not depend on K. It is |n<sub>fault</sub> · n<sub>face</sub>|
for the plane through consecutive traces, with the dip taken from the offset
between the traces and the elevation difference of their layers' mid-depths,
and the exporters multiply your
hydraulic characteristic by it. With flow normal to the fault the staircase then
has the plane's resistance; flow along the fault still sees the stair. Give the
hydraulic characteristic (>= 0): a negative HFB value (a conductance multiplier)
cannot carry the factor.

Flat-topped cells make any dipping unit or fault a staircase. Heads, flow
direction and the flux per square metre can all be right while the total water
crossing the unit is too low by about Δx·tanθ/b (column width Δx, dip θ,
vertical thickness b); only narrower columns reduce it. Keep Δx·tanθ/b at or
below about 5 % where that water matters, and check total flows against a finer
grid. `examples/disu_grids.ipynb` reads the lost face from the connections;
`examples/disu_faults_and_gnc.ipynb` builds faults and ghost-node corrections
and checks them against an exact solution.

The optional K-aware method needs current positive scalar isotropic conductivity
in active-node order (use `d.to_nodes(...)` to map layer values):

```python
k_corrections = d.estimate_gnc(k, method="k_aware")              # contrast=10
k_corrections = d.estimate_gnc(k, method="k_aware", contrast=2)  # more faces count
```

Where the conductivity ratio across a face reaches `contrast` (either
direction), the face is treated as a material contact: the neighbour's head is
weighted with Darcy's law across that face. Smaller differences are treated as
the same material and weighted geometrically, so smoothly varying fields, such
as pilot-point interpolations, behave like the geometric method. `contrast=1`
treats every difference as a contact.

This method is experimental. It is exact where contacts follow cell faces and
flow crosses them. Where cells stair-step a dipping contact, the faces are not
the real contact; flows are then in error even when heads match. Recompute the
corrections whenever K changes. It is not a verified copy of ModelMuse or
Gridgen and is not a general replacement for XT3D. Confined scalar isotropic
flow is the tested scope; anisotropy tensors and head-dependent fractions are unsupported.
