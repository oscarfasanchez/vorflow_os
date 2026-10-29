"""Fully unstructured MODFLOW grids (DISU) from vorflow meshes.

The workflow follows FloPy conventions: give the 2D grid and the usual layer
arrays (``top``, ``botm``, ``idomain``), call :meth:`DisuGrid.build`, then
unpack :meth:`DisuGrid.get_gridprops_disu6` into ``flopy.mf6.ModflowGwfdisu``,
as with ``flopy.utils.gridgen.Gridgen``. vorflow reads no rasters and writes
no model files.

Full connectivity (vertically staggered horizontal connections between cells
in different layers) follows the algorithm of C. D. Langevin's USGS
``disv2disu.py``, published with

    Provost, A.M., Bardot, K., Langevin, C.D. and McCallum, J.L. (2025).
    Accurate simulation of flow through dipping aquifers with MODFLOW 6 using
    enhanced cell connectivity. Groundwater 63(3), 399-408.
    https://doi.org/10.1111/gwat.13459 (code: https://doi.org/10.5066/P13BNARA)

and motivated by Bardot, K., Lesueur, M., Siade, A.J. and McCallum, J.L.
(2023), Revisiting MODFLOW's capability to model flow through sedimentary
structures, Groundwater 61, 663-673 (https://doi.org/10.1111/gwat.13273).
Vertical pass-through of pinched cells follows LoopFlopy
(https://github.com/kerrybardot/loopflopy; Bardot et al. 2025, Environmental
Modelling & Software 192, 106557). This module is an independent, vectorised
implementation with its own pinch-out and sliver handling.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from . import _columns
from ._columns import ABSENT, ACTIVE, NOFLOW, PASSTHROUGH, STATUS_NAMES, subdivide  # noqa: F401
from ._connect3d import concat_links, layered_links, overlay_links, staggered_links, vertical_links
from ._disu_export import mf6_gridprops, mfusg_gridprops, write_gsf

__all__ = ["DisuGrid", "DisuOptions", "splice", "subdivide"]

CONNECTIVITY = ("full", "layered")
CL12_MODES = ("plan", "connector")


@dataclass(frozen=True)
class DisuOptions:
    """Validated :class:`DisuGrid` settings; see the class docstring for their meaning."""

    min_thickness: float = 0.0
    collapse: str | None = "split"
    connectivity: str = "full"
    sliver_tol: float = 0.05
    sliver_abs: float = 1e-3
    sliver_dip: float | None = 1.0
    cl12: str = "plan"
    strict: bool = False

    def __post_init__(self):
        for name in ("min_thickness", "sliver_tol", "sliver_abs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite number >= 0, got {value!r}.")
        if self.sliver_dip is not None and (
            isinstance(self.sliver_dip, bool) or not 0 <= self.sliver_dip < 90
        ):
            raise ValueError(f"sliver_dip must be None or in [0, 90) degrees, got {self.sliver_dip!r}.")
        if self.collapse not in _columns.COLLAPSE_MODES:
            raise ValueError(f"collapse must be one of {_columns.COLLAPSE_MODES}, got {self.collapse!r}.")
        if self.connectivity not in CONNECTIVITY:
            raise ValueError(f"connectivity must be one of {CONNECTIVITY}, got {self.connectivity!r}.")
        if self.cl12 not in CL12_MODES:
            raise ValueError(f"cl12 must be one of {CL12_MODES}, got {self.cl12!r}.")
        if not isinstance(self.strict, bool):
            raise ValueError("strict must be True or False.")


class DisuGrid:
    """A DISU grid built from vorflow 2D grids and FloPy-style layer arrays.

    Args:
        grids: The vorflow grid (GeoDataFrame with ``x``/``y`` generator
            columns), or a list with one grid per layer. Layers may use
            different grids, e.g. a :func:`splice` that is finer inside an area.
        top: Model top, scalar or one value per cell of the first layer's grid.
            With several grids it may also be a list with one entry per layer:
            ``None``, or the tops of that layer's cells. Where a layer's grid
            has cells that the layer above lacks, their tops are taken from
            the bottom of the cell above them unless given here.
        botm: Layer bottoms: ``(nlay,)`` or ``(nlay, ncpl)`` for one grid, or a
            list with one value or array per layer.
        idomain: ``None`` (all active), ``(nlay, ncpl)`` or a list per layer:
            >0 active, 0 no-flow (never passed through), -1 pass-through.
        min_thickness: Cells this thin or thinner are treated as pass-through.
        collapse: What happens to the interval of pass-through cells:
            ``"split"`` (default) shares it between the active cells above and
            below, ``"up"``/``"down"`` gives it to one side, and ``None`` keeps
            the gap, as MODFLOW 6 and LoopFlopy do. With a gap-free column the
            vertical connection lengths are exact and no thickness is lost.
        connectivity: ``"full"`` adds vertically staggered connections between
            offset neighbours (IHC=2); ``"layered"`` reproduces DISV
            connectivity (same-layer neighbours only).
        sliver_tol, sliver_abs, sliver_dip: An offset across a face is noise,
            and treated as aligned, when it is below ``sliver_tol`` (5 %) of the
            thinner cell and also flatter than ``sliver_dip`` degrees (1) between
            the two centres; ``sliver_abs`` is a floor in length units. Both
            conditions matter: a small offset on a steep dip between small cells
            is real. ``sliver_dip=None`` uses the thickness rule alone. Plan
            overlaps between grids below ``sliver_tol`` of the smaller cell are
            added to that cell's largest overlap.
        cl12: ``"plan"`` (perpendicular plan distance to the face, as in the
            paper; exact for XT3D) or ``"connector"``.
        strict: Raise instead of warn when a check fails.

    After :meth:`build`, ``nodemap`` maps ``(layer, cell)`` to the DISU node
    (-1 for removed cells): an ``(nlay, ncpl)`` array for one grid, a list of
    per-layer arrays for several. ``cellmap`` describes every layer cell.

    Cells with identical footprints in different grids (as kept by
    :func:`splice`) form one column through all their layers, so full
    connectivity applies across a change of grid. Other cells meet the cells
    above and below them through their plan overlaps.
    """

    def __init__(self, grids, top, botm, idomain=None, *, min_thickness=0.0, collapse="split",
                 connectivity="full", sliver_tol=0.05, sliver_abs=1e-3, sliver_dip=1.0,
                 cl12="plan", strict=False):
        self.options = DisuOptions(
            min_thickness=min_thickness, collapse=collapse, connectivity=connectivity,
            sliver_tol=sliver_tol, sliver_abs=sliver_abs, sliver_dip=sliver_dip,
            cl12=cl12, strict=strict,
        )
        self._grids, self._layer_grid, self._layers = _layer_inputs(grids, top, botm, idomain)
        self.nodemap = None
        self.cellmap = None
        self.links = None

    @classmethod
    def from_disv(cls, vertices, cell2d, top, botm, idomain=None, **options):
        """Build from MODFLOW 6 DISV arrays, e.g. an existing FloPy or LoopFlopy model.

        ``vertices`` is ``[[iv, x, y], ...]`` and ``cell2d`` is
        ``[[icell2d, xc, yc, ncvert, iv1, ...], ...]`` as for
        ``flopy.mf6.ModflowGwfdisv``; ``xc``/``yc`` are kept as the cell centres.
        The other arguments are those of :class:`DisuGrid`.
        """
        xy = {int(row[0]): (float(row[1]), float(row[2])) for row in vertices}
        polygons, xc, yc, ids = [], [], [], []
        for row in cell2d:
            ncvert = int(row[3])
            ring = [xy[int(iv)] for iv in row[4:4 + ncvert]]
            if len(ring) > 1 and ring[0] == ring[-1]:
                ring = ring[:-1]
            polygons.append(shapely.Polygon(ring))
            ids.append(int(row[0]))
            xc.append(float(row[1]))
            yc.append(float(row[2]))
        grid = gpd.GeoDataFrame({"node_id": ids, "x": xc, "y": yc}, geometry=polygons)
        return cls(grid, top, botm, idomain, **options)

    @property
    def nlay(self):
        return len(self._layer_grid)

    @property
    def single_grid(self):
        return len(self._grids) == 1

    @property
    def grid(self):
        """The plan grid when every layer uses the same one."""
        if not self.single_grid:
            raise AttributeError("Layers use different grids; see grids_per_layer.")
        return self._grids[0]

    @property
    def grids_per_layer(self):
        return [self._grids[g] for g in self._layer_grid]

    @property
    def nodes(self):
        self._require_build()
        return int((self._nodemap >= 0).sum())

    def build(self):
        """Classify cells, close pass-through gaps and connect the active cells. Returns ``self``."""
        from ._mfusg import _planar_faces

        opts = self.options
        planar = [_planar_faces(grid) for grid in self._grids]
        self._col_of, columns = _plan_columns(self._grids)
        self._columns = columns
        self._area = shapely.area(columns.geometry.to_numpy())
        tolerance = max(p[3] for p in planar)
        self._faces = _global_faces([p[2] for p in planar], self._col_of)
        cross = self._cross_faces(tolerance)
        if len(cross):
            self._faces = pd.concat([self._faces, cross.astype(self._faces.dtypes.to_dict())],
                                    ignore_index=True)
        from .utils import grid_to_disv

        self._planar = grid_to_disv(columns, tolerance=tolerance)
        top_l, bot_l, idomain = self._lattice()
        self.top_l, self.bot_l = top_l, bot_l
        status = _columns.classify(top_l, bot_l, idomain, opts.min_thickness)
        top_l, bot_l = _columns.collapse(top_l, bot_l, status, opts.collapse)
        active = status == ACTIVE
        nodemap = np.full(status.shape, -1, dtype=np.int64)
        nodemap[active] = np.arange(int(active.sum()))
        self.status, self._nodemap = status, nodemap
        self._node_top, self._node_bot = top_l[active], bot_l[active]
        self._node_layer, self._node_cell = np.nonzero(active)

        self.diagnostics = {"slivers_merged": 0, "largest_sliver": 0.0, "plan_slivers_merged": 0}
        overlaps = self._overlaps()
        vertical = concat_links(
            vertical_links(status, nodemap, top_l, bot_l, self._area),
            overlay_links(status, nodemap, top_l, bot_l, overlaps),
        )
        if opts.connectivity == "full":
            centres = columns[["x", "y"]].to_numpy(dtype=float)
            horizontal, found = staggered_links(
                status, nodemap, top_l, bot_l, self._faces, centres, sliver_tol=opts.sliver_tol,
                sliver_abs=opts.sliver_abs, sliver_dip=opts.sliver_dip, cl12=opts.cl12,
            )
            self.diagnostics.update(found)
        else:
            horizontal = layered_links(status, nodemap, self._faces, top_l, bot_l)
        self.links = concat_links(vertical, horizontal)
        self.diagnostics.update(_offset_ratio(status, top_l, bot_l, self._faces))
        self.diagnostics["vertical_offset"] = _vertical_offset(self.links, self._node_top, self._node_bot)
        self.nodemap = self._public(nodemap)
        self.cellmap = self._cellmap(top_l, bot_l)
        return self

    def get_gridprops_disu6(self):
        """Keyword arguments for ``flopy.mf6.ModflowGwfdisu`` (gridgen key names).

        The one-based layer number is written on the IHC diagonal, which
        MODFLOW 6 does not read. IDOMAIN cannot carry tags: MODFLOW 6 DISU
        accepts only 0 and 1 there, so use :attr:`cellmap` (or
        :meth:`save_cellmap`) to map nodes to layers, cells and zones.

        Where layers use different grids and the cells above and below a
        change of grid overlap slightly in elevation, ``vertical_offset_tolerance``
        is set to the largest overlap so MODFLOW 6 accepts the connections.
        """
        self._require_build()
        props = mf6_gridprops(
            self.nodes, self._node_top, self._node_bot, self._area[self._node_cell],
            self._node_layer, self.links, self._planar, self._node_cell,
        )
        if self.diagnostics["vertical_offset"] > 0:
            props["vertical_offset_tolerance"] = self.diagnostics["vertical_offset"] * (1 + 1e-9)
        return props

    def get_gridprops_disu5(self):
        """Keyword arguments for ``flopy.mfusg.MfUsgDisU`` (MODFLOW-USG; gridgen key names).

        Uses ``ivsd=1`` with an explicit ``ivc`` so staggered connections stay
        horizontal. ``fahl`` is the interface area: for a horizontal connection
        the face width times its (merged) overlap height. MODFLOW-USG has no
        XT3D; offset layers rely on the two-point formulation (or GNC).
        """
        self._require_build()
        return mfusg_gridprops(
            self.nodes, self.nlay, self._node_top, self._node_bot, self._area[self._node_cell],
            self._node_layer, self.links,
        )

    def write_gsf(self, filepath):
        """Write the grid specification file (GSF) used by MODFLOW-USG post-processors."""
        self._require_build()
        return write_gsf(filepath, self._planar, self._node_cell, self._node_top, self._node_bot,
                         self._node_layer, self.nlay)

    def to_nodes(self, values):
        """Pick the values of the active cells, in DISU node order.

        ``values`` is shaped like ``botm``: ``(nlay, ncpl)`` for one grid, or a
        list with one array (or scalar) per layer.
        """
        self._require_build()
        lattice = np.full(self._nodemap.shape, np.nan, dtype=object)
        for layer, layer_values in enumerate(self._per_layer(values, "values")):
            lattice[layer, self._col_of[self._layer_grid[layer]]] = layer_values
        picked = lattice[self._nodemap >= 0]
        return np.array(picked.tolist())

    def to_layers(self, values, fill=np.nan):
        """Spread per-node values back to the layers, with ``fill`` for removed cells.

        Returns ``(nlay, ncpl)`` for one grid, or a list of per-layer arrays.
        """
        self._require_build()
        values = np.asarray(values).ravel()
        if values.shape != (self.nodes,):
            raise ValueError(f"values must have {self.nodes} entries, one per node.")
        lattice = np.full(self._nodemap.shape, fill, dtype=np.result_type(values, np.asarray(fill)))
        lattice[self._nodemap >= 0] = values
        return self._public(lattice)

    def column(self, x, y):
        """Active DISU nodes at plan point ``(x, y)``, from the top down."""
        self._require_build()
        cols = self._columns_at(np.atleast_1d(float(x)), np.atleast_1d(float(y)))
        nodes = [self._nodemap[k, cols[g][0]] for k, g in enumerate(self._layer_grid) if cols[g][0] >= 0]
        return np.array([n for n in nodes if n >= 0], dtype=np.int64)

    def locate(self, x, y, z):
        """DISU node containing each point ``(x, y, z)``, or -1 outside every active cell.

        Uses the cell elevations after pass-through gaps are closed, so a
        point in a collapsed pass-through cell maps to the active cell that
        took its interval. Accepts scalars or arrays.
        """
        self._require_build()
        x, y, z = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (x, y, z))
        cols = self._columns_at(x, y)
        found = np.full(len(x), -1, dtype=np.int64)
        for k, g in enumerate(self._layer_grid):
            col = cols[g]
            node = np.where(col >= 0, self._nodemap[k, np.maximum(col, 0)], -1)
            hit = (node >= 0) & (found < 0)
            node_safe = np.maximum(node, 0)
            hit &= (self._node_bot[node_safe] <= z) & (z <= self._node_top[node_safe])
            found[hit] = node[hit]
        return found if found.size > 1 else int(found[0])

    def save_cellmap(self, path):
        """Write :attr:`cellmap` to CSV, so results can be mapped back without vorflow."""
        self._require_build()
        self.cellmap.to_csv(path, index=False)
        return path

    def _columns_at(self, x, y):
        """Per grid, the global column containing each plan point (-1 outside)."""
        if not hasattr(self, "_trees"):
            geoms = self._columns.geometry.to_numpy()
            self._trees = [shapely.STRtree(geoms[cols]) for cols in self._col_of]
        points = shapely.points(np.column_stack([x, y]))
        out = []
        for tree, cols in zip(self._trees, self._col_of):
            found = np.full(len(points), -1, dtype=np.int64)
            point, hit = tree.query(points, predicate="intersects")
            found[point[::-1]] = cols[hit[::-1]]     # First hit wins on shared edges.
            out.append(found)
        return out

    def report(self):
        """Plain-text summary of the grid and its connections."""
        self._require_build()
        counts = {STATUS_NAMES[code]: int((self.status == code).sum()) for code in STATUS_NAMES}
        ihc = np.bincount(self.links["ihc"], minlength=3)
        gained = (self._node_top - self._node_bot).sum() - (
            self.top_l - self.bot_l)[self.status == ACTIVE].sum()
        shape = (f"{self.nlay} layers x {self._grids[0].shape[0]} cells" if self.single_grid
                 else f"{self.nlay} layers on {len(self._grids)} grids")
        lines = [
            f"DISU grid: {self.nodes} nodes from {shape}",
            f"  cells: {counts['active']} active, {counts['passthrough']} pass-through, "
            f"{counts['noflow']} no-flow",
            f"  pass-through thickness given to active cells: {gained:.6g} (collapse={self.options.collapse!r})",
            f"  connections: {len(self.links['n'])} ({ihc[0]} vertical, {ihc[1]} horizontal, "
            f"{ihc[2]} staggered)",
        ]
        if len(self._faces):
            lines.append(f"  max plan orthogonality error: {self._faces['ortho_error'].max():.3g} deg")
        diag = self.diagnostics
        if self.options.connectivity == "full":
            lines.append(f"  slivers merged: {diag['slivers_merged']} (largest {diag['largest_sliver']:.3g})")
        if not self.single_grid:
            lines.append(f"  plan-overlap slivers merged: {diag['plan_slivers_merged']}")
            if diag["vertical_offset"] > 0:
                lines.append(
                    f"  cells above and below a change of grid overlap by up to {diag['vertical_offset']:.3g}; "
                    "MODFLOW 6 vertical_offset_tolerance is set to accept it."
                )
        ratio = diag["max_offset_ratio"]
        lines.append(f"  max offset ratio (offset / thinner cell, same layer): {ratio:.3g}")
        if ratio >= 1:
            joined = ("joined only through other layers" if self.options.connectivity == "full"
                      else "still joined as same-layer neighbours")
            lines.append(
                f"  {diag['disconnected_faces']} same-layer neighbour pairs are offset by a whole cell "
                f"or more and are {joined}; about {int(np.ceil(ratio)) + 1} sublayers would keep "
                "each layer connected (Provost et al. 2025)."
            )
        if ratio > 0:
            if self.options.connectivity == "layered":
                lines.append("  layers are vertically offset: connectivity='full' is recommended.")
            lines.append(
                "  for offset layers use XT3D, and give NPF angle2 and angle3 (zeros are fine): "
                "without them MODFLOW 6 XT3D ignores the vertical offsets."
            )
        return "\n".join(lines)

    def _lattice(self):
        """Elevations and idomain over every plan column, NaN where a layer's grid lacks the column."""
        ncol = len(self._columns)
        top_l = np.full((self.nlay, ncol), np.nan)
        bot_l = np.full((self.nlay, ncol), np.nan)
        idomain = np.zeros((self.nlay, ncol), dtype=np.int64)
        for layer, (top, bot, ido) in enumerate(self._layers):
            cols = self._col_of[self._layer_grid[layer]]
            bot_l[layer, cols] = bot
            idomain[layer, cols] = ido
            if layer == 0:
                top_l[0, cols] = top
                continue
            above = bot_l[layer - 1, cols]
            new = np.isnan(above)
            layer_top = above.copy()
            if top is not None:
                layer_top[new] = np.broadcast_to(top, cols.shape)[new]
            elif new.any():
                layer_top[new] = self._sample_above(layer, cols[new], bot_l[layer - 1])
            top_l[layer, cols] = layer_top
        with np.errstate(invalid="ignore"):
            bad = np.argwhere(top_l < bot_l)
        if len(bad):
            layer, col = bad[0]
            raise ValueError(
                f"Negative thickness in layer {layer} (column {col}): each bottom must lie at or "
                "below the bottom of the layer above."
            )
        return top_l, bot_l, idomain

    def _sample_above(self, layer, cols, bot_above):
        """Bottom of the layer-above cell that contains each column's centre."""
        above_cols = self._col_of[self._layer_grid[layer - 1]]
        geoms = self._columns.geometry.to_numpy()
        tree = shapely.STRtree(geoms[above_cols])
        centres = shapely.points(self._columns[["x", "y"]].to_numpy(dtype=float)[cols])
        point, hit = tree.query(centres, predicate="intersects")
        found = np.full(len(cols), -1)
        found[point[::-1]] = hit[::-1]           # First hit wins on shared edges.
        if (found < 0).any():
            raise ValueError(
                f"Layer {layer}: a cell centre lies outside the layer above; give its top explicitly."
            )
        return bot_above[above_cols[found]]

    def _cross_faces(self, tolerance):
        """Faces between cells of different grids that touch along an edge without overlapping.

        Inside a refined area, a coarse cell above a change of grid and a fine
        cell below it can be plan neighbours; with offset layers their walls
        overlap vertically and need a staggered connection.
        """
        from ._mfusg import _straight_face

        geoms = self._columns.geometry.to_numpy()
        centres = self._columns[["x", "y"]].to_numpy(dtype=float)
        records = []
        for g1 in range(len(self._grids)):
            for g2 in range(g1 + 1, len(self._grids)):
                a_cols, b_cols = self._col_of[g1], self._col_of[g2]
                a_only = a_cols[~np.isin(a_cols, b_cols)]
                b_only = b_cols[~np.isin(b_cols, a_cols)]
                if not len(a_only) or not len(b_only):
                    continue
                ia, ib = shapely.STRtree(geoms[b_only]).query(geoms[a_only], predicate="intersects")
                for i, j in zip(a_only[ia], b_only[ib]):
                    if geoms[i].intersection(geoms[j]).area > tolerance * geoms[i].length:
                        continue                 # Overlapping cells meet vertically, not side by side.
                    shared = shapely.intersection(geoms[i].boundary, geoms[j].boundary, grid_size=tolerance)
                    try:
                        record = _straight_face(shared, centres[[i, j]], (int(i), int(j)), tolerance)
                    except ValueError:
                        record = _bent_face(shared, centres[[i, j]], (int(i), int(j)))
                    if record is not None and record["shared_length"] > tolerance:
                        records.append(record)
        columns = ["cell_id_1", "cell_id_2", "shared_length", "distance_1", "distance_2",
                   "ortho_error", "normal_x", "normal_y"]
        return pd.DataFrame(records, columns=columns)

    def _overlaps(self):
        """Plan overlaps between the columns of grids used in consecutive layers."""
        pairs = {tuple(sorted(p)) for p in zip(self._layer_grid[:-1], self._layer_grid[1:]) if p[0] != p[1]}
        geoms = self._columns.geometry.to_numpy()
        frames = []
        for g1, g2 in sorted(pairs):
            a_cols, b_cols = self._col_of[g1], self._col_of[g2]
            a_only = a_cols[~np.isin(a_cols, b_cols)]
            b_only = b_cols[~np.isin(b_cols, a_cols)]
            if not len(a_only) or not len(b_only):
                continue
            ia, ib = shapely.STRtree(geoms[b_only]).query(geoms[a_only], predicate="intersects")
            area = shapely.area(shapely.intersection(geoms[a_only][ia], geoms[b_only][ib]))
            frames.append(pd.DataFrame({"col_a": a_only[ia], "col_b": b_only[ib], "area": area}))
        if not frames:
            return pd.DataFrame({"col_a": [], "col_b": [], "area": []})
        overlaps = pd.concat(frames, ignore_index=True)
        overlaps = overlaps[overlaps.area > 1e-12 * self._area[overlaps.col_a.to_numpy(dtype=int)]]
        overlaps, merged = _merge_plan_slivers(overlaps, self._area, self.options.sliver_tol)
        self.diagnostics["plan_slivers_merged"] = merged
        return overlaps

    def _per_layer(self, values, name):
        """Split values shaped like ``botm`` into one array per layer."""
        if isinstance(values, (list, tuple)):
            if len(values) != self.nlay:
                raise ValueError(f"{name} needs one entry per layer ({self.nlay}).")
            parts = values
        else:
            values = np.asarray(values)
            if not self.single_grid or values.ndim != 2 or values.shape[0] != self.nlay:
                raise ValueError(f"{name} must be shaped like botm.")
            parts = list(values)
        out = []
        for layer, part in enumerate(parts):
            size = len(self._col_of[self._layer_grid[layer]])
            part = np.broadcast_to(np.asarray(part), (size,))
            out.append(part)
        return out

    def _public(self, lattice):
        if self.single_grid:
            return lattice[:, self._col_of[0]]
        return [lattice[layer, self._col_of[g]] for layer, g in enumerate(self._layer_grid)]

    def _cellmap(self, top_l, bot_l):
        frames = []
        for layer, g in enumerate(self._layer_grid):
            cols = self._col_of[g]
            grid = self._grids[g]
            frame = pd.DataFrame({
                "node": self._nodemap[layer, cols], "layer": layer, "icpl": np.arange(len(cols)),
                "grid": g, "column": cols,
                "node_id": grid["node_id"].to_numpy() if "node_id" in grid else np.arange(len(cols)),
                "status": self.status[layer, cols],
                "x": grid["x"].to_numpy(dtype=float), "y": grid["y"].to_numpy(dtype=float),
                "top": top_l[layer, cols], "bot": bot_l[layer, cols],
            })
            if "zone_id" in grid:
                frame["zone_id"] = grid["zone_id"].to_numpy()
            frames.append(frame)
        cellmap = pd.concat(frames, ignore_index=True)
        cellmap["status"] = pd.Categorical.from_codes(
            cellmap["status"].to_numpy(), [STATUS_NAMES[code] for code in range(len(STATUS_NAMES))],
        )
        return cellmap

    def _require_build(self):
        if self.nodemap is None:
            raise RuntimeError("Call build() first.")


def splice(base, fine, polygon):
    """A plan grid with ``base`` cells outside ``polygon`` and ``fine`` cells inside it.

    Both grids must have the polygon outline as cell faces, which vorflow
    gives when the outline is added as a barrier line
    (``add_line(..., is_barrier=True)``) in the blueprint of both. No cell may
    cross the outline. The nodes along the outline need not match: the
    connections there come from the shared face segments (hanging nodes).

    The base cells are kept unchanged, so in :class:`DisuGrid` they continue
    the base grid's columns. Fine cells get new ``node_id`` values after the
    base ones; ``splice_source`` and ``source_node_id`` record their origin.
    """
    for name, grid in (("base", base), ("fine", fine)):
        missing = {"geometry", "x", "y"}.difference(grid.columns)
        if missing:
            raise ValueError(f"{name} grid missing required columns: {sorted(missing)}.")
    polygon = shapely.make_valid(polygon)
    parts = []
    for name, grid, keep_inside in (("base", base, False), ("fine", fine, True)):
        geoms = grid.geometry.to_numpy()
        area = shapely.area(geoms)
        inside = shapely.area(shapely.intersection(geoms, polygon))
        tolerance = 1e-9 * area
        crossing = (inside > tolerance) & (area - inside > tolerance)
        if crossing.any():
            raise ValueError(
                f"{int(crossing.sum())} {name} cells cross the splice outline; add the outline as a "
                "barrier line (is_barrier=True) so it becomes cell faces."
            )
        selected = grid[(inside > tolerance) == keep_inside].copy()
        selected["splice_source"] = name
        selected["source_node_id"] = (selected["node_id"].to_numpy() if "node_id" in selected
                                      else np.flatnonzero((inside > tolerance) == keep_inside))
        parts.append(selected)
    outside, inside = parts
    inside_area, polygon_area = inside.geometry.area.sum(), polygon.intersection(
        shapely.union_all(base.geometry.to_numpy())).area
    if abs(inside_area - polygon_area) > 1e-9 * max(polygon_area, 1.0):
        raise ValueError(
            f"Fine cells cover {inside_area:.6g} of the {polygon_area:.6g} splice area; the fine grid "
            "must cover the whole area inside the outline."
        )
    start = int(outside["node_id"].max()) + 1 if "node_id" in outside and len(outside) else 0
    inside["node_id"] = np.arange(start, start + len(inside))
    if "node_id" not in outside:
        outside["node_id"] = outside["source_node_id"]
    return gpd.GeoDataFrame(pd.concat([outside, inside], ignore_index=True), crs=base.crs)


def _layer_inputs(grids, top, botm, idomain):
    """Normalize grids and layer arrays into ``(unique grids, grid per layer, per-layer arrays)``."""
    if isinstance(grids, (list, tuple)):
        if not grids:
            raise ValueError("grids must not be empty.")
        unique, layer_grid = [], []
        for grid in grids:
            index = next((i for i, g in enumerate(unique) if g is grid), None)
            if index is None:
                unique.append(grid)
                index = len(unique) - 1
            layer_grid.append(index)
    else:
        unique, layer_grid = [grids], None
    for grid in unique:
        missing = {"geometry", "x", "y"}.difference(getattr(grid, "columns", ()))
        if missing:
            raise ValueError(f"grid missing required columns: {sorted(missing)}.")
        if len(grid) == 0:
            raise ValueError("grid must not be empty.")

    if len(unique) == 1:
        top_l, bot_l, ido = _columns.layer_arrays(top, botm, idomain, ncpl=len(unique[0]))
        nlay = bot_l.shape[0]
        if layer_grid is not None and len(layer_grid) != nlay:
            raise ValueError(f"grids lists {len(layer_grid)} layers but botm has {nlay}.")
        layers = [(top_l[0] if k == 0 else None, bot_l[k], ido[k]) for k in range(nlay)]
        return unique, [0] * nlay, layers

    if not isinstance(botm, (list, tuple)):
        raise ValueError("With different grids per layer, botm must be a list with one entry per layer.")
    nlay = len(botm)
    if layer_grid is None:
        layer_grid = [0] * nlay
    if len(layer_grid) != nlay:
        raise ValueError(f"grids lists {len(layer_grid)} layers but botm has {nlay}.")
    tops = top if isinstance(top, (list, tuple)) else [top] + [None] * (nlay - 1)
    idos = idomain if isinstance(idomain, (list, tuple)) else [idomain] * nlay
    if len(tops) != nlay or len(idos) != nlay:
        raise ValueError(f"Per-layer top and idomain lists need {nlay} entries.")
    if tops[0] is None:
        raise ValueError("The top of the first layer is required.")
    layers = []
    for k in range(nlay):
        size = len(unique[layer_grid[k]])
        bot = np.broadcast_to(np.asarray(botm[k], dtype=float), (size,)).copy()
        layer_top = None if tops[k] is None else np.broadcast_to(np.asarray(tops[k], dtype=float), (size,)).copy()
        ido = np.ones(size, dtype=np.int64) if idos[k] is None else np.broadcast_to(
            np.asarray(idos[k]), (size,)).astype(np.int64)
        if not np.isfinite(bot).all() or (layer_top is not None and not np.isfinite(layer_top).all()):
            raise ValueError(f"Layer {k} elevations must be finite.")
        layers.append((layer_top, bot, ido))
    return unique, layer_grid, layers


def _plan_columns(grids):
    """Global plan columns: cells with the same generator and footprint in several grids are one column."""
    rows, col_of, index = [], [], {}
    for g, grid in enumerate(grids):
        geoms = grid.geometry.to_numpy()
        xy = grid[["x", "y"]].to_numpy(dtype=float)
        ids = np.empty(len(grid), dtype=np.int64)
        for i, (geom, (x, y)) in enumerate(zip(geoms, xy)):
            match = next((c for c in index.get((x, y), ()) if rows[c][0].equals(geom)), None)
            if match is None:
                match = len(rows)
                rows.append((geom, x, y, g, i))
                index.setdefault((x, y), []).append(match)
            ids[i] = match
        col_of.append(ids)
    geom, x, y, g, i = zip(*rows)
    columns = gpd.GeoDataFrame({"x": x, "y": y, "grid": g, "icpl": i, "node_id": np.arange(len(rows))},
                               geometry=list(geom))
    return col_of, columns


def _global_faces(face_tables, col_of):
    """Plan faces of every grid, renumbered to global columns, each column pair once."""
    frames = []
    for faces, cols in zip(face_tables, col_of):
        faces = faces.copy()
        faces["cell_id_1"] = cols[faces["cell_id_1"].to_numpy(dtype=np.int64)]
        faces["cell_id_2"] = cols[faces["cell_id_2"].to_numpy(dtype=np.int64)]
        frames.append(faces)
    faces = pd.concat(frames, ignore_index=True)
    key = np.sort(faces[["cell_id_1", "cell_id_2"]].to_numpy(), axis=1)
    keep = ~pd.DataFrame(key).duplicated().to_numpy()
    return faces[keep].reset_index(drop=True)


def _bent_face(shared, centres, pair):
    """One face for a contact that is not a single straight segment: total length, mean normal."""
    from ._mfusg import _line_parts

    parts = _line_parts(shared)
    if not parts:
        return None
    lengths = np.array([part.length for part in parts])
    vectors = np.array([np.subtract(part.coords[-1], part.coords[0])[:2] for part in parts])
    normals = np.column_stack([-vectors[:, 1], vectors[:, 0]]) / lengths[:, None]
    towards = centres[1] - centres[0]
    normals *= np.sign(normals @ towards)[:, None]
    normal = (normals * lengths[:, None]).sum(axis=0)
    normal /= np.linalg.norm(normal)
    midpoints = np.array([part.interpolate(0.5, normalized=True).coords[0][:2] for part in parts])
    face_point = (midpoints * lengths[:, None]).sum(axis=0) / lengths.sum()
    d1, d2 = (face_point - centres[0]) @ normal, (centres[1] - face_point) @ normal
    angle = np.degrees(np.arccos(np.clip(abs(towards @ normal) / np.linalg.norm(towards), 0, 1)))
    return {"cell_id_1": pair[0], "cell_id_2": pair[1], "shared_length": float(lengths.sum()),
            "distance_1": float(abs(d1)), "distance_2": float(abs(d2)), "ortho_error": float(angle),
            "normal_x": float(normal[0]), "normal_y": float(normal[1])}


def _merge_plan_slivers(overlaps, area, sliver_tol):
    """Add overlaps smaller than ``sliver_tol`` of the smaller cell to that cell's largest overlap."""
    if overlaps.empty or sliver_tol == 0:
        return overlaps.reset_index(drop=True), 0
    overlaps = overlaps.reset_index(drop=True)
    a = overlaps.col_a.to_numpy(dtype=np.int64)
    b = overlaps.col_b.to_numpy(dtype=np.int64)
    small_cell = np.where(area[a] <= area[b], a, b)
    tiny = overlaps.area.to_numpy() < sliver_tol * area[small_cell]
    if not tiny.any():
        return overlaps, 0
    merged_area = overlaps.area.to_numpy(dtype=float).copy()
    kept = ~tiny
    for row in np.flatnonzero(tiny):
        cell = small_cell[row]
        candidates = np.flatnonzero(kept & ((a == cell) | (b == cell)))
        if not len(candidates):
            kept[row] = True                    # The only overlap of that cell: keep it.
            continue
        target = candidates[np.argmax(merged_area[candidates])]
        merged_area[target] += merged_area[row]
    overlaps = overlaps.assign(area=merged_area)[kept].reset_index(drop=True)
    return overlaps, int((~kept).sum())


def _vertical_offset(links, top, bot):
    """Largest elevation overlap between the upper and lower cells of a vertical connection."""
    vertical = links["ihc"] == 0
    if not vertical.any():
        return 0.0
    n, m = links["n"][vertical], links["m"][vertical]
    upper = np.where(top[n] >= top[m], n, m)
    lower = np.where(upper == n, m, n)
    return float(max(0.0, (top[lower] - bot[upper]).max()))


def _offset_ratio(status, top_l, bot_l, faces):
    """Largest vertical offset between same-layer neighbours, relative to the thinner cell.

    Provost et al. (2025) show connectivity changes where this ratio reaches 1
    (same-layer neighbours stop touching), 2 and 3.
    """
    i = faces["cell_id_1"].to_numpy(dtype=np.int64)
    j = faces["cell_id_2"].to_numpy(dtype=np.int64)
    both = (status[:, i] == ACTIVE) & (status[:, j] == ACTIVE)
    if not both.any():
        return {"max_offset_ratio": 0.0, "disconnected_faces": 0}
    with np.errstate(invalid="ignore"):
        offset = np.maximum(np.abs(top_l[:, i] - top_l[:, j]), np.abs(bot_l[:, i] - bot_l[:, j]))
        thinner = np.minimum(top_l[:, i] - bot_l[:, i], top_l[:, j] - bot_l[:, j])
    ratio = np.where(both, offset / np.where(both, thinner, 1.0), 0.0)
    return {"max_offset_ratio": float(ratio.max()), "disconnected_faces": int((ratio >= 1).sum())}
