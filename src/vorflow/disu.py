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

import numpy as np
import pandas as pd

from . import _columns
from ._columns import ACTIVE, NOFLOW, PASSTHROUGH, STATUS_NAMES, subdivide  # noqa: F401
from ._connect3d import concat_links, layered_links, staggered_links, vertical_links
from ._disu_export import mf6_gridprops, mfusg_gridprops, write_gsf

__all__ = ["DisuGrid", "DisuOptions", "subdivide"]

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
    sliver_dip: float | None = None
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
    """A DISU grid built from a vorflow 2D grid and FloPy-style layer arrays.

    Args:
        grids: The vorflow grid (GeoDataFrame with ``x``/``y`` generator
            columns), or a list with one grid per layer.
        top: Model top, scalar or ``(ncpl,)``.
        botm: Layer bottoms, ``(nlay,)`` or ``(nlay, ncpl)``.
        idomain: ``None`` (all active) or ``(nlay, ncpl)``: >0 active, 0
            no-flow (never passed through), -1 vertical pass-through.
        min_thickness: Cells this thin or thinner are treated as pass-through.
        collapse: What happens to the interval of pass-through cells:
            ``"split"`` (default) shares it between the active cells above and
            below, ``"up"``/``"down"`` gives it to one side, and ``None`` keeps
            the gap, as MODFLOW 6 and LoopFlopy do. With a gap-free column the
            vertical connection lengths are exact and no thickness is lost.
        connectivity: ``"full"`` adds vertically staggered connections between
            offset neighbours (IHC=2); ``"layered"`` reproduces DISV
            connectivity (same-layer neighbours only).
        sliver_tol, sliver_abs, sliver_dip: Offsets smaller than
            ``max(sliver_abs, sliver_tol * thinner cell)`` (or than
            ``sliver_dip`` degrees across a face) are treated as aligned.
        cl12: ``"plan"`` (perpendicular plan distance to the face, as in the
            paper; exact for XT3D) or ``"connector"``.
        strict: Raise instead of warn when a check fails.

    After :meth:`build`, ``nodemap`` maps ``(layer, cell)`` to the DISU node
    (-1 for removed cells) and ``cellmap`` describes every layer cell.
    """

    def __init__(self, grids, top, botm, idomain=None, *, min_thickness=0.0, collapse="split",
                 connectivity="full", sliver_tol=0.05, sliver_abs=1e-3, sliver_dip=None,
                 cl12="plan", strict=False):
        self.options = DisuOptions(
            min_thickness=min_thickness, collapse=collapse, connectivity=connectivity,
            sliver_tol=sliver_tol, sliver_abs=sliver_abs, sliver_dip=sliver_dip,
            cl12=cl12, strict=strict,
        )
        self.grid = _single_grid(grids)
        self.top_l, self.bot_l, self.idomain = _columns.layer_arrays(top, botm, idomain, ncpl=len(self.grid))
        self.nodemap = None
        self.cellmap = None
        self.links = None

    @property
    def nlay(self):
        return self.top_l.shape[0]

    @property
    def ncpl(self):
        return self.top_l.shape[1]

    @property
    def nodes(self):
        self._require_build()
        return int((self.nodemap >= 0).sum())

    def build(self):
        """Classify cells, close pass-through gaps and connect the active cells. Returns ``self``."""
        from ._mfusg import _planar_faces

        self._planar, self._area, self._faces, _ = _planar_faces(self.grid)
        status = _columns.classify(self.top_l, self.bot_l, self.idomain, self.options.min_thickness)
        top_l, bot_l = _columns.collapse(self.top_l, self.bot_l, status, self.options.collapse)
        active = status == ACTIVE
        nodemap = np.full(status.shape, -1, dtype=np.int64)
        nodemap[active] = np.arange(int(active.sum()))
        self.status, self.nodemap = status, nodemap
        self._node_top, self._node_bot = top_l[active], bot_l[active]
        self._node_layer, self._node_cell = np.nonzero(active)
        opts = self.options
        self.diagnostics = {"slivers_merged": 0, "largest_sliver": 0.0}
        if opts.connectivity == "full":
            centres = self.grid[["x", "y"]].to_numpy(dtype=float)
            horizontal, found = staggered_links(
                status, nodemap, top_l, bot_l, self._faces, centres, sliver_tol=opts.sliver_tol,
                sliver_abs=opts.sliver_abs, sliver_dip=opts.sliver_dip, cl12=opts.cl12,
            )
            self.diagnostics.update(found)
        else:
            horizontal = layered_links(status, nodemap, self._faces, top_l, bot_l)
        self.links = concat_links(vertical_links(status, nodemap, top_l, bot_l, self._area), horizontal)
        self.diagnostics.update(_offset_ratio(status, top_l, bot_l, self._faces))
        self.cellmap = self._cellmap(top_l, bot_l)
        return self

    def get_gridprops_disu6(self, idomain_tag=None):
        """Keyword arguments for ``flopy.mf6.ModflowGwfdisu`` (gridgen key names).

        Args:
            idomain_tag: ``None`` (no IDOMAIN), ``"layer"`` (one-based layer
                number) or a ``(nlay, ncpl)`` integer array of tags >= 1, written
                as the DISU IDOMAIN so every node stays active and carries its tag.
        """
        self._require_build()
        return mf6_gridprops(
            self.nodes, self._node_top, self._node_bot, self._area[self._node_cell],
            self._node_layer, self.links, self._planar, self._node_cell,
            idomain=self._idomain_tags(idomain_tag),
        )

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
        """Pick the ``(nlay, ncpl)`` values of the active cells, in DISU node order."""
        self._require_build()
        values = np.asarray(values)
        if values.shape != self.nodemap.shape:
            raise ValueError(f"values must have shape {self.nodemap.shape}.")
        return values[self.nodemap >= 0]

    def to_layers(self, values, fill=np.nan):
        """Spread per-node values back to ``(nlay, ncpl)``, with ``fill`` for removed cells."""
        self._require_build()
        values = np.asarray(values).ravel()
        if values.shape != (self.nodes,):
            raise ValueError(f"values must have {self.nodes} entries, one per node.")
        result = np.full(self.nodemap.shape, fill, dtype=np.result_type(values, np.asarray(fill)))
        result[self.nodemap >= 0] = values
        return result

    def report(self):
        """Plain-text summary of the grid and its connections."""
        self._require_build()
        counts = {STATUS_NAMES[code]: int((self.status == code).sum()) for code in STATUS_NAMES}
        ihc = np.bincount(self.links["ihc"], minlength=3)
        gained = (self._node_top - self._node_bot).sum() - (
            self.top_l - self.bot_l)[self.status == ACTIVE].sum()
        lines = [
            f"DISU grid: {self.nodes} nodes from {self.nlay} layers x {self.ncpl} cells",
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

    def _cellmap(self, top_l, bot_l):
        layer, cell = np.indices(self.status.shape).reshape(2, -1)
        frame = pd.DataFrame({
            "node": self.nodemap.ravel(),
            "layer": layer, "icpl": cell,
            "node_id": np.asarray(self._planar["node_id"])[cell],
            # Status codes are 0..2, so they double as category codes.
            "status": pd.Categorical.from_codes(
                self.status.ravel(), [STATUS_NAMES[code] for code in range(len(STATUS_NAMES))],
            ),
            "x": self.grid["x"].to_numpy(dtype=float)[cell],
            "y": self.grid["y"].to_numpy(dtype=float)[cell],
            "top": top_l.ravel(), "bot": bot_l.ravel(),
        })
        if "zone_id" in self.grid:
            frame["zone_id"] = self.grid["zone_id"].to_numpy()[cell]
        return frame

    def _idomain_tags(self, tag):
        if tag is None:
            return None
        if isinstance(tag, str):
            if tag != "layer":
                raise ValueError("idomain_tag must be None, 'layer' or an array.")
            return self._node_layer + 1
        tags = self.to_nodes(tag)
        if not np.issubdtype(tags.dtype, np.integer) or (tags < 1).any():
            raise ValueError("idomain_tag values must be integers >= 1 for active cells.")
        return tags

    def _require_build(self):
        if self.nodemap is None:
            raise RuntimeError("Call build() first.")


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
    offset = np.maximum(np.abs(top_l[:, i] - top_l[:, j]), np.abs(bot_l[:, i] - bot_l[:, j]))
    thinner = np.minimum(top_l[:, i] - bot_l[:, i], top_l[:, j] - bot_l[:, j])
    ratio = np.where(both, offset / np.where(both, thinner, 1.0), 0.0)
    return {"max_offset_ratio": float(ratio.max()), "disconnected_faces": int((ratio >= 1).sum())}


def _single_grid(grids):
    if isinstance(grids, (list, tuple)):
        if not grids or any(grid is not grids[0] for grid in grids):
            raise NotImplementedError("Different grids per layer are not available yet.")
        grids = grids[0]
    missing = {"geometry", "x", "y"}.difference(getattr(grids, "columns", ()))
    if missing:
        raise ValueError(f"grid missing required columns: {sorted(missing)}.")
    if len(grids) == 0:
        raise ValueError("grid must not be empty.")
    return grids
