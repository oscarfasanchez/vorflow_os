"""Barrier connections of a built DISU grid and their HFB package data.

A horizontal connection is a barrier connection when the plan segment between
its two cell centres crosses the barrier trace of either cell's layer. On a
grid meshed with ``is_barrier=True`` these are exactly the faces lying on the
trace; elsewhere they are the faces nearest the trace (a zig-zag), which still
separates the two sides. This module returns FloPy constructor arguments
without importing FloPy. Node numbers are zero-based.
"""

from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

_LINEAR = {shapely.GeometryType.LINESTRING, shapely.GeometryType.LINEARRING,
           shapely.GeometryType.MULTILINESTRING}


def _traces(trace, nlay):
    """One validated linear geometry (or None) per layer."""
    if trace is None or isinstance(trace, BaseGeometry):
        traces = [trace] * nlay
    else:
        try:
            traces = list(trace)
        except TypeError as error:
            raise ValueError("trace must be a line geometry or one per layer.") from error
        if len(traces) != nlay:
            raise ValueError(f"trace needs one line geometry (or None) per layer: {nlay}, got {len(traces)}.")
    for t in traces:
        if t is not None and (not isinstance(t, BaseGeometry) or shapely.get_type_id(t) not in _LINEAR
                              or t.is_empty):
            raise ValueError("Each trace must be a non-empty LineString, LinearRing or MultiLineString, or None.")
    return traces


def barrier_faces(grid, trace):
    """Find the connections a barrier trace cuts; see ``DisuGrid.barrier_faces``."""
    grid._require_build()
    traces = _traces(trace, grid.nlay)
    links = grid.links
    n, m = np.asarray(links["n"], dtype=np.int64), np.asarray(links["m"], dtype=np.int64)
    cells = grid.cellmap[grid.cellmap.node >= 0].sort_values("node")
    xy = cells[["x", "y"]].to_numpy(dtype=float)
    layer = cells.layer.to_numpy()
    horizontal = np.flatnonzero(np.asarray(links["ihc"]) != 0)
    tree = shapely.STRtree(shapely.linestrings(np.stack((xy[n[horizontal]], xy[m[horizontal]]), axis=1)))
    hit = np.zeros(len(n), dtype=bool)
    for t in {id(t): t for t in traces if t is not None}.values():
        found = horizontal[tree.query(t, predicate="crosses")]
        own = [k for k, other in enumerate(traces) if other is t]
        hit[found[np.isin(layer[n[found]], own) | np.isin(layer[m[found]], own)]] = True
    found = np.flatnonzero(hit)
    on_trace = np.zeros(len(found), dtype=bool)
    if len(found):
        tolerance = 1e-7 * max(float(np.ptp(xy, axis=0).max()), 1.0)
        polygons = [grid._grids[int(g)].geometry.iloc[int(i)] for g, i in zip(cells.grid, cells.icpl)]
        shared = shapely.intersection(shapely.boundary([polygons[i] for i in n[found]]),
                                      shapely.boundary([polygons[i] for i in m[found]]))
        for k, link in enumerate(found):
            face = shared[k]
            zone = [t.buffer(tolerance) for t in (traces[layer[n[link]]], traces[layer[m[link]]]) if t is not None]
            on_trace[k] = face.length > tolerance and any(z.covers(face) for z in zone)
    return BarrierFaces(found, n[found], m[found], on_trace, np.ones(len(found)), grid.nodes)


@dataclass
class BarrierFaces:
    """Connections cut by a barrier, with MF6 and MODFLOW-USG HFB package data.

    ``links`` indexes ``DisuGrid.links``; ``n`` and ``m`` are the zero-based
    nodes on either side. ``on_trace`` is True where the shared face lies on
    the trace, False where the face is only the nearest one (zig-zag).
    ``factor`` multiplies the hydraulic characteristic of each face (1 on a
    vertical trace).
    """

    links: np.ndarray
    n: np.ndarray
    m: np.ndarray
    on_trace: np.ndarray
    factor: np.ndarray
    nodes: int

    def __len__(self):
        return len(self.links)

    @property
    def pairs(self):
        """``(n, m)`` node pairs, for ``DisuGrid.estimate_gnc(barriers=...)``."""
        return np.column_stack((self.n, self.m))

    def _hydchr(self, hydchr):
        try:
            values = np.broadcast_to(np.asarray(hydchr, dtype=float), (len(self),))
        except (TypeError, ValueError) as error:
            raise ValueError("hydchr must be a number or have one value per barrier connection.") from error
        if not np.isfinite(values).all():
            raise ValueError("hydchr must be finite.")
        if np.any(values < 0) and np.any(self.factor != 1):
            raise ValueError("A negative hydchr (conductance multiplier) cannot carry a face factor; "
                             "give the hydraulic characteristic (>= 0).")
        return np.where(values < 0, values, values * self.factor)

    def get_gridprops_mf6_hfb(self, hydchr):
        """Return ``ModflowGwfhfb`` kwargs for one stress period, or None when empty.

        ``hydchr`` is the barrier conductivity divided by its width (>= 0; a
        negative value is MODFLOW's conductance multiplier), one number or one
        per connection. Vertical barrier connections need MF6 6.7 or later.
        """
        values = self._hydchr(hydchr)
        if not len(self):
            return None
        return {
            "maxhfb": len(self),
            "stress_period_data": {0: [((int(a),), (int(b),), float(h)) for a, b, h in zip(self.n, self.m, values)]},
        }

    def get_gridprops_mfusg_hfb(self, hydchr):
        """Return ``flopy.modflow.ModflowHfb`` kwargs for a USG model, or None when empty."""
        values = self._hydchr(hydchr)
        if not len(self):
            return None
        return {
            "nhfbnp": len(self),
            "hfb_data": [[int(a), int(b), float(h)] for a, b, h in zip(self.n, self.m, values)],
        }
