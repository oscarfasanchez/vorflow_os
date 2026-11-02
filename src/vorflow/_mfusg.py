"""MODFLOW-USG DISU serialization; never use these arrays for MF6 DISU.

Geometry and layer preparation are shared by DISU and original 3D GSF output.
FloPy is deliberately absent: only the tessellator's package wrapper needs it.
"""

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
import shapely


logger = logging.getLogger(__name__)
METADATA_KEYS = {"node_id", "cell_map", "connectivity"}
PACKAGE_KEYS = {
    "nodes", "nlay", "njag", "ivsd", "idsymrd", "nodelay",
    "top", "bot", "area", "iac", "ja", "cl12", "fahl",
}


@dataclass
class _LayeredGrid:
    planar: dict
    top: np.ndarray
    bot: np.ndarray
    area: np.ndarray
    cell_map: pd.DataFrame
    faces: pd.DataFrame

    @property
    def nlay(self):
        return self.bot.shape[0]

    @property
    def ncpl(self):
        return self.planar["ncpl"]


def _layer_arrays(top, botm, nlay, ncpl):
    if isinstance(nlay, (bool, np.bool_)) or not isinstance(nlay, (int, np.integer)) or nlay < 1:
        raise ValueError("nlay must be a positive integer.")
    top = np.asarray(top, dtype=float)
    if top.ndim == 0:
        top = np.full(ncpl, top.item())
    elif top.shape != (ncpl,):
        raise ValueError(f"top must be scalar or have shape ({ncpl},).")
    bot = np.asarray(botm, dtype=float)
    if bot.ndim == 0:
        bot = np.full((nlay, ncpl), bot.item())
    elif bot.shape == (nlay,):
        bot = np.broadcast_to(bot[:, None], (nlay, ncpl)).copy()
    elif bot.shape != (nlay, ncpl):
        raise ValueError(f"botm must be scalar, shape ({nlay},), or shape ({nlay}, {ncpl}).")
    layer_top = np.vstack([top[None, :], bot[:-1]])
    if not np.isfinite(layer_top).all() or not np.isfinite(bot).all():
        raise ValueError("Layer elevations must be finite.")
    with np.errstate(over="ignore"):
        thickness = layer_top - bot
    if not np.isfinite(thickness).all():
        raise ValueError("Layer thicknesses must be finite.")
    bad = np.argwhere(layer_top <= bot)
    if len(bad):
        layer, cell = bad[0]
        raise ValueError(
            f"Non-positive thickness at layer {layer}, cell {cell}; "
            "pinch-outs and vertical pass-through are not supported."
        )
    return layer_top.copy(), bot.copy()


def _line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry] if geometry.length > 0 else []
    if hasattr(geometry, "geoms"):
        return [line for part in geometry.geoms for line in _line_parts(part)]
    return []


def _straight_face(shared, centres, pair, tolerance):
    """Combine contiguous collinear pieces, rejecting bent/disconnected faces."""
    parts = _line_parts(shared)
    if not parts:
        return None  # Corner-only contact.
    longest = max(parts, key=lambda part: part.length)
    xy = np.asarray(longest.coords)[:, :2]
    origin = xy[0]
    direction = xy[-1] - origin
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError(f"Unsupported shared face between cells {pair}.")
    direction /= norm
    normal = np.array([-direction[1], direction[0]])
    intervals = []
    for part in parts:
        offsets = np.asarray(part.coords)[:, :2] - origin
        if np.max(np.abs(offsets @ normal)) > tolerance:
            raise ValueError(
                f"Noncollinear shared face between cells {pair}: the cells meet along a "
                "bent line, which breaks the straight, perpendicular face that CVFD "
                "connections assume (MF6 DISV would silently use only its first edge). "
                "In vorflow grids this comes from cells cut along a barrier; check the "
                "barrier warnings from generate()."
            )
        projected = offsets @ direction
        intervals.append((projected.min(), projected.max()))
    intervals.sort()
    end = intervals[0][1]
    for start, stop in intervals[1:]:
        if start > end + tolerance:
            raise ValueError(f"Disconnected shared face between cells {pair}.")
        end = max(end, stop)
    signed = (centres - origin) @ normal
    if signed[0] * signed[1] >= 0 or not np.isfinite(signed).all():
        raise ValueError(f"Centres must lie on opposite sides of the shared face for cells {pair}.")
    delta = centres[1] - centres[0]
    angle = np.degrees(np.arccos(np.clip(abs(delta @ normal) / np.linalg.norm(delta), 0, 1)))
    outward = normal if signed[1] > 0 else -normal  # From cell_1 towards cell_2.
    return {
        "cell_id_1": pair[0], "cell_id_2": pair[1],
        "shared_length": sum(part.length for part in parts),
        "distance_1": abs(signed[0]), "distance_2": abs(signed[1]),
        "ortho_error": float(angle),
        "normal_x": float(outward[0]), "normal_y": float(outward[1]),
    }


def _prepare_grid(gdf, top, botm, nlay):
    planar, area, faces, _ = _planar_faces(gdf)
    layer_top, bot = _layer_arrays(top, botm, nlay, len(gdf))
    ncpl = len(gdf)
    mapping = pd.DataFrame({
        "model_node": np.arange(nlay * ncpl),
        "layer": np.repeat(np.arange(nlay), ncpl),
        "cell_id": np.tile(np.arange(ncpl), nlay),
        "orig_index": np.tile(gdf.index.to_numpy(), nlay),
        "node_id": np.tile(planar["node_id"], nlay),
    })
    if "zone_id" in gdf:
        mapping["zone_id"] = np.tile(gdf["zone_id"].to_numpy(), nlay)
    return _LayeredGrid(planar, layer_top, bot, area, mapping, faces)


FACE_COLUMNS = ("cell_id_1", "cell_id_2", "shared_length", "distance_1", "distance_2", "ortho_error",
                "normal_x", "normal_y")


def _simple_faces(shared, c1, c2, tolerance):
    """Faces ``_face_records`` handles in bulk: one straight segment with the centres on either side.

    Anything else (several pieces, a bent line, a corner contact) goes
    through ``_straight_face``, which accepts or rejects it with a message.
    """
    ok = (shapely.get_type_id(shared) == shapely.GeometryType.LINESTRING) & (shapely.length(shared) > 0)
    xy, index = shapely.get_coordinates(shared, return_index=True)
    count = np.bincount(index, minlength=len(shared))
    first = np.cumsum(count) - count
    has = count > 0
    start = np.full((len(shared), 2), np.nan)
    end = np.full((len(shared), 2), np.nan)
    start[has], end[has] = xy[first[has]], xy[(first + count - 1)[has]]
    direction = end - start
    norm = np.hypot(direction[:, 0], direction[:, 1])
    ok &= norm > 0
    normal = np.column_stack((-direction[:, 1], direction[:, 0])) / np.where(norm > 0, norm, 1.0)[:, None]
    off = np.abs(np.einsum("ij,ij->i", xy - start[index], normal[index]))
    ok &= np.bincount(index, weights=off > tolerance, minlength=len(shared)) == 0
    s1 = np.einsum("ij,ij->i", c1 - start, normal)
    s2 = np.einsum("ij,ij->i", c2 - start, normal)
    return ok & (s1 * s2 < 0) & np.isfinite(s1 * s2)


def _face_records(shared, c1, c2, i, j):
    """``_straight_face`` records, in bulk, for faces that passed ``_simple_faces``."""
    first = shapely.get_point(shared, 0)
    last = shapely.get_point(shared, -1)
    start, end = shapely.get_coordinates(first), shapely.get_coordinates(last)
    direction = (end - start) / np.hypot(*(end - start).T)[:, None]
    normal = np.column_stack((-direction[:, 1], direction[:, 0]))
    s1 = np.einsum("ij,ij->i", c1 - start, normal)
    s2 = np.einsum("ij,ij->i", c2 - start, normal)
    delta = c2 - c1
    cos = np.abs(np.einsum("ij,ij->i", delta, normal)) / np.hypot(*delta.T)
    outward = np.where((s2 > 0)[:, None], normal, -normal)
    return pd.DataFrame({
        "cell_id_1": i.astype(np.int64), "cell_id_2": j.astype(np.int64),
        "shared_length": shapely.length(shared), "distance_1": np.abs(s1), "distance_2": np.abs(s2),
        "ortho_error": np.degrees(np.arccos(np.clip(cos, 0, 1))),
        "normal_x": outward[:, 0], "normal_y": outward[:, 1],
    })


def _planar_faces(gdf, disv=True):
    """Validated plan geometry shared by every layer: DISV rings, areas and straight faces.

    Returns ``(planar, area, faces, tolerance)``. ``faces`` has one row per
    pair of cells sharing a straight face of positive length, with the face
    length, each centre's perpendicular distance to it, the orthogonality
    error and the unit normal pointing from ``cell_id_1`` to ``cell_id_2``.
    ``planar`` (the DISV rings) is ``None`` when ``disv`` is False.
    """
    # Local import avoids a cycle with the thin public wrappers in utils.
    from .utils import grid_to_disv

    required = {"geometry", "x", "y"}
    if not required.issubset(gdf.columns):
        raise ValueError(f"Grid missing required columns: {sorted(required.difference(gdf.columns))}.")
    if gdf.empty:
        raise ValueError("MFUSG export needs a non-empty grid.")
    geoms = gdf.geometry.to_numpy()
    polygon = shapely.get_type_id(geoms) == shapely.GeometryType.POLYGON
    bad = ~polygon | shapely.is_empty(geoms) | ~shapely.is_valid(geoms)
    if bad.any():
        raise ValueError(f"Cell {np.flatnonzero(bad)[0]} must be a valid non-empty single polygon.")
    holes = shapely.get_num_interior_rings(geoms) > 0
    if holes.any():
        raise ValueError(f"Cell {np.flatnonzero(holes)[0]} has holes, which GSF cannot represent.")
    centres = gdf[["x", "y"]].to_numpy(dtype=float)
    if not np.isfinite(centres).all():
        raise ValueError("Cell centres must be finite.")
    coords = shapely.get_coordinates(geoms)
    tolerance = 32 * np.finfo(float).eps * max(1.0, float(np.abs(coords).max()))
    # Clipping a slanted edge can leave a boundary generator a few ulps
    # outside its polygon. Accept roundoff without moving the generator.
    outside = ~shapely.dwithin(geoms, shapely.points(centres), tolerance)
    if outside.any():
        raise ValueError(f"Centre of cell {np.flatnonzero(outside)[0]} lies outside its polygon.")
    # Reuse conforming rings, not MF6 solver semantics. Limit insertion to
    # coordinate roundoff: a looser map-scale tolerance can bend very short
    # faces, making GSF footprints disagree with the numerical geometry.
    planar = grid_to_disv(gdf, tolerance=tolerance) if disv else None
    tree = shapely.STRtree(geoms)
    left, right = tree.query(geoms, predicate="intersects")
    order = np.lexsort((right, left))
    left, right = left[order], right[order]
    left, right = left[left < right], right[left < right]
    boundaries = shapely.boundary(geoms)
    overlapping = shapely.relate_pattern(geoms[left], geoms[right], "T********")
    shared_all = shapely.intersection(boundaries[left], boundaries[right])
    fast = ~overlapping & _simple_faces(shared_all, centres[left], centres[right], tolerance)
    faces = _face_records(shared_all[fast], centres[left[fast]], centres[right[fast]], left[fast], right[fast])
    records = []
    # Corner-only contacts carry no face.
    corner = ~overlapping & np.isin(shapely.get_type_id(shared_all),
                                    (shapely.GeometryType.POINT, shapely.GeometryType.MULTIPOINT))
    rest = ~fast & ~corner
    for i, j, shared, overlaps in zip(left[rest], right[rest], shared_all[rest], overlapping[rest]):
        overlap = geoms[i].intersection(geoms[j]) if overlaps else None
        if overlaps and overlap.area > 0:
            # Only forgive slivers contained within BOTH boundary roundoff
            # bands. An area-only cutoff could accept small but deep overlaps.
            for geom in (geoms[i], geoms[j]):
                if not overlap.difference(geom.boundary.buffer(tolerance)).is_empty:
                    raise ValueError(f"Overlapping cells {i} and {j}.")
            # Recover the common face as well as accepting the sliver; otherwise
            # exact boundary intersection could silently lose this connection.
            shared = shapely.intersection(
                geoms[i].boundary, geoms[j].boundary, grid_size=tolerance,
            )
            if shared.is_empty:
                raise ValueError(f"Cannot resolve roundoff overlap between cells {i} and {j}.")
        record = _straight_face(shared, centres[[i, j]], (int(i), int(j)), tolerance)
        if record is not None:
            records.append(record)
    if records:
        slow = pd.DataFrame(records, columns=list(FACE_COLUMNS))
        faces = pd.concat([faces, slow], ignore_index=True) if len(faces) else slow
    faces = faces.astype({"cell_id_1": np.int64, "cell_id_2": np.int64})
    faces = faces.sort_values(["cell_id_1", "cell_id_2"], ignore_index=True)[list(FACE_COLUMNS)]
    if len(faces):
        logger.info("Grid export: %s horizontal faces; maximum orthogonality error %.6g degrees",
                    len(faces), faces["ortho_error"].max())
    return planar, shapely.area(geoms), faces, tolerance


def _disu_arrays(grid):
    """Full-matrix USG CL12 stores each node's own half-connection length.

    The connections come from the shared DISU core (``vorflow.disu``) with
    DISV-style layered connectivity; this keeps the original USG output:
    ``ivsd=-1`` (layers share one mesh) and one layer of areas.
    """
    from ._columns import ACTIVE
    from ._connect3d import concat_links, layered_links, vertical_links
    from ._disu_export import mfusg_gridprops

    ncpl, nlay = grid.ncpl, grid.nlay
    status = np.full((nlay, ncpl), ACTIVE, dtype=np.int8)
    nodemap = np.arange(nlay * ncpl).reshape(nlay, ncpl)
    links = concat_links(
        vertical_links(status, nodemap, grid.top, grid.bot, grid.area),
        layered_links(status, nodemap, grid.faces, grid.top, grid.bot),
    )
    props = mfusg_gridprops(
        nlay * ncpl, nlay, grid.top.ravel(), grid.bot.ravel(), np.tile(grid.area, nlay),
        np.repeat(np.arange(nlay), ncpl), links,
    )
    del props["ivc"]
    props.update({
        "ivsd": -1, "area": grid.area.copy(),
        "nodelay": props["nodelay"].astype(np.int32),
        "iac": props["iac"].astype(np.int32), "ja": props["ja"].astype(np.int32),
        "node_id": grid.cell_map["node_id"].to_numpy(copy=True),
        "cell_map": grid.cell_map.copy(deep=True),
        "connectivity": grid.faces.copy(deep=True),
    })
    return props


def _write_gsf(grid, filepath):
    """Original 3D GSF: top and bottom vertex rings, all IDs one-based."""
    from ._disu_export import write_gsf

    nodes = grid.ncpl * grid.nlay
    layer, cell = np.divmod(np.arange(nodes), grid.ncpl)
    return write_gsf(filepath, grid.planar, cell, grid.top.ravel(), grid.bot.ravel(), layer, grid.nlay)
