"""Construct a model outline with ordinary exact circle barriers.

The surrounding construction region is temporary. Its generators close the
Voronoi diagram across the physical outline; their cells never enter the model.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import affinity
from shapely.geometry import LineString, box

from ._barrier import BarrierMeshError
from ._barrier.lines import BarrierLine, node_lines
from ._features import is_embedded, is_face_line, polygon_parts


def construction_frames(polygons, lines, points, background_lc):
    """Return construction frames and the unchanged physical domain.

    Node the closed perimeters with existing features without running line
    simplification or noise removal: no physical corner may be dropped.
    Feature indices already in the clean frames remain unchanged.
    """
    embedded = polygons.loc[[is_embedded(row) for _, row in polygons.iterrows()]]
    domain = shapely.union_all(embedded.geometry.to_numpy())
    if domain.is_empty or not domain.is_valid or domain.area <= 0:
        raise ValueError("boundary_method='conforming' requires a valid embedded polygon domain.")
    h = float(background_lc)
    xmin, ymin, xmax, ymax = domain.bounds
    # This padding is only construction space, not a change to the active area.
    envelope = box(xmin - 4 * h, ymin - 4 * h, xmax + 4 * h, ymax + 4 * h)
    support = envelope.difference(domain)
    row = {name: None for name in polygons.columns}
    row.update(geometry=support, zone_id="__vorflow_boundary_support__", lc=h,
               z_order=float(embedded.z_order.min()) - 1, fields=[], embed=True,
               quad_buffer=False, quad_buffer_thickness=1, gap_wall=False)
    next_polygon = int(max(polygons.index, default=-1)) + 1
    extra = gpd.GeoDataFrame([row], index=[next_polygon], crs=polygons.crs)
    construction = gpd.GeoDataFrame(pd.concat([polygons, extra]), crs=polygons.crs)
    construction.attrs = polygons.attrs.copy()

    next_line = int(max(lines.index, default=-1)) + 1
    used = set(lines.get("line_id", []))
    rows, ids = [], []
    for polygon in polygon_parts(domain):
        for ring in [polygon.exterior, *polygon.interiors]:
            geometry = LineString(ring.coords)
            name = "__vorflow_domain_boundary__"
            while name in used:
                name += "_"
            used.add(name)
            row = {key: None for key in lines.columns}
            # Native barrier sizing evaluates the zone/line/point fields along
            # the perimeter; do not impose the finest zone on the entire ring.
            row.update(geometry=geometry, line_id=name, lc=h,
                       is_barrier=True, embed=True, quad_buffer=False,
                       quad_buffer_thickness=1, fields=[], z_order=0,
                       simplify_tolerance=0., barrier_events=[], gap_wall=False)
            rows.append(row)
            ids.append(next_line)
            next_line += 1
    extra = gpd.GeoDataFrame(rows, index=ids, crs=lines.crs)
    prepared = gpd.GeoDataFrame(pd.concat([lines, extra]), crs=lines.crs)
    prepared.attrs = lines.attrs.copy()
    origin = np.asarray(envelope.bounds[:2])
    local_domain = affinity.translate(envelope, -origin[0], -origin[1])
    barriers = [BarrierLine(index, np.asarray(row.geometry.coords)[:, :2] - origin,
                            float(row.lc) if pd.notna(row.get("lc")) else h,
                            0., source=index)
                for index, row in prepared.iterrows() if is_face_line(row)]
    standard = [affinity.translate(row.geometry, -origin[0], -origin[1])
                for _, row in prepared.iterrows()
                if is_embedded(row) and not is_face_line(row)]
    tolerance = max(1e-12, 1e-10 * max(xmax - xmin, ymax - ymin))
    node_lines(barriers, standard, local_domain, tolerance)
    for barrier in barriers:
        prepared.at[barrier.source, "geometry"] = LineString(barrier.xy + origin)
    return construction, prepared, points, domain, envelope, tuple(ids)


def retain_domain_cells(tessellator, raw):
    """Keep whole active cells, rejecting any boundary that needs a cut.

    Intersections below are measurements only. The returned polygons and
    numerical centres are the original Voronoi geometry and generators.
    """
    domain = tessellator.mg.model_domain
    tolerance = 64 * np.finfo(float).eps * max(1., *map(abs, tessellator.mg.construction_domain.bounds))
    limits = 8 * tolerance * np.maximum(raw.length.to_numpy(), 1.)
    inside = raw.geometry.intersection(domain).area.to_numpy()
    outside = raw.geometry.difference(domain).area.to_numpy()
    crossing = (inside > limits) & (outside > limits)
    if crossing.any():
        cell = raw.iloc[np.flatnonzero(crossing)[0]]
        raise BarrierMeshError(
            f"Conforming domain boundary crosses cell {cell.node_id} near ({cell.x:.6g}, {cell.y:.6g}); "
            "refusing to clip it. Refine the boundary resolution or simplify the domain outline.")
    if tessellator.n_exact_barrier_mirrors:
        raise BarrierMeshError("Conforming domain boundary required mirror repairs; refusing clipped or repaired cells.")
    keep = (inside > limits) & (outside <= limits)
    retained = raw.loc[keep].copy()
    covered = shapely.union_all(retained.geometry.to_numpy())
    error = covered.symmetric_difference(domain).area
    area_limit = 8 * tolerance * domain.length
    overlap = max(0., float(retained.area.sum()) - covered.area)
    if retained.empty or error > area_limit or overlap > area_limit:
        raise BarrierMeshError(
            "Whole conforming-boundary cells do not cover the requested domain without gaps or overlaps; "
            "refine the boundary resolution or simplify the outline. No clipping was applied.")
    concavity = retained.geometry.convex_hull.area.to_numpy() - retained.area.to_numpy()
    if (concavity > limits[keep]).any():
        raise BarrierMeshError("Conforming domain construction produced a concave cell; no clipping was applied.")
    tessellator.boundary_construction_grid = raw.copy()
    site_ids = np.asarray(tessellator.mg.node_tags)[tessellator.mg.node_is_site]
    retained["source_node_id"] = retained.node_id.astype(object)
    retained["generator_origin"] = np.where(retained.node_id.isin(site_ids), "barrier_site", "gmsh")
    tessellator.boundary_report = {
        "method": "conforming", "construction_cells": len(raw), "active_cells": len(retained),
        "discarded_cells": int((~keep).sum()),
        "discarded_node_ids": tuple(map(int, raw.loc[~keep, "node_id"])),
        "coverage_error_area": float(error), "overlap_area": float(overlap), "cell_clip_area": 0.,
    }
    return retained
