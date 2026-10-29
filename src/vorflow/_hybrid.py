"""Fill a predefined polygon hole in an exterior Voronoi mesh with triangles."""

from __future__ import annotations

import logging
import math

import geopandas as gpd
import gmsh
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from ._mfusg import _prepare_grid
from .engine import _activate_size_fields
from .fields import ConstantField
from ._barrier import BarrierMeshError
from ._features import is_embedded, row_bool

logger = logging.getLogger(__name__)


def register_subregion(cm, geometry, resolution):
    """Register one meshing reservation, independently of geological zones."""
    if cm.raw_subregions:
        raise ValueError("Only one triangle subregion is supported.")
    if not isinstance(geometry, Polygon) or geometry.is_empty or not geometry.is_valid or geometry.interiors:
        raise ValueError("A triangle subregion must be one valid polygon without holes.")
    if isinstance(resolution, (bool, np.bool_)) or resolution is None or not math.isfinite(float(resolution)) or resolution <= 0:
        raise ValueError("A triangle subregion requires a positive finite resolution.")
    if geometry.buffer(-0.25 * float(resolution)).is_empty:
        raise ValueError("Triangle subregion is too narrow; enlarge the buffer or refine the mesh.")
    cm.raw_subregions.append({"geometry": geometry, "lc": float(resolution)})


def prepare_reservations(cm):
    """Add the perimeter as an ordinary exact barrier in the working copy."""
    embedded = [row["geometry"] for row in cm.raw_polygons if row.get("embed", True)]
    if not embedded:
        raise ValueError("Triangle subregions need an embedded model domain.")
    domain = unary_union(embedded)
    rows = []
    used = {row["line_id"] for row in cm.raw_lines}
    for reservation in cm.raw_subregions:
        requested = reservation["geometry"].intersection(domain)
        if not isinstance(requested, Polygon) or requested.is_empty or requested.interiors:
            raise ValueError("Triangle subregion must meet the domain in one polygon without holes.")
        if domain.difference(requested).area <= 0:
            raise ValueError("Triangle subregion must leave an exterior meshing domain.")
        name = "__vorflow_triangle_interface__"
        while name in used:
            name += "_"
        used.add(name)
        cm.add_line(LineString(requested.exterior.coords), line_id=name,
                    resolution=reservation["lc"], is_barrier=True, simplify_tolerance=0)
        rows.append({**reservation, "geometry": requested, "interface_id": name})
    cm.clean_subregions = gpd.GeoDataFrame(rows, crs=cm.crs)


def first_pass_features(polygons, lines, points):
    """Let the circle barrier own the perimeter; defer its interior faults."""
    reservations = polygons.attrs.get("triangle_subregions", ())
    if not reservations:
        return lines, points, None, (), set()
    if len(reservations) != 1:
        raise ValueError("Only one triangle subregion is supported.")
    reservation = reservations[0]
    hole = reservation["geometry"]
    tolerance = 64 * np.finfo(float).eps * max(1., *map(abs, hole.bounds))
    owned, faults = set(), []
    for index, row in lines.iterrows():
        if row["line_id"] == reservation["interface_id"] or not is_embedded(row):
            continue
        if row_bool(row, "is_barrier"):
            if not hole.buffer(tolerance).covers(row.geometry):
                raise ValueError("A fault lies outside the triangle subregion; enlarge its buffer.")
            owned.add(int(index))
            faults.extend(_parts(row.geometry))
        elif row.geometry.intersection(hole).length > tolerance:
            raise ValueError("Embedded non-barrier lines in a triangle subregion are not yet supported.")
    for _, row in points.iterrows():
        if is_embedded(row) and hole.buffer(tolerance).covers(row.geometry):
            raise ValueError("Embedded point features in a triangle subregion are not yet supported.")
    if not faults:
        raise ValueError("A triangle subregion requires a fault inside its perimeter.")
    return lines.loc[~lines.index.isin(owned)].copy(), points, reservation, tuple(faults), owned


def finish_hybrid(tessellator, construction):
    """Remove complete interior cells, then fill the same closed-barrier hole."""
    mg = tessellator.mg
    reservation = mg.triangle_subregion
    hole = reservation["geometry"]
    tolerance = 64 * np.finfo(float).eps * max(1., *map(abs, construction.total_bounds))
    limits = 8 * tolerance * np.maximum(construction.length.to_numpy(), reservation["lc"])
    inside_area = construction.geometry.intersection(hole).area.to_numpy()
    outside_area = construction.geometry.difference(hole).area.to_numpy()
    crossing = (inside_area > limits) & (outside_area > limits)
    if crossing.any():
        raise BarrierMeshError("Triangle perimeter crosses a whole Voronoi cell; refusing clipping or cell repair.")
    if tessellator.n_exact_barrier_cuts or tessellator.n_exact_barrier_mirrors or tessellator.n_barrier_mirrors:
        raise BarrierMeshError("Triangle perimeter required the barrier safety net; whole cells are required.")
    # A cell at a domain contact may have its generator on the outer model
    # boundary, which is also the reservation boundary. Classify complete
    # polygons instead of excluding those valid boundary-centred cells.
    inside = (outside_area <= limits) & (inside_area > limits)
    removed = construction.loc[inside]
    exterior = construction.loc[~inside].copy()
    error = unary_union(removed.geometry).symmetric_difference(hole).area
    domain = unary_union(tessellator._embedded_polygons().geometry)
    if error > 8 * tolerance * (hole.length + domain.length):
        raise BarrierMeshError("Whole interior cells do not cover the requested triangle hole; refusing a jagged or shifted footprint.")
    if exterior.empty or removed.empty:
        raise ValueError("A triangle subregion needs whole cells on both sides of its perimeter.")
    site_ids = set(map(int, np.asarray(mg.node_tags)[mg.node_is_site]))
    exterior["source_x"], exterior["source_y"] = exterior.x, exterior.y
    exterior["centre_adjusted"] = False
    exterior["source_node_id"] = exterior.node_id.astype(object)
    exterior["generator_origin"] = ["barrier_site" if int(tag) in site_ids else "gmsh" for tag in exterior.node_id]
    tessellator.construction_grid = construction.copy()
    tessellator.exterior_grid = exterior.copy()
    mixed, realised = build_hybrid_grid(
        exterior, hole, reservation["lc"], mg.triangle_faults,
        tessellator._embedded_polygons(), domain,
        first_triangle_id=int(construction.node_id.max()) + 1,
    )
    retained = mixed.loc[mixed.cell_type == "voronoi"]
    if not np.array_equal(retained.node_id, exterior.node_id) or not np.array_equal(
        retained[["x", "y"]].to_numpy(), exterior[["x", "y"]].to_numpy()
    ):
        raise BarrierMeshError("Triangle pass changed an exterior generator or node ID.")
    changes = np.asarray([a.symmetric_difference(b).area for a, b in zip(retained.geometry, exterior.geometry)])
    if (changes > 8 * tolerance * np.maximum(exterior.length.to_numpy(), reservation["lc"])).any():
        raise BarrierMeshError("Triangle pass changed a whole exterior cell; refusing geometry repair.")
    tessellator.realised_triangle_footprint = realised
    tessellator.hybrid_report = {
        "construction_cells": len(construction), "exterior_cells": len(exterior),
        "removed_cells": len(removed), "removed_node_ids": tuple(map(int, removed.node_id)),
        "interface_error_area": float(error), "cell_clip_area": 0.,
        "max_exterior_change_area": float(changes.max()),
        "barrier_mirrors": 0, "barrier_cuts": 0,
    }
    return mixed


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry] if geometry.length > 0 else []
    if hasattr(geometry, "geoms"):
        return [part for child in geometry.geoms for part in _parts(child)]
    return []


def _ring_with_endpoints(hole, faults, tolerance):
    """Node the hole perimeter at fault ends that touch it."""
    ring = list(hole.exterior.coords)[:-1]
    ends = [
        Point(coord) for line in faults for coord in (line.coords[0], line.coords[-1])
    ]
    result = []
    for a, b in zip(ring, ring[1:] + ring[:1]):
        result.append(a)
        segment = LineString([a, b])
        inserts = []
        for endpoint in ends:
            if segment.distance(endpoint) <= tolerance:
                projected = segment.project(endpoint)
                if tolerance < projected < segment.length - tolerance:
                    inserts.append(
                        (projected, segment.interpolate(projected).coords[0])
                    )
        result.extend(coord for _, coord in sorted(set(inserts)))
    return result


def _interface_ring(hole, exterior, faults, resolution, tolerance):
    """Node the requested perimeter at exterior cell faces and fault ends."""
    # The exterior faces already discretise the interface. An independent
    # segmentize(hole, resolution) interleaves a second set of points and can
    # halve the edge lengths before the size field ever runs.
    # Square-cap buffers can contain redundant collinear vertices at the
    # original line ends. They are not corners or interface endpoints.
    scale = max(1.0, *map(abs, hole.bounds))
    shape = hole.simplify(64 * np.finfo(float).eps * scale)
    ring = _ring_with_endpoints(shape, faults, tolerance)
    # Roundoff clipping vertices are not additional interface constraints.
    clean = shapely.simplify(exterior.geometry.to_numpy(), 64 * np.finfo(float).eps * scale)
    candidates = shapely.get_coordinates(clean)
    candidates = candidates[
        shapely.dwithin(hole.boundary, shapely.points(candidates), tolerance)
    ]
    result = []
    for a, b in zip(ring, ring[1:] + ring[:1]):
        a, b = np.asarray(a), np.asarray(b)
        delta = b - a
        length = np.linalg.norm(delta)
        projected = ((candidates - a) @ delta) / length**2
        distances = np.abs(_cross2(delta, candidates - a)) / length
        selected = (
            (distances <= tolerance)
            & (projected > tolerance / length)
            & (projected < 1 - tolerance / length)
        )
        extra = candidates[selected]
        extra = extra[np.argsort((extra - a) @ delta)]
        result.extend([tuple(a), *map(tuple, extra)])
    cleaned = []
    for xy in result:
        if not cleaned or math.dist(cleaned[-1], xy) > tolerance:
            cleaned.append(xy)
    if len(cleaned) > 1 and math.dist(cleaned[0], cleaned[-1]) <= tolerance:
        cleaned.pop()
    return cleaned


def _triangle_elements(hole, faults, resolution, tolerance, exterior):
    """Mesh the realised hole and embed noded fault segments as Gmsh edges."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("vorflow_triangle_subregion")
        ring = _interface_ring(hole, exterior, faults, resolution, tolerance)
        points = {}
        origin = np.asarray(hole.bounds[:2])

        def point_tag(xy):
            xy = (float(xy[0]), float(xy[1]))
            for old, tag in points.items():
                if math.dist(old, xy) <= tolerance:
                    return tag
            local = np.asarray(xy) - origin
            tag = gmsh.model.geo.addPoint(*local, 0)
            points[xy] = tag
            return tag

        boundary_points = [point_tag(xy) for xy in ring]
        if len(set(boundary_points)) < 3:
            raise ValueError(
                "The realised triangle hole has fewer than three distinct vertices."
            )
        boundary_lines = [
            gmsh.model.geo.addLine(a, b)
            for a, b in zip(boundary_points, boundary_points[1:] + boundary_points[:1])
            if a != b
        ]
        loop = gmsh.model.geo.addCurveLoop(boundary_lines)
        surface = gmsh.model.geo.addPlaneSurface([loop])
        edge_tags = []
        # Unary union nodes branch intersections and shared vertices before
        # making any Gmsh curves, so every junction is one mesh point.
        noded = unary_union(faults)
        for part in _parts(noded):
            coords = list(part.coords)
            tags = [point_tag(xy) for xy in coords]
            for a, b in zip(tags, tags[1:]):
                if a != b:
                    edge_tags.append(gmsh.model.geo.addLine(a, b))
        gmsh.model.geo.synchronize()
        for curve, a, b in zip(boundary_lines, ring, ring[1:] + ring[:1]):
            # Preserve each existing face endpoint. Subdivide long faces on
            # their own interval instead of adding an independent sampling.
            segments = max(1, math.ceil((math.dist(a, b) - tolerance) / resolution))
            gmsh.model.mesh.setTransfiniteCurve(curve, segments + 1)
        if edge_tags:
            gmsh.model.mesh.embed(1, edge_tags, 2, surface)
        size_field = ConstantField(resolution).create(
            gmsh,
            tags_dict={},
            background_lc=resolution,
        )
        # Use the same field-only policy as the exterior pass. Gmsh starts
        # a fresh model here, so the first pass's options cannot be assumed.
        _activate_size_fields([size_field])
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Smoothing", 0)
        gmsh.model.mesh.generate(2)
        # Improve interior triangles while Gmsh keeps the prescribed curves
        # and interface endpoints constrained.
        gmsh.model.mesh.optimize("Relocate2D")
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(coords, dtype=float).reshape((-1, 3))
        coordinates = {int(tag): xy[:2] + origin for tag, xy in zip(node_tags, xyz)}
        for xy, geometric_tag in points.items():
            mesh_tags, _, _ = gmsh.model.mesh.getNodes(0, geometric_tag)
            for mesh_tag in mesh_tags:
                coordinates[int(mesh_tag)] = np.asarray(xy)
        element_types, element_tags, element_nodes = gmsh.model.mesh.getElements(
            2, surface
        )
        triangles = []
        element_ids = []
        for element_type, ids, flat in zip(element_types, element_tags, element_nodes):
            _, _, _, count, _, _ = gmsh.model.mesh.getElementProperties(element_type)
            if count != 3:
                raise ValueError(
                    "Hybrid subregion requires first-order Gmsh triangles."
                )
            for element_id, tags in zip(ids, np.asarray(flat).reshape((-1, 3))):
                triangle = Polygon([coordinates[int(tag)] for tag in tags])
                if not triangle.is_valid or triangle.area <= tolerance * tolerance:
                    raise ValueError(
                        "The triangle subregion contains a degenerate cell."
                    )
                triangles.append(triangle)
                element_ids.append(int(element_id))
        # The constrained edges must represent the full original fault.
        element_edges = []
        for triangle in triangles:
            vertices = np.asarray(triangle.exterior.coords) - origin
            element_edges.extend(
                LineString([a, b]) for a, b in zip(vertices, vertices[1:])
            )
        edge_union = unary_union(element_edges)
        for fault in faults:
            local_fault = LineString(np.asarray(fault.coords) - origin)
            if local_fault.difference(edge_union.buffer(tolerance)).length > tolerance:
                raise ValueError("The fault was not retained as triangle edges.")
        return triangles, element_ids
    finally:
        gmsh.finalize()


def _minimum_angle(triangle):
    coordinates = np.asarray(triangle.exterior.coords)[:3]
    lengths = np.linalg.norm(coordinates - np.roll(coordinates, 1, axis=0), axis=1)
    angles = []
    for opposite, adjacent_a, adjacent_b in (
        (lengths[0], lengths[1], lengths[2]),
        (lengths[1], lengths[2], lengths[0]),
        (lengths[2], lengths[0], lengths[1]),
    ):
        cosine = (adjacent_a**2 + adjacent_b**2 - opposite**2) / (
            2 * adjacent_a * adjacent_b
        )
        angles.append(math.degrees(math.acos(float(np.clip(cosine, -1, 1)))))
    return min(angles)


def build_hybrid_grid(grid, requested, resolution, fault_lines, zones, domain, first_triangle_id=None):
    """Fill the exact, predefined hole after tessellating the exterior domain."""
    if not isinstance(requested, Polygon) or requested.interiors:
        raise ValueError("A triangle reservation must be one polygon without holes.")
    if requested.buffer(-0.25 * resolution).is_empty:
        raise ValueError(
            "Triangle subregion is too narrow for its target resolution; enlarge the buffer or refine the mesh."
        )
    if not fault_lines:
        raise ValueError("A triangle subregion requires at least one fault line.")
    faults = [part for line in fault_lines for part in _parts(line)]
    if not faults:
        raise ValueError("The triangle subregion has no nonempty fault lines.")
    scale = max(1.0, *map(abs, requested.bounds))
    tolerance = max(64 * np.finfo(float).eps * scale, resolution * 1e-6)
    hole = requested
    exterior = grid.copy()
    numerical_tolerance = 64 * np.finfo(float).eps * scale
    for row in exterior.itertuples():
        if row.geometry.distance(hole) <= tolerance:
            concavity = row.geometry.convex_hull.area - row.geometry.area
            if concavity > 8 * numerical_tolerance * max(
                row.geometry.length, resolution
            ):
                raise ValueError(
                    f"Nonconvex Voronoi interface cell {row.node_id}; refusing polygon or centre repair."
                )
    if any(polygon.interiors for polygon in exterior.geometry):
        raise ValueError(
            "A Voronoi cell wraps the entire hole; refine the exterior mesh."
        )
    for fault in faults:
        if not hole.buffer(tolerance).covers(fault):
            raise ValueError(
                "A fault lies outside the predefined triangle hole; enlarge the buffer."
            )
    outside_union = unary_union(exterior.geometry.to_numpy())
    expected_exterior = domain.difference(hole)
    area_tolerance = max(tolerance * domain.length, 1e-9 * domain.area)
    if outside_union.symmetric_difference(expected_exterior).area > area_tolerance:
        raise ValueError("Exterior Voronoi mesh does not match the predefined hole.")
    triangles, element_ids = _triangle_elements(
        hole, faults, resolution, tolerance, exterior
    )
    if not triangles:
        raise ValueError("Gmsh produced no triangles in the realised hole.")
    triangle_union = unary_union(triangles)
    area_tolerance = max(tolerance * hole.length, 1e-9 * hole.area)
    if hole.symmetric_difference(triangle_union).area > area_tolerance:
        raise ValueError("Triangle mesh leaves gaps or overlaps the realised hole.")
    if (
        sum(triangle.area for triangle in triangles) - triangle_union.area
        > area_tolerance
    ):
        raise ValueError("Triangle mesh contains overlapping cells.")
    if not domain.buffer(tolerance).covers(triangle_union):
        raise ValueError("Triangle mesh exceeds the model domain.")

    new = gpd.GeoDataFrame(
        {
            "node_id": range(
                max(int(grid["node_id"].max()) + 1, first_triangle_id or 0),
                max(int(grid["node_id"].max()) + 1, first_triangle_id or 0) + len(triangles),
            ),
            "x": [poly.centroid.x for poly in triangles],
            "y": [poly.centroid.y for poly in triangles],
            "cell_type": ["triangle"] * len(triangles),
            "source_element_id": element_ids,
            "source_voronoi_index": [pd.NA] * len(triangles),
            "minimum_angle": [_minimum_angle(poly) for poly in triangles],
        },
        geometry=triangles,
        crs=grid.crs,
    )
    new["node_id"] = new["node_id"].astype(grid["node_id"].dtype)
    new["source_x"] = new["x"]
    new["source_y"] = new["y"]
    new["centre_adjusted"] = False
    new["generator_origin"] = "triangle"
    new["source_node_id"] = pd.Series([None] * len(new), dtype=object)
    if "zone_id" in grid:
        zone_ids, priorities = [], []
        for centre in new.geometry.centroid:
            covering = zones[zones.geometry.covers(centre)]
            if covering.empty:
                raise ValueError("A triangle centre has no geological zone.")
            chosen = covering.sort_values("z_order", ascending=False).iloc[0]
            zone_ids.append(chosen["zone_id"])
            priorities.append(chosen["z_order"])
        new["zone_id"] = zone_ids
        new["z_order"] = priorities
    exterior["cell_type"] = "voronoi"
    exterior["source_element_id"] = pd.NA
    exterior["source_voronoi_index"] = exterior.index.to_numpy()
    exterior["minimum_angle"] = np.nan
    mixed = gpd.GeoDataFrame(
        pd.concat([exterior, new], ignore_index=True), crs=grid.crs
    )
    # Insert hanging interface vertices into both rings without changing the
    # fixed geometric footprint. DISU and GSF then use identical edge segments.
    from .utils import grid_to_disv
    from .tessellator import _merge_close_vertices

    # Gmsh and Qhull can represent the same prescribed corner a few ULPs
    # apart. Unify these aliases before inserting hanging vertices, avoiding
    # a zero-length spike in an otherwise valid triangle.
    mixed = _merge_close_vertices(mixed, rel_tol=32 * np.finfo(float).eps)
    planar = grid_to_disv(mixed, tolerance=32 * np.finfo(float).eps * scale)
    vertices = np.asarray(planar["vertices"])[:, 1:]
    mixed["geometry"] = [Polygon(vertices[row[4:]]) for row in planar["cell2d"]]
    # Reuse the exact MFUSG face checks. This also rejects bent, disconnected,
    # zero-distance, overlapping and outside-centre connections.
    _prepare_grid(mixed, 1.0, 0.0, 1)
    exterior_edges, triangle_counts = set(), {}
    for kind, row in zip(mixed["cell_type"], planar["cell2d"]):
        ids = row[4:]
        for a, b in zip(ids, ids[1:] + ids[:1]):
            key = tuple(sorted((a, b)))
            if kind == "voronoi":
                exterior_edges.add(key)
            else:
                triangle_counts[key] = triangle_counts.get(key, 0) + 1
    triangle_boundary = {key for key, count in triangle_counts.items() if count == 1}
    for a, b in triangle_boundary:
        edge = LineString(vertices[[a, b]])
        if edge.difference(domain.boundary.buffer(tolerance)).length > tolerance:
            if (a, b) not in exterior_edges:
                raise ValueError("Triangle and Voronoi interface edges do not match.")
    for a, b in exterior_edges:
        edge = LineString(vertices[[a, b]])
        if edge.difference(hole.boundary.buffer(tolerance)).length <= tolerance:
            if (a, b) not in triangle_boundary:
                raise ValueError("Voronoi interface edge has no matching triangle.")
    minimum_angle = float(new["minimum_angle"].min())
    if minimum_angle < 10:
        logger.warning(
            "Hybrid triangle minimum angle is %.2f degrees; review cell quality, "
            "enlarge the reservation or refine the first mesh.",
            minimum_angle,
        )
    logger.info(
        "Hybrid grid: %s exterior cells, %s triangles; minimum fault clearance %.6g; "
        "minimum triangle angle %.2f degrees",
        len(exterior),
        len(new),
        hole.boundary.distance(unary_union(faults)),
        minimum_angle,
    )
    return mixed, hole
