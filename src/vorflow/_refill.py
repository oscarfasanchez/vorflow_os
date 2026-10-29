"""Re-mesh the inside of a barrier outline so the new cells share the outline's vertices.

Pure geometry for :func:`vorflow.disu.refill`. The outline nodes are the
vertices that the base grid has on the outline. Each outline segment gets a
pair of generators mirrored across it, on circles centred at its two end
nodes (the empty-circle rule of the exact barriers, Berge et al. 2019): with
no other generator inside a circle, the segment is exactly the Voronoi face
between the pair. The inside generator of each pair is kept as a cell, the
outside one only shapes that face, and new generators fill the rest of the
polygon outside the circles.
"""

from __future__ import annotations

import numpy as np
import shapely
from scipy.spatial import cKDTree

RADIUS_FACTOR = 0.6     # circle radius / longer adjacent segment; must exceed 0.5
CLEARANCE = 1.05        # fill nodes must lie this much outside every circle
INSIDE_EPS = 1e-9


def outline_nodes(geoms, ring, tolerance):
    """Vertices of ``geoms`` on ``ring``, ordered along it (the ring's own vertices included)."""
    coords = shapely.get_coordinates(shapely.boundary(geoms))
    on_ring = shapely.distance(shapely.points(coords), ring) <= tolerance
    points = np.vstack([coords[on_ring], np.asarray(ring.coords)[:-1]])
    position = shapely.line_locate_point(ring, shapely.points(points))
    order = np.argsort(position)
    position, points = position[order], points[order]
    keep = np.r_[True, np.diff(position) > tolerance]
    if position[-1] > ring.length - tolerance:      # The start point, reached again at the end.
        keep[-1] = False
    return points[keep]


def subdivide_outline(nodes, divisions):
    """Split every outline segment into ``divisions`` equal parts."""
    if divisions == 1:
        return nodes
    nxt = np.roll(nodes, -1, axis=0)
    steps = np.arange(divisions)[None, :, None] / divisions
    return (nodes[:, None, :] + steps * (nxt - nodes)[:, None, :]).reshape(-1, 2)


def mirrored_sites(nodes, polygon):
    """One generator pair per outline segment, mirrored across it.

    Returns ``(inside, outside, centres, radii)``: the inside generators (one
    per segment), their mirrors outside the polygon, and the circle at each
    node.
    """
    nxt = np.roll(nodes, -1, axis=0)
    length = np.linalg.norm(nxt - nodes, axis=1)
    radii = RADIUS_FACTOR * np.maximum(length, np.roll(length, 1))
    r_a, r_b = radii, np.roll(radii, -1)
    # Intersections of the circles at both ends of each segment.
    along = (length ** 2 + r_a ** 2 - r_b ** 2) / (2 * length)
    offset = np.sqrt(np.maximum(r_a ** 2 - along ** 2, 0.0))
    unit = (nxt - nodes) / length[:, None]
    normal = np.column_stack([-unit[:, 1], unit[:, 0]])
    foot = nodes + along[:, None] * unit
    left, right = foot + offset[:, None] * normal, foot - offset[:, None] * normal
    left_inside = shapely.contains_xy(polygon, left[:, 0], left[:, 1])
    inside = np.where(left_inside[:, None], left, right)
    outside = np.where(left_inside[:, None], right, left)
    return inside, outside, nodes, radii


def fill_nodes(candidates, centres, radii, sites, spacing):
    """Candidate generators outside every circle and not crowding a site."""
    candidates = np.asarray(candidates, dtype=float).reshape(-1, 2)
    if not len(candidates):
        return candidates
    clear = np.ones(len(candidates), dtype=bool)
    for hits in cKDTree(candidates).query_ball_point(centres, CLEARANCE * radii):
        clear[hits] = False
    gap, _ = cKDTree(sites).query(candidates)
    return candidates[clear & (gap > 0.5 * spacing)]


def check_circles(generators, centres, radii, own):
    """Raise if a generator lies strictly inside a circle other than the two of its own segment.

    ``own`` gives, per generator, the indices of the two circles it sits on
    (-1 for fill nodes).
    """
    tree = cKDTree(generators)
    for k, hits in enumerate(tree.query_ball_point(centres, radii * (1 - INSIDE_EPS))):
        bad = [h for h in hits if k not in own[h]]
        if bad:
            x, y = generators[bad[0]]
            raise ValueError(
                f"refill: a generator at ({x:.2f}, {y:.2f}) lies inside an outline circle, so the outline "
                "faces would not be exact. Use a coarser resolution near the outline or a larger growth."
            )


def voronoi_cells(generators, polygon, keep):
    """Voronoi cells of ``generators`` clipped to ``polygon``, for the generators in ``keep``."""
    cells = shapely.voronoi_polygons(shapely.multipoints(generators), extend_to=polygon.buffer(polygon.length),
                                     ordered=True)
    cells = np.asarray(cells.geoms)[keep]
    return shapely.intersection(cells, polygon)
