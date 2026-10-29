"""disu.refill: finer cells inside an outline that share the outline's vertices with the base grid."""

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box

from vorflow import disu
from vorflow._mfusg import _planar_faces
from vorflow._refill import outline_nodes
from vorflow.fields import ExponentialField, MeshField

from test_disu_multigrid import PIT, VORONOI_PIT, _boxes, linear_field_error, voronoi_pit


def _outline_vertices(grid, polygon, inside):
    geoms = grid.geometry.to_numpy()
    in_poly = shapely.area(shapely.intersection(geoms, polygon)) > 1e-9 * shapely.area(geoms)
    cells = geoms[in_poly == inside]
    nodes = outline_nodes(cells[shapely.dwithin(cells, polygon.exterior, 1e-9)], polygon.exterior, 1e-7)
    return {(round(x, 6), round(y, 6)) for x, y in nodes}


def test_refill_shares_every_outline_vertex_and_keeps_the_outside():
    base = _boxes(4, 1.0)
    grid = disu.refill(base, PIT, 0.25)
    outside = grid[grid.splice_source == "base"]
    # Base cells outside the outline are unchanged, with their node_ids.
    kept = base.set_index("node_id").loc[outside.node_id]
    assert all(a.equals(b) for a, b in zip(outside.geometry, kept.geometry))
    assert len(outside) == 12
    inside = grid[grid.splice_source == "fine"]
    assert inside.geometry.area.sum() == pytest.approx(PIT.area)
    assert len(inside) > 4 * 4 and grid.attrs["refill"]["cells"] == len(inside)
    # The new cells end exactly at the base vertices on the outline: one face per base face.
    assert _outline_vertices(grid, PIT, True) == _outline_vertices(base, PIT, False)
    assert grid.attrs["refill"]["outline_vertices"] == 8


def test_border_divisions_add_vertices_between_the_shared_ones():
    base = _boxes(4, 1.0)
    grid = disu.refill(base, PIT, 0.25, border_divisions=2)
    inner, outer = _outline_vertices(grid, PIT, True), _outline_vertices(base, PIT, False)
    assert outer < inner and len(inner) == 2 * len(outer)
    assert grid.attrs["refill"]["outline_spacing"] == pytest.approx(0.5)


def test_refill_reports_the_size_jump_between_neighbours():
    base = _boxes(4, 1.0)
    gentle = disu.refill(base, PIT, 0.25, growth_factor=1.2).attrs["refill"]["max_neighbour_area_ratio"]
    assert 1.0 <= gentle < 4.0


def test_refill_takes_size_fields_like_add_polygon():
    base = _boxes(4, 1.0)
    grid = disu.refill(base, PIT, 0.25, fields=[ExponentialField(size_min=0.25, decay_length=0.3)])
    assert _outline_vertices(grid, PIT, True) == _outline_vertices(base, PIT, False)
    assert grid.attrs["refill"]["cells"] > 4 * 4


class _NoSizeField(MeshField):
    def create(self, gmsh_api, tags_dict, background_lc, feature_lc=None):
        return None


def test_refill_rejects_bad_input():
    base = _boxes(4, 1.0)
    with pytest.raises(ValueError, match="cross the refill outline"):
        disu.refill(base, box(0.5, 0.5, 3, 3), 0.25)
    with pytest.raises(ValueError, match="without holes"):
        disu.refill(base, Polygon(PIT.exterior.coords, [box(1.5, 1.5, 2.5, 2.5).exterior.coords]), 0.25)
    with pytest.raises(ValueError, match="growth_factor"):
        disu.refill(base, PIT, 0.25, growth_factor=1.0)
    with pytest.raises(ValueError, match="MeshField"):
        disu.refill(base, PIT, 0.25, fields=[1.2])
    with pytest.raises(ValueError, match="python_size"):
        disu.refill(base, PIT, 0.25, fields=[_NoSizeField()])
    with pytest.raises(ValueError, match="border_divisions"):
        disu.refill(base, PIT, 0.25, border_divisions=0)
    with pytest.raises(ValueError, match="resolution"):
        disu.refill(base, PIT, -1.0)


@pytest.mark.slow
def test_refill_of_the_voronoi_pit_shares_all_outline_vertices():
    _, base = voronoi_pit()
    grid = disu.refill(base, VORONOI_PIT, 12)
    assert _outline_vertices(grid, VORONOI_PIT, True) == _outline_vertices(base, VORONOI_PIT, False)
    info = grid.attrs["refill"]
    assert info["outline_vertices"] == 28 and info["max_neighbour_area_ratio"] < 4
    # Every face, the outline included, is perpendicular to the line between its two generators.
    faces = _planar_faces(grid)[2]
    assert faces["ortho_error"].max() < 1e-4          # degrees; roundoff only
    assert faces["shared_length"].min() > 1e-3        # no roundoff-length faces on the outline
    # Refined at depth: faces between a coarse cell above and a fine cell below
    # are skewed but carry no connection, so the report's angle stays at roundoff.
    d = disu.DisuGrid([base, base, grid, grid], 100.0, [80.0, 60.0, 50.0, 40.0]).build()
    assert d._faces["ortho_error"].max() > 1
    assert d.diagnostics["max_ortho_error"] < 1e-4


@pytest.mark.slow
@pytest.mark.parametrize("border_divisions", [1, 2])
@pytest.mark.parametrize("dip", [0.0, 0.03])
def test_voronoi_pit_refilled_at_depth_is_exact_with_xt3d(tmp_path, dip, border_divisions):
    _, base = voronoi_pit()
    grid = disu.refill(base, VORONOI_PIT, 12, border_divisions=border_divisions)
    error, d = linear_field_error(tmp_path, [base] * 3 + [grid] * 3, dip)
    assert d.diagnostics["vertical_offset"] == 0
    assert error < 1e-6
