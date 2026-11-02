"""Barrier connections of a built DISU grid, their HFB data and GNC across them."""

import warnings

import geopandas as gpd
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString, Point, box

from vorflow import disu

FAULT = LineString([(2, -1), (2, 5)])


def boxes(n, size=1.0):
    polygons = [box(i * size, j * size, (i + 1) * size, (j + 1) * size) for j in range(n) for i in range(n)]
    return gpd.GeoDataFrame({"node_id": np.arange(len(polygons)),
                             "x": [p.centroid.x for p in polygons],
                             "y": [p.centroid.y for p in polygons]}, geometry=polygons)


def spliced():
    """Three 4 x 4 layers over three layers refined to 0.5 m in [1, 3] x [1, 3]."""
    base = boxes(4)
    pit = disu.splice(base, boxes(8, 0.5), box(1, 1, 3, 3))
    return disu.DisuGrid([base] * 3 + [pit] * 3, 10, [8, 6, 4, 2, 0, -2]).build()


def dipping():
    """4 x 4 columns, three layers dipping along x, so links across x = 2 are staggered."""
    grid = boxes(4)
    x = grid.x.to_numpy()
    return disu.DisuGrid(grid, 10.0 - 0.6 * x, [7.0 - 0.6 * x, 4.0 - 0.6 * x, 1.0 - 0.6 * x]).build()


def cells(d):
    c = d.cellmap[d.cellmap.node >= 0].sort_values("node")
    return np.column_stack((c.x, c.y, (c.top + c.bot) / 2)), c.layer.to_numpy()


def crossing(d, x0=2.0):
    """Horizontal links whose two centres lie on opposite sides of x = x0."""
    xyz, _ = cells(d)
    n, m = d.links["n"], d.links["m"]
    return np.flatnonzero((d.links["ihc"] != 0) & ((xyz[n, 0] - x0) * (xyz[m, 0] - x0) < 0))


def test_vertical_trace_finds_the_faces_on_it_in_every_layer():
    d = spliced()
    faces = d.barrier_faces(FAULT)
    _, layer = cells(d)
    np.testing.assert_array_equal(np.sort(faces.links), crossing(d))
    np.testing.assert_array_equal(np.bincount(layer[faces.n]), [4, 4, 4, 6, 6, 6])
    assert len(faces) == 30 and faces.on_trace.all() and (faces.factor == 1).all()
    np.testing.assert_array_equal(faces.pairs, np.column_stack((faces.n, faces.m)))


def test_one_trace_per_layer_limits_the_barrier_to_those_layers():
    d = spliced()
    faces = d.barrier_faces([None] * 3 + [FAULT] * 3)
    _, layer = cells(d)
    assert len(faces) == 18 and set(layer[faces.n]) == {3, 4, 5}


def test_staggered_links_across_a_dipping_stack_are_cut():
    d = dipping()
    faces = d.barrier_faces(FAULT)
    _, layer = cells(d)
    np.testing.assert_array_equal(np.sort(faces.links), crossing(d))
    assert (layer[faces.n] != layer[faces.m]).any() and faces.on_trace.all()


def test_trace_off_the_faces_cuts_the_nearest_faces_and_is_watertight():
    d = disu.DisuGrid(boxes(6), 1.0, [0.0]).build()
    trace = LineString([(-0.1, 0.35), (6.1, 4.9)])
    faces = d.barrier_faces(trace)
    assert len(faces) and not faces.on_trace.any()
    xyz, _ = cells(d)
    side = np.sign((trace.coords[1][0] - trace.coords[0][0]) * (xyz[:, 1] - trace.coords[0][1])
                   - (trace.coords[1][1] - trace.coords[0][1]) * (xyz[:, 0] - trace.coords[0][0]))
    keep = np.ones(len(d.links["n"]), dtype=bool)
    keep[faces.links] = False
    n, m = d.links["n"][keep], d.links["m"][keep]
    graph = coo_matrix((np.ones(len(n)), (n, m)), shape=(d.nodes, d.nodes))
    count, label = connected_components(graph, directed=False)
    assert count == 2
    assert len(set(label[side > 0])) == 1 and len(set(label[side < 0])) == 1


def test_hfb_package_data():
    faces = spliced().barrier_faces(FAULT)
    mf6 = faces.get_gridprops_mf6_hfb(0.5)
    assert mf6["maxhfb"] == 30 and len(mf6["stress_period_data"][0]) == 30
    assert mf6["stress_period_data"][0][0] == ((int(faces.n[0]),), (int(faces.m[0]),), 0.5)
    hydchr = np.linspace(0.0, 1.0, 30)
    usg = faces.get_gridprops_mfusg_hfb(hydchr)
    assert usg["nhfbnp"] == 30
    assert usg["hfb_data"][-1] == [int(faces.n[-1]), int(faces.m[-1]), 1.0]
    assert faces.get_gridprops_mf6_hfb(-0.1)["stress_period_data"][0][0][2] == -0.1


@pytest.mark.parametrize("hydchr", [np.nan, np.inf, np.ones(3), "low"])
def test_invalid_hydchr_is_rejected(hydchr):
    with pytest.raises(ValueError, match="hydchr"):
        spliced().barrier_faces(FAULT).get_gridprops_mf6_hfb(hydchr)


def test_trace_outside_the_grid_gives_no_package():
    faces = spliced().barrier_faces(LineString([(10, 0), (10, 4)]))
    assert len(faces) == 0
    assert faces.get_gridprops_mf6_hfb(0.0) is None and faces.get_gridprops_mfusg_hfb(0.0) is None


@pytest.mark.parametrize("trace", [box(0, 0, 1, 1), Point(1, 1), LineString(), [FAULT] * 5, 3.0])
def test_invalid_trace_is_rejected(trace):
    with pytest.raises(ValueError, match="trace"):
        spliced().barrier_faces(trace)


def exact_flows(d, faces, heads, gradient, hydchr):
    """Two-point flows with HFB series conductance, GNC-corrected flows and exact flows (K = 1)."""
    xyz, _ = cells(d)
    links = d.links
    n, m = links["n"], links["m"]
    area = np.asarray(links["fahl"], dtype=float)
    conductance = area / (links["cl_n"] + links["cl_m"])
    cut = np.zeros(len(n), dtype=bool)
    cut[faces.links] = True
    conductance = np.where(cut, 1 / (1 / conductance + 1 / (hydchr * area)), conductance)
    normal = np.column_stack((np.cos(np.radians(links["angle"])), np.sin(np.radians(links["angle"])), np.zeros(len(n))))
    vertical = links["ihc"] == 0
    normal[vertical] = 0.0
    normal[vertical, 2] = np.sign(xyz[m[vertical], 2] - xyz[n[vertical], 2])
    exact = -(normal @ gradient) * area

    def corrected(result):
        q = conductance * (heads[n] - heads[m])
        for r in result.records:
            extra = conductance[r["link"]] * np.dot(r["weights"], heads[r["contributors"]] - heads[r["n"]])
            q[r["link"]] += extra if r["n"] == n[r["link"]] else -extra
        return q
    return exact, corrected


@pytest.mark.parametrize("grid, plain_crosses, unsupported_links", [(spliced, True, 8), (dipping, False, 0)])
def test_gnc_does_not_interpolate_across_a_barrier(grid, plain_crosses, unsupported_links):
    # Spliced: corner ghosts beside the barrier lose their only support and keep standard conductance.
    # Dipping: ghosts use their own column, so they never reached across the barrier.
    d = grid()
    faces = d.barrier_faces(FAULT)
    xyz, _ = cells(d)
    gradient, hydchr = np.array([-1.0, 0.5, -0.3]), 0.5
    heads = 50 + xyz @ gradient - (xyz[:, 0] > 2) / hydchr  # q_x = 1 makes a jump of 1 / hydchr
    exact, corrected = exact_flows(d, faces, heads, gradient, hydchr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        plain = d.estimate_gnc(connections="all")
        aware = d.estimate_gnc(connections="all", barriers=faces.pairs)
    assert aware.records
    for r in aware.records:
        assert all(tuple(sorted((r["n"], j))) not in set(map(tuple, np.sort(faces.pairs, axis=1)))
                   for j in r["contributors"])
    error = abs(corrected(aware) - exact)
    unsupported = [u["link"] for u in aware.unsupported]
    assert len(unsupported) == unsupported_links
    assert np.all(np.delete(error, unsupported) < 1e-9)
    assert (abs(corrected(plain) - exact).max() > 0.1) == plain_crosses


def test_invalid_barrier_pairs_are_rejected():
    with pytest.raises(ValueError, match="barriers"):
        spliced().estimate_gnc(barriers=[(0, 0)])
