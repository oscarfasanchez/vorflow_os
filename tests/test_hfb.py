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
    """Exact flows and GNC-corrected flows with HFB series conductance (K = 1); hydchr per face or one value."""
    xyz, _ = cells(d)
    links = d.links
    n, m = links["n"], links["m"]
    area = np.asarray(links["fahl"], dtype=float)
    conductance = area / (links["cl_n"] + links["cl_m"])
    barrier = np.full(len(n), np.inf)
    barrier[faces.links] = hydchr
    conductance = 1 / (1 / conductance + 1 / (barrier * area))
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


def section(ncol=12, nlay=6):
    """One row of unit columns over unit layers: a vertical x-z section."""
    polygons = [box(i, 0, i + 1, 1) for i in range(ncol)]
    grid = gpd.GeoDataFrame({"node_id": np.arange(ncol), "x": [p.centroid.x for p in polygons],
                             "y": [0.5] * ncol}, geometry=polygons)
    return disu.DisuGrid(grid, float(nlay), [float(nlay - 1 - k) for k in range(nlay)]).build()


def test_dipping_fault_has_walls_and_treads_with_the_plane_resistance():
    d = section()
    # 45 degrees: each layer's trace one column further than the layer above.
    faces = d.barrier_faces([LineString([(3 + k, -1), (3 + k, 2)]) for k in range(6)])
    xyz, _ = cells(d)
    far = xyz[:, 0] + xyz[:, 2] > 3 + 5.5
    n, m = d.links["n"], d.links["m"]
    np.testing.assert_array_equal(np.sort(faces.links), np.flatnonzero(far[n] != far[m]))
    assert len(faces) == 11 and faces.vertical.sum() == 5
    np.testing.assert_allclose(faces.factor, np.sqrt(0.5))
    # Flow normal to the fault: with the factor every connection carries the exact flow.
    normal, q, hydchr = np.array([1.0, 0.0, 1.0]) / np.sqrt(2), 0.7, 0.4
    heads = 50 - q * xyz @ normal - far * q / hydchr
    with_factor = exact_flows(d, faces, heads, -q * normal, hydchr * faces.factor)
    without = exact_flows(d, faces, heads, -q * normal, hydchr)
    assert np.abs(with_factor[1](d.estimate_gnc(connections="all")) - with_factor[0]).max() < 1e-12
    assert np.abs(without[1](d.estimate_gnc(connections="all")) - without[0]).max() > 0.1


def stacked(name):
    if name == "uneven offsets":
        xs = [2, 3, 3, 5]
        d = disu.DisuGrid(boxes(6), 4.0, [3.0, 2.0, 1.0, 0.0]).build()
        return d, [LineString([(x, -1), (x, 7)]) for x in xs], lambda x, y, k: x > np.array(xs)[k]
    if name == "oblique strike":
        d = disu.DisuGrid(boxes(8), 3.0, [2.0, 1.0, 0.0]).build()
        return (d, [LineString([(1 + k, -1), (6 + k, 9)]) for k in range(3)],
                lambda x, y, k: x > 1 + k + 0.5 * (y + 1))
    grid = boxes(8)
    x = grid.x.to_numpy()
    d = disu.DisuGrid(grid, 12 - 0.3 * x, [10 - 0.3 * x - 2 * k for k in range(4)]).build()
    return d, [LineString([(3 + k, -1), (3 + k, 9)]) for k in range(4)], lambda x, y, k: x > 3 + k


@pytest.mark.parametrize("name", ["uneven offsets", "oblique strike", "dipping layers"])
def test_stacked_traces_cut_every_connection_that_changes_side(name):
    d, traces, right = stacked(name)
    faces = d.barrier_faces(traces)
    c = d.cellmap[d.cellmap.node >= 0].sort_values("node")
    side = right(c.x.to_numpy(), c.y.to_numpy(), c.layer.to_numpy())
    n, m = d.links["n"], d.links["m"]
    np.testing.assert_array_equal(np.sort(faces.links), np.flatnonzero(side[n] != side[m]))
    assert faces.vertical.any() and np.all((faces.factor > 0) & (faces.factor < 1))


@pytest.mark.parametrize("slope", [0.6, -0.6, 0.0])
@pytest.mark.parametrize("partial", [0, 1])
@pytest.mark.parametrize("flip", [False, True])
def test_traces_of_different_length_are_paired_end_to_end(slope, partial, flip):
    # A fault that dies out along the strike in one layer: x = 3 in layer 0, x = 4 in layer 1,
    # one of them only for y in [6, 11]. Where both exist, every change of side is cut, and nothing else.
    grid = boxes(10)
    x = grid.x.to_numpy()
    d = disu.DisuGrid(grid, 10 - slope * x, [7 - slope * x, 4 - slope * x]).build()
    traces = [LineString([(3 + k, 6 if k == partial else -1), (3 + k, 11)]) for k in range(2)]
    if flip:
        traces[partial] = LineString(traces[partial].coords[::-1])
    faces = d.barrier_faces(traces)
    xyz, layer = cells(d)
    side = xyz[:, 0] > 3 + layer
    n, m = d.links["n"], d.links["m"]
    both = (xyz[n, 1] > 6.5) & (xyz[m, 1] > 6.5)
    cut = np.zeros(len(n), dtype=bool)
    cut[faces.links] = True
    assert not (both & (side[n] != side[m]) & ~cut).any()
    assert not (both & (side[n] == side[m]) & cut).any()


def sealed(d, faces, side):
    """With the cut connections removed, no cell on one side reaches the other."""
    keep = np.ones(len(d.links["n"]), dtype=bool)
    keep[faces.links] = False
    n, m = d.links["n"][keep], d.links["m"][keep]
    _, label = connected_components(coo_matrix((np.ones(len(n)), (n, m)), shape=(d.nodes,) * 2), directed=False)
    return not set(label[side]) & set(label[~side])


def test_a_trace_through_cell_centres_puts_them_on_its_left():
    d = disu.DisuGrid(boxes(5), 2.0, [1.0, 0.0]).build()
    xyz, _ = cells(d)
    for trace, west in ((LineString([(2.5, -1), (2.5, 6)]), True), (LineString([(2.5, 6), (2.5, -1)]), False)):
        faces = d.barrier_faces(trace)
        # Drawn northwards the centres on the trace join the west side, drawn southwards the east.
        side = xyz[:, 0] > 2.5 if west else xyz[:, 0] >= 2.5
        assert sealed(d, faces, side) and len(faces) == 10
        np.testing.assert_array_equal(np.sort(faces.links), np.flatnonzero(side[d.links["n"]] != side[d.links["m"]]))


def test_a_trace_that_crosses_a_connection_twice_does_not_cut_it():
    d = disu.DisuGrid(boxes(5), 1.0, [0.0]).build()
    xyz, _ = cells(d)
    faces = d.barrier_faces(LineString([(1.9, 2.4), (2.6, 2.6), (1.9, 2.8)]))
    pairs = {frozenset(map(tuple, xyz[p, :2].round(1))) for p in faces.pairs}
    assert frozenset({(2.5, 2.5), (2.5, 3.5)}) not in pairs


@pytest.mark.slow
def test_a_fault_line_meshed_without_a_barrier_is_still_sealed():
    from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator

    fault = LineString([(70, 0), (110, 120)])
    cm = ConceptualMesh()
    cm.add_polygon(box(0, 0, 200, 120), zone_id=1, resolution=10)
    cm.add_line(fault, line_id="fault", resolution=6, is_barrier=False)
    mg = MeshGenerator(background_lc=10, conceptual_mesh=cm, verbosity=0)
    mg.generate()
    grid = VoronoiTessellator(mg, cm).generate()
    d = disu.DisuGrid(grid, 2.0, [1.0, 0.0]).build()
    xyz, _ = cells(d)
    on = np.array([fault.distance(Point(p)) < 1e-6 for p in xyz[:, :2]])
    assert on.any()
    west = (110 - 70) * xyz[:, 1] - 120 * (xyz[:, 0] - 70) > 0
    assert sealed(d, d.barrier_faces(fault), west | on)


def test_uneven_offsets_give_the_local_dip():
    d, traces, _ = stacked("uneven offsets")
    faces = d.barrier_faces(traces)
    _, layer = cells(d)
    walls, treads = ~faces.vertical & (layer[faces.n] == layer[faces.m]), faces.vertical
    # Walls in layer 3 see an offset of 2 (factor 1/sqrt(5)); treads from layer 2 to 3 the same offset.
    np.testing.assert_allclose(faces.factor[walls & (layer[faces.n] == 3)], 1 / np.sqrt(5))
    np.testing.assert_allclose(faces.factor[treads & (layer[faces.n] == 2)], 2 / np.sqrt(5))
    np.testing.assert_allclose(faces.factor[walls & (layer[faces.n] == 1)], 1 / np.sqrt(1.25))


def test_dip_is_measured_across_the_strike_up_to_the_grid_edge():
    # Oblique traces that end on the grid edges, 1 m apart across the strike, through 1 m layers: 45 degrees.
    d = disu.DisuGrid(boxes(8), 3.0, [2.0, 1.0, 0.0]).build()
    shift = np.hypot(1.0, 0.5)
    faces = d.barrier_faces([LineString([(1 + k * shift, 0), (5 + k * shift, 8)]) for k in range(3)])
    assert faces.vertical.any()
    np.testing.assert_allclose(faces.factor, np.sqrt(0.5), atol=1e-9)


def test_fault_tips_close_the_step_polygon():
    d, traces, _ = stacked("uneven offsets")
    faces = d.barrier_faces([LineString([(t.coords[0][0], 1.5), (t.coords[0][0], 4.5)]) for t in traces])
    xyz, _ = cells(d)
    assert len(faces) and np.all((xyz[faces.pairs, 1] > 1) & (xyz[faces.pairs, 1] < 5))


def test_usg_export_divides_vertical_hydchr_by_the_mean_thickness():
    # MODFLOW-USG and USG-Transport multiply a vertical barrier's hydchr by the mean cell thickness
    # (gwf2hfb7u1.f); dividing by it beforehand gives hydchr * area, as in MODFLOW 6.
    d = section()
    faces = d.barrier_faces([LineString([(3 + k, -1), (3 + k, 2)]) for k in range(6)])
    mf6 = faces.get_gridprops_mf6_hfb(0.1)
    usg = faces.get_gridprops_mfusg_hfb(0.1)
    assert mf6["maxhfb"] == usg["nhfbnp"] == 11 and faces.vertical.any()
    hydchr6 = np.array([h for _, _, h in mf6["stress_period_data"][0]])
    hydchr_usg = np.array([h for _, _, h in usg["hfb_data"]])
    thickness = d._node_top - d._node_bot
    mean = (thickness[faces.n] + thickness[faces.m]) / 2
    np.testing.assert_allclose(hydchr_usg, np.where(faces.vertical, hydchr6 / mean, hydchr6))
    with pytest.raises(ValueError, match="negative"):
        faces.get_gridprops_mf6_hfb(-0.5)


def test_traces_that_differ_must_be_single_lines():
    two = LineString([(3, -1), (3, 2)]).union(LineString([(8, -1), (8, 2)]))
    with pytest.raises(ValueError, match="one LineString"):
        section(nlay=2).barrier_faces([two, LineString([(4, -1), (4, 2)])])


def test_dipping_fault_through_dipping_layers_is_exact_with_gnc():
    # Layers 0.7 thick slope 0.3 along x, so traces one column apart make a 45 degree fault plane
    # through each layer's mid-depth; links across the fault include staggered ones.
    grid = boxes(12)
    x = grid.x.to_numpy()
    top = 12 - 0.3 * x
    d = disu.DisuGrid(grid, top, [top - 0.7 * (k + 1) for k in range(6)]).build()
    faces = d.barrier_faces([LineString([(3 + k, -1), (3 + k, 13)]) for k in range(6)])
    xyz, layer = cells(d)
    plane = 3 + 12 - 0.3 * 3 - 0.35          # x + z at the top layer's mid-depth on x = 3
    far = xyz[:, 0] + xyz[:, 2] > plane
    n, m = d.links["n"], d.links["m"]
    np.testing.assert_array_equal(np.sort(faces.links), np.flatnonzero(far[n] != far[m]))
    assert (layer[faces.n] != layer[faces.m]).sum() > faces.vertical.sum()   # staggered links are cut too
    np.testing.assert_allclose(faces.factor, np.sqrt(0.5))
    normal, q, hydchr = np.array([1.0, 0.0, 1.0]) / np.sqrt(2), 0.7, 0.4
    heads = 50 - q * xyz @ normal - far * q / hydchr
    exact, corrected = exact_flows(d, faces, heads, -q * normal, hydchr * faces.factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = d.estimate_gnc(connections="all", barriers=faces.pairs)
    error = np.delete(abs(corrected(result) - exact), [u["link"] for u in result.unsupported])
    assert error.max() < 1e-9
