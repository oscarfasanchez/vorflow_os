"""Behavioural checks for geometric GNC and the one-ring K-aware workflow."""

import warnings

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPoint, Point, Polygon, box

from vorflow.disu import DisuGrid, splice


def plan(polygons):
    return gpd.GeoDataFrame(
        {
            "node_id": np.arange(len(polygons)),
            "x": [p.centroid.x for p in polygons],
            "y": [p.centroid.y for p in polygons],
        },
        geometry=polygons,
    )


def refined_grid(**options):
    polygons = [box(-2, 0, 0, 2), box(-2, -2, 0, 0)]
    polygons += [box(x, y, x + 1, y + 1) for y in range(-2, 2) for x in range(2)]
    return DisuGrid(plan(polygons), 1.0, [0.0], **options).build()


def sandwich():
    grids = [
        plan(
            [
                box(x / n, y / n, (x + 1) / n, (y + 1) / n)
                for y in range(n)
                for x in range(n)
            ]
        )
        for n in [2, 3, 2]
    ]
    return DisuGrid(grids, 3.0, [2.0, 1.0, 0.0]).build()


def centres(grid):
    c = grid.cellmap[grid.cellmap.node >= 0].sort_values("node")
    return np.column_stack((c.x, c.y, (c.top + c.bot) / 2))


def ghost_head(record, heads):
    return heads[record["n"]] + np.asarray(record["weights"]) @ (
        heads[record["contributors"]] - heads[record["n"]]
    )


def test_geometry_is_default_and_has_no_conductivity_dependency():
    d = refined_grid()
    a = d.estimate_gnc(connections="all")
    b = d.estimate_gnc(
        np.geomspace(1e-8, 1e8, d.nodes), method="geometric", connections="all"
    )
    assert a.records and a.records == b.records
    assert a.method == "geometric" and a.rings == 1 and not a.experimental
    xyz = centres(d)
    for gradient in np.eye(3).tolist() + [[0.71, -0.39, 0.2]]:
        heads = 31.0 + xyz @ gradient
        for record in a.records:
            np.testing.assert_allclose(
                ghost_head(record, heads),
                31.0 + np.asarray(record["target"]) @ gradient,
                atol=1e-10,
            )


@pytest.mark.parametrize("ratio", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_k_aware_reproduces_admissible_interface_heads(ratio):
    d = refined_grid()
    xyz = centres(d)
    k = np.where(xyz[:, 1] < 0, ratio, 1.0)
    result = d.estimate_gnc(k, method="k_aware", connections="all")
    assert result.records and result.experimental and result.rings == 1

    def resistance_coordinates(points):
        p = np.array(points, dtype=float, copy=True)
        p[..., 1] = np.where(p[..., 1] < 0, p[..., 1] / ratio, p[..., 1])
        return p

    for gradient in np.eye(3).tolist() + [[0.73, -0.21, 0.37]]:
        heads = 23.0 + resistance_coordinates(xyz) @ gradient
        for record in result.records:
            target_head = 23.0 + resistance_coordinates(record["target"]) @ gradient
            np.testing.assert_allclose(
                ghost_head(record, heads), target_head, atol=1e-9
            )


def test_material_boundary_changes_weight_without_solved_head_input():
    d = refined_grid()
    xyz = centres(d)
    host = int(np.flatnonzero(np.all(np.isclose(xyz[:, :2], [-1, 1]), axis=1))[0])
    other = int(np.flatnonzero(np.all(np.isclose(xyz[:, :2], [0.5, 0.5]), axis=1))[0])
    below = int(np.flatnonzero(np.all(np.isclose(xyz[:, :2], [-1, -1]), axis=1))[0])
    k = np.where(xyz[:, 1] < 0, 0.1, 1.0)
    for method, weight in [("geometric", 0.25), ("k_aware", 1 / 22)]:
        result = d.estimate_gnc(k, method=method, connections="all")
        record = next(r for r in result.records if r["n"] == host and r["m"] == other)
        assert record["contributors"] == [below]
        np.testing.assert_allclose(record["weights"], [weight], atol=1e-12)


def test_homogeneous_k_limit_matches_geometric_corrections():
    d = refined_grid()
    geometric = d.estimate_gnc(connections="all")
    aware = d.estimate_gnc(1.0, method="k_aware", connections="all")
    assert geometric.records == aware.records


@pytest.mark.parametrize("ratio", [0.2, 5.0, 9.9])
def test_contrast_below_threshold_is_weighted_geometrically(ratio):
    d = refined_grid()
    k = np.where(centres(d)[:, 1] < 0, ratio, 1.0)
    aware = d.estimate_gnc(k, method="k_aware", connections="all")
    geometric = d.estimate_gnc(connections="all")
    assert aware.records == geometric.records
    assert aware.contrast == 10.0 and geometric.contrast is None


@pytest.mark.parametrize("ratio", [10.0, 0.1])
def test_contrast_at_the_threshold_is_a_contact(ratio):
    d = refined_grid()
    k = np.where(centres(d)[:, 1] < 0, ratio, 1.0)
    aware = d.estimate_gnc(k, method="k_aware", connections="all")
    assert aware.records != d.estimate_gnc(connections="all").records


@pytest.mark.parametrize("contrast", [1.0, 2.0, 5.0])
def test_lower_contrast_treats_moderate_interface_as_contact(contrast):
    d = refined_grid()
    xyz = centres(d)
    ratio = 5.0
    k = np.where(xyz[:, 1] < 0, ratio, 1.0)
    result = d.estimate_gnc(k, method="k_aware", connections="all", contrast=contrast)
    assert result.records and result.contrast == contrast
    assert result.records != d.estimate_gnc(connections="all").records

    def resistance_coordinates(points):
        p = np.array(points, dtype=float, copy=True)
        p[..., 1] = np.where(p[..., 1] < 0, p[..., 1] / ratio, p[..., 1])
        return p

    for gradient in np.eye(3).tolist() + [[0.73, -0.21, 0.37]]:
        heads = 23.0 + resistance_coordinates(xyz) @ gradient
        for record in result.records:
            target_head = 23.0 + resistance_coordinates(record["target"]) @ gradient
            np.testing.assert_allclose(
                ghost_head(record, heads), target_head, atol=1e-9
            )


def test_geometric_ignores_contrast():
    d = refined_grid()
    assert (
        d.estimate_gnc(connections="all", contrast=1.0).records
        == d.estimate_gnc(connections="all", contrast=1e6).records
    )


@pytest.mark.parametrize("contrast", [0.5, np.nan, np.inf, True, "10", [10.0], None])
def test_invalid_contrast_fails_before_estimation(contrast):
    with pytest.raises(ValueError, match="contrast"):
        refined_grid().estimate_gnc(1.0, method="k_aware", contrast=contrast)


@pytest.mark.parametrize("method", ["geometric", "k_aware"])
def test_one_ring_sandwich_rejects_unit_sum_fit_before_export(method):
    d = sandwich()
    layer = d.cellmap[d.cellmap.node >= 0].sort_values("node").layer.to_numpy()
    k = np.array([1.0, 0.01, 1.0])[layer]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = d.estimate_gnc(k, method=method)
    if method == "k_aware":
        assert any("left without GNC" in str(w.message) for w in caught)
        assert result.corrected_connections == 16 and len(result.unsupported) == 16
        assert result.rings == 1
    else:
        assert not caught and not result.unsupported
        assert result.corrected_connections == 32 and result.rings == 2
    for record in result.records:
        assert min(record["weights"]) >= 0
        assert 1 - sum(record["weights"]) >= 1e-8 - 1e-12
    assert result.get_gridprops_mf6_gnc() is not None
    assert result.get_gridprops_mfusg_gnc() is not None


def boundary_grid():
    """Fourteen Voronoi generators, including generators clipped at the edge."""
    xy = np.array(
        [
            (0, 0),
            (2, 0),
            (4, 0),
            (4, 2),
            (4, 4),
            (2, 4),
            (0, 4),
            (0, 2),
            (0.6, 0.8),
            (1.7, 0.6),
            (3.2, 0.7),
            (0.8, 2.8),
            (2.1, 2.2),
            (3.4, 3.1),
        ]
    )
    domain = box(0, 0, 4, 4)
    polygons = [
        p.intersection(domain)
        for p in shapely.voronoi_polygons(
            MultiPoint(xy), extend_to=domain, ordered=True
        ).geoms
    ]
    grid = plan(polygons)
    grid["x"], grid["y"] = xy.T
    dip = 0.6 * xy[:, 0]
    return DisuGrid(grid, 10 - dip, [7 - dip, 4 - dip, 1 - dip]).build()


def test_geometric_fallback_recovers_boundary_flux_and_preserves_valid_records():
    d = boundary_grid()
    # Uniform K makes the unchanged K-aware path the old one-ring affine
    # construction; no heterogeneous transmission assumption enters this check.
    with pytest.warns(RuntimeWarning, match="left without GNC"):
        original = d.estimate_gnc(1.0, method="k_aware", connections="all")
    assert any(u["reason"] == "ghost outside host prism" for u in original.unsupported)
    revised = d.estimate_gnc(connections="all", strict=True)
    assert not revised.unsupported and revised.diagnostics["support_union_links"] > 0
    by_host = {(r["link"], r["n"]): r for r in revised.records}
    assert all(r == by_host[(r["link"], r["n"])] for r in original.records)
    cm = d.cellmap[d.cellmap.node >= 0].sort_values("node")
    xyz = centres(d)
    gradient = np.array([0.2, -0.1, 0.7])
    heads = 31 + xyz @ gradient
    links = d.links
    a, b = links["n"], links["m"]
    area = links["fahl"]
    conductance = area / (links["cl_n"] + links["cl_m"])
    flux = conductance * (heads[b] - heads[a])
    normals = np.column_stack(
        [
            np.cos(np.radians(links["angle"])),
            np.sin(np.radians(links["angle"])),
            np.zeros(len(a)),
        ]
    )
    vertical = links["ihc"] == 0
    normals[vertical] = [0, 0, -1]
    outside_host = False
    for r in revised.records:
        shift = ghost_head(r, heads) - heads[r["n"]]
        flux[r["link"]] += (
            (-1 if r["n"] == a[r["link"]] else 1) * conductance[r["link"]] * shift
        )

        def covers(j):
            polygon = d._grids[int(cm.iloc[j].grid)].geometry.iloc[int(cm.iloc[j].icpl)]
            return (
                polygon.buffer(1e-8).covers(Point(*r["target"][:2]))
                and cm.iloc[j].bot - 1e-8 <= r["target"][2] <= cm.iloc[j].top + 1e-8
            )

        outside_host |= not covers(r["n"])
        assert any(covers(j) for j in [r["n"], *r["contributors"]])
    assert outside_host
    np.testing.assert_allclose(flux, area * (normals @ gradient), atol=1e-8)


def test_second_ring_paths_respect_blocked_and_barrier_connections():
    d = sandwich()
    original = d.estimate_gnc()
    record = next(r for r in original.records if any(len(p) == 2 for p in r["paths"]))
    path = next(p for p in record["paths"] if len(p) == 2)
    pair = tuple(sorted((path[0], path[1])))
    for option in ("blocked_pairs", "barriers"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = d.estimate_gnc(**{option: [pair]})
        for r in result.records:
            for support in r["paths"]:
                route = [r["n"], *support]
                assert all(
                    tuple(sorted(edge)) != pair for edge in zip(route, route[1:])
                )
            assert all(tuple(sorted((r["n"], j))) != pair for j in r["contributors"])


def test_roundoff_at_a_spliced_hanging_vertex_keeps_the_paired_face():
    base = plan([box(x, y, x + 1, y + 1) for y in range(4) for x in range(4)])
    fine = plan(
        [
            box(x / 2, y / 2, (x + 1) / 2, (y + 1) / 2)
            for y in range(8)
            for x in range(8)
        ]
    )
    grid = splice(base, fine, box(1, 1, 3, 3))
    dip = 0.6 * grid.x.to_numpy()
    d = DisuGrid(grid, 10 - dip, [7 - dip, 4 - dip, 1 - dip]).build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        before = d.estimate_gnc(1.0, method="k_aware", connections="all")
    cm = d.cellmap[d.cellmap.node >= 0].sort_values("node")
    counts = {}
    for r in before.records:
        counts[r["link"]] = counts.get(r["link"], 0) + 1
    for r in before.records:
        a, b = int(cm.iloc[r["n"]].icpl), int(cm.iloc[r["m"]].icpl)
        if (
            counts[r["link"]] == 2
            and grid.iloc[a].splice_source != grid.iloc[b].splice_source
        ):
            break
    else:
        pytest.fail("Fixture needs a paired correction on the splice outline")
    polygon = d._grids[0].geometry.iloc[a]
    other = d._grids[0].geometry.iloc[b]
    midpoint = polygon.boundary.intersection(other.boundary).centroid
    xy = np.array(midpoint.coords[0])
    direction = np.array(polygon.centroid.coords[0]) - xy
    inset = xy + 5e-14 * direction / np.linalg.norm(direction)
    ring = list(polygon.exterior.coords)
    for i, (start, end) in enumerate(zip(ring, ring[1:])):
        if shapely.LineString([start, end]).distance(midpoint) < 1e-14:
            ring.insert(i + 1, tuple(inset))
            break
    # Stored outlines can differ from the built contact by roundoff, as at a
    # hanging vertex on independently meshed splice outlines.
    d._grids[0].at[a, "geometry"] = Polygon(ring)
    assert (
        d._grids[0].geometry.iloc[a].boundary.intersection(other.boundary).length
        < 1e-10
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        after = d.estimate_gnc(1.0, method="k_aware", connections="all")
    assert not any(u["link"] == r["link"] for u in after.unsupported)
    assert sum(s["link"] == r["link"] for s in after.records) == 2


def test_scope_threshold_and_existing_exports_are_explicit():
    d = refined_grid()
    before = d.get_gridprops_mf6_disu()
    assert (
        not d.estimate_gnc().records
    )  # Single-layer in-plane offsets are outside 3D scope.
    assert d.estimate_gnc(connections="all").records
    assert not d.estimate_gnc(connections="all", min_offset=1e6).records
    after = d.get_gridprops_mf6_disu()
    assert before.keys() == after.keys()
    for key in before:
        np.testing.assert_equal(before[key], after[key])


def test_blocked_face_is_not_used_for_correction_or_support():
    d = refined_grid()
    original = d.estimate_gnc(connections="all")
    record = original.records[0]
    pair = tuple(sorted((record["n"], record["m"])))
    with pytest.warns(RuntimeWarning):
        result = d.estimate_gnc(connections="all", blocked_pairs=[pair])
    assert any(tuple(sorted((r["n"], r["m"]))) == pair for r in result.unsupported)
    for r in result.records:
        assert tuple(sorted((r["n"], r["m"]))) != pair
        assert all(tuple(sorted((r["n"], j))) != pair for j in r["contributors"])
    with pytest.raises(ValueError, match="lack support"):
        d.estimate_gnc(connections="all", blocked_pairs=[pair], strict=True)


@pytest.mark.parametrize(
    "k",
    [
        None,
        0.0,
        -1.0,
        np.nan,
        np.inf,
        np.ones((10, 3)),
        np.ones(9),
        np.ones(10, complex),
    ],
)
def test_k_aware_requires_valid_scalar_k_in_active_node_order(k):
    with pytest.raises(ValueError):
        refined_grid().estimate_gnc(k, method="k_aware")


@pytest.mark.parametrize(
    "options",
    [
        dict(method="resistance"),
        dict(connections="bad"),
        dict(min_offset=-1),
        dict(min_offset=np.inf),
        dict(min_offset=True),
        dict(strict="yes"),
        dict(blocked_pairs=[(-1, 2)]),
        dict(blocked_pairs=[(0, 0)]),
    ],
)
def test_invalid_options_fail_before_returning_corrections(options):
    with pytest.raises(ValueError):
        refined_grid().estimate_gnc(**options)


def test_perpendicular_distances_and_build_are_required():
    with pytest.raises(ValueError, match="cl12"):
        refined_grid(cl12="connector").estimate_gnc(connections="all")
    d = DisuGrid(plan([box(0, 0, 1, 1)]), 1.0, [0.0])
    with pytest.raises((ValueError, RuntimeError), match="build"):
        d.estimate_gnc()


def test_no_heads_or_numerical_runs_are_needed():
    d = refined_grid()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = d.estimate_gnc(connections="all")
    assert result.records and result.corrected_connections <= result.record_count
