"""Behavioural checks for the public one-ring GNC workflow."""

import warnings

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from vorflow.disu import DisuGrid


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


@pytest.mark.parametrize("method", ["geometric", "k_aware"])
def test_one_ring_sandwich_rejects_unit_sum_fit_before_export(method):
    d = sandwich()
    layer = d.cellmap[d.cellmap.node >= 0].sort_values("node").layer.to_numpy()
    k = np.array([1.0, 0.01, 1.0])[layer]
    with pytest.warns(RuntimeWarning, match="left without GNC"):
        result = d.estimate_gnc(k, method=method)
    assert result.corrected_connections == 16 and len(result.unsupported) == 16
    for record in result.records:
        assert min(record["weights"]) >= 0
        assert 1 - sum(record["weights"]) >= 1e-8 - 1e-12
    assert result.get_gridprops_mf6_gnc() is not None
    assert result.get_gridprops_mfusg_gnc() is not None


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
