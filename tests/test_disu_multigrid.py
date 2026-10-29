"""Different plan grids per layer: splice, global columns, overlay links and MODFLOW 6 accuracy."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from vorflow import disu
from vorflow.disu import _merge_plan_slivers

from test_disu_grid import _mf6, _run


def _boxes(n, size, start_id=0):
    polygons = [box(i * size, j * size, (i + 1) * size, (j + 1) * size) for j in range(n) for i in range(n)]
    return gpd.GeoDataFrame({
        "node_id": np.arange(start_id, start_id + len(polygons)),
        "x": [p.centroid.x for p in polygons], "y": [p.centroid.y for p in polygons],
    }, geometry=polygons)


PIT = box(1, 1, 3, 3)


@pytest.fixture
def grids():
    base = _boxes(4, 1.0)                    # 16 cells of 1 m
    fine = _boxes(8, 0.5, start_id=500)      # 64 cells of 0.5 m
    return base, disu.splice(base, fine, PIT)


def test_splice_keeps_base_outside_and_fine_inside(grids):
    base, pit = grids
    assert len(pit) == 12 + 16
    outside = pit[pit.splice_source == "base"]
    assert list(outside.node_id) == list(outside.source_node_id)
    assert all(a.equals(b) for a, b in zip(outside.geometry, base.set_index("node_id").loc[outside.node_id].geometry))
    inside = pit[pit.splice_source == "fine"]
    assert inside.node_id.min() == 16 and inside.node_id.is_unique
    assert inside.geometry.area.sum() == pytest.approx(PIT.area)


def test_splice_rejects_cells_crossing_the_outline():
    base, fine = _boxes(4, 1.0), _boxes(8, 0.5)
    with pytest.raises(ValueError, match="cross the splice outline"):
        disu.splice(base, fine, box(0.5, 0.5, 3, 3))
    with pytest.raises(ValueError, match="cross the splice outline"):
        disu.splice(base, fine, box(1, 1, 2.75, 2.75))


def _pit_model(grids, botm_value=None, **kwargs):
    base, pit = grids
    nb, np_ = len(base), len(pit)
    botm = botm_value or [np.full(nb, 6.0), np.full(nb, 4.0), np.full(np_, 2.0), np.full(np_, 0.0)]
    return disu.DisuGrid([base, base, pit, pit], 10.0, botm, **kwargs).build()


def test_columns_and_links_across_a_change_of_grid(grids):
    base, pit = grids
    d = _pit_model(grids)
    assert d.nodes == 16 * 2 + 28 * 2
    assert [len(m) for m in d.nodemap] == [16, 16, 28, 28]
    links = pd.DataFrame(d.links)
    layer_of = np.repeat(np.arange(4), [16, 16, 28, 28])
    # Overlay links join the 4 coarse pit cells of layer 1 to the 16 fine cells of layer 2.
    overlay = links[(links.ihc == 0) & (layer_of[links.n] == 1) & (layer_of[links.m] == 2)]
    shared = overlay.groupby("n").hwva.sum()
    assert len(overlay) == 12 + 16                         # 12 exterior columns continue, 16 overlaps.
    np.testing.assert_allclose(shared, 1.0)                # Area conserved for every upper cell.
    # On the ring, each exterior face of 1 m meets two fine faces of 0.5 m.
    ring = links[(links.ihc > 0) & (layer_of[links.n] == 2) & (layer_of[links.m] == 2)]
    assert sorted(set(np.round(ring.hwva, 9))) == [0.5, 1.0]
    assert d.diagnostics["vertical_offset"] == 0
    assert "on 2 grids" in d.report()


def test_tops_of_new_columns_come_from_the_cell_above(grids):
    base, pit = grids
    slope = 6.0 - 0.1 * base.x.to_numpy()
    d = _pit_model(grids, [slope, slope - 2, np.full(len(pit), 1.0), np.full(len(pit), 0.0)])
    cm = d.cellmap
    fine_l2 = cm[(cm.layer == 2) & (cm.grid == 1) & (cm.node_id >= 16)]
    # A fine cell centred at x=1.25 lies in the coarse cell centred at x=1.5: bottom 6 - 0.15 - 2.
    first = fine_l2[np.isclose(fine_l2.x, 1.25)].iloc[0]
    assert first.top == pytest.approx(3.85)


def test_lists_shaped_like_botm_round_trip(grids):
    d = _pit_model(grids)
    k = [np.full(16, 1.0), np.full(16, 2.0), np.full(28, 3.0), np.full(28, 4.0)]
    nodes = d.to_nodes(k)
    assert nodes.shape == (88,) and list(np.unique(nodes)) == [1, 2, 3, 4]
    back = d.to_layers(nodes)
    assert [len(b) for b in back] == [16, 16, 28, 28]
    np.testing.assert_array_equal(back[2], 3.0)
    with pytest.raises(ValueError, match="one entry per layer"):
        d.to_nodes(k[:3])


def test_plan_slivers_join_the_largest_overlap_of_the_smaller_cell():
    area = np.array([10.0, 4.0, 5.0, 0.2])
    overlaps = pd.DataFrame({"col_a": [0, 0, 3], "col_b": [1, 2, 1], "area": [3.9, 5.0, 0.02]})
    merged, count = _merge_plan_slivers(overlaps, area, 0.05)
    assert count == 0                                 # 0.02 is 10 % of cell 3: not a sliver.
    overlaps.loc[0, "area"] = 0.1                     # 0.1 is 2.5 % of cell 1, the smaller cell.
    merged, count = _merge_plan_slivers(overlaps, area, 0.05)
    assert count == 1
    # It joins cell 1's other (largest) overlap; total area is kept.
    np.testing.assert_allclose(sorted(merged.area), [0.12, 5.0])
    # A sliver that is a cell's only overlap stays.
    alone = pd.DataFrame({"col_a": [0], "col_b": [1], "area": [0.01]})
    assert _merge_plan_slivers(alone, area, 0.05)[1] == 0


@pytest.mark.parametrize("dip", [0.0, 0.2])
def test_pit_refined_at_depth_reproduces_a_linear_head_with_xt3d(tmp_path, grids, dip):
    """Hanging-node ring faces and overlay links are exact for XT3D on a linear field."""
    flopy, exe = _mf6()
    base, pit = grids
    tilt_b, tilt_p = dip * base.x.to_numpy(), dip * pit.x.to_numpy()
    botm = [8.0 + tilt_b, 6.0 + tilt_b, 4.0 + tilt_b, 2.0 + tilt_p, 0.0 + tilt_p, -2.0 + tilt_p]
    d = disu.DisuGrid([base, base, base, pit, pit, pit], 10.0 + tilt_b, botm).build()
    cm = d.cellmap[d.cellmap.node >= 0].sort_values("node")
    z = ((cm.top + cm.bot) / 2).to_numpy()
    x, y = cm.x.to_numpy(), cm.y.to_numpy()
    exact = 50.0 - 1.0 * x + 0.5 * y - 0.3 * z
    # A vertical gradient needs fixed heads on the top and bottom layers as well.
    layer = cm.layer.to_numpy()
    edge = (x < 1) | (x > 3) | (y < 1) | (y > 3) | (layer == 0) | (layer == 5)
    chd = [[(int(n),), float(exact[n])] for n in np.flatnonzero(edge)]
    props = d.get_gridprops_disu6()
    heads = _run(flopy, exe, tmp_path, lambda gwf: flopy.mf6.ModflowGwfdisu(gwf, **props), chd,
                 np.ones(d.nodes), k33=1.0, xt3doptions=True, angle2=0.0, angle3=0.0).ravel()
    assert np.abs(heads - exact)[~edge].max() < 1e-6
