"""DisuGrid: layered DISU arrays, tracking and MODFLOW 6 equivalence with DISV."""

import os
import shutil

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from vorflow import disu
from vorflow.utils import grid_to_disv


def _grid(nx=3, ny=2, widths=None):
    widths = np.ones(nx) if widths is None else np.asarray(widths, dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    polygons = [box(edges[i], j, edges[i + 1], j + 1) for j in range(ny) for i in range(nx)]
    return gpd.GeoDataFrame({
        "node_id": np.arange(100, 100 + len(polygons)),
        "zone_id": 1,
        "x": [p.centroid.x for p in polygons], "y": [p.centroid.y for p in polygons],
    }, geometry=polygons)


def _rows(props, key):
    return np.split(np.asarray(props[key]), np.cumsum(props["iac"])[:-1])


def _entries(props):
    """{(n, m): (ihc, cl12, hwva, angldegx)} for every off-diagonal entry."""
    result = {}
    for n, (ja, ihc, cl, hw, ang) in enumerate(zip(*(_rows(props, k) for k in
                                                     ("ja", "ihc", "cl12", "hwva", "angldegx")))):
        for m, *values in zip(ja[1:], ihc[1:], cl[1:], hw[1:], ang[1:]):
            result[n, int(m)] = tuple(values)
    return result


def test_options_are_validated_eagerly():
    grid = _grid()
    with pytest.raises(ValueError, match="collapse"):
        disu.DisuGrid(grid, 10, [0], collapse="sideways")
    with pytest.raises(ValueError, match="min_thickness"):
        disu.DisuGrid(grid, 10, [0], min_thickness=-1)
    with pytest.raises(ValueError, match="6 cells per layer"):
        disu.DisuGrid(grid, 10, [[0, 0]])
    with pytest.raises(RuntimeError, match="build"):
        disu.DisuGrid(grid, 10, [0], connectivity="layered").get_gridprops_disu6()


def test_layered_csr_structure_and_symmetry():
    grid = _grid()
    d = disu.DisuGrid(grid, 10.0, [5.0, 0.0], connectivity="layered").build()
    props = d.get_gridprops_disu6()
    assert props["nodes"] == 12 and len(props["iac"]) == 12
    assert props["nja"] == props["iac"].sum() == len(props["ja"])
    for node, (ja, ihc) in enumerate(zip(_rows(props, "ja"), _rows(props, "ihc"))):
        assert ja[0] == node and ihc[0] == node // 6 + 1     # Diagonal first; layer on the IHC diagonal.
        assert list(ja[1:]) == sorted(ja[1:])
    entries = _entries(props)
    for (n, m), (ihc, cl, hwva, angle) in entries.items():
        ihc_r, cl_r, hwva_r, angle_r = entries[m, n]
        assert ihc == ihc_r and hwva == hwva_r
        if ihc:
            assert (angle - angle_r) % 360 == pytest.approx(180)
    # Cell 0 (x=0.5, y=0.5): east neighbour at 0 degrees, north at 90, below it node 6.
    assert entries[0, 1] == (1, 0.5, 1.0, 0.0)
    assert entries[0, 3] == (1, 0.5, 1.0, 90.0)
    assert entries[0, 6] == (0, 2.5, 1.0, 0.0)
    assert entries[6, 0] == (0, 2.5, 1.0, 0.0)          # Vertical links carry no angle either way.
    assert len(props["cell2d"]) == 12 and props["cell2d"][7][0] == 7
    np.testing.assert_array_equal(props["top"][:6], 10)
    np.testing.assert_array_equal(props["bot"][6:], 0)


def test_pass_through_and_noflow_cells_are_removed_and_tracked():
    grid = _grid(nx=2, ny=1)
    idomain = [[1, 1], [-1, 0], [1, 1]]
    d = disu.DisuGrid(grid, 10.0, [[6, 6], [4, 4], [0, 0]], idomain, connectivity="layered").build()
    np.testing.assert_array_equal(d.nodemap, [[0, 1], [-1, -1], [2, 3]])
    assert list(d.cellmap["status"]) == ["active", "active", "passthrough", "noflow", "active", "active"]
    assert list(d.cellmap["node_id"]) == [100, 101] * 3
    props = d.get_gridprops_disu6()
    entries = _entries(props)
    # Pass-through links node 0 to node 2; no-flow breaks the second column.
    assert (0, 2) in entries and (1, 3) not in entries
    # collapse="split" closes the 2 m gap at its midpoint.
    assert props["bot"][0] == 5 and props["top"][2] == 5
    assert entries[0, 2][1] == 2.5 and entries[2, 0][1] == 2.5
    # Second column keeps its gap next to no-flow only on the no-flow side: both actives take their run.
    assert props["bot"][1] == 6 and props["top"][3] == 4


def test_to_nodes_to_layers_round_trip_and_idomain_tags():
    grid = _grid(nx=2, ny=1)
    d = disu.DisuGrid(grid, 10.0, [[6, 6], [4, 4], [0, 0]], [[1, 1], [-1, 1], [1, 1]],
                      connectivity="layered").build()
    k = np.arange(6).reshape(3, 2) + 1
    np.testing.assert_array_equal(d.to_nodes(k), [1, 2, 4, 5, 6])
    back = d.to_layers(d.to_nodes(k).astype(float))
    np.testing.assert_array_equal(np.isnan(back), [[0, 0], [1, 0], [0, 0]])
    np.testing.assert_array_equal(np.nan_to_num(back), np.where(np.isnan(back), 0, k))
    np.testing.assert_array_equal(d.get_gridprops_disu6(idomain_tag="layer")["idomain"], [1, 1, 2, 3, 3])
    np.testing.assert_array_equal(d.get_gridprops_disu6(idomain_tag=k)["idomain"], [1, 2, 4, 5, 6])
    assert "idomain" not in d.get_gridprops_disu6()
    with pytest.raises(ValueError, match=">= 1"):
        d.get_gridprops_disu6(idomain_tag=k - 1)
    assert "5 nodes" in d.report()


def _mf6():
    flopy = pytest.importorskip("flopy")
    exe = os.environ.get("MF6_EXE") or shutil.which("mf6")
    if not exe:
        pytest.skip("MODFLOW 6 executable not installed (set MF6_EXE or PATH).")
    return flopy, exe


def _run(flopy, exe, ws, add_dis, chd, k, **npf):
    sim = flopy.mf6.MFSimulation(sim_ws=str(ws), exe_name=exe)
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(sim, outer_maximum=50, inner_maximum=500, linear_acceleration="bicgstab",
                         outer_dvclose=1e-10, inner_dvclose=1e-11, rcloserecord=1e-8)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="m")
    add_dis(gwf)
    flopy.mf6.ModflowGwfnpf(gwf, k=k, **(npf or {"k33": 0.1}))
    flopy.mf6.ModflowGwfic(gwf, strt=8.0)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="m.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    success, output = sim.run_simulation(silent=True, report=True)
    assert success, "\n".join(output)
    return gwf.output.head().get_data()


def test_layered_disu_reproduces_disv_heads_with_mf6(tmp_path):
    """Same geometry and connectivity as DISV (incl. pass-through) must give identical heads."""
    flopy, exe = _mf6()
    grid = _grid(nx=4, ny=3, widths=[1, 2, 1.5, 3])
    ncpl = len(grid)
    rng = np.random.default_rng(3)
    botm = np.array([6.0, 3.0, 0.0])[:, None] + rng.uniform(-0.5, 0.5, (3, ncpl))
    idomain = np.ones((3, ncpl), dtype=int)
    idomain[1, 5] = -1
    k_layers = np.array([5.0, 0.5, 2.0])[:, None] * np.ones(ncpl)
    left, right = grid.x.to_numpy() < 1, grid.x.to_numpy() > 4.5
    planar = grid_to_disv(grid)

    def add_disv(gwf):
        flopy.mf6.ModflowGwfdisv(gwf, nlay=3, ncpl=ncpl, nvert=planar["nvert"],
                                 vertices=planar["vertices"], cell2d=planar["cell2d"],
                                 top=10.0, botm=botm, idomain=idomain)

    chd_disv = [[(0, i), 10.0] for i in np.flatnonzero(left)] + [[(2, i), 5.0] for i in np.flatnonzero(right)]
    heads_disv = _run(flopy, exe, tmp_path / "disv", add_disv, chd_disv, k_layers).reshape(3, ncpl)

    d = disu.DisuGrid(grid, 10.0, botm, idomain, connectivity="layered", collapse=None).build()
    props = d.get_gridprops_disu6()
    chd_disu = [[(int(d.nodemap[0, i]),), 10.0] for i in np.flatnonzero(left)] + [
        [(int(d.nodemap[2, i]),), 5.0] for i in np.flatnonzero(right)]
    heads_disu = _run(flopy, exe, tmp_path / "disu",
                      lambda gwf: flopy.mf6.ModflowGwfdisu(gwf, **props), chd_disu, d.to_nodes(k_layers))
    heads_disu = d.to_layers(heads_disu.ravel())
    active = idomain > 0
    np.testing.assert_allclose(heads_disu[active], heads_disv[active], rtol=0, atol=1e-8)
    assert np.isnan(heads_disu[~active]).all()
