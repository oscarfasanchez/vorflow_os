"""MODFLOW-USG output of DisuGrid: IVSD/IVC, FAHL from overlaps, GSF and an MFUSG run."""

import os
import shutil

import numpy as np
import pytest

from vorflow import disu
from vorflow.utils import grid_to_mfusg_disu

from test_disu_full import _dipping_aquifer, _two_columns
from test_disu_grid import _grid, _rows


def test_disu5_marks_vertical_connections_explicitly():
    d = _two_columns(np.array([20.0, 20.0]), [12, 4], [16, 0])
    props = d.get_gridprops_disu5()
    assert (props["ivsd"], props["idsymrd"], props["nlay"]) == (1, 0, 2)
    np.testing.assert_array_equal(props["nodelay"], [2, 2])
    a1, b1, a2, b2 = d.nodemap[0, 0], d.nodemap[0, 1], d.nodemap[1, 0], d.nodemap[1, 1]
    rows = {n: dict(zip(ja[1:], zip(ivc[1:], fahl[1:], cl[1:]))) for n, (ja, ivc, fahl, cl) in
            enumerate(zip(*(_rows(props, k) for k in ("ja", "ivc", "fahl", "cl12"))))}
    assert rows[a1][a2] == (1, 1.0, 4.0)            # Vertical: IVC=1, plan area, half thickness.
    assert rows[a1][b2] == (0, 4.0, 0.5)            # Staggered stays horizontal: width x overlap.
    assert rows[a1][b1] == (0, 4.0, 0.5)
    assert rows[a2][b2] == (0, 8.0, 0.5)
    assert all(ivc[0] == 0 for ivc in _rows(props, "ivc"))


def test_disu5_counts_nodes_per_layer_after_removing_cells():
    grid = _grid(nx=3, ny=1)
    d = disu.DisuGrid(grid, 10.0, [[6, 6, 6], [4, 4, 4], [0, 0, 0]], [[1, 1, 1], [1, -1, 0], [1, 1, 1]]).build()
    np.testing.assert_array_equal(d.get_gridprops_disu5()["nodelay"], [3, 1, 3])
    empty = disu.DisuGrid(grid, 10.0, [[6, 6, 6], [6, 6, 6]]).build()
    with pytest.raises(ValueError, match="Layer 1 has no active cells"):
        empty.get_gridprops_disu5()


def test_layered_disu5_matches_the_original_usg_export():
    grid = _grid(nx=3, ny=2, widths=[1, 2, 1])
    legacy = grid_to_mfusg_disu(grid, 10.0, [6.0, 0.0], nlay=2)
    new = disu.DisuGrid(grid, 10.0, [6.0, 0.0], connectivity="layered").build().get_gridprops_disu5()
    for key in ("iac", "ja", "cl12", "fahl", "top", "bot", "nodelay"):
        np.testing.assert_allclose(new[key], legacy[key], err_msg=key)
    # The original export keeps one layer of areas and lets USG infer vertical connections.
    assert legacy["ivsd"] == -1 and new["ivsd"] == 1
    np.testing.assert_allclose(new["area"], np.tile(legacy["area"], 2))


def test_gsf_lists_each_node_with_its_own_elevations(tmp_path):
    d = _two_columns(np.array([20.0, 20.0]), [12, 4], [16, 0])
    path = d.write_gsf(tmp_path / "grid.gsf")
    lines = path.read_text().splitlines()
    assert lines[0] == "UNSTRUCTURED" and lines[1] == "4 2 1 1"
    cells = lines[3 + int(lines[2]):]
    assert len(cells) == 4
    # Node 2 is column A, layer 2 (12 to 4): centre elevation 8, layer 2.
    node, x, y, z, layer, nverts = cells[2].split()[:6]
    assert (int(node), float(z), int(layer), int(nverts)) == (3, 8.0, 2, 8)


def _mfusg():
    flopy = pytest.importorskip("flopy", minversion="3.10")
    exe = os.environ.get("MFUSG_EXE") or shutil.which("mfusg")
    if not exe:
        pytest.skip("MFUSG executable not installed (set MFUSG_EXE or PATH).")
    return flopy, exe


def test_full_connectivity_improves_a_dipping_aquifer_in_mfusg(tmp_path):
    """MODFLOW-USG has no XT3D, but staggered connections still let flow follow the dip."""
    flopy, exe = _mfusg()
    errors = {}
    for connectivity in ("layered", "full"):
        d, exact, edge, aquifer = _dipping_aquifer(30, connectivity)
        # MODFLOW-USG resets heads below a cell's bottom even in confined layers; shift the field up.
        exact = exact + 100.0
        props = d.get_gridprops_disu5()
        model = flopy.mfusg.MfUsg(modelname="dip", structured=False, model_ws=tmp_path / connectivity,
                                  exe_name=exe)
        flopy.mfusg.MfUsgDisU(model, **props)
        layer = np.repeat(np.arange(d.nlay), props["nodelay"])
        per_layer = lambda values: [values[layer == k] for k in range(d.nlay)]  # noqa: E731
        k = np.where(aquifer, 1.0, 1e-4)
        flopy.mfusg.MfUsgBas(model, ibound=per_layer(np.where(edge, -1, 1)), strt=per_layer(exact))
        flopy.mfusg.MfUsgLpf(model, laytyp=0, hk=per_layer(k), vka=per_layer(k))
        flopy.mfusg.MfUsgSms(model, hclose=1e-9, hiclose=1e-9, mxiter=200, iter1=1000)
        flopy.modflow.ModflowOc(model, stress_period_data={(0, 0): ["save head"]})
        model.write_input()
        success, output = model.run_model(silent=True, report=True)
        assert success, "\n".join(output[-20:])
        heads = np.concatenate(flopy.utils.HeadUFile(tmp_path / connectivity / "dip.hds").get_data())
        errors[connectivity] = np.abs(heads - exact)[aquifer & ~edge].max()
    assert errors["full"] < 0.5 * errors["layered"]
