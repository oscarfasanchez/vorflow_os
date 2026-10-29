"""Whole-cell hybrid holes using the upstream exact-circle barrier mesher."""
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import shapely
from shapely.affinity import translate
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator
from vorflow._barrier import BarrierMeshError
from vorflow._hybrid import finish_hybrid


def _inputs(kind="zigzag", offset=0):
    lines = {
        "rectangle": [[(75, 50), (125, 50)]],
        "zigzag": [[(55, 40), (80, 60), (105, 40), (145, 60)]],
        "junction": [[(55, 50), (100, 50), (145, 50)], [(100, 50), (125, 75)]],
        "contact": [[(0, 50), (60, 50)]],
    }[kind]
    domain = translate(box(0, 0, 200, 100), offset, offset)
    faults = [translate(LineString(xy), offset, offset) for xy in lines]
    footprint = unary_union(faults).buffer(np.sqrt(3) * 5, cap_style="square", join_style="mitre")
    cm = ConceptualMesh()
    cm.add_polygon(domain, zone_id="aquifer", resolution=10)
    cm.add_polygon(footprint, resolution=5, subregion="triangle")
    for i, line in enumerate(faults):
        cm.add_line(line, line_id=f"fault_{i}", resolution=5, is_barrier=True)
    return cm, domain, footprint.intersection(domain), faults


def _build(kind="zigzag", offset=0, explicit=False):
    cm, domain, footprint, faults = _inputs(kind, offset)
    if explicit:
        mg = MeshGenerator(background_lc=10, verbosity=0)
        mg.generate(*cm.generate())
    else:
        mg = MeshGenerator(background_lc=10, conceptual_mesh=cm, verbosity=0)
        mg.generate()
    vt = VoronoiTessellator(mg, cm)
    grid = vt.generate()
    return cm, mg, vt, grid, domain, footprint, faults


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["rectangle", "zigzag", "junction", "contact"])
@pytest.mark.parametrize("explicit", [False, True])
def test_closed_barrier_produces_whole_cell_hole(kind, explicit):
    cm, mg, vt, grid, domain, footprint, faults = _build(kind, explicit=explicit)
    assert vt.n_barrier_mirrors == vt.n_exact_barrier_mirrors == vt.n_exact_barrier_cuts == 0
    assert vt.hybrid_report["cell_clip_area"] == 0
    assert vt.hybrid_report["removed_cells"] > 0
    assert len(vt.construction_grid) == len(mg.nodes)
    assert vt.requested_triangle_footprint.equals(footprint)
    assert vt.realised_triangle_footprint.equals(footprint)
    combined = unary_union(grid.geometry)
    assert combined.symmetric_difference(domain).area < 1e-7
    assert abs(grid.area.sum() - combined.area) < 1e-7
    exterior = grid[grid.cell_type == "voronoi"]
    before = vt.construction_grid.set_index("node_id").loc[exterior.node_id]
    np.testing.assert_array_equal(exterior[["x", "y"]], before[["x", "y"]])
    assert all(a.symmetric_difference(b).area < 1e-7 for a, b in zip(exterior.geometry, before.geometry))
    assert (exterior.geometry.convex_hull.area - exterior.area).max() < 1e-7
    assert grid.node_id.is_unique and grid.node_id.dtype == np.int64
    triangles = grid[grid.cell_type == "triangle"]
    assert triangles.node_id.min() > vt.construction_grid.node_id.max()
    assert set(vt.hybrid_report["removed_node_ids"]).isdisjoint(grid.node_id)
    edges = unary_union(triangles.geometry.boundary)
    for fault in faults:
        assert fault.difference(edges.buffer(1e-6)).length < 1e-6
    assert triangles.minimum_angle.min() > 20
    assert not grid.centre_adjusted.any()
    assert set(grid.generator_origin).issubset({"gmsh", "barrier_site", "triangle"})
    assert len(cm.raw_lines) == len(faults)  # The generated perimeter is not registered twice.


@pytest.mark.slow
def test_large_coordinates_and_layered_round_trip(tmp_path):
    flopy = pytest.importorskip("flopy")
    _, _, vt, grid, _, _, _ = _build(offset=5_000_000)
    arrays = vt.to_mfusg_disu(top=10, botm=[0, -5], nlay=2)
    assert arrays["cell_map"].cell_type.tolist() == grid.cell_type.tolist() * 2
    assert arrays["cell_map"].generator_origin.tolist() == grid.generator_origin.tolist() * 2
    model = flopy.mfusg.MfUsg(structured=False, model_ws=tmp_path)
    disu = vt.to_mfusgdisu(model, top=10, botm=[0, -5], nlay=2)
    disu.write_file()
    loaded = flopy.mfusg.MfUsgDisU.load(disu.fn_path, model)
    np.testing.assert_array_equal(loaded.ja.array, arrays["ja"] + 1)
    np.testing.assert_allclose(loaded.cl12.array, arrays["cl12"], rtol=2e-6)
    gsf = vt.to_gsf(tmp_path / "hybrid.gsf", top=10, botm=[0, -5], nlay=2)
    geometry = flopy.discretization.UnstructuredGrid.from_gridspec(gsf)
    np.testing.assert_allclose(geometry.xcellcenters, np.tile(grid.x, 2), rtol=0, atol=1e-7)
    np.testing.assert_allclose(geometry.ycellcenters, np.tile(grid.y, 2), rtol=0, atol=1e-7)


@pytest.mark.parametrize("resolution", [None, 0, -1, float("nan"), float("inf"), True])
def test_invalid_reservation_size(resolution):
    with pytest.raises(ValueError, match="positive finite"):
        ConceptualMesh().add_polygon(box(0, 0, 10, 10), resolution=resolution, subregion="triangle")


def test_invalid_multiple_and_narrow_reservations():
    cm = ConceptualMesh()
    with pytest.raises(ValueError, match="valid polygon"):
        cm.add_polygon(Polygon([(0, 0), (10, 10), (10, 0), (0, 10)]), resolution=5, subregion="triangle")
    with pytest.raises(ValueError, match="too narrow"):
        cm.add_polygon(box(0, 0, 10, 1), resolution=5, subregion="triangle")
    cm.add_polygon(box(0, 0, 10, 10), resolution=5, subregion="triangle")
    with pytest.raises(ValueError, match="Only one"):
        cm.add_polygon(box(20, 0, 30, 10), resolution=5, subregion="triangle")


def test_reservation_does_not_become_a_zone_and_repeated_cleaning_is_stable():
    cm, _, _, faults = _inputs()
    first = cm.generate()
    records = cm.clean_subregions.copy()
    second = cm.generate()
    assert set(second[0].zone_id) == {"aquifer"}
    assert first[1].line_id.tolist() == second[1].line_id.tolist()
    assert len(cm.raw_lines) == len(faults)
    assert records.geometry.iloc[0].equals(cm.clean_subregions.geometry.iloc[0])
    assert len(second[0].attrs["triangle_subregions"]) == 1


@pytest.mark.parametrize("feature", ["fault", "point", "line"])
def test_incompatible_interior_features_fail_clearly(feature):
    cm, _, _, _ = _inputs("rectangle")
    if feature == "fault":
        cm.add_line(LineString([(10, 10), (30, 10)]), line_id="outside", resolution=5, is_barrier=True)
        message = "outside the triangle"
    elif feature == "point":
        cm.add_point(shapely.Point(100, 50), point_id="well", resolution=5)
        message = "point features"
    else:
        cm.add_line(LineString([(80, 48), (120, 48)]), line_id="river", resolution=5)
        message = "non-barrier lines"
    with pytest.raises(ValueError, match=message):
        MeshGenerator(background_lc=10, conceptual_mesh=cm).generate()


@pytest.mark.slow
def test_finish_refuses_damaged_whole_cells():
    _, _, vt, _, _, footprint, _ = _build("rectangle")
    damaged = vt.construction_grid.copy()
    index = damaged[damaged.intersects(footprint.boundary)].index[0]
    damaged.loc[index, "geometry"] = damaged.loc[index, "geometry"].buffer(0.5)
    with pytest.raises(BarrierMeshError, match="refusing clipping"):
        finish_hybrid(vt, damaged)


@pytest.mark.slow
def test_triangle_pass_cannot_change_whole_exterior_cells(monkeypatch):
    import vorflow._hybrid as hybrid
    _, _, vt, grid, _, footprint, _ = _build("rectangle")
    damaged = grid.copy()
    index = damaged[damaged.cell_type == "voronoi"].index[0]
    damaged.loc[index, "geometry"] = damaged.loc[index, "geometry"].buffer(0.1)
    monkeypatch.setattr(hybrid, "build_hybrid_grid", lambda *args, **kwargs: (damaged, footprint))
    with pytest.raises(BarrierMeshError, match="changed a whole exterior"):
        finish_hybrid(vt, vt.construction_grid)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["zigzag", "junction"])
def test_homogeneous_hybrid_mfusg_flow(tmp_path, kind):
    flopy = pytest.importorskip("flopy")
    executable = os.environ.get("MFUSG_EXE") or shutil.which("mfusg")
    if not executable:
        pytest.skip("Install mfusg or set MFUSG_EXE.")
    _, _, vt, grid, _, _, _ = _build(kind)
    model = flopy.mfusg.MfUsg(structured=False, model_ws=tmp_path, exe_name=executable, modelname="hybrid")
    vt.to_mfusgdisu(model, top=10, botm=[0], nlay=1, nper=1, steady=True)
    bound = np.ones(len(grid), dtype=int)
    left, right = np.isclose(grid.x, 0), np.isclose(grid.x, 200)
    bound[left | right] = -1
    initial = np.full(len(grid), 15.)
    initial[left], initial[right] = 20., 10.
    flopy.mfusg.MfUsgBas(model, ibound=bound, strt=initial)
    flopy.mfusg.MfUsgLpf(model, laytyp=0, hk=10, vka=10, ipakcb=53)
    flopy.mfusg.MfUsgSms(model, nonlinmeth=0, linmeth=2, clin="cg", hclose=1e-8,
                       hiclose=1e-8, rclosepcgu=1e-8, mxiter=100, iter1=500)
    flopy.modflow.ModflowOc(model, stress_period_data={(0, 0): ["save head", "save budget", "print budget"]}, compact=True)
    model.write_input()
    success, report = model.run_model(silent=True, report=True)
    assert success and any("Normal termination" in line for line in report)
    heads = flopy.utils.HeadUFile(Path(tmp_path) / "hybrid.hds").get_data()[0].ravel()
    assert np.isfinite(heads).all() and heads.min() >= 10 - 1e-6 and heads.max() <= 20 + 1e-6
    budget, _ = flopy.utils.MfusgListBudget(Path(tmp_path) / "hybrid.list").get_budget()
    assert abs(float(budget["CONSTANT_HEAD_IN"].max()) / 500 - 1) < .02
    assert float(np.abs(budget["PERCENT_DISCREPANCY"]).max()) < .1
