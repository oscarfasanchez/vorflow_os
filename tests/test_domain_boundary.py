"""Whole-cell model boundaries built with the native circle barriers."""
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box

from vorflow import BarrierMeshError, ConceptualMesh, MeshGenerator, VoronoiTessellator
from vorflow._domain_boundary import retain_domain_cells


def _model(domain, *, barrier=None, standard=None, zones=False):
    cm = ConceptualMesh()
    cm.add_polygon(domain, zone_id='aquifer', resolution=10)
    if zones:
        cm.add_polygon(box(0, 0, 50, 80), zone_id='refined', resolution=5, z_order=1)
    if barrier is not None:
        cm.add_line(barrier, line_id='fault', resolution=5, is_barrier=True)
    if standard is not None:
        cm.add_line(standard, line_id='river', resolution=5)
    return cm


def _build(cm, *, explicit=False, **tessellation):
    mg = MeshGenerator(background_lc=10, conceptual_mesh=None if explicit else cm,
                       boundary_method='conforming', verbosity=0)
    mg.generate(*cm.generate()) if explicit else mg.generate()
    vt = VoronoiTessellator(mg, cm, **tessellation)
    return mg, vt, vt.generate()


@pytest.mark.slow
@pytest.mark.parametrize('domain', [
    box(0, 0, 100, 80),
    Polygon([(0, 0), (120, 0), (120, 80), (80, 80), (80, 40), (40, 40), (40, 80), (0, 80)]),
    box(0, 0, 120, 100).difference(box(40, 30, 70, 65)),
    affinity.translate(box(0, 0, 120, 100).difference(box(40, 30, 70, 65)), 500000, 6500000),
    box(0, 0, 180, 100).difference(shapely.union_all([box(30, 20, 55, 55), box(105, 40, 135, 75)])),
    box(0, 0, 120, 100).difference(Point(60, 50).buffer(20, quad_segs=8)),
    shapely.union_all([box(0, 0, 40, 40), box(60, 0, 100, 40)]),
    box(0, 0, 100, 6),
], ids=['rectangle', 'reflex', 'hole', 'projected', 'multiple-holes', 'polygonised-hole', 'disconnected', 'narrow'])
def test_domain_keeps_whole_voronoi_cells(domain, monkeypatch):
    import vorflow.tessellator as tessellator
    def no_clip(*args):
        raise AssertionError('Conforming boundaries must never call the domain clipper.')
    monkeypatch.setattr(tessellator, '_clip_to_domain', no_clip)
    cm = _model(domain)
    mg, vt, grid = _build(cm)
    covered = shapely.union_all(grid.geometry)
    assert covered.symmetric_difference(domain).area < 1e-6
    assert abs(grid.area.sum() - domain.area) < 1e-6
    assert (grid.geometry.convex_hull.area - grid.area).max() < 1e-6
    assert vt.n_barrier_mirrors == vt.n_exact_barrier_cuts == 0
    assert vt.boundary_report['cell_clip_area'] == 0
    assert vt.boundary_report['method'] == 'conforming'
    assert set(vt.boundary_report['discarded_node_ids']).isdisjoint(grid.node_id)
    assert set(grid.zone_id) == {'aquifer'}
    assert grid.node_id.is_unique
    source = vt.boundary_construction_grid.set_index('node_id').loc[grid.node_id]
    np.testing.assert_array_equal(grid[['x', 'y']], source[['x', 'y']])
    for before, after in zip(source.geometry, grid.geometry):
        assert before.symmetric_difference(after).area < 1e-6
    vertices = shapely.get_coordinates(shapely.get_exterior_ring(grid.geometry.to_numpy()))
    for polygon in getattr(domain, 'geoms', [domain]):
        for ring in [polygon.exterior, *polygon.interiors]:
            for xy in ring.coords:
                assert np.hypot(*(vertices - xy).T).min() < 1e-6
    assert mg.model_domain.equals(domain)
    assert not mg.construction_domain.equals(domain)
    assert len(cm.raw_lines) == 0
    assert len(mg.zones_gdf) == len(cm.clean_polygons)


@pytest.mark.slow
@pytest.mark.parametrize('feature', ['barrier', 'standard', 'zone'])
def test_existing_features_with_conforming_domain(feature):
    domain = box(0, 0, 120, 80)
    args = {'barrier': LineString([(0, 20), (60, 40), (120, 60)])} if feature == 'barrier' else (
        {'standard': LineString([(0, 40), (60, 40)])} if feature == 'standard' else
        {'zones': True})
    cm = _model(domain, **args)
    _, vt, grid = _build(cm, explicit=True)
    assert shapely.union_all(grid.geometry).symmetric_difference(domain).area < 1e-6
    assert vt.n_exact_barrier_mirrors == vt.n_exact_barrier_cuts == 0
    if feature == 'zone':
        assert set(grid.zone_id) == {'aquifer', 'refined'}


@pytest.mark.slow
@pytest.mark.parametrize('kind', ['zone', 'field_only', 'point'])
def test_native_fields_control_spacing_along_the_boundary(kind):
    cm = _model(box(0, 0, 120, 80))
    if kind == 'zone':
        cm.add_polygon(box(0, 0, 40, 80), zone_id='fine', resolution=5, z_order=1)
    elif kind == 'field_only':
        cm.add_polygon(box(-5, -5, 25, 85), zone_id='field', resolution=3, embed=False)
    else:
        cm.add_point(Point(5, 40), point_id='well', resolution=2)
    _, vt, grid = _build(cm)
    medians = []
    for x in [0, 120]:
        line = LineString([(x, 0), (x, 80)])
        lengths = grid.geometry.boundary.intersection(line).length
        medians.append(float(lengths[lengths > 1e-8].median()))
    assert medians[0] < .7 * medians[1]
    assert medians[1] == pytest.approx(10)
    assert vt.boundary_report['cell_clip_area'] == 0
    assert vt.n_exact_barrier_mirrors == vt.n_exact_barrier_cuts == 0


@pytest.mark.slow
def test_relaxation_and_determinism_preserve_boundary():
    cm = _model(box(0, 0, 100, 80))
    _, _, first = _build(cm, lloyd_iterations=2, min_face_fraction=.15)
    _, _, second = _build(cm, lloyd_iterations=2, min_face_fraction=.15)
    np.testing.assert_array_equal(first.node_id, second.node_id)
    np.testing.assert_allclose(first[['x', 'y']], second[['x', 'y']], rtol=0, atol=1e-10)
    assert all(a.equals_exact(b, 1e-9) for a, b in zip(first.geometry, second.geometry))


def test_option_validation_and_legacy_default():
    assert MeshGenerator(background_lc=10).boundary_method == 'clip'
    with pytest.raises(ValueError, match='boundary_method'):
        MeshGenerator(background_lc=10, boundary_method='unknown')
    cm = _model(box(0, 0, 100, 80))
    mg = MeshGenerator(background_lc=10, conceptual_mesh=cm, boundary_method='conforming')
    with pytest.raises(ValueError, match='owns the domain boundary'):
        VoronoiTessellator(mg, clip_to_boundary=False)
    with pytest.raises(ValueError, match='owns the domain boundary'):
        VoronoiTessellator(mg, boundary_centering='inset_mirror')


@pytest.mark.slow
def test_invalid_outline_and_fixed_boundary_point_fail_explicitly():
    with pytest.raises(BarrierMeshError):
        _build(_model(Polygon([(0, 0), (100, 0), (100, 2)])))
    cm = _model(box(0, 0, 100, 80))
    cm.add_point(Point(0, 40), point_id='well', resolution=5)
    with pytest.raises(BarrierMeshError, match='cannot sit on a barrier face'):
        _build(cm)


@pytest.mark.slow
def test_never_accept_straddling_cells_or_missing_coverage():
    _, vt, _ = _build(_model(box(0, 0, 100, 80)))
    raw = vt.boundary_construction_grid.copy()
    active = raw.geometry.intersection(vt.mg.model_domain).area > 1e-4
    index = raw.loc[active].index[0]
    missing = raw.drop(index)
    with pytest.raises(BarrierMeshError, match='do not cover'):
        retain_domain_cells(vt, missing)
    # Force a material crossing at an actual boundary cell.
    boundary = raw.geometry.distance(vt.mg.model_domain.boundary) < 1e-8
    index = raw.loc[active & boundary].index[0]
    raw.at[index, 'geometry'] = raw.loc[index].geometry.buffer(.1)
    with pytest.raises(BarrierMeshError, match='refusing to clip'):
        retain_domain_cells(vt, raw)


@pytest.mark.slow
def test_disv_round_trip_excludes_outside_support(tmp_path):
    flopy = pytest.importorskip('flopy')
    _, vt, grid = _build(_model(box(0, 0, 120, 100).difference(box(40, 30, 70, 65))))
    arrays = vt.to_disv()
    assert arrays['ncpl'] == len(grid)
    np.testing.assert_array_equal(arrays['node_id'], grid.node_id)
    assert set(arrays['node_id']).isdisjoint(vt.boundary_report['discarded_node_ids'])
    sim = flopy.mf6.MFSimulation(sim_ws=tmp_path)
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname='domain')
    vt.to_modflowgwfdisv(gwf, top=10, botm=[0, -10], nlay=2)
    sim.write_simulation(silent=True)
    loaded = flopy.mf6.MFSimulation.load(sim_ws=tmp_path, verbosity_level=0).get_model('domain')
    np.testing.assert_allclose(loaded.modelgrid.xcellcenters, grid.x, rtol=0, atol=1e-7)
    np.testing.assert_allclose(loaded.modelgrid.ycellcenters, grid.y, rtol=0, atol=1e-7)
    for node, polygon in enumerate(grid.geometry):
        restored = Polygon(loaded.modelgrid.get_cell_vertices(node))
        assert restored.symmetric_difference(polygon).area < 1e-6


@pytest.mark.slow
def test_conforming_domain_mf6_analytical_flow(tmp_path):
    flopy = pytest.importorskip('flopy')
    executable = os.environ.get('MF6_EXE') or shutil.which('mf6')
    if not executable:
        pytest.skip('Install mf6 or set MF6_EXE.')
    _, vt, grid = _build(_model(box(0, 0, 200, 100)))
    sim = flopy.mf6.MFSimulation(sim_ws=tmp_path, exe_name=executable)
    flopy.mf6.ModflowTdis(sim, time_units='DAYS', perioddata=[(1, 1, 1)])
    flopy.mf6.ModflowIms(sim, outer_dvclose=1e-9, inner_dvclose=1e-9,
                       rcloserecord=1e-8, inner_maximum=500)
    gwf = flopy.mf6.ModflowGwf(sim, modelname='domain', save_flows=True)
    vt.to_modflowgwfdisv(gwf, top=10, botm=0, nlay=1, length_units='METERS')
    flopy.mf6.ModflowGwfic(gwf, strt=12.)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=10)
    ghb = []
    # Apply heads at the physical faces, allowing for the true generator offset.
    for x, head in [(0., 20.), (200., 10.)]:
        face = LineString([(x, 0), (x, 100)])
        for node, (_, cell) in enumerate(grid.iterrows()):
            length = cell.geometry.boundary.intersection(face).length
            if length > 1e-8:
                ghb.append([(0, node), head, 10 * 10 * length / abs(cell.x - x)])
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=ghb, save_flows=True)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord='domain.hds', budget_filerecord='domain.cbc',
                        saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
    sim.write_simulation(silent=True)
    success, report = sim.run_simulation(silent=True, report=True)
    assert success and any('Normal termination' in line for line in report)
    heads = gwf.output.head().get_data().ravel()
    assert np.isfinite(heads).all() and heads.min() >= 10 and heads.max() <= 20
    np.testing.assert_allclose(heads, 20 - grid.x.to_numpy() / 20, rtol=0, atol=1e-4)
    budget, _ = flopy.utils.Mf6ListBudget(Path(tmp_path) / 'domain.lst').get_budget()
    assert float(np.abs(budget['PERCENT_DISCREPANCY']).max()) < .1
    assert abs(float(budget['GHB_IN'].max()) / 500 - 1) < .02
