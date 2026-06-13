import warnings

import gmsh
import pytest
from shapely.geometry import LineString, Polygon, box

from vorflow import ConceptualMesh, MeshGenerator, VoronoiTessellator


@pytest.fixture(autouse=True)
def ensure_gmsh_finalized():
    if gmsh.is_initialized():
        gmsh.finalize()
    yield
    if gmsh.is_initialized():
        gmsh.finalize()


def _generate_line_buffer_mesh(*, thickness=1, add_crossing_line=False, return_context=False):
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(
        box(0, 0, 10, 4),
        zone_id="domain",
        resolution=1.0,
        densify=True,
    )
    cm.add_line(
        LineString([(0, 2), (10, 2)]),
        line_id="buffered-line",
        resolution=1.0,
        is_barrier=True,
        quad_buffer=True,
        quad_buffer_thickness=thickness,
        dist_min=0.5,
        dist_max=2.0,
    )
    if add_crossing_line:
        cm.add_line(
            LineString([(5, 0), (5, 4)]),
            line_id="crossing-line",
            resolution=0.5,
            is_barrier=False,
            dist_min=0.5,
            dist_max=1.5,
        )

    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(
        background_lc=1.5,
        verbosity=0,
        smoothing_steps=0,
        optimization_cycles=0,
    )
    assert mesher.generate(clean_polys, clean_lines, clean_points)
    if return_context:
        return mesher, cm
    return mesher


def test_line_structured_quad_buffer_produces_mixed_element_grid():
    mesher = _generate_line_buffer_mesh(thickness=1)
    element_grid = mesher.get_element_grid()
    quads = mesher.get_element_grid("quads")
    triangles = mesher.get_element_grid("triangles")

    assert not element_grid.empty
    assert not quads.empty
    assert not triangles.empty
    assert len(quads) + len(triangles) == len(element_grid)
    assert quads["is_quad"].all()
    assert quads["centroid_y"].between(1.45, 2.55).all()
    # Quads inside the domain must receive a valid zone assignment.
    assert quads["zone_id"].notna().all()
    assert quads["zone_id"].eq("domain").all()
    # A thickness-1 strip on a length-10 line at lc=1 is a single structured
    # row of ~10 quads, not an arbitrary recombination.
    assert 7 <= len(quads) <= 14


def test_line_structured_quad_buffer_thickness_controls_quad_band_area():
    thin = _generate_line_buffer_mesh(thickness=1).get_element_grid("quads")
    thick = _generate_line_buffer_mesh(thickness=2).get_element_grid("quads")

    assert not thin.empty
    assert not thick.empty
    assert thick.geometry.area.sum() > thin.geometry.area.sum()


def test_structured_quad_buffer_protects_crossing_feature_corridor():
    baseline_quads = _generate_line_buffer_mesh(thickness=1).get_element_grid("quads")
    if gmsh.is_initialized():
        gmsh.finalize()

    mesher = _generate_line_buffer_mesh(thickness=1, add_crossing_line=True)
    quads = mesher.get_element_grid("quads")
    triangles = mesher.get_element_grid("triangles")

    assert not quads.empty
    # The crossing standard line is trimmed out of the strip corridor, so the
    # strip interior stays quad-dominated. (The crossing line's size field may
    # still grade into the strip and leave a couple of split triangles, but no
    # feature nodes are injected.)
    strip_interior = LineString([(0, 2), (10, 2)]).buffer(0.4, cap_style=2)
    strip_quads = quads[quads.geometry.centroid.within(strip_interior)]
    strip_triangles = triangles[triangles.geometry.centroid.within(strip_interior)]
    assert len(strip_quads) >= 8
    assert len(strip_triangles) <= max(2, len(strip_quads) // 4)
    # The strip structure is preserved despite the crossing feature.
    assert abs(len(quads) - len(baseline_quads)) <= max(2, int(0.25 * len(baseline_quads)))


def test_curved_line_buffer_produces_structured_quads():
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 10, 6), zone_id="domain", resolution=1.0, densify=True)
    line = LineString([(1, 2), (5, 4), (9, 2)])
    cm.add_line(
        line,
        line_id="curved",
        resolution=1.0,
        is_barrier=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
        dist_min=0.5,
        dist_max=2.0,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.5, verbosity=0, smoothing_steps=0, optimization_cycles=0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert mesher.generate(clean_polys, clean_lines, clean_points)
    degradations = [
        w for w in caught if "transfinite" in str(w.message) or "recombine-only" in str(w.message)
    ]
    assert not degradations, [str(w.message) for w in degradations]

    quads = mesher.get_element_grid("quads")
    assert not quads.empty
    assert quads["is_quad"].all()
    # A full structured row of quads hugs the whole curved line.
    assert quads.geometry.centroid.apply(lambda p: line.distance(p) < 0.75).all()
    assert len(quads) >= round(line.length) - 3


def test_crossing_quad_buffers_are_protected_and_warn():
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 10, 4), zone_id="domain", resolution=1.0, densify=True)
    cm.add_line(
        LineString([(0, 2), (10, 2)]),
        line_id="horizontal",
        resolution=1.0,
        is_barrier=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    cm.add_line(
        LineString([(5, 0), (5, 4)]),
        line_id="vertical",
        resolution=1.0,
        is_barrier=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.5, verbosity=0, smoothing_steps=0, optimization_cycles=0)

    with pytest.warns(UserWarning, match="crosses another protected feature"):
        assert mesher.generate(clean_polys, clean_lines, clean_points)

    quads = mesher.get_element_grid("quads")
    assert not quads.empty
    horizontal_arm = quads[
        (quads["centroid_y"].sub(2).abs() < 0.5) & (quads["centroid_x"].sub(5).abs() > 1)
    ]
    vertical_arm = quads[
        (quads["centroid_x"].sub(5).abs() < 0.5) & (quads["centroid_y"].sub(2).abs() > 1)
    ]
    assert not horizontal_arm.empty
    assert not vertical_arm.empty


def test_transfinite_survives_fragmentation():
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 10, 4), zone_id="domain", resolution=1.0, densify=True)
    line = LineString([(1, 2), (9, 2)])
    cm.add_line(
        line,
        line_id="buffered",
        resolution=1.0,
        is_barrier=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    cm.add_line(
        LineString([(5, 0), (5, 4)]),
        line_id="crossing-standard",
        resolution=0.5,
        is_barrier=False,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.5, verbosity=0, smoothing_steps=0, optimization_cycles=0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert mesher.generate(clean_polys, clean_lines, clean_points)
    degradations = [
        w for w in caught if "transfinite" in str(w.message) or "recombine-only" in str(w.message)
    ]
    assert not degradations, [str(w.message) for w in degradations]

    quads = mesher.get_element_grid("quads")
    assert not quads.empty
    # Structured rows: near-uniform quad areas and a count matching length/lc.
    areas = quads.geometry.area
    assert areas.std() / areas.mean() < 0.35
    assert round(line.length) - 2 <= len(quads) <= round(line.length) + 4


def test_structured_quad_buffer_barrier_is_not_cut_again_by_tessellator():
    mesher, cm = _generate_line_buffer_mesh(thickness=1, return_context=True)

    tessellator = VoronoiTessellator(mesher, cm, clip_to_boundary=True)
    grid = tessellator.generate()

    assert len(grid) == len(mesher.node_tags)


def test_polygon_structured_quad_buffer_produces_quads():
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(
        box(0, 0, 10, 6),
        zone_id="domain",
        resolution=1.5,
        densify=True,
    )
    cm.add_polygon(
        Polygon([(3, 2), (7, 2), (7, 4), (3, 4)]),
        zone_id="inner",
        resolution=1.0,
        z_order=1,
        densify=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    clean_polys, clean_lines, clean_points = cm.generate()

    mesher = MeshGenerator(
        background_lc=1.5,
        verbosity=0,
        smoothing_steps=0,
        optimization_cycles=0,
    )
    # Polygon bands are recombine-only by design (annulus), which must not be
    # reported as a degradation.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert mesher.generate(clean_polys, clean_lines, clean_points)
    degradations = [
        w for w in caught if "transfinite" in str(w.message) or "recombine-only" in str(w.message)
    ]
    assert not degradations, [str(w.message) for w in degradations]

    quads = mesher.get_element_grid("quads")
    assert not quads.empty


def test_node_extraction_covers_all_regions_split_by_buffer_strip():
    # A full-width strip crossing an inner zone splits the domain into pieces
    # OCC's fragment map can drop (regression: all generators above the strip
    # were lost, producing giant merged-looking Voronoi cells).
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 12, 7), zone_id="domain", resolution=1.5, densify=True)
    cm.add_polygon(
        Polygon([(4, 2), (8, 2), (8, 5), (4, 5)]),
        zone_id="inner-zone",
        resolution=1.0,
        z_order=1,
        densify=True,
    )
    cm.add_line(
        LineString([(0, 3.5), (12, 3.5)]),
        line_id="fault-buffer",
        resolution=1.0,
        is_barrier=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.5, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    assert mesher.generate(clean_polys, clean_lines, clean_points)

    element_grid = mesher.get_element_grid()
    corner_xy = {
        (round(x, 6), round(y, 6))
        for geom in element_grid.geometry
        for x, y in geom.exterior.coords[:-1]
    }
    node_xy = {(round(float(x), 6), round(float(y), 6)) for x, y in mesher.nodes}
    missing = corner_xy - node_xy
    assert not missing, f"{len(missing)} gmsh mesh nodes missing from Voronoi generators"


def test_strip_crossing_embedded_zone_stays_transfinite():
    # Regression: a strip split by an embedded zone boundary degraded to
    # recombined triangles. Pieces are now re-cornered from the offset side
    # lines and kept transfinite, and densified ring vertices falling inside
    # the strip are pushed onto its boundary so the caps stay single curves.
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 12, 7), zone_id="domain", resolution=1.5, densify=True)
    cm.add_polygon(
        Polygon([(4, 2), (8, 2), (8, 5), (4, 5)]),
        zone_id="inner",
        resolution=1.0,
        z_order=1,
        densify=True,
    )
    cm.add_line(
        LineString([(0, 3.5), (12, 3.5)]),
        line_id="fault",
        resolution=1.0,
        is_barrier=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.5, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert mesher.generate(clean_polys, clean_lines, clean_points)
    degradations = [
        w for w in caught if "transfinite" in str(w.message) or "recombine-only" in str(w.message)
    ]
    assert not degradations, [str(w.message) for w in degradations]

    quads = mesher.get_element_grid("quads")
    triangles = mesher.get_element_grid("triangles")
    strip = LineString([(0, 3.5), (12, 3.5)]).buffer(0.4, cap_style=2)
    assert triangles[triangles.geometry.centroid.within(strip)].empty
    # One structured row of ~12 quads along the 12-length fault.
    assert 10 <= len(quads) <= 16


def test_size_field_attached_to_quad_buffer_line_refines_halo():
    # Regression: distance-growth fields on quad_buffer lines were silently
    # dropped because the strip surfaces were not listed as embedded surfaces.
    from vorflow.fields import AutoExponentialField

    def cell_count(with_field):
        if gmsh.is_initialized():
            gmsh.finalize()
        cm = ConceptualMesh(crs=None)
        cm.add_polygon(box(0, 0, 20, 12), zone_id="domain", resolution=4.0, densify=True)
        cm.add_line(
            LineString([(4, 6), (16, 6)]),
            line_id="buffered",
            resolution=0.5,
            quad_buffer=True,
            quad_buffer_thickness=1,
            fields=[AutoExponentialField(growth_factor=1.2)] if with_field else None,
        )
        clean_polys, clean_lines, clean_points = cm.generate()
        mesher = MeshGenerator(background_lc=4.0, verbosity=0, smoothing_steps=0, optimization_cycles=0)
        assert mesher.generate(clean_polys, clean_lines, clean_points)
        return len(mesher.get_element_grid())

    without_field = cell_count(False)
    with_field = cell_count(True)
    # The exponential halo must grade sizes outward from the strip, producing
    # clearly more elements than the bare strip in a coarse background.
    assert with_field > without_field * 1.3


def _generate_polygon_buffer_voronoi(thickness):
    zone = Polygon([(3, 3), (9, 3), (9, 7), (3, 7)])
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 12, 10), zone_id="domain", resolution=1.0, densify=False)
    cm.add_polygon(
        zone,
        zone_id="inner",
        resolution=1.0,
        z_order=1,
        densify=True,
        quad_buffer=True,
        quad_buffer_thickness=thickness,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.0, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    assert mesher.generate(clean_polys, clean_lines, clean_points)
    grid = VoronoiTessellator(mesher, cm, clip_to_boundary=True).generate()
    return zone, grid


def test_polygon_quad_buffer_thickness1_faces_follow_boundary():
    # The zone outline never becomes mesh edges: a single quad row straddles it
    # (nodes at +/- lc/2), so the Voronoi faces trace the shapefile shape and
    # no cell straddles the outline.
    zone, grid = _generate_polygon_buffer_voronoi(1)
    centroids = grid.geometry.centroid
    assert (centroids.distance(zone.exterior) < 0.05).sum() == 0
    inside_frac = grid.geometry.intersection(zone).area / grid.geometry.area
    crossing = ((inside_frac > 0.03) & (inside_frac < 0.97)).sum()
    assert crossing == 0


def test_polygon_quad_buffer_thickness2_centers_cells_on_boundary():
    # Two quad rows put a node row on the outline, giving a ring of ~square
    # (lc x lc) Voronoi cells centered on the shape.
    zone, grid = _generate_polygon_buffer_voronoi(2)
    centroids = grid.geometry.centroid
    ring = grid[centroids.distance(zone.exterior) < 0.05]
    perimeter = zone.exterior.length
    assert 0.65 * perimeter <= len(ring) <= 1.35 * perimeter
    assert abs(ring.geometry.area.mean() - 1.0) < 0.15


def test_line_quad_buffer_thickness2_produces_square_cells_on_line():
    line = LineString([(1, 5), (11, 5)])
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 12, 10), zone_id="domain", resolution=1.0, densify=False)
    cm.add_line(
        line,
        line_id="drain",
        resolution=1.0,
        quad_buffer=True,
        quad_buffer_thickness=2,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.0, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    assert mesher.generate(clean_polys, clean_lines, clean_points)
    grid = VoronoiTessellator(mesher, cm, clip_to_boundary=True).generate()

    on_line = grid[grid.geometry.centroid.distance(line) < 0.05]
    assert round(line.length) - 2 <= len(on_line) <= round(line.length) + 3
    areas = on_line.geometry.area
    assert areas.std() / areas.mean() < 0.2


def test_narrow_zone_quad_buffer_falls_back_with_warning():
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 12, 10), zone_id="domain", resolution=1.0, densify=False)
    cm.add_polygon(
        Polygon([(3, 4.7), (9, 4.7), (9, 5.3), (3, 5.3)]),
        zone_id="sliver",
        resolution=1.0,
        z_order=1,
        densify=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.0, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    with pytest.warns(UserWarning, match="too narrow for a quad_buffer band"):
        assert mesher.generate(clean_polys, clean_lines, clean_points)
    assert not mesher.get_element_grid().empty


def test_buffer_surfaces_do_not_double_mesh_crossing_configuration():
    # Regression: OCC fragment silently refused to cut overlapping faces when a
    # trimmed line strip crossed a trimmed polygon band, leaving the domain and
    # zone surfaces triangulated on top of the strip quads (duplicate nodes,
    # junk Voronoi cells). Embedded polygon surfaces are now built disjoint
    # from every buffer footprint, so the elements tile the domain exactly.
    cm = ConceptualMesh(crs=None)
    cm.add_polygon(box(0, 0, 12, 10), zone_id="domain", resolution=1.0, densify=False)
    cm.add_polygon(
        Polygon([(3, 3), (9, 3), (9, 7), (3, 7)]),
        zone_id="inner",
        resolution=1.0,
        z_order=1,
        densify=True,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    cm.add_line(
        LineString([(0, 5), (12, 5)]),
        line_id="fault",
        resolution=1.0,
        quad_buffer=True,
        quad_buffer_thickness=1,
    )
    clean_polys, clean_lines, clean_points = cm.generate()
    mesher = MeshGenerator(background_lc=1.5, verbosity=0, smoothing_steps=0, optimization_cycles=0)
    with pytest.warns(UserWarning, match="crosses another protected feature"):
        assert mesher.generate(clean_polys, clean_lines, clean_points)

    element_grid = mesher.get_element_grid()
    assert abs(element_grid.geometry.area.sum() - 120.0) < 0.01


def test_quad_buffer_thickness_is_validated():
    cm = ConceptualMesh(crs=None)

    with pytest.raises(ValueError, match="quad_buffer_thickness"):
        cm.add_line(
            LineString([(0, 0), (1, 0)]),
            line_id="bad",
            resolution=1.0,
            quad_buffer=True,
            quad_buffer_thickness=3,
        )

    with pytest.raises(ValueError, match="quad_buffer_thickness"):
        cm.add_polygon(
            box(0, 0, 1, 1),
            zone_id="bad",
            resolution=1.0,
            quad_buffer=True,
            quad_buffer_thickness=0,
        )
