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


def test_line_structured_quad_buffer_thickness_controls_quad_band_area():
    thin = _generate_line_buffer_mesh(thickness=1).get_element_grid("quads")
    thick = _generate_line_buffer_mesh(thickness=2).get_element_grid("quads")

    assert not thin.empty
    assert not thick.empty
    assert thick.geometry.area.sum() > thin.geometry.area.sum()


def test_structured_quad_buffer_protects_crossing_feature_corridor():
    mesher = _generate_line_buffer_mesh(thickness=1, add_crossing_line=True)
    quads = mesher.get_element_grid("quads")

    assert not quads.empty


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
    assert mesher.generate(clean_polys, clean_lines, clean_points)

    quads = mesher.get_element_grid("quads")
    assert not quads.empty


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
