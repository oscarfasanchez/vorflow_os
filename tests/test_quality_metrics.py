import gmsh
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from vorflow import ConceptualMesh, MeshGenerator
from vorflow.utils import build_connectivity, calculate_mesh_quality, calculate_orthogonality


TRIANGULAR_QUALITY_COLUMNS = [
    "element_tag",
    "element_type",
    "element_name",
    "is_triangle",
    "minSICN",
    "minDetJac",
    "maxDetJac",
    "minSJ",
    "minSIGE",
    "gamma",
    "innerRadius",
    "outerRadius",
    "minIsotropy",
    "angleShape",
    "minEdge",
    "maxEdge",
]


@pytest.fixture(autouse=True)
def ensure_gmsh_finalized():
    if gmsh.is_initialized():
        gmsh.finalize()
    yield
    if gmsh.is_initialized():
        gmsh.finalize()


@pytest.fixture
def paired_polygons():
    poly_a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly_b = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    return gpd.GeoDataFrame(
        {
            "node_id": [10, 20],
            "x": [0.5, 1.5],
            "y": [0.5, 0.5],
            "geometry": [poly_a, poly_b],
        }
    )


def test_build_connectivity_reports_generator_based_pair_metrics(paired_polygons):
    connectivity = build_connectivity(paired_polygons)

    assert len(connectivity) == 1
    row = connectivity.iloc[0]
    assert row["cell_id_1"] != row["cell_id_2"]
    assert {row["node_id_1"], row["node_id_2"]} == {10, 20}
    assert row["center_mode"] == "generator"
    assert row["shared_edge"].length == pytest.approx(1.0)
    assert row["connector"].length == pytest.approx(1.0)
    assert row["angle"] == pytest.approx(90.0, abs=1e-6)
    assert row["ortho_error"] == pytest.approx(0.0, abs=1e-6)
    assert row["skewness"] == pytest.approx(0.5, abs=1e-6)


def test_build_connectivity_centroid_mode_works_without_generator_columns(paired_polygons):
    gdf = paired_polygons.drop(columns=["x", "y"])

    connectivity = build_connectivity(gdf, center="centroid")

    assert len(connectivity) == 1
    row = connectivity.iloc[0]
    assert row["center_mode"] == "centroid"
    assert row["angle"] == pytest.approx(90.0, abs=1e-6)
    assert row["ortho_error"] == pytest.approx(0.0, abs=1e-6)
    assert row["skewness"] == pytest.approx(0.5, abs=1e-6)


def test_build_connectivity_generator_mode_still_requires_generator_columns(paired_polygons):
    gdf = paired_polygons.drop(columns=["x"])

    with pytest.raises(ValueError, match="center='generator'"):
        build_connectivity(gdf)


def test_centroid_mode_exposes_asymmetric_cell_center_error():
    left = Polygon([(0, 0), (1, 0), (1, 2), (0, 2)])
    right = Polygon([(1, 0), (3, 0), (3, 1), (1, 1)])
    gdf = gpd.GeoDataFrame(
        {
            "node_id": [1, 2],
            "x": [0.5, 2.0],
            "y": [0.5, 0.5],
            "geometry": [left, right],
        }
    )

    generator_report = build_connectivity(gdf)
    centroid_report = build_connectivity(gdf, center="centroid")
    centroid_quality = calculate_mesh_quality(
        gdf.drop(columns=["x", "y"]),
        calc_ortho=True,
        calc_skewness=True,
        connectivity=centroid_report,
    )

    assert generator_report.iloc[0]["ortho_error"] == pytest.approx(0.0, abs=1e-6)
    assert centroid_report.iloc[0]["center_mode"] == "centroid"
    assert centroid_report.iloc[0]["ortho_error"] > 0.0
    assert centroid_report.iloc[0]["skewness"] != pytest.approx(0.5, abs=1e-6)
    assert "ortho_error" in centroid_quality.columns
    assert "skewness" in centroid_quality.columns


def test_build_connectivity_uses_unique_row_ids_when_index_is_duplicated(paired_polygons):
    duplicated_index = paired_polygons.copy()
    duplicated_index.index = [7, 7]

    connectivity = build_connectivity(duplicated_index)
    quality = calculate_mesh_quality(
        duplicated_index,
        calc_ortho=True,
        calc_skewness=True,
        connectivity=connectivity,
    )

    assert connectivity[["cell_id_1", "cell_id_2"]].iloc[0].tolist() == [0, 1]
    assert connectivity[["orig_index_1", "orig_index_2"]].iloc[0].tolist() == [7, 7]
    assert quality["ortho_error"].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    assert quality["skewness"].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_calculate_mesh_quality_validates_supplied_connectivity_schema(paired_polygons):
    connectivity = build_connectivity(paired_polygons).drop(columns=["skewness"])

    with pytest.raises(ValueError, match="connectivity missing required columns"):
        calculate_mesh_quality(
            paired_polygons,
            calc_skewness=True,
            connectivity=connectivity,
        )


def test_build_connectivity_skips_point_only_touching_cells():
    gdf = gpd.GeoDataFrame(
        {
            "x": [0.5, 1.5],
            "y": [0.5, 1.5],
            "geometry": [box(0, 0, 1, 1), box(1, 1, 2, 2)],
        }
    )

    assert build_connectivity(gdf).empty


def test_build_connectivity_uses_longest_multilinestring_shared_boundary():
    left = MultiPolygon([box(0, 0, 1, 1), box(0, 2, 1, 4)])
    right = MultiPolygon([box(1, 0, 2, 1), box(1, 2, 2, 5)])
    gdf = gpd.GeoDataFrame(
        {
            "x": [0.5, 1.5],
            "y": [3.0, 3.0],
            "geometry": [left, right],
        }
    )

    connectivity = build_connectivity(gdf)

    assert len(connectivity) == 1
    assert connectivity.iloc[0]["shared_edge"].length == pytest.approx(2.0)


def test_build_connectivity_projects_shared_face_midpoint_when_connector_does_not_cross():
    gdf = gpd.GeoDataFrame(
        {
            "x": [0.5, 1.5],
            "y": [2.0, 2.0],
            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        }
    )

    connectivity = build_connectivity(gdf)

    assert len(connectivity) == 1
    assert connectivity.iloc[0]["skewness"] == pytest.approx(0.5, abs=1e-6)


def test_calculate_orthogonality_regression_value_unchanged(paired_polygons):
    result = calculate_orthogonality(paired_polygons)
    expected = pd.Series([0.0, 0.0], index=paired_polygons.index)

    pd.testing.assert_series_equal(result, expected)


def test_calculate_mesh_quality_skewness_is_opt_in(paired_polygons):
    default_quality = calculate_mesh_quality(paired_polygons)

    assert "ortho_error" not in default_quality.columns
    assert "skewness" not in default_quality.columns

    connectivity = build_connectivity(paired_polygons)
    skew_quality = calculate_mesh_quality(
        paired_polygons,
        calc_ortho=True,
        calc_skewness=True,
        connectivity=connectivity,
    )

    assert "ortho_error" in skew_quality.columns
    assert "skewness" in skew_quality.columns
    assert skew_quality["ortho_error"].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    assert skew_quality["skewness"].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_get_triangular_quality_requires_generation():
    mesher = MeshGenerator(background_lc=2.0, verbosity=0)

    with pytest.raises(RuntimeError, match="Call MeshGenerator.generate"):
        mesher.get_triangular_quality()


def test_get_triangular_quality_returns_cached_gmsh_metrics_after_generate():
    cm = ConceptualMesh()
    cm.add_polygon(
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        zone_id=1,
        resolution=2.0,
        dist_max=4.0,
    )
    clean_polys, clean_lines, clean_points = cm.generate()

    mesher = MeshGenerator(
        background_lc=2.0,
        verbosity=0,
        smoothing_steps=0,
        optimization_cycles=0,
    )
    assert mesher.generate(clean_polys, clean_lines, clean_points)

    quality = mesher.get_triangular_quality()

    assert list(quality.columns) == TRIANGULAR_QUALITY_COLUMNS
    assert not quality.empty
    assert quality["element_tag"].is_unique
    assert quality["is_triangle"].all()
    metric_values = quality.drop(
        columns=["element_tag", "element_type", "element_name", "is_triangle"]
    )
    assert np.isfinite(metric_values.to_numpy()).all()
    assert quality["gamma"].between(0.0, 1.0).all()
    assert quality["minSICN"].between(-1.0, 1.0).all()

    quality_copy = mesher.get_triangular_quality()
    quality_copy.loc[quality_copy.index[0], "gamma"] = -999.0
    assert mesher.get_triangular_quality()["gamma"].iloc[0] != -999.0
