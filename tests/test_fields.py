"""Direct unit tests for the mesh size field classes.

Each test builds a minimal gmsh model, calls the field's create(), and reads
the resulting gmsh field options back with gmsh.model.mesh.field.get* so the
refinement parameters the classes promise are actually what gmsh receives.
"""
import math

import gmsh
import pytest

from vorflow.fields import (
    AutoExponentialField,
    AutoLinearField,
    ConstantField,
    DistanceField,
    ExponentialField,
    MeshField,
    ThresholdField,
)


@pytest.fixture
def gmsh_model():
    """A tiny synchronized OCC model: one point, one line, one surface."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("fields_test")
    pt = gmsh.model.occ.addPoint(5, 5, 0)
    l1 = gmsh.model.occ.addPoint(0, -2, 0)
    l2 = gmsh.model.occ.addPoint(10, -2, 0)
    line = gmsh.model.occ.addLine(l1, l2)
    rect = gmsh.model.occ.addRectangle(0, 0, 0, 10, 10)
    gmsh.model.occ.synchronize()
    yield {"point": pt, "line": line, "surface": rect}
    gmsh.finalize()


def _line_tags(model):
    return {"points": [], "lines": [model["line"]], "surfaces": []}


class TestDistanceField:
    def test_creates_distance_field_with_curve_list(self, gmsh_model):
        tag = DistanceField(sampling=33).create(gmsh, _line_tags(gmsh_model))
        assert tag is not None
        assert gmsh.model.mesh.field.getType(tag) == "Distance"
        curves = gmsh.model.mesh.field.getNumbers(tag, "CurvesList")
        assert list(map(int, curves)) == [gmsh_model["line"]]
        assert gmsh.model.mesh.field.getNumber(tag, "Sampling") == 33

    def test_returns_none_for_empty_tags(self, gmsh_model):
        tag = DistanceField().create(gmsh, {"points": [], "lines": [], "surfaces": []})
        assert tag is None


class TestConstantField:
    def test_sets_vin_and_vout(self, gmsh_model):
        tag = ConstantField(size=7.5).create(gmsh, {}, background_lc=50.0)
        assert gmsh.model.mesh.field.getType(tag) == "Constant"
        assert gmsh.model.mesh.field.getNumber(tag, "VIn") == 7.5
        assert gmsh.model.mesh.field.getNumber(tag, "VOut") == 50.0


class TestThresholdField:
    def test_sets_all_threshold_options(self, gmsh_model):
        field = ThresholdField(size_min=2.0, dist_min=4.0, dist_max=40.0, size_max=25.0)
        tag = field.create(gmsh, _line_tags(gmsh_model), background_lc=100.0)
        assert gmsh.model.mesh.field.getType(tag) == "Threshold"
        assert gmsh.model.mesh.field.getNumber(tag, "SizeMin") == 2.0
        assert gmsh.model.mesh.field.getNumber(tag, "SizeMax") == 25.0
        assert gmsh.model.mesh.field.getNumber(tag, "DistMin") == 4.0
        assert gmsh.model.mesh.field.getNumber(tag, "DistMax") == 40.0

    def test_size_max_defaults_to_background(self, gmsh_model):
        field = ThresholdField(size_min=2.0, dist_min=4.0, dist_max=40.0)
        tag = field.create(gmsh, _line_tags(gmsh_model), background_lc=100.0)
        assert gmsh.model.mesh.field.getNumber(tag, "SizeMax") == 100.0

    def test_polygon_surface_gets_constant_interior_via_min(self, gmsh_model):
        tags = {"points": [], "lines": [], "surfaces": [gmsh_model["surface"]]}
        tag = ThresholdField(2.0, 4.0, 40.0).create(gmsh, tags, background_lc=100.0)
        # growth from boundary curves + spatial constant inside -> combined Min
        assert gmsh.model.mesh.field.getType(tag) == "Min"


class TestExponentialField:
    def test_matheval_embeds_decay_and_sizes(self, gmsh_model):
        field = ExponentialField(size_min=1.5, decay_length=30.0, size_max=20.0)
        tag = field.create(gmsh, _line_tags(gmsh_model), background_lc=100.0)
        assert gmsh.model.mesh.field.getType(tag) == "MathEval"
        expr = gmsh.model.mesh.field.getString(tag, "F")
        assert "30.0" in expr and "1.5" in expr and "20.0" in expr


class TestAutoLinearField:
    def test_delegates_to_threshold_with_transition_math(self, gmsh_model):
        cs, cs_dom, fac = 2.0, 100.0, 1.3
        tag = AutoLinearField(growth_factor=fac).create(
            gmsh, _line_tags(gmsh_model), background_lc=cs_dom, feature_lc=cs
        )
        assert gmsh.model.mesh.field.getType(tag) == "Threshold"
        n_cells = math.log(cs_dom / cs) / math.log(fac)
        expected_dist_max = (cs * ((fac ** n_cells) - 1) / math.log(fac)) / 2.0
        assert gmsh.model.mesh.field.getNumber(tag, "DistMin") == pytest.approx(cs / 2.0)
        assert gmsh.model.mesh.field.getNumber(tag, "DistMax") == pytest.approx(expected_dist_max)
        assert gmsh.model.mesh.field.getNumber(tag, "SizeMin") == cs
        assert gmsh.model.mesh.field.getNumber(tag, "SizeMax") == cs_dom

    def test_returns_none_without_feature_lc(self, gmsh_model):
        assert AutoLinearField().create(gmsh, _line_tags(gmsh_model), 100.0) is None

    def test_returns_none_when_feature_not_finer_than_background(self, gmsh_model):
        tag = AutoLinearField().create(
            gmsh, _line_tags(gmsh_model), background_lc=2.0, feature_lc=5.0
        )
        assert tag is None

    def test_rejects_growth_factor_at_or_below_one(self, gmsh_model):
        with pytest.raises(ValueError, match="[Gg]rowth factor"):
            AutoLinearField(growth_factor=1.0).create(
                gmsh, _line_tags(gmsh_model), background_lc=100.0, feature_lc=2.0
            )


class TestAutoExponentialField:
    def test_matheval_expression_uses_growth_factor(self, gmsh_model):
        tag = AutoExponentialField(growth_factor=1.2).create(
            gmsh, _line_tags(gmsh_model), background_lc=100.0, feature_lc=2.0
        )
        assert gmsh.model.mesh.field.getType(tag) == "MathEval"
        expr = gmsh.model.mesh.field.getString(tag, "F")
        assert "1.2^" in expr
        assert expr.startswith("2.0 *")

    def test_returns_none_without_feature_lc(self, gmsh_model):
        tag = AutoExponentialField().create(gmsh, _line_tags(gmsh_model), 100.0)
        assert tag is None

    def test_rejects_growth_factor_at_or_below_one(self, gmsh_model):
        with pytest.raises(ValueError, match="[Gg]rowth factor"):
            AutoExponentialField(growth_factor=0.9).create(
                gmsh, _line_tags(gmsh_model), background_lc=100.0, feature_lc=2.0
            )


class TestFieldEqualityGrouping:
    """__eq__/__hash__ let the engine group identical field specs."""

    def test_equal_parameters_hash_and_compare_equal(self):
        a = ThresholdField(2.0, 4.0, 40.0, 25.0)
        b = ThresholdField(2.0, 4.0, 40.0, 25.0)
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_different_parameters_or_types_differ(self):
        a = ThresholdField(2.0, 4.0, 40.0)
        b = ThresholdField(3.0, 4.0, 40.0)
        c = ExponentialField(2.0, 4.0)
        assert a != b
        assert a != c

    def test_base_class_create_is_abstract(self):
        with pytest.raises(NotImplementedError):
            MeshField().create(gmsh, {}, 100.0)
