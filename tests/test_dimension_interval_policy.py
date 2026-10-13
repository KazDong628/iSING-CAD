"""Source tolerance bounds coexist with exact geometry; midpoints stay optional."""
from copy import deepcopy
import math

import pytest

from contour_agent.dimension_policy import (
    source_dimension_interval, source_dimension_target,
    tolerance_interval_check, tolerance_interval_residual,
)
from contour_agent.ocr import parse_dimension


def thickness_policy():
    nominal, metadata = source_dimension_interval(parse_dimension("20^{+3}_{0}"))
    assert nominal == 20.
    return metadata


@pytest.mark.parametrize("actual,passed,residual", [
    (20., True, 0.), (21.5, True, 0.), (23., True, 0.),
    (19.9, False, -.1), (23.1, False, .1), (-20., True, 0.),
])
def test_thickness_interval_permits_exact_radius_difference(actual, passed, residual):
    policy = thickness_policy()
    check = tolerance_interval_check(policy, actual)
    assert check["passed"] is passed
    assert tolerance_interval_residual(policy, actual) == pytest.approx(residual)
    assert "construction_target" not in policy
    assert "construction_target" not in check
    # R180/R160 concentric arcs have exactly 20 mm radial separation. The
    # source tolerance permits that derived value without moving either R.
    assert tolerance_interval_check(policy, 180. - 160.)["passed"]


def test_unilateral_positive_deviations_do_not_promote_nominal_to_equality():
    nominal, policy = source_dimension_interval(parse_dimension("10^{+2}_{+1}"))
    assert nominal == 10.
    assert (policy["minimum"], policy["maximum"]) == (11., 12.)
    assert not tolerance_interval_check(policy, nominal)["passed"]
    assert tolerance_interval_check(policy, 11.)["passed"]


def test_old_midpoint_policy_retains_its_declared_construction_target():
    parsed = parse_dimension("20^{+3}_{0}")
    target, policy = source_dimension_target(parsed)
    assert target == 21.5
    assert tolerance_interval_check(policy, target)["construction_target"] == target
    assert source_dimension_target(parsed) == (target, policy)


@pytest.mark.parametrize("text", ["R40", "R40±1", "R40^{+3}_{0}"])
def test_radius_never_acquires_an_interval_relaxation(text):
    assert source_dimension_interval(parse_dimension(text)) == (40., None)
    assert source_dimension_target(parse_dimension(text)) == (40., None)


def test_exact_dimension_without_tolerances_remains_an_equality_candidate():
    assert source_dimension_interval(parse_dimension("26")) == (26., None)


@pytest.mark.parametrize("change", [
    {"minimum": 19.}, {"maximum": 24.}, {"construction_target": 21.5},
    {"nominal": math.nan}, {"minimum": math.nan}, {"maximum": math.inf},
    {"lower_deviation": False}, {"upper_deviation": "3"},
    {"reference_geometry_used": True}, {"policy": "fitted_interval"},
])
def test_forged_or_nonfinite_interval_metadata_is_rejected(change):
    policy = {**thickness_policy(), **change}
    with pytest.raises(ValueError):
        tolerance_interval_check(policy, 20.)
    with pytest.raises(ValueError):
        tolerance_interval_residual(policy, 20.)


@pytest.mark.parametrize("policy_builder", [source_dimension_target, source_dimension_interval])
def test_extra_metadata_fields_cannot_smuggle_coordinates(policy_builder):
    _, policy = policy_builder(parse_dimension("20^{+3}_{0}"))
    forged = deepcopy(policy)
    forged["source_coordinate"] = 123.
    with pytest.raises(ValueError):
        tolerance_interval_check(forged, 20.)


@pytest.mark.parametrize("change", [
    {"nominal": True}, {"nominal": math.inf}, {"nominal": -1.},
    {"lower_deviation": 4., "upper_deviation": 3.},
    {"lower_deviation": None, "upper_deviation": 3.},
    {"lower_deviation": math.nan},
])
def test_invalid_source_intervals_fail_before_entering_the_solver(change):
    parsed = {**parse_dimension("20^{+3}_{0}"), **change}
    with pytest.raises(ValueError):
        source_dimension_interval(parsed)


@pytest.mark.parametrize("actual", [math.nan, math.inf, -math.inf])
def test_nonfinite_actual_can_never_receive_a_passing_certificate(actual):
    policy = thickness_policy()
    assert not tolerance_interval_check(policy, actual)["passed"]
    with pytest.raises(ValueError):
        tolerance_interval_residual(policy, actual)
