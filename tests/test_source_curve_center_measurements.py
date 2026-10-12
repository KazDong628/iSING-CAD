"""Explicit radial-leader centers are symbolic measurement locations, not fits."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from contour_agent.measurement_anchors import (
    derived_anchor_point, source_curve_center_locations, validate_derived_anchor,
)
from contour_agent.parametric_solver import (
    _measurement_anchor_point, _validate_measurement_anchors, _strict_measurement_contract,
)


def source_fixture():
    gray = np.full((500, 500), 255, np.uint8)
    center = np.array([240., 360.])
    observations = []
    for index, tip in enumerate((np.array([80., 70.]), np.array([370., 85.]))):
        direction = (tip-center)/np.linalg.norm(tip-center)
        tail = tip+.45*(center-tip)
        cv2.line(gray, tuple(tip.astype(int)), tuple(center.astype(int)), 0, 2)
        observations.append({"record_id": "r"+str(index), "nominal": 10.+index,
            "segment_px": [tip.tolist(), tail.tolist()], "arrowhead_verified": True,
            "arrowhead": {"tip_px": tip.tolist(), "direction_px": direction.tolist(), "verified": True},
            "shaft_evidence": {"verified": True}, "source_arrow_ownership": {
                "physical_arrow_id": "arrow"+str(index), "status": "globally_unique_source_claim"},
            "target_candidates": [{"entity_id": "g0", "entity_type": "ARC"}] if index == 0 else [
                {"entity_id": "g1", "entity_type": "ARC"}, {"entity_id": "g2", "entity_type": "ARC"}]})
    # Deliberately unrelated fit coordinates: source rays, never this center,
    # identify the measurement station.
    graph = {"entities": [{"id": "g0", "type": "ARC", "start": [15., 0.], "end": [5., 10.],
                           "center": [5., 0.], "radius": 10., "clockwise": False}]}
    return gray, graph, observations, center


def test_observed_common_termination_yields_symbolic_center_without_fitted_target():
    gray, graph, observations, center = source_fixture()
    result = source_curve_center_locations(gray, graph, observations, "y")
    assert len(result) == 1
    row = result[0]
    np.testing.assert_allclose(row["source_px"], center, atol=1e-8)
    assert row["anchor"] == {"type": "curve_center", "entity_id": "g0", "axis": "y"}
    assert row["x"] == 5. and row["y"] == 0.
    assert row["evidence"]["center_semantics_verified"]
    assert not row["evidence"]["nominal_used"] and not row["evidence"]["coordinates_locked"]
    assert row["evidence"]["other_arc_concentricity_not_implicitly_asserted"]
    changed = deepcopy(observations)
    for observation in changed:
        observation["nominal"] *= 20.
    assert source_curve_center_locations(gray, graph, changed, "y") == result


def test_infinite_shaft_crossing_without_common_termination_is_not_circle_center():
    gray, graph, observations, center = source_fixture()
    for row in observations:
        beyond = center-70*np.asarray(row["arrowhead"]["direction_px"])
        cv2.line(gray, tuple(center.astype(int)), tuple(beyond.astype(int)), 0, 2)
    assert source_curve_center_locations(gray, graph, observations, "y") == []


def test_missing_radial_tail_cannot_be_extrapolated_to_dimension_station():
    gray, graph, observations, center = source_fixture()
    point = (np.asarray(observations[0]["segment_px"])[1]+center)/2
    cv2.circle(gray, tuple(point.astype(int)), 18, 255, -1)
    assert source_curve_center_locations(gray, graph, observations, "y") == []


@pytest.mark.parametrize("invalid", ["arrow", "shaft", "ownership", "duplicate", "target"])
def test_unverified_or_ambiguous_source_ownership_does_not_create_center_anchor(invalid):
    gray, graph, observations, center = source_fixture()
    if invalid == "arrow":
        observations[0]["arrowhead_verified"] = False
    elif invalid == "shaft":
        observations[0]["shaft_evidence"]["verified"] = False
    elif invalid == "ownership":
        observations[0]["source_arrow_ownership"]["status"] = "ambiguous_source_claim"
    elif invalid == "duplicate":
        observations[1]["source_arrow_ownership"]["physical_arrow_id"] = "arrow0"
    else:
        observations[0]["target_candidates"].append({"entity_id": "g2", "entity_type": "ARC"})
    assert source_curve_center_locations(gray, graph, observations, "y") == []


def test_center_measurement_recomputes_current_unknowns_and_native_certificate():
    gray, graph, observations, center = source_fixture()
    entity = graph["entities"][0]
    anchor = {"type": "curve_center", "entity_id": "g0", "axis": "y"}
    line = {"id": "line", "type": "LINE", "start": [0., 110.], "end": [50., 110.]}
    constraint = {"id": "measure", "record_id": "r110", "kind": "distance_y",
        "entities": ["g0", "line"], "nodes": [], "value": 110.,
        "source": "source_nominal_measurement_equality",
        "anchors": [anchor, {"type": "support_level", "entity_id": "line", "axis": "y"}]}
    indexed = {"g0": entity, "line": line}
    assert validate_derived_anchor(anchor, indexed, "y") == (("curve_center", "g0", "y"), {"g0"})
    _validate_measurement_anchors(constraint, {}, indexed)
    np.testing.assert_array_equal(_measurement_anchor_point(anchor, {}, indexed), [5., 0.])
    assert _strict_measurement_contract({}, list(indexed.values()), [constraint])["satisfied"]
    entity["center"][1] += .01
    np.testing.assert_array_equal(derived_anchor_point(anchor, indexed), [5., .01])
    result = _strict_measurement_contract({}, list(indexed.values()), [constraint])
    assert result["required_count"] == 1 and not result["satisfied"]
    with pytest.raises(ValueError, match="literal coordinates"):
        validate_derived_anchor({**anchor, "source_px": center.tolist()}, indexed, "y")


def test_joint_optimization_solves_center_station_with_exact_radii_and_tangency():
    from contour_agent.parametric_solver import solve_parametric
    from test_parametric_solver import rounded_graph, dimensions_and_relations
    graph = rounded_graph()
    constraints = dimensions_and_relations()
    constraints[1] = {"id": "center_height", "record_id": "r_center_height", "kind": "distance_y",
        "nodes": [], "entities": ["g001", "g004"], "value": 9., "source": "ocr_local_binding",
        "anchors": [{"type": "curve_center", "entity_id": "g001", "axis": "y"},
                    {"type": "support_level", "entity_id": "g004", "axis": "y"}]}
    before = deepcopy(graph)
    result = solve_parametric(graph, constraints)
    assert result["accepted"], result
    assert graph == before
    assert result["strict_measurement_contract"]["satisfied"]
    assert result["strict_tangent_contract"]["satisfied"]
    assert result["strict_radius_contract"]["satisfied"]
    assert result["diagnostics"]["remaining_shape_dof"] == 0
    entities = {row["id"]: row for row in result["entities"]}
    assert entities["g004"]["start"][1]-entities["g001"]["center"][1] == pytest.approx(9., abs=1e-7)
    assert entities["g004"]["start"][1]-entities["g000"]["start"][1] == pytest.approx(12., abs=1e-7)
