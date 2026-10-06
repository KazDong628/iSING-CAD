from copy import deepcopy

import pytest

from contour_agent.reconstruction_feedback import reconstruction_feedback


def fixture():
    graph = {"candidate_id": "binding-retarget", "units": "mm", "entities": [
        {"id": "g004", "type": "LINE", "start": [0., 0.], "end": [1., 0.]},
        {"id": "g011", "type": "ARC", "radius": 65., "center": [1., 65.],
         "clockwise": False, "start": [1., 0.], "end": [0., 0.]}],
        "annotation_support": [{"record_id": "r039", "kind": "radius", "nominal": 65.,
                                "candidate_entity_id": "g004", "arrowhead_verified": True}]}
    constraint = {"id": "kc039", "kind": "radius", "record_id": "r039", "entities": ["g011"],
                  "value": 65., "source_arrow_verified": True, "passed": True}
    bindings = {"constraints": [constraint], "radius_binding_coverage": {"confirmed_arrow_records": ["r039"]}}
    solution = {"accepted": True, "status": "accepted", "entities": graph["entities"], "constraints": [constraint]}
    return graph, bindings, solution


def test_verified_radius_on_retargeted_object_does_not_report_stale_line():
    graph, bindings, solution = fixture()
    report = reconstruction_feedback(graph, bindings, solution)
    assert report["issues"] == []
    assert report["verified_radius_record_ids"] == ["r039"]
    assert report["bound_record_entities"] == {"r039": ["g011"]}


def test_unsolved_binding_still_reports_actual_current_target():
    graph, bindings, _ = fixture()
    report = reconstruction_feedback(graph, bindings)
    assert report["issues"] == [{"code": "radius_annotation_unbound", "record_id": "r039", "entity_id": "g011",
                                  "stable_id": None, "binding_verified": True, "constraint_satisfied": False,
                                  "arrowhead_verified": True}]
    assert report["verified_radius_record_ids"] == []


def test_real_radius_mismatch_is_not_hidden_by_retargeted_binding():
    graph, bindings, solution = fixture()
    solution = deepcopy(solution)
    solution["entities"][1]["radius"] = 64.
    report = reconstruction_feedback(graph, bindings, solution)
    assert report["issues"][0]["code"] == "radius_value_unresolved"
    assert report["issues"][0]["entity_id"] == "g011"


def test_fresh_constraint_nominal_takes_precedence_over_initial_hypothesis():
    graph, bindings, solution = fixture()
    graph["annotation_support"][0]["nominal"] = 63.
    assert reconstruction_feedback(graph, bindings, solution)["issues"] == []


def test_unbound_record_keeps_initial_target_and_duplicate_hypotheses_are_deduplicated():
    graph, bindings, solution = fixture()
    graph["annotation_support"].append(deepcopy(graph["annotation_support"][0]))
    assert reconstruction_feedback(graph, bindings, solution)["issues"] == []
    report = reconstruction_feedback(graph)
    assert len(report["issues"]) == 1
    assert report["issues"][0]["code"] == "radius_target_is_line"
    assert report["issues"][0]["entity_id"] == "g004"
    assert not report["issues"][0]["binding_verified"]
