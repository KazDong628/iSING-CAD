"""Object types and current solve evidence cannot be replaced by good fitting."""
from copy import deepcopy

from contour_agent.reconstruction_feedback import reconstruction_feedback, source_object_obligations


def _case():
    graph = {"entities": [{"id": "arc", "type": "ARC", "radius": 20.,
                           "start": [0., 0.], "end": [20., 20.], "center": [0., 20.], "clockwise": False},
                          {"id": "line", "type": "LINE", "start": [20., 20.], "end": [0., 0.]}]}
    constraints = [{"id": "kr", "kind": "radius", "record_id": "r", "value": 20., "entities": ["arc"]},
                   {"id": "ka", "kind": "angle", "record_id": "a", "value": 24., "entities": ["line"]}]
    bindings = {"constraints": constraints, "radius_binding_coverage": {
        "recognized_radius_records": ["r"], "confirmed_arrow_records": ["r"],
        "required_mappings": [{"record_id": "r", "candidate_entity_ids": ["arc"]}]},
        "angle_source_observations": [{"record_id": "a", "nominal": 24., "verified": True,
            "reference_axis": "vertical", "target_candidates": [
                {"entity_id": "line", "whole_line_supported": True}]}]}
    solution = {"accepted": True, "constraints": [{**row, "passed": True} for row in constraints]}
    return graph, bindings, solution


def test_current_source_objects_and_accepted_solver_subset_are_required():
    graph, bindings, solution = _case()
    original = deepcopy((graph, bindings, solution))
    ledger = source_object_obligations(graph, bindings, solution)
    assert ledger["satisfied"] and ledger["unresolved_record_ids"] == []
    assert (graph, bindings, solution) == original
    solution["accepted"] = False
    ledger = source_object_obligations(graph, bindings, solution)
    assert not ledger["satisfied"] and ledger["unresolved_record_ids"] == ["a", "r"]


def test_even_passed_radius_receipt_cannot_certify_a_line_or_wrong_radius():
    graph, bindings, solution = _case()
    graph["entities"][0]["type"] = "LINE"
    ledger = source_object_obligations(graph, bindings, solution)
    assert ledger["missing_object_record_ids"] == ["r"]
    graph["entities"][0].update(type="ARC", radius=20.00001)
    assert not source_object_obligations(graph, bindings, solution)["satisfied"]


def test_angle_on_a_curve_is_a_missing_line_not_completed_numeric_coverage():
    graph, bindings, solution = _case()
    bindings["angle_source_observations"][0]["target_candidates"] = [
        {"entity_id": "arc", "whole_line_supported": False}]
    bindings["constraints"][1]["entities"] = ["arc"]
    ledger = source_object_obligations(graph, bindings, solution)
    assert ledger["missing_object_record_ids"] == ["a"]
    solution["accepted"] = False
    feedback = reconstruction_feedback(graph, bindings, solution)
    assert any(row["code"] == "angle_requires_finite_line_restoration" for row in feedback["issues"])


def test_missing_radius_arrow_and_scope_conflict_remain_unresolved():
    graph, bindings, solution = _case()
    coverage = bindings["radius_binding_coverage"]
    coverage["confirmed_arrow_records"] = []
    coverage["unresolved"] = [{"record_id": "r", "reason": "outside_selected_boundary_or_scope_conflict"}]
    ledger = source_object_obligations(graph, bindings, solution)
    row = next(row for row in ledger["obligations"] if row["record_id"] == "r")
    assert row["status"] == "unresolved" and row["reason"] == "outside_selected_boundary_or_scope_conflict"
    assert not ledger["satisfied"]
