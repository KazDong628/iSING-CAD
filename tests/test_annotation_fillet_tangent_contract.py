"""Fillet tangency is reported only from current, independent source evidence."""
import json

from contour_agent.parametric_pipeline import (
    _fillet_tangent_contract, _record_fillet_tangent_contract,
)


def _example():
    graph = {"entities": [
        {"id": "left", "type": "LINE", "start_node": "v0", "end_node": "v1"},
        {"id": "round", "type": "ARC", "start_node": "v1", "end_node": "v2"},
        {"id": "right", "type": "LINE", "start_node": "v2", "end_node": "v3"},
    ]}
    radius = {"id": "kradius", "kind": "radius", "record_id": "r5",
              "entities": ["round"], "nodes": [], "value": 5.}
    tangent0 = {"id": "krel000", "kind": "tangent", "record_id": None,
                "entities": ["left", "round"], "nodes": ["v1"], "value": None}
    tangent1 = {"id": "krel001", "kind": "tangent", "record_id": None,
                "entities": ["round", "right"], "nodes": ["v2"], "value": None}
    bindings = {"constraints": [radius, tangent0, tangent1],
                "radius_binding_coverage": {"confirmed_arrow_records": ["r5"]},
                "bindings": [{"relation_id": "rel000", "accepted": True,
                              "source": "source_geometry", "evidence": {"verified": True}},
                             {"relation_id": "rel001", "accepted": True,
                              "source": "source_geometry", "evidence": {"verified": True}}]}
    solution = {"accepted": True, "constraints": [{"id": "kradius", "passed": True},
                                {"id": "krel000", "passed": True},
                                {"id": "krel001", "passed": True}]}
    return graph, bindings, solution


def test_both_source_verified_rounded_line_joints_are_audited(tmp_path):
    graph, bindings, solution = _example()
    feedback = {}
    result = _record_fillet_tangent_contract(tmp_path, graph, bindings, solution, feedback)
    assert result["source_verified_joints"] == 2
    assert result["satisfied_source_verified_joints"] == 2
    assert result["all_line_arc_joints_certified"] is True
    assert all(row["status"] == "source_verified_and_satisfied"
               for row in result["fillets"][0]["line_joints"])
    assert feedback["fillet_tangent_contract"] == result
    assert json.loads((tmp_path / "fillet-tangent-contract.json").read_text()) == result


def test_radius_and_plausible_geometry_do_not_fabricate_missing_tangent():
    graph, bindings, solution = _example()
    bindings["bindings"][1]["evidence"] = {"verified": False}
    result = _fillet_tangent_contract(graph, bindings, solution)
    assert result["source_verified_joints"] == 1
    assert result["unverified_joints"] == 1
    assert result["all_line_arc_joints_certified"] is False
    assert {row["status"] for row in result["fillets"][0]["line_joints"]} == {
        "source_verified_and_satisfied", "source_tangency_unverified"}


def test_certified_but_failed_or_missing_solver_receipt_is_not_success():
    graph, bindings, solution = _example()
    solution["constraints"][2]["passed"] = False
    result = _fillet_tangent_contract(graph, bindings, solution)
    assert result["source_verified_joints"] == 2
    assert result["satisfied_source_verified_joints"] == 1
    assert result["all_source_verified_joints_satisfied"] is False
    assert "source_verified_but_unsatisfied" in {
        row["status"] for row in result["fillets"][0]["line_joints"]}
    solution["constraints"] = solution["constraints"][:2]
    result = _fillet_tangent_contract(graph, bindings, solution)
    assert "source_verified_without_solver_receipt" in {
        row["status"] for row in result["fillets"][0]["line_joints"]}


def test_unverified_radius_arrow_cannot_certify_fillet():
    graph, bindings, solution = _example()
    bindings["radius_binding_coverage"]["confirmed_arrow_records"] = []
    result = _fillet_tangent_contract(graph, bindings, solution)
    assert result["arrow_verified_radius_arcs"] == 0
    assert result["all_line_arc_joints_certified"] is False


def test_radius_bound_to_line_is_reported_without_tangent_success():
    graph, bindings, solution = _example()
    graph["entities"][1]["type"] = "LINE"
    result = _fillet_tangent_contract(graph, bindings, solution)
    assert result["fillets"][0]["status"] == "radius_target_not_arc"
    assert result["all_line_arc_joints_certified"] is False


def _design_example():
    graph, bindings, solution = _example()
    for row in bindings["bindings"]:
        row.update(source="source_bound_design_fillet", evidence={
            "verified": True, "evidence_class": "source_bound_design_construction",
            "independent_source_tangent_measurement": False})
    for row in bindings["constraints"][1:]:
        row.update(source="source_geometry", evidence_class="source_bound_design_construction",
                   independent_source_tangent_measurement=False)
    return graph, bindings, solution


def test_source_bound_design_fillet_is_separate_from_measured_source_tangency():
    graph, bindings, solution = _design_example()
    result = _fillet_tangent_contract(graph, bindings, solution)
    assert result["source_verified_joints"] == 0
    assert result["all_source_verified_joints_satisfied"] is False
    assert result["design_constructed_joints"] == 2
    assert result["satisfied_design_constructed_joints"] == 2
    assert result["all_design_constructed_joints_satisfied"] is True
    assert result["all_line_arc_joints_certified"] is True


def test_design_label_without_matching_constraint_or_solver_receipt_is_not_certified():
    graph, bindings, solution = _design_example()
    bindings["constraints"][1].pop("evidence_class")
    solution["constraints"][2]["passed"] = False
    result = _fillet_tangent_contract(graph, bindings, solution)
    assert result["unverified_joints"] == 1
    assert result["satisfied_design_constructed_joints"] == 0
    assert result["all_line_arc_joints_certified"] is False


def test_constructed_design_cannot_be_relabelled_as_independent_source_measurement():
    graph, bindings, solution = _design_example()
    bindings["bindings"][0]["source"] = "source_geometry"
    result = _fillet_tangent_contract(graph, bindings, solution)
    assert result["source_verified_joints"] == 0
    assert result["unverified_joints"] == 1
    assert result["all_line_arc_joints_certified"] is False


def test_rejected_solve_cannot_certify_returned_baseline_from_candidate_receipts():
    for example in (_example, _design_example):
        graph, bindings, solution = example()
        solution["accepted"] = False
        result = _fillet_tangent_contract(graph, bindings, solution)
        assert result["solver_accepted"] is False
        assert result["all_line_arc_joints_certified"] is False
        assert result["satisfied_source_verified_joints"] == 0
        assert result["satisfied_design_constructed_joints"] == 0
        assert result["cohort"] == "bound_arrow_verified_radius_arcs"
