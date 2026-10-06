from copy import deepcopy
import json

import pytest

from contour_agent.parametric_pipeline import _solver_source_observation, _topology_edit_loop
from contour_agent.reconstruction_feedback import (
    geometry_fingerprint, source_failure_feedback, source_mask_interval_diagnostics,
)
from contour_agent.topology_search import (
    SearchBudget, candidate_priority, operation_fingerprint, retain_distinct,
)


def candidate(name, length=2.):
    return {"id": name, "graph": {"candidate_id": name, "units": "mm", "entities": [
        {"id": "g000", "type": "LINE", "start": [0., 0.], "end": [length, 0.]},
        {"id": "g001", "type": "LINE", "start": [length, 0.], "end": [0., 0.]},
    ]}}


def test_operation_visited_is_parent_scoped_and_budget_is_finite():
    now = [0.]
    budget = SearchBudget(max_preflights=2, max_provider_calls=2, max_seconds=10., clock=lambda: now[0])
    operation = {"action": "split_chain_at_source_features", "entity_ids": ["g000"], "record_id": "r001"}
    first = geometry_fingerprint(candidate("first")["graph"])
    other = geometry_fingerprint(candidate("other", 3.)["graph"])
    assert budget.visit_operation(first, operation)[0]
    assert not budget.visit_operation(first, deepcopy(operation))[0]
    assert budget.visit_operation(other, operation)[0]
    assert operation_fingerprint(first, operation) != operation_fingerprint(other, operation)
    assert budget.reserve_preflight() and budget.reserve_preflight()
    assert not budget.reserve_preflight()
    assert budget.reserve_provider("editor", first) == 10.
    assert budget.reserve_provider("evaluator", first) == 10.
    assert budget.reserve_provider("editor", other) is None
    now[0] = 11.
    assert budget.remaining_seconds() == 0.
    assert not budget.reserve_preflight()
    json.dumps(budget.summary(), allow_nan=False)


def test_distinct_beam_prefers_fresh_verified_constraints_over_construction_and_count():
    verified = candidate("verified", 3.)
    verified["constraint_feedback"] = {"solver_accepted": True, "constraint_count": 2,
        "verified_radius_record_ids": ["r1", "r1"], "satisfied_record_ids": ["r1"],
        "remaining_shape_dof": 4, "source_validation": {"passed": True}}
    constructed = candidate("constructed", 4.)
    constructed["graph"]["entities"][0]["radius_binding"] = {"record_id": "r2", "nominal": 5.}
    constructed["constraint_feedback"] = {"remaining_shape_dof": 0}
    failed = candidate("failed", 5.)
    failed["constraint_feedback"] = {"solver_accepted": True, "constraint_count": 3,
        "verified_radius_record_ids": ["r1", "r2", "r3"], "satisfied_record_ids": ["r1", "r2", "r3"],
        "source_validation": {"passed": False}, "remaining_shape_dof": 0}
    alias = deepcopy(verified)
    alias["id"] = "verified-alias"
    result = retain_distinct([constructed, failed, alias, verified], width=3, preferred_id="verified")
    assert result[0]["id"] == "verified"
    assert len(result) == 3
    assert candidate_priority(verified) < candidate_priority(constructed) < candidate_priority(failed)
    assert candidate_priority(failed)[1:4] == (0, 0, 0)


def test_original_mask_budget_is_mapped_without_using_reference_tolerance():
    baseline = {"oracle_mask_conditioned": True, "scale": {"pixels_per_mm": 2.},
        "curve_fit": {"total_deviation_budget_px": 3.8},
        "extraction": {"raw_polyline_px": [[0., 0.], [10., 0.], [0., -6.]]},
        "coordinate_system": {"origin_source_px": [0., 0.], "units": "mm"},
        "reference_tolerance_mm": 1.}
    observation = _solver_source_observation(baseline, candidate("base")["graph"])
    assert observation["boundary_error_budget"] == {"units": "mm", "maximum_deviation": 1.9,
        "sampling_step": .25, "source": "initial_curve_fit_total_deviation_budget_px"}
    assert observation["reference_dxf_read"] is False


def test_already_bound_radius_mask_failure_remains_an_editor_target():
    feedback = {"issues": [], "bound_record_entities": {"r005": ["g000"]},
        "bound_record_ids": ["r005"], "satisfied_record_ids": ["r005"], "verified_radius_record_ids": ["r005"]}
    graph = {"entities": [{"id": "g000", "type": "ARC", "radius": 5.}]}
    diagnostics = {"original_deviation_budget_px": 3.8, "entities": [{"entity_id": "g000",
        "exceeds_original_budget": True, "conservative_max_deviation_px": 6.4, "failed_quarters": [1, 2]}]}
    result = source_failure_feedback(feedback, graph, {"passed": False, "reasons": ["original_oracle_mask_budget_exceeded"]}, diagnostics)
    issue = result["issues"][0]
    assert issue["code"] == "bound_radius_source_interval_failed"
    assert issue["record_id"] == "r005" and issue["binding_verified"]
    assert issue["source_failed_quarters"] == [1, 2]
    assert "not_radius_relaxation" in issue["required_action"]


def test_source_mask_diagnostics_localize_shifted_line_against_immutable_ring():
    baseline = {"curve_fit": {"total_deviation_budget_px": .5},
        "extraction": {"raw_polyline_px": [[0., 0.], [10., 0.], [10., -6.], [0., -6.]]}}
    points = [[0., 2.], [10., 2.], [10., 6.], [0., 6.]]
    graph = {"units": "pixel", "entities": [{"id": f"g{i:03d}", "type": "LINE", "start": point,
        "end": points[(i + 1) % 4]} for i, point in enumerate(points)]}
    diagnostics = source_mask_interval_diagnostics(baseline, graph, graph["entities"])
    top = next(row for row in diagnostics["entities"] if row["entity_id"] == "g000")
    assert top["exceeds_original_budget"] and top["conservative_max_deviation_px"] > 2.
    assert top["failed_quarters"] == [1, 2, 3, 4]
    assert diagnostics["original_deviation_budget_px"] == .5
    assert diagnostics["ground_truth_used"] is False and not diagnostics["acceptance_thresholds_changed"]


def _mock_preflight(monkeypatch):
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_source_validation", lambda *args: {"passed": True})
    monkeypatch.setattr("contour_agent.parametric_pipeline._preflight_input_identity", lambda *args:
                        {"schema_version":"synthetic_preflight_identity","reference_geometry_used":False})
    def bindings(image, document, baseline, graph, output, **kwargs):
        constraints = [] if graph["candidate_id"] == "base" else [
            {"kind": "distance", "record_id": "r1", "entities": ["g000"]}]
        return {"constraints": constraints}
    def solve(graph, constraints, **kwargs):
        return {"status": "accepted", "accepted": True, "entities": graph["entities"],
            "constraints": [{**row, "passed": True} for row in constraints],
            "diagnostics": {"remaining_shape_dof": 4 if graph["candidate_id"] == "base" else 2}}
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings", bindings)
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric", solve)
    monkeypatch.setattr("contour_agent.parametric_pipeline._solved_source_validation",
        lambda image, doc, baseline, graph, entities: {"passed": graph["candidate_id"] != "explore",
            "reasons": ["original_oracle_mask_budget_exceeded"] if graph["candidate_id"] == "explore" else []})
    monkeypatch.setattr("contour_agent.source_support_diagnostics.source_support_diagnostics", lambda *args: {"status": "measured"})


@pytest.mark.parametrize("allow_child", [False, True])
def test_exploratory_branch_can_produce_a_publishable_child_but_is_never_selected_itself(tmp_path, monkeypatch, allow_child):
    _mock_preflight(monkeypatch)
    calls = []
    def stage(image, document, baseline, parent, bundle, output, **kwargs):
        calls.append((kwargs["round_index"], parent["id"]))
        proposed = parent
        trusted, exploratory = [parent["id"]], []
        if kwargs["round_index"] == 1:
            explore = candidate("explore", 3.)
            explore["constraint_feedback"] = kwargs["verifier"](explore)
            bundle.setdefault("candidates", []).append(explore)
            exploratory = ["explore"]
        elif parent["id"] == "explore":
            trusted = []
            if allow_child:
                proposed = candidate("ready-child", 4.)
                proposed["constraint_feedback"] = kwargs["verifier"](proposed)
                bundle["candidates"].append(proposed)
                trusted = [proposed["id"]]
        return proposed, {"base_candidate_id": parent["id"], "final_candidate_id": proposed["id"],
            "acceptance_gate": {"accepted": proposed["id"] != parent["id"]},
            "execution": {"operations": []}, "editor": {}, "evaluator": {},
            "trusted_candidate_ids": trusted, "exploratory_candidate_ids": exploratory}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_stage", stage)
    final, report = _topology_edit_loop("unused", {}, {}, candidate("base"), {}, tmp_path, max_rounds=2)
    assert final["id"] == ("ready-child" if allow_child else "base")
    assert final["id"] != "explore" and not final.get("diagnostic_only")
    assert (2, "explore") in calls
    assert report["rounds"][0]["final_candidate_id"] == "base"
    assert report["rounds"][0]["exploratory_candidate_ids"] == ["explore"]
    assert len(report["rounds"][1]["branches"]) == 2
    assert report["budget"]["preflights_used"] <= 18
    assert all(row["validation"]["passed"] for row in report["final_source_rechecks"])


def test_loop_provider_budget_is_shared_across_branches(tmp_path, monkeypatch):
    _mock_preflight(monkeypatch)
    def stage(image, document, baseline, parent, bundle, output, **kwargs):
        assert kwargs["provider_guard"]("editor") is not None
        assert kwargs["provider_guard"]("evaluator") is not None
        return parent, {"base_candidate_id": parent["id"], "final_candidate_id": parent["id"],
            "acceptance_gate": {"accepted": False}, "execution": {"operations": []},
            "editor": {}, "evaluator": {}}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_stage", stage)
    final, report = _topology_edit_loop("unused", {}, {}, candidate("base"), {}, tmp_path,
        use_api=True, max_provider_calls=2)
    assert final["id"] == "base" and len(report["rounds"]) == 1
    assert report["stop_reason"] == "topology_search_provider_budget_exhausted"
    assert report["budget"]["provider_calls_reserved"] == 2


def test_rejected_solver_candidate_is_localized_instead_of_retained_entities(tmp_path,monkeypatch):
    _mock_preflight(monkeypatch)
    baseline_entities=candidate("base")["graph"]["entities"]
    rejected_entities=candidate("rejected",7.)["graph"]["entities"]
    monkeypatch.setattr("contour_agent.parametric_solver.solve_parametric",lambda *args,**kwargs:
        {"status":"source_budget_search_exhausted","accepted":False,"entities":baseline_entities,
         "candidate_entities":rejected_entities,"diagnostics":{"source_budget_search":{"infeasibility_proven":False}}})
    seen=[]
    monkeypatch.setattr("contour_agent.parametric_pipeline._solved_source_validation",
        lambda image,doc,baseline,graph,entities:seen.append(entities) or
            {"passed":False,"reasons":["original_oracle_mask_budget_exceeded"]})
    observed=[]
    def stage(image,doc,baseline,parent,bundle,output,**kwargs):
        observed.append(kwargs["feedback"])
        return parent,{"acceptance_gate":{"accepted":False},"execution":{"operations":[]}}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_stage",stage)
    _topology_edit_loop("unused",{}, {},candidate("base"),{},tmp_path,max_rounds=1)
    assert seen==[rejected_entities]
    assert observed[0]["solver_status"]=="source_budget_search_exhausted"
    assert observed[0]["source_mask_diagnostics"]["geometry_stage"]=="rejected_solver_candidate"
    assert not observed[0]["post_solve_source_accepted"]
