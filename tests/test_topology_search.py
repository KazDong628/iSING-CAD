from copy import deepcopy
import json

import pytest

from contour_agent.parametric_pipeline import (_edit_observation_parent, _preflight_input_identity,
                                               _solver_source_observation, _source_joint_bootstrap,
                                               _topology_edit_loop, _topology_edit_stage)
from contour_agent.reconstruction_feedback import (
    geometry_fingerprint, source_failure_feedback, source_mask_interval_diagnostics,
)
from contour_agent.topology_search import (
    SearchBudget, candidate_priority, operation_fingerprint, preflight_admissible,
    near_nominal_joint_bootstrap_refits, preflight_order, retain_distinct,
)


def _near_nominal_joint_fixture():
    graph = {"units": "mm", "source_grid_pitch_px": 4., "proposal_tolerance_px": 2.,
        "entities": [
            {"id": "g000", "type": "ARC", "start": [0., 0.], "end": [10., 0.], "radius": 14.9},
            {"id": "g001", "type": "LINE", "start": [10., 0.], "end": [10., 10.]},
            {"id": "g002", "type": "ARC", "start": [10., 10.], "end": [0., 0.], "radius": 25.},
        ],
        "annotation_support": [{"kind": "radius", "status": "candidate_supported",
                                "record_id": "r-radius", "candidate_entity_id": "g000",
                                "arrowhead_verified": True, "target_gap_px": .5}],
        "angle_source_observations": [{"record_id": "r-angle", "nominal": 24.,
            "reference_axis": "vertical", "verified": True,
            "source_line": {"start_px": [0., 0.], "end_px": [24., 40.]},
            "target_candidates": [{"entity_id": "g002", "supported_span_px": 24.,
                                   "whole_line_supported": False}]}]}
    inventory = [{"record_id": "r-radius", "kind": "radius", "nominal": 15., "text": "R15"},
                 {"record_id": "r-angle", "kind": "angle", "nominal": 24., "text": "24°"}]
    feedback = {"satisfied_record_ids": [],
                "radius_binding_coverage": {"unresolved": [{"record_id": "r-radius"}]}}
    return graph, inventory, feedback


def test_near_nominal_joint_refit_requires_arrow_compatible_radius_and_unmet_angle():
    graph, inventory, feedback = _near_nominal_joint_fixture()
    expected = {"action": "refit_chain_as_annotated_arc", "entity_ids": ["g000"],
                "record_id": "r-radius", "evidence_tags": ["annotation_target", "source_boundary"]}
    assert near_nominal_joint_bootstrap_refits(graph, inventory, feedback) == [expected]
    missing_arrow = deepcopy(graph)
    missing_arrow["annotation_support"][0]["arrowhead_verified"] = False
    assert near_nominal_joint_bootstrap_refits(missing_arrow, inventory, feedback) == []
    incompatible = deepcopy(graph)
    incompatible["entities"][0]["radius"] = 30.
    assert near_nominal_joint_bootstrap_refits(incompatible, inventory, feedback) == []
    missing_angle = deepcopy(graph)
    missing_angle["angle_source_observations"] = []
    assert near_nominal_joint_bootstrap_refits(missing_angle, inventory, feedback) == []


def test_source_joint_bootstrap_rejects_refit_without_fresh_joint(tmp_path, monkeypatch):
    graph, inventory, feedback = _near_nominal_joint_fixture()
    selected = {"id": "base", "graph": graph}
    bundle = {"annotation_inventory": inventory}
    calls = []
    def execute(image, document, baseline, parent, source_bundle, operations, output):
        calls.append(operations)
        return [{"id": "intermediate", "graph": deepcopy(graph)}], {"operations": [{"status": "accepted_as_candidate"}]}
    monkeypatch.setattr("contour_agent.topology_editing.execute_topology_edits", execute)
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_source_validation",
                        lambda *args: {"passed": True, "reasons": []})
    monkeypatch.setattr("contour_agent.topology_search.refresh_source_angle_observations",
                        lambda image, document, baseline, current: current.update(angle_source_observations=[]) or [])
    compound, audit, source = _source_joint_bootstrap(
        "unused", {}, {}, selected, bundle, tmp_path, feedback,
        remaining_seconds=lambda: 600., edit_time_reserve=300.)
    assert compound is None and source is None
    assert audit["reason"] == "source_joint_bootstrap_missing_verified_two_radius_joint"
    assert len(calls) == 1
    assert audit["intermediate"]["source_validation_passed"] is True


def test_source_joint_bootstrap_rechecks_time_before_second_source_edit(tmp_path, monkeypatch):
    graph, inventory, feedback = _near_nominal_joint_fixture()
    selected = {"id": "base", "graph": graph}
    calls = []
    def execute(image, document, baseline, parent, bundle, operations, output):
        calls.append(operations)
        return [{"id": "intermediate", "graph": deepcopy(graph)}], {"operations": [{"status": "accepted_as_candidate"}]}
    monkeypatch.setattr("contour_agent.topology_editing.execute_topology_edits", execute)
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_source_validation",
                        lambda *args: {"passed": True, "reasons": []})
    monkeypatch.setattr("contour_agent.topology_search.refresh_source_angle_observations",
                        lambda *args: [])
    monkeypatch.setattr("contour_agent.topology_search.near_nominal_joint_bootstrap_refits",
                        lambda *args, **kwargs: [{"action": "refit_chain_as_annotated_arc",
                            "entity_ids": ["g000"], "record_id": "r-radius"}])
    monkeypatch.setattr("contour_agent.topology_search.unmet_source_angle_joint_operations",
                        lambda *args, **kwargs: [{"action": "restore_annotated_line_support",
                            "entity_ids": ["g000", "g002"], "record_id": "r-angle"}])
    remaining = iter([600., 600., 315.])
    compound, audit, source = _source_joint_bootstrap(
        "unused", {}, {}, selected, {"annotation_inventory": inventory}, tmp_path, feedback,
        remaining_seconds=lambda: next(remaining), edit_time_reserve=300.)
    assert compound is None and source is None
    assert audit["reason"] == "source_joint_bootstrap_edit_time_reserved"
    assert audit["joint_witness"]["angle_record_id"] == "r-angle"
    assert len(calls) == 1


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
    assert budget.preflight_block_reason() == "topology_search_preflight_count_budget_exhausted"
    assert not budget.reserve_preflight()
    assert budget.reserve_provider("editor", first) == 10.
    assert budget.reserve_provider("evaluator", first) == 10.
    assert budget.reserve_provider("editor", other) is None
    now[0] = 11.
    assert budget.remaining_seconds() == 0.
    assert budget.preflight_block_reason() == "topology_search_time_budget_exhausted"
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


def test_preflight_order_reserves_angular_line_and_combination_before_other_singles():
    ordinary = candidate("ordinary")
    angular = candidate("angular", 3.)
    angular["graph"]["source_evidence"] = {"topology_edit": {
        "action": "restore_annotated_line_support", "resegmentation_applied": True,
        "angle_support_record_ids": ["r035"]}}
    combined = candidate("combined", 4.)
    combined["graph"]["source_evidence"] = {"topology_edit": {
        "action": "apply_nonoverlapping_edits", "operations": []}}
    rows = [ordinary, angular, combined]
    valid = {row["id"]: {"passed": True} for row in rows}
    assert [row["id"] for row in preflight_order(rows, valid)] == ["angular", "combined", "ordinary"]
    valid["angular"]["passed"] = False
    assert [row["id"] for row in preflight_order(rows, valid)] == ["combined", "ordinary", "angular"]


def test_stage_preflights_angular_line_and_combination_before_time_budget_skips_single(tmp_path, monkeypatch):
    import contour_agent.topology_editing as editing
    import contour_agent.planning_provider as planning

    base = candidate("base")
    ordinary = candidate("ordinary", 3.)
    angular = candidate("angular", 4.)
    angular["graph"]["source_evidence"] = {"topology_edit": {
        "action": "restore_annotated_line_support", "resegmentation_applied": True,
        "angle_support_record_ids": ["r035"], "record_id": "r035"}}
    combined = candidate("combined", 5.)
    combined["graph"]["source_evidence"] = {"topology_edit": {
        "action": "apply_nonoverlapping_edits", "operations": [angular["graph"]["source_evidence"]["topology_edit"]]}}
    edited = [ordinary, angular, combined]
    operations = [{"action": "restore_annotated_line_support", "entity_ids": ["g000"], "record_id": "r035"}]
    monkeypatch.setattr(editing, "propose_annotation_arc_edits", lambda *args, **kwargs: operations)
    monkeypatch.setattr(editing, "execute_topology_edits", lambda *args, **kwargs: (edited, {
        "operations": [{"operation_index": index, "operation": operations[0], "candidate_id": row["id"],
                        "status": "accepted_as_candidate"} for index, row in enumerate(edited, 1)]}))
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_source_validation",
                        lambda *args: {"passed": True, "reasons": []})
    def local_evaluation(rows, *, max_candidates=5):
        ids = [row["id"] for row in rows]
        return {"admissible_candidate_ids": ids, "evaluated": [
            {"candidate_id": name, "admissible": True, "score": 1., "metrics": {
                "source_boundary_support": 1., "unsupported_primitive_count": 0, "entity_count": 2}}
            for name in ids]}
    monkeypatch.setattr(planning, "evaluate_candidates", local_evaluation)
    seen = []
    def verifier(row, *, source_validation=None):
        assert source_validation == {"passed": True, "reasons": []}
        seen.append(row["id"])
        if len(seen) > 2:
            return {"solver_status": "preflight_not_run", "preflight_skip_reason": "topology_search_time_budget_exhausted",
                    "binding_status": "not_run", "source_validation": {"passed": False}}
        return {"solver_status": "accepted", "solver_accepted": True, "constraint_count": 0,
                "binding_status": "local", "source_validation": {"passed": True}, "issue_count": 0}
    _, report = _topology_edit_stage("unused", {}, {}, base,
        {"annotation_inventory": [], "candidates": [base], "generation": {}}, tmp_path,
        verifier=verifier, feedback={"issue_count": 1})
    assert seen == ["angular", "combined", "ordinary"]
    assert [row["status"] for row in report["preflight_queue"]] == ["passed", "passed", "not_run_budget"]
    assert [row["preflight_status"] for row in report["execution"]["operations"]] == [
        "not_run_budget", "passed", "passed"]
    assert preflight_admissible(ordinary["constraint_feedback"]) is False
    assert base["id"] in report["trusted_candidate_ids"]


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


def test_preflight_keeps_binding_graph_immutable_and_caches_only_exact_input(tmp_path, monkeypatch):
    _mock_preflight(monkeypatch)
    monkeypatch.setattr("contour_agent.parametric_pipeline._preflight_input_identity", _preflight_input_identity)
    source = tmp_path / "source.png"
    source.write_bytes(b"fixed source for input identity")
    calls = []
    segment = [[1., 2.], [3., 4.]]
    def bindings(image, document, baseline, graph, output, **kwargs):
        hinted = bool(graph.get("radius_source_segment_hypotheses"))
        calls.append(hinted)
        return {"constraints": [] if hinted else [{"kind": "radius", "record_id": "r045",
            "entities": ["g000"], "value": 15.}],
            "angle_source_observations": [{"record_id": "a1", "verified": True}],
            "radius_binding_coverage": {"confirmed_arrow_records": ["r045"],
                "required_mappings": [{"record_id": "r045", "source_evidence": [
                    {"segment_px": segment}]}]}}
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings", bindings)
    base = candidate("base")
    original = deepcopy(base["graph"])
    observed = {}
    def stage(image, document, baseline, parent, bundle, output, **kwargs):
        assert parent["graph"] == original
        assert parent["edit_source_observations"]["radius_source_segment_hypotheses"][0]["source_evidence"]["segment_px"] == segment
        assert _edit_observation_parent(parent)["graph"]["radius_source_segment_hypotheses"]
        assert _edit_observation_parent(parent)["graph"]["angle_source_observations"]
        assert kwargs["verifier"](parent)["bound_record_ids"] == ["r045"]
        changed = deepcopy(parent)
        changed["graph"]["radius_source_segment_hypotheses"] = deepcopy(
            parent["edit_source_observations"]["radius_source_segment_hypotheses"])
        observed["changed"] = kwargs["verifier"](changed)
        assert parent["graph"] == original
        return parent, {"base_candidate_id": parent["id"], "final_candidate_id": parent["id"],
            "acceptance_gate": {"accepted": False}, "execution": {"operations": []},
            "editor": {}, "evaluator": {}, "trusted_candidate_ids": [parent["id"]]}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_stage", stage)
    final, report = _topology_edit_loop(source, {}, {}, base, {}, tmp_path / "run", max_rounds=1)
    assert final["graph"] == original
    assert calls == [False, True]
    assert report["budget"]["preflights_used"] == 2
    assert final["constraint_feedback"]["bound_record_ids"] == ["r045"]
    assert observed["changed"]["bound_record_ids"] == []
    artifact = tmp_path / "run" / "topology-iterations" / "constraint-preflight"
    exact = [path for path in artifact.iterdir() if
             (path / "reconstruction-feedback.json").is_file() and
             json.loads((path / "reconstruction-feedback.json").read_text())["bound_record_ids"] == ["r045"]]
    assert len(exact) == 1
    assert json.loads((exact[0] / "topology.json").read_text()) == original
    assert json.loads((exact[0] / "preflight-input-identity.json").read_text()) == _preflight_input_identity(
        source, {}, {}, original)


def test_preflight_rejects_mutated_binding_input_without_claiming_artifact(tmp_path, monkeypatch):
    _mock_preflight(monkeypatch)
    def bindings(image, document, baseline, graph, output, **kwargs):
        graph["radius_source_segment_hypotheses"] = [{"record_id": "r045", "kind": "radius",
            "source_evidence": {"segment_px": [[1., 2.], [3., 4.]]}}]
        return {"constraints": []}
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings", bindings)
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_stage",
        lambda image, document, baseline, parent, bundle, output, **kwargs: (parent,
            {"base_candidate_id": parent["id"], "final_candidate_id": parent["id"],
             "acceptance_gate": {"accepted": False}, "execution": {"operations": []},
             "editor": {}, "evaluator": {}, "trusted_candidate_ids": [parent["id"]]}))
    base = candidate("base")
    original = deepcopy(base["graph"])
    final, report = _topology_edit_loop("unused", {}, {}, base, {}, tmp_path, max_rounds=1)
    assert final["graph"] == original
    assert final.get("edit_source_observations") is None
    assert final["constraint_feedback"]["solver_status"] == "preflight_failed"
    artifact = next((tmp_path / "topology-iterations" / "constraint-preflight").iterdir())
    assert not (artifact / "topology.json").exists()
    assert not (artifact / "preflight-input-identity.json").exists()


def test_preflight_rejects_late_source_check_mutation_before_certifying(tmp_path, monkeypatch):
    _mock_preflight(monkeypatch)
    def mutate_after_solve(image, document, baseline, graph, entities):
        graph["radius_source_segment_hypotheses"] = [{"record_id": "r045", "kind": "radius",
            "source_evidence": {"segment_px": [[1., 2.], [3., 4.]]}}]
        return {"passed": True, "reasons": []}
    monkeypatch.setattr("contour_agent.parametric_pipeline._solved_source_validation",
                        mutate_after_solve)
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_stage",
        lambda image, document, baseline, parent, bundle, output, **kwargs: (parent,
            {"base_candidate_id": parent["id"], "final_candidate_id": parent["id"],
             "acceptance_gate": {"accepted": False}, "execution": {"operations": []},
             "editor": {}, "evaluator": {}, "trusted_candidate_ids": [parent["id"]]}))
    base = candidate("base")
    original = deepcopy(base["graph"])
    final, _ = _topology_edit_loop("unused", {}, {}, base, {}, tmp_path, max_rounds=1)
    assert final["graph"] == original
    assert final.get("edit_source_observations") is None
    assert final["constraint_feedback"]["solver_status"] == "preflight_failed"
    assert final["constraint_feedback"]["source_validation"]["reasons"] == [
        "preflight_input_mutated_during_preflight"]
    artifact = next((tmp_path / "topology-iterations" / "constraint-preflight").iterdir())
    assert not (artifact / "topology.json").exists()
    assert not (artifact / "preflight-input-identity.json").exists()


def test_edit_stage_uses_fresh_source_observations_only_on_proposal_copy(tmp_path, monkeypatch):
    import contour_agent.topology_editing as editing
    import contour_agent.planning_provider as planning

    base = candidate("base")
    original = deepcopy(base["graph"])
    base["edit_source_observations"] = {
        "angle_source_observations": [{"record_id": "a1", "verified": True}],
        "radius_source_segment_hypotheses": [{"record_id": "r045", "kind": "radius",
            "source_evidence": {"segment_px": [[1., 2.], [3., 4.]]}}]}
    overlay = tmp_path / "overlay.png"
    overlay.write_bytes(b"local overlay")
    base["overlay_path"] = str(overlay)
    operation = {"action": "refit_chain_as_annotated_arc", "entity_ids": ["g000"], "record_id": "r045"}
    seen = []
    def checked_graph(name, graph):
        seen.append(name)
        assert graph["angle_source_observations"] == base["edit_source_observations"]["angle_source_observations"]
        assert graph["radius_source_segment_hypotheses"] == base["edit_source_observations"]["radius_source_segment_hypotheses"]
    class Editor:
        def propose(self, image, overlay_path, parent, inventory, **kwargs):
            checked_graph("provider", parent["graph"])
            return {"schema_success": True, "operations": [operation], "status": "completed"}
    def propose(graph, inventory, **kwargs):
        checked_graph("local", graph)
        return [operation]
    def execute(image, document, baseline, parent, bundle, operations, output):
        checked_graph("executor", parent["graph"])
        return [], {"operations": []}
    monkeypatch.setattr(editing, "propose_annotation_arc_edits", propose)
    monkeypatch.setattr(editing, "execute_topology_edits", execute)
    monkeypatch.setattr(planning, "evaluate_candidates", lambda rows, **kwargs: {
        "admissible_candidate_ids": ["base"], "evaluated": [{"candidate_id": "base",
            "admissible": True, "score": 1., "metrics": {"entity_count": 2,
            "source_boundary_support": 1., "unsupported_primitive_count": 0}}]})
    final, _ = _topology_edit_stage("unused", {}, {}, base,
        {"annotation_inventory": [], "candidates": [base]}, tmp_path,
        editor_provider=Editor(), use_api=True,
        provider_guard=lambda role: 10., feedback={"constraint_count": 0})
    assert seen == ["provider", "local", "executor"]
    assert final is base and base["graph"] == original


def test_joint_priority_uses_edit_observations_without_mutating_ranked_graph(monkeypatch):
    import contour_agent.topology_search as search

    base = candidate("base")
    original = deepcopy(base["graph"])
    base["constraint_feedback"] = {"solver_accepted": True, "constraint_count": 0,
        "satisfied_record_ids": [], "source_validation": {"passed": True}}
    base["edit_source_observations"] = {"angle_source_observations": [{"record_id": "a1"}]}
    monkeypatch.setattr(search, "_unmet_source_angle_joint", lambda row, *args:
        bool(row["graph"].get("angle_source_observations")))
    assert candidate_priority(base, annotation_inventory=[{"record_id": "a1"}])[3] == -1
    assert base["graph"] == original


def test_initial_source_seeds_are_preflighted_and_visited_in_first_round(tmp_path, monkeypatch):
    _mock_preflight(monkeypatch)
    parents = []
    def stage(image, document, baseline, parent, bundle, output, **kwargs):
        parents.append((kwargs["round_index"], parent["id"]))
        return parent, {"base_candidate_id": parent["id"], "final_candidate_id": parent["id"],
            "acceptance_gate": {"accepted": False}, "execution": {"operations": []},
            "editor": {}, "evaluator": {}, "trusted_candidate_ids": [parent["id"]]}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_stage", stage)
    base, first, second = candidate("base"), candidate("first", 3.), candidate("second", 4.)
    final, report = _topology_edit_loop("unused", {}, {}, base,
        {"candidates": [base, first, second]}, tmp_path, max_rounds=1,
        initial_seed_ids=("base", "first", "second"))
    assert parents == [(1, "first"), (1, "second"), (1, "base")]
    assert report["initial_seed_audit"]["first_round_parent_ids"] == ["first", "second", "base"]
    assert report["initial_seed_audit"]["admitted_candidate_ids"] == ["first", "second"]
    assert final["id"] in {"first", "second"}
    assert report["final_selection_origin"] == "initial_seed_promotion"
    assert report["accepted_seed_promotion_count"] == 1
    assert report["accepted_local_edit_count"] == 0
    assert report["budget"]["preflights_used"] == 3
    assert report["budget"]["preflights_used"] <= report["budget"]["max_preflights"]


def test_compound_source_joint_uses_one_numeric_preflight_before_original_seeds(tmp_path, monkeypatch):
    _mock_preflight(monkeypatch)
    base, alternate, compound = candidate("base"), candidate("alternate", 3.), candidate("seed-joint-base", 4.)
    parents = []
    def stage(image, document, baseline, parent, bundle, output, **kwargs):
        parents.append(parent["id"])
        return parent, {"base_candidate_id": parent["id"], "final_candidate_id": parent["id"],
            "acceptance_gate": {"accepted": False}, "execution": {"operations": []},
            "editor": {}, "evaluator": {}, "trusted_candidate_ids": [parent["id"]]}
    def bootstrap(*args, **kwargs):
        return compound, {"status": "source_candidate", "operations": [
            {"action": "refit_chain_as_annotated_arc", "entity_ids": ["g000"], "record_id": "r1"},
            {"action": "restore_annotated_line_support", "entity_ids": ["g000", "g001"], "record_id": "r2"}],
            "intermediate": {"geometry_sha256": "intermediate-source-geometry"},
            "ground_truth_used": False}, {"passed": True, "reasons": []}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_stage", stage)
    monkeypatch.setattr("contour_agent.parametric_pipeline._source_joint_bootstrap", bootstrap)
    _, report = _topology_edit_loop("unused", {}, {}, base,
        {"candidates": [base, alternate]}, tmp_path, max_rounds=1,
        initial_seed_ids=("alternate",))
    assert parents == ["seed-joint-base", "base"]
    assert report["budget"]["preflights_used"] == 2
    assert report["initial_seed_audit"]["admitted_candidate_ids"] == ["seed-joint-base"]
    assert report["initial_seed_audit"]["attempts"][0]["reason"] == "source_joint_bootstrap_preferred_for_edit_budget"
    assert report["initial_seed_audit"]["source_joint_bootstrap"]["status"] == "admitted"


def test_failed_compound_preflight_reserves_measured_time_before_fallback_seed(tmp_path, monkeypatch):
    _mock_preflight(monkeypatch)
    clock = [0.]
    monkeypatch.setattr("contour_agent.topology_search.time.monotonic", lambda: clock[0])
    def bindings(image, document, baseline, graph, output, **kwargs):
        clock[0] += 120. if graph["candidate_id"] == "base" else 100.
        return {"constraints": []}
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings", bindings)
    base, alternate, compound = candidate("base"), candidate("alternate", 3.), candidate("explore", 4.)
    monkeypatch.setattr("contour_agent.parametric_pipeline._source_joint_bootstrap",
        lambda *args, **kwargs: (compound,
            {"status": "source_candidate", "operations": [], "ground_truth_used": False},
            {"passed": True, "reasons": []}))
    parents = []
    def stage(image, document, baseline, parent, bundle, output, **kwargs):
        parents.append(parent["id"])
        return parent, {"base_candidate_id": parent["id"], "final_candidate_id": parent["id"],
            "acceptance_gate": {"accepted": False}, "execution": {"operations": []},
            "editor": {}, "evaluator": {}, "trusted_candidate_ids": [parent["id"]]}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_stage", stage)
    _, report = _topology_edit_loop("unused", {}, {}, base,
        {"candidates": [base, alternate]}, tmp_path, max_rounds=1,
        initial_seed_ids=("alternate",))
    assert parents == ["base"]
    assert report["budget"]["preflights_used"] == 2
    assert report["initial_seed_audit"]["selected_preflight"]["elapsed_seconds"] == 120.
    assert report["initial_seed_audit"]["source_joint_bootstrap"]["status"] == "rejected"
    assert report["initial_seed_audit"]["attempts"][0]["reason"] == "post_compound_failure_edit_time_reserved"
    assert report["budget"]["remaining_seconds"] == 380.


def test_initial_seeds_require_distinct_source_valid_preflight(tmp_path, monkeypatch):
    _mock_preflight(monkeypatch)
    def source_check(image, baseline, graph):
        passed = graph["candidate_id"] != "invalid"
        return {"passed": passed, "reasons": [] if passed else ["source_stroke_support_degraded"]}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_source_validation", source_check)
    parents = []
    def stage(image, document, baseline, parent, bundle, output, **kwargs):
        parents.append(parent["id"])
        return parent, {"base_candidate_id": parent["id"], "final_candidate_id": parent["id"],
            "acceptance_gate": {"accepted": False}, "execution": {"operations": []},
            "editor": {}, "evaluator": {}, "trusted_candidate_ids": [parent["id"]]}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_stage", stage)
    base = candidate("base")
    duplicate, invalid, valid = candidate("duplicate"), candidate("invalid", 3.), candidate("valid", 4.)
    _, report = _topology_edit_loop("unused", {}, {}, base,
        {"candidates": [base, duplicate, invalid, valid]}, tmp_path, max_rounds=1,
        initial_seed_ids=("duplicate", "invalid", "valid"))
    assert parents == ["valid", "base"]
    assert report["budget"]["preflights_used"] == 2
    assert [(row["candidate_id"], row["reason"]) for row in report["initial_seed_audit"]["attempts"]] == [
        ("duplicate", "repeated_geometry"), ("invalid", "source_validation_failed"), ("valid", None)]


def test_unpreflighted_initial_seed_cannot_enter_beam(tmp_path, monkeypatch):
    _mock_preflight(monkeypatch)
    parents = []
    def stage(image, document, baseline, parent, bundle, output, **kwargs):
        parents.append(parent["id"])
        return parent, {"base_candidate_id": parent["id"], "final_candidate_id": parent["id"],
            "acceptance_gate": {"accepted": False}, "execution": {"operations": []},
            "editor": {}, "evaluator": {}, "trusted_candidate_ids": [parent["id"]]}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_stage", stage)
    base, alternate = candidate("base"), candidate("alternate", 3.)
    final, report = _topology_edit_loop("unused", {}, {}, base,
        {"candidates": [base, alternate]}, tmp_path, max_rounds=1, max_preflights=1,
        initial_seed_ids=("alternate",))
    assert final["id"] == "base"
    assert parents == ["base"]
    assert report["initial_seed_audit"]["admitted_candidate_ids"] == []
    assert report["initial_seed_audit"]["attempts"][0]["reason"] == "topology_search_preflight_count_budget_exhausted"


def test_slow_first_seed_reserves_time_for_editing_instead_of_preflighting_second(tmp_path, monkeypatch):
    _mock_preflight(monkeypatch)
    clock = [0.]
    monkeypatch.setattr("contour_agent.topology_search.time.monotonic", lambda: clock[0])
    def bindings(image, document, baseline, graph, output, **kwargs):
        if graph["candidate_id"] == "first":
            clock[0] += 350.
        return {"constraints": []}
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings", bindings)
    parents = []
    def stage(image, document, baseline, parent, bundle, output, **kwargs):
        parents.append(parent["id"])
        return parent, {"base_candidate_id": parent["id"], "final_candidate_id": parent["id"],
            "acceptance_gate": {"accepted": False}, "execution": {"operations": []},
            "editor": {}, "evaluator": {}, "trusted_candidate_ids": [parent["id"]]}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_stage", stage)
    base, first, second = candidate("base"), candidate("first", 3.), candidate("second", 4.)
    _, report = _topology_edit_loop("unused", {}, {}, base,
        {"candidates": [base, first, second]}, tmp_path, max_rounds=1,
        initial_seed_ids=("first", "second"))
    assert parents == ["first", "base"]
    assert report["initial_seed_audit"]["attempts"][1]["reason"] == "initial_seed_edit_time_reserved"
    assert report["budget"]["preflights_used"] == 2
    assert report["budget"]["remaining_seconds"] == 250.


def test_initial_seed_with_failed_solved_source_check_is_diagnostic_only(tmp_path, monkeypatch):
    _mock_preflight(monkeypatch)
    parents = []
    def stage(image, document, baseline, parent, bundle, output, **kwargs):
        parents.append(parent["id"])
        return parent, {"base_candidate_id": parent["id"], "final_candidate_id": parent["id"],
            "acceptance_gate": {"accepted": False}, "execution": {"operations": []},
            "editor": {}, "evaluator": {}, "trusted_candidate_ids": [parent["id"]]}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_stage", stage)
    base, alternate = candidate("base"), candidate("explore", 3.)
    final, report = _topology_edit_loop("unused", {}, {}, base,
        {"candidates": [base, alternate]}, tmp_path, max_rounds=1,
        initial_seed_ids=("explore",))
    assert parents == ["base", "explore"]
    assert final["id"] == "base"
    assert report["initial_seed_audit"]["attempts"][0]["status"] == "diagnostic_only"
    assert report["final_selection_origin"] == "initial_selection"


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


def test_time_exhausted_candidate_is_recorded_as_not_run_not_failed(tmp_path, monkeypatch):
    _mock_preflight(monkeypatch)
    clock = [0.]
    monkeypatch.setattr("contour_agent.topology_search.time.monotonic", lambda: clock[0])
    skipped = []
    def stage(image, document, baseline, parent, bundle, output, **kwargs):
        clock[0] = 2.
        skipped.append(kwargs["verifier"](candidate("later", 3.), source_validation={"passed": True}))
        return parent, {"base_candidate_id": parent["id"], "final_candidate_id": parent["id"],
            "acceptance_gate": {"accepted": False}, "execution": {"operations": []},
            "editor": {}, "evaluator": {}, "trusted_candidate_ids": [parent["id"]]}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_stage", stage)
    _, report = _topology_edit_loop("unused", {}, {}, candidate("base"), {}, tmp_path,
                                    max_rounds=1, max_seconds=1.)
    assert report["budget"]["preflights_used"] == 1
    assert skipped[0]["solver_status"] == "preflight_not_run"
    assert skipped[0]["preflight_status"] == "not_run_budget"
    assert skipped[0]["preflight_skip_reason"] == "topology_search_time_budget_exhausted"
    assert skipped[0]["topology_source_validation"] == {"passed": True}
    assert skipped[0]["source_validation"] == {"status": "not_run", "passed": None, "reasons": []}
    assert not preflight_admissible(skipped[0])


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
