"""A verified primary edit can preserve time for the next bounded round."""
from copy import deepcopy

import pytest

from contour_agent.parametric_pipeline import _topology_edit_loop, _topology_edit_stage
from contour_agent.topology_search import preflight_admissible


def _candidate(name, length=2.):
    return {"id": name, "graph": {"candidate_id": name, "units": "mm",
        "source_grid_pitch_px": 1., "entities": [
            {"id": "g000", "type": "LINE", "start": [0., 0.], "end": [length, 0.]},
            {"id": "g001", "type": "LINE", "start": [length, 0.], "end": [0., 0.]},
        ]}}


def _stage_fixture(tmp_path, monkeypatch, *, angle_gain=True, angle_certificate=True,
                   another_round=True, sibling_source_valid=True):
    import contour_agent.planning_provider as planning
    import contour_agent.topology_editing as editing

    base = _candidate("base")
    angular = _candidate("angular", 3.)
    angular["graph"]["entities"].append({"id": "g002", "type": "LINE",
        "start": [0., 0.], "end": [8., 0.], "angle_support_evidence": {
            "record_id": "r035", "requires_angle_binding_and_solve": True,
            "ground_truth_used": False,
            "source_interval_endpoints_px": [[0., 0.], [8., 0.]]}})
    angular["graph"]["source_evidence"] = {"topology_edit": {
        "action": "restore_annotated_line_support", "record_id": "r035",
        "resegmentation_applied": True, "net_entity_reduction": -1,
        "angle_support_record_ids": ["r035"]}}
    combined = _candidate("combined", 4.)
    combined["graph"]["source_evidence"] = {"topology_edit": {
        "action": "apply_nonoverlapping_edits", "operations": []}}
    sibling = _candidate("sibling", 5.)
    sibling["graph"]["source_evidence"] = {"topology_edit": {
        "action": "insert_annotated_fillet", "record_id": "r040"}}
    edited = [sibling, angular, combined]
    operations = [{"action": "insert_annotated_fillet", "entity_ids": ["g000"],
                   "record_id": "r040"}]
    monkeypatch.setattr(editing, "propose_annotation_arc_edits", lambda *args, **kwargs: operations)
    monkeypatch.setattr(editing, "execute_topology_edits", lambda *args, **kwargs: (edited, {
        "operations": [{"operation": {"action": "synthetic", "entity_ids": [row["id"]]},
                        "candidate_id": row["id"], "status": "accepted_as_candidate"}
                       for row in edited]}))
    source_checked = []
    def source_check(_image, _baseline, graph):
        source_checked.append(graph["candidate_id"])
        if graph["candidate_id"] == "sibling" and not sibling_source_valid:
            return {"passed": False, "reasons": ["source_stroke_support_degraded"]}
        return {"passed": True, "reasons": []}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_source_validation", source_check)
    def local_evaluation(rows, *, max_candidates=5):
        evaluated = []
        for row in rows:
            name = row["id"]
            evaluated.append({"candidate_id": name, "admissible": True,
                "score": .85 if name == "base" else .84,
                "metrics": {"entity_count": len(row["graph"]["entities"]),
                            "source_boundary_support": .83,
                            "unsupported_primitive_count": 0}})
        return {"admissible_candidate_ids": [row["id"] for row in rows],
                "evaluated": evaluated}
    monkeypatch.setattr(planning, "evaluate_candidates", local_evaluation)
    seen = []
    def verifier(row, *, source_validation):
        seen.append(row["id"])
        if source_validation["passed"] is False:
            return {"solver_status": "source_validation_failed", "solver_accepted": False,
                "constraint_count": 0, "binding_status": "not_run",
                "source_validation": source_validation,
                "topology_source_validation": source_validation}
        if row["id"] == "combined":
            return {"solver_status": "source_budget_search_exhausted", "solver_accepted": False,
                "post_solve_source_accepted": True, "constraint_count": 2,
                "binding_status": "local", "bound_record_ids": ["r001"],
                "satisfied_record_ids": ["r001"], "issue_count": 9,
                "source_validation": {"passed": True},
                "topology_source_validation": {"passed": True}}
        ids = ["r001", "r035"] if row["id"] == "angular" and angle_gain else ["r001"]
        return {"solver_status": "accepted", "solver_accepted": True,
            "post_solve_source_accepted": angle_certificate if row["id"] == "angular" else True,
            "constraint_count": 2, "binding_status": "local", "bound_record_ids": ids,
            "satisfied_record_ids": ids, "issue_count": 4 if len(ids) == 2 else 5,
            "source_validation": {"passed": True},
            "topology_source_validation": {"passed": True}}
    feedback = {"solver_accepted": True, "constraint_count": 1,
                "bound_record_ids": ["r001"], "satisfied_record_ids": ["r001"],
                "issue_count": 5, "source_validation": {"passed": True},
                "topology_source_validation": {"passed": True}}
    selected, receipt = _topology_edit_stage(
        "unused", {}, {}, base,
        {"annotation_inventory": [], "candidates": [base], "generation": {}}, tmp_path,
        verifier=verifier, feedback=feedback,
        defer_after_primary_progress=another_round)
    return selected, receipt, edited, seen, source_checked


def test_angle_progress_defers_siblings_after_combined_numeric_failure(tmp_path, monkeypatch):
    selected, receipt, edited, seen, source_checked = _stage_fixture(tmp_path, monkeypatch)
    assert source_checked == ["sibling", "angular", "combined"]
    assert seen == ["angular", "combined"]
    assert [row["status"] for row in receipt["preflight_queue"]] == [
        "passed", "rejected", "deferred_not_run"]
    assert receipt["deferred_preflight_candidate_ids"] == ["sibling"]
    assert receipt["deferred_for_next_round_after_candidate_id"] == "angular"
    sibling = edited[0]
    assert sibling["constraint_feedback"]["solver_status"] == "preflight_not_run"
    assert sibling["constraint_feedback"]["binding_status"] == "not_run"
    assert sibling["constraint_feedback"].get("preflight_artifact") is None
    assert preflight_admissible(sibling["constraint_feedback"]) is False
    assert sibling["id"] not in receipt["trusted_candidate_ids"]
    assert sibling["id"] not in receipt["exploratory_candidate_ids"]
    assert receipt["execution"]["operations"][0]["preflight_status"] == "deferred_not_run"
    assert selected["id"] == "angular"


@pytest.mark.parametrize("angle_gain,angle_certificate,another_round", [
    (False, True, True),
    (True, False, True),
    (True, True, False),
])
def test_no_deferral_without_certified_progress_and_another_round(
        tmp_path, monkeypatch, angle_gain, angle_certificate, another_round):
    _, receipt, _, seen, _ = _stage_fixture(tmp_path, monkeypatch,
        angle_gain=angle_gain, angle_certificate=angle_certificate,
        another_round=another_round)
    assert seen == ["angular", "combined", "sibling"]
    assert receipt["deferred_preflight_candidate_ids"] == []
    assert "deferred_not_run" not in [row["status"] for row in receipt["preflight_queue"]]


def test_source_invalid_sibling_keeps_its_source_failure_audit(tmp_path, monkeypatch):
    _, receipt, edited, seen, checked = _stage_fixture(
        tmp_path, monkeypatch, sibling_source_valid=False)
    assert checked == ["sibling", "angular", "combined"]
    assert seen == ["angular", "combined", "sibling"]
    assert receipt["deferred_preflight_candidate_ids"] == []
    assert edited[0]["constraint_feedback"]["solver_status"] == "source_validation_failed"
    assert receipt["preflight_queue"][-1]["reasons"] == ["source_stroke_support_degraded"]


def test_deferred_operation_is_retryable_on_same_parent_next_round(tmp_path, monkeypatch):
    from contour_agent.reconstruction_feedback import geometry_fingerprint

    base = _candidate("base")
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_source_validation",
                        lambda *args: {"passed": True, "reasons": []})
    monkeypatch.setattr("contour_agent.parametric_pipeline._preflight_input_identity",
                        lambda *args: {"schema_version": "synthetic", "reference_geometry_used": False})
    monkeypatch.setattr("contour_agent.constraint_binding.analyze_constraint_bindings",
                        lambda *args, **kwargs: {"constraints": []})
    monkeypatch.setattr("contour_agent.parametric_pipeline._solve_with_source_observation",
                        lambda graph, *args, **kwargs: {"status": "accepted", "accepted": True,
                            "entities": graph["entities"], "constraints": []})
    monkeypatch.setattr("contour_agent.parametric_pipeline._solved_source_validation",
                        lambda *args: {"passed": True, "reasons": []})
    first = {"action": "refit_entity_as_line", "entity_ids": ["g000"], "record_id": "r001"}
    deferred = {"action": "insert_annotated_fillet", "entity_ids": ["g001"], "record_id": "r002"}
    attempts = []
    def stage(_image, _document, _baseline, parent, _bundle, _output, **kwargs):
        operation_filter = kwargs["operation_filter"]
        number = kwargs["round_index"]
        receipts = []
        for operation in (first, deferred):
            allowed = operation_filter(operation)
            attempts.append((number, operation["record_id"], allowed))
            if allowed:
                receipts.append({"operation": deepcopy(operation),
                    "status": "accepted_as_candidate",
                    "preflight_status": "deferred_not_run" if number == 1 and operation == deferred else "rejected",
                    "preflight_reasons": ["topology_preflight_deferred_for_next_round"]
                    if number == 1 and operation == deferred else ["synthetic_rejection"]})
        return parent, {"base_candidate_id": parent["id"], "final_candidate_id": parent["id"],
            "acceptance_gate": {"accepted": False}, "execution": {"operations": receipts},
            "editor": {}, "evaluator": {}, "trusted_candidate_ids": [parent["id"]]}
    monkeypatch.setattr("contour_agent.parametric_pipeline._topology_edit_stage", stage)
    final, report = _topology_edit_loop("unused", {}, {}, base,
        {"annotation_inventory": [], "candidates": [base]}, tmp_path / "run",
        use_api=True, max_rounds=2)
    assert final["id"] == "base"
    assert len(report["rounds"]) == 2
    assert attempts == [(1, "r001", True), (1, "r002", True),
                        (2, "r001", False), (2, "r002", True)]
    assert len(report["visited_operations"]) == 2
    assert all(row["status"] == "rejected" for row in report["visited_operations"])
    assert report["budget"]["preflights_used"] == 1
    assert report["visited_geometries"][0]["geometry_sha256"] == geometry_fingerprint(base["graph"])
