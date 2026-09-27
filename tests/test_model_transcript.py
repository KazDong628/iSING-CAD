import json

from contour_agent.model_transcript import build_model_transcript


def test_transcript_exposes_validated_answers_and_transport_without_private_fields(tmp_path):
    output = tmp_path / "jobs" / "a" / "automatic-001"
    output.mkdir(parents=True)
    (output / "topology-plan.json").write_text(json.dumps({
        "selected_candidate_id": "cand-03", "provider": {
            "status": "succeeded", "model": "gpt-test", "network_requests": 1,
            "http_success": True, "schema_success": True, "elapsed_seconds": 18.8,
            "selected_candidate_id": "cand-03", "relation_ids": ["rel001"],
            "binding_candidate_ids": ["bind002"], "observed_evidence_ids": ["ev003"],
            "rationale_code": "boundary_and_annotation_agree", "confidence": "high",
            "response_text_sha256": "hidden-hash", "source_image_sha256": "hidden-source",
        }
    }), encoding="utf-8")
    (output / "dimension-analysis.json").write_text(json.dumps({"provider": {
        "status": "succeeded", "network_requests": 1, "http_success": True,
        "schema_success": True, "elapsed_seconds": 11.6,
        "dimensions": [{"id": "r001", "kind": "radius", "nominal": 40}],
    }}), encoding="utf-8")
    job = {"id": "a" * 32, "mode": "autonomous_image", "artifact_directory": str(output), "provider": {
        "status": "succeeded", "network_requests": 1, "http_status": 200,
        "http_success": True, "schema_success": True, "elapsed_seconds": 56.1,
        "verdict": "mismatch", "roi": [1, 2, 3, 4], "issues": ["right edge"], "units": "mm",
    }}

    result = build_model_transcript(job, tmp_path)

    assert [stage["id"] for stage in result["stages"]] == ["planning", "dimensions", "vision"]
    assert result["stages"][0]["transport"]["elapsed_seconds"] == 18.8
    assert result["stages"][0]["answer"]["candidate_id"] == "cand-03"
    assert result["stages"][1]["answer"]["dimensions"][0]["nominal"] == 40
    assert result["stages"][2]["answer"]["issues"] == ["right edge"]
    serialized = json.dumps(result)
    assert "hidden-hash" not in serialized and "hidden-source" not in serialized
    assert result["private_chain_of_thought_exposed"] is False


def test_transcript_uses_only_sanitized_excerpt_when_schema_fails(tmp_path):
    output = tmp_path / "jobs" / "b"
    output.mkdir(parents=True)
    (output / "constraint-bindings.json").write_text(json.dumps({"provider": {
        "status": "failed", "network_requests": 1, "http_success": True,
        "schema_success": False, "elapsed_seconds": 135.2, "error_code": "invalid_json",
        "response_excerpt": "[REDACTED] malformed response", "api_key": "must-not-appear",
    }}), encoding="utf-8")
    job = {"id": "b" * 32, "mode": "autonomous_image", "artifact_directory": str(output),
           "provider": {"status": "disabled", "network_requests": 0}}

    result = build_model_transcript(job, tmp_path)

    assert len(result["stages"]) == 1
    answer = result["stages"][0]["answer"]
    assert answer == {"error_code": "invalid_json", "sanitized_response_excerpt": "[REDACTED] malformed response"}
    assert "must-not-appear" not in json.dumps(result)


def test_topology_editor_and_evaluator_outputs_follow_planning_in_transcript(tmp_path):
    output = tmp_path / "jobs" / "c"
    output.mkdir(parents=True)
    (output / "topology-plan.json").write_text(json.dumps({"provider": {
        "status": "succeeded", "network_requests": 1, "http_success": True,
        "schema_success": True, "selected_candidate_id": "cand-base", "confidence": "medium",
    }}), encoding="utf-8")
    (output / "topology-edit-proposals.json").write_text(json.dumps({
        "editor": {"status": "succeeded", "network_requests": 1, "http_success": True,
                   "schema_success": True, "elapsed_seconds": 9.1,
                   "observation": "g002 到 g004 是同一圆弧。", "confidence": "high",
                   "operations": [{"action": "merge_chain_as_arc", "entity_ids": ["g002", "g003", "g004"],
                                   "record_id": "r040", "evidence_tags": ["cocircular_support"]}]},
        "evaluator": {"status": "succeeded", "network_requests": 1, "http_success": True,
                      "schema_success": True, "elapsed_seconds": 4.2,
                      "selected_candidate_id": "cand-edit-01-arc", "decision": "accept_edit",
                      "observation": "圆弧连续并与 R40 指向一致。", "evidence_tags": ["continuity"],
                      "confidence": "high"},
        "private_prompt": "must-not-appear",
    }), encoding="utf-8")
    result = build_model_transcript({"id": "c" * 32, "mode": "autonomous_image",
                                     "artifact_directory": str(output),
                                     "provider": {"status": "disabled", "network_requests": 0}}, tmp_path)
    assert [stage["id"] for stage in result["stages"]] == [
        "planning", "topology_edit", "topology_evaluate"]
    assert result["stages"][1]["answer"]["operations"][0]["action"] == "merge_chain_as_arc"
    assert result["stages"][2]["answer"]["decision"] == "accept_edit"
    assert "must-not-appear" not in json.dumps(result)


def test_topology_edit_receipts_remain_visible_when_initial_planner_was_skipped(tmp_path):
    output = tmp_path / "jobs" / "d"
    output.mkdir(parents=True)
    (output / "topology-plan.json").write_text(json.dumps({"provider": {
        "status": "skipped", "network_requests": 0, "schema_success": False,
        "reason": "no_source_annotation_leader_evidence",
    }}), encoding="utf-8")
    (output / "topology-edit-proposals.json").write_text(json.dumps({
        "editor": {"status": "succeeded", "network_requests": 1, "http_success": True,
                   "schema_success": True, "observation": "短线段应合并。", "confidence": "high",
                   "operations": [{"action": "merge_chain_as_line", "entity_ids": ["g001", "g002"],
                                   "record_id": None, "evidence_tags": ["micro_segment"]}]},
        "evaluator": {"status": "succeeded", "network_requests": 1, "http_success": True,
                      "schema_success": True, "selected_candidate_id": "cand-edit-01-line",
                      "decision": "accept_edit", "observation": "通过", "evidence_tags": ["continuity"],
                      "confidence": "high"},
    }), encoding="utf-8")
    result = build_model_transcript({"id": "d" * 32, "mode": "autonomous_image",
                                     "artifact_directory": str(output),
                                     "provider": {"status": "disabled", "network_requests": 0}}, tmp_path)
    assert [stage["id"] for stage in result["stages"]] == ["topology_edit", "topology_evaluate"]


def test_missing_failure_code_is_classified_without_schema_not_available(tmp_path):
    output = tmp_path / "jobs" / "e"
    output.mkdir(parents=True)
    (output / "dimension-analysis.json").write_text(json.dumps({"provider": {
        "status": "failed", "network_requests": 1, "http_success": True,
        "schema_success": False,
    }}), encoding="utf-8")
    result = build_model_transcript({"id": "e" * 32, "mode": "autonomous_image",
                                     "artifact_directory": str(output),
                                     "provider": {"status": "disabled", "network_requests": 0}}, tmp_path)
    assert result["stages"][0]["answer"]["error_code"] == "schema_validation_failed"
    assert "schema_not_available" not in json.dumps(result)


def test_each_iteration_exposes_proposal_execution_and_independent_acceptance(tmp_path):
    output = tmp_path / "jobs" / "iterations"
    output.mkdir(parents=True)
    rounds = []
    for number in (1, 2):
        rounds.append({"round": number, "base_candidate_id": f"cand-{number}", "final_candidate_id": f"cand-{number+1}",
                       "editor": {"status": "succeeded", "network_requests": 1, "http_success": True,
                                  "schema_success": True, "observation": "检查圆角", "confidence": "high",
                                  "operations": []},
                       "evaluator": {"status": "succeeded", "network_requests": 1, "http_success": True,
                                     "schema_success": True, "decision": "accept_edit", "selected_candidate_id": f"cand-{number+1}"},
                       "acceptance_gate": {"accepted": True, "selection_source": "online_evaluator", "private_prompt": "DO-NOT-EXPOSE"},
                       "feedback": {"issue_count": 1, "issues": [{"code": "radius_value_unresolved", "record_id": "r003",
                                                                  "entity_id": "g001", "private_reasoning": "DO-NOT-EXPOSE"}]},
                       "execution": {"operations": [{"status": "accepted_as_candidate", "candidate_id": f"cand-{number+1}",
                                                      "operation": {"action": "refit_chain_as_annotated_arc", "entity_ids": ["g001"],
                                                                    "record_id": "r003", "hidden_reasoning": "DO-NOT-EXPOSE"},
                                                      "execution": {"radius_binding_applied": False,
                                                                    "radius_binding_status": "unresolved_fixed_radius_fit_failed",
                                                                    "radius_binding": {"secret": "DO-NOT-EXPOSE"}}}]}})
    document = {"status": "completed", "max_rounds": 3, "stop_reason": "no_accepted_improvement", "rounds": rounds,
                "final_feedback": {"remaining_shape_dof": 12, "constraint_count": 4, "solver_accepted": True}}
    (output / "topology-iterations.json").write_text(json.dumps(document), encoding="utf8")
    (output / "topology-edit-proposals.json").write_text(json.dumps(rounds[-1]), encoding="utf8")
    result = build_model_transcript({"id": "i" * 32, "artifact_directory": str(output)}, tmp_path)
    assert [(stage["id"], stage["round"]) for stage in result["stages"]] == [
        ("topology_edit", 1), ("topology_evaluate", 1), ("topology_edit", 2), ("topology_evaluate", 2)]
    assert result["iterations"]["rounds"][0]["operations"][0]["radius_binding_applied"] is False
    assert result["iterations"]["rounds"][0]["acceptance_gate"]["accepted"] is True
    assert result["iterations"]["final_feedback"]["remaining_shape_dof"] == 12
    assert "DO-NOT-EXPOSE" not in json.dumps(result)


def test_local_iterations_and_constraint_residuals_are_visible_without_online_receipts(tmp_path):
    output = tmp_path / "jobs" / "local"
    output.mkdir(parents=True)
    (output / "topology-iterations.json").write_text(json.dumps({
        "status": "completed", "max_rounds": 3, "stop_reason": "no_accepted_improvement", "rounds": [{
            "round": 1, "base_candidate_id": "base", "final_candidate_id": "base",
            "editor": {"status": "disabled", "network_requests": 0},
            "evaluator": {"status": "skipped", "network_requests": 0},
            "acceptance_gate": {"accepted": False, "reason": "evaluator_preserved_base"},
            "execution": {"operations": []}}]}), encoding="utf8")
    (output / "parametric-solution.json").write_text(json.dumps({
        "status": "accepted", "accepted": True, "underconstrained": True,
        "diagnostics": {"remaining_shape_dof": 15, "constraint_rank": 4, "private_reasoning": "DO-NOT-EXPOSE"},
        "constraints": [{"id": "kc001", "kind": "radius", "record_id": "r003", "value": 3, "actual": 3.02,
                         "absolute_residual": .02, "tolerance": .05, "passed": True}]}), encoding="utf8")
    (output / "constraint-bindings.json").write_text(json.dumps({
        "counts": {"bound_source_records": 2, "unbound_dimensions": 8, "recognized_dimensions": 10,
                   "constraints": 4, "structural_local_accepted": 2}}), encoding="utf8")
    (output / "reconstruction-feedback.json").write_text(json.dumps({
        "issue_count": 1, "issues": [{"code": "radius_value_unresolved", "entity_id": "g008", "record_id": "r009"}]
    }), encoding="utf8")
    result = build_model_transcript({"id": "j" * 32, "artifact_directory": str(output)}, tmp_path)
    assert result["stages"] == []
    assert result["iterations"]["rounds"][0]["acceptance_gate"]["accepted"] is False
    assert result["parameterization"]["counts"]["bound_source_records"] == 2
    assert result["parameterization"]["constraints"][0]["absolute_residual"] == .02
    assert result["parameterization"]["feedback"]["issues"][0]["record_id"] == "r009"
    assert result["parameterization"]["reference_verified"] is False
    assert "DO-NOT-EXPOSE" not in json.dumps(result)
