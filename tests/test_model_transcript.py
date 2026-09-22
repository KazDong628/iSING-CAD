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
