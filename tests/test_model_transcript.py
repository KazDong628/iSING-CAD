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


def test_radius_target_stage_exposes_only_valid_source_points_and_record_ids(tmp_path):
    output = tmp_path / "jobs" / "radius-targets"
    output.mkdir(parents=True)
    (output / "radius-targets.json").write_text(json.dumps({
        "status": "succeeded", "network_requests": 1, "http_success": True,
        "schema_success": True, "model": "source-agent", "elapsed_seconds": 2.4,
        "proposals": [
            {"record_id": "r003", "tip_px": [140.5, 260.], "shaft_px": [150., 200.],
             "private_prompt": "DO-NOT-EXPOSE", "evidence": {"api_key": "DO-NOT-EXPOSE"}},
            {"record_id": "r004", "tip_px": [{"secret": "DO-NOT-EXPOSE"}, 3], "shaft_px": [1, 2]},
            {"record_id": "r005", "tip_px": [True, 3], "shaft_px": [1, 2]}],
        "unknown_record_ids": ["r004", {"secret": "DO-NOT-EXPOSE"}],
        "omitted_record_ids": ["r016"], "response_text": "DO-NOT-EXPOSE",
        "input_images": [{"payload": "DO-NOT-EXPOSE"}], "arrowheads_verified": True,
    }), encoding="utf8")
    result = build_model_transcript({"id": "r" * 32, "artifact_directory": str(output)}, tmp_path)
    assert [stage["id"] for stage in result["stages"]] == ["radius_targets"]
    stage = result["stages"][0]
    assert stage["label"] == "半径箭头定位"
    assert stage["transport"]["elapsed_seconds"] == 2.4
    assert stage["answer"]["proposals"] == [
        {"record_id": "r003", "tip_px": [140.5, 260.], "shaft_px": [150., 200.]}]
    assert stage["answer"]["unknown_record_ids"] == ["r004"]
    assert stage["answer"]["omitted_record_ids"] == ["r016"]
    assert not stage["answer"]["arrowheads_verified"]
    assert not stage["answer"]["missing_arrow_detection_is_exemption"]
    assert "DO-NOT-EXPOSE" not in json.dumps(result)


def test_radius_audit_distinguishes_subset_acceptance_coverage_and_dxf_readback(tmp_path):
    output = tmp_path / "jobs" / "radius-audit"
    output.mkdir(parents=True)
    documents = {
        "parametric-solution.json": {"status": "accepted", "accepted": True, "constraints": []},
        "parametric-stage.json": {"status": "completed_with_unresolved_radii", "accepted": False,
                                  "constraint_subset_accepted": True, "all_dimensions_verified": False},
        "radius-contract.json": {
            "satisfied": False, "all_annotated_radii_verified": False,
            "current_dxf_verified": True, "publication_status": "committed",
            "reasons": ["radius_arrow_detection_unresolved"],
            "coverage": {"recognized_count": 3, "required_count": 2, "bound_count": 1,
                         "unresolved_count": 2, "ambiguous_count": 1,
                         "recognized_radius_records": ["r001", "r002", "r003"],
                         "confirmed_arrow_records": ["r001", "r002"], "unknown_arrow_records": ["r003"],
                         "all_radius_records_resolved": False, "all_confirmed_arrows_bound": False,
                         "required_mappings": [{"source_evidence": {"secret": "DO-NOT-EXPOSE"}}],
                         "unresolved": [{"record_id": "r003", "nominal": 40., "reason": "source_arrow_not_verified",
                                         "candidate_entity_ids": ["g003"], "private_reasoning": "DO-NOT-EXPOSE"}]},
            "exact_radius_validation": {"required_count": 1, "passed": True, "dxf_readback_performed": False,
                                        "checks": [{"record_id": "r001", "entity_id": "g001", "nominal": 3.,
                                                    "actual": 3., "dxf_radius": None, "tolerance": 0., "passed": True}]}},
        "validation.json": {"exact_radius_validation": {
            "mode": "exact_native_arc_radius", "required_count": 1, "passed": False, "dxf_readback_performed": True,
            "checks": [{"record_id": "r001", "entity_id": "g001", "nominal": 3., "actual": 3.,
                        "dxf_radius": 3.01, "absolute_residual": 0., "tolerance": 0., "enforcement": "exact",
                        "passed": False, "nested": {"secret": "DO-NOT-EXPOSE"}}]}}
    }
    for name, document in documents.items():
        (output / name).write_text(json.dumps(document), encoding="utf8")
    result = build_model_transcript({"id": "s" * 32, "artifact_directory": str(output)}, tmp_path)
    audit = result["parameterization"]
    assert audit["constraint_subset_accepted"] and audit["accepted"]
    assert not audit["pipeline_accepted"] and not audit["all_dimensions_verified"]
    assert audit["pipeline_status"] == "completed_with_unresolved_radii"
    contract = audit["radius_contract"]
    assert not contract["satisfied"] and not contract["missing_arrow_detection_is_exemption"]
    assert contract["coverage"]["recognized_count"] == 3
    assert contract["coverage"]["confirmed_count"] == 2
    assert contract["coverage"]["bound_count"] == 1
    assert contract["coverage"]["unknown_count"] == 1 and contract["coverage"]["ambiguous_count"] == 1
    exact = contract["exact_radius_validation"]
    assert exact["dxf_readback_performed"] and not exact["passed"]
    assert exact["checks"][0]["value"] == exact["checks"][0]["actual"] == 3.
    assert exact["checks"][0]["dxf_radius"] == 3.01 and exact["checks"][0]["tolerance"] == 0.
    assert "DO-NOT-EXPOSE" not in json.dumps(result)


def test_candidate_radius_readback_cannot_certify_retained_dxf():
    from contour_agent.model_transcript import _public_radius_contract
    contract={"satisfied":False,"candidate_satisfied":True,"current_dxf_verified":False,
              "publication_status":"rolled_back","exact_radius_validation":{
                  "passed":True,"dxf_readback_performed":True,"required_count":1,"checks":[]}}
    public=_public_radius_contract(contract,{}, {"exact_radius_validation":{
        "passed":True,"dxf_readback_performed":True,"required_count":0,"checks":[]}})
    assert public["current_dxf_verified"] is False and public["candidate_satisfied"] is True
    assert public["publication_status"]=="rolled_back" and public["satisfied"] is False
    assert public["exact_radius_validation"]["required_count"]==1


def test_radius_contract_without_solver_receipt_does_not_claim_subset_acceptance(tmp_path):
    output = tmp_path / "jobs" / "radius-only"
    output.mkdir(parents=True)
    (output / "radius-contract.json").write_text(json.dumps({
        "satisfied": False, "coverage": {"recognized_count": 1, "unknown_arrow_records": ["r001"],
                                          "recognized_radius_records": ["r001"], "confirmed_arrow_records": [],
                                          "bound_count": 0, "ambiguous_count": 0}}), encoding="utf8")
    result = build_model_transcript({"id": "t" * 32, "artifact_directory": str(output)}, tmp_path)
    audit = result["parameterization"]
    assert not audit["constraint_subset_accepted"] and not audit["pipeline_accepted"]
    assert audit["radius_contract"]["coverage"]["unknown_count"] == 1
    assert audit["radius_contract"]["exact_radius_validation"]["dxf_readback_performed"] is None


def _published_joint_validation():
    return {"passed":True,"strict_relation_validation":{
        "passed":True,"required_count":1,"satisfied_count":1,"dxf_readback_performed":True,
        "native_mapping_verified":True,"angle_tolerance_deg":1e-7,"endpoint_tolerance":1e-7,
        "checks":[{"constraint_id":"krel001","entity_ids":["g000","g001"],"node_id":"v001","passed":True,
                   "model":{"angle_residual_deg":1e-12,"endpoint_gap":0.,"passed":True},
                   "dxf":{"angle_residual_deg":2e-12,"endpoint_gap":0.,"passed":True}}]},
        "reconstruction_contract":{"satisfied":False,"recognized_dimensions":3,"bound_source_records":2,
        "unbound_dimensions":1,"all_recognized_attributes_covered":False,
        "all_join_relationships_certified":False,"remaining_shape_dof":4,"shape_fully_determined":False,
        "entity_count":3,"entity_count_by_type":{"LINE":1,"ARC":2},"unresolved_joint_count":1,
        "joints":[{"node_id":"v001","entities":["g000","g001"],"types":["LINE","ARC"],
                   "relationship":"source_admitted_tangent","constraint_id":"krel001","passed":True},
                  {"node_id":"v002","entities":["g001","g002"],"types":["ARC","ARC"],
                   "relationship":"unresolved","passed":False},
                  {"node_id":"v000","entities":["g002","g000"],"types":["ARC","LINE"],
                   "relationship":"unresolved","passed":False}]}}


def test_published_joint_audit_is_visible_without_candidate_solution_and_is_whitelisted(tmp_path):
    output=tmp_path/"jobs"/"published-joints";output.mkdir(parents=True)
    validation=_published_joint_validation()
    strict=validation["strict_relation_validation"];coverage=validation["reconstruction_contract"]
    strict["provider"]={"api_key":"DO-NOT-EXPOSE"};strict["path"]="C:/private/secret"
    strict["checks"][0]["dxf"]["secret"]="DO-NOT-EXPOSE"
    coverage["joints"][0]["evidence"]={"provider":{"token":"DO-NOT-EXPOSE"}}
    coverage["remaining_shape_dof"]={"secret":"DO-NOT-EXPOSE"}
    coverage["entity_count_by_type"]["secret"]="DO-NOT-EXPOSE"
    (output/"validation.json").write_text(json.dumps(validation),encoding="utf8")
    result=build_model_transcript({"id":"j"*32,"artifact_directory":str(output)},tmp_path)
    audit=result["parameterization"];public=audit["strict_relation_validation"]
    assert public["certificate_source"]=="published_validation"
    assert public["passed"] and public["current_dxf_verified"]
    assert public["required_count"]==public["satisfied_count"]==1
    coverage=audit["reconstruction_contract"]
    assert coverage["unresolved_joint_count"]==2  # Never trust an understated summary over its joint inventory.
    assert coverage["unresolved_arc_arc_joint_count"]==1
    assert coverage["remaining_shape_dof"] is None
    assert not coverage["satisfied"] and not coverage["all_join_relationships_certified"]
    assert not coverage["reference_accuracy_verified"]
    serialized=json.dumps(result)
    assert "DO-NOT-EXPOSE" not in serialized and "C:/private" not in serialized


def test_current_joint_certificate_never_falls_back_to_a_newer_candidate(tmp_path):
    output=tmp_path/"jobs"/"retained-current";output.mkdir(parents=True)
    candidate=_published_joint_validation()
    (output/"parametric-solution.json").write_text(json.dumps({"accepted":True,
        "strict_relation_validation":candidate["strict_relation_validation"],
        "strict_tangent_contract":{"satisfied":True,"required_count":12}}),encoding="utf8")
    (output/"reconstruction-contract.json").write_text(json.dumps({"satisfied":True,"remaining_shape_dof":0}),encoding="utf8")
    # A standalone/latest candidate receipt cannot certify the published DXF.
    result=build_model_transcript({"id":"k"*32,"artifact_directory":str(output)},tmp_path)
    assert result["parameterization"]["strict_relation_validation"] is None
    assert result["parameterization"]["reconstruction_contract"] is None
    published=_published_joint_validation();published["strict_relation_validation"]["passed"]=False
    published["strict_relation_validation"]["checks"][0]["dxf"].update(passed=False,angle_residual_deg=.03)
    (output/"validation.json").write_text(json.dumps(published),encoding="utf8")
    result=build_model_transcript({"id":"k"*32,"artifact_directory":str(output)},tmp_path)
    assert not result["parameterization"]["strict_relation_validation"]["passed"]
    assert result["parameterization"]["strict_relation_validation"]["satisfied_count"]==0
    assert result["parameterization"]["reconstruction_contract"]["remaining_shape_dof"]==4


def test_unknown_arc_arc_never_inherits_complete_relation_flag(tmp_path):
    output=tmp_path/"jobs"/"false-complete";output.mkdir(parents=True)
    validation=_published_joint_validation();contract=validation["reconstruction_contract"]
    contract.update(satisfied=True,unresolved_joint_count=0,all_join_relationships_certified=True,
                    all_recognized_attributes_covered=True,remaining_shape_dof=0,shape_fully_determined=True)
    contract["joints"][0]["relationship"]={"provider":"DO-NOT-EXPOSE"}
    contract["joints"][0]["types"]=[{"provider":"DO-NOT-EXPOSE"},"ARC"]
    (output/"validation.json").write_text(json.dumps(validation),encoding="utf8")
    result=build_model_transcript({"id":"m"*32,"artifact_directory":str(output)},tmp_path)
    contract=result["parameterization"]["reconstruction_contract"]
    assert contract["unresolved_joint_count"]==3
    assert contract["unresolved_arc_arc_joint_count"]==1
    assert not contract["satisfied"] and not contract["all_join_relationships_certified"]
    assert "DO-NOT-EXPOSE" not in json.dumps(result)


def test_whitelisted_complete_source_audit_still_does_not_claim_gt(tmp_path):
    output=tmp_path/"jobs"/"complete-source";output.mkdir(parents=True)
    validation=_published_joint_validation();contract=validation["reconstruction_contract"]
    contract.update(satisfied=True,unresolved_joint_count=0,all_join_relationships_certified=True,
                    all_recognized_attributes_covered=True,remaining_shape_dof=0,shape_fully_determined=True,
                    unbound_dimensions=0,bound_source_records=3,reference_accuracy_verified=True)
    for row in contract["joints"]:row.update(relationship="source_admitted_tangent",passed=True)
    (output/"validation.json").write_text(json.dumps(validation),encoding="utf8")
    result=build_model_transcript({"id":"n"*32,"artifact_directory":str(output)},tmp_path)
    contract=result["parameterization"]["reconstruction_contract"]
    assert contract["satisfied"] and contract["all_join_relationships_certified"]
    assert not contract["reference_accuracy_verified"]
