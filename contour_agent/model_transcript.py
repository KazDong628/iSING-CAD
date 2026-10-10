"""Public, allow-listed model receipts for the conversational workbench."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re


_STAGES = (
    ("radius_targets", "半径箭头定位", "radius-targets.json"),
    ("planning", "规划", "topology-plan.json"),
    ("dimensions", "尺寸解析", "dimension-analysis.json"),
    ("binding", "约束绑定", "constraint-bindings.json"),
    ("feedback", "截图修订", "feedback-plan.json"),
)


def _read_object(directory: Path | None, name: str):
    if directory is None:
        return None
    path = directory / name
    try:
        if not path.is_file() or path.stat().st_size > 8_000_000:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _value(primary, fallback, key, default=None):
    if isinstance(primary, dict) and key in primary:
        return primary[key]
    if isinstance(fallback, dict) and key in fallback:
        return fallback[key]
    return default


def _transport(provider):
    http_success = provider.get("http_success")
    http_status = provider.get("http_status")
    if http_status is None and http_success is True:
        http_status = 200
    return {
        "status": provider.get("status", "unknown"),
        "http_status": http_status,
        "http_success": http_success if isinstance(http_success, bool) else None,
        "schema_success": provider.get("schema_success") is True,
        "elapsed_seconds": provider.get("elapsed_seconds") if isinstance(provider.get("elapsed_seconds"), (int, float)) else None,
        "network_requests": provider.get("network_requests") if isinstance(provider.get("network_requests"), int) else None,
        "model": provider.get("model") if isinstance(provider.get("model"), str) else None,
        "protocol": provider.get("protocol") if isinstance(provider.get("protocol"), str) else None,
        "total_timeout_seconds": provider.get("total_timeout_seconds") if isinstance(provider.get("total_timeout_seconds"), (int, float)) else None,
    }


def _answer(stage_id, document, provider):
    if provider.get("schema_success") is not True:
        code = provider.get("error_code") or provider.get("code") or provider.get("reason")
        if not code:
            if provider.get("http_success") is True:
                code = "schema_validation_failed"
            elif provider.get("http_success") is False:
                code = "transport_failure_without_code"
            else:
                code = "stage_receipt_incomplete"
        result = {"error_code": code}
        excerpt = provider.get("response_excerpt")
        if isinstance(excerpt, str) and excerpt:
            result["sanitized_response_excerpt"] = excerpt
        return result, "sanitized_failure_receipt"
    if stage_id == "radius_targets":
        answer = _public_radius_targets(provider)
    elif stage_id == "planning":
        answer = {
            "candidate_id": _value(provider, document, "selected_candidate_id"),
            "relation_ids": _value(provider, document, "relation_ids", []),
            "binding_candidate_ids": _value(provider, document, "binding_candidate_ids", []),
            "observed_evidence_ids": _value(provider, document, "observed_evidence_ids", []),
            "rationale_code": _value(provider, document, "rationale_code"),
            "confidence": _value(provider, document, "confidence"),
        }
    elif stage_id == "dimensions":
        answer = {"dimensions": provider.get("dimensions", [])}
    elif stage_id == "binding":
        answer = {"bindings": provider.get("bindings", []), "relations": provider.get("relations", [])}
    elif stage_id == "topology_edit":
        answer = {
            "observation": _value(provider, document, "observation", ""),
            "operations": _value(provider, document, "operations", []),
            "confidence": _value(provider, document, "confidence"),
        }
    elif stage_id == "topology_evaluate":
        answer = {
            "candidate_id": _value(provider, document, "selected_candidate_id"),
            "observation": _value(provider, document, "observation", ""),
            "decision": _value(provider, document, "decision"),
            "evidence_tags": _value(provider, document, "evidence_tags", []),
            "confidence": _value(provider, document, "confidence"),
        }
    elif stage_id == "feedback":
        answer = {
            "candidate_id": _value(provider, document, "selected_candidate_id"),
            "observation": _value(provider, document, "observation", ""),
            "proposed_action": _value(provider, document, "proposed_action", ""),
            "evidence_tags": _value(provider, document, "evidence_tags", []),
            "operations": _value(provider, document, "operations", []),
            "rationale_code": _value(provider, document, "rationale_code"),
            "confidence": _value(provider, document, "confidence"),
        }
    else:
        answer = {}
    return answer, "validated_structured_fields"


def _stage(stage_id, label, document, provider):
    answer, source = _answer(stage_id, document, provider)
    return {
        "id": stage_id,
        "label": label,
        "transport": _transport(provider),
        "answer": answer,
        "answer_source": source,
        "raw_response_persisted": False,
    }


def _identifier(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,159}", value) else None


def _scalars(value, keys):
    """Allow-listed audit fields do not inherit arbitrary nested provider content."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in keys:
        item = value.get(key)
        if item is None or isinstance(item, bool):
            result[key] = item
        elif isinstance(item, (int, float)) and math.isfinite(item):
            result[key] = item
        elif _identifier(item):
            result[key] = item
    return result


def _identifiers(value, limit=128):
    return list(dict.fromkeys(item for item in value[:limit] if _identifier(item))) if isinstance(value, list) else []


def _pixel_point(value):
    if (isinstance(value, list) and len(value) == 2 and
            all(isinstance(item, (int, float)) and not isinstance(item, bool)
                and math.isfinite(item) and item >= 0 for item in value)):
        return value
    return None


def _public_radius_targets(provider):
    """Only source-pixel hypotheses; nested provider evidence is never public."""
    proposals = []
    raw = provider.get("proposals")
    for item in raw[:24] if isinstance(raw, list) else []:
        if not isinstance(item, dict) or not _identifier(item.get("record_id")):
            continue
        tip, shaft = _pixel_point(item.get("tip_px")), _pixel_point(item.get("shaft_px"))
        if tip is not None and shaft is not None:
            proposals.append({"record_id": item["record_id"], "tip_px": tip, "shaft_px": shaft})
    return {"proposals": proposals,
            "unknown_record_ids": _identifiers(provider.get("unknown_record_ids")),
            "omitted_record_ids": _identifiers(provider.get("omitted_record_ids")),
            "arrowheads_verified": False, "dimensions_verified": False,
            "missing_arrow_detection_is_exemption": False}


def _public_radius_contract(contract, bindings, validation):
    if not isinstance(contract, dict):
        contract = {}
    coverage = contract.get("coverage")
    if not isinstance(coverage, dict):
        coverage = (bindings or {}).get("radius_binding_coverage")
    if not contract and not isinstance(coverage, dict):
        return None
    coverage = coverage if isinstance(coverage, dict) else {}
    public = _scalars(contract, ("satisfied", "all_annotated_radii_verified", "candidate_satisfied",
                                "current_dxf_verified", "publication_status"))
    public["current_dxf_verified"] = contract.get("current_dxf_verified") is True
    public["reasons"] = _identifiers(contract.get("reasons"), 32)
    public["missing_arrow_detection_is_exemption"] = False
    exposed = _scalars(coverage, ("recognized_count", "required_count", "bound_count", "unresolved_count",
                                  "ambiguous_count", "unresolved_radius_text_count", "all_confirmed_arrows_bound", "all_radius_records_resolved"))
    for key in ("recognized_radius_records", "confirmed_arrow_records", "unknown_arrow_records", "verified_absent_arrow_records", "unresolved_radius_text_records"):
        exposed[key] = _identifiers(coverage.get(key))
    exposed["confirmed_count"] = len(exposed["confirmed_arrow_records"])
    exposed["unknown_count"] = len(exposed["unknown_arrow_records"])
    exposed["unresolved"] = []
    raw = coverage.get("unresolved")
    for item in raw[:128] if isinstance(raw, list) else []:
        if isinstance(item, dict):
            row = _scalars(item, ("record_id", "nominal", "reason", "source_arrow_verified"))
            row["candidate_entity_ids"] = _identifiers(item.get("candidate_entity_ids"), 16)
            exposed["unresolved"].append(row)
    public["coverage"] = exposed
    exact = contract.get("exact_radius_validation")
    final_exact = (validation or {}).get("exact_radius_validation")
    if (public["current_dxf_verified"] and isinstance(final_exact, dict) and
            final_exact.get("dxf_readback_performed") is True):
        exact = final_exact
    exact = exact if isinstance(exact, dict) else {}
    public_exact = _scalars(exact, ("mode", "required_count", "passed", "dxf_readback_performed"))
    public_exact["checks"] = []
    checks = exact.get("checks")
    for item in checks[:128] if isinstance(checks, list) else []:
        if isinstance(item, dict):
            row = _scalars(item, ("record_id", "entity_id", "nominal", "actual", "dxf_radius",
                                  "absolute_residual", "tolerance", "enforcement", "passed"))
            row["value"] = row.get("nominal")
            public_exact["checks"].append(row)
    public["exact_radius_validation"] = public_exact
    public["reference_verified"] = False
    return public


def _audit_number(value, *, integer=False):
    if (isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value) and value >= 0 and (not integer or isinstance(value, int))):
        return value
    return None


def _public_strict_relations(validation):
    """Published validation is authoritative; a newer candidate is not a receipt."""
    raw = (validation or {}).get("strict_relation_validation")
    if not isinstance(raw, dict):
        return None
    result = {"certificate_source": "published_validation", "checks": [],
              "required_count": _audit_number(raw.get("required_count"), integer=True),
              "angle_tolerance_deg": _audit_number(raw.get("angle_tolerance_deg")),
              "endpoint_tolerance": _audit_number(raw.get("endpoint_tolerance")),
              "dxf_readback_performed": raw.get("dxf_readback_performed") is True,
              "native_mapping_verified": raw.get("native_mapping_verified") is True,
              "complete_relation_coverage_verified": False, "reference_accuracy_verified": False}
    checks = raw.get("checks")
    for item in checks[:256] if isinstance(checks, list) else []:
        if not isinstance(item, dict):
            continue
        row = {"constraint_id": _identifier(item.get("constraint_id")),
               "entity_ids": _identifiers(item.get("entity_ids"), 2),
               "node_id": _identifier(item.get("node_id")), "passed": item.get("passed") is True}
        for source in ("model", "dxf"):
            measurement = item.get(source)
            if isinstance(measurement, dict):
                row[source] = {key: _audit_number(measurement.get(key))
                               for key in ("angle_residual_deg", "endpoint_gap")}
                row[source]["passed"] = measurement.get("passed") is True
        row["passed"] = bool(row["passed"] and all(
            row.get(source, {}).get("passed") is True and
            row[source].get("angle_residual_deg") is not None and row[source].get("endpoint_gap") is not None
            for source in ("model", "dxf")))
        result["checks"].append(row)
    result["satisfied_count"] = sum(row["passed"] for row in result["checks"])
    result["current_dxf_verified"] = bool(result["dxf_readback_performed"] and result["native_mapping_verified"])
    result["passed"] = bool(raw.get("passed") is True and result["current_dxf_verified"] and
                            result["required_count"] == len(result["checks"]) == result["satisfied_count"])
    return result


def _public_reconstruction_contract(validation, strict):
    raw = (validation or {}).get("reconstruction_contract")
    if not isinstance(raw, dict):
        return None
    result = {"certificate_source": "published_validation", "joints": [],
              "reference_accuracy_verified": False, "reference_object_count_verified": False,
              "complete_source_annotation_detection_verified": False}
    for key in ("recognized_dimensions", "bound_source_records", "unbound_dimensions", "entity_count"):
        result[key] = _audit_number(raw.get(key), integer=True)
    result["remaining_shape_dof"] = _audit_number(raw.get("remaining_shape_dof"))
    counts = raw.get("entity_count_by_type")
    result["entity_count_by_type"] = {kind: _audit_number((counts if isinstance(counts, dict) else {}).get(kind), integer=True)
                                      for kind in ("LINE", "ARC")}
    joints = raw.get("joints")
    for item in joints[:256] if isinstance(joints, list) else []:
        if not isinstance(item, dict):
            continue
        relation = item.get("relationship")
        if not isinstance(relation, str) or relation not in {"source_admitted_tangent", "sourced_line_directions"}:
            relation = "unresolved"
        types = item.get("types")
        result["joints"].append({"node_id": _identifier(item.get("node_id")),
                                 "entities": _identifiers(item.get("entities"), 2),
                                 "types": [kind if isinstance(kind, str) and kind in {"LINE", "ARC"} else None for kind in types[:2]]
                                          if isinstance(types, list) else [],
                                 "relationship": relation, "passed": item.get("passed") is True,
                                 "constraint_id": _identifier(item.get("constraint_id"))})
    observed_unknown = sum(row["relationship"] == "unresolved" for row in result["joints"])
    declared_unknown = _audit_number(raw.get("unresolved_joint_count"), integer=True)
    result["unresolved_joint_count"] = max(declared_unknown, observed_unknown) if declared_unknown is not None else None
    result["unresolved_arc_arc_joint_count"] = sum(row["relationship"] == "unresolved" and row["types"] == ["ARC", "ARC"]
                                                  for row in result["joints"])
    result["joint_inventory_complete"] = bool(isinstance(joints, list) and len(joints) <= 256 and
                                               result["entity_count"] == len(joints) == len(result["joints"]))
    result["all_join_relationships_certified"] = bool(raw.get("all_join_relationships_certified") is True and
        result["joint_inventory_complete"] and result["unresolved_joint_count"] == 0 and
        all(row["passed"] for row in result["joints"]) and (strict or {}).get("passed") is True)
    result["all_recognized_attributes_covered"] = bool(raw.get("all_recognized_attributes_covered") is True and
        result["recognized_dimensions"] is not None and result["recognized_dimensions"] > 0 and
        result["bound_source_records"] == result["recognized_dimensions"] and result["unbound_dimensions"] == 0)
    result["shape_fully_determined"] = bool(raw.get("shape_fully_determined") is True and result["remaining_shape_dof"] == 0)
    result["satisfied"] = bool(raw.get("satisfied") is True and (validation or {}).get("passed") is True and result["all_join_relationships_certified"] and
                               result["all_recognized_attributes_covered"] and result["shape_fully_determined"])
    result["status"] = "source_obligations_satisfied" if result["satisfied"] else "unresolved_source_obligations"
    return result


def _public_feedback(feedback):
    if not isinstance(feedback, dict):
        return {}
    result = _scalars(feedback, ("candidate_id", "issue_count", "constraint_count", "structural_constraint_count",
                                  "solver_status", "solver_accepted", "remaining_shape_dof", "unbound_dimensions"))
    result["issues"] = []
    issues = feedback.get("issues")
    for issue in issues[:64] if isinstance(issues, list) else []:
        if not isinstance(issue, dict) or not _identifier(issue.get("code")):
            continue
        row = _scalars(issue, ("code", "entity_id", "record_id", "stable_id", "kind", "relation_id",
                              "binding_verified", "arrowhead_verified"))
        if isinstance(issue.get("entity_ids"), list):
            row["entity_ids"] = [item for item in issue["entity_ids"][:8] if _identifier(item)]
        result["issues"].append(row)
    result["dimensions_verified"] = False
    result["reference_verified"] = False
    return result


def _public_iterations(document):
    if not isinstance(document, dict) or not isinstance(document.get("rounds"), list):
        return None
    result = _scalars(document, ("status", "max_rounds", "stop_reason", "final_candidate_id"))
    result["rounds"] = []
    for item in document["rounds"][:3]:
        if not isinstance(item, dict):
            continue
        row = _scalars(item, ("round", "base_candidate_id", "final_candidate_id"))
        gate = item.get("acceptance_gate") or {}
        row["acceptance_gate"] = _scalars(gate, ("accepted", "reason", "selection_source",
                                                 "source_validated_feature_restoration"))
        row["feedback"] = _public_feedback(item.get("feedback"))
        row["operations"] = []
        execution = item.get("execution") or {}
        operations = execution.get("operations") if isinstance(execution, dict) else []
        for operation in operations[:6] if isinstance(operations, list) else []:
            if not isinstance(operation, dict):
                continue
            public = _scalars(operation, ("operation_index", "status", "reason", "candidate_id"))
            proposal = operation.get("operation") or {}
            if isinstance(proposal, dict):
                public.update(_scalars(proposal, ("action", "record_id")))
                if isinstance(proposal.get("entity_ids"), list):
                    public["entity_ids"] = [item for item in proposal["entity_ids"][:8] if _identifier(item)]
            detail = operation.get("execution") or {}
            public.update(_scalars(detail, ("replacement_type", "replacement_count", "net_entity_reduction",
                                            "radius_binding_applied", "radius_binding_status",
                                            "feature_restoration_validated")))
            row["operations"].append(public)
        result["rounds"].append(row)
    result["final_feedback"] = _public_feedback(document.get("final_feedback"))
    result["reference_verified"] = False
    return result


def _public_parameterization(directory):
    solution = _read_object(directory, "parametric-solution.json")
    bindings = _read_object(directory, "constraint-bindings.json")
    stage = _read_object(directory, "parametric-stage.json")
    contract = _read_object(directory, "radius-contract.json")
    validation = _read_object(directory, "validation.json")
    if not solution and not bindings and not contract and not stage and not validation:
        return None
    result = _scalars(solution, ("status", "accepted", "underconstrained"))
    result["constraint_subset_accepted"] = (solution or {}).get("accepted") is True
    result["pipeline_accepted"] = (stage or {}).get("accepted") is True
    result["pipeline_status"] = _scalars(stage, ("status",)).get("status")
    result["all_dimensions_verified"] = (stage or {}).get("all_dimensions_verified") is True
    result["radius_contract"] = _public_radius_contract(contract, bindings, validation)
    result["strict_relation_validation"] = _public_strict_relations(validation)
    result["reconstruction_contract"] = _public_reconstruction_contract(validation, result["strict_relation_validation"])
    result["counts"] = _scalars((bindings or {}).get("counts"),
                                ("recognized_dimensions", "bound_source_records", "unbound_dimensions", "constraints",
                                 "structural_local_accepted", "structural_api_accepted"))
    result["diagnostics"] = _scalars((solution or {}).get("diagnostics"),
                                     ("constraint_rank", "remaining_shape_dof", "independent_dimension_record_count"))
    result["constraints"] = []
    for constraint in (solution or {}).get("constraints", [])[:128]:
        if isinstance(constraint, dict):
            result["constraints"].append(_scalars(constraint, ("id", "kind", "record_id", "passed",
                                                               "value", "actual", "absolute_residual", "tolerance")))
    result["feedback"] = _public_feedback(_read_object(directory, "reconstruction-feedback.json"))
    result["reference_verified"] = False
    return result


def build_model_transcript(job: dict, runtime_root: Path):
    """Return only model-returned allow-listed fields, never prompts, secrets or CoT."""
    directory = None
    raw_directory = job.get("artifact_directory")
    if isinstance(raw_directory, str) and raw_directory:
        try:
            candidate = Path(raw_directory).resolve(strict=True)
            root = Path(runtime_root).resolve(strict=True)
            if candidate.is_relative_to(root):
                directory = candidate
        except OSError:
            pass
    stages = []
    iterations = _read_object(directory, "topology-iterations.json")
    for stage_id, label, filename in _STAGES:
        document = _read_object(directory, filename)
        provider = document.get("provider") if isinstance(document, dict) else None
        if stage_id == "radius_targets" and isinstance(document, dict):
            # This artifact stores the receipt directly. Its newly introduced
            # stage never exposes arbitrary response excerpts or nested fields.
            provider = {**_scalars(document, ("status", "http_status", "http_success", "schema_success",
                                               "elapsed_seconds", "network_requests", "model", "protocol",
                                               "total_timeout_seconds", "error_code", "reason")),
                        **_public_radius_targets(document)}
        visible = isinstance(provider, dict) and not (
            not provider.get("network_requests") and
            provider.get("status") in {"disabled", "skipped", "not_configured"}
        )
        if visible:
            stages.append(_stage(stage_id, label, document, provider))
        if stage_id == "planning":
            editing = _read_object(directory, "topology-edit-proposals.json")
            rounds = iterations.get("rounds") if isinstance(iterations, dict) else None
            if not isinstance(rounds, list) and isinstance(editing, dict):
                rounds = editing.get("rounds")
            per_round = isinstance(rounds, list)
            for record in rounds[:3] if per_round else [editing]:
                if not isinstance(record, dict):
                    continue
                for edit_id, edit_label, key in (("topology_edit", "局部拓扑编辑", "editor"),
                                                 ("topology_evaluate", "拓扑编辑评估", "evaluator")):
                    edit_provider = record.get(key)
                    if not isinstance(edit_provider, dict):
                        continue
                    if (not edit_provider.get("network_requests") and
                            edit_provider.get("status") in {"disabled", "skipped", "not_configured"}):
                        continue
                    round_number = record.get("round") if per_round else None
                    label = f"第 {round_number} 轮 · {edit_label}" if type(round_number) is int else edit_label
                    stage = _stage(edit_id, label, record, edit_provider)
                    if type(round_number) is int:
                        stage["round"] = round_number
                    stages.append(stage)
    provider = job.get("provider")
    if job.get("mode") != "autonomous_revision" and isinstance(provider, dict) and (
        provider.get("network_requests") or provider.get("status") not in {"disabled", "skipped", "not_configured", "pending"}
    ):
        document = {"provider": provider}
        if provider.get("schema_success") is True:
            answer = {
                "verdict": provider.get("verdict"), "roi": provider.get("roi"),
                "issues": provider.get("issues", []), "units": provider.get("units"),
            }
            source = "validated_structured_fields"
        else:
            answer, source = _answer("vision", document, provider)
        visual = _stage("vision", "视觉复核", document, provider)
        visual["answer"], visual["answer_source"] = answer, source
        stages.append(visual)
    return {
        "job_id": job.get("id"),
        "scope": "validated_model_outputs",
        "private_chain_of_thought_exposed": False,
        "notice": "显示模型返回后通过结构校验的字段；不保存或展示私有思维链。",
        "stages": stages,
        "iterations": _public_iterations(iterations),
        "parameterization": _public_parameterization(directory),
    }
