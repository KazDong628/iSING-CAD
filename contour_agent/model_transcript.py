"""Public, allow-listed model receipts for the conversational workbench."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re


_STAGES = (
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
    if stage_id == "planning":
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
    if not solution and not bindings:
        return None
    result = _scalars(solution, ("status", "accepted", "underconstrained"))
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
