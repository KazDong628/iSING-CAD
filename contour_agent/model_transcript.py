"""Public, allow-listed model receipts for the conversational workbench."""
from __future__ import annotations

import json
from pathlib import Path


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
            if not isinstance(editing, dict):
                continue
            for edit_id, edit_label, key in (("topology_edit", "局部拓扑编辑", "editor"),
                                             ("topology_evaluate", "拓扑编辑评估", "evaluator")):
                edit_provider = editing.get(key)
                if not isinstance(edit_provider, dict):
                    continue
                if (not edit_provider.get("network_requests") and
                        edit_provider.get("status") in {"disabled", "skipped", "not_configured"}):
                    continue
                stages.append(_stage(edit_id, edit_label, editing, edit_provider))
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
    }
