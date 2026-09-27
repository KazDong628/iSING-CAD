"""Wire-level budget and truncation contracts across all topology stages."""
import hashlib
import json

import httpx
from PIL import Image
import pytest

from contour_agent.api_wire import numeric_token_usage, output_token_budget
from contour_agent.binding_provider import BindingProvider
from contour_agent.config import Settings
from contour_agent.planning_provider import PlanningProvider
from contour_agent.provider import DimensionProvider
from contour_agent.topology_edit_provider import TopologyEditProvider, TopologyEvaluationProvider


STAGES = ("planning", "binding", "editing", "evaluation")
PRIVATE = "PRIVATE_THINKING_MUST_NOT_APPEAR"
ANSWERS = {
    "planning": {"candidate_id": "cand-base", "relation_ids": [], "binding_candidate_ids": [],
                 "observed_evidence_ids": [], "rationale_code": "strongest_boundary_support", "confidence": "high"},
    "binding": {"bindings": [], "relations": []},
    "editing": {"observation": "Source supports a straight segment.", "confidence": "high", "operations": [
        {"action": "refit_entity_as_line", "entity_ids": ["g000"], "record_id": None,
         "evidence_tags": ["source_boundary"]}]},
    "evaluation": {"candidate_id": "cand-base", "observation": "Preserve supported base.",
                   "decision": "preserve_base", "evidence_tags": ["source_boundary"], "confidence": "high"},
}


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (96, 64), "white").save(path)
    return path


def _candidate(path, candidate_id):
    return {"id": candidate_id, "overlay_path": str(path), "ground_truth_used": False,
            "source_stroke_support": {"edge_supported_fraction": .9},
            "annotation_support": {"compatibility_fraction": .8}, "unsupported_primitive_count": 0,
            "graph": {"entities": [{"id": "g000", "type": "LINE"}], "nodes": [], "relations": [],
                      "annotation_support": [], "ground_truth_used": False,
                      "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "validation": {"closed": True, "connected": True, "simple": True,
                                     "ordered_entity_cycle": True}}}


def _invoke(stage, settings, source):
    base = _candidate(source, "cand-base")
    if stage == "planning":
        return PlanningProvider(settings).select(source, [base])
    if stage == "binding":
        return BindingProvider(settings).select(source, source, {"records": [], "candidates": [], "relations": []})
    if stage == "editing":
        return TopologyEditProvider(settings).propose(source, source, base, [])
    return TopologyEvaluationProvider(settings).select(source, [base, _candidate(source, "cand-edit")], "cand-base")


def _mock(monkeypatch, envelope):
    calls, options = [], []
    original = httpx.AsyncClient
    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=envelope)
    def factory(**kwargs):
        options.append(kwargs)
        return original(**kwargs, transport=httpx.MockTransport(handle))
    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return calls, options


def _envelope(wire, text):
    usage = {"output_tokens": 17, "input_tokens": 31, "reasoning": PRIVATE,
             "output_tokens_details": {"reasoning_tokens": 9, "text": PRIVATE}}
    if wire == "anthropic_messages":
        return {"stop_reason": "end_turn", "content": [{"type": "thinking", "thinking": PRIVATE},
                {"type": "text", "text": text}], "usage": usage}
    if wire == "responses":
        return {"status": "completed", "output": [{"type": "reasoning", "text": PRIVATE},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
                "usage": usage}
    return {"choices": [{"finish_reason": "stop", "message": {"content": text, "reasoning_content": PRIVATE}}],
            "usage": usage}


@pytest.mark.parametrize("wire", ["chat_completions", "responses", "anthropic_messages"])
@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("thinking_mode", ["provider_default", "disabled"])
def test_stage_budget_is_protocol_specific_audited_and_still_single_request(monkeypatch, source, wire, stage, thinking_mode):
    calls, options = _mock(monkeypatch, _envelope(wire, json.dumps(ANSWERS[stage])))
    receipt = _invoke(stage, Settings(api_key="test-key", wire_api=wire, api_timeout=600,
                                     anthropic_thinking_mode=thinking_mode), source)
    expected = {"planning": 900, "binding": 1200 if wire == "responses" else 2200,
                "editing": 1600, "evaluation": 1000}[stage]
    if wire == "anthropic_messages":
        expected = 12288 if stage == "editing" else 8192
    assert receipt["schema_success"] is True and receipt["network_requests"] == len(calls) == 1
    assert calls[0]["max_output_tokens" if wire == "responses" else "max_tokens"] == expected
    assert receipt["request_max_output_tokens"] == expected
    assert calls[0].get("thinking") == ({"type": "disabled"}
                                         if wire == "anthropic_messages" and thinking_mode == "disabled" else None)
    assert receipt["anthropic_thinking_mode_requested"] == (thinking_mode if wire == "anthropic_messages" else None)
    assert receipt["total_timeout_seconds"] == 600
    assert options[0]["timeout"].read == 600 and options[0]["timeout"].connect == 10
    assert receipt["usage"] == {"input_tokens": 31, "output_tokens": 17, "output_tokens_details.reasoning_tokens": 9}
    assert PRIVATE not in json.dumps(receipt)


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("body", ["thinking_only", "partial_json", "complete_json"])
def test_increased_anthropic_budget_never_admits_truncated_answers(monkeypatch, source, stage, body):
    text = {"thinking_only": None, "partial_json": '{"candidate_id":',
            "complete_json": json.dumps(ANSWERS[stage])}[body]
    envelope = _envelope("anthropic_messages", text or "")
    envelope["stop_reason"] = "max_tokens"
    if text is None:
        envelope["content"] = envelope["content"][:1]
    calls, _ = _mock(monkeypatch, envelope)
    receipt = _invoke(stage, Settings(api_key="test-key", wire_api="anthropic_messages", api_timeout=45), source)
    assert receipt["http_success"] is True and receipt["schema_success"] is False
    assert receipt["error_code"] == "truncated_output" and receipt["finish_reason"] == "length"
    assert receipt["network_requests"] == len(calls) == 1
    assert receipt["total_timeout_seconds"] == 45
    assert not receipt.get("selected_candidate_id") and not receipt.get("operations") and not receipt.get("bindings")
    assert receipt["usage"]["output_tokens"] == 17 and PRIVATE not in json.dumps(receipt)


@pytest.mark.parametrize("stage", STAGES)
def test_invalid_reasoning_only_envelope_cannot_enter_failure_receipt(monkeypatch, source, stage):
    envelope = _envelope("anthropic_messages", "")
    envelope["content"] = envelope["content"][:1]
    calls, _ = _mock(monkeypatch, envelope)
    receipt = _invoke(stage, Settings(api_key="test-key", wire_api="anthropic_messages"), source)
    assert not receipt["schema_success"] and receipt["network_requests"] == len(calls) == 1
    assert PRIVATE not in json.dumps(receipt)


@pytest.mark.parametrize("wire,expected", [("chat_completions", 1000), ("responses", 1000), ("anthropic_messages", 2200)])
def test_text_dimension_request_retains_its_existing_budget(monkeypatch, wire, expected):
    calls, _ = _mock(monkeypatch, _envelope(wire, '{"dimensions":[["r000","radius",40,null,null]]}'))
    result = DimensionProvider(Settings(api_key="test-key", wire_api=wire), max_attempts=1).normalize(
        [{"id": "r000", "text": "R40"}])
    assert result["schema_success"] is True and len(calls) == 1
    assert calls[0]["max_output_tokens" if wire == "responses" else "max_tokens"] == expected


def test_numeric_token_usage_rejects_text_booleans_nested_payloads_and_non_counts():
    assert numeric_token_usage({"input_tokens": True, "output_tokens": -1, "total_tokens": 10.5,
                                "prompt_tokens": "17", "cache_read_input_tokens": 8, "reasoning": PRIVATE,
                                "completion_tokens_details": {"reasoning_tokens": 4, "content": PRIVATE},
                                "input_tokens_details": {"cached_tokens": False}}) == {
        "cache_read_input_tokens": 8, "completion_tokens_details.reasoning_tokens": 4}
    assert numeric_token_usage([PRIVATE]) == {}


@pytest.mark.parametrize("default", [0, 12289, True, float("inf")])
def test_output_budget_rejects_unbounded_or_invalid_defaults(default):
    with pytest.raises(ValueError):
        output_token_budget(Settings(wire_api="anthropic_messages"), "editing", default)
