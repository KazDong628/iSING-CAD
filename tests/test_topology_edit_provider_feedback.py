import asyncio
import json
import time

import httpx
import pytest
from PIL import Image

from contour_agent.config import Settings
from contour_agent.topology_edit_provider import (
    TopologyEditProvider, TopologyEvaluationProvider, bounded_iteration_feedback, validate_edit_response,
)
from contour_agent.vision_provider import _InspectionError


def _graph():
    return {"entities": [{"id": f"g{i:03d}", "type": "LINE"} for i in range(5)]}


def _operation(action, ids, record_id=None):
    return {"action": action, "entity_ids": ids, "record_id": record_id, "evidence_tags": ["source_boundary"]}


@pytest.mark.parametrize("action,ids,record_id", [
    ("refit_entity_as_line", ["g001"], None),
    ("split_chain_at_source_features", ["g004", "g000"], None),
    ("insert_annotated_fillet", ["g001", "g002"], "r003"),
])
def test_extended_editor_actions_remain_id_only(action, ids, record_id):
    value = {"observation": "原图支持此处局部重建。", "confidence": "high",
             "operations": [_operation(action, ids, record_id)]}
    assert validate_edit_response(json.dumps(value), _graph(), {"r003"}) == value
    value["operations"][0]["center"] = [20, 30]
    with pytest.raises(_InspectionError):
        validate_edit_response(json.dumps(value), _graph(), {"r003"})


@pytest.mark.parametrize("operation", [
    _operation("refit_entity_as_line", ["g001", "g002"]),
    _operation("insert_annotated_fillet", ["g001"]),
    _operation("split_chain_at_source_features", ["g000", "g002"]),
    _operation("split_chain_at_source_features", [[1, 2]]),
])
def test_invalid_extended_action_is_rejected(operation):
    value = {"observation": "证据", "confidence": "high", "operations": [operation]}
    with pytest.raises(_InspectionError):
        validate_edit_response(json.dumps(value), _graph(), {"r003"})


def test_feedback_is_bounded_to_current_evidence_and_historical_operations():
    feedback = {
        "round": 2, "issues": [{"code": "radius_value_unresolved", "entity_id": "g001", "record_id": "r003",
                                 "coordinates": [123, 456], "private_reasoning": "never-send"},
                                {"code": "unbound", "entity_id": "other-candidate-id", "record_id": "unknown"}],
        "previous_operations": [{**_operation("refit_chain_as_annotated_arc", ["g017"], "r003"),
                                  "status": "rejected", "reason": "source_deviation", "private_prompt": "never-send"}],
        "previous_rounds": [{"private_response": "never-send"}], "ground_truth": "never-send",
    }
    result = bounded_iteration_feedback(feedback, _graph(), {"r003"})
    assert result["issues"][0] == {"code": "radius_value_unresolved", "entity_id": "g001", "record_id": "r003"}
    assert result["issues"][1] == {"code": "unbound"}
    assert result["previous_operations"][0]["entity_ids"] == ["g017"]
    assert result["historical_ids_are_not_current_ids"] is True
    assert "never-send" not in json.dumps(result)
    assert "coordinates" not in json.dumps(result)


def test_editor_sends_compact_iteration_feedback(monkeypatch, tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (64, 64), "white").save(image)
    observed = []
    def handler(request):
        observed.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "observation": "缺少进一步证据。", "operations": [], "confidence": "abstain"})}}]})
    original = httpx.AsyncClient
    monkeypatch.setattr("contour_agent.topology_edit_provider.httpx.AsyncClient",
                        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)))
    receipt = TopologyEditProvider(Settings(api_key="test-key")).propose(
        image, image, {"id": "cand-02", "graph": _graph()}, [], feedback={
            "round": 2, "issues": [{"code": "radius_value_unresolved", "entity_id": "g001"}],
            "review_items": [{"code": "tangent_jump_requires_source_review",
                              "entity_ids": ["g001", "g002"], "tangent_jump_deg": 90.,
                              "tangency_required": True, "advisory_only": False}],
            "private_reasoning": "never-send", "ground_truth": {"x": 600}})
    packet = json.loads(observed[0]["messages"][1]["content"][0]["text"])
    assert packet["iteration_feedback"]["round"] == 2
    assert packet["iteration_feedback"]["issues"][0]["entity_id"] == "g001"
    review = packet["iteration_feedback"]["review_items"][0]
    assert review["tangent_jump_deg"] == 90.
    assert review["advisory_only"] is True and review["tangency_required"] is False
    assert "Never force tangency" in observed[0]["messages"][0]["content"]
    assert "never-send" not in json.dumps(observed)
    assert receipt["schema_success"] is True and receipt["feedback_issue_count"] == 1
    assert receipt["feedback_review_item_count"] == 1
    assert receipt["total_timeout_seconds"] <= 600


def test_review_feedback_whitelists_current_adjacent_ids_and_finite_bounded_angles():
    graph = {"entities": [{"id": f"g{i:03d}", "stable_id": f"edge-{i}", "type": "LINE"}
                          for i in range(30)]}
    def item(ids, angle=12.):
        return {"code": "tangent_jump_requires_source_review", "entity_ids": ids,
                "tangent_jump_deg": angle, "stable_ids": ["untrusted", "untrusted"],
                "coordinates": [1, 2], "private_reasoning": "never-send"}
    review = [item(["g000", "unknown"]), item(["g000", "g002"]),
              item(["g000", "g001"], float("nan")), item(["g000", "g001"], True),
              item(["g000", "g001"], 181.)]
    review.extend(item([f"g{i:03d}", f"g{i + 1:03d}"]) for i in range(18))
    result = bounded_iteration_feedback({"review_items": review}, graph, set())
    assert len(result["review_items"]) == 12 and result["issues"] == []
    assert result["review_items"][0]["entity_ids"] == ["g000", "g001"]
    assert result["review_items"][0]["stable_ids"] == ["edge-0", "edge-1"]
    serialized = json.dumps(result, allow_nan=False)
    assert all(word not in serialized for word in ("coordinates", "never-send", "untrusted"))


def test_editor_total_deadline_interrupts_slow_service(monkeypatch, tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (64, 64), "white").save(image)
    async def handler(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json={})
    original = httpx.AsyncClient
    monkeypatch.setattr("contour_agent.topology_edit_provider.httpx.AsyncClient",
                        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)))
    started = time.monotonic()
    receipt = TopologyEditProvider(Settings(api_key="test-key", api_timeout=.06)).propose(
        image, image, {"id": "cand-01", "graph": _graph()}, [])
    assert time.monotonic() - started < .5
    assert receipt["error_code"] == "timeout"
    assert receipt["schema_success"] is False and receipt["network_requests"] == 1
    assert receipt["request_started"] is True
    assert receipt["total_timeout_seconds"] == .06


def test_default_timeout_allows_ten_minutes_but_preserves_explicit_short_budget(monkeypatch):
    monkeypatch.delenv("CONTOUR_API_TIMEOUT", raising=False)
    assert Settings().api_timeout == 600
    monkeypatch.setenv("CONTOUR_API_TIMEOUT", "45")
    assert Settings().api_timeout == 45
    monkeypatch.setenv("CONTOUR_API_TIMEOUT", "1200")
    assert Settings().api_timeout == 600


@pytest.mark.parametrize("wire_api,envelope", [
    ("chat_completions", {"choices": [{"finish_reason": "length", "message": {"content": "{\"observation\":"}}]}),
    ("responses", {"status": "incomplete", "output": []}),
    ("anthropic_messages", {"stop_reason": "max_tokens", "content": [{"type": "thinking", "thinking": "private"}]}),
])
def test_truncated_provider_output_is_reported_separately_from_schema_failure(monkeypatch, tmp_path, wire_api, envelope):
    image = tmp_path / "source.png"
    Image.new("RGB", (64, 64), "white").save(image)
    original = httpx.AsyncClient
    monkeypatch.setattr("contour_agent.topology_edit_provider.httpx.AsyncClient",
                        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(
                            lambda request: httpx.Response(200, json=envelope))))
    receipt = TopologyEditProvider(Settings(api_key="test-key", wire_api=wire_api)).propose(
        image, image, {"id": "cand-01", "graph": _graph()}, [])
    assert receipt["http_success"] is True
    assert receipt["schema_success"] is False
    assert receipt["error_code"] == "truncated_output"
    assert "private" not in json.dumps(receipt)


def test_evaluator_reserves_admissible_base_within_five_candidate_budget(monkeypatch, tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (64, 64), "white").save(image)
    candidates = []
    for index, count in enumerate((12, 6, 5, 4, 7, 8, 9)):
        candidates.append({"id": f"cand-{index}", "overlay_path": str(image), "ground_truth_used": False,
                           "graph": {"entities": [{"id": f"g{i:03d}", "type": "LINE"} for i in range(count)],
                                     "relations": [], "annotation_support": [], "ground_truth_used": False,
                                     "validation": {"closed": True, "connected": True, "simple": True,
                                                    "ordered_entity_cycle": True}},
                           "source_stroke_support": {"edge_supported_fraction": .56 if index == 0 else .95},
                           "annotation_support": {"compatibility_fraction": .8},
                           "unsupported_primitive_count": 1, "binding_candidate_ids": []})
    observed = []
    def handler(request):
        observed.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "candidate_id": "cand-0", "observation": "保留可验证的基线。", "decision": "preserve_base",
            "evidence_tags": ["source_boundary"], "confidence": "high"})}}]})
    original = httpx.AsyncClient
    monkeypatch.setattr("contour_agent.topology_edit_provider.httpx.AsyncClient",
                        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)))
    receipt = TopologyEvaluationProvider(Settings(api_key="test-key")).select(image, candidates, "cand-0")
    assert receipt["schema_success"] is True
    assert receipt["selected_candidate_id"] == "cand-0"
    assert receipt["local_evaluation"]["ranked_candidate_ids"][-1] == "cand-0"
    packet = json.loads(observed[0]["messages"][1]["content"][0]["text"])
    assert packet["candidate_order"][0] == "cand-0"
    assert len(packet["candidate_order"]) == 5
    assert len(packet["candidate_image_order"]) == 5
