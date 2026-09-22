import json

import httpx
from PIL import Image

from contour_agent.config import Settings
from contour_agent.feedback_provider import FeedbackProvider


def _candidate(candidate_id, overlay_path, *, valid=True):
    return {
        "id": candidate_id,
        "overlay_path": str(overlay_path),
        "ground_truth_used": False,
        "graph": {
            "ground_truth_used": False,
            "validation": {"closed": valid, "connected": True, "simple": True, "ordered_entity_cycle": True},
            "entities": [
                {"id": "g000", "type": "LINE", "start": [0, 0], "end": [1, 0]},
                {"id": "g001", "type": "ARC", "start": [1, 0], "end": [0, 0]},
            ],
            "relations": [],
            "annotation_support": [
                {"record_id": "ev001", "candidate_entity_id": "g000", "status": "candidate_supported", "type_compatible": True}
            ],
        },
        "source_stroke_support": {"edge_supported_fraction": 0.9},
        "annotation_support": {"compatibility_fraction": 0.9},
        "unsupported_primitive_count": 1,
        "binding_candidate_ids": [],
        "ground_truth_secret": "NEVER-SEND-GT",
    }


def _image(path):
    Image.new("RGB", (96, 64), "white").save(path)
    return path


def _patch_client(monkeypatch, handler):
    original = httpx.AsyncClient

    def factory(*args, **kwargs):
        return original(*args, **kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr("contour_agent.feedback_provider.httpx.AsyncClient", factory)


def test_feedback_provider_without_key_abstains_before_image_read(tmp_path):
    missing = tmp_path / "missing.png"
    receipt = FeedbackProvider(Settings(api_key="")).select(
        missing, missing, [_candidate("topo-a", missing)], "右上角不连续"
    )
    assert receipt["status"] == "not_configured"
    assert receipt["network_requests"] == 0
    assert receipt["selected_candidate_id"] is None


def test_feedback_provider_selects_only_admissible_candidate_and_redacts_geometry(monkeypatch, tmp_path):
    screenshot = _image(tmp_path / "feedback.png")
    current = _image(tmp_path / "current.png")
    overlay_a = _image(tmp_path / "a.png")
    overlay_b = _image(tmp_path / "b.png")
    candidates = [_candidate("topo-a", overlay_a), _candidate("topo-b", overlay_b, valid=False)]
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "candidate_id": "topo-a",
            "observation": "反馈区域在候选 A 中保持连续。",
            "proposed_action": "切换到候选 A 并重新执行本地几何门禁。",
            "evidence_tags": ["user_feedback", "continuity"],
            "rationale_code": "feedback_region_improved",
            "confidence": "high",
            "operations": [{"action": "replace_boundary_chain_with_line", "side": "top", "basis": "顶部应为直线"}],
        }, ensure_ascii=False)}}]})

    _patch_client(monkeypatch, handler)
    receipt = FeedbackProvider(Settings(api_key="private-test-key")).select(
        screenshot, current, candidates, "这里不连续", current_candidate_id="topo-old"
    )

    assert receipt["status"] == "succeeded" and receipt["schema_success"]
    assert receipt["selected_candidate_id"] == "topo-a"
    assert receipt["input_candidate_ids"] == ["topo-a"]
    assert receipt["operations"][0]["side"] == "top"
    wire = json.dumps(payloads[0], ensure_ascii=False)
    assert wire.count('"type": "image_url"') == 3
    assert "NEVER-SEND-GT" not in wire and '"start"' not in wire and '"end"' not in wire
    assert "private-test-key" not in json.dumps(receipt)


def test_feedback_provider_rejects_unknown_candidate(monkeypatch, tmp_path):
    screenshot = _image(tmp_path / "feedback.png")
    current = _image(tmp_path / "current.png")
    overlay = _image(tmp_path / "candidate.png")

    def handler(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "candidate_id": "topo-unknown", "observation": "观察", "proposed_action": "修改",
            "evidence_tags": ["user_feedback"], "rationale_code": "feedback_region_improved", "confidence": "high",
            "operations": [],
        }, ensure_ascii=False)}}]})

    _patch_client(monkeypatch, handler)
    receipt = FeedbackProvider(Settings(api_key="test")).select(
        screenshot, current, [_candidate("topo-a", overlay)], "修复此处"
    )
    assert receipt["http_success"] and not receipt["schema_success"]
    assert receipt["error_code"] == "unknown_candidate"
    assert receipt["selected_candidate_id"] is None


def test_feedback_provider_discards_unknown_explanatory_tags_but_keeps_strict_operation(monkeypatch, tmp_path):
    screenshot = _image(tmp_path / "feedback.png")
    current = _image(tmp_path / "current.png")
    overlay = _image(tmp_path / "candidate.png")

    def handler(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "candidate_id": "topo-a", "observation": "顶部和右侧应为直线。", "proposed_action": "执行局部直线替换。",
            "evidence_tags": ["user_feedback", "straight_edge", "hatching_excluded"],
            "rationale_code": "simplify_unsupported_primitives", "confidence": "high",
            "operations": [{"action": "replace_boundary_chain_with_line", "side": "right", "basis": "右侧是一条线段"}],
        }, ensure_ascii=False)}}]})

    _patch_client(monkeypatch, handler)
    receipt = FeedbackProvider(Settings(api_key="test")).select(
        screenshot, current, [_candidate("topo-a", overlay)], "右侧是一条线段"
    )
    assert receipt["schema_success"]
    assert receipt["evidence_tags"] == ["user_feedback", "hatching_excluded"]
    assert receipt["discarded_evidence_tag_count"] == 1
    assert receipt["operations"][0]["action"] == "replace_boundary_chain_with_line"
