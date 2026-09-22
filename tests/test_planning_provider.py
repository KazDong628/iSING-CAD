import hashlib
import json

import httpx
from PIL import Image
import pytest

from contour_agent.config import Settings
from contour_agent.planning_provider import PlanningProvider, evaluate_candidates, validate_plan
from contour_agent.vision_provider import _InspectionError


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (320, 180), "white").save(path)
    return path


def candidate(candidate_id, *, source_hash=None, support=.9, annotation=.8, count=4,
              unsupported=1, valid=True):
    points = [[20, 20], [280, 20], [280, 150], [20, 150]][:count]
    if count > len(points):
        points += [[20+i, 80] for i in range(count-len(points))]
    entities = [{"id": f"g{i:03d}", "type": "LINE", "start": [i, i], "end": [i+1, i+1]}
                for i in range(count)]
    graph = {
        "source_sha256": source_hash,
        "ground_truth_used": False,
        "validation": {"closed": valid, "connected": True, "simple": True,
                       "ordered_entity_cycle": True},
        "nodes": [{"id": f"v{i:03d}", "source_px": point} for i, point in enumerate(points)],
        "entities": entities,
        "relations": [{"id": "rel001"}],
        "annotation_support": [{"record_id": "ev001", "candidate_entity_id": "g000",
                                "status": "candidate_supported", "type_compatible": True}],
    }
    return {
        "id": candidate_id,
        "graph": graph,
        "source_stroke_support": {"edge_supported_fraction": support},
        "annotation_support": {"compatibility_fraction": annotation},
        "unsupported_primitive_count": unsupported,
        "binding_candidate_ids": ["bind001"],
        "planner_signals": {"prior_rank": 1},
        "ground_truth_used": False,
        "ground_truth_secret": "MUST NOT SEND",
        "invented_dimension": {"nominal": 12345},
    }


def _patch_client(monkeypatch, handler, options=None):
    original = httpx.AsyncClient

    def factory(*args, **kwargs):
        if options is not None:
            options.append(kwargs.copy())
        return original(*args, **kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr("contour_agent.planning_provider.httpx.AsyncClient", factory)


def _response(content):
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]})


def test_local_evaluator_makes_annotation_primary_and_rejects_hard_invalidity():
    boundary_best = candidate("topo-a", support=.99, annotation=.20, unsupported=1)
    annotation_best = candidate("topo-b", support=.80, annotation=.95, unsupported=2)
    invalid = candidate("topo-c", support=.99, annotation=1.0, unsupported=0, valid=False)
    weak_boundary = candidate("topo-d", support=.54, annotation=1.0, unsupported=0)

    result = evaluate_candidates([boundary_best, annotation_best, invalid, weak_boundary])

    assert result["ranked_candidate_ids"][0] == "topo-b"
    assert result["weights"]["annotation_coverage"] == .50
    by_id = {row["candidate_id"]: row for row in result["evaluated"]}
    assert not by_id["topo-c"]["admissible"]
    assert "closed_validation_failed" in by_id["topo-c"]["rejection_reasons"]
    assert "insufficient_source_boundary_support" in by_id["topo-d"]["rejection_reasons"]
    assert "topo-c" not in result["bounded_candidate_ids"]
    assert result["ground_truth_used"] is False


def test_local_evaluator_bounds_online_set_to_five_and_sends_no_geometry():
    rows = [candidate(f"topo-{i}", annotation=.1*i, unsupported=0) for i in range(7)]
    result = evaluate_candidates(rows)
    assert len(result["bounded_candidates"]) == 5
    assert all("overlay_path" not in row for row in result["bounded_candidates"])
    wire = json.dumps(result["bounded_candidates"])
    assert "source_px" not in wire and '"start"' not in wire and "nominal" not in wire


def test_response_validation_allows_abstention_and_only_whitelisted_ids():
    allowed = {"topo-a": {"relation_ids": ["rel001"], "binding_candidate_ids": ["bind001"],
                           "evidence_ids": ["ev001"]}}
    selected = validate_plan(json.dumps({
        "candidate_id": "topo-a", "relation_ids": ["rel001"],
        "binding_candidate_ids": ["bind001"], "observed_evidence_ids": ["ev001"],
        "rationale_code": "boundary_and_annotation_agree", "confidence": "high",
    }), allowed)
    assert selected["candidate_id"] == "topo-a"
    abstained = validate_plan(json.dumps({
        "candidate_id": None, "relation_ids": [], "binding_candidate_ids": [],
        "observed_evidence_ids": [], "rationale_code": "ambiguous_candidates", "confidence": "abstain",
    }), allowed)
    assert abstained["candidate_id"] is None

    with pytest.raises(_InspectionError):
        validate_plan(json.dumps({
            "candidate_id": "topo-a", "relation_ids": [], "binding_candidate_ids": ["unknown"],
            "observed_evidence_ids": [], "rationale_code": "boundary_and_annotation_agree", "confidence": "high",
        }), allowed)
    with pytest.raises(_InspectionError):
        validate_plan(json.dumps({
            "candidate_id": "topo-a", "relation_ids": [], "binding_candidate_ids": [],
            "observed_evidence_ids": [], "rationale_code": "boundary_and_annotation_agree", "confidence": "high",
            "dimension_values": [40],
        }), allowed)


def test_provider_one_bounded_call_uses_source_rendered_overlays_and_redacted_receipt(monkeypatch, source):
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    candidates = [candidate(f"topo-{i}", source_hash=source_hash, annotation=.9-.05*i, unsupported=0)
                  for i in range(6)]
    payloads, options = [], []

    def handler(request):
        payloads.append(json.loads(request.content))
        return _response(json.dumps({
            "candidate_id": "topo-0", "relation_ids": ["rel001"],
            "binding_candidate_ids": ["bind001"], "observed_evidence_ids": ["ev001"],
            "rationale_code": "boundary_and_annotation_agree", "confidence": "high",
        }))

    _patch_client(monkeypatch, handler, options)
    receipt = PlanningProvider(Settings(api_key="test-private-token")).select(source, candidates)

    assert receipt["status"] == "succeeded" and receipt["schema_success"]
    assert receipt["selected_candidate_id"] == "topo-0"
    assert receipt["network_requests"] == 1 and receipt["input_image_count"] == 6
    assert all(row["overlay_sent"] for row in receipt["candidate_overlays"])
    assert len(receipt["candidate_summary_sha256"]) == 64
    wire = payloads[0]
    assert wire["max_tokens"] == 900
    assert sum(part["type"] == "image_url" for part in wire["messages"][1]["content"]) == 6
    serialized = json.dumps(wire)
    assert "MUST NOT SEND" not in serialized and "12345" not in serialized and "source_px" not in serialized
    assert "test-private-token" not in json.dumps(receipt)
    assert options[0]["verify"] is True and options[0]["follow_redirects"] is False
    assert options[0]["trust_env"] is False


def test_unknown_or_hard_invalid_api_selection_cannot_override_local_gate(monkeypatch, source):
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    valid = candidate("topo-valid", source_hash=source_hash)
    invalid = candidate("topo-invalid", source_hash=source_hash, valid=False)
    _patch_client(monkeypatch, lambda request: _response(json.dumps({
        "candidate_id": "topo-invalid", "relation_ids": [], "binding_candidate_ids": [],
        "observed_evidence_ids": [], "rationale_code": "strongest_boundary_support", "confidence": "high",
    })))
    receipt = PlanningProvider(Settings(api_key="test")).select(source, [valid, invalid])
    assert receipt["http_success"] and not receipt["schema_success"]
    assert receipt["error_code"] == "unknown_candidate_id"
    assert receipt["selected_candidate_id"] is None


def test_no_admissible_candidate_skips_network(monkeypatch, source):
    calls = []
    _patch_client(monkeypatch, lambda request: calls.append(request), [])
    row = candidate("topo-invalid", source_hash=hashlib.sha256(source.read_bytes()).hexdigest(), valid=False)
    receipt = PlanningProvider(Settings(api_key="test")).select(source, [row])
    assert receipt["status"] == "skipped" and receipt["error_code"] == "no_admissible_candidates"
    assert receipt["network_requests"] == 0 and calls == []


def test_schema_failure_redacts_echoed_credentials(monkeypatch, source):
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    _patch_client(monkeypatch, lambda request: _response("bad sk-echo Bearer echoed.token secret-key"))
    receipt = PlanningProvider(Settings(api_key="secret-key")).select(source, [candidate("topo-a", source_hash=source_hash)])
    assert receipt["http_success"] and not receipt["schema_success"]
    assert receipt["error_code"] == "invalid_json"
    assert "secret-key" not in json.dumps(receipt) and "echoed.token" not in receipt["response_excerpt"]
