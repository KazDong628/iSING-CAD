import asyncio
import copy
import json
import time

import httpx
import numpy as np
from PIL import Image
import pytest

from contour_agent.config import Settings
from contour_agent.radius_target_provider import RadiusTargetProvider, validate_radius_target_response, locate_radius_targets
from contour_agent import topology_candidates
from contour_agent.vision_provider import _InspectionError
from contour_agent.topology_edit_provider import _rank_radius_detail_records, _radius_detail_payload


def document(count=2):
    return {"records": [{"text": f"R{10+i}", "box": [[40, 40], [80, 40], [80, 70], [40, 70]],
                         "irrelevant": "DO-NOT-SEND", "ground_truth": {"center": [3, 4]}}
                        for i in range(count)], "ground_truth": "DO-NOT-SEND"}


def proposal(record_id="r000"):
    return {"record_id": record_id, "tip_px": [130, 110], "shaft_px": [70, 75]}


def test_schema_accepts_source_pixels_and_accounts_for_unknown():
    payload = {"proposals": [proposal()], "unknown_record_ids": ["r001"]}
    result = validate_radius_target_response(json.dumps(payload), {"r000", "r001"}, (300, 200))
    assert result == payload
    assert "arrowhead_verified" not in result["proposals"][0]


@pytest.mark.parametrize("change", [
    lambda p: p["proposals"][0].update(radius=10),
    lambda p: p["proposals"][0].update(tip_px=[float("nan"), 110]),
    lambda p: p["proposals"][0].update(tip_px=[300, 110]),
    lambda p: p["proposals"][0].update(tip_px=[True, 110]),
    lambda p: p["proposals"][0].update(shaft_px=[130, 110]),
    lambda p: p["proposals"].append(proposal()),
    lambda p: p.update(unknown_record_ids=[]),
    lambda p: p.update(unknown_record_ids=["r000", "r001"]),
    lambda p: p["proposals"][0].update(record_id="r999"),
    lambda p: p.update(arrow_absent=True),
])
def test_schema_rejects_false_certification_and_bad_coordinates(change):
    value = {"proposals": [proposal()], "unknown_record_ids": ["r001"]}
    change(value)
    with pytest.raises(_InspectionError):
        validate_radius_target_response(json.dumps(value), {"r000", "r001"}, (300, 200))


def test_source_only_request_keeps_leaderless_records_and_record_budget(monkeypatch, tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (1200, 900), "white").save(image)
    original = document(14)
    before = copy.deepcopy(original)
    observed = []

    def handler(request):
        body = json.loads(request.content)
        observed.append(body)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "proposals": [proposal()], "unknown_record_ids": [f"r{i:03d}" for i in range(1, 12)]})}}]})

    client = httpx.AsyncClient
    monkeypatch.setattr("contour_agent.radius_target_provider.httpx.AsyncClient",
                        lambda **kwargs: client(**kwargs, transport=httpx.MockTransport(handler)))
    result = RadiusTargetProvider(Settings(api_key="not-for-receipt")).locate(image, original)
    assert result["http_success"] and result["schema_success"]
    assert result["network_requests"] == 1
    assert result["omitted_record_ids"] == ["r012", "r013"]
    assert result["unknown_record_ids"] == [f"r{i:03d}" for i in range(1, 12)]
    assert result["arrowheads_verified"] is False and result["dimensions_verified"] is False
    assert original == before
    packet = json.loads(observed[0]["messages"][1]["content"][0]["text"])
    assert len(packet["record_ids"]) == 12 and len(packet["crop_order"]) == 12
    assert len(observed[0]["messages"][1]["content"]) == 14  # text + source + 12 crops
    assert "DO-NOT-SEND" not in json.dumps(observed)
    assert "nominal" not in json.dumps(packet)
    assert all(row["kind"] == "radius_source_crop" and not row["overlay_sent"] for row in result["input_images"][1:])
    assert "not-for-receipt" not in json.dumps(result)
    assert "response_excerpt" not in result


def test_transport_success_does_not_mean_schema_or_local_verification_success(monkeypatch, tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (300, 200), "white").save(image)
    client = httpx.AsyncClient
    monkeypatch.setattr("contour_agent.radius_target_provider.httpx.AsyncClient",
                        lambda **kwargs: client(**kwargs, transport=httpx.MockTransport(
                            lambda request: httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
                                "content": '{"proposals":[],"unknown_record_ids":[],"verified":true}'}}]}))))
    result = RadiusTargetProvider(Settings(api_key="test")).locate(image, document())
    assert result["http_success"] and not result["schema_success"]
    assert result["proposals"] == [] and result["unknown_record_ids"] == ["r000", "r001"]
    assert result["error_code"] == "schema_mismatch"


def test_record_filter_preserves_original_ids_and_crop_coordinate_mapping(monkeypatch, tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (1200, 900), "white").save(image)
    observed = []

    def handler(request):
        observed.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "proposals": [], "unknown_record_ids": ["r001", "r003"]})}}]})

    client = httpx.AsyncClient
    monkeypatch.setattr("contour_agent.radius_target_provider.httpx.AsyncClient",
                        lambda **kwargs: client(**kwargs, transport=httpx.MockTransport(handler)))
    result = RadiusTargetProvider(Settings(api_key="test")).locate(image, document(5), record_ids=["r003", "r001"])
    assert result["schema_success"] and result["input_record_ids"] == ["r001", "r003"]
    assert result["unrequested_record_ids"] == ["r000", "r002", "r004"]
    packet = json.loads(observed[0]["messages"][1]["content"][0]["text"])
    assert packet["record_ids"] == ["r001", "r003"]
    for crop in packet["crop_order"]:
        mapping = crop["pixel_mapping"]
        assert mapping["source_origin_px"] == crop["crop_box_source_px"][:2]
        assert mapping["content_origin_display_px"] == [48, 32]
        assert mapping["content_size_display_px"][0] < crop["input_image_size"][0]


@pytest.mark.parametrize("ids", [["r999"], ["r000", "r000"], "r000"])
def test_invalid_record_filter_fails_before_network(tmp_path, ids):
    result = RadiusTargetProvider(Settings(api_key="test")).locate(tmp_path/"missing.png", document(), record_ids=ids)
    assert result["error_code"] == "invalid_record_selection" and result["network_requests"] == 0


def test_single_request_deadline_preserves_unknown_without_retry(monkeypatch, tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (300, 200), "white").save(image)

    async def handler(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json={})

    client = httpx.AsyncClient
    monkeypatch.setattr("contour_agent.radius_target_provider.httpx.AsyncClient",
                        lambda **kwargs: client(**kwargs, transport=httpx.MockTransport(handler)))
    started = time.monotonic()
    result = RadiusTargetProvider(Settings(api_key="test", api_timeout=.08)).locate(image, document())
    assert time.monotonic() - started < .6
    assert result["error_code"] == "timeout" and result["network_requests"] == 1
    assert not result["schema_success"] and result["unknown_record_ids"] == ["r000", "r001"]


def test_inventory_calls_local_verifier_and_rejects_unverified_proposals(monkeypatch):
    ring = np.array([[100., 40.], [180., 40.], [180., 80.], [100., 80.], [100., 40.]])
    record = {"id": "r000", "text": "R36", "parsed": {"kind": "radius", "nominal": 36},
              "box": [[230, 30], [265, 30], [265, 50], [230, 50]],
              "source_arrow_proposals": [{"tip_px": [180, 40], "shaft_px": [230, 40]}]}
    monkeypatch.setattr(topology_candidates, "_leaders", lambda *args: [])
    calls = []

    def verifier(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr(topology_candidates, "verify_source_arrow_proposal", verifier)
    rows, _ = topology_candidates._annotation_inventory(np.full((140, 300), 255, np.uint8), [record], ring, 2.)
    assert calls
    assert rows[0]["leader_status"] == "not_detected"
    assert rows[0]["source_arrow_proposal_count"] == 1
    assert rows[0]["locally_verified_source_arrow_proposal_count"] == 0
    assert "leader" not in rows[0]


def test_inventory_retains_locally_verified_crossing_source_evidence(monkeypatch):
    ring = np.array([[100., 40.], [180., 40.], [180., 80.], [100., 80.], [100., 40.]])
    record = {"id": "r000", "text": "R36", "parsed": {"kind": "radius", "nominal": 36},
              "box": [[10, 30], [40, 30], [40, 50], [10, 50]],
              "source_arrow_proposals": [{"tip_px": [180, 40], "shaft_px": [45, 40]}]}
    monkeypatch.setattr(topology_candidates, "_leaders", lambda *args: [])
    monkeypatch.setattr(topology_candidates, "verify_source_arrow_proposal", lambda *args, **kwargs: {
        "method": "agent_proposed_source_arrow_locally_verified", "arrowhead_verified": True,
        "arrowhead": {"tip_px": [180, 40]}, "segment_px": [[45, 40], [180, 40]], "score": 1.,
        "shaft_evidence": {"verified": True}, "crossing_source_contour": True,
        "contour_visibility": {"verified": False},
        "crossing_admission": "explicit_directed_proposal_with_full_source_shaft_and_arrow"})
    rows, _ = topology_candidates._annotation_inventory(np.full((140, 300), 255, np.uint8), [record], ring, 2.)
    assert rows[0]["leader_status"] == "directed_arrow_candidate"
    assert rows[0]["leader"]["arrowhead_verified"] is True
    assert rows[0]["leader"]["crossing_source_contour"] is True
    assert rows[0]["leader"]["target_source_px"] == [180., 40.]


def test_editor_reserves_detail_for_missing_leader_and_can_crop_it(tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (1200, 900), "white").save(image)
    located = [{"record_id": f"r{i:03d}", "kind": "radius", "nominal": 10,
                "source_box": [[40, 40], [80, 70]], "leader": {"target_source_px": [150, 110]}}
               for i in range(6)]
    missing = {"record_id": "r006", "kind": "radius", "nominal": 170,
               "source_box": [[600, 400], [700, 500]], "leader_status": "not_detected"}
    graph = {"entities": [{"id": f"g{i:03d}", "type": "ARC", "radius": 100} for i in range(6)],
             "annotation_support": [{"record_id": f"r{i:03d}", "kind": "radius", "status": "candidate_supported",
                                     "candidate_entity_id": f"g{i:03d}"} for i in range(6)]}
    selected = _rank_radius_detail_records([*located, missing], graph, limit=6)
    assert len(selected) == 6 and any(row[1]["record_id"] == "r006" for row in selected)
    detail = _radius_detail_payload(image, image, missing)
    assert detail is not None and detail[1]["record_id"] == "r006"


def test_bounded_second_call_retries_only_locally_unverified_and_preserves_ids(monkeypatch, tmp_path):
    source = document(3)
    untouched = copy.deepcopy(source)
    calls = []
    stages = []

    class Provider:
        settings = Settings(api_timeout=25.)

        def locate(self, path, received_document, **kwargs):
            assert received_document == untouched
            calls.append(kwargs)
            if len(calls) == 1:
                return {"status": "succeeded", "http_success": True, "schema_success": True, "network_requests": 1,
                        "proposals": [proposal("r000"), proposal("r001")], "unknown_record_ids": ["r002"]}
            assert kwargs == {"record_ids": ["r001", "r002"]}
            return {"status": "succeeded", "http_success": True, "schema_success": True, "network_requests": 1,
                    "proposals": [{**proposal("r001"), "tip_px": [131, 110]}, proposal("r002")],
                    "unknown_record_ids": []}

    local_calls = []

    def local(path, doc, baseline):
        local_calls.append(copy.deepcopy(doc))
        return {"status": "completed", "verified_record_ids": ["r000"] if len(local_calls) == 1 else ["r000", "r001", "r002"]}

    monkeypatch.setattr("contour_agent.radius_target_provider._local_radius_inventory", local)
    result_doc, receipt = locate_radius_targets("source.png", source, {"oracle_mask_conditioned": True}, tmp_path,
                                               Provider(), lambda *args: stages.append(args))
    assert source == untouched
    assert calls == [{}, {"record_ids": ["r001", "r002"]}]
    assert receipt["attempt_count"] == receipt["network_requests"] == 2
    assert receipt["total_request_budget_seconds"] == 50.
    assert receipt["http_success"] and receipt["schema_success"]
    assert receipt["all_arrows_locally_verified"] and not receipt["dimensions_verified"]
    assert receipt["model_unknown_record_ids"] == receipt["locally_unverified_record_ids"] == []
    assert result_doc["records"][0]["source_arrow_proposals"] == [proposal("r000")]
    assert result_doc["records"][1]["source_arrow_proposals"][0]["tip_px"] == [131, 110]
    assert all(len(row["source_arrow_proposals"]) <= 2 for row in result_doc["records"])
    assert len(stages) == 4
    assert all((tmp_path/name).is_file() for name in ("radius-targets-attempt-01.json", "radius-targets-attempt-02.json",
                                                    "radius-targets-local-01.json", "radius-targets-local-02.json", "radius-targets.json"))


def test_failed_focused_retry_preserves_first_evidence_and_separates_status(monkeypatch, tmp_path):
    class Provider:
        settings = Settings(api_timeout=600.)
        count = 0

        def locate(self, *args, **kwargs):
            self.count += 1
            if self.count == 1:
                return {"status": "succeeded", "http_success": True, "schema_success": True, "network_requests": 1,
                        "proposals": [proposal("r000"), proposal("r001")], "unknown_record_ids": []}
            return {"status": "failed", "http_success": True, "schema_success": False, "network_requests": 1,
                    "error_code": "invalid_json", "proposals": [], "unknown_record_ids": ["r001"]}

    monkeypatch.setattr("contour_agent.radius_target_provider._local_radius_inventory", lambda *args:
                        {"status": "completed", "verified_record_ids": ["r000"]})
    enriched, receipt = locate_radius_targets("source.png", document(), {}, tmp_path, Provider())
    assert receipt["http_success"] and not receipt["schema_success"] and receipt["any_schema_success"]
    assert receipt["locally_verified_record_ids"] == ["r000"]
    assert receipt["locally_unverified_record_ids"] == ["r001"]
    assert receipt["model_unknown_record_ids"] == []  # A bad schema cannot assert model absence/unknown.
    assert enriched["records"][0]["source_arrow_proposals"] == [proposal("r000")]
    assert enriched["records"][1]["source_arrow_proposals"] == [proposal("r001")]
    assert receipt["verified_absent_arrow_records"] == []


def test_complete_local_verification_stops_without_second_network_call(monkeypatch, tmp_path):
    class Provider:
        count = 0

        def locate(self, *args, **kwargs):
            self.count += 1
            return {"status": "succeeded", "http_success": True, "schema_success": True, "network_requests": 1,
                    "proposals": [proposal("r000")], "unknown_record_ids": []}

    monkeypatch.setattr("contour_agent.radius_target_provider._local_radius_inventory", lambda *args:
                        {"status": "completed", "verified_record_ids": ["r000"]})
    provider = Provider()
    _, receipt = locate_radius_targets("source.png", document(1), {}, tmp_path, provider)
    assert provider.count == receipt["attempt_count"] == receipt["network_requests"] == 1
    assert not (tmp_path/"radius-targets-attempt-02.json").exists()


def test_provider_interruption_is_not_converted_to_ordinary_failure(monkeypatch, tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (300, 200), "white").save(image)

    def handler(request):
        raise InterruptedError("cancelled")

    client = httpx.AsyncClient
    monkeypatch.setattr("contour_agent.radius_target_provider.httpx.AsyncClient",
                        lambda **kwargs: client(**kwargs, transport=httpx.MockTransport(handler)))
    with pytest.raises(InterruptedError):
        RadiusTargetProvider(Settings(api_key="test")).locate(image, document())


def test_aggregate_provider_interruption_stops_without_retry(tmp_path):
    class Provider:
        calls = 0

        def locate(self, *args, **kwargs):
            self.calls += 1
            raise InterruptedError("cancelled")

    provider = Provider()
    with pytest.raises(InterruptedError):
        locate_radius_targets("source.png", document(), {}, tmp_path, provider)
    assert provider.calls == 1
    assert not (tmp_path/"radius-targets.json").exists()


@pytest.mark.parametrize("interrupted", [False, True])
def test_second_local_verifier_failure_preserves_first_verified_evidence(monkeypatch, tmp_path, interrupted):
    class Provider:
        calls = 0

        def locate(self, *args, **kwargs):
            self.calls += 1
            ids = ["r000", "r001"] if self.calls == 1 else ["r001"]
            return {"status": "succeeded", "http_success": True, "schema_success": True, "network_requests": 1,
                    "proposals": [proposal(rid) for rid in ids], "unknown_record_ids": []}

    count = 0

    def local(*args):
        nonlocal count
        count += 1
        if count == 1:
            return {"status": "completed", "verified_record_ids": ["r000"]}
        if interrupted:
            raise InterruptedError("cancelled")
        raise ValueError("internal message must not be persisted")

    monkeypatch.setattr("contour_agent.radius_target_provider._local_radius_inventory", local)
    if interrupted:
        with pytest.raises(InterruptedError):
            locate_radius_targets("source.png", document(), {}, tmp_path, Provider())
        assert json.loads((tmp_path/"radius-targets-attempt-01.json").read_text())["proposals"][0] == proposal("r000")
    else:
        enriched, receipt = locate_radius_targets("source.png", document(), {}, tmp_path, Provider())
        assert receipt["locally_verified_record_ids"] == ["r000"]
        assert receipt["local_verification_status"] == "unavailable"
        assert receipt["locally_unverified_record_ids"] == ["r001"]
        assert enriched["records"][0]["source_arrow_proposals"] == [proposal("r000")]
        assert "internal message" not in (tmp_path/"radius-targets-local-02.json").read_text()
    assert (tmp_path/"radius-targets-attempt-02.json").exists()
    assert (tmp_path/"radius-targets-local-01.json").exists()
