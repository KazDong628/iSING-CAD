"""Every source binding page is bounded, auditable, and independently checked."""
import asyncio
import copy
import json
import time

import httpx
import pytest
from PIL import Image

from contour_agent.binding_provider import (BindingProvider, _binding_pages, _merge_page_selections,
                                           validate_receipt_selection, MAX_BINDING_PAGES)
from contour_agent.config import Settings
from contour_agent.vision_provider import _InspectionError


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (32, 32), "white").save(path)
    return path


def _inventory(count=35, groups=2, relations=35):
    records = [{"id": f"r{i:03d}", "text": "R5", "parsed": {"kind": "radius", "nominal": 5.}, "box": []}
               for i in range(count)]
    candidates = [{"id": f"c{i:03d}_{j}", "record_id": f"r{i:03d}", "kind": "radius", "entities": ["g001"],
                   "nodes": [], "evidence": {}, "value": 5.} for i in range(count) for j in range(groups)]
    relation_rows = [{"id": f"rel{i:03d}", "type": "tangent", "entities": ["g000", "g001"], "nodes": ["v001"]}
                     for i in range(relations)]
    return {"units": "mm", "records": records[:24], "candidates": candidates[:48],
            "all_records": records, "all_candidates": candidates, "relations": relation_rows,
            "ground_truth_secret": "GT_MUST_NOT_BE_SENT"}


def _patch(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr("contour_agent.binding_provider.httpx.AsyncClient",
                        lambda *args, **kwargs: original(*args, **kwargs, transport=httpx.MockTransport(handler)))


def _packet(request):
    wire = json.loads(request.content)
    parts = wire["input"][0]["content"] if "input" in wire else wire["messages"][1]["content"]
    return json.loads(parts[0]["text"])


def _response(packet):
    chosen = []
    for record in packet["records"]:
        candidate = next(row for row in packet["candidates"] if row["record_id"] == record["id"])
        chosen.append({"record_id": record["id"], "candidate_id": candidate["id"], "observed_text": record["text"]})
    content = json.dumps({"bindings": chosen, "relations": [{"relation_id": row["id"]} for row in packet["relations"]]})
    return httpx.Response(200, json={"status": "completed", "output": [{"type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": content}]}]})


def test_full_inventory_pages_reach_after_old_caps_with_complete_groups(monkeypatch, image):
    original = _inventory(); previous = copy.deepcopy(original); packets = []
    def handler(request):
        packet = _packet(request); packets.append(packet)
        assert "GT_MUST_NOT_BE_SENT" not in request.content.decode()
        return _response(packet)
    _patch(monkeypatch, handler)
    receipt = BindingProvider(Settings(api_key="test", wire_api="responses")).select(image, image, original)
    assert receipt["network_requests"] == 3
    assert receipt["status"] == "succeeded" and receipt["coverage_complete"]
    assert receipt["all_pages_http_success"] and receipt["all_pages_schema_success"]
    assert len(receipt["bindings"]) == 35 and len(receipt["relations"]) == 35
    assert receipt["inventory_coverage"]["record_ids_not_sent"] == []
    assert receipt["inventory_coverage"]["relation_ids_not_sent"] == []
    for packet in packets:
        assert len(packet["records"]) <= 16 and len(packet["candidates"]) <= 32 and len(packet["relations"]) <= 16
        for record in packet["records"]:
            assert sum(row["record_id"] == record["id"] for row in packet["candidates"]) == 2
    assert len(validate_receipt_selection(receipt)["bindings"]) == 35
    assert original == previous


def test_verified_page_union_may_exceed_single_page_character_limit(monkeypatch, image):
    inventory = _inventory(count=64, groups=1, relations=64)
    record_ids = {row["id"]: row["id"] + "x"*50 for row in inventory["all_records"]}
    for row in inventory["all_records"]:
        row["id"] = record_ids[row["id"]]
        row["text"] = "R5" + " "*158
    for row in inventory["all_candidates"]:
        row["record_id"] = record_ids[row["record_id"]]
        row["id"] += "y"*50
    for row in inventory["relations"]:
        row["id"] += "z"*50
    _patch(monkeypatch, lambda request: _response(_packet(request)))
    receipt = BindingProvider(Settings(api_key="test", wire_api="responses")).select(image, image, inventory)
    assert receipt["network_requests"] == 4
    assert len(json.dumps({"bindings":receipt["bindings"],"relations":receipt["relations"]})) > 24000
    result = validate_receipt_selection(receipt)
    assert len(result["bindings"]) == 64 and len(result["relations"]) == 64
    assert receipt["inventory_coverage"]["relation_ids_not_sent"] == []


def test_failure_page_is_sent_but_never_inferred_as_model_abstention(monkeypatch, image):
    calls = []
    def handler(request):
        packet = _packet(request); calls.append(packet)
        return httpx.Response(503, text="untrusted private error") if len(calls) == 1 else _response(packet)
    _patch(monkeypatch, handler)
    receipt = BindingProvider(Settings(api_key="test", wire_api="responses")).select(image, image, _inventory(20, 1, 20))
    assert receipt["status"] == "partial" and not receipt["http_success"]
    assert receipt["schema_success"] and receipt["selection_subset_verified"]
    assert not receipt["all_pages_succeeded"] and not receipt["coverage_complete"]
    assert len(receipt["sent_record_ids"]) == 20
    assert receipt["input_record_ids"] == [f"r{i:03d}" for i in range(16, 20)]
    assert receipt["pages"][0]["http_status"] == 503
    assert "untrusted private error" not in json.dumps(receipt)
    assert len(validate_receipt_selection(receipt)["bindings"]) == 4


def test_single_large_record_group_is_excluded_whole_not_cherry_picked(monkeypatch, image):
    inventory = _inventory(1, 33, 0)
    extra = _inventory(2, 1, 0)
    inventory["all_records"].append(extra["all_records"][1])
    inventory["all_candidates"].append(extra["all_candidates"][1])
    packets = []
    _patch(monkeypatch, lambda request: (packets.append(_packet(request)) or _response(packets[-1])))
    receipt = BindingProvider(Settings(api_key="test", wire_api="responses")).select(image, image, inventory)
    assert receipt["input_record_ids"] == ["r001"]
    assert len(packets[0]["candidates"]) == 1
    assert receipt["status"] == "partial" and not receipt["coverage_complete"]
    assert {"record_id": "r000", "reason": "complete_candidate_group_exceeds_page_limit"} in receipt["inventory_coverage"]["record_exclusions"]


def test_page_count_cap_keeps_remaining_obligations_explicit(monkeypatch, image):
    _patch(monkeypatch, lambda request: _response(_packet(request)))
    receipt = BindingProvider(Settings(api_key="test", wire_api="responses")).select(image, image, _inventory(75, 1, 75))
    assert receipt["network_requests"] == MAX_BINDING_PAGES == 4
    assert len(receipt["sent_record_ids"]) == len(receipt["sent_relation_ids"]) == 64
    assert receipt["inventory_coverage"]["record_ids_not_sent"] == [f"r{i:03d}" for i in range(64, 75)]
    assert receipt["inventory_coverage"]["relation_ids_not_sent"] == [f"rel{i:03d}" for i in range(64, 75)]
    assert receipt["all_pages_succeeded"] and not receipt["coverage_complete"]
    assert receipt["status"] == "partial" and receipt["error_code"] == "partial_inventory_coverage"


def test_unknown_text_is_not_sent_and_missing_candidates_remain_unresolved(monkeypatch, image):
    inventory = _inventory(3, 1, 0)
    inventory["all_records"][0]["parsed"] = {"kind": "unknown"}
    inventory["all_candidates"] = inventory["all_candidates"][:2]
    _patch(monkeypatch, lambda request: _response(_packet(request)))
    receipt = BindingProvider(Settings(api_key="test", wire_api="responses")).select(image, image, inventory)
    coverage = receipt["inventory_coverage"]
    assert coverage["all_record_ids"] == ["r000", "r001", "r002"]
    assert coverage["source_eligible_record_ids"] == ["r001", "r002"]
    assert receipt["input_record_ids"] == ["r001"]
    assert {"record_id": "r000", "reason": "unsupported_or_unresolved_source_dimension"} in coverage["record_exclusions"]
    assert {"record_id": "r002", "reason": "no_source_candidate"} in coverage["record_exclusions"]
    assert not receipt["coverage_complete"]


def test_total_deadline_is_shared_across_all_pages(monkeypatch, image):
    calls = []
    async def handler(request):
        calls.append(_packet(request)); await asyncio.sleep(.15)
        return _response(calls[-1])
    _patch(monkeypatch, handler)
    started = time.monotonic()
    receipt = BindingProvider(Settings(api_key="test", wire_api="responses", api_timeout=.23)).select(image, image, _inventory(50, 1, 50))
    assert time.monotonic() - started < .65
    assert receipt["network_requests"] <= 2 < receipt["planned_pages"]
    assert receipt["total_timeout_seconds"] == .23
    assert not receipt["all_pages_succeeded"] and not receipt["coverage_complete"]
    assert receipt["pagination_stop_reason"] == "total_deadline_exhausted"
    assert receipt["pages"][-1]["error_code"] == "timeout"


def test_auth_failure_stops_without_spending_more_page_requests(monkeypatch, image):
    _patch(monkeypatch, lambda request: httpx.Response(401))
    receipt = BindingProvider(Settings(api_key="test", wire_api="responses")).select(image, image, _inventory())
    assert receipt["network_requests"] == 1
    assert receipt["pagination_stop_reason"] == "authentication"
    assert receipt["input_record_ids"] == []
    assert len(receipt["sent_record_ids"]) == 16
    assert not receipt["coverage_complete"]


def test_conflicting_aggregate_selections_are_removed_not_overwritten():
    pages = [{"schema_success": True, "selection_payload_verified": True,
              "bindings": [{"record_id": "r001", "candidate_id": "c001", "observed_text": "R5"}], "relations": []},
             {"schema_success": True, "selection_payload_verified": True,
              "bindings": [{"record_id": "r001", "candidate_id": "c002", "observed_text": "R5"}], "relations": []}]
    bindings, _, conflicts = _merge_page_selections(pages)
    assert bindings == [] and conflicts[0]["reason"] == "conflicting_page_selections"


@pytest.mark.parametrize("mutation", ["aggregate_row", "aggregate_input", "page_candidate_owner", "page_foreign_selection", "excess_pages"])
def test_receipt_validator_rejects_tampered_union_or_page_evidence(monkeypatch, image, mutation):
    _patch(monkeypatch, lambda request: _response(_packet(request)))
    receipt = BindingProvider(Settings(api_key="test", wire_api="responses")).select(image, image, _inventory(20, 1, 20))
    if mutation == "aggregate_row":
        receipt["bindings"].pop()
    elif mutation == "aggregate_input":
        receipt["input_record_ids"].append("r999")
    elif mutation == "page_candidate_owner":
        receipt["pages"][0]["input_candidate_record_ids"]["c000_0"] = "r002"
    elif mutation == "page_foreign_selection":
        receipt["pages"][0]["bindings"][0]["candidate_id"] = "foreign"
    else:
        receipt["pages"] *= 3
    with pytest.raises(_InspectionError):
        validate_receipt_selection(receipt)


@pytest.mark.parametrize("field", ["all_records", "all_candidates", "relations"])
def test_duplicate_inventory_ids_fail_before_network(monkeypatch, image, field):
    inventory = _inventory(); inventory[field].append(copy.deepcopy(inventory[field][0]))
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid inventory must not reach transport")
    monkeypatch.setattr("contour_agent.binding_provider.httpx.AsyncClient", unexpected)
    receipt = BindingProvider(Settings(api_key="test", wire_api="responses")).select(image, image, inventory)
    assert receipt["error_code"] == "invalid_inventory" and receipt["network_requests"] == 0


def test_complete_record_groups_never_exceed_packet_candidate_limit():
    pages, _ = _binding_pages(_inventory(30, 3, 30))
    assert [len(page["records"]) for page in pages] == [10, 10, 10]
    assert all(len(page["candidates"]) == 30 for page in pages)


def test_legacy_single_response_validator_preserves_original_limit():
    receipt = {"bindings": [{"record_id": f"r{i}", "candidate_id": f"c{i}"} for i in range(25)], "relations": []}
    with pytest.raises(_InspectionError):
        validate_receipt_selection(receipt)
