import copy
import hashlib
import json

import httpx
from PIL import Image
import pytest

from contour_agent.config import Settings
from contour_agent import radius_target_provider as candidate
from contour_agent.vision_provider import _InspectionError


def document(count=3):
    return {"records": [{"text": "R" + str(10+i), "box": [[40, 40], [80, 40], [80, 70], [40, 70]]}
                        for i in range(count)]}


def proposal(rid, *, bad=False):
    return {"record_id": rid, "tip_px": [130, 110], "shaft_px": [130, 110] if bad else [70, 75]}


def mixed():
    return {"proposals": [proposal("r000"), proposal("r001", bad=True), proposal("r002")], "unknown_record_ids": []}


def call_adapter(monkeypatch, tmp_path, value):
    image = tmp_path / "source.png"
    Image.new("RGB", (300, 200), "white").save(image)
    text = json.dumps(value)
    client = httpx.AsyncClient
    monkeypatch.setattr(candidate.httpx, "AsyncClient", lambda **kwargs: client(**kwargs, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": text}}]}))))
    return candidate.RadiusTargetProvider(Settings(api_key="NEVER-PUBLISH-TEST-SECRET")).locate(image, document()), text


def test_adapter_preserves_strict_other_records_after_one_bad_shaft(monkeypatch, tmp_path):
    receipt, text = call_adapter(monkeypatch, tmp_path, mixed())
    assert receipt["status"] == "partial"
    assert receipt["http_success"] is True and receipt["schema_success"] is False
    assert receipt["partial_schema_success"] is True and receipt["response_structure_valid"] is True
    assert [row["record_id"] for row in receipt["proposals"]] == ["r000", "r002"]
    assert receipt["unknown_record_ids"] == receipt["rejected_record_ids"] == ["r001"]
    assert receipt["response_text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert receipt["arrowheads_verified"] is False and receipt["dimensions_verified"] is False
    assert "NEVER-PUBLISH-TEST-SECRET" not in json.dumps(receipt)
    assert "response_excerpt" not in receipt
    assert "response_text" not in receipt
    # Retained subset passes the unmodified strict validator contract.
    strict = candidate.validate_radius_target_response(json.dumps({"proposals": receipt["proposals"],
        "unknown_record_ids": receipt["unknown_record_ids"]}), {"r000", "r001", "r002"}, (300, 200))
    assert strict["proposals"] == receipt["proposals"]


def test_default_validator_still_rejects_bad_shaft():
    with pytest.raises(_InspectionError):
        candidate.validate_radius_target_response(json.dumps(mixed()), {"r000", "r001", "r002"}, (300, 200))


@pytest.mark.parametrize("point", [[300, 110], [-1, 110], [True, 110], [12], "secret", {"x": 10}])
def test_invalid_pixel_is_unknown_not_relaxed(monkeypatch, tmp_path, point):
    value = mixed()
    value["proposals"][1]["tip_px"] = point
    receipt, _ = call_adapter(monkeypatch, tmp_path, value)
    assert receipt["schema_success"] is False
    assert receipt["partial_schema_success"] is True
    assert receipt["rejected_records"] == [{"record_id": "r001", "error_code": "invalid_source_pixel"}]
    assert {row["record_id"] for row in receipt["proposals"]} == {"r000", "r002"}


def test_bad_alternative_discards_all_alternatives_for_that_record(monkeypatch, tmp_path):
    value = mixed()
    value["proposals"].append({**proposal("r001"), "tip_px": [131, 110]})
    receipt, _ = call_adapter(monkeypatch, tmp_path, value)
    assert receipt["rejected_record_ids"] == ["r001"]
    assert not any(row["record_id"] == "r001" for row in receipt["proposals"])


@pytest.mark.parametrize("change", [
    lambda p: p.update(extra_verdict=True),
    lambda p: p["proposals"][2].update(record_id="r999"),
    lambda p: p["proposals"][2].update(radius=10),
    lambda p: p.update(unknown_record_ids=["r000"]),
    lambda p: p.update(unknown_record_ids=["r999"]),
    lambda p: p["proposals"].pop(),
    lambda p: p["proposals"].append(proposal("r000")),
    lambda p: p["proposals"].extend([{**proposal("r000"), "tip_px": [131, 110]}, {**proposal("r000"), "tip_px": [132, 110]}]),
    lambda p: p["proposals"].append(proposal("r001", bad=True)),
    lambda p: p["proposals"].append(None),
])
def test_global_structural_error_rejects_entire_batch_even_with_good_records(monkeypatch, tmp_path, change):
    value = mixed()
    change(value)
    receipt, text = call_adapter(monkeypatch, tmp_path, value)
    assert receipt["status"] == "failed"
    assert receipt["schema_success"] is False
    assert not receipt.get("partial_schema_success", False)
    assert receipt["proposals"] == []
    assert receipt["unknown_record_ids"] == ["r000", "r001", "r002"]
    assert receipt["response_text_sha256"] == hashlib.sha256(text.encode()).hexdigest()


def test_all_bad_records_do_not_claim_partial_success(monkeypatch, tmp_path):
    value = {"proposals": [proposal(f"r{i:03d}", bad=True) for i in range(3)], "unknown_record_ids": []}
    receipt, _ = call_adapter(monkeypatch, tmp_path, value)
    assert receipt["status"] == "failed"
    assert receipt["schema_success"] is False and receipt["partial_schema_success"] is False
    assert receipt["proposals"] == [] and receipt["unknown_record_ids"] == ["r000", "r001", "r002"]


def partial_receipt():
    return {"status": "partial", "http_success": True, "schema_success": False,
            "partial_schema_success": True, "response_structure_valid": True,
            "schema_rejection_scope": "record_pixel_semantics", "network_requests": 1,
            "proposals": [proposal("r000"), proposal("r002")], "unknown_record_ids": ["r001"],
            "model_unknown_record_ids": [], "rejected_record_ids": ["r001"],
            "rejected_records": [{"record_id": "r001", "error_code": "degenerate_shaft"}]}


def test_partial_then_timeout_preserves_validated_subset_and_full_denominator(monkeypatch, tmp_path):
    original = document()
    calls = []
    class Provider:
        settings = Settings(api_timeout=25)
        def locate(self, path, received, **kwargs):
            assert received == original
            calls.append(kwargs)
            if len(calls) == 1:
                return partial_receipt()
            return {"status": "failed", "http_success": False, "schema_success": False, "network_requests": 1,
                    "error_code": "timeout", "proposals": [], "unknown_record_ids": ["r001"]}
    def local(path, doc, baseline):
        verified = [f"r{i:03d}" for i, row in enumerate(doc["records"]) if row.get("source_arrow_proposals")]
        return {"status": "completed", "verified_record_ids": verified}
    monkeypatch.setattr(candidate, "_local_radius_inventory", local)
    enriched, receipt = candidate.locate_radius_targets("unused.png", copy.deepcopy(original), {}, tmp_path, Provider())
    assert calls == [{}, {"record_ids": ["r001"]}]
    assert receipt["status"] == "partial" and receipt["schema_success"] is False
    assert receipt["any_partial_schema_success"] is True and receipt["any_schema_success"] is False
    assert receipt["network_requests"] == 2
    assert receipt["recognized_radius_record_ids"] == ["r000", "r001", "r002"]
    assert receipt["locally_verified_record_ids"] == ["r000", "r002"]
    assert receipt["unknown_record_ids"] == receipt["unresolved_rejected_record_ids"] == ["r001"]
    assert receipt["verified_absent_arrow_records"] == [] and receipt["all_arrows_locally_verified"] is False
    assert enriched["records"][0]["source_arrow_proposals"] == [proposal("r000")]
    assert "source_arrow_proposals" not in enriched["records"][1]


def test_partial_flag_alone_cannot_admit_proposals(monkeypatch, tmp_path):
    class Provider:
        def locate(self, *args, **kwargs):
            value = partial_receipt()
            value.pop("response_structure_valid")
            return value
    seen = []
    def local(path, doc, baseline):
        seen.append(copy.deepcopy(doc))
        return {"status": "completed", "verified_record_ids": []}
    monkeypatch.setattr(candidate, "_local_radius_inventory", local)
    candidate.locate_radius_targets("unused.png", document(), {}, tmp_path, Provider())
    assert all(not row.get("source_arrow_proposals") for doc in seen for row in doc["records"])


def test_partial_proposals_still_require_local_original_ink_verification(monkeypatch, tmp_path):
    class Provider:
        def locate(self, *args, **kwargs):
            return partial_receipt()
    monkeypatch.setattr(candidate, "_local_radius_inventory", lambda *args: {"status": "completed", "verified_record_ids": []})
    _, receipt = candidate.locate_radius_targets("unused.png", document(), {}, tmp_path, Provider())
    assert receipt["locally_verified_record_ids"] == []
    assert receipt["unknown_record_ids"] == ["r000", "r001", "r002"]
    assert receipt["all_arrows_locally_verified"] is False and receipt["dimensions_verified"] is False


@pytest.mark.parametrize("token", ["1e999", "-1e999", "9" * 500])
def test_overflow_number_isolated_as_invalid_pixel(token):
    text = json.dumps(mixed()).replace('"tip_px": [130, 110], "shaft_px": [130, 110]',
        '"tip_px": [' + token + ', 110], "shaft_px": [130, 110]')
    value = candidate.validate_radius_target_response(text, {"r000", "r001", "r002"}, (300, 200), isolate_record_errors=True)
    assert value["rejected_records"] == [{"record_id": "r001", "error_code": "invalid_source_pixel"}]
    assert [row["record_id"] for row in value["proposals"]] == ["r000", "r002"]
    with pytest.raises(_InspectionError):
        candidate.validate_radius_target_response(text, {"r000", "r001", "r002"}, (300, 200))

@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_json_constants_remain_global_parse_failure(token):
    text = json.dumps(mixed()).replace('[130, 110]', '[' + token + ', 110]', 1)
    with pytest.raises(_InspectionError) as error:
        candidate.validate_radius_target_response(text, {"r000", "r001", "r002"}, (300, 200), isolate_record_errors=True)
    assert error.value.code == "invalid_json"
