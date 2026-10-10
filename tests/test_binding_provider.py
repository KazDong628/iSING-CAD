import asyncio
import json
import time

import httpx
from PIL import Image
import pytest

from contour_agent.binding_provider import BindingProvider, validate_selection
from contour_agent.config import Settings
from contour_agent.vision_provider import _InspectionError


@pytest.fixture
def image(tmp_path):
    path=tmp_path/"source.png"
    Image.new("RGB",(1600,900),"white").save(path)
    return path


def patch(monkeypatch,handler,options=None):
    original=httpx.AsyncClient
    def factory(*args,**kwargs):
        if options is not None:options.append(kwargs.copy())
        return original(*args,**kwargs,transport=httpx.MockTransport(handler))
    monkeypatch.setattr("contour_agent.binding_provider.httpx.AsyncClient",factory)


def response(content=None,reason="stop"):
    return httpx.Response(200,json={"choices":[{"finish_reason":reason,"message":{"content":content if content is not None else '{"bindings":[],"relations":[]}'}}]})


def test_bounded_multimodal_payload_pages_complete_record_groups(monkeypatch,image):
    options=[];payloads=[]
    def handler(request):
        payloads.append(json.loads(request.content));return response()
    patch(monkeypatch,handler,options)
    inventory={"units":"mm","records":[{"id":f"r{i:03d}","text":"R40","parsed":{"kind":"radius","nominal":40},"box":[]} for i in range(40)],
               "candidates":[{"id":f"c{i:03d}","record_id":f"r{i%24:03d}","kind":"radius","entities":["g000"],"nodes":[],"evidence":{}} for i in range(80)],
               "relations":[],"ground_truth_secret":"MUST NOT SEND"}
    result=BindingProvider(Settings(api_key="test-private-token")).select(image,image,inventory)
    assert result["http_success"] and result["schema_success"] and result["network_requests"]==3
    assert result["input_records"]==24 and result["input_candidates"]==80
    assert result["inventory_coverage"]["record_ids_not_sent"] == [f"r{i:03d}" for i in range(24,40)]
    wire=payloads[0];parts=wire["messages"][1]["content"]
    assert wire["max_tokens"]==2200 and sum(p["type"]=="image_url" for p in parts)==2
    assert "MUST NOT SEND" not in json.dumps(wire) and "test-private-token" not in json.dumps(result)
    assert options[0]["verify"] is True and options[0]["follow_redirects"] is False and options[0]["trust_env"] is False
    assert "response_text_sha256" in result and "response_excerpt" not in result


@pytest.mark.parametrize("content",[
    '{"bindings":[],"relations":[]}',
    'Result: ```json\n{"bindings":[{"record_id":"r001","candidate_id":"c005"}],"relations":[{"relation_id":"rel001"}]}\n```',
])
def test_valid_selection_can_abstain_or_select_only_ids(content):
    assert set(validate_selection(content))=={"bindings","relations"}


def test_visible_text_is_preserved_but_not_interpreted_as_new_geometry():
    value=validate_selection('{"bindings":[{"record_id":"r032","candidate_id":"c009","observed_text":"Δ1"}],"relations":[]}')
    assert value["bindings"][0]["observed_text"]=="Δ1"
    with pytest.raises(_InspectionError):
        validate_selection('{"bindings":[{"record_id":"r032","candidate_id":"c009","observed_text":41}],"relations":[]}')


@pytest.mark.parametrize("content",[
    '{"bindings":[{"record_id":"r000","candidate_id":"c000","value":40}],"relations":[]}',
    '{"bindings":[],"relations":[],"points":[[0,0],[1,1]]}',
    '{"bindings":[],"relations":[]} {"bindings":[],"relations":[]}',
    '{"bindings":[{"record_id":null,"candidate_id":"c000"}],"relations":[]}',
    '{"bindings":[],"relations":[]',
])
def test_output_schema_rejects_numbers_points_multiple_objects_and_incomplete(content):
    with pytest.raises(_InspectionError):validate_selection(content)


def test_schema_failure_keeps_http_success_and_redacts_excerpt(monkeypatch,image):
    patch(monkeypatch,lambda request:response("invalid secret-test-token sk-echoed Bearer echoed.token"))
    result=BindingProvider(Settings(api_key="secret-test-token")).select(image,image,{})
    assert result["http_success"] and not result["schema_success"] and result["error_code"]=="invalid_json"
    assert "secret-test-token" not in json.dumps(result) and "echoed.token" not in result["response_excerpt"]
    assert result["response_excerpt"].count("[REDACTED]")==3


@pytest.mark.parametrize("status",[401,429,503])
def test_transport_status_never_retries_or_retains_raw_error(monkeypatch,image,status):
    calls=[]
    def handler(request):calls.append(request);return httpx.Response(status,text="secret echo")
    patch(monkeypatch,handler)
    result=BindingProvider(Settings(api_key="test")).select(image,image,{})
    assert len(calls)==result["network_requests"]==1 and not result["http_success"]
    assert "secret echo" not in json.dumps(result)


def test_total_deadline_and_truncation_are_not_repaired(monkeypatch,image):
    async def handler(request):await asyncio.sleep(.7);return response()
    patch(monkeypatch,handler)
    start=time.monotonic()
    result=BindingProvider(Settings(api_key="test",api_timeout=.06)).select(image,image,{})
    assert time.monotonic()-start<.5 and result["error_code"]=="timeout" and result["network_requests"]<=1


def test_finish_length_is_rejected_even_for_complete_json(monkeypatch,image):
    patch(monkeypatch,lambda request:response(reason="length"))
    result=BindingProvider(Settings(api_key="test")).select(image,image,{})
    assert result["http_success"] and result["error_code"]=="truncated_output" and not result["schema_success"]


def test_source_detail_panels_are_bounded_and_audited(monkeypatch,image):
    payloads=[]
    patch(monkeypatch,lambda request:(payloads.append(json.loads(request.content)) or response()))
    inventory={"records":[{"id":f"r{i:03d}","text":"R3","box":[[x,y],[x+30,y+30]],"parsed":{"kind":"radius","nominal":3}}
                          for i,(x,y) in enumerate([(100,100),(700,100),(1400,100),(100,700),(700,700)])],
               "candidates":[{"id":f"c{i:03d}","record_id":f"r{i:03d}","kind":"radius","entities":["g000"],"nodes":[]}
                             for i in range(5)]}
    receipt=BindingProvider(Settings(api_key="test")).select(image,image,inventory)
    assert receipt["schema_success"] and receipt["input_image_count"]==6
    assert len(receipt["detail_panels"])==4
    assert all(len(p["input_image_sha256"])==64 and p["bytes"]<2_000_000 for p in receipt["detail_panels"])
    assert all(0<=p["source_box_px"][0]<p["source_box_px"][2]<=1600 for p in receipt["detail_panels"])
    assert sum(p["type"]=="image_url" for p in payloads[0]["messages"][1]["content"])==6


def _large_inventory():
    return {"units": "mm",
            "records": [{"id": f"r{i:03d}", "text": "R40", "box": [],
                         "parsed": {"kind": "radius", "nominal": 40.}} for i in range(30)],
            "candidates": [{"id": f"c{i:03d}", "record_id": f"r{i//3:03d}",
                            "kind": "radius", "entities": ["g000"], "nodes": [], "evidence": {}}
                           for i in range(90)],
            "relations": [{"id": f"rel{i:03d}", "type": "horizontal", "entities": ["g000"], "nodes": []}
                          for i in range(30)]}


def _wire_response(wire_api, selection):
    content = json.dumps(selection)
    if wire_api == "responses":
        return httpx.Response(200, json={"status": "completed", "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content}]}]})
    return response(content)


@pytest.mark.parametrize("wire_api,limits", [("responses", (16, 32, 16)), ("chat_completions", (16, 32, 16))])
def test_receipt_ids_equal_actual_bounded_packet(monkeypatch, image, wire_api, limits):
    packets = []

    def handler(request):
        wire = json.loads(request.content)
        parts = wire["input"][0]["content"] if wire_api == "responses" else wire["messages"][1]["content"]
        packet = json.loads(parts[0]["text"])
        packets.append(packet)
        return _wire_response(wire_api, {"bindings": [{"record_id": packet["records"][0]["id"], "candidate_id": packet["candidates"][0]["id"],
                                                      "observed_text": "R40"}],
                                         "relations": [{"relation_id": packet["relations"][0]["id"]}] if packet["relations"] else []})

    patch(monkeypatch, handler)
    result = BindingProvider(Settings(api_key="test", wire_api=wire_api)).select(image, image, _large_inventory())
    assert len(packets) == result["network_requests"] == 3
    assert result["schema_success"] and result["semantic_success"] and result["selection_payload_verified"]
    for name, field, limit in zip(("records", "candidates", "relations"),
                                  ("input_record_ids", "input_candidate_ids", "input_relation_ids"), limits):
        assert result[field] == [row["id"] for packet in packets for row in packet[name]]
        assert all(len(packet[name]) <= limit for packet in packets)


@pytest.mark.parametrize("selection,error", [
    ({"bindings": [{"record_id": "r016", "candidate_id": "c048"}], "relations": []}, "record_not_sent"),
    ({"bindings": [{"record_id": "r000", "candidate_id": "c039"}], "relations": []}, "candidate_not_sent"),
    ({"bindings": [{"record_id": "r001", "candidate_id": "c000"}], "relations": []}, "record_candidate_mismatch"),
    ({"bindings": [{"record_id": "r000", "candidate_id": "c000"},
                   {"record_id": "r000", "candidate_id": "c001"}], "relations": []}, "duplicate_record_selection"),
    ({"bindings": [], "relations": [{"relation_id": "rel016"}]}, "relation_not_sent"),
    ({"bindings": [], "relations": [{"relation_id": "rel000"}, {"relation_id": "rel000"}]}, "duplicate_relation_selection"),
])
def test_semantic_validation_rejects_unsent_or_inconsistent_ids(monkeypatch, image, selection, error):
    calls = []
    patch(monkeypatch, lambda request: (calls.append(request) or _wire_response("responses", selection)))
    inventory = _large_inventory()
    inventory["records"] = inventory["records"][:10]
    inventory["candidates"] = inventory["candidates"][:30]
    inventory["relations"] = inventory["relations"][:16]
    result = BindingProvider(Settings(api_key="test", wire_api="responses")).select(image, image, inventory)
    assert len(calls) == result["network_requests"] == 1
    assert result["http_success"] and result["error_code"] == error
    assert not result["schema_success"] and not result["semantic_success"]
    assert not result["selection_payload_verified"]
    assert result["bindings"] == [] and result["relations"] == []
    assert result["input_record_ids"] == result["input_candidate_ids"] == []
    assert len(result["sent_record_ids"]) == 10 and len(result["sent_candidate_ids"]) == 30
