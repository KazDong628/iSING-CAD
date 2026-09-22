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


def test_bounded_multimodal_payload_and_single_request(monkeypatch,image):
    options=[];payloads=[]
    def handler(request):
        payloads.append(json.loads(request.content));return response()
    patch(monkeypatch,handler,options)
    inventory={"units":"mm","records":[{"id":f"r{i:03d}","text":"R40","parsed":{"kind":"radius","nominal":40},"box":[]} for i in range(40)],
               "candidates":[{"id":f"c{i:03d}","record_id":f"r{i%24:03d}","kind":"radius","entities":["g000"],"nodes":[],"evidence":{}} for i in range(80)],
               "relations":[],"ground_truth_secret":"MUST NOT SEND"}
    result=BindingProvider(Settings(api_key="test-private-token")).select(image,image,inventory)
    assert result["http_success"] and result["schema_success"] and result["network_requests"]==1
    assert result["input_records"]==24 and result["input_candidates"]==48
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
                          for i,(x,y) in enumerate([(100,100),(700,100),(1400,100),(100,700),(700,700)])]}
    receipt=BindingProvider(Settings(api_key="test")).select(image,image,inventory)
    assert receipt["schema_success"] and receipt["input_image_count"]==6
    assert len(receipt["detail_panels"])==4
    assert all(len(p["input_image_sha256"])==64 and p["bytes"]<2_000_000 for p in receipt["detail_panels"])
    assert all(0<=p["source_box_px"][0]<p["source_box_px"][2]<=1600 for p in receipt["detail_panels"])
    assert sum(p["type"]=="image_url" for p in payloads[0]["messages"][1]["content"])==6
