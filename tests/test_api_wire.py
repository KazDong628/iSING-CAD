import json

from contour_agent.api_wire import extract_text, prepare_request, request_headers
from contour_agent.config import Settings, provider_registry


class Response:
    def __init__(self,value):self.value=value
    def json(self):return self.value


def test_responses_wire_converts_text_images_and_disables_storage():
    settings=Settings(base_url="https://api.example/v1",model="gpt-test",wire_api="responses",
                      disable_response_storage=True)
    endpoint,payload=prepare_request(settings,{"model":"gpt-test","temperature":0,"max_tokens":123,
        "messages":[{"role":"system","content":"rules"},{"role":"user","content":[
            {"type":"text","text":"inspect"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]})
    assert endpoint=="https://api.example/v1/responses"
    assert payload=={"model":"gpt-test","instructions":"rules","max_output_tokens":123,"store":False,
                     "input":[{"role":"user","content":[{"type":"input_text","text":"inspect"},
                         {"type":"input_image","image_url":"data:image/png;base64,AAAA"}]}]}
    assert "temperature" not in payload and "messages" not in payload


def test_responses_wire_extracts_only_assistant_output_text():
    settings=Settings(base_url="https://api.example/v1",wire_api="responses")
    content,source,reason,usage=extract_text(settings,Response({"status":"completed","output":[
        {"type":"reasoning","summary":[]},
        {"type":"message","role":"assistant","content":[{"type":"output_text","text":"{\"ok\":true}"}]}],
        "usage":{"output_tokens":7}}))
    assert json.loads(content)=={"ok":True}
    assert source=="responses.output_text" and reason=="stop" and usage["output_tokens"]==7


def test_chat_wire_remains_backward_compatible():
    settings=Settings(base_url="https://api.example/v1",wire_api="chat_completions")
    original={"model":"qwen","messages":[{"role":"user","content":"x"}],"max_tokens":20}
    endpoint,payload=prepare_request(settings,original)
    assert endpoint.endswith("/chat/completions") and payload==original and payload is not original
    text,source,reason,_=extract_text(settings,Response({"choices":[{"finish_reason":"stop","message":{"content":"ok"}}]}))
    assert (text,source,reason)==("ok","message.content","stop")


def test_anthropic_messages_wire_converts_system_text_and_base64_image():
    settings=Settings(base_url="http://inference.example:30539",model="vision-model",
                      wire_api="anthropic_messages",api_key="private",allow_insecure_http=True,
                      auth_scheme="x-api-key")
    endpoint,payload=prepare_request(settings,{"model":"vision-model","temperature":0,"max_tokens":321,
        "messages":[{"role":"system","content":"rules"},{"role":"user","content":[
            {"type":"text","text":"inspect"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]})
    assert endpoint=="http://inference.example:30539/v1/messages"
    assert payload["system"]=="rules" and payload["max_tokens"]==321
    assert payload["messages"][0]["content"][1]=={
        "type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}
    assert request_headers(settings)=={"x-api-key":"private","anthropic-version":"2023-06-01",
                                       "content-type":"application/json"}
    text,source,reason,usage=extract_text(settings,Response({"content":[{"type":"text","text":"ok"}],
        "stop_reason":"end_turn","usage":{"output_tokens":2}}))
    assert (text,source,reason,usage["output_tokens"])==("ok","anthropic.content.text","stop",2)


def test_anthropic_reasoning_only_length_response_reports_truncation():
    settings=Settings(base_url="http://inference.example:30539",wire_api="anthropic_messages",
                      allow_insecure_http=True)
    text,source,reason,usage=extract_text(settings,Response({
        "content":[{"type":"thinking","thinking":"still reasoning"}],
        "stop_reason":"max_tokens","usage":{"output_tokens":900}}))
    assert (text,source,reason,usage["output_tokens"])==("","anthropic.content.text","length",900)


def test_provider_registry_exposes_safe_profiles_and_never_keys(monkeypatch):
    monkeypatch.setenv("NINEE_API_KEY","secret-nine")
    monkeypatch.setenv("CAD_AGENT_API_KEY","secret-h800")
    monkeypatch.setenv("CADRECON_API_KEY","secret-ustc")
    settings=Settings(base_url="https://api.9e.lv/v1",model="gpt-5.6-sol",model_provider="9eCode",
                      wire_api="responses",api_key="active-secret")
    profiles,default_id=provider_registry(settings)
    assert default_id=="9ecode-gpt-5.6-sol"
    assert set(profiles)=={"ustc-qwen-chat","ustc-glm-5.3-flash","9ecode-gpt-5.6-sol","h800-qwen3.8-27b"}
    public=settings.public()
    serialized=json.dumps(public)
    assert all(secret not in serialized for secret in ("secret-nine","secret-h800","secret-ustc","active-secret"))
    assert all(row["configured"] for row in public["providers"])


def test_http_provider_requires_an_explicit_allow_flag():
    import pytest
    with pytest.raises(ValueError):
        Settings(base_url="http://inference.example:30539",api_key="test")
    assert Settings(base_url="http://inference.example:30539",api_key="test",
                    wire_api="anthropic_messages",allow_insecure_http=True).allow_insecure_http
