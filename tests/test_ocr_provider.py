import asyncio
import json
import httpx
import pytest
from contour_agent.config import Settings
from contour_agent.ocr import parse_dimension, bind_parameters
from contour_agent.provider import DimensionProvider, ProviderError, extract_json, validate_dimensions

@pytest.mark.parametrize("text,kind,value,upper,lower", [
    ("⌀274^{+5}_{0}", "diameter", 274, 5, 0),
    ("⌀710^{0}_{-10}", "diameter", 710, 0, -10),
    ("17.5±1", "length", 17.5, 1, -1),
    ("R62", "radius", 62, None, None), ("12°", "angle", 12, None, None),
    ("R80Ra6.3", "unknown", None, None, None), ("Ra6.3", "surface_finish", None, None, None),
    ("δ1", "symbol", None, None, None),
    ("17178", "length", 17178, None, None),
])
def test_parse(text, kind, value, upper, lower):
    result = parse_dimension(text)
    assert (result["kind"], result["nominal"], result["upper_deviation"], result["lower_deviation"]) == (kind, value, upper, lower)

def test_bad_digits_are_not_repaired_to_template():
    document = {"meta": {"original_size": {"width": 4170, "height": 2551}},
                "records": [{"text": "17178", "box": [[70, 1210], [94, 1210], [94, 1230], [70, 1230]]}]}
    schema = {"parameters": [{"id": "left_height", "default": 178, "min": 1, "max": 1000}]}
    rows, records = bind_parameters(document, schema, calibrated_layout=True)
    assert rows[0]["value"] is None
    assert rows[0]["suggested_value"] == 178
    assert rows[0]["needs_review"]

@pytest.mark.parametrize("text", ["{\"dimensions\": [", "{\"value\":NaN}", "[]"])
def test_json_rejects_incomplete_and_nonfinite(text):
    with pytest.raises(ProviderError):
        extract_json(text)

def test_provider_cannot_invent_source_ids():
    with pytest.raises(ProviderError):
        validate_dimensions({"dimensions": [{"id": "invented", "kind": "length", "nominal": 7}]}, {"r001"})

def test_compact_dimensions_are_expanded_and_validated():
    rows = validate_dimensions({"dimensions": [["r001", "diameter", 274, 5, 0]]}, {"r001"})
    assert rows == [{"id": "r001", "kind": "diameter", "nominal": 274, "upper_deviation": 5, "lower_deviation": 0}]
    with pytest.raises(ProviderError):
        validate_dimensions({"dimensions": [["r001", "radius", 5]]}, {"r001"})
    with pytest.raises(ProviderError):
        validate_dimensions({"dimensions": [[["r001"], "radius", 5, None, None]]}, {"r001"})

def patch_client(monkeypatch, handler):
    real = httpx.AsyncClient
    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)
    monkeypatch.setattr("contour_agent.provider.httpx.AsyncClient", factory)

def test_wire_payload_excludes_tools_images_and_secret_logs(monkeypatch):
    captured = []
    def handler(request):
        payload = json.loads(request.content)
        captured.append(payload)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"dimensions": [{"id": "r001", "kind": "radius", "nominal": 62, "upper_deviation": None, "lower_deviation": None}]})}}]})
    patch_client(monkeypatch, handler)
    settings = Settings(api_key="private-test-token")
    result = DimensionProvider(settings).normalize([{"id": "r001", "text": "R62", "gt": "must-not-be-sent", "image": "must-not-be-sent"}])
    assert result["status"] == "succeeded"
    assert result["http_status"] == 200 and result["finish_reason"] == "stop"
    assert "private-test-token" not in repr(result)
    assert "must-not-be-sent" not in repr(captured)
    assert set(captured[0]) == {"model", "messages", "temperature", "max_tokens"}

def test_http_error_body_is_not_logged(monkeypatch):
    patch_client(monkeypatch, lambda request: httpx.Response(401, json={"error": "secret-echoed-key"}))
    with pytest.raises(ProviderError) as error:
        DimensionProvider(Settings(api_key="private-test-token")).normalize([{"id": "a", "text": "R1"}])
    assert error.value.code == "authentication"
    assert "secret" not in str(error.value)

def test_truncated_output_rejected_even_if_json_parseable(monkeypatch):
    patch_client(monkeypatch, lambda request: httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": '{"dimensions":[]}'}}]}))
    with pytest.raises(ProviderError) as error:
        DimensionProvider(Settings(api_key="test")).normalize([])
    assert error.value.code == "truncated_output"


def test_http_200_schema_failure_preserves_transport_and_provider_metadata(monkeypatch):
    patch_client(monkeypatch, lambda request: httpx.Response(200, json={
        "choices": [{"finish_reason": "length", "message": {"content": ""}}]}))
    settings = Settings(api_key="test", model="test-model")
    with pytest.raises(ProviderError) as error:
        DimensionProvider(settings).normalize([{"id": "a", "text": "R1"}])
    assert error.value.status_code == 200
    assert error.value.http_success is True
    assert error.value.model == "test-model"
    assert error.value.protocol == "chat_completions-text-dimensions-compact-v3"
    assert isinstance(error.value.elapsed_seconds, float)

def test_total_deadline(monkeypatch):
    async def handler(request):
        await asyncio.sleep(3)
        return httpx.Response(200, json={})
    patch_client(monkeypatch, handler)
    with pytest.raises(ProviderError) as error:
        DimensionProvider(Settings(api_key="test", api_timeout=.001)).normalize([{"id": "a", "text": "R1"}])
    assert error.value.code == "timeout"

@pytest.mark.parametrize("url", ["https://name:secret@example.com/v1", "https://example.com/v1?key=secret", "https://example.com/v1#secret", "http://example.com/v1"])
def test_secret_bearing_or_insecure_urls_rejected(url):
    with pytest.raises(ValueError) as error:
        Settings(base_url=url)
    assert "secret" not in str(error.value)

def test_http_success_is_separate_from_schema_success(monkeypatch):
    patch_client(monkeypatch, lambda request: httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "not-json"}}]}))
    with pytest.raises(ProviderError) as error:
        DimensionProvider(Settings(api_key="test")).normalize([])
    assert error.value.http_success is True
    assert error.value.network_requests == 1
