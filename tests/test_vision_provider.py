import asyncio
import base64
import hashlib
from io import BytesIO
import json
import time

import httpx
from PIL import Image
import pytest

from contour_agent.config import Settings
from contour_agent.vision_provider import VisionProvider, _InspectionError, _observation


@pytest.fixture
def drawing(tmp_path):
    path = tmp_path / "drawing.png"
    Image.new("RGB", (1800, 1400), "white").save(path)
    return path


def patch_client(monkeypatch, handler, calls=None):
    original = httpx.AsyncClient
    def factory(*args, **kwargs):
        if calls is not None:
            calls.append(kwargs.copy())
        kwargs["transport"] = httpx.MockTransport(handler)
        return original(*args, **kwargs)
    monkeypatch.setattr("contour_agent.vision_provider.httpx.AsyncClient", factory)


def response(value=None, *, reason="stop"):
    value = value or {"verdict": "match", "roi": [10, 10, 990, 990], "issues": [], "units": "unknown"}
    return httpx.Response(200, json={"choices": [{"finish_reason": reason, "message": {"content": json.dumps(value)}}]})


def test_image_payload_overlay_resize_and_credential_hygiene(monkeypatch, drawing):
    payloads, client_options = [], []
    def handler(request):
        payloads.append(json.loads(request.content))
        return response({"verdict": "mismatch", "roi": [10, 10, 990, 990], "issues": ["private-test-token"], "units": "unknown"})
    patch_client(monkeypatch, handler, client_options)
    receipt = VisionProvider(Settings(api_key="private-test-token")).inspect(drawing, [(0, 0), (1800, 0), (1800, 1400), (0, 1400)])
    assert receipt["schema_success"] and receipt["http_success"]
    assert receipt["network_requests"] == 1 and receipt["dimension_certified"] is False
    assert "private-test-token" not in json.dumps(receipt)
    assert receipt["issues"] == ["[REDACTED]"]
    assert client_options[0]["verify"] is True and client_options[0]["trust_env"] is False
    assert client_options[0]["follow_redirects"] is False
    wire = payloads[0]
    assert wire["max_tokens"] == 300 and "tools" not in wire
    parts = wire["messages"][1]["content"]
    image = Image.open(BytesIO(base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1])))
    assert image.format == "JPEG" and max(image.size) <= 1280
    assert image.getpixel((1, 1))[0] > image.getpixel((1, 1))[1] + 50


def test_no_candidate_can_never_receive_match(monkeypatch, drawing):
    patch_client(monkeypatch, lambda request: response())
    receipt = VisionProvider(Settings(api_key="test-key")).inspect(drawing)
    assert receipt["schema_success"] and receipt["verdict"] == "uncertain"
    assert receipt["overlay_sent"] is False


@pytest.mark.parametrize("status", [401, 429, 503])
def test_http_failures_are_single_call_and_do_not_echo_body(monkeypatch, drawing, status):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": "sensitive-echoed-value"})
    patch_client(monkeypatch, handler)
    receipt = VisionProvider(Settings(api_key="test-key")).inspect(drawing)
    assert len(requests) == receipt["network_requests"] == 1
    assert not receipt["http_success"] and not receipt["schema_success"]
    assert "sensitive" not in repr(receipt)
    assert "response_excerpt" not in receipt


@pytest.mark.parametrize("content,reason,error", [
    ("not json", "stop", "invalid_json"),
    (json.dumps({"verdict": "match", "roi": [700, 0, 200, 900], "issues": [], "units": "mm"}), "stop", "schema_mismatch"),
    (json.dumps({"verdict": "match", "roi": [0, 0, 900, 900], "issues": [], "units": "mm"}), "length", "truncated_output"),
])
def test_http_success_is_not_schema_success(monkeypatch, drawing, content, reason, error):
    patch_client(monkeypatch, lambda request: httpx.Response(200, json={"choices": [{"finish_reason": reason, "message": {"content": content}}]}))
    receipt = VisionProvider(Settings(api_key="test-key")).inspect(drawing)
    assert receipt["http_success"] is True and receipt["schema_success"] is False
    assert receipt["error_code"] == error
    assert receipt["response_text_sha256"] == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert receipt["response_text_chars"] == len(content)
    assert receipt["response_text_source"] == "message.content"
    assert receipt["response_excerpt"] == content


def test_real_total_deadline_and_no_retry(monkeypatch, drawing):
    async def handler(request):
        await asyncio.sleep(1)
        return response()
    patch_client(monkeypatch, handler)
    start = time.monotonic()
    receipt = VisionProvider(Settings(api_key="test-key", api_timeout=.05)).inspect(drawing)
    assert time.monotonic() - start < .5
    assert receipt["network_requests"] <= 1 and receipt["error_code"] == "timeout"


def test_missing_key_and_invalid_candidate_never_call(monkeypatch, drawing):
    def forbidden(request):
        pytest.fail("unexpected network request")
    patch_client(monkeypatch, forbidden)
    assert VisionProvider(Settings(api_key="")).inspect(drawing)["network_requests"] == 0
    result = VisionProvider(Settings(api_key="test-key")).inspect(drawing, [(0, 0), (9000, 0), (1, 1)])
    assert result["error_code"] == "invalid_candidate_boundary" and result["network_requests"] == 0


OBSERVATION = {"verdict": "match", "roi": [10, 10, 990, 990], "issues": [], "units": "unknown"}
OBSERVATION_JSON = json.dumps(OBSERVATION)


@pytest.mark.parametrize("content", [
    OBSERVATION_JSON,
    "```json\n" + OBSERVATION_JSON + "\n```",
    "以下是图像复核结果：\n" + OBSERVATION_JSON + "\n仅表示视觉一致。",
    "The result follows.\n```JSON\n" + OBSERVATION_JSON + "\n```\nNo dimensions certified.",
    json.dumps({**OBSERVATION, "issues": ['A brace { or } in a quoted issue is text, not another object.']}),
])
def test_unique_complete_json_object_accepts_prose_and_markdown(content):
    result = _observation(content, overlay=True)
    assert result["verdict"] == "match" and result["roi"] == OBSERVATION["roi"]
    assert set(result) == set(OBSERVATION)


@pytest.mark.parametrize("content", [
    OBSERVATION_JSON + "\n" + OBSERVATION_JSON,
    "```json\n" + OBSERVATION_JSON + "\n```\n```json\n" + OBSERVATION_JSON + "\n```",
    OBSERVATION_JSON + '\n{"verdict":',
    '{"unfinished": ' + OBSERVATION_JSON,
    OBSERVATION_JSON[:-1],
    "```json\n" + OBSERVATION_JSON,
    "Result:\n[" + OBSERVATION_JSON + "]",
    "[" + OBSERVATION_JSON + "]",
    OBSERVATION_JSON[:-1] + ', "verdict":"mismatch"}',
    OBSERVATION_JSON.replace('"unknown"', 'NaN'),
    json.dumps({key: value for key, value in OBSERVATION.items() if key != "units"}),
])
def test_does_not_choose_between_objects_repair_truncation_or_invent_fields(content):
    with pytest.raises(_InspectionError):
        _observation(content, overlay=True)


def test_success_receipt_hashes_original_prose_without_saving_excerpt(monkeypatch, drawing):
    content = "视觉复核结果：\n```json\n" + OBSERVATION_JSON + "\n```"
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]})
    patch_client(monkeypatch, handler)
    receipt = VisionProvider(Settings(api_key="test-key")).inspect(drawing, [(0, 0), (100, 0), (100, 100)])
    assert receipt["schema_success"] and receipt["verdict"] == "match"
    assert receipt["response_text_sha256"] == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert receipt["response_text_chars"] == len(content)
    assert "response_excerpt" not in receipt
    assert len(requests) == receipt["network_requests"] == 1


def test_failed_excerpt_redacts_before_length_limit(monkeypatch, drawing):
    key = "private-credential-" + "x" * 100
    content = "sk-unrelated-secret Bearer arbitrary.access.token " + "a" * 1110 + key + " END" * 100
    patch_client(monkeypatch, lambda request: httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]}))
    receipt = VisionProvider(Settings(api_key=key)).inspect(drawing)
    assert receipt["http_success"] and not receipt["schema_success"]
    assert receipt["error_code"] == "invalid_json"
    assert receipt["response_text_sha256"] == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert receipt["response_text_chars"] == len(content)
    excerpt = receipt["response_excerpt"]
    assert len(excerpt) <= 1200 and excerpt.count("[REDACTED]") == 3
    assert "private-credential" not in excerpt and "unrelated-secret" not in excerpt
    assert "arbitrary.access.token" not in excerpt


def test_invalid_envelope_uses_sanitized_body_diagnostic(monkeypatch, drawing):
    body = 'unexpected envelope: sk-echoed-key Bearer other.secret'
    patch_client(monkeypatch, lambda request: httpx.Response(200, text=body))
    receipt = VisionProvider(Settings(api_key="test-key")).inspect(drawing)
    assert receipt["error_code"] == "invalid_envelope"
    assert receipt["response_text_source"] == "http_body"
    assert receipt["response_text_chars"] == len(body)
    assert receipt["response_text_sha256"] == hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert receipt["response_excerpt"] == 'unexpected envelope: [REDACTED] Bearer [REDACTED]'
