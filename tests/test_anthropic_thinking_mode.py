"""Explicit request policy never asserts provider-side thinking was disabled."""
import json

import pytest

from contour_agent.api_wire import prepare_request
from contour_agent.config import Settings
from scripts.run_oracle_mask_reconstruction import _providers, run_oracle_mask_case


def test_settings_default_environment_override_and_public_declaration(monkeypatch):
    monkeypatch.delenv("CONTOUR_ANTHROPIC_THINKING_MODE", raising=False)
    assert Settings().anthropic_thinking_mode == "provider_default"
    monkeypatch.setenv("CONTOUR_ANTHROPIC_THINKING_MODE", "disabled")
    settings = Settings(wire_api="anthropic_messages", api_key="private-test-key")
    assert settings.anthropic_thinking_mode == "disabled"
    public = settings.public()
    assert public["anthropic_thinking_mode_requested"] == "disabled"
    assert "private-test-key" not in json.dumps(public)
    assert not any("effective" in key for key in public)
    for profile in public["providers"]:
        assert profile["anthropic_thinking_mode_requested"] == (
            "disabled" if profile["wire_api"] == "anthropic_messages" else None)


@pytest.mark.parametrize("mode", ["enabled", "auto", "", None])
def test_settings_reject_invalid_thinking_modes(mode):
    with pytest.raises(ValueError, match="CONTOUR_ANTHROPIC_THINKING_MODE"):
        Settings(anthropic_thinking_mode=mode)


def test_settings_reject_invalid_environment_mode(monkeypatch):
    monkeypatch.setenv("CONTOUR_ANTHROPIC_THINKING_MODE", "adaptive")
    with pytest.raises(ValueError, match="CONTOUR_ANTHROPIC_THINKING_MODE"):
        Settings()


@pytest.mark.parametrize("wire", ["chat_completions", "responses", "anthropic_messages"])
@pytest.mark.parametrize("mode", ["provider_default", "disabled"])
def test_thinking_control_is_explicit_and_protocol_isolated(wire, mode):
    settings = Settings(wire_api=wire, anthropic_thinking_mode=mode)
    original = {"model": "test-model", "max_tokens": 1000, "messages": [{"role": "user", "content": "inspect"}]}
    _, payload = prepare_request(settings, original)
    assert payload.get("thinking") == ({"type": "disabled"}
                                        if wire == "anthropic_messages" and mode == "disabled" else None)
    assert "thinking" not in original
    assert payload.get("max_tokens", payload.get("max_output_tokens")) == 1000


@pytest.fixture
def provider_environment(monkeypatch):
    # Do not load local credentials or make network requests in these tests.
    monkeypatch.setattr("contour_agent.config.load_local_env", lambda: None)
    monkeypatch.setenv("H800_API_KEY", "test-h800-key")
    monkeypatch.setenv("USTC_API_KEY", "test-ustc-key")
    monkeypatch.setenv("NINEE_API_KEY", "test-nine-key")
    monkeypatch.setenv("CONTOUR_ANTHROPIC_THINKING_MODE", "provider_default")
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: pytest.fail("must reject or configure before any request"))


@pytest.mark.parametrize("profile", ["ustc-qwen-chat", "9ecode-gpt-5.6-sol"])
def test_runner_rejects_explicit_thinking_mode_for_other_protocols(provider_environment, profile):
    with pytest.raises(ValueError, match="requires an Anthropic provider"):
        _providers(True, profile, anthropic_thinking="disabled")


def test_runner_rejects_explicit_thinking_mode_offline_before_creating_run(tmp_path):
    with pytest.raises(ValueError, match="requires --online"):
        _providers(False, None, anthropic_thinking="disabled")
    run = tmp_path / "offline"
    with pytest.raises(ValueError, match="requires --online"):
        run_oracle_mask_case("unused", run, anthropic_thinking="disabled")
    assert not run.exists()


@pytest.mark.parametrize("mode", ["enabled", "provider_default", ""])
def test_runner_rejects_unsupported_explicit_override(tmp_path, mode):
    with pytest.raises(ValueError, match="accepts only disabled"):
        _providers(True, None, anthropic_thinking=mode)
    run = tmp_path / "invalid"
    with pytest.raises(ValueError, match="accepts only disabled"):
        run_oracle_mask_case("unused", run, online=True, anthropic_thinking=mode)
    assert not run.exists()


def test_runner_default_unchanged_explicit_override_audited_and_does_not_mutate_environment(provider_environment):
    default_bundle, default_receipt = _providers(True, "h800-qwen3.8-27b")
    assert default_receipt["anthropic_thinking_mode_requested"] == "provider_default"
    bundle, receipt = _providers(True, "h800-qwen3.8-27b", anthropic_thinking="disabled")
    assert receipt["anthropic_thinking_mode_requested"] == "disabled"
    assert receipt["timeout_seconds"] == default_receipt["timeout_seconds"]
    assert all(provider.settings.anthropic_thinking_mode == "disabled" for provider in bundle.values())
    assert all(provider.settings.anthropic_thinking_mode == "provider_default" for provider in default_bundle.values())
    assert Settings().anthropic_thinking_mode == "provider_default"
    assert "api_key" not in json.dumps(receipt) and "test-h800-key" not in json.dumps(receipt)
