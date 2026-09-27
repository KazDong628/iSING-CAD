"""Small explicit configuration surface. Never serialize credentials."""
from __future__ import annotations
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent

def load_local_env(path: Path | None = None) -> None:
    paths = [path] if path is not None else [ROOT / ".env.local", ROOT / ".env"]
    for candidate in paths:
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            if name.startswith(("CONTOUR_", "USTC_", "NINEE_", "H800_", "CAD_AGENT_", "CADRECON_")):
                os.environ.setdefault(name, value.strip().strip("\"'"))

@dataclass
class Settings:
    dataset_root: Path = field(default_factory=lambda: ROOT / "__dataset")
    runtime_root: Path = field(default_factory=lambda: ROOT / "runtime")
    base_url: str = field(default_factory=lambda: os.getenv("CONTOUR_API_BASE", "https://api.llm.ustc.edu.cn/v1").rstrip("/"))
    model: str = field(default_factory=lambda: os.getenv("CONTOUR_MODEL", "qwen-chat"))
    model_provider: str = field(default_factory=lambda: os.getenv("CONTOUR_MODEL_PROVIDER", "default"))
    wire_api: str = field(default_factory=lambda: os.getenv("CONTOUR_WIRE_API", "chat_completions").strip().lower())
    disable_response_storage: bool = field(default_factory=lambda: os.getenv("CONTOUR_DISABLE_RESPONSE_STORAGE", "true").lower() == "true")
    api_key: str = field(default_factory=lambda: os.getenv("CONTOUR_API_KEY") or os.getenv("USTC_API_KEY") or
                              os.getenv("CADRECON_API_KEY", ""), repr=False)
    api_timeout: float = field(default_factory=lambda: max(5., min(600., float(os.getenv("CONTOUR_API_TIMEOUT", "600")))))
    trust_env: bool = field(default_factory=lambda: os.getenv("CONTOUR_TRUST_ENV", "false").lower() == "true")
    allow_insecure_http: bool = False
    auth_scheme: str = "bearer"
    segmentation_checkpoint: str = field(default_factory=lambda: os.getenv("CONTOUR_SEGMENTATION_CHECKPOINT", ""))
    segmentation_manifest: str = field(default_factory=lambda: os.getenv("CONTOUR_SEGMENTATION_MANIFEST", ""))
    anthropic_thinking_mode: str = field(default_factory=lambda: os.getenv("CONTOUR_ANTHROPIC_THINKING_MODE", "provider_default").strip().lower())

    def __post_init__(self):
        parsed = urlparse(self.base_url)
        allowed_schemes = {"https", "http"} if self.allow_insecure_http else {"https"}
        if parsed.scheme not in allowed_schemes or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("API地址必须使用允许的传输协议，且不得包含凭据、查询参数或片段。")
        self.base_url = self.base_url.rstrip("/")
        if self.wire_api not in {"chat_completions","responses","anthropic_messages"}:
            raise ValueError("CONTOUR_WIRE_API must be chat_completions, responses or anthropic_messages")
        if self.auth_scheme not in {"bearer", "x-api-key"}:
            raise ValueError("API auth scheme must be bearer or x-api-key")
        if self.anthropic_thinking_mode not in {"provider_default", "disabled"}:
            raise ValueError("CONTOUR_ANTHROPIC_THINKING_MODE must be provider_default or disabled")

    def public(self) -> dict:
        profiles, default_id = provider_registry(self)
        active = profiles[default_id]
        return {"provider_configured": bool(active.api_key), "base_url": active.base_url,
                "model": active.model, "model_provider":active.model_provider,"wire_api":active.wire_api,
                "anthropic_thinking_mode": self.anthropic_thinking_mode,
                "anthropic_thinking_mode_requested": self.anthropic_thinking_mode if active.wire_api == "anthropic_messages" else None,
                "default_provider_id": default_id,
                "providers": [profile.public(default=profile_id == default_id, anthropic_thinking_mode=self.anthropic_thinking_mode)
                              for profile_id, profile in profiles.items()],
                "response_storage_disabled":self.disable_response_storage,
                "timeout_seconds": self.api_timeout,
                "tls_verification": True, "trust_environment_proxy": self.trust_env,
                "segmentation_available": bool(self.segmentation_checkpoint and Path(self.segmentation_checkpoint).is_file()),
                "segmentation_status": "experimental_supervised_model"}


@dataclass(frozen=True)
class ProviderProfile:
    id: str
    name: str
    model_provider: str
    base_url: str
    model: str
    wire_api: str
    api_key: str = field(default="", repr=False)
    allow_insecure_http: bool = False
    auth_scheme: str = "bearer"

    def settings(self, base: Settings) -> Settings:
        return replace(base, base_url=self.base_url, model=self.model, model_provider=self.model_provider,
                       wire_api=self.wire_api, api_key=self.api_key,
                       allow_insecure_http=self.allow_insecure_http, auth_scheme=self.auth_scheme)

    def public(self, *, default=False, anthropic_thinking_mode="provider_default") -> dict:
        return {"id": self.id, "name": self.name, "model_provider": self.model_provider,
                "model": self.model, "wire_api": self.wire_api,
                "configured": bool(self.api_key), "default": bool(default),
                "auth_scheme": self.auth_scheme,
                "anthropic_thinking_mode_requested": anthropic_thinking_mode if self.wire_api == "anthropic_messages" else None,
                "transport_security": "http" if self.allow_insecure_http else "https"}


def _first_secret(*names: str) -> str:
    return next((os.getenv(name, "") for name in names if os.getenv(name)), "")


def provider_registry(base: Settings) -> tuple[dict[str, ProviderProfile], str]:
    """Return the allow-listed provider profiles without serializing secrets."""
    templates = [
        ProviderProfile("ustc-qwen-chat", "中科大 · qwen-chat", "USTC", "https://api.llm.ustc.edu.cn/v1",
                        "qwen-chat", "chat_completions", _first_secret("USTC_API_KEY", "CADRECON_API_KEY")),
        ProviderProfile("ustc-glm-5.3-flash", "中科大 · glm-5.3-flash", "USTC", "https://api.llm.ustc.edu.cn/v1",
                        "glm-5.3-flash", "chat_completions", _first_secret("USTC_API_KEY", "CADRECON_API_KEY")),
        ProviderProfile("9ecode-gpt-5.6-sol", "玖亿AI · gpt-5.6-sol", "9eCode", "https://api.9e.lv/v1",
                        "gpt-5.6-sol", "responses", _first_secret("NINEE_API_KEY")),
        ProviderProfile("h800-qwen3.8-27b", "H800 · Qwen3.8-27B-FP8", "H800-vLLM",
                        os.getenv("CAD_AGENT_BASE_URL", "http://60.171.65.125:30539").rstrip("/"),
                        os.getenv("CAD_AGENT_MODEL", "Qwen3.8-27B-FP8"), "anthropic_messages",
                        _first_secret("H800_API_KEY", "CAD_AGENT_API_KEY"), True),
    ]
    signatures = {(row.base_url.rstrip("/"), row.model, row.wire_api): row.id for row in templates}
    active_signature = (base.base_url.rstrip("/"), base.model, base.wire_api)
    matched_id = signatures.get(active_signature)
    if matched_id:
        templates = [replace(row, api_key=base.api_key) if row.id == matched_id else row for row in templates]
    if not matched_id:
        matched_id = "server-default"
        templates.insert(0, ProviderProfile(matched_id, f"{base.model_provider} · {base.model}", base.model_provider,
                                            base.base_url, base.model, base.wire_api, base.api_key,
                                            base.allow_insecure_http))
    return {row.id: row for row in templates}, matched_id
