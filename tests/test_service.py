import json
import threading
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from contour_agent.config import ROOT, Settings
from contour_agent.geometry import template_schema
from contour_agent.provider import ProviderError
from contour_agent.server import create_app
from contour_agent.service import AgentService

@pytest.fixture
def service(tmp_path):
    result = AgentService(Settings(runtime_root=tmp_path, api_key=""))
    yield result
    result.close()

def test_missing_key_retains_parameters_and_requires_review(service):
    job = service.create_case("solid-arrow-ping__img_000293", use_api=True, asynchronous=False)
    assert job["status"] == "needs_review"
    assert job["provider"]["code"] == "not_configured"
    assert any(row["value"] is not None for row in job["parameters"])
    assert not job["artifacts"]
    assert not job["automatic_completion"]
    assert next(p for p in job["parameters"] if p["id"] == "left_height")["value"] is None

def test_explicit_confirmation_produces_independently_readable_dxf(service):
    schema = template_schema()
    job = service.create_case(schema["calibration_case"], use_api=False, asynchronous=False)
    values = {p["id"]: p["default"] for p in schema["parameters"]}
    with pytest.raises(ValueError):
        service.confirm(job["id"], values, [], True, asynchronous=False)
    with pytest.raises(ValueError):
        service.confirm(job["id"], {**values, "left_height": float("nan")}, [a["id"] for a in schema["assumptions"]], True, asynchronous=False)
    result = service.confirm(job["id"], values, [a["id"] for a in schema["assumptions"]], True, asynchronous=False, actor="test_fixture")
    assert result["status"] == "completed"
    assert result["validation"]["passed"]
    assert not result["engineering_accepted"]
    assert not result["automatic_completion"]
    import ezdxf
    document = ezdxf.readfile(Path(result["artifact_directory"]) / "drawing.dxf")
    assert len(document.modelspace().query("LINE ARC")) == 21

def test_unknown_topology_does_not_run_api(service, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported case sent to API")
    monkeypatch.setattr(service.provider, "normalize", forbidden)
    case = next(c for c in service.catalog["cases"] if c.get("image") and c.get("ocr") and not c.get("supported_template"))
    job = service.create_case(case["id"], use_api=True, asynchronous=False)
    assert job["status"] == "unsupported"
    with pytest.raises(ValueError):
        service.create_case(case["id"], template_id=template_schema()["id"])

def test_timeout_fallback(service, monkeypatch):
    def fail(*args):
        raise ProviderError("timeout", "deadline")
    monkeypatch.setattr(service.provider, "normalize", fail)
    job = service.create_case(template_schema()["calibration_case"], use_api=True, asynchronous=False)
    assert job["status"] == "needs_review" and job["parameters"]
    assert job["provider"]["code"] == "timeout"

def test_cancel_discards_late_provider_result(service, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def slow(rows):
        entered.set()
        assert release.wait(5)
        raise ProviderError("timeout", "deadline")
    monkeypatch.setattr(service.provider, "normalize", slow)
    job = service.create_case(template_schema()["calibration_case"], use_api=True)
    assert entered.wait(5)
    service.cancel(job["id"])
    release.set()
    service.executor.shutdown(wait=True)
    assert service.store.get(job["id"])["status"] == "cancelled"


def test_delete_cancelled_job_cleans_artifacts_written_by_late_worker(service):
    import threading
    import uuid
    from datetime import datetime, timezone

    job_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    service.store.save({"id": job_id, "status": "cancelled", "created_at": now,
                        "updated_at": now, "events": [], "artifacts": {}})
    started, release = threading.Event(), threading.Event()
    target = service.settings.runtime_root / "jobs" / job_id / "automatic-001"

    def late_write():
        started.set()
        assert release.wait(5)
        target.mkdir(parents=True, exist_ok=True)
        (target / "late.json").write_text("{}", encoding="utf-8")

    future = service._submit(job_id, late_write)
    assert started.wait(5)
    service.delete_job(job_id)
    release.set()
    future.result(timeout=5)
    import time
    deadline = time.monotonic() + 2
    while (service.settings.runtime_root / "jobs" / job_id).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not (service.settings.runtime_root / "jobs" / job_id).exists()
    with pytest.raises(KeyError):
        service.store.get(job_id)

def test_http_metadata_and_origin_boundary(tmp_path):
    app = create_app(Settings(runtime_root=tmp_path, api_key="do-not-expose"))
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        config = client.get("/api/config")
        assert config.json()["provider_configured"]
        assert "do-not-expose" not in config.text
        assert client.post("/api/jobs", json={"case_id": "x"}, headers={"Origin": "https://external.invalid"}).status_code == 403
        assert client.get("/api/jobs/not-found/artifacts/.env").status_code == 404
        assert client.get("/api/jobs/missing").status_code == 404


def test_static_assets_have_explicit_browser_mime_types(tmp_path, monkeypatch):
    import mimetypes
    mimetypes.init()
    # Reproduce a host with incorrect extension associations without changing
    # its registry. Explicit response types must override all of these guesses.
    for extension in (".js", ".css", ".html"):
        monkeypatch.setitem(mimetypes.types_map, extension, "text/plain")
    app = create_app(Settings(runtime_root=tmp_path, api_key=""))
    with TestClient(app) as client:
        for route, expected in (("/static/app.js", "text/javascript"), ("/static/styles.css", "text/css"), ("/", "text/html")):
            response = client.get(route)
            assert response.status_code == 200
            assert response.headers["content-type"].split(";", 1)[0] == expected
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.content

def test_new_failed_revision_cannot_serve_old_artifact(service, monkeypatch):
    schema = template_schema()
    job = service.create_case(schema["calibration_case"], use_api=False, asynchronous=False)
    values = {p["id"]: p["default"] for p in schema["parameters"]}
    assumptions = [a["id"] for a in schema["assumptions"]]
    first = service.confirm(job["id"], values, assumptions, True, asynchronous=False)
    assert first["artifact_directory"]
    def invalid(*args):
        raise ValueError("new geometry rejected")
    monkeypatch.setattr("contour_agent.service.solve_profile", invalid)
    second = service.confirm(job["id"], {**values, "left_height": 179}, assumptions, True, asynchronous=False)
    assert second["status"] == "needs_review"
    assert "artifact_directory" not in second
    assert not second["artifacts"] and second["validation"] is None and second["geometry"] is None
    assert not service._persist_if_active(first)

def test_other_service_does_not_recover_active_job(service):
    job = service._new("test", template_schema()["id"], Path("unused"), Path("unused"), False)
    other = AgentService(service.settings)
    try:
        assert other.store.get(job["id"])["status"] == "queued"
    finally:
        other.close()

def test_runtime_has_exclusive_server_lease(tmp_path):
    from contour_agent.store import RuntimeLease
    lease = RuntimeLease(tmp_path)
    try:
        with pytest.raises(ValueError):
            RuntimeLease(tmp_path)
    finally:
        lease.close()
    again = RuntimeLease(tmp_path)
    again.close()


def test_feedback_evidence_walks_back_from_failed_revision(service, tmp_path):
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    original_id, failed_id = "1" * 32, "2" * 32
    original_output = tmp_path / "jobs" / original_id / "automatic-001"
    original_output.mkdir(parents=True)
    for name in ("topology-candidates.json", "topology-plan.json", "baseline-model.json"):
        (original_output / name).write_text("{}", encoding="utf-8")
    (original_output / "overlay.png").write_bytes(b"png")
    service.store.save({
        "id": original_id, "mode": "autonomous_image", "status": "completed",
        "created_at": now, "updated_at": now, "events": [], "artifacts": {},
        "artifact_directory": str(original_output), "parent_job_id": None,
    })
    failed_output = tmp_path / "jobs" / failed_id / "feedback-001"
    failed_output.mkdir(parents=True)
    (failed_output / "feedback-plan.json").write_text("{}", encoding="utf-8")
    failed = {
        "id": failed_id, "mode": "autonomous_revision", "status": "needs_review",
        "created_at": now, "updated_at": now, "events": [], "artifacts": {},
        "artifact_directory": str(failed_output), "parent_job_id": original_id,
    }
    service.store.save(failed)

    evidence_parent, evidence_path, selection_file, depth = service._resolve_feedback_evidence(failed)

    assert evidence_parent["id"] == original_id
    assert evidence_path == original_output.resolve()
    assert selection_file == "topology-plan.json"
    assert depth == 1
