import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from contour_agent import server
from contour_agent.config import Settings


RUN_ID = "20260920T103926355535Z"
REPORT_NAME = RUN_ID + "_autonomous.json"
CASE_ID = "solid-arrow-ping__img_000293"
JOB_ID = "a" * 32
NAMES = ("overlay.png", "drawing.dxf", "model.json", "validation.json", "preview.svg", "dimension-evidence.json")


@pytest.fixture
def evaluation_client(tmp_path, monkeypatch):
    class FakeService:
        def __init__(self, settings, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(server, "AgentService", FakeService)
    root = tmp_path / "runtime"
    artifact_dir = root / "autonomous-qualification" / RUN_ID / "agent-runtime" / "jobs" / JOB_ID / "automatic-001"
    artifact_dir.mkdir(parents=True)
    for name in NAMES:
        (artifact_dir / name).write_bytes(("latest " + name).encode())
    report_path = root / "evaluations" / REPORT_NAME
    report_path.parent.mkdir()
    data = {"mode": "autonomous_image", "run_id": RUN_ID,
            "trials": [{"case_id": CASE_ID, "repeat": 1, "job_id": "b" * 32, "artifacts": {}},
                       {"case_id": CASE_ID, "repeat": 2, "job_id": JOB_ID,
                        "artifacts": {name: str(artifact_dir / name) for name in NAMES}}]}
    report_path.write_text(json.dumps(data), encoding="utf8")
    app = server.create_app(Settings(runtime_root=root, api_key=""))
    with TestClient(app) as client:
        yield client, root, report_path, data, artifact_dir


def route(name, *, report_name=REPORT_NAME, case_id=CASE_ID):
    return f"/api/evaluations/{report_name}/cases/{case_id}/artifacts/{name}"


def test_latest_trial_artifacts_are_read_only_and_have_explicit_types(evaluation_client):
    client, _, _, _, _ = evaluation_client
    expected = {"overlay.png": "image/png", "preview.svg": "image/svg+xml", "drawing.dxf": "application/dxf"}
    for name in NAMES:
        response = client.get(route(name))
        assert response.status_code == 200
        assert response.content == ("latest " + name).encode()
        assert response.headers["content-type"].split(";", 1)[0] == expected.get(name, "application/json")
        assert response.headers["x-content-type-options"] == "nosniff"
        if name not in {"overlay.png", "preview.svg"}:
            assert "attachment" in response.headers["content-disposition"]
        assert client.post(route(name)).status_code == 405


def test_latest_trial_without_artifacts_does_not_fall_back(evaluation_client):
    client, _, report_path, data, _ = evaluation_client
    data["trials"].append({"case_id": CASE_ID, "repeat": 3, "job_id": "c" * 32, "artifacts": {}})
    report_path.write_text(json.dumps(data), encoding="utf8")
    assert client.get(route("overlay.png")).status_code == 404


def test_collision_safe_run_id_can_serve_its_own_artifact(evaluation_client):
    client, root, report_path, data, _ = evaluation_client
    run_id = RUN_ID + "-1a2b3c4d"
    destination = root / "autonomous-qualification" / run_id / "agent-runtime" / "jobs" / JOB_ID / "automatic-001" / "drawing.dxf"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"new isolated drawing")
    data["run_id"] = run_id
    data["trials"][-1]["artifacts"] = {"drawing.dxf":str(destination)}
    report_path.write_text(json.dumps(data), encoding="utf8")
    response = client.get(route("drawing.dxf"))
    assert response.status_code == 200 and response.content == b"new isolated drawing"


@pytest.mark.parametrize("name", [".env", "jobs.sqlite3", "model.json:secret", "..%5Cmodel.json", "%2e%2e%2fmodel.json"])
def test_non_whitelisted_files_are_not_served(evaluation_client, name):
    client, _, _, _, _ = evaluation_client
    assert client.get(route(name)).status_code == 404


@pytest.mark.parametrize("location", ["outside", "inside_other_directory"])
def test_report_cannot_redirect_artifact_paths(evaluation_client, location):
    client, root, report_path, data, _ = evaluation_client
    directory = root.parent if location == "outside" else root / "private"
    directory.mkdir(exist_ok=True)
    secret = directory / "model.json"
    secret.write_text("secret", encoding="utf8")
    data["trials"][-1]["artifacts"]["model.json"] = str(secret)
    report_path.write_text(json.dumps(data), encoding="utf8")
    response = client.get(route("model.json"))
    assert response.status_code == 404
    assert "secret" not in response.text


@pytest.mark.parametrize("mutation", ["bad_json", "wrong_type", "bad_run_id", "bad_job_id", "missing_case", "missing_artifact", "bad_trials"])
def test_malformed_or_missing_resources_return_404(evaluation_client, mutation):
    client, _, report_path, data, artifact_dir = evaluation_client
    if mutation == "bad_json":
        report_path.write_text("{", encoding="utf8")
    else:
        if mutation == "wrong_type":
            data = []
        elif mutation == "bad_run_id":
            data["run_id"] = "../../private"
        elif mutation == "bad_job_id":
            data["trials"][-1]["job_id"] = ".."
        elif mutation == "missing_case":
            data["trials"][-1]["case_id"] = "other"
        elif mutation == "missing_artifact":
            (artifact_dir / "model.json").unlink()
        elif mutation == "bad_trials":
            data["trials"] = "not a list"
        report_path.write_text(json.dumps(data), encoding="utf8")
    assert client.get(route("model.json")).status_code == 404


@pytest.mark.parametrize("target_kind", ["artifact", "report", "directory"])
def test_symlinked_files_and_directories_are_rejected(evaluation_client, target_kind):
    client, root, report_path, _, artifact_dir = evaluation_client
    target = root.parent / "target"
    target.mkdir()
    try:
        if target_kind == "artifact":
            (target / "model.json").write_text("secret", encoding="utf8")
            (artifact_dir / "model.json").unlink()
            (artifact_dir / "model.json").symlink_to(target / "model.json")
        elif target_kind == "report":
            (target / "report.json").write_text(report_path.read_text(encoding="utf8"), encoding="utf8")
            report_path.unlink()
            report_path.symlink_to(target / "report.json")
        else:
            moved = artifact_dir.with_name("original-output")
            artifact_dir.rename(moved)
            artifact_dir.symlink_to(moved, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires Windows developer mode or an appropriate privilege")
    assert client.get(route("model.json")).status_code == 404


def test_report_name_traversal_and_unknown_case_return_404(evaluation_client):
    client, _, _, _, _ = evaluation_client
    for name in ("..%5Coutside.json", "C:%5Coutside.json", "missing.json"):
        assert client.get(route("model.json", report_name=name)).status_code == 404
    assert client.get(route("model.json", case_id="unknown")).status_code == 404


@pytest.mark.parametrize("redirect_report", [False, True])
def test_reparse_path_boundary_without_symlink_privileges(evaluation_client, monkeypatch, redirect_report):
    client, root, report_path, _, artifact_dir = evaluation_client
    checked = report_path if redirect_report else artifact_dir / "model.json"
    redirected = root.parent / "private.json"
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == checked:
            return redirected
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    assert client.get(route("model.json")).status_code == 404
