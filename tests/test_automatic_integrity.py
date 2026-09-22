"""Regressions for persisted autonomous artifacts and exact DXF readback."""
import json

import ezdxf
import pytest

from contour_agent.automatic import _verify_dxf_readback
from contour_agent.config import Settings
from contour_agent.service import AgentService
from contour_agent.store import JobStore


@pytest.fixture
def saved_geometry(tmp_path):
    expected = [
        {"id": "line", "type": "LINE", "start": [0., 0.], "end": [10., 0.]},
        {"id": "ccw", "type": "ARC", "start": [10., 0.], "end": [15., 5.], "center": [10., 5.], "radius": 5., "clockwise": False},
        {"id": "cw", "type": "ARC", "start": [15., 5.], "end": [20., 10.], "center": [20., 5.], "radius": 5., "clockwise": True},
    ]
    document = ezdxf.new("R2010")
    document.units = 4
    document.modelspace().add_line((0, 0), (10, 0))
    document.modelspace().add_arc((10, 5), 5, 270, 0)
    # Clockwise chains are stored with exchanged endpoints by DXF ARC.
    document.modelspace().add_arc((20, 5), 5, 90, 180)
    filename = tmp_path / "roundtrip.dxf"
    document.saveas(filename)
    return ezdxf.readfile(filename), expected


def test_readback_accepts_saved_line_and_both_arc_directions(saved_geometry):
    document, expected = saved_geometry
    result = _verify_dxf_readback(document, expected, expected_units=4)
    assert result["passed"]
    assert result["checked_entities"] == 3
    assert all(row["max_geometry_error"] < 1e-10 for row in result["entities"])


@pytest.mark.parametrize("mutate", [
    lambda items: setattr(items[0].dxf, "end", (10.2, 0, 0)),
    lambda items: setattr(items[0].dxf, "start", (0, 0, .2)),
    lambda items: setattr(items[1].dxf, "radius", 5.2),
    lambda items: setattr(items[1].dxf, "center", (10.2, 5, 0)),
    lambda items: setattr(items[1].dxf, "start_angle", 269),
    lambda items: setattr(items[2].dxf, "end_angle", 181),
    lambda items: setattr(items[2].dxf, "extrusion", (0, 0, -1)),
])
def test_same_count_same_units_cannot_hide_changed_geometry(saved_geometry, mutate):
    document, expected = saved_geometry
    mutate(list(document.modelspace()))
    result = _verify_dxf_readback(document, expected, expected_units=4)
    assert result["entity_count_matches"] and result["units_match"]
    assert not result["passed"]
    assert any(not row["passed"] for row in result["entities"])


def test_readback_rejects_entity_type_count_and_units(saved_geometry):
    document, expected = saved_geometry
    model = document.modelspace()
    model.delete_entity(list(model)[-1])
    assert not _verify_dxf_readback(document, expected, expected_units=4)["entity_count_matches"]
    model.add_circle((20, 5), 5)
    result = _verify_dxf_readback(document, expected, expected_units=4)
    assert result["entity_count_matches"] and not result["passed"]
    assert result["entities"][-1]["reason"] == "entity_type_mismatch"
    document.units = 0
    assert not _verify_dxf_readback(document, expected, expected_units=4)["units_match"]


def seed_job(tmp_path, *, stage="auditing", published=True, mode="autonomous_image"):
    output = tmp_path / "jobs" / "persisted" / "automatic-001"
    output.mkdir(parents=True)
    validation={"passed":published,"scaled_mm":True,"dimensions_verified":False}
    if published:
        # Recovery must validate actual files, not accept a stale journal flag.
        from PIL import Image
        points=[[0.,0.],[10.,0.],[10.,10.],[0.,10.]]
        entities=[{"id":f"g{i}","type":"LINE","start":p,"end":points[(i+1)%4]} for i,p in enumerate(points)]
        document=ezdxf.new("R2010");document.units=4
        for entity in entities:document.modelspace().add_line(entity["start"],entity["end"])
        document.saveas(output/"drawing.dxf")
        curve_fit={"passed":True}
        model={"entities":entities,"validation":validation,"automatic_completion":True,"curve_fit":curve_fit,
               "scale":{"status":"resolved","pixels_per_mm":2.},"coordinate_system":{"units":"mm","origin_source_px":[0,20]},
               "bounds":{"min_x":0,"min_y":0,"max_x":10,"max_y":10},"extraction":{}}
        for name,value in (("model.json",model),("validation.json",validation),("curve-fit.json",curve_fit)):
            (output/name).write_text(json.dumps(value),encoding="utf8")
        (output/"preview.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0H10V10H0Z"/></svg>')
        Image.new("RGB",(32,32),"white").save(output/"overlay.png")
    job = {"id": "persisted", "mode": mode, "status": stage, "updated_at": "2026-09-20T00:00:00+00:00",
           "parameters": [], "events": [], "issues": [], "automatic_completion": published,
           "validation": validation, "artifact_directory": str(output),
           "artifacts": {"dxf": "/api/jobs/persisted/artifacts/drawing.dxf"} if published else {},
           "provider": {"status": "pending", "network_requests": 0}, "use_api": True}
    JobStore(tmp_path).save(job)
    return job, output


def recovered_service(tmp_path, monkeypatch):
    monkeypatch.setattr("contour_agent.service.build_catalog", lambda root: {"cases": [], "counts": {}})
    def forbidden(*args, **kwargs):
        pytest.fail("recovery must not start an online request")
    monkeypatch.setattr("contour_agent.service.VisionProvider.inspect", forbidden)
    return AgentService(Settings(runtime_root=tmp_path, api_key=""), recover_running=True)


def test_restart_preserves_valid_artifacts_and_marks_audit_interrupted(tmp_path, monkeypatch):
    before, output = seed_job(tmp_path)
    content = {p.name: p.read_bytes() for p in output.iterdir()}
    service = recovered_service(tmp_path, monkeypatch)
    try:
        after = service.store.get("persisted")
        assert after["status"] == "completed" and after["automatic_completion"]
        assert all(after["artifacts"][key] == value for key,value in before["artifacts"].items())
        assert all(after["artifacts"].get(key) for key in ("dxf","svg","model","validation","overlay"))
        assert after["validation"] == before["validation"]
        assert after["provider"]["status"] == "interrupted"
        assert after["provider"]["network_request_state"] == "unknown"
        assert after["provider"]["network_requests"] is None and after["provider"]["http_success"] is None
        assert after["provider"]["verdict"] == "uncertain" and not after["provider"]["schema_success"]
        assert "确认参数" not in after["events"][-1]["message"]
        assert content == {p.name: p.read_bytes() for p in output.iterdir()}
    finally:
        service.close()


@pytest.mark.parametrize("stage", ["queued", "extracting", "solving"])
def test_restart_without_completed_output_remains_failed(tmp_path, monkeypatch, stage):
    seed_job(tmp_path, stage=stage, published=False)
    service = recovered_service(tmp_path, monkeypatch)
    try:
        after = service.store.get("persisted")
        assert after["status"] == "failed" and not after["automatic_completion"]
        assert not after["artifacts"] and "重新运行" in after["events"][-1]["message"]
    finally:
        service.close()


def test_restart_does_not_restore_missing_files_or_forge_audit_success(tmp_path, monkeypatch):
    before, output = seed_job(tmp_path)
    (output / "drawing.dxf").unlink()
    service = recovered_service(tmp_path, monkeypatch)
    try:
        after = service.store.get("persisted")
        assert after["status"] == "failed" and not after["automatic_completion"]
        assert after["provider"]["status"] == "interrupted"
    finally:
        service.close()


def test_legacy_template_recovery_keeps_existing_review_behavior(tmp_path, monkeypatch):
    before, _ = seed_job(tmp_path, stage="solving", mode="template_assisted")
    before["parameters"] = [{"id": "old-parameter", "value": 10}]
    JobStore(tmp_path).save(before)
    service = recovered_service(tmp_path, monkeypatch)
    try:
        assert service.store.get("persisted")["status"] == "needs_review"
    finally:
        service.close()
