"""Screenshot edits export real source-only drafts, never dimensional success."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import ezdxf
import pytest

from contour_agent.config import Settings
from contour_agent.parametric_pipeline import CORE
from contour_agent.service import AgentService, _parameterization_needs_review
from test_parametric_pipeline import prepared, supported_graph


def feedback_draft(tmp_path, monkeypatch, *, selected="alternative"):
    image, baseline, parent_output = prepared(tmp_path)
    raw_ocr = tmp_path / "source-ocr.json"
    raw_ocr.write_text(json.dumps({"meta": {"original_size": {"width": 200, "height": 200}},
                                   "records": []}), encoding="utf8")
    graph = supported_graph(image, baseline)
    candidates = [{"id": name, "graph": deepcopy(graph), "ground_truth_used": False,
                   "source_sha256": graph["source_sha256"], "parent_topology_sha256": "synthetic-base"}
                  for name in ("current", "alternative")]
    bundle = {"candidates": candidates, "source_sha256": graph["source_sha256"],
              "base_graph_sha256": "synthetic-base"}
    for name, value in (("baseline-model.json", baseline), ("topology-candidates.json", bundle),
                        ("topology-plan.json", {"selected_candidate_id": "current"})):
        (parent_output / name).write_text(json.dumps(value), encoding="utf8")
    before = {name: (parent_output / name).read_bytes() for name in CORE}
    monkeypatch.setattr("contour_agent.service.build_catalog", lambda *a: {"cases": []})
    service = AgentService(Settings(runtime_root=tmp_path / "runtime", api_key=""))
    receipt = {"status": "succeeded", "schema_success": True, "http_success": True,
               "network_requests": 1, "selected_candidate_id": selected, "operations": []}
    # Only transport is substituted.  Candidate admission, source/coordinate
    # checks, the native DXF writer/readback and durable publication are real.
    monkeypatch.setattr(service, "_provider_bundle", lambda job: {
        "feedback": SimpleNamespace(select=lambda *a, **k: deepcopy(receipt))})
    timestamp = "2026-10-10T00:00:00+00:00"
    job = {"id": "feedback", "mode": "autonomous_revision", "status": "queued",
           "created_at": timestamp, "updated_at": timestamp, "events": [], "issues": [],
           "artifacts": {}, "source_image": str(image), "source_ocr": str(raw_ocr),
           "feedback_screenshot": str(image), "feedback_instruction": "检查并修正主轮廓。",
           "parent_job_id": "persisted", "parent_artifact_directory": str(parent_output),
           "use_api": True, "solve_revision": 1, "automatic_completion": False}
    service.store.save(job)
    service._feedback_revision(job["id"])
    return service, service.store.get(job["id"]), parent_output, before


@pytest.mark.parametrize("selected", ["current", "alternative"])
def test_feedback_dxf_available_without_claiming_constraints_complete(tmp_path, monkeypatch, selected):
    service, job, parent_output, before = feedback_draft(tmp_path, monkeypatch, selected=selected)
    try:
        assert job["status"] == "needs_review"
        assert job["completion_class"] == "source_topology_draft"
        assert job["artifact_available"] and not job["constraints_complete"]
        assert not job["automatic_completion"]
        assert job["validation"]["passed"] and job["validation"]["dxf_readback"]["passed"]
        assert job["provider"]["http_success"]  # Transport success stays independent.
        assert job["parameterization"]["topology_exported"]
        assert not job["parameterization"]["constraint_subset_accepted"]
        assert not job["parameterization"]["accepted"]
        assert not job["parameterization"]["all_dimensions_verified"]
        assert _parameterization_needs_review(job)
        assert any("标注绑定与严格约束求解" in issue for issue in job["issues"])
        assert any("没有变化" in issue for issue in job["issues"]) == (selected == "current")
        assert {"dxf", "model", "overlay", "parameterization"} <= set(job["artifacts"])
        output = Path(job["artifact_directory"])
        assert len(ezdxf.readfile(output / "drawing.dxf").modelspace()) == 4
        model = json.loads((output / "model.json").read_text("utf8"))
        assert model["artifact_available"] and not model["constraints_complete"]
        assert not model["automatic_completion"]
        assert model["parameterization"] == job["parameterization"]
        assert before == {name: (parent_output / name).read_bytes() for name in CORE}
    finally:
        service.close()


@pytest.mark.parametrize("corrupt_native", [False, True])
def test_restart_retains_only_verified_feedback_draft_without_upgrading_it(tmp_path, monkeypatch, corrupt_native):
    service, job, _, _ = feedback_draft(tmp_path, monkeypatch)
    output = Path(job["artifact_directory"])
    job["status"] = "auditing"  # Simulate interruption after CORE publication.
    service.store.save(job)
    service.close()
    if corrupt_native:
        document = ezdxf.readfile(output / "drawing.dxf")
        list(document.modelspace())[0].dxf.start = (1000, 1000, 0)
        document.saveas(output / "drawing.dxf")
    before = {name: (output / name).read_bytes() for name in CORE}
    restored_service = AgentService(Settings(runtime_root=tmp_path / "runtime", api_key=""), recover_running=True)
    try:
        restored = restored_service.store.get(job["id"])
        assert restored["status"] == ("failed" if corrupt_native else "needs_review")
        assert not restored["automatic_completion"] and not restored["constraints_complete"]
        assert not restored["parameterization"]["accepted"]
        assert not restored["parameterization"]["constraint_subset_accepted"]
        assert restored["artifacts"]["dxf"]  # Even failed evidence remains available.
        assert before == {name: (output / name).read_bytes() for name in CORE}
        if not corrupt_native:
            assert restored["artifact_available"] and _parameterization_needs_review(restored)
            assert restored["geometry"]["entities"] == job["geometry"]["entities"]
        else:
            assert not restored["validation"]["passed"]
    finally:
        restored_service.close()
