"""Stage isolation and preservation for the oracle-mask development runner."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts.run_oracle_mask_reconstruction import run_oracle_mask_case


def _fake_inputs(monkeypatch, records=None):
    def frozen(_, case_id, output):
        assert case_id == "case"
        output.mkdir()
        image, ocr, mask = output / "source-image.png", output / "source-ocr.json", output / "oracle-mask.png"
        Image.fromarray(np.full((20, 30), 255, np.uint8)).save(image)
        Image.fromarray(np.full((10, 15), 255, np.uint8)).save(mask)
        ocr.write_text(json.dumps({"meta": {"original_size": {"width": 30, "height": 20},
                                               "box_coord_space": "original_image"}, "records": records or []}))
        receipt = {"reference_dxf_sha256": "f" * 64, "mask_sha256": "a" * 64,
                   "source_image_sha256": "b" * 64, "source_ocr_sha256": "c" * 64,
                   "source_split_label": "test", "reference_components": 1, "reference_holes": 0}
        (output / "input-manifest.json").write_text(json.dumps(receipt))
        return {"source_image": image, "source_ocr": ocr, "mask": mask,
                "input_manifest": output / "input-manifest.json", "receipt": receipt}
    monkeypatch.setattr("scripts.run_oracle_mask_reconstruction.load_oracle_mask_case", frozen)


def test_offline_prediction_finishes_before_optional_gt_evaluation(tmp_path, monkeypatch):
    _fake_inputs(monkeypatch)
    events = []

    def build(image, document, after, *, segmentation_mask, segmentation_review, progress):
        assert image.name == "source-image.png" and segmentation_mask.name == "oracle-mask.png"
        assert document["meta"]["original_size"] == {"width": 30, "height": 20}
        assert segmentation_review == {"status": "oracle_mask", "source": "registered_dxf_gt",
                                       "reviewed": False}
        after.mkdir()
        (after / "drawing.dxf").write_bytes(b"prediction")
        events.append("build")
        progress("automatic_saved", "")
        return {"entities": []}

    def refine(image, document, model, after, *, use_api, progress, **providers):
        assert use_api is False and providers == {}
        assert (after / "drawing.dxf").read_bytes() == b"prediction"
        events.append("refine")
        progress("parametric_saved", "")
        return model, {"status": "completed", "accepted": True}

    def evaluate(run, *, case_id):
        assert case_id == "case" and (run / "after" / "drawing.dxf").is_file()
        assert json.loads((run / "run-manifest.json").read_text())["status"] == "completed"
        events.append("evaluate")
        (run / "evaluation").mkdir()
        (run / "evaluation" / "summary.json").write_text('{"status":"compared"}')
        return {"status": "compared"}

    monkeypatch.setattr("contour_agent.automatic.build_automatic", build)
    monkeypatch.setattr("contour_agent.parametric_pipeline.refine_parametric", refine)
    monkeypatch.setattr("scripts.evaluate_oracle_mask_run.evaluate_oracle_mask_run", evaluate)
    run = tmp_path / "oracle-run"
    result = run_oracle_mask_case("case", run, manifest_path=tmp_path / "frozen.json", evaluate=True)
    assert events == ["build", "refine", "evaluate"]
    manifest = json.loads(result["manifest"].read_text())
    assert manifest["held_out"] is False and manifest["oracle_mask_conditioned"] is True
    assert manifest["source_gt_sha256"] == "f" * 64
    assert manifest["online_requested"] is False and manifest["provider"]["network_requests"] == 0
    assert manifest["evaluation_status"] == "compared" and manifest["status"] == "completed"
    dimensions = json.loads((run / "after" / "dimension-analysis.json").read_text())
    assert dimensions["provider"]["status"] == "disabled"
    stages = [row["stage"] for row in manifest["stage_history"]]
    assert stages.index("dimension_analysis_finished") < stages.index("topology_and_constraints")
    assert "reference.dxf" not in [path.name for path in (run / "inputs").iterdir()]
    with pytest.raises(ValueError, match="existing evidence"):
        run_oracle_mask_case("case", run)


def test_prediction_failure_keeps_artifact_and_checkpoint(tmp_path, monkeypatch):
    _fake_inputs(monkeypatch)

    def build(image, document, after, *, segmentation_mask, segmentation_review, progress):
        after.mkdir()
        (after / "contour-overlay.png").write_bytes(b"preserved")
        progress("boundary_extracted", "")
        raise RuntimeError("controlled vectorization failure")

    monkeypatch.setattr("contour_agent.automatic.build_automatic", build)
    run = tmp_path / "failed-run"
    with pytest.raises(RuntimeError, match="controlled vectorization"):
        run_oracle_mask_case("case", run)
    assert (run / "after" / "contour-overlay.png").read_bytes() == b"preserved"
    manifest = json.loads((run / "run-manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure_stage"] == "boundary_extracted"
    assert manifest["failure_type"] == "RuntimeError"


def test_explicit_thinking_request_is_frozen_in_provider_manifest(tmp_path, monkeypatch):
    _fake_inputs(monkeypatch)
    def configured(online, provider_profile, *, anthropic_thinking):
        assert online and anthropic_thinking == "disabled"
        return {}, {"id": provider_profile, "wire_api": "anthropic_messages",
                    "anthropic_thinking_mode_requested": anthropic_thinking}
    def build(image, document, after, **kwargs):
        after.mkdir()
        (after / "drawing.dxf").write_bytes(b"prediction")
        return {"entities": []}
    monkeypatch.setattr("scripts.run_oracle_mask_reconstruction._providers", configured)
    monkeypatch.setattr("contour_agent.automatic.build_automatic", build)
    monkeypatch.setattr("contour_agent.parametric_pipeline.refine_parametric",
                        lambda image, document, model, after, **kwargs: (model, {"accepted": False}))
    result = run_oracle_mask_case("case", tmp_path / "explicit", online=True,
                                  provider_profile="selected", anthropic_thinking="disabled")
    manifest = json.loads(result["manifest"].read_text())
    assert manifest["provider"]["anthropic_thinking_mode_requested"] == "disabled"
    assert "thinking_disabled" not in json.dumps(manifest)


def test_online_bundle_is_explicit_and_sanitized(tmp_path, monkeypatch):
    _fake_inputs(monkeypatch)
    secret = "must-never-be-in-run-manifest"

    def configured(online, provider_profile):
        assert online and provider_profile == "selected-provider"
        return {"provider": object(), "planner_provider": object(),
                "editor_provider": object(), "evaluator_provider": object()}, \
               {"id": provider_profile, "status": "configured", "model": "model",
                "wire_api": "responses", "timeout_seconds": 600}

    def build(image, document, after, *, segmentation_mask, segmentation_review, progress):
        after.mkdir()
        (after / "drawing.dxf").write_bytes(b"prediction")
        return {"entities": []}

    def refine(image, document, model, after, *, use_api, progress, **providers):
        assert use_api and set(providers) == {"provider", "planner_provider", "editor_provider", "evaluator_provider"}
        return model, {"status": "candidate_rejected", "accepted": False}

    monkeypatch.setattr("scripts.run_oracle_mask_reconstruction._providers", configured)
    monkeypatch.setattr("contour_agent.automatic.build_automatic", build)
    monkeypatch.setattr("contour_agent.parametric_pipeline.refine_parametric", refine)
    result = run_oracle_mask_case("case", tmp_path / "online", online=True,
                                  provider_profile="selected-provider")
    manifest_text = result["manifest"].read_text()
    manifest = json.loads(manifest_text)
    assert manifest["provider"]["id"] == "selected-provider"
    assert manifest["status"] == "completed_with_retained_draft"
    assert manifest["post_export_evaluation_requested"] is False
    assert secret not in manifest_text and "api_key" not in manifest_text


@pytest.mark.parametrize("published",[False,True])
def test_partial_parametric_publication_is_distinct_from_retained_draft(tmp_path,monkeypatch,published):
    _fake_inputs(monkeypatch)
    monkeypatch.setattr("scripts.run_oracle_mask_reconstruction._providers",lambda *a,**k:({},{}))
    def build(image,document,after,**kwargs):
        after.mkdir()
        (after/"drawing.dxf").write_bytes(b"initial")
        return {"entities":[]}
    def refine(image,document,model,after,**kwargs):
        if published:(after/"drawing.dxf").write_bytes(b"partial-parametric")
        return model,{"status":"completed_with_unresolved_radii","accepted":False,
                      "constraint_subset_accepted":True,
                      "publication":{"status":"committed" if published else "rolled_back","kind":"parametric"}}
    monkeypatch.setattr("contour_agent.automatic.build_automatic",build)
    monkeypatch.setattr("contour_agent.parametric_pipeline.refine_parametric",refine)
    result=run_oracle_mask_case("case",tmp_path/"subset")
    manifest=json.loads(result["manifest"].read_text())
    assert manifest["status"]==("completed_with_unresolved_radii" if published else "completed_with_retained_draft")
    assert manifest["constraint_subset_published"] is published
    assert manifest["parameterization_accepted"] is False


@pytest.mark.parametrize("parser_failure", [False, True])
def test_dimension_parse_once_before_topology_preserves_source(tmp_path, monkeypatch, parser_failure):
    from contour_agent.provider import ProviderError

    records = [{"text": "R36", "box": [[1, 2], [3, 2], [3, 4], [1, 4]]}]
    _fake_inputs(monkeypatch, records)
    calls = []

    class Parser:
        def normalize(self, rows):
            assert rows == [{"id": "r000", "text": "R36"}]
            calls.append("dimension_api")
            if parser_failure:
                raise ProviderError("timeout", "private-provider-error", network_requests=1)
            # A conflicting proposal is visible evidence, not an OCR rewrite.
            return {"status": "succeeded", "network_requests": 1,
                    "http_success": True, "schema_success": True,
                    "dimensions": [{"id": "r000", "kind": "radius", "nominal": 360.,
                                    "upper_deviation": None, "lower_deviation": None}]}

    def configured(online, provider_profile):
        assert online
        return {"dimension_provider": Parser()}, {"id": "test", "status": "configured"}

    def build(image, document, after, **kwargs):
        after.mkdir()
        (after / "drawing.dxf").write_bytes(b"initial-cad")
        calls.append("build")
        return {"entities": [{"id": "arc", "radius": 36.}]}

    def refine(image, document, model, after, *, use_api, progress, **providers):
        assert providers == {} and use_api
        assert document["records"][0]["text"] == "R36"
        assert document["records"][0]["box"] == records[0]["box"]
        assert model["entities"][0]["radius"] == 36.
        assert (after / "drawing.dxf").read_bytes() == b"initial-cad"
        analysis = json.loads((after / "dimension-analysis.json").read_text())
        assert analysis["provider"]["status"] == ("failed" if parser_failure else "succeeded")
        assert analysis["decisions"][0]["local"]["nominal"] == 36.
        assert analysis["counts"]["api_conflicts"] == (0 if parser_failure else 1)
        calls.append("refine")
        return model, {"status": "candidate_rejected", "accepted": False}

    monkeypatch.setattr("scripts.run_oracle_mask_reconstruction._providers", configured)
    monkeypatch.setattr("contour_agent.automatic.build_automatic", build)
    monkeypatch.setattr("contour_agent.parametric_pipeline.refine_parametric", refine)
    result = run_oracle_mask_case("case", tmp_path / "parse-run", online=True)
    assert calls == ["build", "dimension_api", "refine"]
    receipt = result["state"]["dimension_analysis"]
    assert receipt["provider"]["network_requests"] == 1
    assert receipt["geometry_updated_by_api"] is False
    assert "private-provider-error" not in result["manifest"].read_text()
    assert result["prediction"].read_bytes() == b"initial-cad"


def test_configured_dimension_provider_has_one_attempt(monkeypatch):
    from contour_agent.config import Settings
    from scripts.run_oracle_mask_reconstruction import _providers

    class Profile:
        model = "mock-model"
        wire_api = "responses"

        def settings(self, original):
            return Settings(api_key="unit-test-only", model=self.model)

    monkeypatch.setattr("contour_agent.config.load_local_env", lambda: None)
    monkeypatch.setattr("contour_agent.config.provider_registry", lambda settings: ({"mock": Profile()}, "mock"))
    providers, receipt = _providers(True, "mock")
    assert providers["dimension_provider"].max_attempts == 1
    assert receipt["id"] == "mock"
