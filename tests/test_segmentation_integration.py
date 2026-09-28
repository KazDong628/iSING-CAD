"""Server -> service -> real CAD export, with model inference stubbed at its edge.

Only synthetic source images are used. No training, GPU, checkpoint loading,
network calls, dataset labels, or reference geometry are needed.
"""
import json
import io
import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from contour_agent import automatic, segmentation
from contour_agent.config import Settings
from contour_agent.server import create_app


def extraction(*, learned=False):
    return {"status": "needs_review", "polyline_px": [[20, 20], [108, 20], [108, 76], [20, 76], [20, 20]],
            "image_size": {"width": 128, "height": 96}, "issues": [], "evidence": {"synthetic_test": True},
            "learned_segmentation": learned}


@pytest.fixture
def integration(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    origin = dataset / "origin"
    origin.mkdir(parents=True)
    image = origin / "synthetic.png"
    Image.new("RGB", (128, 96), "white").save(image)
    ocr = origin / "synthetic.json"
    ocr.write_text(json.dumps({"meta": {"original_size": {"width": 128, "height": 96}}, "records": []}))
    checkpoint = tmp_path / "configured-model.pt"
    checkpoint.write_bytes(b"not loaded: inference is stubbed for this integration test")
    settings = Settings(dataset_root=dataset, runtime_root=tmp_path / "runtime", api_key="",
                        segmentation_checkpoint=str(checkpoint))
    calls = {"learned": [], "primary": [], "fallback": []}

    def learned(model_path, source_path, output_dir):
        calls["learned"].append((Path(model_path), Path(source_path), Path(output_dir)))
        return extraction(learned=True)

    def primary(source_path, output_dir):
        calls["primary"].append((Path(source_path), Path(output_dir)))
        return extraction()

    def fallback(*args):
        calls["fallback"].append(args)
        pytest.fail("Unexpected heuristic fallback")

    monkeypatch.setattr(segmentation, "cached_extract", learned)
    monkeypatch.setattr(automatic, "extract_main_profile", primary)
    monkeypatch.setattr(automatic, "extract_unhatched", fallback)
    monkeypatch.setattr(automatic, "estimate_scale", lambda *args: {"status": "unresolved", "pixels_per_mm": None, "issues": []})
    app = create_app(settings)
    # The endpoint still exercises the real service/state machine, but makes
    # its worker immediate for deterministic tests without polling or sleeps.
    monkeypatch.setattr(app.state.service.executor, "submit", lambda fn, *args, **kwargs: fn(*args, **kwargs))
    monkeypatch.setattr(app.state.service.vision_provider, "inspect", lambda *args, **kwargs: pytest.fail("Unexpected online call"))
    with TestClient(app) as client:
        yield {"client": client, "service": app.state.service, "settings": settings,
               "image": image, "ocr": ocr, "checkpoint": checkpoint, "calls": calls}


def create_case(context, **options):
    return context["client"].post("/api/jobs", json={"case_id": "synthetic", "use_api": False, **options})


def read_model(context, job):
    response = context["client"].get(job["artifacts"]["model"])
    assert response.status_code == 200
    return response.json()


@pytest.mark.parametrize("explicit_false", [False, True])
def test_configured_checkpoint_does_not_change_default_method(integration, explicit_false):
    context = integration
    assert context["client"].get("/api/config").json()["segmentation_available"] is True
    response = create_case(context, **({"use_segmentation": False} if explicit_false else {}))
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "completed" and job["use_segmentation"] is False
    assert len(context["calls"]["primary"]) == 1
    assert not context["calls"]["learned"] and not context["calls"]["fallback"]
    model = read_model(context, job)
    assert model["algorithm_version"] == "hatch-dimension-vector-v3"
    assert model["validation"]["passed"] and not model["validation"]["scaled_mm"]


@pytest.mark.parametrize("upload", [False, True])
def test_explicit_selection_uses_configured_checkpoint_through_export(integration, upload):
    context = integration
    if upload:
        response = context["client"].post("/api/uploads", data={"use_api": "false", "use_segmentation": "true"},
                    files={"image": ("source.png", context["image"].read_bytes(), "image/png"),
                           "ocr": ("ocr.json", context["ocr"].read_bytes(), "application/json")})
    else:
        response = create_case(context, use_segmentation=True)
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "completed" and job["use_segmentation"] is True
    assert len(context["calls"]["learned"]) == 1
    configured, source, evidence_dir = context["calls"]["learned"][0]
    assert configured == context["checkpoint"]
    assert source.read_bytes() == context["image"].read_bytes()
    assert evidence_dir.name == "learned-evidence"
    assert not context["calls"]["primary"] and not context["calls"]["fallback"]
    model = read_model(context, job)
    assert model["algorithm_version"] == "unet-resnet18-weak-v1"
    assert model["extraction"]["learned_segmentation"] is True
    assert model["validation"]["dxf_readback"]["passed"]
    assert not model["validation"]["dimensions_verified"]
    assert not model["validation"]["reference_verified"]
    assert not model["ground_truth_used"] and not model["template_used"]


def test_default_heuristic_fallback_remains_available(integration, monkeypatch):
    context = integration
    calls = context["calls"]

    def empty_primary(*args):
        calls["primary"].append(args)
        return {"status": "needs_review", "polyline_px": [], "issues": ["No periodic hatch"]}

    def fallback(*args):
        calls["fallback"].append(args)
        return extraction()

    monkeypatch.setattr(automatic, "extract_main_profile", empty_primary)
    monkeypatch.setattr(automatic, "extract_unhatched", fallback)
    response = create_case(context)
    assert response.status_code == 202 and response.json()["status"] == "completed"
    assert len(calls["primary"]) == len(calls["fallback"]) == 1
    assert not calls["learned"]
    model = read_model(context, response.json())
    assert model["extraction"]["fallback_used"] is True
    assert model["algorithm_version"] == "hatch-dimension-vector-v3"


@pytest.mark.parametrize("no_result", [None, {"status": "needs_review", "polyline_px": [], "issues": ["No predicted material"]}])
def test_empty_learned_result_fails_without_switching_to_heuristics(integration, monkeypatch, no_result):
    context = integration

    def empty_model(*args):
        context["calls"]["learned"].append(args)
        return no_result

    monkeypatch.setattr(segmentation, "cached_extract", empty_model)
    response = create_case(context, use_segmentation=True)
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "failed" and job["use_segmentation"] is True
    assert not job["automatic_completion"] and not job["artifacts"]
    assert job["issues"] and len(context["calls"]["learned"]) == 1
    assert not context["calls"]["primary"] and not context["calls"]["fallback"]
    assert not list(context["settings"].runtime_root.rglob("drawing.dxf"))


@pytest.mark.parametrize("missing_kind", ["unconfigured", "missing_file"])
def test_missing_checkpoint_is_actionable_and_creates_no_job(integration, missing_kind):
    context = integration
    context["settings"].segmentation_checkpoint = "" if missing_kind == "unconfigured" else str(context["checkpoint"].with_name("absent.pt"))
    assert context["client"].get("/api/config").json()["segmentation_available"] is False
    response = create_case(context, use_segmentation=True)
    assert response.status_code == 400
    assert "checkpoint" in response.json()["detail"] and "训练" in response.json()["detail"]
    assert not context["service"].store.list()
    assert not any(context["calls"].values())


def test_incompatible_checkpoint_failure_does_not_export_heuristic_result(integration, monkeypatch):
    context = integration

    def incompatible(*args):
        context["calls"]["learned"].append(args)
        raise ValueError("Unsupported segmentation checkpoint specification")

    monkeypatch.setattr(segmentation, "cached_extract", incompatible)
    job = create_case(context, use_segmentation=True).json()
    assert job["status"] == "failed" and not job["artifacts"]
    assert any("checkpoint" in issue for issue in job["issues"])
    assert len(context["calls"]["learned"]) == 1
    assert not context["calls"]["primary"] and not context["calls"]["fallback"]


def test_client_cannot_supply_an_arbitrary_checkpoint_path(integration):
    response = create_case(integration, use_segmentation=True, segmentation_checkpoint="../../private.pt")
    assert response.status_code == 422
    assert not integration["service"].store.list()
    assert not any(integration["calls"].values())


def test_segmentation_review_gate_pauses_accepts_pixel_edits_and_resumes_export(integration, monkeypatch):
    context = integration

    def learned_with_artifacts(model_path, source_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", (128, 96), 0)
        for x in range(20, 109):
            for y in range(20, 77):
                mask.putpixel((x, y), 255)
        mask.save(output_dir / "prediction-mask.png")
        Image.open(source_path).convert("RGB").save(output_dir / "prediction-overlay.png")
        (output_dir / "segmentation.json").write_text(json.dumps({
            "model": {"label_status": "registered_dxf_gt_experimental"},
            "status": "needs_review",
        }), encoding="utf8")
        context["calls"]["learned"].append((Path(model_path), Path(source_path), output_dir))
        return extraction(learned=True)

    monkeypatch.setattr(segmentation, "cached_extract", learned_with_artifacts)
    response = create_case(context, use_segmentation=True, require_segmentation_review=True)
    assert response.status_code == 202
    pending = response.json()
    assert pending["status"] == "awaiting_segmentation_review"
    assert pending["segmentation_review"]["status"] == "pending"
    assert pending.get("geometry") is None
    assert "dxf" not in pending["artifacts"]
    assert context["client"].get(pending["artifacts"]["segmentation_mask"]).status_code == 200

    wrong = io.BytesIO()
    Image.new("L", (32, 24), 255).save(wrong, format="PNG")
    rejected = context["client"].put(
        f"/api/jobs/{pending['id']}/segmentation-review",
        files={"mask": ("wrong.png", wrong.getvalue(), "image/png")},
    )
    assert rejected.status_code == 400
    assert "尺寸" in rejected.json()["detail"]

    reviewed = Image.new("L", (128, 96), 0)
    for x in range(18, 111):
        for y in range(18, 79):
            reviewed.putpixel((x, y), 255)
    payload = io.BytesIO()
    reviewed.save(payload, format="PNG")
    accepted = context["client"].put(
        f"/api/jobs/{pending['id']}/segmentation-review",
        files={"mask": ("reviewed.png", payload.getvalue(), "image/png")},
    )
    assert accepted.status_code == 202
    job = accepted.json()
    assert job["status"] in {"completed", "needs_review"}
    assert job["segmentation_review"]["status"] == "approved"
    assert job["segmentation_review"]["added_pixels"] > 0
    assert "mask_path" not in job["segmentation_review"]
    assert job["manual_intervention"] is True
    assert context["client"].get(job["artifacts"]["model_segmentation_mask"]).status_code == 200
    assert context["client"].get(job["artifacts"]["reviewed_segmentation_mask"]).status_code == 200
    assert context["client"].get(job["artifacts"]["segmentation_review"]).status_code == 200
    model = read_model(context, job)
    assert model["algorithm_version"] == "human-reviewed-unet-mask-v1"
    assert model["manual_intervention"] is True
    assert model["ground_truth_used"] is False
    assert model["validation"]["dxf_readback"]["passed"]


def test_uploaded_gt_dxf_generates_verified_oracle_mask(integration, monkeypatch):
    context = integration

    def learned_with_artifacts(model_path, source_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        Image.new("L", (128, 96), 0).save(output_dir / "prediction-mask.png")
        Image.open(source_path).convert("RGB").save(output_dir / "prediction-overlay.png")
        context["calls"]["learned"].append((Path(model_path), Path(source_path), output_dir))
        return extraction(learned=True)

    monkeypatch.setattr(segmentation, "cached_extract", learned_with_artifacts)
    pending = create_case(context, use_segmentation=True, require_segmentation_review=True).json()
    assert pending["status"] == "awaiting_segmentation_review"

    mask = Image.new("L", (128, 96), 0)
    for x in range(18, 111):
        for y in range(18, 79):
            mask.putpixel((x, y), 255)
    payload = io.BytesIO()
    mask.save(payload, format="PNG")
    pixels = payload.getvalue()
    rejected = context["client"].put(
        f"/api/jobs/{pending['id']}/segmentation-review",
        files={"mask": ("oracle.png", pixels, "image/png"),
               "mask_source": (None, "gt_oracle")},
    )
    assert rejected.status_code == 400

    def generated(dxf_bytes, image_path, ocr_path, hint_path, mask_size, *, temporary_root):
        assert dxf_bytes == b"uploaded reference DXF"
        assert mask_size == (128, 96) and Path(hint_path).is_file()
        return pixels, {
            "schema_version": "job-oracle-dxf-import-v1", "source": "uploaded_gt_dxf",
            "source_gt_sha256": hashlib.sha256(dxf_bytes).hexdigest(),
            "mask_sha256": hashlib.sha256(pixels).hexdigest(),
            "source_image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
            "source_ocr_sha256": hashlib.sha256(Path(ocr_path).read_bytes()).hexdigest(),
            "mask_size": {"width": 128, "height": 96}, "foreground_pixels": 93 * 61,
            "ground_truth_dxf_coordinates_sent_to_provider": False,
        }

    monkeypatch.setattr("contour_agent.oracle_mask_import.generate_oracle_mask", generated)
    imported = context["client"].post(
        f"/api/jobs/{pending['id']}/oracle-mask-from-dxf",
        files={"dxf": ("reference.dxf", b"uploaded reference DXF", "application/dxf")},
    )
    assert imported.status_code == 200
    assert imported.json()["artifacts"]["oracle_generated_mask"]
    assert context["client"].get(imported.json()["artifacts"]["oracle_generated_mask"]).content == pixels
    assert context["client"].get(imported.json()["artifacts"]["oracle_mask_import"]).json()["ground_truth_dxf_coordinates_sent_to_provider"] is False
    assert context["client"].get(f"/api/jobs/{pending['id']}/artifacts/reference.dxf").status_code == 404

    tampered = io.BytesIO()
    altered = mask.copy()
    altered.putpixel((18, 18), 0)
    altered.save(tampered, format="PNG")
    rejected = context["client"].put(
        f"/api/jobs/{pending['id']}/segmentation-review",
        files={"mask": ("tampered.png", tampered.getvalue(), "image/png"),
               "mask_source": (None, "gt_oracle")},
    )
    assert rejected.status_code == 400
    response = context["client"].put(
        f"/api/jobs/{pending['id']}/segmentation-review",
        files={"mask": ("oracle.png", pixels, "image/png"),
               "mask_source": (None, "gt_oracle")},
    )
    assert response.status_code == 202
    job = response.json()
    assert job["status"] in {"completed", "needs_review"}
    assert job["segmentation_review"]["source"] == "registered_dxf_gt"
    assert job["segmentation_review"]["ground_truth_used"] is True
    assert job["segmentation_review"]["reviewed"] is False
    assert job["segmentation_review"]["ground_truth_dxf_coordinates_used_for_mask_creation"] is True
    assert job["segmentation_review"]["ground_truth_dxf_coordinates_used_for_prediction"] is False
    assert job["manual_intervention"] is False
    model = read_model(context, job)
    assert model["algorithm_version"] == "oracle-gt-mask-conditioned-v1"
    assert model["oracle_mask_conditioned"] is True
    assert model["ground_truth_used"] is True
    assert model["manual_intervention"] is False
    assert model["validation"]["dxf_readback"]["passed"]

    edited_pending = create_case(context, use_segmentation=True, require_segmentation_review=True).json()
    edited_import = context["client"].post(
        f"/api/jobs/{edited_pending['id']}/oracle-mask-from-dxf",
        files={"dxf": ("reference.dxf", b"uploaded reference DXF", "application/dxf")},
    )
    assert edited_import.status_code == 200
    edited_response = context["client"].put(
        f"/api/jobs/{edited_pending['id']}/segmentation-review",
        files={"mask": ("edited.png", tampered.getvalue(), "image/png"),
               "mask_source": (None, "gt_oracle_edited")},
    )
    assert edited_response.status_code == 202
    edited_job = edited_response.json()
    assert edited_job["segmentation_review"]["source"] == "registered_dxf_gt_edited"
    assert edited_job["segmentation_review"]["ground_truth_used"] is True
    assert edited_job["segmentation_review"]["oracle_mask_conditioned"] is False
    edited_model = read_model(context, edited_job)
    assert edited_model["algorithm_version"] == "gt-assisted-human-reviewed-mask-v1"
    assert edited_model["ground_truth_used"] is True
    assert edited_model["manual_intervention"] is True
