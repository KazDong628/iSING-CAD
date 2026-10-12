import hashlib
import json
from types import SimpleNamespace
import zipfile

from scripts.check_dxf_output_coverage import render, worker


def test_report_keeps_failure_denominator_and_distinguishes_input_conditions(tmp_path):
    for case in ("oracle", "source"):
        directory = tmp_path / case / "after"
        directory.mkdir(parents=True)
        (directory / "drawing.dxf").write_bytes(case.encode())
    rows = [
        {"case_id": "oracle", "status": "initial_cad_exported", "mask_condition": "gt_derived_raster_mask", "dxf_availability": {"available": True}},
        {"case_id": "source", "status": "source_draft_retained", "mask_condition": "source_only_no_usable_gt_mask", "dxf_availability": {"available": True}},
        {"case_id": "failed", "status": "failed", "mask_condition": None, "dxf_availability": {"available": False}},
    ]
    result = render(tmp_path, rows)
    assert result["total_cases"] == 3 and result["exported_readable_dxf_cases"] == 2
    assert result["oracle_mask_cases"] == result["oracle_mask_dxf_cases"] == 1
    assert not result["engineering_accuracy_tested"] and not result["online_api_called"]
    with zipfile.ZipFile(tmp_path / "automatic-dxf-outputs.zip") as bundle:
        assert set(bundle.namelist()) == {"oracle/drawing.dxf", "oracle/output-status.json", "source/drawing.dxf", "source/output-status.json"}


def test_inadmissible_gt_input_is_not_silently_relabelled_as_oracle(tmp_path, monkeypatch):
    image, ocr = tmp_path / "image.png", tmp_path / "ocr.json"
    image.write_bytes(b"source image")
    ocr.write_text('{"meta":{},"records":[]}', encoding="utf8")
    row = {"id": "case", "artifact_status": "ready", "source_image": str(image), "source_ocr": str(ocr),
           "source_image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
           "source_ocr_sha256": hashlib.sha256(ocr.read_bytes()).hexdigest()}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [row]}))
    def reject(*a, **k):
        raise ValueError("ambiguous multi-component GT")
    observed = []
    def build(image, document, output, *, progress, **kwargs):
        observed.append(kwargs)
        return {"entities": [], "complete_material_exterior": False}
    monkeypatch.setattr("scripts.oracle_mask_inputs.load_oracle_mask_case", reject)
    monkeypatch.setattr("contour_agent.automatic.build_automatic", build)
    monkeypatch.setattr("contour_agent.draft_artifacts.dxf_availability", lambda *a: {"available": False})
    args = SimpleNamespace(code_root=tmp_path, manifest=manifest, output=tmp_path / "run", case="case", source_only_missing_mask=True)
    worker(args)
    state = json.loads((args.output / "run-manifest.json").read_text(encoding="utf8"))
    assert state["mask_condition"] == "source_only_no_usable_gt_mask"
    assert "oracle_input_unavailable" in state and observed == [{}]
    assert not state["engineering_accepted"] and not state["online_api_called"]
