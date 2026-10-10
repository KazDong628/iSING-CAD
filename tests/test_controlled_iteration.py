import importlib.util
import json
from pathlib import Path

import ezdxf
import numpy as np
import pytest
from PIL import Image


SPEC = importlib.util.spec_from_file_location("controlled_runner", Path(__file__).parents[1] / "scripts/run_controlled_iteration.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def frozen_case(tmp_path):
    source = tmp_path / "source" / "inputs"
    source.mkdir(parents=True)
    Image.new("RGB", (8, 6), "white").save(source / "source-image.png")
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[1:5, 1:7] = 255
    Image.fromarray(mask).save(source / "oracle-mask.png")
    (source / "source-ocr.json").write_text('{"records": []}', encoding="utf8")
    receipt = {"schema_version": "oracle-mask-input-v1", "case_id": "synthetic", "oracle_mask": True,
               "held_out": False, "original_image_size": {"width": 8, "height": 6},
               "mask_size": {"width": 8, "height": 6}, "mask_geometry": {"foreground_pixels": 24},
               "reference_dxf_sha256": "a" * 64,
               "unexpected_reference_coordinates": [1, 2, 3]}
    for field, hash_field, name in (("source_image", "source_image_sha256", "source-image.png"),
                                   ("source_ocr", "source_ocr_sha256", "source-ocr.json"),
                                   ("mask", "mask_sha256", "oracle-mask.png")):
        receipt[field] = str(source / name)
        receipt[hash_field] = runner.digest(source / name)
    runner.write(source / "input-manifest.json", receipt)
    (source / "reference.dxf").write_text("must never copy or open", encoding="utf8")
    return source.parent


def test_only_three_frozen_inputs_and_whitelisted_receipt(tmp_path):
    source = frozen_case(tmp_path)
    output = tmp_path / "new-inputs"
    paths, receipt = runner.freeze_inputs(source, output)
    assert len(paths) == 3
    assert {p.name for p in output.iterdir()} == {"source-image.png", "source-ocr.json", "oracle-mask.png", "input-manifest.json"}
    assert "unexpected_reference_coordinates" not in receipt
    for field, path in paths.items():
        assert runner.digest(path) == runner.digest(source / "inputs" / path.name)


def test_changed_source_rejected_before_prediction(tmp_path):
    source = frozen_case(tmp_path)
    (source / "inputs" / "source-ocr.json").write_text('{"records": [1]}', encoding="utf8")
    with pytest.raises(ValueError, match="hash_or_path"):
        runner.freeze_inputs(source, tmp_path / "new-inputs")


def test_current_native_radius_readback_overrules_stale_contract(tmp_path):
    doc = ezdxf.new("R2010")
    doc.units = 4
    doc.modelspace().add_arc((0, 0), 4, 0, 90)
    doc.saveas(tmp_path / "drawing.dxf")
    runner.write(tmp_path / "model.json", {
        "entities": [{"id": "g0", "type": "ARC", "radius": 3}],
        "parameterization": {"constraints": [{"kind": "radius", "entities": ["g0"], "value": 3, "record_id": "r0"}]},
        "validation": {"annotation_radius_contract": {"satisfied": True}},
    })
    result = runner.local_export_audit(tmp_path)
    assert result["exact_radius_count"] == 0
    assert result["radius_contract_complete"] is False
    assert result["exact_published_radii"]["checks"][0]["dxf_radius"] == 4


def test_no_radius_constraints_does_not_prove_coverage(tmp_path):
    doc = ezdxf.new("R2010")
    doc.units = 4
    doc.modelspace().add_arc((0, 0), 3, 0, 90)
    doc.saveas(tmp_path / "drawing.dxf")
    runner.write(tmp_path / "model.json", {
        "entities": [{"id": "g0", "type": "ARC", "radius": 3}],
        "validation": {"annotation_radius_contract": {"satisfied": True}},
    })
    result = runner.local_export_audit(tmp_path)
    assert result["exact_published_radii"]["passed"] is True
    assert result["radius_contract_complete"] is False


def test_missing_evaluation_inventory_fails_before_prediction(tmp_path, capsys):
    config = tmp_path / "config"
    output = config / "runtime" / "cad-controlled-20261008" / "new-run"
    output.parent.mkdir(parents=True)
    evaluation = tmp_path / "code-only-snapshot"
    evaluation.mkdir()
    with pytest.raises(SystemExit) as error:
        runner.main(["--source-run", str(frozen_case(tmp_path)), "--code-root", str(tmp_path),
                     "--config-root", str(config), "--evaluation-root", str(evaluation),
                     "--output", str(output), "--evaluate", "--online"])
    assert error.value.code == 2
    assert "evaluation-root must contain __dataset" in capsys.readouterr().err
    assert not output.exists()
