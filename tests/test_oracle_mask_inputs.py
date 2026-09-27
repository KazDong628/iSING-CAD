"""Oracle input checks use synthetic source/GT files, never the real dataset."""
from __future__ import annotations

import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from scripts.oracle_mask_inputs import load_oracle_mask_case


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    dataset = tmp_path / "__dataset"
    origin = dataset / "origin"
    gt = dataset / "GT"
    origin.mkdir(parents=True)
    gt.mkdir()
    source_image = origin / "case.jpg"
    Image.fromarray(np.full((30, 40), 255, dtype=np.uint8)).save(source_image)
    source_ocr = origin / "case.json"
    source_ocr.write_text(json.dumps({"meta": {"original_size": {"width": 40, "height": 30},
                                               "box_coord_space": "original_image"}, "records": []}))
    gt_source = gt / "reference.dxf"
    gt_source.write_bytes(b"synthetic reference bytes for hash verification")
    root = tmp_path / "runtime" / "segmentation"
    frozen = root / "gt-data-v3"
    for folder in (frozen / "images", frozen / "masks", frozen / "references", root / "data"):
        folder.mkdir(parents=True)
    base = root / "data" / "manifest.json"
    base.write_text("{}")
    prepared = frozen / "images" / "case.png"
    Image.fromarray(np.full((15, 20), 255, dtype=np.uint8)).save(prepared)
    mask = frozen / "masks" / "case.png"
    values = np.zeros((15, 20), dtype=np.uint8)
    values[2:13, 3:17] = 255
    Image.fromarray(values).save(mask)
    gt_sha = _sha(gt_source)
    reference = {"status": "ready", "topology_repaired": False,
                 "selection_ambiguous": False,
                 "source": {"path": str(gt_source), "member": None},
                 "source_sha256": gt_sha,
                 "polygons": [{"exterior": [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
                               "holes": []}]}
    (frozen / "references" / "case.json").write_text(json.dumps(reference))
    quality = {name: 1.0 for name in ("frame_inside_ratio", "edge_support",
                                      "hint_precision", "hint_coverage", "ambiguity_margin")}
    row = {"id": "case", "label_source": "registered_dxf_gt", "artifact_status": "ready",
           "trainable": True, "reference_status": "ready", "split": "test",
           "source_image": str(source_image), "source_image_sha256": _sha(source_image),
           "image_sha256": _sha(source_image), "source_ocr": str(source_ocr),
           "source_ocr_sha256": _sha(source_ocr), "image": str(prepared),
           "prepared_image_sha256": _sha(prepared), "mask": str(mask), "mask_sha256": _sha(mask),
           "foreground_pixels": 154, "original_size": {"width": 40, "height": 30},
           "prepared_size": {"width": 20, "height": 15},
           "reference_source": reference["source"],
           "registration": {"status": "accepted", "source_gt_sha256": gt_sha,
                            "failed_quality_checks": [], "transform": [[.5, 0, 0], [0, -.5, 15]],
                            "quality": quality}}
    manifest = {"format_version": "registered-dxf-gt-v1", "label_source": "registered_dxf_gt",
                "ground_truth_used": True, "split_claim": "development_only_not_blind_test",
                "dataset_root": str(dataset), "output_dir": str(frozen),
                "base_manifest_sha256": _sha(base),
                "quality_policy": {key + "_min": .5 for key in quality}, "cases": [row]}
    manifest_path = frozen / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, reference, mask, gt_source


def test_oracle_mask_is_frozen_with_original_ocr_coordinates(tmp_path):
    manifest, _, _, gt = _fixture(tmp_path)
    result = load_oracle_mask_case(manifest, "case", tmp_path / "replay" / "inputs")
    receipt = result["receipt"]
    assert receipt["oracle_mask"] is True and receipt["held_out"] is False
    assert receipt["calibration_or_development"] is True
    assert receipt["source_split_label"] == "test"
    assert receipt["original_image_size"] == {"width": 40, "height": 30}
    assert receipt["mask_size"] == {"width": 20, "height": 15}
    assert receipt["mask_geometry"]["connected_components"] == 1
    assert receipt["mask_geometry"]["holes"] == 0
    assert _sha(result["mask"]) == receipt["mask_sha256"]
    assert gt.name not in [path.name for path in result["mask"].parent.iterdir()]
    with pytest.raises(ValueError, match="immutable"):
        load_oracle_mask_case(manifest, "case", result["mask"].parent)


def test_oracle_rejects_changed_mask_or_uncopied_hole(tmp_path):
    manifest, reference, mask, _ = _fixture(tmp_path)
    mask.write_bytes(mask.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="GT mask SHA-256"):
        load_oracle_mask_case(manifest, "case", tmp_path / "bad-mask")
    assert not (tmp_path / "bad-mask").exists()

    manifest, reference, _, _ = _fixture(tmp_path / "other")
    reference["polygons"][0]["holes"] = [[[.2, .2], [.8, .2], [.8, .8], [.2, .2]]]
    (manifest.parent / "references" / "case.json").write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="uncopied holes"):
        load_oracle_mask_case(manifest, "case", tmp_path / "bad-hole")
    assert not (tmp_path / "bad-hole").exists()


def test_oracle_rejects_reference_change(tmp_path):
    manifest, _, _, gt = _fixture(tmp_path)
    gt.write_bytes(b"changed GT")
    with pytest.raises(ValueError, match="GT DXF bytes differ"):
        load_oracle_mask_case(manifest, "case", tmp_path / "bad-reference")
    assert not (tmp_path / "bad-reference").exists()
