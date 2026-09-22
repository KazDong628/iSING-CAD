import json
from pathlib import Path

import pytest

from contour_agent.dataset import CALIBRATION_CASE, build_catalog, read_ocr, resolve_inside


def _source(root, case):
    origin = root / "origin"
    origin.mkdir(parents=True, exist_ok=True)
    (origin / f"{case}.jpg").write_bytes(b"image-placeholder")
    (origin / f"{case}.json").write_text(json.dumps({"meta": {"original_size": {"width": 100, "height": 100}}, "records": [{"id": "untrusted-id", "text": "R40", "box": [[0, 0], [20, 0], [20, 10], [0, 10]], "score": 0.9}]}), encoding="utf8")


def test_catalog_is_deterministic_and_keeps_unsupported(tmp_path):
    _source(tmp_path, CALIBRATION_CASE)
    _source(tmp_path, "other")
    (tmp_path / "origin" / "orphan.json").write_text('{"records":[]}', encoding="utf8")
    catalog = build_catalog(tmp_path)
    assert catalog == build_catalog(tmp_path)
    assert catalog["counts"]["total"] == 3
    assert catalog["counts"]["unsupported"] == 2
    assert catalog["counts"]["missing_gt"] == 3
    assert catalog["counts"]["missing_image"] == 1
    calibration = next(c for c in catalog["cases"] if c["id"] == CALIBRATION_CASE)
    assert calibration["split"] == "calibration"
    assert calibration["supported_template"] == "solid293-v1"
    assert all(c["split"] == "holdout" for c in catalog["cases"] if c != calibration)


@pytest.mark.parametrize("path", ["../secret", "origin/../../secret", "C:/outside", "/outside", "\\\\server\\outside"])
def test_paths_cannot_escape_root(tmp_path, path):
    with pytest.raises(ValueError):
        resolve_inside(tmp_path, path)


def test_manifest_cannot_redirect_reference_outside_root(tmp_path):
    _source(tmp_path, CALIBRATION_CASE)
    (tmp_path / "vis").mkdir()
    (tmp_path / "vis" / "splits.json").write_text(json.dumps({"records": [{"sample_id": CALIBRATION_CASE, "dxf": "../../secret.dxf"}]}), encoding="utf8")
    catalog = build_catalog(tmp_path)
    assert catalog["cases"][0]["gt_dxf"] is None
    assert "Rejected unsafe" in catalog["warnings"][0]


def test_ocr_assigns_ids_and_does_not_forward_metadata_paths(tmp_path):
    _source(tmp_path, "sample")
    ocr = read_ocr(tmp_path / "origin" / "sample.json")
    assert ocr["records"][0]["id"] == "r000"
    assert set(ocr["meta"]) == {"original_size", "box_coord_space"}


def test_ocr_rejects_non_finite_coordinates(tmp_path):
    _source(tmp_path, "sample")
    file = tmp_path / "origin" / "sample.json"
    data = json.loads(file.read_text())
    data["records"][0]["box"][0][0] = float("nan")
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="coordinate"):
        read_ocr(file)


def test_actual_dataset_inventory_and_ocr_validation():
    root = Path(__file__).resolve().parents[1] / "__dataset"
    if not root.is_dir():
        pytest.skip("Repository dataset not installed")
    catalog = build_catalog(root)
    assert catalog["counts"]["total"] == 50
    assert catalog["counts"]["paired"] == 50
    assert catalog["counts"]["gt_dxf"] == 44
    assert catalog["counts"]["archived_gt_only"] == 1
    assert catalog["counts"]["missing_gt"] == 5
    assert catalog["counts"]["holdout"] == 49
    assert sum(len(read_ocr(resolve_inside(root, case["ocr"]))["records"]) for case in catalog["cases"]) == 2754
