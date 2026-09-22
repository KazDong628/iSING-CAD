"""Source-only weak labels, frozen provenance, and split-leakage regressions."""
import copy
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image
import pytest

import contour_agent.segmentation_data as data


def source(dataset, case_id, *, size=(120, 80), pixels=None):
    origin = dataset / "origin"
    origin.mkdir(parents=True, exist_ok=True)
    path = origin / (case_id + ".png")
    if pixels is None:
        pixels = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
        cv2.rectangle(pixels, (size[0] // 4, size[1] // 4), (size[0] * 3 // 4, size[1] * 3 // 4), (0, 0, 0), 2)
    Image.fromarray(pixels).save(path)
    path.with_suffix(".json").write_text(json.dumps({"meta": {}, "records": []}), encoding="utf-8")
    return path


def rectangle(path):
    with Image.open(path) as image:
        w, h = image.size
    return {"polyline_px": [[w / 4, h / 4], [w * .75, h / 4], [w * .75, h * .75],
                            [w / 4, h * .75], [w / 4, h / 4]],
            "status": "needs_review", "issues": ["Source proposal remains a weak label."]}


def forbid(*args, **kwargs):
    pytest.fail("Unexpected extraction or forbidden file access")


def test_prepared_pair_resize_binary_mask_and_read_only_sources(tmp_path, monkeypatch):
    dataset, output = tmp_path / "dataset", tmp_path / "prepared"
    original = source(dataset, data.CALIBRATION_CASE, size=(3200, 1200))
    for folder in ("GT", "vis"):
        (dataset / folder).mkdir()
        (dataset / folder / "never-read.txt").write_text("No reference data may enter labels.", encoding="utf-8")
    before = {str(path): path.read_bytes() for path in dataset.rglob("*") if path.is_file()}
    actual_open, forbidden_accesses = Path.open, []
    def guarded_open(path, *args, **kwargs):
        if dataset / "GT" in path.parents or dataset / "vis" in path.parents:
            forbidden_accesses.append(path)
            raise AssertionError("Reference read")
        return actual_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(data, "extract_main_profile", rectangle)
    monkeypatch.setattr(data, "extract_unhatched", forbid)
    manifest = data.prepare_dataset(dataset, output)
    row = manifest["cases"][0]
    assert not forbidden_accesses
    assert manifest["split_claim"] == "development_only_not_blind_test"
    assert not manifest["ground_truth_used"] and not manifest["previous_predictions_used"]
    assert row["split"] == "train" and row["artifact_status"] == "ready" and row["trainable"]
    assert row["label_source"] == "source_heuristic" and row["reviewed"] is False
    assert row["source_image_sha256"] == row["image_sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert row["original_size"] == {"width": 3200, "height": 1200}
    assert row["prepared_size"] == {"width": 1536, "height": 576}
    assert row["resize_scale"] == {"x": .48, "y": .48}
    assert all(Path(row[key]).is_relative_to(output) for key in ("image", "mask"))
    with Image.open(row["image"]) as image, Image.open(row["mask"]) as mask:
        assert image.size == mask.size == (1536, 576)
        values = np.asarray(mask)
        assert set(np.unique(values)) == {0, 255}
        assert values[288, 768] == 255 and values[0, 0] == 0
    monkeypatch.setattr(Path, "open", actual_open)
    assert before == {str(path): path.read_bytes() for path in dataset.rglob("*") if path.is_file()}


def test_fallback_only_for_empty_primary_and_failed_cases_remain(tmp_path, monkeypatch):
    dataset, output = tmp_path / "dataset", tmp_path / "prepared"
    for case_id in ("primary", "fallback", "failed"):
        source(dataset, case_id)
    calls = []
    def primary(path):
        return rectangle(path) if Path(path).stem == "primary" else {"polyline_px": []}
    def fallback(path, document):
        calls.append(Path(path).stem)
        assert document["records"] == []
        return rectangle(path) if Path(path).stem == "fallback" else {"polyline_px": [], "issues": ["No hatch evidence."]}
    monkeypatch.setattr(data, "extract_main_profile", primary)
    monkeypatch.setattr(data, "extract_unhatched", fallback)
    manifest = data.prepare_dataset(dataset, output)
    rows = {row["id"]: row for row in manifest["cases"]}
    assert sorted(calls) == ["failed", "fallback"]
    assert manifest["summary"]["total"] == 3 and manifest["summary"]["failed"] == 1
    failed = rows["failed"]
    assert failed["artifact_status"] == "failed" and not failed["trainable"]
    assert failed["mask"] is None and Path(failed["image"]).is_file()
    assert failed["group"] and failed["split"] in {"train", "val", "test"}
    assert not rows["primary"]["fallback_used"] and rows["fallback"]["fallback_used"]
    assert all(row["reviewed"] is False for row in rows.values())


def test_exact_near_duplicate_and_transitive_groups_stay_together():
    rows = [
        {"id": data.CALIBRATION_CASE, "image_sha256": "a", "phash": "0000000000000000", "aspect_ratio": 1.0},
        {"id": "near", "image_sha256": "b", "phash": "000000000000000f", "aspect_ratio": 1.01},
        {"id": "transitive", "image_sha256": "c", "phash": "00000000000000ff", "aspect_ratio": 1.02},
        {"id": "other_aspect", "image_sha256": "d", "phash": "0000000000000000", "aspect_ratio": 1.2},
        {"id": "same_bytes", "image_sha256": "a", "phash": None},
    ]
    data._assign_groups(rows, 42)
    assert len({rows[index]["group"] for index in (0, 1, 2, 4)}) == 1
    assert all(rows[index]["split"] == "train" for index in (0, 1, 2, 4))
    assert rows[3]["group"] != rows[0]["group"]


def test_deterministic_group_splits_roughly_seventy_fifteen_fifteen():
    rng = random.Random(4)
    rows = [{"id": data.CALIBRATION_CASE if i == 0 else f"case-{i:02}",
             "image_sha256": str(i), "phash": f"{rng.getrandbits(64):016x}", "aspect_ratio": 1.4}
            for i in range(50)]
    a, b, c = copy.deepcopy(rows), copy.deepcopy(rows), copy.deepcopy(rows)
    data._assign_groups(a, 42)
    data._assign_groups(b, 42)
    data._assign_groups(c, 43)
    assert a == b and a != c
    assert len({row["group"] for row in a}) == 50
    counts = {split: sum(row["split"] == split for row in a) for split in ("train", "val", "test")}
    assert counts["train"] == 35 and sorted([counts["val"], counts["test"]]) == [7, 8]
    assert a[0]["split"] == c[0]["split"] == "train"


def test_actual_phash_groups_reencoded_resized_same_image(tmp_path, monkeypatch):
    dataset, output = tmp_path / "dataset", tmp_path / "prepared"
    pixels = np.random.default_rng(6).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    original = source(dataset, data.CALIBRATION_CASE, pixels=pixels)
    duplicate = original.with_name("duplicate.jpg")
    with Image.open(original) as image:
        image.resize((192, 128), Image.Resampling.LANCZOS).save(duplicate, quality=97)
    monkeypatch.setattr(data, "extract_main_profile", rectangle)
    manifest = data.prepare_dataset(dataset, output)
    a, b = manifest["cases"]
    assert a["image_sha256"] != b["image_sha256"]
    assert (int(a["phash"], 16) ^ int(b["phash"], 16)).bit_count() <= 4
    assert a["group"] == b["group"] and a["split"] == b["split"] == "train"


def test_reuse_frozen_manifest_preserves_separate_reviews(tmp_path, monkeypatch):
    dataset, output = tmp_path / "dataset", tmp_path / "prepared"
    source(dataset, "source")
    monkeypatch.setattr(data, "extract_main_profile", rectangle)
    first = data.prepare_dataset(dataset, output, seed=7)
    manifest_bytes = (output / "manifest.json").read_bytes()
    reviews = output / "reviews.json"
    reviews.write_text('{"reviewed": "independent review ledger"}', encoding="utf-8")
    original_reviews = reviews.read_bytes()
    monkeypatch.setattr(data, "extract_main_profile", forbid)
    monkeypatch.setattr(data, "extract_unhatched", forbid)
    assert data.prepare_dataset(dataset, output, seed=7) == first
    assert (output / "manifest.json").read_bytes() == manifest_bytes
    assert reviews.read_bytes() == original_reviews


@pytest.mark.parametrize("change", ["seed", "source", "ocr", "mask", "missing_image"])
def test_frozen_manifest_rejects_changed_sources_configuration_or_artifacts(tmp_path, monkeypatch, change):
    dataset, output = tmp_path / "dataset", tmp_path / "prepared"
    original = source(dataset, "source")
    monkeypatch.setattr(data, "extract_main_profile", rectangle)
    manifest = data.prepare_dataset(dataset, output)
    before = (output / "manifest.json").read_bytes()
    if change == "source":
        Image.new("RGB", (120, 80), "black").save(original)
    elif change == "ocr":
        original.with_suffix(".json").write_text('{"records":[],"meta":{"revision":2}}', encoding="utf-8")
    elif change == "mask":
        Path(manifest["cases"][0]["mask"]).write_bytes(b"changed")
    elif change == "missing_image":
        Path(manifest["cases"][0]["image"]).unlink()
    monkeypatch.setattr(data, "extract_main_profile", forbid)
    with pytest.raises(ValueError, match="Frozen"):
        data.prepare_dataset(dataset, output, seed=43 if change == "seed" else 42)
    assert (output / "manifest.json").read_bytes() == before


def test_corrupt_source_is_retained_as_failed_not_trainable(tmp_path, monkeypatch):
    dataset, output = tmp_path / "dataset", tmp_path / "prepared"
    source(dataset, "broken").write_bytes(b"not an image")
    monkeypatch.setattr(data, "extract_main_profile", forbid)
    manifest = data.prepare_dataset(dataset, output)
    row = manifest["cases"][0]
    assert row["artifact_status"] == "failed" and row["trainable"] is False
    assert row["image"] is None and row["mask"] is None
    assert row["source_image_sha256"] and row["group"] and row["split"]


def test_output_cannot_modify_sources_or_overwrite_partial_directory(tmp_path):
    dataset, output = tmp_path / "dataset", tmp_path / "prepared"
    source(dataset, "source")
    with pytest.raises(ValueError, match="read-only"):
        data.prepare_dataset(dataset, dataset / "labels")
    output.mkdir()
    partial = output / "partial.png"
    partial.write_bytes(b"preserve me")
    with pytest.raises(ValueError, match="not empty"):
        data.prepare_dataset(dataset, output)
    assert partial.read_bytes() == b"preserve me"


def test_real_source_extractor_builds_weak_mask_without_reference(tmp_path):
    dataset, output = tmp_path / "dataset", tmp_path / "prepared"
    pixels = np.full((420, 620, 3), 255, dtype=np.uint8)
    for offset in range(-420, 1040, 24):
        layer = np.full_like(pixels, 255)
        cv2.line(layer, (offset, 419), (offset + 420, 0), (0, 0, 0), 1)
        pixels[100:330, 140:490] = np.minimum(pixels[100:330, 140:490], layer[100:330, 140:490])
    cv2.rectangle(pixels, (140, 100), (490, 330), (0, 0, 0), 3)
    source(dataset, "synthetic-hatch", pixels=pixels)
    manifest = data.prepare_dataset(dataset, output)
    row = manifest["cases"][0]
    assert row["trainable"] and not row["reviewed"]
    with Image.open(row["mask"]) as image:
        predicted = np.asarray(image) > 0
    expected = np.zeros(predicted.shape, dtype=bool)
    expected[100:331, 140:491] = True
    assert np.count_nonzero(predicted & expected) / np.count_nonzero(predicted | expected) > .9
