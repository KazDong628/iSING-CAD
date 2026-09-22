"""Synthetic shared-background registration checks; no real GT is needed."""
from hashlib import sha256
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

from contour_agent.gt_overlay_hint import overlay_hint


def fixture_files(tmp_path, *, unrelated=False, colored=True):
    source = tmp_path / "source.png"
    rng = np.random.default_rng(29)
    texture = rng.integers(30, 240, (480, 640), dtype=np.uint8)
    rgb = np.repeat(texture[:, :, None], 3, axis=2)
    Image.fromarray(rgb).save(source)
    package = tmp_path / "GT/case"
    package.mkdir(parents=True)
    reference = package / "case_main_profile.dxf"
    reference.write_text("not opened by the overlay helper", encoding="utf-8")
    if unrelated:
        texture = np.random.default_rng(310).integers(30, 240, (480, 640), dtype=np.uint8)
        rgb = np.repeat(texture[:, :, None], 3, axis=2)
    # The official overlay background is a crop: its mapping to source pixels
    # must include translation rather than assuming equal image dimensions.
    overlay = rgb[40:430, 60:600].copy()
    cv2.rectangle(overlay, (70, 60), (360, 250), (240, 10, 30) if colored else (0, 0, 0), 3)
    overlay_path = package / "case_main_overlay.png"
    Image.fromarray(overlay).save(overlay_path)
    return source, reference, overlay_path


def test_cropped_shared_background_maps_hint_to_source_and_records_provenance(tmp_path):
    source, reference, overlay = fixture_files(tmp_path)
    before = {path: path.read_bytes() for path in (source, reference, overlay)}
    result = overlay_hint({"path": str(reference), "member": None}, source, tmp_path / "output")
    assert result is not None
    hint, receipt = result["hint"], result["provenance"]
    assert hint.shape == (480, 640) and hint.dtype == np.uint8
    assert set(np.unique(hint)) == {0, 255}
    expected = np.zeros(hint.shape, bool); expected[100:291, 130:421] = True
    predicted = hint > 0
    assert np.count_nonzero(predicted & expected) / np.count_nonzero(predicted | expected) > .94
    assert np.allclose(receipt["affine_overlay_to_source"], [[1, 0, 60], [0, 1, 40]], atol=.4)
    assert receipt["quality"]["inliers"] >= 20 and receipt["quality"]["inlier_ratio"] >= .5
    assert receipt["localization_only"] and receipt["final_label_source"] == "rendered_gt_dxf_not_overlay_pixels"
    assert receipt["source_image_sha256"] == sha256(before[source]).hexdigest()
    assert receipt["overlay_sha256"] == sha256(before[overlay]).hexdigest()
    assert receipt["registration_reviewed"] is False
    assert (tmp_path / "output/overlay-hint-mask.png").is_file()
    assert before == {path: path.read_bytes() for path in before}


def test_unrelated_background_cannot_supply_a_hint(tmp_path):
    source, reference, _ = fixture_files(tmp_path, unrelated=True)
    assert overlay_hint({"path": str(reference)}, source, tmp_path / "output") is None


def test_black_drawing_lines_are_not_treated_as_colored_gt(tmp_path):
    source, reference, _ = fixture_files(tmp_path, colored=False)
    assert overlay_hint({"path": str(reference)}, source) is None


def test_zoom_or_labeled_diagnostics_are_not_selected(tmp_path):
    source, reference, overlay = fixture_files(tmp_path)
    overlay.rename(overlay.with_name("case_main_overlay_labeled_zoom.png"))
    assert overlay_hint({"path": str(reference)}, source) is None


def test_zip_package_overlay_is_read_without_extracting(tmp_path):
    source, reference, overlay = fixture_files(tmp_path)
    archive = tmp_path / "package.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.write(reference, "inside/main_profile.dxf")
        package.write(overlay, "inside/main_overlay.png")
    before = set(tmp_path.rglob("*"))
    result = overlay_hint({"path": str(archive), "member": "inside/main_profile.dxf"}, source)
    assert result is not None
    assert result["provenance"]["overlay_source"]["member"] == "inside/main_overlay.png"
    assert set(tmp_path.rglob("*")) == before
