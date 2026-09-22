"""Synthetic tests of registration, not a claim about dataset label accuracy."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from contour_agent.gt_registration import register_training_polygon


POLYGON = np.array([[0, 0], [65, 0], [65, 22], [31, 22], [31, 56], [0, 44], [0, 0]], float)


def make_source(tmp_path, matrix=None, shape=(280, 360), *, second_island=False):
    if matrix is None:
        matrix = np.array([[2.5, 0, 54], [0, -2.5, 222]], float)
    points = POLYGON @ matrix[:, :2].T+matrix[:, 2]
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    image = np.full(shape, 255, np.uint8)
    hatch = np.zeros(shape, np.uint8)
    for x in range(-shape[0], shape[1], 11):
        cv2.line(hatch, (x, 0), (x+shape[0], shape[0]-1), 255, 1)
    image[(hatch > 0) & (mask > 0)] = 0
    cv2.polylines(image, [np.rint(points).astype(np.int32)], True, 0, 2)
    if second_island:
        cv2.rectangle(mask, (270, 30), (325, 94), 255, -1)
        cv2.rectangle(image, (270, 30), (325, 94), 0, 2)
    path = tmp_path/"drawing.png"
    Image.fromarray(image).save(path)
    return path, mask, matrix, points


def test_supplied_calibration_kept_and_writes_original_mask(tmp_path):
    path, mask, matrix, points = make_source(tmp_path)
    result = register_training_polygon(path, POLYGON, tmp_path/"labels", foreground_hint=mask,
                                       initial_transforms=[{"matrix": matrix, "source": "synthetic_calibration"}])
    assert result["status"] == "registration_candidate"
    assert not result["reviewed"] and not result["geometry_changed"]
    supplied = [c for c in result["candidates"] if c["source"] == "synthetic_calibration"]
    assert {c["variant"] for c in supplied} == {"unoptimized", "refined"}
    np.testing.assert_allclose(next(c for c in supplied if c["variant"] == "unoptimized")["transform_2x3"], matrix)
    prediction = np.asarray(result["registered_polyline_px"])
    assert np.max(np.linalg.norm(prediction-points, axis=1)) < 2
    with Image.open(result["artifacts"]["mask"]) as saved:
        assert saved.size == (360, 280)
        assert np.unique(saved).tolist() == [0, 255]
    assert Path(result["artifacts"]["overlay"]).is_file()
    for key in ("frame_inside_ratio", "edge_support", "hint_precision", "hint_coverage", "score", "ambiguity_margin"):
        assert 0 <= result["quality"][key] <= 1


def test_rotation_reflection_from_foreground_initializer(tmp_path):
    matrix = np.array([[0., -2.6, 230], [-2.6, 0, 235]])
    path, mask, _, points = make_source(tmp_path, matrix)
    result = register_training_polygon(path, POLYGON, foreground_hint=mask)
    predicted = np.asarray(result["registered_polyline_px"])
    assert np.max(np.linalg.norm(predicted-points, axis=1)) < 3
    assert result["quality"]["best_component_hint_iou"] > .95


def test_resize_mapping_and_fixed_affine_are_preserved(tmp_path):
    matrix = np.array([[6.0, 1.3, 75.25], [0, -5.8, 495.5]])
    path, mask, _, _ = make_source(tmp_path, matrix, (601, 877))
    result = register_training_polygon(path, POLYGON, tmp_path/"labels", foreground_hint=mask,
                                       initial_transforms=[matrix], max_dimension=320)
    fixed = [c for c in result["candidates"] if c["source"] == "provided_0"]
    assert len(fixed) == 1 and fixed[0]["variant"] == "unoptimized"
    assert fixed[0]["transform_kind"] == "provided_affine_fixed"
    np.testing.assert_allclose(fixed[0]["transform_2x3"], matrix, atol=1e-10)
    assert result["similarity_transform"] is None
    for candidate in result["candidates"]:
        if candidate["transform_kind"] == "similarity":
            linear = np.asarray(candidate["transform_2x3"])[:, :2]
            gram = linear.T@linear
            np.testing.assert_allclose(gram, np.eye(2)*np.trace(gram)/2, atol=1e-9)
    with Image.open(result["artifacts"]["mask"]) as saved:
        assert saved.size == (877, 601)


def test_multiple_hints_do_not_force_union_registration(tmp_path):
    path, mask, _, _ = make_source(tmp_path, second_island=True)
    result = register_training_polygon(path, POLYGON, foreground_hint=mask)
    assert result["quality"]["best_component_hint_iou"] > .95
    assert .6 < result["quality"]["hint_coverage"] < .95
    assert any("hint_1" in c["source"] for c in result["candidates"])


@pytest.mark.parametrize("colour", [0, 255])
def test_blank_or_no_initializer_is_explicit(tmp_path, colour):
    path = tmp_path/"blank.png"
    Image.new("L", (200, 200), colour).save(path)
    result = register_training_polygon(path, POLYGON, initial_transforms=[[[1, 0, 30], [0, 1, 30]]])
    assert result["status"] == "needs_review"
    assert result["registered_polyline_px"] == []
    assert result["transform_2x3"] is None
    assert result["candidates"] == []


def test_hints_optional_with_known_calibration(tmp_path):
    path, _, matrix, _ = make_source(tmp_path)
    result = register_training_polygon(path, POLYGON, initial_transforms=[matrix])
    assert result["quality"]["edge_support"] > .95
    assert result["quality"]["hint_precision"] is None
    assert result["quality"]["hint_coverage"] is None


def test_partial_foreground_and_known_calibration(tmp_path):
    path, mask, matrix, points = make_source(tmp_path)
    mask[:, 125:] = 0
    mask_before, polygon_before = mask.copy(), POLYGON.copy()
    result = register_training_polygon(path, POLYGON, foreground_hint=mask, initial_transforms=[matrix])
    assert result["quality"]["edge_support"] > .95
    assert np.max(np.linalg.norm(np.asarray(result["registered_polyline_px"])-points, axis=1)) < 2
    np.testing.assert_array_equal(mask, mask_before)
    np.testing.assert_array_equal(POLYGON, polygon_before)


@pytest.mark.parametrize("polygon", [POLYGON[:-1], [[0, 0], [1, 1], [0, 1], [1, 0], [0, 0]],
                                     [[0, 0], [1, 0], [1, float("nan")], [0, 0]]])
def test_invalid_reference_is_not_repaired(tmp_path, polygon):
    path, _, _, _ = make_source(tmp_path)
    with pytest.raises(ValueError):
        register_training_polygon(path, polygon)


@pytest.mark.parametrize("matrix", [[[1, 0], [0, 1]], [[0, 0, 2], [0, 0, 3]],
                                    [[1, 0, float("inf")], [0, 1, 0]]])
def test_invalid_initial_matrix_rejected(tmp_path, matrix):
    path, _, _, _ = make_source(tmp_path)
    with pytest.raises(ValueError):
        register_training_polygon(path, POLYGON, initial_transforms=[matrix])


def test_provided_reflection_respects_explicit_disable(tmp_path):
    path, _, matrix, _ = make_source(tmp_path)
    result = register_training_polygon(path, POLYGON, initial_transforms=[matrix], allow_reflection=False)
    assert result["status"] == "needs_review"
    assert any("reflection is disabled" in issue for issue in result["issues"])


@pytest.mark.parametrize("hint_kind", ["none", "tiny_wrong_fragment"])
def test_global_source_edge_search_recovers_pose_without_reliable_hint(tmp_path, monkeypatch, hint_kind):
    matrix = np.array([[0., -2.6, 230], [-2.6, 0, 235]])
    path, mask, _, points = make_source(tmp_path, matrix)
    # Dimension-like lines are deliberate distractors; the full boundary must
    # select a coherent pose, independent of a local fragment's bounding box.
    source = np.asarray(Image.open(path)).copy()
    cv2.line(source, (5, 15), (340, 15), 0, 1)
    cv2.line(source, (8, 5), (8, 260), 0, 1)
    Image.fromarray(source).save(path)
    hint = None
    if hint_kind == "tiny_wrong_fragment":
        hint = np.zeros_like(mask)
        hint[15:32, 20:44] = 255
    monkeypatch.setattr("contour_agent.dimension_evidence.estimate_scale",
                        lambda image, document: {"status": "resolved", "pixels_per_mm": 2.6})
    result = register_training_polygon(path, POLYGON, foreground_hint=hint, ocr_document={})
    assert result["global_search"]["candidate_count"] > 0
    assert result["candidates"][0]["source"].startswith("global_edge:")
    assert np.max(np.linalg.norm(np.asarray(result["registered_polyline_px"])-points, axis=1)) < 3
    linear = np.asarray(result["transform_2x3"])[:, :2]
    np.testing.assert_allclose(linear.T@linear, np.eye(2)*abs(np.linalg.det(linear)), atol=1e-6)


def test_global_search_does_not_invent_a_resolved_scale(tmp_path, monkeypatch):
    path, _, _, _ = make_source(tmp_path)
    monkeypatch.setattr("contour_agent.dimension_evidence.estimate_scale",
                        lambda image, document: {"status": "ambiguous", "pixels_per_mm": None})
    result = register_training_polygon(path, POLYGON, ocr_document={})
    assert result["global_search"]["candidate_count"] == 0
    assert result["status"] == "needs_review"
