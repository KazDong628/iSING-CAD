import hashlib

import ezdxf
import numpy as np
import pytest

from contour_agent.autonomous_evaluation import (
    evaluate_autonomous_artifact, summarize_autonomous_evaluation,
)


def polygon(path, matrix=None, offset=(0, 0), scale=1., units=4, close=True):
    doc = ezdxf.new()
    doc.units = units
    points = np.array([[0., 0.], [21., 0.], [21., 7.], [14., 7.], [14., 12.], [0., 15.]]) * scale
    if matrix is not None:
        points = points @ np.asarray(matrix).T
    points += np.asarray(offset)
    doc.modelspace().add_lwpolyline(points.tolist(), close=close, dxfattribs={"layer": "MAIN_PROFILE"})
    doc.saveas(path)
    return path


def test_arbitrary_origin_is_registered_but_not_engineering_verified(tmp_path):
    reference = polygon(tmp_path / "reference.dxf")
    prediction = polygon(tmp_path / "prediction.dxf", offset=(132., -74.))
    before = hashlib.sha256(prediction.read_bytes()).hexdigest()
    result = evaluate_autonomous_artifact(prediction, reference)
    assert result["status"] == "compared"
    assert result["geometry_valid"]
    assert result["reference_within_0_1mm"]
    assert result["registered_metrics"]["max_error_mm"] < 1e-7
    assert result["direct_metrics"]["max_error_mm"] > 50
    assert result["alignment_kind"] == "shape_diagnostic"
    assert result["transform"]["scale"] == 1
    assert result["transform"]["candidates_evaluated"] == 8
    assert not result["engineering_verified"]
    assert hashlib.sha256(prediction.read_bytes()).hexdigest() == before


def test_axes_and_reflection_are_disclosed_not_hidden(tmp_path):
    reference = polygon(tmp_path / "reference.dxf")
    prediction = polygon(tmp_path / "prediction.dxf", matrix=[[0, 1], [1, 0]], offset=(90., -30.))
    result = evaluate_autonomous_artifact(prediction, reference)
    assert result["reference_within_0_1mm"]
    assert result["transform"]["axis_searched"]
    assert result["transform"]["matrix"] == [[0., 1.], [1., 0.]]
    assert not result["engineering_verified"]


def test_scale_error_cannot_be_removed_by_registration(tmp_path):
    reference = polygon(tmp_path / "reference.dxf")
    prediction = polygon(tmp_path / "prediction.dxf", scale=1.05, offset=(40., 60.))
    result = evaluate_autonomous_artifact(prediction, reference)
    assert not result["reference_within_0_1mm"]
    assert result["registered_metrics"]["max_error_mm"] > .2
    assert result["registered_metrics"]["length_error_fraction"] == pytest.approx(.05)
    assert result["transform"]["scale"] == 1
    assert result["scale_fit_permitted"] is False


def test_unitless_prediction_never_receives_physical_score(tmp_path):
    reference = polygon(tmp_path / "reference.dxf")
    prediction = polygon(tmp_path / "prediction.dxf", units=0)
    result = evaluate_autonomous_artifact(prediction, reference)
    assert result["artifact_completed"] and result["geometry_valid"]
    assert not result["scaled_mm"]
    assert result["status"] == "unscaled_prediction"
    assert not result["reference_compared"]
    assert not result["reference_within_0_1mm"]


def test_reference_inches_converted_from_declared_units(tmp_path):
    reference = polygon(tmp_path / "reference_inches.dxf", scale=1 / 25.4, units=1)
    prediction = polygon(tmp_path / "prediction.dxf")
    result = evaluate_autonomous_artifact(prediction, reference, declared_axis_transform=[[1, 0], [0, 1]], declared_translation_mm=[0, 0])
    assert result["reference_within_0_1mm"]
    assert result["reference_info"]["millimetre_conversion_factor"] == pytest.approx(25.4)
    assert result["alignment_kind"] == "declared_alignment"
    assert result["transform"]["candidates_evaluated"] == 1


def test_declared_alignment_is_not_optimized_away(tmp_path):
    reference = polygon(tmp_path / "reference.dxf")
    prediction = polygon(tmp_path / "prediction.dxf", offset=(.4, 0.))
    result = evaluate_autonomous_artifact(prediction, reference, declared_axis_transform=[[1, 0], [0, 1]], declared_translation_mm=[0., 0.], tolerance_mm=1.)
    assert result["reference_within_tolerance"]
    assert not result["reference_within_0_1mm"]
    assert not result["transform"]["translation_fitted"]
    assert result["transform"]["translation_mm"] == [0., 0.]


def test_axis_declarations_cannot_smuggle_scale(tmp_path):
    with pytest.raises(ValueError, match="scale"):
        evaluate_autonomous_artifact(tmp_path / "none", None, declared_axis_transform=[[2, 0], [0, 2]])


def test_open_or_partial_geometry_cannot_claim_complete_pass(tmp_path):
    open_path = polygon(tmp_path / "open.dxf", close=False)
    result = evaluate_autonomous_artifact(open_path, open_path)
    assert not result["geometry_valid"]
    assert not result["reference_within_0_1mm"]
    partial = polygon(tmp_path / "reference_scored_only.dxf")
    result = evaluate_autonomous_artifact(partial, partial)
    assert result["reference_scope"] == "partial_profile"
    assert not result["reference_within_0_1mm"]


def test_missing_reference_preserves_artifact_completion(tmp_path):
    prediction = polygon(tmp_path / "prediction.dxf")
    result = evaluate_autonomous_artifact(prediction, None)
    assert result["status"] == "missing_reference"
    assert result["artifact_completed"] and result["scaled_mm"] and result["geometry_valid"]
    assert not result["reference_compared"]


def test_reflection_preserves_trimmed_arc_geometry(tmp_path):
    reference = tmp_path / "reference.dxf"
    prediction = tmp_path / "prediction.dxf"
    for path, mirrored in ((reference, False), (prediction, True)):
        doc = ezdxf.new()
        doc.units = 4
        doc.modelspace().add_arc((0, 0), 10, 0 if not mirrored else 180, 180 if not mirrored else 360)
        doc.modelspace().add_line((-10, 0), (10, 0))
        doc.saveas(path)
    result = evaluate_autonomous_artifact(prediction, reference, declared_axis_transform=[[1, 0], [0, -1]], declared_translation_mm=[0, 0])
    assert result["reference_within_0_1mm"]
    assert result["registered_metrics"]["max_error_mm"] < 1e-8


def test_generic_engine_is_not_gated_by_old_template_support():
    catalog = {"cases": [{"id": str(i), "supported_template": None, "split": "holdout"} for i in range(50)]}
    comparison = {"artifact_completed": True, "geometry_valid": True, "scaled_mm": True, "reference_compared": True, "reference_within_0_1mm": True, "alignment_kind": "shape_diagnostic"}
    rows = [
        {"case_id": "0", "attempted": True, "status": "completed", "comparison": comparison},
        {"case_id": "1", "attempted": True, "status": "completed", "manual_confirmation": {"actor": "user"}, "comparison": comparison},
        {"case_id": "2", "attempted": True, "status": "failed"},
    ]
    summary = summarize_autonomous_evaluation(catalog, rows)
    assert summary["total"] == 50
    assert summary["attempted"] == 3
    assert summary["auto_generated"] == 1
    assert summary["reference_within_0_1mm"] == 2
    assert summary["autonomous_reference_within_0_1mm"] == 1
    assert summary["manual_interventions"] == 1
    assert summary["not_attempted"] == 47
    assert summary["autonomous_reference_within_0_1mm_rate"] == .02
    assert not summary["template_gate_used"]
    assert not summary["engineering_verified"]


def test_duplicate_or_unknown_cases_cannot_change_denominator():
    catalog = {"cases": [{"id": "known"}]}
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_autonomous_evaluation(catalog, [{"case_id": "known"}, {"case_id": "known"}])
    with pytest.raises(ValueError, match="outside"):
        summarize_autonomous_evaluation(catalog, [{"case_id": "other"}])
