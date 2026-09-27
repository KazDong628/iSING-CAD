import hashlib
import json
import math

import ezdxf
import pytest

from contour_agent.dxf_comparison import audit_dxf
from scripts.evaluate_oracle_mask_run import _single_cycle_area, evaluate_oracle_mask_run


def _rectangle(path, *, origin=(0, 0), split=False, units=4):
    document = ezdxf.new()
    document.units = units
    ox, oy = origin
    points = [(ox, oy), (ox + 10, oy), (ox + 10, oy + 5), (ox, oy + 5)]
    for start, end in zip(points, points[1:] + points[:1]):
        middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        for left, right in (((start, middle), (middle, end)) if split else ((start, end),)):
            document.modelspace().add_line(left, right, dxfattribs={"layer": "MAIN"})
    document.saveas(path)


def _fixture(tmp_path, *, prediction=True, split=False, mismatched_hash=False):
    dataset = tmp_path / "dataset"
    (dataset / "origin").mkdir(parents=True)
    (dataset / "GT").mkdir()
    (dataset / "origin" / "sample-main.jpg").write_bytes(b"source image")
    gt = dataset / "GT" / "sample-main_main_profile.dxf"
    _rectangle(gt)
    run = tmp_path / "run"
    (run / "inputs").mkdir(parents=True)
    (run / "inputs" / "oracle-mask.png").write_bytes(b"oracle mask")
    (run / "after").mkdir()
    if prediction:
        _rectangle(run / "after" / "drawing.dxf", origin=(100, -30), split=split)
    gt_hash = hashlib.sha256(gt.read_bytes()).hexdigest()
    (run / "run-manifest.json").write_text(json.dumps({"case_id": "sample-main",
        "oracle_mask_conditioned": True,
        "source_gt_sha256": "0" * 64 if mismatched_hash else gt_hash}), encoding="utf-8")
    return run, dataset


def test_missing_prediction_never_reads_gt_inventory(tmp_path, monkeypatch):
    run, dataset = _fixture(tmp_path, prediction=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("GT inventory must not be opened")

    monkeypatch.setattr("scripts.evaluate_oracle_mask_run.build_catalog", forbidden)
    report = evaluate_oracle_mask_run(run, dataset_root=dataset)
    assert report["status"] == "missing_prediction"
    assert report["reference_opened"] is False
    assert not (run / "evaluation" / "comparison.json").exists()


def test_oracle_report_separates_shape_registration_from_native_coordinates(tmp_path):
    run, dataset = _fixture(tmp_path)
    report = evaluate_oracle_mask_run(run, dataset_root=dataset)
    assert report["status"] == "compared"
    assert report["held_out"] is False
    assert report["provenance"]["mask_and_scoring_reference_same_bytes"] is True
    assert report["objects"]["prediction_filtered_types"] == {"LINE": 4}
    assert report["connectivity"]["prediction"]["closed_endpoint_degree"]
    assert report["geometry"]["prediction_area"]["area_mm2"] == pytest.approx(50)
    assert report["geometry"]["absolute_area_difference_mm2"] == pytest.approx(0)
    assert report["checks"]["registered_shape_within_0_1mm"]
    assert not report["checks"]["same_native_coordinate_frame_within_0_1mm"]
    assert report["checks"]["all_native_primitive_parameters_match_after_registration"]
    assert report["checks"]["matched_primitive_adjacency_equal"]
    assert report["primitive_matching"]["matched_count"] == 4
    assert (run / "evaluation" / "comparison.json").is_file()
    assert (run / "evaluation" / "overlay.svg").is_file()


def test_oracle_report_does_not_call_fragments_matching_primitives(tmp_path):
    run, dataset = _fixture(tmp_path, split=True)
    report = evaluate_oracle_mask_run(run, dataset_root=dataset)
    assert report["objects"]["prediction_filtered_count"] == 8
    assert report["objects"]["reference_filtered_count"] == 4
    assert report["checks"]["registered_shape_within_0_1mm"]
    assert not report["checks"]["all_native_primitive_parameters_match_after_registration"]
    assert report["checks"]["matched_primitive_adjacency_equal"] is None
    assert report["primitive_matching"]["status_counts"] == {"possible_fragment_coverage": 8}


def test_oracle_source_hash_mismatch_prevents_reference_comparison(tmp_path):
    run, dataset = _fixture(tmp_path, mismatched_hash=True)
    report = evaluate_oracle_mask_run(run, dataset_root=dataset)
    assert report["status"] == "reference_hash_mismatch"
    assert report["reference_compared"] is False
    assert not (run / "evaluation" / "comparison.json").exists()


def test_stage_diagnostics_preserve_predictions_and_expose_fragmentation(tmp_path):
    run, dataset = _fixture(tmp_path)
    baseline = run / "after" / "baseline-drawing.dxf"
    _rectangle(baseline, origin=(100, -30), split=True)
    before = {p: p.read_bytes() for p in (baseline, run / "after" / "drawing.dxf")}
    report = evaluate_oracle_mask_run(run, dataset_root=dataset)
    stages = {row["stage"]: row for row in report["stage_comparison"]}
    assert stages["initial_cad"]["checks"]["registered_shape_within_0_1mm"]
    assert not stages["initial_cad"]["checks"]["all_native_primitive_parameters_match_after_registration"]
    assert stages["initial_cad"]["objects"]["prediction_filtered_count"] == 8
    assert stages["published"]["objects"]["prediction_filtered_count"] == 4
    assert stages["published"]["primitive_matching"]["matched_count"] == 4
    assert stages["parametric_candidate"]["status"] == "not_available"
    assert (run / stages["initial_cad"]["overlay"]).is_file()
    assert all(p.read_bytes() == data for p, data in before.items())


def test_analytic_arc_area_uses_native_radius_not_svg_chords(tmp_path):
    path = tmp_path / "semicircle.dxf"
    doc = ezdxf.new()
    doc.units = 4
    doc.modelspace().add_arc((0, 0), 2, 0, 180)
    doc.modelspace().add_line((-2, 0), (2, 0))
    doc.saveas(path)
    area = _single_cycle_area(audit_dxf(path))
    assert area["status"] == "single_closed_cycle"
    assert area["area_mm2"] == pytest.approx(2 * math.pi)


def test_unitless_area_is_not_mislabeled_square_millimetres(tmp_path):
    path = tmp_path / "unknown.dxf"
    _rectangle(path, units=0)
    assert _single_cycle_area(audit_dxf(path)) == {"status": "unknown_units", "area_mm2": None}
