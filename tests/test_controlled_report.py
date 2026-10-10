import importlib.util
import json
from pathlib import Path

import ezdxf


SPEC = importlib.util.spec_from_file_location("controlled_report", Path(__file__).parents[1] / "scripts/report_controlled_iteration.py")
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def run_fixture(root, name="base", *, label="baseline", status="completed", date="2026-10-08T00:00:00Z", shape_dof=1):
    run = root / name
    after = run / "after"
    after.mkdir(parents=True)
    doc = ezdxf.new("R2010")
    doc.units = 4
    doc.modelspace().add_arc((0, 0), 3, 0, 90)
    doc.saveas(after / "drawing.dxf")
    sha = reporter.digest(after / "drawing.dxf")
    entity = {"id": "g0", "type": "ARC", "radius": 3}
    write(after / "model.json", {"entities": [entity], "parameterization": {
        "constraints": [{"kind": "radius", "entities": ["g0"], "value": 3, "record_id": "r0"}],
        "all_dimensions_verified": True, "solver": {"diagnostics": {"remaining_shape_dof": shape_dof}},
        "radius_binding_coverage": {"recognized_count": 1, "required_count": 1, "all_radius_records_resolved": True}}})
    write(run / "run-manifest.json", {"case_id": "HDSA-65-main", "label": label, "status": status,
        "created_at": date, "prediction_frozen_at": date, "prediction_sha256": sha,
        "code": {"unchanged": True, "sha256_before": {"p.py": "same"}},
        "source_image_sha256": "image", "source_ocr_sha256": "ocr", "oracle_mask_sha256": "mask"})
    write(run / "evaluation/summary.json", {"status": "compared", "provenance": {"prediction_sha256": sha, "reference_sha256": "ref"},
        "geometry": {"registered_shape_diagnostic": {"conservative_max_error_mm": .5}},
        "checks": {"registered_shape_within_0_1mm": False},
        "target_acceptance": {"all_requested_checks_passed": True, "checks": {"registered_shape_within_target": True}}})
    return run


def test_failed_and_missing_slots_remain_in_denominator(tmp_path):
    run_fixture(tmp_path, shape_dof=1)
    write(tmp_path / "failed/run-manifest.json", {"label": "one-fix", "case_id": "HDSA-65-main", "status": "failed"})
    result = reporter.assemble(tmp_path)
    assert result["coverage"] == {"actual_attempts": 2, "actual_attempts_full_pass": 0,
                                  "expected_label_case_slots": 4, "latest_slots_full_pass": 0, "not_run_slots": 2}
    assert result["all_attempts"][0]["native"]["exact_radius_count"] == 1
    assert result["all_attempts"][0]["full_user_target_passed"] is False


def test_latest_is_chronological_not_best_gt(tmp_path):
    run_fixture(tmp_path, "earlier", date="2026-10-08T00:00:00Z", shape_dof=0)
    run_fixture(tmp_path, "later", date="2026-10-08T01:00:00Z", status="failed", shape_dof=3)
    result = reporter.assemble(tmp_path, cases=("HDSA-65-main",), labels=("baseline",))
    assert result["latest_chronological"][0]["attempt"] == "later"
    assert result["coverage"]["actual_attempts"] == 2
    assert result["coverage"]["latest_slots_full_pass"] == 0


def test_tampered_dxf_invalidates_stale_evaluation(tmp_path):
    run = run_fixture(tmp_path, shape_dof=0)
    with (run / "after/drawing.dxf").open("ab") as stream:
        stream.write(b"\n")
    row = reporter.collect_attempt(tmp_path, run)
    assert row["frozen_prediction_still_current"] is False
    assert row["artifact_state"] == "frozen_artifact_mismatch"
    assert row["reference"]["registered_max_mm"] is None
    assert row["reference"]["checks_1mm"] is None
    assert row["full_user_target_passed"] is False


def test_current_native_r_is_not_stale_solver_claim(tmp_path):
    run = run_fixture(tmp_path)
    doc = ezdxf.readfile(run / "after/drawing.dxf")
    list(doc.modelspace())[0].dxf.radius = 4
    doc.saveas(run / "after/drawing.dxf")
    native = reporter.native_export_check(run / "after/drawing.dxf", reporter.read(run / "after/model.json"))
    assert native["exact_radius_count"] == 0
    assert native["exact_radius_checks"][0]["current_dxf_radius"] == 4


def test_report_escapes_untrusted_labels(tmp_path):
    run_fixture(tmp_path, label='<script>alert("x")</script>')
    result = reporter.assemble(tmp_path)
    html = reporter.render(result)
    assert '<script>alert("x")</script>' not in html
    assert '&lt;script&gt;' in html


def test_recognized_radius_denominator_includes_unknown_arrows(tmp_path):
    run = run_fixture(tmp_path)
    path = run / "after/model.json"
    model = reporter.read(path)
    model["parameterization"]["radius_binding_coverage"].update(
        recognized_count=3, required_count=1, all_radius_records_resolved=False,
        unknown_arrow_records=["r1", "r2"], unresolved_count=2)
    write(path, model)
    row = reporter.collect_attempt(tmp_path, run)
    assert row["native"]["exact_radius_count"] == 1
    assert row["native"]["recognized_radius_count"] == 3
    assert row["native"]["required_radius_count"] == 1
    result = reporter.render({"all_attempts": [row], "latest_chronological": [row], "comparisons": [],
                              "coverage": {"actual_attempts": 1, "actual_attempts_full_pass": 0,
                                           "latest_slots_full_pass": 0, "expected_label_case_slots": 1, "not_run_slots": 0}, "limits": []})
    assert "1 / 3" in result
    assert "1 / 1" not in result


def test_nested_radius_receipt_is_not_added_to_stage_aggregate(tmp_path):
    after = tmp_path / "after"
    write(after / "parametric-stage.json", {"radius_target_provider": {
        "network_requests": 2, "attempts": [{"network_requests": 1}, {"network_requests": 1}],
        "status": "completed", "http_success": True, "schema_success": True}})
    receipts = reporter.transport_receipts(after)
    assert len(receipts) == 1
    assert receipts[0]["network_requests"] == 2
    assert "attempts" not in receipts[0]


def test_round_branch_transport_keeps_failures_without_mirror_duplicates(tmp_path):
    after = tmp_path / "after"
    success = {"status": "succeeded", "network_requests": 1, "http_success": True, "schema_success": True}
    failed = {"status": "failed", "network_requests": 1, "http_success": False, "schema_success": False}
    branch = {"round": 1, "branch": 1, "editor": success, "evaluator": failed}
    mirror = {**branch, "branches": [branch, {**branch, "branch": 2}]}
    write(after / "topology-iterations.json", {"rounds": [mirror]})
    write(after / "topology-plan.json", {"topology_editing": {"rounds": [mirror]},
                                         "editor_provider": success, "evaluator_provider": failed})
    receipts = reporter.transport_receipts(after)
    assert len(receipts) == 4
    assert sum(row["network_requests"] for row in receipts) == 4
    assert sum(row["http_success"] is False for row in receipts) == 2
    assert len({row["stage"] for row in receipts}) == 4


def test_native_endpoint_closure_uses_declared_numeric_tolerance():
    doc = ezdxf.new("R2010")
    space = doc.modelspace()
    for first, second in [((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (0, 1)), ((0, 1), (1e-8, 0))]:
        space.add_line(first, second)
    result = reporter.endpoint_connectivity(list(space))
    assert result["closed_endpoint_degree"] is True
    assert result["component_count"] == 1
    list(space)[-1].dxf.end = (1e-3, 0)
    assert reporter.endpoint_connectivity(list(space))["closed_endpoint_degree"] is False


def test_partial_radius_validation_is_not_reported_as_full_schema_success(tmp_path):
    run = run_fixture(tmp_path)
    write(run / "after/parametric-stage.json", {"radius_target_provider": {
        "status": "partial_or_failed", "http_success": True, "schema_success": False,
        "partial_schema_success": True, "any_partial_schema_success": True,
        "rejected_record_ids": ["r_bad"], "unresolved_rejected_record_ids": ["r_bad"],
        "network_requests": 2, "attempts": [{"network_requests": 1}, {"network_requests": 1}]}})
    result = reporter.assemble(tmp_path, cases=("HDSA-65-main",), labels=("baseline",))
    receipt = result["all_attempts"][0]["transport"][0]
    assert receipt["http_success"] is True
    assert receipt["schema_success"] is False
    assert receipt["partial_schema_success"] is True
    assert receipt["rejected_record_ids"] == receipt["unresolved_rejected_record_ids"] == ["r_bad"]
    assert receipt["network_requests"] == 2 and "attempts" not in receipt
    page = reporter.render(result)
    assert "完整结构成功" in page and "部分结构成功" in page
    assert "<td>通过</td><td>未通过</td><td>通过</td>" in page
    assert result["coverage"]["actual_attempts_full_pass"] == 0


def test_interrupted_prediction_stays_in_denominator_and_out_of_comparisons(tmp_path):
    run_fixture(tmp_path)
    run = run_fixture(tmp_path, "interrupted", label="final-candidate", status="interrupted")
    manifest_path = run / "run-manifest.json"
    manifest = reporter.read(manifest_path)
    manifest.update(prediction_frozen=False, interruption_reason="execution_session_interrupted_before_prediction_freeze")
    write(manifest_path, manifest)
    result = reporter.assemble(tmp_path, cases=("HDSA-65-main",), labels=("baseline", "final-candidate"))
    row = next(row for row in result["all_attempts"] if row["label"] == "final-candidate")
    assert result["coverage"]["actual_attempts"] == 2
    assert result["coverage"]["expected_label_case_slots"] == 2
    assert result["coverage"]["actual_attempts_full_pass"] == 0
    assert row["artifact_state"] == "intermediate_only"
    assert row["artifact_frozen"] is False
    assert row["reference"]["registered_max_mm"] is None
    assert row["interruption_reason"] == "execution_session_interrupted_before_prediction_freeze"
    assert result["comparisons"][0]["same_frozen_inputs_and_reference"] is False
    assert result["comparisons"][0]["registered_max_delta_mm"] is None


def test_interruption_after_freeze_cannot_be_compared_as_completed_run(tmp_path):
    run_fixture(tmp_path)
    run_fixture(tmp_path, "interrupted", label="one-fix", status="interrupted")
    result = reporter.assemble(tmp_path, cases=("HDSA-65-main",), labels=("baseline", "one-fix"))
    row = next(row for row in result["all_attempts"] if row["label"] == "one-fix")
    assert row["artifact_state"] == "frozen_current"
    assert row["reference"]["registered_max_mm"] == .5
    assert row["full_user_target_passed"] is False
    assert result["comparisons"][0]["same_frozen_inputs_and_reference"] is False
    assert result["comparisons"][0]["exact_radius_count_delta"] is None


def test_missing_partial_schema_receipt_is_unknown_not_success(tmp_path):
    write(tmp_path / "after/parametric-stage.json", {"radius_target_provider": {
        "status": "failed", "http_success": True, "schema_success": False}})
    receipt = reporter.transport_receipts(tmp_path / "after")[0]
    assert "partial_schema_success" not in receipt and "any_partial_schema_success" not in receipt


def test_completed_unresolved_runs_are_comparable_without_claiming_full_pass(tmp_path):
    run_fixture(tmp_path, status="completed_with_unresolved_attributes")
    run_fixture(tmp_path, "candidate", label="one-fix", status="completed_with_unresolved_attributes")
    result = reporter.assemble(tmp_path, cases=("HDSA-65-main",), labels=("baseline", "one-fix"))
    pair = result["comparisons"][0]
    assert pair["same_frozen_inputs_and_reference"] is True
    assert pair["registered_max_delta_mm"] == 0
    assert pair["exact_radius_count_delta"] == 0
    assert result["coverage"]["actual_attempts_full_pass"] == 0
