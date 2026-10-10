"""The review reads frozen predictions only and leaves all run bytes intact."""
import importlib.util
import json
from pathlib import Path

import ezdxf
import pytest


SPEC = importlib.util.spec_from_file_location("strict_joint_report", Path(__file__).parents[1] / "scripts/report_strict_joint_iteration.py")
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def _fixture(root, version="v11", *, tangent=True, dimensions=False):
    run = root / f"angle-semantics-{version}-hdsa"
    after = run / "after"; after.mkdir(parents=True)
    document = ezdxf.new("R2010"); document.units = 4
    document.modelspace().add_arc((0, 0), 1., 0., 180.)
    document.modelspace().add_arc((0, 0), 1., 180., 360.)
    document.saveas(after / "drawing.dxf")
    entities = [dict(id="upper", type="ARC", start=[1., 0.], end=[-1., 0.], center=[0., 0.], radius=1., clockwise=False, start_node="a", end_node="b"),
                dict(id="lower", type="ARC", start=[-1., 0.], end=[1., 0.], center=[0., 0.], radius=1., clockwise=False, start_node="b", end_node="a")]
    constraints = [dict(id="r"+str(i), kind="radius", entities=[entity["id"]], value=1., record_id="r"+str(i), nodes=[]) for i, entity in enumerate(entities)]
    if tangent:
        constraints += [dict(id="left", kind="tangent", entities=["upper", "lower"], nodes=["b"], value=None),
                        dict(id="right", kind="tangent", entities=["upper", "lower"], nodes=["a"], value=None)]
    published = {"status": "completed", "accepted": True, "constraint_subset_accepted": True,
                 "constraints": constraints, "all_dimensions_verified": dimensions,
                 "binding_counts": {"recognized_dimensions": 2, "bound_source_records": 2, "unbound_dimensions": 0},
                 "solver": {"status": "accepted", "accepted": True, "diagnostics": {"remaining_shape_dof": 0},
                            "validation": {"geometry_valid": True}},
                 "radius_binding_coverage": {"recognized_count": 2, "required_count": 2, "all_radius_records_resolved": True}}
    _write(after / "model.json", {"entities": entities, "parameterization": published})
    _write(after / "parametric-solution.json", {"entities": entities, "candidate_entities": entities,
        "accepted": True, "diagnostics": {"remaining_shape_dof": 0}, "constraints": constraints})
    _write(after / "constraint-bindings.json", {"constraints": constraints, "counts": published["binding_counts"]})
    _write(after / "parametric-stage.json", {"status": "completed", "provider": {"network_requests": 1, "http_success": True,
        "schema_success": True, "input_record_ids": ["r0", "r1"], "input_relation_ids": ["left", "right"],
        "elapsed_seconds": 3., "inventory_coverage": {"all_record_count": 2, "all_relation_count": 2}}})
    _write(after / "binding-candidates.json", {"all_records": [{"id": "r0", "parsed": {"kind": "radius", "nominal": 1.}},
                                                               {"id": "r1", "parsed": {"kind": "radius", "nominal": 1.}}]})
    sha = reporter.digest(after / "drawing.dxf")
    _write(run / "run-manifest.json", {"status": "completed", "label": run.name, "case_id": "HDSA-65-main",
        "created_at": "2026-10-10T00:00:00+00:00", "prediction_frozen_at": "2026-10-10T00:01:00+00:00",
        "stage_history": [{"stage": "finished", "at": "2026-10-10T00:01:10+00:00"}],
        "prediction_sha256": sha, "frozen_artifacts": {name: reporter.digest(after/name) for name in reporter.REQUIRED_FILES},
        "code": {"unchanged": True}, "source_image_sha256": "same_image", "source_ocr_sha256": "same_ocr", "oracle_mask_sha256": "same_mask"})
    _write(run / "evaluation/summary.json", {"status": "compared", "provenance": {"prediction_sha256": sha, "reference_sha256": "same_ref"},
        "target_acceptance": {"all_requested_checks_passed": True, "primitive_matching": {"matched_count": 2}},
        "geometry": {"registered_shape_diagnostic": {"conservative_max_error_mm": .2, "rms_error_mm": .1}}})
    return run


def test_all_expected_slots_remain_and_g1_is_recomputed(tmp_path):
    _fixture(tmp_path)
    report = reporter.assemble(tmp_path)
    assert report["coverage"] == {"expected_slots": 4, "fully_passed_slots": 0, "not_run_slots": 3, "frozen_audited_slots": 1}
    row = report["rows"][0]
    assert row["strict_relations"]["required_count"] == row["strict_relations"]["satisfied_count"] == 2
    assert row["curved_joins"]["unknown_count"] == 0
    assert row["elapsed"] == {"prediction_seconds": 60., "total_run_seconds": 70.,
                              "scope": "Wall time from run creation; total includes independent evaluation when finished."}
    assert row["reference"]["matched_primitive_count_1mm"] == 2
    assert row["full_user_target_passed"] is False  # all dimensions receipt is false


def test_unknown_arc_arc_joins_are_not_erased_from_denominator(tmp_path):
    _fixture(tmp_path, tangent=False, dimensions=True)
    row = reporter.assemble(tmp_path)["rows"][0]
    assert row["strict_relations"]["passed"] and row["strict_relations"]["required_count"] == 0
    assert row["curved_joins"]["curved_join_count"] == row["curved_joins"]["unknown_count"] == 2
    assert row["curved_joins"]["unknown_arc_arc_count"] == 2
    assert not row["full_user_target_passed"]


def test_pending_run_never_reads_after_artifacts(tmp_path, monkeypatch):
    run = tmp_path / "angle-semantics-v12-hdsa"
    _write(run / "run-manifest.json", {"status": "running", "last_stage": "solving"})
    _write(run / "after/model.json", {"entities": "not frozen must not read"})
    def unexpected(*args, **kwargs):
        raise AssertionError("collect_attempt must not inspect pending after artifacts")
    monkeypatch.setattr(reporter, "collect_attempt", unexpected)
    report = reporter.assemble(tmp_path)
    row = report["rows"][1]
    assert row["status"] == "running" and row["artifact_state"] == "not_frozen_no_native_audit"
    assert not row["full_user_target_passed"]


def test_changed_model_receipt_cannot_gain_frozen_certificate(tmp_path):
    run = _fixture(tmp_path)
    (run / "after/model.json").write_text((run / "after/model.json").read_text() + " ")
    row = reporter.assemble(tmp_path)["rows"][0]
    assert row["strict_audit_status"] == "frozen_receipt_mismatch"
    assert "strict_relations" not in row
    assert not row["full_user_target_passed"]


def test_report_and_audits_are_siblings_and_do_not_modify_run_or_old_report(tmp_path):
    run = _fixture(tmp_path)
    before = {path.relative_to(run).as_posix(): reporter.digest(path) for path in run.rglob("*") if path.is_file()}
    (tmp_path / "report.html").write_text("old report sentinel")
    report, json_path, html_path = reporter.write_report(tmp_path)
    after = {path.relative_to(run).as_posix(): reporter.digest(path) for path in run.rglob("*") if path.is_file()}
    assert before == after
    assert (tmp_path / "report.html").read_text() == "old report sentinel"
    assert json_path.is_file() and html_path.is_file()
    audit = tmp_path / report["rows"][0]["files"]["audit"]
    assert audit.is_file() and run not in audit.parents
    assert "0 / 4" in html_path.read_text(encoding="utf8")


def test_final_binding_sent_and_successful_page_subsets_are_separate():
    result = reporter._binding_receipt({"provider": {"network_requests": 2, "http_success": False, "schema_success": True,
        "sent_record_ids": ["r0", "r1"], "input_record_ids": ["r1"], "sent_relation_ids": ["a", "b"],
        "all_pages_succeeded": False, "pages": [{"page_index": 1, "http_success": False, "schema_success": False},
                                                 {"page_index": 2, "http_success": True, "schema_success": True}]}})
    assert result["actual_sent_record_count"] == 2 and result["valid_response_record_count"] == 1
    assert not result["http_success"] and not result["all_pages_succeeded"]
    assert result["schema_success_scope"] == "valid retained page subset"


def test_same_frozen_inputs_compared_without_best_result_selection(tmp_path):
    _fixture(tmp_path, "v11")
    _fixture(tmp_path, "v12")
    report = reporter.assemble(tmp_path)
    assert report["comparisons"][0]["same_frozen_inputs_and_reference"]
    assert not report["comparisons"][0]["automatic_promotion"]


def test_untrusted_labels_and_errors_are_escaped(tmp_path):
    report = reporter.assemble(tmp_path)
    report["rows"][0]["status"] = "<script>alert(1)</script>"
    page = reporter.render(report)
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page


def test_three_versions_keep_six_slots_and_only_adjacent_deltas(tmp_path):
    for version, maximum, rms in (("v11", .4, .2), ("v12", .2, .1), ("v13", .1, .08)):
        run = _fixture(tmp_path, version)
        path = run / "evaluation/summary.json"
        evaluation = reporter.read(path)
        evaluation["geometry"]["registered_shape_diagnostic"].update(
            conservative_max_error_mm=maximum, rms_error_mm=rms)
        _write(path, evaluation)
    report = reporter.assemble(tmp_path, versions=("v11", "v12", "v13"))
    assert report["versions"] == ["v11", "v12", "v13"]
    assert report["coverage"] == {"expected_slots": 6, "fully_passed_slots": 0,
                                  "not_run_slots": 3, "frozen_audited_slots": 3}
    pairs = report["comparisons"]
    assert [(pair["from_version"], pair["to_version"]) for pair in pairs] == [
        ("v11", "v12"), ("v12", "v13"), ("v11", "v12"), ("v12", "v13")]
    assert pairs[0]["registered_max_delta_mm"] == pytest.approx(-.2)
    assert pairs[1]["registered_max_delta_mm"] == pytest.approx(-.1)
    assert pairs[1]["registered_rms_delta_mm"] == pytest.approx(-.02)
    assert pairs[2]["registered_max_delta_mm"] is None
    assert pairs[3]["registered_max_delta_mm"] is None
    page = reporter.render(report)
    assert "V11 / V12 / V13" in page and "0 / 6" in page
    assert "v12 → v13" in page and "暂无有效比较" in page


@pytest.mark.parametrize("state", ["missing", "pending", "failed", "stale_evaluation", "changed_frozen_artifact"])
def test_unavailable_or_invalid_later_version_never_produces_numeric_delta(tmp_path, state):
    _fixture(tmp_path, "v11")
    _fixture(tmp_path, "v12")
    if state == "pending":
        _write(tmp_path / "angle-semantics-v13-hdsa/run-manifest.json", {"status": "running"})
    elif state != "missing":
        run = _fixture(tmp_path, "v13")
        if state == "failed":
            manifest = reporter.read(run / "run-manifest.json")
            manifest["status"] = "failed"
            _write(run / "run-manifest.json", manifest)
        elif state == "stale_evaluation":
            evaluation = reporter.read(run / "evaluation/summary.json")
            evaluation["provenance"]["prediction_sha256"] = "obsolete_prediction"
            _write(run / "evaluation/summary.json", evaluation)
        else:
            path = run / "after/model.json"
            path.write_text(path.read_text() + " ")
    report = reporter.assemble(tmp_path, versions=["v11", "v12", "v13"])
    assert report["coverage"]["expected_slots"] == 6
    earlier, later = report["comparisons"][:2]
    assert earlier["same_frozen_inputs_and_reference"]
    assert earlier["registered_max_delta_mm"] == 0
    assert not later["same_frozen_inputs_and_reference"]
    deltas = {key: value for key, value in later.items() if "_delta" in key}
    assert deltas and all(value is None for value in deltas.values())


@pytest.mark.parametrize("versions", [[], ["v11", "v11"], ["../v11"], ["V11"], ["v11", 13], "v11"])
def test_invalid_version_sequences_are_rejected(tmp_path, versions):
    with pytest.raises(ValueError, match="versions"):
        reporter.assemble(tmp_path, versions=versions)


def test_cli_passes_requested_versions_to_written_report(tmp_path, capsys):
    _fixture(tmp_path, "v13")
    reporter.main(["--root", str(tmp_path), "--versions", "v11", "v12", "v13",
                   "--output-prefix", "three-version-review"])
    printed = json.loads(capsys.readouterr().out)
    report = reporter.read(Path(printed["json"]))
    assert report["versions"] == ["v11", "v12", "v13"]
    assert report["coverage"]["expected_slots"] == printed["coverage"]["expected_slots"] == 6
    assert report["rows"][2]["version"] == "v13" and report["rows"][2]["strict_audit_status"] == "read_back"
    assert "V11 / V12 / V13" in Path(printed["html"]).read_text(encoding="utf8")


def _freeze_changed_synthetic_receipts(run):
    manifest = reporter.read(run / "run-manifest.json")
    manifest["frozen_artifacts"] = {name: reporter.digest(run/"after"/name) for name in reporter.REQUIRED_FILES}
    _write(run / "run-manifest.json", manifest)


def test_retained_topology_draft_never_inherits_failed_candidate_bindings_or_dof(tmp_path):
    run = _fixture(tmp_path)
    model = reporter.read(run / "after/model.json")
    model["parameterization"] = dict(status="source_topology_exported", accepted=False,
                                     constraint_subset_accepted=False, topology_exported=True)
    _write(run / "after/model.json", model)
    solution = reporter.read(run / "after/parametric-solution.json")
    # Rejected solvers return baseline entities as their retained output. The
    # DOF and candidate_entities still describe the failed numerical proposal.
    solution.update(accepted=False, status="source_budget_search_failed", diagnostics=dict(remaining_shape_dof=27))
    solution["candidate_entities"][0]["radius"] = 2.
    _write(run / "after/parametric-solution.json", solution)
    _freeze_changed_synthetic_receipts(run)
    before = {p.relative_to(run).as_posix(): reporter.digest(p) for p in run.rglob("*") if p.is_file()}
    report, _, html_path = reporter.write_report(tmp_path)
    row = report["rows"][0]
    assert row["attributes"]["recognized_dimensions"] == 2
    assert row["attributes"]["bound_source_records"] == row["attributes"]["formal_constraint_count"] == 0
    assert row["attributes"]["unbound_dimensions"] == 2
    assert row["attributes"]["remaining_shape_dof"] is None
    assert row["attempt_attributes"]["bound_source_records"] == 2
    assert row["attempt_attributes"]["remaining_shape_dof"] == 27
    assert not row["attempt_attributes"]["solver_accepted"]
    assert not row["attempt_attributes"]["applies_to_current_core"]
    assert row["published"]["geometry_only_solution_match"] is True
    assert not row["published"]["current_solution_receipt_matches_model"]
    assert row["published"]["remaining_shape_dof"] is None
    assert row["published"]["maximum_recorded_constraint_residual_by_unit"] == {"mm": None, "degree": None}
    assert not row["full_user_target_passed"]
    assert before == {p.relative_to(run).as_posix(): reporter.digest(p) for p in run.rglob("*") if p.is_file()}
    page = html_path.read_text(encoding="utf8")
    assert "当前绑定 / 已识别" in page and "最新候选绑定 / 已识别" in page
    assert "最新候选自由度" in page and "source_budget_search_failed" in page


def test_current_published_solver_diagnostics_are_not_overridden_even_by_matching_solution(tmp_path):
    run = _fixture(tmp_path)
    solution = reporter.read(run / "after/parametric-solution.json")
    solution["diagnostics"]["remaining_shape_dof"] = 27
    _write(run / "after/parametric-solution.json", solution)
    bindings = reporter.read(run / "after/constraint-bindings.json")
    bindings["counts"].update(bound_source_records=13, recognized_dimensions=37, unbound_dimensions=24)
    _write(run / "after/constraint-bindings.json", bindings)
    _freeze_changed_synthetic_receipts(run)
    row = reporter.assemble(tmp_path)["rows"][0]
    assert row["solution_publication_link"]["checks"]["solution_geometry_matches"]
    assert row["solution_publication_link"]["checks"]["candidate_geometry_matches"]
    assert not row["solution_publication_link"]["checks"]["solver_diagnostics_agree"]
    assert not row["solution_publication_link"]["verified"]
    assert row["attributes"]["remaining_shape_dof"] == 0
    assert row["attributes"]["bound_source_records"] == 2
    assert row["attempt_attributes"]["remaining_shape_dof"] == 27
    assert row["attempt_attributes"]["bound_source_records"] == 13


@pytest.mark.parametrize("defect", ["rejected", "candidate_mismatch", "obligation_mismatch", "node_mismatch", "unaccepted_publication"])
def test_equal_baseline_geometry_is_insufficient_to_associate_solution_receipt(tmp_path, defect):
    run = _fixture(tmp_path)
    solution = reporter.read(run / "after/parametric-solution.json")
    if defect == "rejected": solution["accepted"] = False
    elif defect == "candidate_mismatch": solution["candidate_entities"][0]["id"] = "other"
    elif defect == "node_mismatch": solution["entities"][0]["start_node"] = "other"
    elif defect == "obligation_mismatch": solution["constraints"][0]["value"] = 2.
    else:
        model = reporter.read(run / "after/model.json")
        model["parameterization"].update(accepted=False, constraint_subset_accepted=False)
        _write(run / "after/model.json", model)
    _write(run / "after/parametric-solution.json", solution)
    _freeze_changed_synthetic_receipts(run)
    row = reporter.assemble(tmp_path)["rows"][0]
    assert row["published"]["geometry_only_solution_match"] is True
    assert not row["solution_publication_link"]["verified"]
    assert not row["published"]["current_solution_receipt_matches_model"]
    assert not row["attempt_attributes"]["applies_to_current_core"]


def test_topology_without_formal_constraints_has_unknown_dof_even_with_stale_cached_solver(tmp_path):
    run = _fixture(tmp_path, dimensions=True)
    model = reporter.read(run / "after/model.json")
    model["parameterization"]["constraints"] = []
    model["parameterization"]["status"] = "source_topology_exported"
    _write(run / "after/model.json", model)
    _freeze_changed_synthetic_receipts(run)
    row = reporter.assemble(tmp_path)["rows"][0]
    assert row["attributes"]["remaining_shape_dof"] is None
    assert row["attributes"]["bound_source_records"] == 0
    assert not row["full_user_target_passed"]
