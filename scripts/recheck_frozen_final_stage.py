"""Re-run online binding/solving on a frozen source topology, never a full run.

The upstream prediction and OCR receipts are immutable. Reference evaluation is
launched in a separate process only after the new prediction has been frozen.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_controlled_iteration as runner


def validate_upstream(source):
    source = Path(source).resolve(strict=True)
    manifest = runner.read(source / "run-manifest.json")
    if not manifest.get("prediction_frozen_at") or not manifest.get("frozen_artifacts"):
        raise ValueError("upstream_prediction_not_frozen")
    for name, sha in manifest["frozen_artifacts"].items():
        if Path(name).name != name or not (source / "after" / name).is_file():
            raise ValueError("invalid_upstream_artifact_path")
        if runner.digest(source / "after" / name) != sha:
            raise ValueError("upstream_artifact_changed")
    return manifest


def replay_baseline(source):
    """Restore the frozen run's policy without copying solved coordinates.

    baseline-model predates source-first solving. Loading it alone silently
    turns a source-first continuation back into a mask-hard reconstruction.
    Only the declared policy is inherited from the frozen exported model.
    """
    from contour_agent.reconstruction_priority import SOURCE_FIRST, source_first_model
    baseline=runner.read(Path(source)/'after/baseline-model.json')
    published=runner.read(Path(source)/'after/model.json')
    if (published.get('reconstruction_priority') or {}).get('policy')==SOURCE_FIRST:
        baseline=source_first_model(baseline)
    return baseline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "code-root", "config-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--provider-profile", default="9ecode-gpt-5.6-sol")
    parser.add_argument("--api-timeout", type=float, default=180.)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--target-tolerance-mm", type=float, default=1.5,
                        help="Additional diagnostic threshold; strict 0.1 mm evaluation remains unchanged")
    parser.add_argument('--repair-topology', action='store_true',
                        help='Permit bounded source-driven topology completion after the frozen initial graph')
    parser.add_argument('--resume-published-subset', action='store_true',
                        help='Revalidate the frozen published source subset and resume its original immutable numerical domain')
    args = parser.parse_args(argv)
    if not 5 <= args.api_timeout <= 600:
        parser.error("api-timeout must be between 5 and 600 seconds")
    import math
    if not math.isfinite(args.target_tolerance_mm) or args.target_tolerance_mm <= 0:
        parser.error("target-tolerance-mm must be positive and finite")
    source, config, run = args.source_run.resolve(), args.config_root.resolve(strict=True), args.output.resolve()
    if run.exists() or not run.is_relative_to(config / "runtime" / "cad-controlled-20261008"):
        parser.error("use a fresh output directory inside the controlled experiment root")
    upstream = validate_upstream(source)
    run.mkdir(parents=True)
    state = {"schema_version": "controlled-cad-iteration-v1", "label": run.name,
        "created_at": runner.stamp(), "status": "initializing", "stage_history": [],
        "execution_scope": "frozen_topology_online_final_stage_replay", "upstream_reconstruction_rerun": False,
        "upstream_run": str(source), "upstream_manifest_sha256": runner.digest(source / "run-manifest.json"),
        "oracle_mask_conditioned": True, "held_out": False, "calibration_or_development": True,
        "ground_truth_coordinates_sent_to_provider": False, "ground_truth_dxf_used_as_prediction_geometry": False,
        "evaluation_requested": args.evaluate, "frozen_strict_evaluation_tolerance_mm": .1,
        "additional_evaluation_tolerance_mm": args.target_tolerance_mm,
        "scope": "Reused frozen topology, OCR review and source arrow proposals. Only final online binding, solving and export are rerun."}
    if args.repair_topology:
        state.update(execution_scope='frozen_initial_topology_online_binding_solve_and_structural_completion',
            topology_completion_enabled=True,
            scope='Frozen upstream OCR and initial graph; rerun online binding, joint solve, bounded source-driven structural completion and export.')
    def checkpoint(stage, message=""):
        state["last_stage"] = stage
        state["stage_history"].append({"stage": stage, "at": runner.stamp()})
        runner.write(run / "run-manifest.json", state)
        print(stage, flush=True)
    checkpoint("initializing")
    try:
        before = runner.code_snapshot(args.code_root)
        root = runner.select_code_root(args.code_root)
        state["code"] = {"root": str(root), "sha256_before": before,
            "runner_path": str(Path(__file__).resolve()), "runner_sha256": runner.digest(__file__)}
        inputs, receipt = runner.freeze_inputs(source, run / "inputs")
        state.update(case_id=receipt["case_id"], source_gt_sha256=receipt.get("reference_dxf_sha256"),
            source_image_sha256=receipt["source_image_sha256"], source_ocr_sha256=receipt["source_ocr_sha256"],
            oracle_mask_sha256=receipt["mask_sha256"])
        from contour_agent.source_ocr_review import load_effective_ocr
        from contour_agent.ocr import canonical_records
        from contour_agent.parametric_pipeline import refine_parametric, CORE
        document = load_effective_ocr(inputs["source_image"], inputs["source_ocr"], source / "after")
        indices = {row["id"]: i for i, row in enumerate(canonical_records(document))}
        proposals = {}
        arrow_path = source / "after" / "radius-targets.json"
        if arrow_path.is_file():
            for row in runner.read(arrow_path).get("proposals", []):
                if row.get("record_id") not in indices:
                    raise ValueError("unknown_upstream_arrow_record")
                proposals.setdefault(row["record_id"], []).append(copy.deepcopy(row))
        for rid, rows in proposals.items():
            document["records"][indices[rid]]["source_arrow_proposals"] = rows[:2]
        baseline = replay_baseline(source)
        state['reconstruction_priority']=copy.deepcopy(baseline.get('reconstruction_priority',{}))
        graph = runner.read(source / "after" / "topology.json")
        trusted = (None if args.resume_published_subset else
            runner.read(source / "after" / "topology-plan.json").get("selected_preflight_artifact"))
        if trusted:
            trusted = Path(trusted).resolve(strict=True)
            if not trusted.is_relative_to(source / "after"):
                raise ValueError("upstream_preflight_outside_source")
        frozen_nested = {str(p.relative_to(source)): runner.digest(p)
            for p in (source / "after").rglob("*") if p.is_file()}
        state["frozen_upstream_receipts_sha256"] = frozen_nested
        state["source_ocr_review"] = {**upstream.get("source_ocr_review", {}),
            "reused_from_upstream": True, "new_network_requests": 0}
        after = run / "after"
        after.mkdir()
        for name in CORE:
            shutil.copyfile(source / "after" / ("baseline-" + name), after / name)
        for name in ("source-ocr-review.json", "effective-source-ocr.json", "radius-targets.json"):
            if (source / "after" / name).is_file():
                shutil.copyfile(source / "after" / name, after / name)
        runner.write(run / "inputs" / "source-ocr-with-upstream-proposals.json", document)
        providers, state["provider"] = runner.providers(config, True, args.provider_profile, args.api_timeout)
        if args.resume_published_subset:
            # Resume only a matched source-image/binding/solve tuple. This
            # validates native equations and original ink, keeps the original
            # displacement domain and still requires fresh online binding.
            from contour_agent.source_state import resume_source_state
            checkpoint('revalidate_published_source_subset')
            trusted=after/'published-source-resume'
            graph=resume_source_state(inputs['source_image'],document,baseline,graph,
                runner.read(source/'after/constraint-bindings.json'),
                runner.read(source/'after/parametric-solution.json'),
                runner.read(source/'after/binding-candidates.json'),trusted)
            state.update(execution_scope='certified_source_subset_online_continuation',
                scope='Frozen source-certified partial prediction revalidated against original pixels and equations; original domain preserved, fresh online binding and bounded source completion rerun.',
                upstream_source_subset_used_as_initialization=True,
                original_numerical_domain_recentered=False)
        state["status"] = "running"
        checkpoint("frozen_topology_online_final_stage")
        model, stage = refine_parametric(inputs["source_image"], document, baseline, after,
            provider=providers["provider"], use_api=True, progress=checkpoint,
            editor_provider=providers.get('editor_provider') if args.repair_topology else None,
            frozen_topology=graph, trusted_preflight_dir=trusted,
            allow_frozen_topology_repair=args.repair_topology)
        state.update(parameterization_status=stage.get("status"), parameterization_accepted=stage.get("accepted") is True,
            constraint_subset_accepted=stage.get("constraint_subset_accepted") is True)
        runner.write(run / "current-export-audit.json", runner.local_export_audit(after))
        state["prediction_sha256"] = runner.digest(after / "drawing.dxf")
        state["frozen_artifacts"] = {p.name: runner.digest(p) for p in after.iterdir() if p.is_file()}
        state["code"].update(imported_sources=runner.imported_sources(root), sha256_after=runner.code_snapshot(root))
        state["code"]["unchanged"] = state["code"]["sha256_after"] == before
        state["upstream_receipts_unchanged"] = all(runner.digest(source / name) == sha for name, sha in frozen_nested.items())
        if not state["code"]["unchanged"] or not state["upstream_receipts_unchanged"]:
            raise RuntimeError("immutable_code_or_upstream_receipt_changed")
        state.update(prediction_frozen_at=runner.stamp(), status="prediction_frozen")
        checkpoint("prediction_frozen")
        if args.evaluate:
            comparator = config / "scripts" / "evaluate_oracle_mask_run.py"
            state["evaluation_code"] = {"root": str(config), "sha256_before": runner.code_snapshot(config),
                "comparator_sha256": runner.digest(comparator)}
            checkpoint("post_export_evaluation")
            completed = subprocess.run([sys.executable, "-X", "utf8", str(comparator), "--run-dir", str(run),
                "--dataset", str(config / "__dataset"), "--case-id", state["case_id"],
                "--target-tolerance-mm", str(args.target_tolerance_mm)],
                cwd=config, capture_output=True, timeout=600, check=False)
            state["evaluation_exit_code"] = completed.returncode
            state["evaluation_code"]["unchanged"] = runner.code_snapshot(config) == state["evaluation_code"]["sha256_before"]
            state["prediction_unchanged_after_evaluation"] = runner.digest(after / "drawing.dxf") == state["prediction_sha256"]
            if completed.returncode or not state["evaluation_code"]["unchanged"] or not state["prediction_unchanged_after_evaluation"]:
                raise RuntimeError("evaluation_failed_or_frozen_artifact_changed")
            state["evaluation_status"] = runner.read(run / "evaluation" / "summary.json").get("status")
        state["status"] = "completed" if state["parameterization_accepted"] else "completed_with_unresolved_attributes"
        checkpoint("finished")
        return 0
    except BaseException as error:
        state.update(status="failed", failure_type=type(error).__name__, failure_stage=state.get("last_stage"))
        runner.write(run / "run-manifest.json", state)
        print(json.dumps({"status": "failed", "failure_type": type(error).__name__}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
