"""Re-run online binding/solving on a frozen source topology, never a full run.

The upstream prediction and OCR receipts are immutable. Reference evaluation is
launched in a separate process only after the new prediction has been frozen.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

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


def refine_initial_line_candidates(image, document, baseline, parent, constraints,
                                   candidates, checks, output, *, budget_seconds=240.):
    """Give at most one source-owned LINE proposal a bounded numerical preflight."""
    from contour_agent.source_topology_initial_guess import (line_seed_admission,
        repair_source_owned_line_seed)
    from contour_agent.parametric_pipeline import _topology_source_validation
    deadline=time.monotonic()+min(240.,max(0.,float(budget_seconds)))
    updated=list(candidates)
    receipt={"scope":"Local source-only initial solution; fresh online binding still required",
             "maximum_attempts":1,"maximum_wall_seconds":min(240.,max(0.,float(budget_seconds))),
             "attempts":[],"publication_approved":False,"ground_truth_used":False}
    for index,row in enumerate(updated):
        if not line_seed_admission(row["graph"],checks[row["id"]])["eligible"]:continue
        repaired,diagnostic=repair_source_owned_line_seed(image,document,baseline,row,
            Path(output)/row["id"],parent_graph=parent,parent_constraints=constraints,
            source_validation=checks[row["id"]],budget_seconds=max(0.,deadline-time.monotonic()))
        receipt["attempts"].append({"candidate_id":row["id"],"receipt":diagnostic})
        if repaired is not None:
            fresh=_topology_source_validation(image,baseline,repaired["graph"])
            if fresh.get("passed") is True:
                updated[index]=repaired
                receipt["attempts"][-1]["materialized_source_check"]=fresh
        break
    return updated,receipt


def restore_frozen_radius_proposals(image, document, source):
    """Restore final source-pixel hints without changing reviewed OCR semantics."""
    path=Path(source)/'after/effective-radius-source-document.json'
    receipt={'restored':False,'new_network_requests':0,'fresh_source_pixel_verification_required':True}
    if not path.is_file():return document,receipt
    stored=runner.read(path)
    def semantic(value):
        result=copy.deepcopy(value)
        for row in result['records']:
            row.pop('source_arrow_proposals',None)
            row.pop('source_angle_proposals',None)
        return result
    if semantic(stored)!=semantic(document):
        raise ValueError('frozen_radius_document_changed_reviewed_ocr')
    from PIL import Image
    with Image.open(image) as raster:width,height=raster.size
    result=copy.deepcopy(document);count=0
    for index,(old,row) in enumerate(zip(result['records'],stored['records'])):
        rid=f'r{index:03d}';proposals=row.get('source_arrow_proposals',[])
        if not isinstance(proposals,list) or len(proposals)>2:
            raise ValueError('frozen_radius_proposal_budget')
        for proposal in proposals:
            if (not isinstance(proposal,dict) or set(proposal)!={'record_id','tip_px','shaft_px'} or
                    proposal['record_id']!=rid):
                raise ValueError('frozen_radius_proposal_identity')
            for key in ('tip_px','shaft_px'):
                point=proposal[key]
                if (not isinstance(point,list) or len(point)!=2 or
                        any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<limit
                            for v,limit in zip(point,(width,height)))):
                    raise ValueError('frozen_radius_proposal_not_source_pixel')
        old.pop('source_arrow_proposals',None)
        if proposals:old['source_arrow_proposals']=copy.deepcopy(proposals)
        angle_proposals=row.get('source_angle_proposals',[])
        if angle_proposals:
            from contour_agent.angle_target_provider import validate_angle_target_response
            from contour_agent.ocr import parse_dimension
            if parse_dimension(row.get('text','')).get('kind')!='angle':
                raise ValueError('frozen_angle_proposal_wrong_record_kind')
            validate_angle_target_response(json.dumps({'proposals':angle_proposals,'unknown_record_ids':[]}),
                {rid},(width,height))
            old['source_angle_proposals']=copy.deepcopy(angle_proposals)
        count+=len(proposals)
    receipt.update(restored=True,source_artifact_sha256=runner.digest(path),proposal_count=count)
    return result,receipt


def fresh_edit_bundle(image, document, baseline, graph):
    """Rebuild hypotheses on the resumed geometry; no upstream bundle needed."""
    import cv2
    from contour_agent.topology_candidates import (_read_image, _affines, _base_source_ring,
                                                   _annotation_inventory)
    from contour_agent.ocr import canonical_records
    from contour_agent.parametric_pipeline import _canonical_sha256
    from contour_agent.constraint_binding import _angle_source_observations
    gray=cv2.cvtColor(_read_image(image),cv2.COLOR_BGR2GRAY)
    transform,_,_=_affines(graph,baseline)
    grid=float(graph.get('source_grid_pitch_px') or 1.)
    inventory,_=_annotation_inventory(gray,canonical_records(document),
        _base_source_ring(graph,transform,grid),grid,
        [*(graph.get('annotation_support') or []),*(graph.get('radius_source_segment_hypotheses') or [])])
    angles=_angle_source_observations(gray,canonical_records(document),graph,transform,
        max(4.,float(graph.get('proposal_tolerance_px') or max(gray.shape)/500)))
    return dict(source_sha256=graph['source_sha256'],base_graph_sha256=_canonical_sha256(graph),
                annotation_inventory=inventory,ground_truth_used=False,
                angle_source_observations=angles,
                scope='Fresh original-image evidence on the matched resumed source geometry')


def check_edit_obligations(image, document, baseline, parent, constraints, candidates, output):
    """Every edit must freshly retain the known source subset, including fillets."""
    from contour_agent.constraint_binding import analyze_constraint_bindings
    from contour_agent.source_topology_initial_guess import required_constraints_preserved
    receipts={}
    for candidate in candidates:
        directory=Path(output)/candidate['id']
        try:
            bindings=analyze_constraint_bindings(image,document,baseline,candidate['graph'],directory,use_api=False)
            receipts[candidate['id']]=required_constraints_preserved(parent,candidate['graph'],
                constraints,bindings['constraints'])
        except (ValueError,TypeError,KeyError,ArithmeticError,OSError) as error:
            receipts[candidate['id']]={'passed':False,'reason':'fresh_known_obligation_check_failed',
                'error_type':type(error).__name__}
    return receipts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "code-root", "config-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--provider-profile", default="9ecode-gpt-5.6-sol")
    parser.add_argument("--api-timeout", type=float, default=180.)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--target-tolerance-mm", type=float, default=1.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.target_tolerance_mm) or args.target_tolerance_mm <= 0:
        parser.error("target-tolerance-mm must be finite and positive")
    if not 5 <= args.api_timeout <= 600:
        parser.error("api-timeout must be between 5 and 600 seconds")
    source, config, run = args.source_run.resolve(), args.config_root.resolve(strict=True), args.output.resolve()
    if run.exists() or not run.is_relative_to(config / "runtime" / "cad-controlled-20261008"):
        parser.error("use a fresh output directory inside the controlled experiment root")
    upstream = validate_upstream(source)
    run.mkdir(parents=True)
    state = {"schema_version": "controlled-cad-iteration-v1", "label": run.name,
        "created_at": runner.stamp(), "status": "initializing", "stage_history": [],
        "execution_scope": "source_object_repair_online_final_stage_replay", "upstream_reconstruction_rerun": False,
        "upstream_run": str(source), "upstream_manifest_sha256": runner.digest(source / "run-manifest.json"),
        "oracle_mask_conditioned": True, "held_out": False, "calibration_or_development": True,
        "ground_truth_coordinates_sent_to_provider": False, "ground_truth_dxf_used_as_prediction_geometry": False,
        "evaluation_requested": args.evaluate, "frozen_strict_evaluation_tolerance_mm": .1,
        "additional_evaluation_tolerance_mm": args.target_tolerance_mm,
        "preserved_comparison_tolerances_mm": [.1, 1.],
        "scope": "Reused frozen topology and OCR review; unresolved arrows are freshly inspected online after source repairs. Local source-nominal object repairs and final online binding, solving and export are rerun."}
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
        document,state['frozen_radius_proposal_recovery']=restore_frozen_radius_proposals(
            inputs['source_image'],document,source)
        baseline = runner.read(source / "after" / "baseline-model.json")
        from contour_agent.reconstruction_priority import source_first_model
        baseline=source_first_model(baseline)
        graph = runner.read(source / "after" / "topology.json")
        trusted = runner.read(source / "after" / "topology-plan.json").get("selected_preflight_artifact")
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
        from contour_agent.topology_editing import propose_annotation_arc_edits, execute_topology_edits
        from contour_agent.parametric_pipeline import _topology_source_validation
        previous_bindings=runner.read(source/'after/constraint-bindings.json')
        previous_solution=runner.read(source/'after/parametric-solution.json')
        previous_inventory=runner.read(source/'after/binding-candidates.json')
        from contour_agent.source_state import resume_source_state
        graph=resume_source_state(inputs['source_image'],document,baseline,graph,previous_bindings,
            previous_solution,previous_inventory,after/'source-state-resume')
        trusted=after/'source-state-resume'
        state['state_resume']={'method':'matched_certified_source_state_v1',
            'original_optimization_domain_recentered':False,'source_annotations_before_mask':True}
        providers,state["provider"]=runner.providers(config,True,args.provider_profile,args.api_timeout)
        bound_angles={r.get('record_id') for r in previous_bindings.get('constraints',[]) if r.get('kind')=='angle'}
        pending_angles=[r['id'] for r in canonical_records(document)
            if r.get('parsed',{}).get('kind')=='angle' and r['id'] not in bound_angles]
        if pending_angles:
            from contour_agent.angle_target_provider import AngleTargetProvider,locate_angle_targets
            checkpoint('multimodal_angle_source_localization')
            document,receipt=locate_angle_targets(inputs['source_image'],document,after/'multimodal-angle-source',
                AngleTargetProvider(providers['editor_provider'].settings),record_ids=pending_angles)
            state['angle_target_provider']={k:v for k,v in receipt.items() if k!='proposals'}
        graph['angle_source_observations']=copy.deepcopy(previous_bindings.get('angle_source_observations',[]))
        source_bundle=fresh_edit_bundle(inputs['source_image'],document,baseline,graph)
        graph['angle_source_observations']=copy.deepcopy(source_bundle['angle_source_observations'])
        from contour_agent.parametric_pipeline import _canonical_sha256
        source_bundle['base_graph_sha256']=_canonical_sha256(graph)
        runner.write(after/'resumed-edit-source-bundle.json',source_bundle)
        unresolved={row['record_id'] for row in previous_bindings.get('radius_binding_coverage',{}).get('unresolved',[])}
        operations=[row for row in propose_annotation_arc_edits(graph,source_bundle['annotation_inventory'],limit=4,
            unresolved_radius_record_ids=unresolved) if row['action'] in ('restore_annotated_line_support','insert_annotated_fillet')]
        repair_dir=after/'source-object-repairs';repair_dir.mkdir()
        checkpoint('source_object_repairs')
        candidates,execution=execute_topology_edits(inputs['source_image'],document,baseline,
            {'id':'frozen-upstream-source-topology','graph':graph},source_bundle,operations,repair_dir)
        runner.write(repair_dir/'execution.json',execution)
        runner.write(repair_dir/'candidates.json',candidates)
        checks={row['id']:_topology_source_validation(inputs['source_image'],baseline,row['graph']) for row in candidates}
        # Preserve the rejected unsolved fit. A successful bounded numerical
        # preflight creates a separately checked graph; it never rewrites the
        # old failure into a pass or grants publication without online binding.
        runner.write(repair_dir/'initial-source-checks.json',checks)
        candidates,seed_receipt=refine_initial_line_candidates(inputs['source_image'],document,baseline,
            graph,previous_bindings.get('constraints') or [],candidates,checks,repair_dir/'initial-guess')
        runner.write(repair_dir/'initial-guess-refinement.json',seed_receipt)
        checks={row['id']:_topology_source_validation(inputs['source_image'],baseline,row['graph']) for row in candidates}
        runner.write(repair_dir/'materialized-candidates.json',candidates)
        valid=[row for row in candidates if checks[row['id']].get('passed') is True]
        # Source-owned LINE edits must preserve the known subset in a joint
        # solve even when mask fidelity is now a diagnostic rather than a veto.
        valid=[row for row in valid if not (row['graph'].get('source_evidence') or {}).get('topology_edit',{}).get(
            'action')=='restore_annotated_line_support' or
            (row['graph'].get('source_evidence') or {}).get('source_line_initial_guess')]
        retained=check_edit_obligations(inputs['source_image'],document,baseline,graph,
            previous_bindings.get('constraints') or [],valid,repair_dir/'known-obligations')
        runner.write(repair_dir/'known-obligations.json',retained)
        valid=[row for row in valid if retained[row['id']]['passed']]
        runner.write(repair_dir/'source-checks.json',checks)
        state['source_repairs']={'proposed':len(operations),'constructed_candidates':len(candidates),
            'source_valid_candidates':len(valid),'selected_candidate_id':None,'numerical_certified':False,
            'initial_guess_refinement':seed_receipt,
            'selection_scope':'Source-only local construction and original gates; no reference count or score used.'}
        if valid:
            # A combined candidate contains independent, non-overlapping source edits.
            selected=next((row for row in valid if row['id']=='cand-edit-all'),valid[0])
            graph=selected['graph'];trusted=None
            state['source_repairs']['selected_candidate_id']=selected['id']
        runner.write(run / "inputs" / "source-ocr-with-upstream-proposals.json", document)
        state["status"] = "running"
        checkpoint("frozen_topology_online_final_stage")
        model, stage = refine_parametric(inputs["source_image"], document, baseline, after,
            provider=providers["provider"], editor_provider=providers["editor_provider"], use_api=True, progress=checkpoint,
            frozen_topology=graph, trusted_preflight_dir=trusted)
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
                "--dataset", str(config / "__dataset"), "--case-id", state["case_id"], "--target-tolerance-mm", str(args.target_tolerance_mm), "--comparison-tolerance-mm", "1"],
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
