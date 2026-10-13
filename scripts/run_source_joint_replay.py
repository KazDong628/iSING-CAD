"""Online source-contact completion of an immutable upstream prediction.

This is a final-stage replay, not a fresh segmentation/end-to-end experiment.
The predictor never opens reference geometry. Evaluation is a separate process
launched only after export hashes have been frozen.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_controlled_iteration as runner
from scripts.run_source_obligation_replay import validate_upstream, restore_frozen_radius_proposals


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--provider-profile',default='9ecode-gpt-5.6-sol')
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--refresh-ocr',action='store_true',help='Fresh two-view source OCR and one source-binding completion round')
    parser.add_argument('--retry-ocr',action='store_true',help='Isolated two-view retry of withheld source OCR and one binding completion round')
    parser.add_argument('--rebind',action='store_true',help='Fresh source association and joint solve using already verified OCR')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=args.source_run.resolve(strict=True);output=args.output.resolve()
    if output.exists() or not output.is_relative_to(root/'runtime'):
        parser.error('use a fresh runtime output directory')
    upstream=validate_upstream(source);output.mkdir(parents=True)
    state=dict(schema_version='source-joint-replay-v1',case_id=upstream['case_id'],
        upstream_run=str(source),upstream_manifest_sha256=runner.digest(source/'run-manifest.json'),
        status='running',stage_history=[],source_gt_sha256=upstream.get('source_gt_sha256'),
        oracle_mask_conditioned=True,held_out=False,calibration_or_development=True,
        execution_scope='online_source_contact_final_stage_replay',fresh_end_to_end_test=False,
        ground_truth_coordinates_sent_to_provider=False,ground_truth_dxf_used_as_prediction_geometry=False,
        original_evaluation_thresholds_changed=False)
    before=runner.code_snapshot(root)
    frozen={str(p.relative_to(source)):runner.digest(p) for p in (source/'after').rglob('*') if p.is_file()}
    def checkpoint(stage):
        state['last_stage']=stage;state['stage_history'].append(dict(stage=stage,at=runner.stamp()))
        runner.write(output/'run-manifest.json',state);print(stage,flush=True)
    try:
        inputs,_=runner.freeze_inputs(source,output/'inputs')
        from contour_agent.source_ocr_review import load_effective_ocr,review_source_ocr
        from contour_agent.joint_review_provider import JointReviewProvider
        from contour_agent.source_joint_completion import complete_source_joints
        from contour_agent.parametric_pipeline import export_parametric,source_input_identity,_record_fillet_tangent_contract
        from contour_agent.reconstruction_contract import reconstruction_contract
        from contour_agent.reconstruction_priority import source_first_model
        baseline=source_first_model(runner.read(source/'after/baseline-model.json'))
        document=load_effective_ocr(inputs['source_image'],inputs['source_ocr'],source/'after')
        reviewed_document=copy.deepcopy(document)
        document,state['frozen_radius_proposal_recovery']=restore_frozen_radius_proposals(inputs['source_image'],document,source)
        graph=runner.read(source/'after/topology.json')
        bindings=runner.read(source/'after/constraint-bindings.json')
        solution=runner.read(source/'after/parametric-solution.json')
        inventory=runner.read(source/'after/binding-candidates.json')
        configured,state['provider']=runner.providers(root,True,args.provider_profile,90.)
        after=output/'after';after.mkdir()
        if not args.refresh_ocr:
            for name in ('source-ocr-review.json','effective-source-ocr.json','source-ocr-retry.json','effective-source-ocr-retry.json'):
                if (source/'after'/name).is_file():shutil.copyfile(source/'after'/name,after/name)
        if args.refresh_ocr or args.retry_ocr or args.rebind:
            from contour_agent.dataset import read_ocr
            from contour_agent.ocr import canonical_records
            from contour_agent.source_constraint_completion import complete_source_constraints
            if args.refresh_ocr:
                checkpoint('fresh_original_image_ocr_review')
                fresh,state['fresh_ocr_review']=review_source_ocr(inputs['source_image'],read_ocr(inputs['source_ocr']),
                    baseline,after,provider=configured['source_ocr_review_provider'],use_api=True,
                    raw_ocr_path=inputs['source_ocr'])
            else:fresh=reviewed_document
            if args.retry_ocr:
                from contour_agent.source_ocr_retry import retry_withheld_source_ocr
                checkpoint('isolated_withheld_source_ocr_retry')
                fresh,state['isolated_ocr_retry']=retry_withheld_source_ocr(inputs['source_image'],fresh,after,
                    provider=configured['source_ocr_review_provider'],use_api=True)
            # Preserve only source-pixel arrow hints belonging to unchanged
            # reviewed annotations; never attach an old hint to new semantics.
            old={r['id']:r for r in canonical_records(document)}
            for i,row in enumerate(canonical_records(fresh)):
                prior=old[row['id']]
                if row['text']==prior['text'] and row['box']==prior['box']:
                    for key in ('source_arrow_proposals','source_angle_proposals'):
                        if prior.get(key):fresh['records'][i][key]=copy.deepcopy(prior[key])
            document=fresh
            checkpoint('online_new_measurement_binding_and_joint_solve')
            bindings,solution,inventory,state['source_constraint_completion']=complete_source_constraints(
                inputs['source_image'],document,baseline,graph,bindings,solution,inventory,output/'source-measurements',
                provider=configured['provider'],use_api=True,max_rounds=1)
        checkpoint('online_two_view_design_review_and_joint_solve')
        bindings,solution,inventory,receipt=complete_source_joints(inputs['source_image'],document,
            baseline,graph,bindings,solution,inventory,output/'source-joints',
            provider=JointReviewProvider(configured['provider'].settings),use_api=True)
        state['source_joint_completion']=receipt
        for name,value in [('baseline-model.json',baseline),('topology.json',graph),
            ('constraint-bindings.json',bindings),('parametric-solution.json',solution),
            ('binding-candidates.json',inventory),('effective-radius-source-document.json',document)]:
            runner.write(after/name,value)
        checkpoint('native_export')
        model=export_parametric(inputs['source_image'],baseline,{**solution,'constraints':bindings['constraints']},after)
        original=runner.read(source/'after/model.json')
        stage=copy.deepcopy(original['parameterization'])
        stage.update(constraints=bindings['constraints'],binding_counts=bindings.get('counts',{}),
            solver={k:solution.get(k) for k in ('accepted','status','validation','diagnostics')},
            underconstrained=solution.get('underconstrained'),solver_constraint_checks=solution.get('constraints',[]),
            source_joint_completion=receipt,constraint_subset_accepted=True,accepted=False,
            status='completed_with_unresolved_attributes')
        # Radius coverage is unchanged by this contact-only solve, but every
        # declared R is freshly verified in native export above.
        radius=copy.deepcopy(original['validation'].get('annotation_radius_contract',{}))
        radius['exact_radius_validation']=model['validation']['exact_radius_validation']
        radius['satisfied']=bool(radius.get('satisfied') and radius['exact_radius_validation']['passed'])
        model['validation']['annotation_radius_contract']=radius
        contract=reconstruction_contract(model['entities'],stage,model['validation'])
        stage['reconstruction_contract']=contract
        stage.update(accepted=contract['satisfied'],status='completed' if contract['satisfied'] else 'completed_with_unresolved_attributes')
        model.update(parameterization=stage,source_input_identity=source_input_identity(inputs['source_image'],document))
        model['validation']['reconstruction_contract']=contract
        for name,value in [('model.json',model),('validation.json',model['validation']),
            ('parametric-stage.json',stage),('reconstruction-contract.json',contract)]:runner.write(after/name,value)
        _record_fillet_tangent_contract(after,graph,bindings,solution)
        state.update(parameterization_accepted=contract['satisfied'],prediction_sha256=runner.digest(after/'drawing.dxf'),
            prediction_frozen_at=runner.stamp(),frozen_artifacts={p.name:runner.digest(p) for p in after.iterdir() if p.is_file()},
            code_unchanged=before==runner.code_snapshot(root),upstream_unchanged=all(runner.digest(source/p)==sha for p,sha in frozen.items()))
        if not state['code_unchanged'] or not state['upstream_unchanged']:raise RuntimeError('immutable_input_changed')
        checkpoint('prediction_frozen')
        if args.evaluate:
            checkpoint('independent_post_freeze_gt_evaluation')
            result=subprocess.run([sys.executable,'-X','utf8',str(root/'scripts/evaluate_oracle_mask_run.py'),
                '--run-dir',str(output),'--dataset',str(root/'__dataset'),'--case-id',state['case_id'],
                '--target-tolerance-mm','1.5','--comparison-tolerance-mm','1'],capture_output=True,timeout=600)
            state['evaluation_exit_code']=result.returncode
            if result.returncode:raise RuntimeError('independent_evaluation_failed')
            state['prediction_unchanged_after_evaluation']=runner.digest(after/'drawing.dxf')==state['prediction_sha256']
            if not state['prediction_unchanged_after_evaluation']:raise RuntimeError('prediction_changed_after_freeze')
        state['status']=stage['status'];checkpoint('finished');return 0
    except Exception as error:
        state.update(status='failed',failure_type=type(error).__name__)
        checkpoint('failed');return 1


if __name__=='__main__':raise SystemExit(main())
