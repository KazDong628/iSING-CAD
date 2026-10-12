"""Evaluate a source-selected, frozen OFFLINE candidate; never select using GT."""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_controlled_iteration as runner


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinement',type=Path,required=True)
    args=parser.parse_args(argv);directory=args.refinement.resolve(strict=True)
    root=Path(__file__).resolve().parents[1]
    if not directory.is_relative_to(root/'runtime'):parser.error('Use an existing runtime refinement')
    state=runner.read(directory/'refinement.json')
    selected=state.get('selected_numerical_round')
    if (state.get('status')!='finished' or type(selected) is not int or selected<0 or
            state.get('known_constraint_values_changed') is not False or state.get('code_unchanged') is not True):
        parser.error('Refinement must be finished with immutable source bindings and code')
    row=next(r for r in state['rounds'] if r['round']==selected)
    if not all(row[k].get('passed') is True and row[k].get('dxf_readback_performed') is True for k in
               ('native_radius_checks','native_angle_checks','native_relation_checks')):
        parser.error('Native source-attribute readback is required')
    source=directory/f'round-{selected:02d}'/'export';upstream=Path(state['upstream_run'])
    run=directory/'post-frozen-evaluation'
    if run.exists():parser.error('Evaluation directory already exists; preserve previous results')
    run.mkdir();shutil.copytree(upstream/'inputs',run/'inputs');shutil.copytree(source,run/'after')
    after=run/'after';frozen=runner.digest(after/'drawing.dxf')
    runner.write(after/'parametric-stage.json',{'status':'diagnostic_candidate_only','accepted':False,
        'publication':{'kind':'diagnostic_candidate','status':'not_published'}})
    manifest=runner.read(upstream/'run-manifest.json');inputs=runner.read(run/'inputs/input-manifest.json')
    receipt=dict(case_id=manifest['case_id'],oracle_mask_conditioned=True,held_out=False,
        execution_scope='offline_fixed_online_binding_numerical_candidate',new_network_requests=0,
        prediction_selected_by='Original source ink and native equations only; no GT score used',
        publication_approved=False,source_gate_passed=row['source_gate_passed'],
        prediction_sha256=frozen,prediction_frozen_at=runner.stamp(),
        source_gt_sha256=manifest['source_gt_sha256'],oracle_mask_sha256=inputs['mask_sha256'])
    runner.write(run/'run-manifest.json',receipt)
    # Only after the candidate is selected, copied and frozen is the independent
    # evaluator imported and allowed to open the read-only reference dataset.
    from scripts.evaluate_oracle_mask_run import evaluate_oracle_mask_run
    result=evaluate_oracle_mask_run(run,dataset_root=root/'__dataset',case_id=manifest['case_id'],
        target_tolerance_mm=1.5,comparison_tolerances_mm=(1.,))
    unchanged=runner.digest(after/'drawing.dxf')==frozen
    runner.write(run/'freeze-audit.json',{'prediction_unchanged':unchanged,'ground_truth_scores_select_prediction':False})
    if not unchanged:raise RuntimeError('Frozen prediction changed during evaluation')
    print(json.dumps({'status':result.get('status'),'native_max_error_mm':result.get('geometry',{}).get(
        'native_coordinate_frame',{}).get('conservative_max_error_mm'),'prediction_unchanged':unchanged}))
    return 0


if __name__=='__main__':raise SystemExit(main())
