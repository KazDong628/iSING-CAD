"""Numerically refine unknown attributes using frozen online source bindings.

No provider is called, no reference DXF is read, and no upstream artifact is
replaced. This is an explicitly offline diagnostic continuation, not a fresh
online or end-to-end reconstruction test.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_controlled_iteration as runner
from scripts.recheck_frozen_final_stage import validate_upstream


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','code-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--rounds',type=int,choices=range(1,5),default=2)
    parser.add_argument('--continue-from',type=Path,
        help='Continue the source-selected, native-verified iterate of an earlier offline run; never select using GT.')
    args=parser.parse_args(argv)
    source=args.source_run.resolve(strict=True);output=args.output.resolve()
    if output.exists() or not output.is_relative_to(Path(__file__).resolve().parents[1]/'runtime'):
        parser.error('Use a fresh runtime output directory')
    validate_upstream(source)
    before=runner.code_snapshot(args.code_root)
    code=runner.select_code_root(args.code_root)
    from contour_agent.parametric_pipeline import _final_source_multistart,_solved_source_validation,export_parametric
    from contour_agent.parametric_solver import solve_parametric,native_source_nominal_angle_checks,native_measurement_checks
    from contour_agent.radius_contract import exact_radius_checks
    from contour_agent.relation_contract import relation_checks
    from contour_agent.reconstruction_priority import source_first_model
    from contour_agent.source_topology_initial_guess import line_seed_problem_graphs
    from contour_agent.topology import stroke_support_fraction

    output.mkdir(parents=True)
    state=dict(schema_version='fixed-source-bindings-refinement-v1',status='running',
        scope='Offline unknown-attribute continuation of frozen online bindings; not a new online/E2E run.',
        upstream_run=str(source),ground_truth_used=False,new_network_requests=0,
        maximum_rounds=args.rounds,maximum_wall_seconds_per_round=120.,rounds=[],
        code_sha256=before,runner_path=str(Path(__file__).resolve()),
        runner_sha256=runner.digest(Path(__file__).resolve()),
        optimization_domain_recentered=False,publication_approved=False)
    (output/'runner-source.py').write_bytes(Path(__file__).read_bytes())
    def save():runner.write(output/'refinement.json',state)
    save()
    try:
        after=source/'after';baseline=source_first_model(runner.read(after/'baseline-model.json'))
        graph=runner.read(after/'topology.json');domain,_=line_seed_problem_graphs(graph)
        document=runner.read(after/'effective-radius-source-document.json')
        image=next(p for p in (source/'inputs').glob('source-image.*') if p.is_file())
        options=[]
        for directory in (after,after/'final-binding-attempt'):
            bp,sp=directory/'constraint-bindings.json',directory/'parametric-solution.json'
            if not (bp.is_file() and sp.is_file()):continue
            bindings,solution=runner.read(bp),runner.read(sp)
            if solution.get('accepted') is not True:continue
            entities=solution['entities'];constraints=bindings.get('constraints') or []
            checks=[exact_radius_checks(entities,constraints),relation_checks(entities,constraints),
                    native_source_nominal_angle_checks(entities,constraints),native_measurement_checks(entities,constraints)]
            if not all(c.get('passed') is True for c in checks):continue
            records={(r.get('kind'),r.get('record_id')) for r in constraints if r.get('record_id')}
            rank=(sum(k=='radius' for k,r in records),sum(k=='angle' for k,r in records),len(records))
            options.append((rank,directory,bindings,solution))
        if not options:raise ValueError('no_native_feasible_online_binding_checkpoint')
        _,directory,bindings,initial=max(options,key=lambda v:v[0])
        constraints=copy.deepcopy(bindings['constraints']);seed=initial['entities']
        state.update(binding_checkpoint=str(directory),
            binding_checkpoint_sha256=runner.digest(directory/'constraint-bindings.json'),
            source_input_sha256=runner.digest(image),source_document_sha256=runner.digest(after/'effective-radius-source-document.json'),
            domain_sha256=__import__('hashlib').sha256(json.dumps(domain,sort_keys=True).encode()).hexdigest())
        if args.continue_from is not None:
            continuation=args.continue_from.resolve(strict=True)
            prior=runner.read(continuation/'refinement.json')
            if (prior.get('status')!='finished' or prior.get('known_constraint_values_changed') is not False or
                    prior.get('upstream_binding_unchanged') is not True or
                    any(prior.get(k)!=state.get(k) for k in ('upstream_run','binding_checkpoint_sha256',
                        'source_input_sha256','source_document_sha256','domain_sha256'))):
                raise ValueError('continuation_source_and_known_bindings_must_match')
            selected_round=prior.get('selected_numerical_round')
            if type(selected_round) is not int or selected_round<0:raise ValueError('invalid_continuation_round')
            path=continuation/f'round-{selected_round:02d}'/'parametric-solution.json'
            prior_solution=runner.read(path)
            seed=prior_solution.get('entities')
            if (prior_solution.get('accepted') is not True or
                    [(e.get('id'),e.get('type')) for e in seed or []]!=
                    [(e.get('id'),e.get('type')) for e in domain['entities']] or
                    not all(c.get('passed') is True for c in (exact_radius_checks(seed,constraints),
                        relation_checks(seed,constraints),native_source_nominal_angle_checks(seed,constraints),
                        native_measurement_checks(seed,constraints)))):
                raise ValueError('continuation_native_known_attributes_not_verified')
            state.update(continued_from=str(continuation),continuation_solution_sha256=runner.digest(path),
                continuation_selection='Prior source-ink-selected numerical iterate; no reference score used')
            save()
        selected=None;best=None
        if args.continue_from is not None:
            # The carried incumbent participates in selection too. Otherwise
            # every new, worse solve could silently replace the best iterate
            # simply because this process had not recorded a round yet.
            current=output/'round-00';current.mkdir()
            runner.write(current/'parametric-solution.json',prior_solution)
            check=_solved_source_validation(image,document,baseline,graph,seed)
            runner.write(current/'solved-source-validation.json',check)
            model=export_parametric(image,baseline,{**prior_solution,'constraints':constraints},current/'export')
            support=stroke_support_fraction(check['after'])
            state['rounds'].append(dict(round=0,scope='Revalidated continuation incumbent; no new numerical solve.',
                numerically_accepted=True,all_known_constraint_ids_preserved=True,
                source_gate_passed=check['passed'],source_gate_reasons=check.get('reasons'),
                stroke_supported_fraction=support,
                native_radius_checks=model['validation']['exact_radius_validation'],
                native_angle_checks=model['validation']['native_source_angle_validation'],
                native_relation_checks=model['validation']['strict_relation_validation'],
                candidate_dxf=str(current/'export/drawing.dxf')))
            selected=0;best=(bool(check['passed']),support,-check['after']['p90_edge_distance_px']);save()
        for index in range(1,args.rounds+1):
            current=output/f'round-{index:02d}'
            result=_final_source_multistart(image,document,baseline,graph,constraints,current,
                solver=solve_parametric,budget_seconds=120.,initial_seed_entities=seed)
            # A better native-verified iterate may still fail the full image
            # gate. Keep it as an explicitly unpublished next numerical seed,
            # never turn that failed gate into a successful reconstruction.
            numerical=runner.read(current/'source-multistart.json').get('unpublished_numerical_candidate') or {}
            if numerical.get('native_attributes_verified') is True and numerical.get('publication_approved') is False:
                draft=runner.read(Path(numerical['solution_path']))
                if draft.get('accepted') is True:result=draft
            runner.write(current/'parametric-solution.json',result)
            row=dict(round=index,numerically_accepted=result.get('accepted') is True,
                     all_known_constraint_ids_preserved={r['id'] for r in result.get('constraints',[])}=={r['id'] for r in constraints})
            if row['numerically_accepted'] and row['all_known_constraint_ids_preserved']:
                entities=result['entities'];check=_solved_source_validation(image,document,baseline,graph,entities)
                runner.write(current/'solved-source-validation.json',check)
                model=export_parametric(image,baseline,{**result,'constraints':constraints},current/'export')
                row.update(source_gate_passed=check['passed'],source_gate_reasons=check.get('reasons'),
                    stroke_supported_fraction=stroke_support_fraction(check['after']),
                    native_radius_checks=model['validation']['exact_radius_validation'],
                    native_angle_checks=model['validation']['native_source_angle_validation'],
                    native_relation_checks=model['validation']['strict_relation_validation'],
                    candidate_dxf=str(current/'export/drawing.dxf'))
                score=(bool(check['passed']),row['stroke_supported_fraction'],-check['after']['p90_edge_distance_px'])
                if best is None or score>best:
                    best=score;selected=index;seed=entities
                if check['passed']:
                    state['rounds'].append(row);save();break
            state['rounds'].append(row);save()
        state.update(status='finished',selected_numerical_round=selected,
            known_constraint_values_changed=False,upstream_binding_unchanged=
            runner.digest(directory/'constraint-bindings.json')==state['binding_checkpoint_sha256'],
            code_unchanged=runner.code_snapshot(code)==before)
        save();print(json.dumps({k:state[k] for k in ['status','selected_numerical_round','new_network_requests','publication_approved']}))
        return 0
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__);save()
        print(json.dumps({'status':'failed','error_type':type(error).__name__}));return 1


if __name__=='__main__':raise SystemExit(main())
