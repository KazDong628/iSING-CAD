"""Reobserve source obligations after a certified solve, in the original domain.

Only initial values move. New observations need fresh binding, all established
source equations need a carried certificate, and the entire enlarged problem
must pass the original source gates before replacing the previous solution.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path


def complete_source_constraints(image_path, document, baseline, graph, bindings,
                                solution, inventory, output, *, provider=None,
                                use_api=False, max_rounds=2, solver=None,
                                binding_analyzer=None, source_validator=None,
                                numerical_refiner=None):
    from .constraint_binding import analyze_constraint_bindings, radius_binding_coverage
    from .parametric_pipeline import (_write, _constraint_signatures,
        _constraint_receipts_complete, _solve_with_source_observation, _solved_source_validation,
        _final_source_multistart)
    from .parametric_solver import solve_parametric
    from .source_radius_hypotheses import (graph_from_solution,
        carried_binding_certificate, merge_carried_bindings)

    if type(max_rounds) is not int or not 0 <= max_rounds <= 2:
        raise ValueError('Source completion requires at most two bounded rounds')
    analyzer = binding_analyzer or analyze_constraint_bindings
    validate = source_validator or _solved_source_validation
    solver = solver or solve_parametric
    refiner = numerical_refiner or _final_source_multistart
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema_version='post-solve-source-completion-v1', accepted=False,
        maximum_rounds=max_rounds, maximum_round_solver_seconds=90., rounds=[],
        ground_truth_used=False, optimization_domain_recentered=False,
        original_graph_and_mask_immutable=True, acceptance_thresholds_changed=False,
        publication_approved=False)
    current_bindings, current_solution, current_inventory = bindings, solution, inventory
    def finish(reason):
        receipt['reason'] = reason
        _write(output/'completion.json', receipt)
        return current_bindings, current_solution, current_inventory, receipt
    if (solution.get('accepted') is not True or
            not _constraint_receipts_complete(bindings, solution, graph)):
        return finish('previous_subset_not_jointly_certified')
    initial = validate(image_path, document, baseline, graph, solution['entities'])
    receipt['initial_original_source_validation'] = initial
    if initial.get('passed') is not True:
        return finish('previous_subset_original_source_check_failed')
    for number in range(max_rounds):
        directory = output/f'round-{number+1:02d}'
        directory.mkdir(parents=True, exist_ok=True)
        row = dict(round=number+1, accepted=False, status='source_reobservation',
                   known_constraint_count=len(current_bindings['constraints']))
        receipt['rounds'].append(row)
        _write(output/'completion.json', receipt)
        started = time.monotonic()
        try:
            observed_graph = graph_from_solution(graph, baseline, current_solution)
            # Geometry is merely a new inspection/initialization state. In
            # particular the optimizer below always receives the FIRST graph.
            _write(directory/'observation-topology.json', observed_graph)
            fresh = analyzer(image_path, document, baseline, observed_graph,
                directory, provider=provider, use_api=use_api)
            fresh_inventory = json.loads(Path(fresh['inventory_artifact']).read_text(encoding='utf8'))
            row['provider'] = copy.deepcopy(fresh.get('provider', {}))
            carry = carried_binding_certificate(image_path, document, baseline, graph,
                observed_graph, current_inventory, current_bindings['constraints'],
                fresh_inventory, fresh, current_solution)
            _write(directory/'known-binding-certificate.json', carry)
            if carry.get('passed') is not True:
                row.update(status='retained', reason='known_source_certificate_failed')
                return finish('new_observation_cannot_preserve_known_subset')
            fresh, fresh_inventory = merge_carried_bindings(current_inventory, fresh_inventory, fresh, carry)
            known = _constraint_signatures(graph, current_bindings['constraints'])
            proposed = _constraint_signatures(graph, fresh['constraints'])
            if not known <= proposed:
                row.update(status='retained', reason='known_constraint_regression')
                return finish('known_constraint_regression')
            row['added_constraint_count'] = len(proposed-known)
            if not proposed-known:
                row.update(status='retained', reason='no_new_source_admitted_constraint')
                return finish('no_new_source_admitted_constraint')
            bound = {c.get('record_id') for c in fresh['constraints']} - {None}
            counts = fresh.setdefault('counts', {})
            counts.update(constraints=len(fresh['constraints']), bound_source_records=len(bound),
                bound_angle_records=sum(c['kind']=='angle' for c in fresh['constraints']),
                carried_source_constraints=len(carry['constraints']))
            if 'recognized_dimensions' in counts:
                counts['unbound_dimensions'] = counts['recognized_dimensions']-len(bound)
            fresh['radius_binding_coverage'] = radius_binding_coverage(
                fresh_inventory, observed_graph, fresh['constraints'], fresh['bindings'])
            fresh['inventory_artifact'] = str(directory/'binding-candidates-certified.json')
            _write(Path(fresh['inventory_artifact']), fresh_inventory)
            _write(directory/'constraint-bindings-certified.json', fresh)
            row['status'] = 'joint_solve'
            _write(output/'completion.json', receipt)
            from .reconstruction_priority import mask_is_hard
            ink = None
            if not mask_is_hard(baseline):
                from .source_stroke_prior import source_stroke_observation
                ink = source_stroke_observation(image_path, document, baseline, graph,
                    probe_entities=current_solution['entities'])
            row['original_image_observation_supplied_to_solver'] = ink is not None
            solve_deadline=time.monotonic()+90.
            candidate = _solve_with_source_observation(solver, graph, fresh['constraints'],
                baseline, directory/'joint-solve', budget_seconds=45.,
                seed_entities=current_solution['entities'], image_stroke_prior=ink)
            _write(directory/'parametric-solution.json', candidate)
            row['solver_status'] = candidate.get('status')
            # A failed first iterate is not a reason to unbind a known R or
            # angle. Try another unknown-attribute initialization of the SAME
            # enlarged equations. This shares the round's original 90 s
            # ceiling and never moves the input graph or source gates.
            certified=(candidate.get('accepted') is True and
                _constraint_receipts_complete(fresh, candidate, graph))
            check=(validate(image_path, document, baseline, graph, candidate['entities'])
                   if certified else {'passed':False,'reasons':['joint_subset_not_certified']})
            _write(directory/'first-attempt-source-validation.json',check)
            remaining=solve_deadline-time.monotonic()
            if not check.get('passed') and remaining>5.:
                from .parametric_solver import numerical_problem_identity
                saved=(candidate.get('diagnostics') or {}).get('numerical_continuation_checkpoint') or {}
                resumable=(saved.get('available') is True and saved.get('seed_only') is True
                    and saved.get('publication_approved') is False
                    and saved.get('problem_sha256')==numerical_problem_identity(graph,fresh['constraints']))
                row['fixed_annotation_recovery']={'attempted':True,
                    'maximum_wall_seconds':remaining,'known_equations_changed':False,
                    'original_domain_recentered':False,'publication_approved':False}
                _write(output/'completion.json',receipt)
                candidate=refiner(image_path, document, baseline, graph, fresh['constraints'],
                    directory/'fixed-annotation-recovery',solver=solver,
                    budget_seconds=remaining,initial_seed_entities=(candidate['entities'] if certified else
                        candidate['candidate_entities'] if resumable else current_solution['entities']))
                row['fixed_annotation_recovery']['continued_unpublished_iterate']=resumable
                _write(directory/'parametric-solution.json',candidate)
                row['solver_status']=candidate.get('status')
                certified=(candidate.get('accepted') is True and
                    _constraint_receipts_complete(fresh,candidate,graph))
                check=(validate(image_path,document,baseline,graph,candidate['entities']) if certified
                       else {'passed':False,'reasons':['joint_subset_not_certified']})
                row['fixed_annotation_recovery']['source_gate_passed']=check.get('passed') is True
            if (candidate.get('accepted') is not True or
                    not _constraint_receipts_complete(fresh, candidate, graph)):
                row.update(status='retained', reason='enlarged_joint_solve_not_certified')
                return finish('enlarged_joint_solve_not_certified')
            before_dof = (current_solution.get('diagnostics') or {}).get('remaining_shape_dof')
            after_dof = (candidate.get('diagnostics') or {}).get('remaining_shape_dof')
            row.update(shape_dof_before=before_dof, shape_dof_after=after_dof)
            if (type(before_dof) in (int,float) and type(after_dof) in (int,float)
                    and after_dof > before_dof):
                row.update(status='retained', reason='physical_shape_dof_increased')
                return finish('physical_shape_dof_increased')
            _write(directory/'original-source-validation.json', check)
            if check.get('passed') is not True:
                row.update(status='retained', reason='original_source_validation_failed')
                return finish('original_source_validation_failed')
            current_bindings, current_solution, current_inventory = fresh, candidate, fresh_inventory
            row.update(status='certified', accepted=True,
                all_known_constraints_preserved=True, current_API_reconfirmed_carried_constraints=False)
            receipt['accepted'] = True
        except InterruptedError:
            row.update(status='interrupted', reason='source_completion_interrupted')
            _write(output/'completion.json', receipt)
            raise
        except Exception as error:
            # Provider errors must not expose credentials or discard artifacts.
            row.update(status='retained', reason='source_completion_failed', failure_type=type(error).__name__)
            return finish('source_completion_failed_previous_valid_subset_retained')
        finally:
            row['elapsed_seconds'] = time.monotonic()-started
            _write(output/'completion.json', receipt)
    return finish('bounded_source_completion_finished')
