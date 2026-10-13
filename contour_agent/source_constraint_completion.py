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

from .execution_control import execution_checkpoint


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
    execution_checkpoint('source_completion.begin')
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
    execution_checkpoint('source_completion.initial_validation.before')
    initial = validate(image_path, document, baseline, graph, solution['entities'])
    execution_checkpoint('source_completion.initial_validation.after')
    receipt['initial_original_source_validation'] = initial
    if initial.get('passed') is not True:
        return finish('previous_subset_original_source_check_failed')
    for number in range(max_rounds):
        execution_checkpoint('source_completion.round.before')
        directory = output/f'round-{number+1:02d}'
        directory.mkdir(parents=True, exist_ok=True)
        row = dict(round=number+1, accepted=False, status='source_reobservation',
                   known_constraint_count=len(current_bindings['constraints']))
        receipt['rounds'].append(row)
        _write(output/'completion.json', receipt)
        started = time.monotonic()
        try:
            execution_checkpoint('source_completion.observation.before')
            observed_graph = graph_from_solution(graph, baseline, current_solution)
            execution_checkpoint('source_completion.observation.after')
            # Geometry is merely a new inspection/initialization state. In
            # particular the optimizer below always receives the FIRST graph.
            _write(directory/'observation-topology.json', observed_graph)
            execution_checkpoint('source_completion.binding.before')
            fresh = analyzer(image_path, document, baseline, observed_graph,
                directory, provider=provider, use_api=use_api,
                **({'source_support_graph':graph} if binding_analyzer is None else {}))
            execution_checkpoint('source_completion.binding.after')
            fresh_inventory = json.loads(Path(fresh['inventory_artifact']).read_text(encoding='utf8'))
            row['provider'] = copy.deepcopy(fresh.get('provider', {}))
            execution_checkpoint('source_completion.certificate.before')
            carry = carried_binding_certificate(image_path, document, baseline, graph,
                observed_graph, current_inventory, current_bindings['constraints'],
                fresh_inventory, fresh, current_solution)
            execution_checkpoint('source_completion.certificate.after')
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
                if ((current_solution.get('diagnostics') or {}).get('optimizer_converged') is False
                        and current_solution.get('underconstrained') is True):
                    from .source_subset_refinement import refine_source_subset
                    execution_checkpoint('source_completion.known_refinement.before')
                    refined, audit = refine_source_subset(image_path, document, baseline, graph,
                        current_bindings, current_solution, directory/'known-subset-refinement',
                        solver=solver, source_validator=validate, budget_seconds=90.)
                    execution_checkpoint('source_completion.known_refinement.after')
                    row['known_subset_refinement'] = audit
                    if audit.get('accepted'):
                        current_solution = refined
                        receipt['accepted'] = True
                        row.update(status='certified', accepted=True, all_known_constraints_preserved=True)
                        return finish('known_equations_preserved_unknown_attributes_refined')
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
                execution_checkpoint('source_completion.ink.before')
                ink = source_stroke_observation(image_path, document, baseline, graph,
                    probe_entities=current_solution['entities'])
                execution_checkpoint('source_completion.ink.after')
            row['original_image_observation_supplied_to_solver'] = ink is not None
            solve_deadline=time.monotonic()+90.
            execution_checkpoint('source_completion.solve.before')
            candidate = _solve_with_source_observation(solver, graph, fresh['constraints'],
                baseline, directory/'joint-solve', budget_seconds=45.,
                seed_entities=current_solution['entities'], image_stroke_prior=ink)
            execution_checkpoint('source_completion.solve.after')
            _write(directory/'parametric-solution.json', candidate)
            row['solver_status'] = candidate.get('status')
            # A failed first iterate is not a reason to unbind a known R or
            # angle. Try another unknown-attribute initialization of the SAME
            # enlarged equations. This shares the round's original 90 s
            # ceiling and never moves the input graph or source gates.
            certified=(candidate.get('accepted') is True and
                _constraint_receipts_complete(fresh, candidate, graph))
            execution_checkpoint('source_completion.validation.before')
            check=(validate(image_path, document, baseline, graph, candidate['entities'])
                   if certified else {'passed':False,'reasons':['joint_subset_not_certified']})
            execution_checkpoint('source_completion.validation.after')
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
                execution_checkpoint('source_completion.recovery.before')
                candidate=refiner(image_path, document, baseline, graph, fresh['constraints'],
                    directory/'fixed-annotation-recovery',solver=solver,
                    budget_seconds=remaining,initial_seed_entities=(candidate['entities'] if certified else
                        candidate['candidate_entities'] if resumable else current_solution['entities']))
                execution_checkpoint('source_completion.recovery.after')
                row['fixed_annotation_recovery']['continued_unpublished_iterate']=resumable
                _write(directory/'parametric-solution.json',candidate)
                row['solver_status']=candidate.get('status')
                certified=(candidate.get('accepted') is True and
                    _constraint_receipts_complete(fresh,candidate,graph))
                execution_checkpoint('source_completion.recovery_validation.before')
                check=(validate(image_path,document,baseline,graph,candidate['entities']) if certified
                       else {'passed':False,'reasons':['joint_subset_not_certified']})
                execution_checkpoint('source_completion.recovery_validation.after')
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
            remaining = solve_deadline-time.monotonic()
            if (remaining > 7. and candidate.get('underconstrained') is True and
                    (candidate.get('diagnostics') or {}).get('optimizer_converged') is False):
                from .source_subset_refinement import refine_source_subset
                execution_checkpoint('source_completion.refinement.before')
                candidate, audit = refine_source_subset(image_path, document, baseline, graph,
                    fresh, candidate, directory/'known-subset-refinement', solver=solver,
                    source_validator=validate, ink_observation=ink, budget_seconds=remaining)
                execution_checkpoint('source_completion.refinement.after')
                row['known_subset_refinement'] = audit
            execution_checkpoint('source_completion.commit.before')
            current_bindings, current_solution, current_inventory = fresh, candidate, fresh_inventory
            row.update(status='certified', accepted=True,
                all_known_constraints_preserved=True, current_API_reconfirmed_carried_constraints=False)
            receipt['accepted'] = True
        except InterruptedError as error:
            row.update(status='interrupted', reason='source_completion_interrupted')
            stop_reason = getattr(error, 'reason', None)
            if isinstance(stop_reason, str) and stop_reason in {'cancelled', 'deadline'}:
                row['execution_stop_reason'] = stop_reason
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
