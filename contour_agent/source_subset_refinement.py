"""Continue a feasible but unconverged source problem without changing equations."""
from __future__ import annotations

import copy
import math
from pathlib import Path


def refine_source_subset(image, document, baseline, graph, bindings, solution, output, *,
                         solver=None, source_validator=None, exporter=None,
                         ink_observation=None, budget_seconds=45.):
    from .parametric_pipeline import (_write, _constraint_receipts_complete,
        _solve_with_source_observation, _solved_source_validation, export_parametric)
    from .parametric_solver import solve_parametric
    from .source_stroke_prior import source_stroke_observation, stroke_prior

    budget = float(budget_seconds)
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError('Source refinement needs a positive bounded budget')
    budget = min(90., budget)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema_version='fixed-source-subset-refinement-v1', accepted=False,
        maximum_solver_seconds=budget, original_graph_and_equations_unchanged=True,
        ground_truth_used=False, source_targets_frozen=True, thresholds_changed=False)
    def finish(reason, selected=solution):
        receipt['reason'] = reason
        _write(output/'refinement.json', receipt)
        return selected, receipt
    if (solution.get('accepted') is not True or
            not _constraint_receipts_complete(bindings, solution, graph)):
        return finish('previous_subset_not_certified')
    validate = source_validator or _solved_source_validation
    export = exporter or export_parametric
    try:
        initial_check = validate(image, document, baseline, graph, solution['entities'])
        if initial_check.get('passed') is not True:
            return finish('previous_source_gate_failed')
        ink = (ink_observation if ink_observation is not None else
            source_stroke_observation(image, document, baseline, graph, probe_entities=solution['entities']))
        # The SAME finite original-image targets and quadrature weights score
        # both shapes. Re-extracting targets after each step would move the
        # objective and could manufacture an apparent improvement.
        residual, evidence = stroke_prior(ink, graph, graph['entities'], 1.)
        receipt['original_image_objective'] = evidence
        if residual is None or not evidence.get('enabled'):
            return finish('original_image_objective_unavailable')
        def score(entities):
            values = residual(entities)
            value = float(values@values/2)
            if not math.isfinite(value):
                raise ValueError('nonfinite_source_refinement_score')
            return value
        before = score(solution['entities'])
        receipt['source_cost_before'] = before
        candidate = _solve_with_source_observation(solver or solve_parametric, graph,
            bindings['constraints'], baseline, output/'solve', budget_seconds=budget,
            seed_entities=solution['entities'], image_stroke_prior=ink)
        _write(output/'candidate-solution.json', candidate)
        receipt['optimizer_converged'] = candidate.get('diagnostics', {}).get('optimizer_converged')
        if (candidate.get('accepted') is not True or
                not _constraint_receipts_complete(bindings, candidate, graph)):
            return finish('candidate_subset_not_certified')
        after = score(candidate['entities'])
        receipt['source_cost_after'] = after
        if after >= before-max(1e-10, abs(before)*1e-8):
            return finish('frozen_original_image_objective_not_improved')
        check = validate(image, document, baseline, graph, candidate['entities'])
        _write(output/'source-validation.json', check)
        if check.get('passed') is not True:
            return finish('unchanged_source_gate_failed')
        old_dof = solution.get('diagnostics', {}).get('remaining_shape_dof')
        new_dof = candidate.get('diagnostics', {}).get('remaining_shape_dof')
        if isinstance(old_dof, (int, float)) and isinstance(new_dof, (int, float)) and new_dof > old_dof:
            return finish('physical_shape_dof_increased')
        model = export(image, baseline, {**candidate, 'constraints': copy.deepcopy(bindings['constraints'])},
                       output/'native-check')
        native = model['validation']
        receipt['native_checks'] = {k: native.get(k, {}).get('passed') is True and
            native.get(k, {}).get('dxf_readback_performed') is True for k in
            ('exact_radius_validation', 'strict_relation_validation', 'native_measurement_validation',
             'native_source_angle_validation', 'native_axis_relation_validation')}
        if not all(receipt['native_checks'].values()):
            return finish('native_equations_not_certified')
        receipt['accepted'] = True
        return finish('frozen_original_image_objective_improved_all_known_equations_preserved', candidate)
    except InterruptedError:
        finish('interrupted')
        raise
    except Exception as error:
        receipt['failure_type'] = type(error).__name__
        return finish('optional_refinement_failed_previous_subset_retained')
