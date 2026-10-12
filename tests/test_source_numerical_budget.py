"""Source annotations can drive optimization without a mask inequality."""
from copy import deepcopy

import numpy as np
import pytest

from contour_agent.parametric_solver import solve_parametric
from test_feasibility_first import inputs


def source_first_inputs():
    graph, constraints, observation = inputs(source_width=10., tolerance=.03)
    diagnostic = observation.pop('boundary_error_budget')
    observation['mask_representation_budget_diagnostic'] = diagnostic
    observation['optimization_budget'] = {'max_wall_seconds': 30., 'max_objective_evaluations': 30000}
    constraints[0].update(required=True, enforcement='exact', nominal_source='source_ocr')
    return graph, constraints, observation


def test_exact_source_width_can_override_soft_mask_without_unbounding_solve():
    graph, constraints, observation = source_first_inputs()
    original = deepcopy((graph, constraints, observation))
    result = solve_parametric(graph, constraints, source_observation=observation)
    assert result['accepted'] and result['strict_measurement_contract']['satisfied']
    assert result['constraints'][0]['actual'] == pytest.approx(10.5, abs=1e-6)
    budget = result['diagnostics']['optimization_budget']
    assert budget['enabled'] and not budget['mask_fidelity_constraint']
    assert 0 < budget['objective_evaluations'] <= 30000
    assert not result['diagnostics']['source_budget_search']['enabled']
    assert (graph, constraints, observation) == original


def test_exhaustion_keeps_original_artifacts_and_does_not_claim_infeasibility(tmp_path):
    graph, constraints, observation = source_first_inputs()
    observation['optimization_budget']['max_objective_evaluations'] = 1
    result = solve_parametric(graph, constraints, source_observation=observation, output_dir=tmp_path)
    assert result['status'] == 'optimization_budget_exhausted' and not result['accepted']
    assert result['entities'] == graph['entities'] and result['nodes'] == graph['nodes']
    assert result['diagnostics']['optimization_budget']['exhausted']
    assert result['diagnostics']['optimization_budget']['objective_evaluations'] == 1
    assert result['diagnostics']['infeasibility_proven'] is False
    assert (tmp_path / 'parametric-solve.json').is_file()


def test_certified_restart_is_independent_from_hard_mask_equation():
    graph, constraints, observation = source_first_inputs()
    solved = solve_parametric(graph, constraints, source_observation=observation)
    restart = solve_parametric(graph, constraints, source_observation=observation,
                               seed_entities=solved['entities'])
    assert restart['accepted'] and restart['strict_measurement_contract']['satisfied']
    assert not restart['diagnostics']['source_budget_search']['enabled']
    assert restart['diagnostics']['optimization_budget']['enabled']
    wrong = deepcopy(solved['entities'])
    wrong[0]['id'] = 'different-object'
    with pytest.raises(ValueError, match='topology or traversal'):
        solve_parametric(graph, constraints, source_observation=observation, seed_entities=wrong)


def test_budget_exhaustion_revalidates_and_retains_complete_feasible_seed(tmp_path):
    graph, constraints, observation = source_first_inputs()
    solved = solve_parametric(graph, constraints, source_observation=observation)
    assert solved['accepted']
    observation['optimization_budget']['max_objective_evaluations'] = 1
    original = deepcopy((graph, constraints, observation, solved['entities']))
    result = solve_parametric(graph, constraints, source_observation=observation,
        seed_entities=solved['entities'], output_dir=tmp_path)
    assert result['accepted']
    assert not result['diagnostics']['optimizer_converged']
    assert result['diagnostics']['acceptance_basis'] == 'independently_verified_feasible_checkpoint'
    assert result['diagnostics']['optimization_budget']['exhausted']
    assert result['diagnostics']['optimization_budget']['objective_evaluations'] == 1
    assert result['strict_measurement_contract']['satisfied']
    assert all(row['passed'] for row in result['constraints'])
    assert result['entities'] == solved['entities']
    receipt = result['diagnostics']['numerical_feasible_checkpoint']
    assert receipt['selected'] and receipt['complete_known_subset_rechecked']
    assert receipt['original_source_publication_gate_still_required']
    assert not receipt['original_domain_recentered']
    assert not result['diagnostics']['optimality_proven']
    assert (graph, constraints, observation, solved['entities']) == original


def test_saved_seed_cannot_satisfy_new_changed_source_equation_by_old_certificate():
    graph, constraints, observation = source_first_inputs()
    solved = solve_parametric(graph, constraints, source_observation=observation)
    constraints[0]['value'] += .25
    observation['optimization_budget']['max_objective_evaluations'] = 1
    result = solve_parametric(graph, constraints, source_observation=observation,
                              seed_entities=solved['entities'])
    assert not result['accepted'] and result['status'] == 'optimization_budget_exhausted'
    assert result['entities'] == graph['entities']


def test_unfinished_iterate_is_persisted_only_as_same_problem_continuation(monkeypatch):
    from contour_agent import parametric_solver as module
    graph,constraints,observation=source_first_inputs()
    solved=solve_parametric(graph,constraints,source_observation=observation)
    _,_,_,_,scale,_,_,_=module._model(graph['nodes'],graph['entities'],{})
    origin=np.array([[n['x'],n['y']] for n in graph['nodes']]).mean(axis=0)
    points={n['id']:np.array([n['x'],n['y']]) for n in solved['nodes']}
    partial=np.concatenate([((points[n['id']]+np.array([n['x'],n['y']]))/2-origin)/scale
                            for n in graph['nodes']])
    def unfinished(fun,starting,**kwargs):
        kwargs['callback'](partial)
        kwargs['callback'](np.full_like(partial,np.nan))
        raise module._NumericalBudgetExhausted('synthetic limit')
    monkeypatch.setattr(module,'minimize',unfinished)
    result=solve_parametric(graph,constraints,source_observation=observation)
    assert not result['accepted'] and result['entities']==graph['entities']
    assert result['candidate_entities'] != graph['entities']
    checkpoint=result['diagnostics']['numerical_continuation_checkpoint']
    assert checkpoint['seed_only'] and not checkpoint['known_equations_certified']
    assert not checkpoint['publication_approved']
    assert checkpoint['problem_sha256']==module.numerical_problem_identity(graph,constraints)
    changed=deepcopy(constraints);changed[0]['value']+=1
    assert checkpoint['problem_sha256']!=module.numerical_problem_identity(graph,changed)


def test_new_fixed_measurement_restores_feasibility_before_source_objective():
    graph,constraints,observation=source_first_inputs()
    old=solve_parametric(graph,constraints,source_observation=observation)
    constraints[0]['value']+=.25
    result=solve_parametric(graph,constraints,source_observation=observation,seed_entities=old['entities'])
    assert result['accepted'] and result['strict_measurement_contract']['satisfied']
    assert result['constraints'][0]['actual']==pytest.approx(10.75,abs=1e-6)
    proof=result['diagnostics']['same_problem_feasibility_projection']
    assert proof['all_known_equations_preserved'] and not proof['original_domain_recentered']
    assert not proof['publication_approved'] and not proof['thresholds_changed']


def test_valid_search_iterate_survives_later_invalid_iterate_and_interruption(monkeypatch):
    from contour_agent import parametric_solver as module
    graph, constraints, observation = source_first_inputs()
    solved = solve_parametric(graph, constraints, source_observation=observation)
    assert solved['accepted']
    _, _, _, _, scale, _, _, _ = module._model(graph['nodes'], graph['entities'], {})
    origin = np.array([[node['x'],node['y']] for node in graph['nodes']]).mean(axis=0)
    node_map = {node['id']: [node['x'],node['y']] for node in solved['nodes']}
    feasible = np.concatenate([(np.array(node_map[node['id']])-origin)/scale for node in graph['nodes']])
    def interrupted(fun, starting, **kwargs):
        kwargs['callback'](starting)  # Old fit violates the new exact width.
        kwargs['callback'](feasible)
        kwargs['callback'](np.full_like(feasible,np.nan))
        raise module._NumericalBudgetExhausted('synthetic bounded search end')
    monkeypatch.setattr(module,'minimize',interrupted)
    result = solve_parametric(graph,constraints,source_observation=observation)
    assert result['accepted'] and result['strict_measurement_contract']['satisfied']
    assert result['entities'] == solved['entities']
    assert result['diagnostics']['numerical_feasible_checkpoint']['selected']
    assert not result['diagnostics']['optimizer_converged']


def test_source_first_joint_offsets_use_sampling_receipt_without_mask_gate():
    graph, constraints, observation = source_first_inputs()
    result = solve_parametric(graph, constraints, source_observation=observation,
        seed_node_offsets={'v1': [0., .01]})
    assert result['accepted']
    assert result['diagnostics']['source_seed']['maximum_offset'] == pytest.approx(.04)
    assert not result['diagnostics']['source_budget_search']['enabled']
    with pytest.raises(ValueError, match='offset exceeds'):
        solve_parametric(graph, constraints, source_observation=observation,
            seed_node_offsets={'v1': [0., .05]})


@pytest.mark.parametrize('field,value', [('max_wall_seconds', 0), ('max_wall_seconds', 121),
    ('max_objective_evaluations', 0), ('max_objective_evaluations', True),
    ('max_objective_evaluations', 30001)])
def test_invalid_budget_is_rejected_before_optimization(field, value):
    graph, constraints, observation = source_first_inputs()
    observation['optimization_budget'][field] = value
    with pytest.raises(ValueError, match='budget'):
        solve_parametric(graph, constraints, source_observation=observation)
