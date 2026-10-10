"""Scheduling regressions exercise the real solver and unchanged source gates."""
from pathlib import Path
import copy
import importlib.util
import os
import sys

import numpy as np
import pytest

from contour_agent import parametric_solver as solver

def rectangle(width):
    points = [[0., 0.], [width, 0.], [width, 5.], [0., 5.]]
    return {'units': 'mm', 'nodes': [{'id': f'v{i}', 'x': p[0], 'y': p[1], 'source_px': p}
            for i, p in enumerate(points)], 'entities': [{'id': f'g{i}', 'type': 'LINE',
            'start_node': f'v{i}', 'end_node': f'v{(i+1)%4}', 'start': p,
            'end': points[(i+1)%4]} for i, p in enumerate(points)]}

def inputs(source_width=10.5, tolerance=.15):
    source = rectangle(source_width)
    points = [*[[n['x'], n['y']] for n in source['nodes']], [0., 0.]]
    observation = {'units': 'mm', 'points': points, 'provenance': 'input_mask_boundary',
        'reference_dxf_read': False, 'boundary_error_budget': {'units': 'mm',
        'maximum_deviation': tolerance, 'sampling_step': .02,
        'source': 'initial_curve_fit_total_deviation_budget_px'}}
    constraints = [{'id': 'width', 'kind': 'distance_x', 'nodes': ['v0', 'v1'],
        'entities': [], 'value': 10.5, 'record_id': 'r0', 'source': 'ocr_local_binding'}]
    return rectangle(10.), constraints, observation

def recording_optimizer(monkeypatch, *, exhaust_rms=False):
    calls = []
    original = solver.minimize
    def minimize(fun, *args, **kwargs):
        calls.append(fun.__name__)
        if exhaust_rms and fun.__name__ == 'scalar_objective':
            raise solver._SourceBudgetExhausted('simulated_warm_or_refinement_budget')
        return original(fun, *args, **kwargs)
    monkeypatch.setattr(solver, 'minimize', minimize)
    return calls

def test_infeasible_seed_spends_budget_on_feasibility_before_rms(monkeypatch):
    graph, constraints, observation = inputs()
    original = copy.deepcopy(graph)
    calls = recording_optimizer(monkeypatch)
    result = solver.solve_parametric(graph, constraints, source_observation=observation)
    assert calls[0] == 'feasibility_objective'
    assert result['accepted'] and result['validation']['source_boundary_budget_passed']
    search = result['diagnostics']['source_budget_search']
    assert search['warm_RMS_skipped'] and search['feasible_checkpoint_found']
    assert not search['acceptance_thresholds_changed']
    assert search['maximum_rounds'] == 5 and search['maximum_wall_seconds'] == 120.
    assert graph == original

def test_feasible_seed_keeps_existing_warm_path(monkeypatch):
    graph, constraints, observation = inputs(source_width=10., tolerance=.8)
    calls = recording_optimizer(monkeypatch)
    result = solver.solve_parametric(graph, constraints, source_observation=observation)
    assert calls[0] == 'scalar_objective'
    assert result['accepted']
    assert result['diagnostics']['source_budget_search']['initial_exact_radius_seed_audit']['passed']
    assert not result['diagnostics']['source_budget_search'].get('warm_RMS_skipped', False)

def test_rms_budget_exhaustion_preserves_certified_feasible_checkpoint(monkeypatch):
    graph, constraints, observation = inputs()
    calls = recording_optimizer(monkeypatch, exhaust_rms=True)
    result = solver.solve_parametric(graph, constraints, source_observation=observation)
    assert calls[0] == 'feasibility_objective'
    assert result['accepted'] and all(c['passed'] for c in result['constraints'])
    search = result['diagnostics']['source_budget_search']
    assert search['feasible_checkpoint_found'] and search['final_audit']['passed']
    assert not search['RMS_refinement_accepted']

def test_incompatible_annotation_still_retains_original_graph():
    graph, constraints, observation = inputs(source_width=10., tolerance=.03)
    original = copy.deepcopy(graph)
    result = solver.solve_parametric(graph, constraints, source_observation=observation)
    assert not result['accepted']
    assert result['entities'] == original['entities'] and result['nodes'] == original['nodes']
    search = result['diagnostics']['source_budget_search']
    assert not search['infeasibility_proven'] and not search['acceptance_thresholds_changed']

def test_iteration_limited_feasible_point_is_independently_certified(monkeypatch, tmp_path):
    graph, constraints, observation = inputs()
    actual_minimize = solver.minimize
    calls = []
    def limit_after_completed_iteration(fun, *args, **kwargs):
        calls.append(fun.__name__)
        if fun.__name__ == 'scalar_objective':
            raise solver._SourceBudgetExhausted('stop_optional_rms')
        fit = actual_minimize(fun, *args, **kwargs)
        if fun.__name__ == 'feasibility_objective':
            fit.success = False
            fit.status = 9
            fit.message = 'Iteration limit reached'
        return fit
    monkeypatch.setattr(solver, 'minimize', limit_after_completed_iteration)
    result = solver.solve_parametric(graph, constraints, source_observation=observation, output_dir=tmp_path)
    assert result['accepted'] and result['status'] == 'accepted'
    assert calls == ['feasibility_objective']
    assert all(row['passed'] for row in result['constraints'])
    assert result['validation']['geometry_valid'] and result['validation']['source_boundary_budget_passed']
    diagnostics = result['diagnostics']
    assert diagnostics['optimizer_converged'] is False
    assert diagnostics['optimality_proven'] is False
    assert diagnostics['acceptance_basis'] == 'independently_verified_feasible_checkpoint'
    assert diagnostics['source_budget_search']['rounds'][0]['feasibility_converged'] is False
    assert diagnostics['source_budget_search']['rounds'][0]['feasibility_found'] is True
    assert diagnostics['source_budget_search']['RMS_refinement_skipped_reason'] == 'feasible_iteration_limit_checkpoint'
    checkpoint = __import__('json').loads((tmp_path/'source-budget-checkpoint.json').read_text(encoding='utf8'))
    assert checkpoint['optimizer_converged'] is False
    assert checkpoint['optimality_proven'] is False

def test_iteration_limited_point_failing_independent_checks_retains_baseline(monkeypatch):
    graph, constraints, observation = inputs()
    original = copy.deepcopy(graph)
    def return_unrepaired_seed(fun, x0, **kwargs):
        assert fun.__name__ == 'feasibility_objective'
        return solver.OptimizeResult(x=np.asarray(x0).copy(), jac=np.zeros_like(x0),
                                     success=False, status=9, nfev=1, nit=120,
                                     message='Iteration limit reached')
    monkeypatch.setattr(solver, 'minimize', return_unrepaired_seed)
    result = solver.solve_parametric(graph, constraints, source_observation=observation)
    assert not result['accepted']
    assert result['entities'] == original['entities']
    assert result['nodes'] == original['nodes']
    assert not result['diagnostics']['source_budget_search']['feasible_checkpoint_found']
