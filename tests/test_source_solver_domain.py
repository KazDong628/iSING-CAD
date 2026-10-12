"""Original displacement bounds guide search without changing certification."""
from copy import deepcopy

import numpy as np
import pytest

from contour_agent import parametric_solver as solver
from test_adaptive_source_refinement import observation, polygon, width_constraint
from test_feasibility_first import inputs


def test_source_search_exposes_only_existing_node_movement_box(monkeypatch):
    graph, constraints, raw = inputs()
    before = deepcopy((graph, constraints, raw))
    initial, _, _, _, scale, *_ = solver._model(graph["nodes"], graph["entities"])
    expected = max(.25, scale * solver.MAX_DISPLACEMENT_FRACTION) / scale
    actual_minimize = solver.minimize
    calls = []

    def inspect_bounds(fun, x0, **kwargs):
        bounds = np.asarray(kwargs["bounds"])
        count = 2 * len(graph["nodes"])
        assert bounds[:count, 0] == pytest.approx(initial[:count] - expected)
        assert bounds[:count, 1] == pytest.approx(initial[:count] + expected)
        calls.append(fun.__name__)
        return actual_minimize(fun, x0, **kwargs)

    monkeypatch.setattr(solver, "minimize", inspect_bounds)
    result = solver.solve_parametric(graph, constraints, source_observation=raw)
    assert result["accepted"] and calls
    assert result["validation"]["source_displacement_passed"]
    assert result["validation"]["source_boundary_budget_passed"]
    search = result["diagnostics"]["source_budget_search"]
    assert search["node_coordinate_box"]["guard_implied_bound_only"]
    assert search["node_coordinate_box"]["radial_movement_guard_still_active"]
    assert search["maximum_deviation"] == raw["boundary_error_budget"]["maximum_deviation"]
    assert not search["acceptance_thresholds_changed"]
    assert (graph, constraints, raw) == before


def test_invalid_later_trial_keeps_valid_rejected_checkpoint_for_restart(monkeypatch):
    # The first real solve fixes an exact width but cannot cover every source
    # peak in one bounded witness batch. A deliberately invalid later solver
    # return must not erase that strict-equation/geometry checkpoint or turn it
    # into a certified source solution.
    monkeypatch.setattr(solver._SourceBoundaryBudget, "witnesses_per_direction", 1)
    width = 3.5
    points = [[0., 0.], [width, 0.], [width, 1.]] + [
        [i * .25, 1. + (.2 if i % 2 else 0.)] for i in range(13, -1, -1)]
    graph = polygon([[x, min(y, 1.)] for x, y in points])
    raw = observation(points)
    before = deepcopy(graph)
    original = solver.minimize
    calls = []

    def invalid_second_trial(fun, x0, **kwargs):
        calls.append(fun.__name__)
        if len(calls) == 2:
            # A failed line search may return an out-of-domain iterate. This
            # return is never allowed to replace the last useful source seed.
            bad = np.asarray(x0).copy() + 2.
            return solver.OptimizeResult(x=bad, jac=np.zeros_like(bad),
                success=False, status=8, nfev=1, nit=1, message="invalid trial")
        return original(fun, x0, **kwargs)

    monkeypatch.setattr(solver, "minimize", invalid_second_trial)
    result = solver.solve_parametric(graph, [width_constraint(width)], source_observation=raw)
    search = result["diagnostics"]["source_budget_search"]
    assert len(calls) == 2
    assert not result["accepted"] and result["entities"] == before["entities"]
    assert result["status"] == "source_budget_search_failed"
    assert result["validation"]["geometry_valid"]
    assert result["validation"]["constraint_subset_satisfied"]
    assert result["candidate_strict_measurement_contract"]["satisfied"]
    assert not result["validation"]["source_boundary_budget_passed"]
    assert search["invalid_trial_replaced_by_valid_checkpoint"]
    assert not search["retained_checkpoint_is_certified_solution"]
    assert not search["feasible_checkpoint_found"]
    assert not search["infeasibility_proven"]
    assert graph == before


def test_incompatible_exact_dimension_cannot_pass_new_domain_bounds():
    graph, constraints, raw = inputs(source_width=10., tolerance=.03)
    constraints[0].update(enforcement="exact", nominal_source="source_ocr")
    before = deepcopy(graph)
    result = solver.solve_parametric(graph, constraints, source_observation=raw)
    assert not result["accepted"]
    assert result["entities"] == before["entities"]
    assert result["diagnostics"]["source_budget_search"]["maximum_deviation"] == .03
    assert not result["diagnostics"]["source_budget_search"]["infeasibility_proven"]

def test_already_exact_source_graph_survives_failed_optional_warmup(monkeypatch):
    graph, constraints, raw = inputs(source_width=10., tolerance=.15)
    constraints[0]['value'] = 10.
    before = deepcopy(graph)
    def exhaust_optional(fun, *args, **kwargs):
        assert fun.__name__ == 'scalar_objective'
        raise solver._SourceBudgetExhausted('optional warmup exhausted')
    monkeypatch.setattr(solver, 'minimize', exhaust_optional)
    result = solver.solve_parametric(graph, constraints, source_observation=raw)
    assert result['accepted'] and all(row['passed'] for row in result['constraints'])
    assert result['diagnostics']['source_budget_search']['initial_graph_feasible_checkpoint_revalidated']
    assert result['validation']['source_boundary_budget_passed']
    assert graph == before
