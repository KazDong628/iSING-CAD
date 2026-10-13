from copy import deepcopy

import numpy as np
import pytest

from contour_agent import parametric_pipeline as p, source_stroke_prior as ink
from contour_agent.parametric_solver import solve_parametric
from contour_agent.source_subset_refinement import refine_source_subset
from test_parametric_solver import rectangle


def setup_refinement(monkeypatch, *, defect=None):
    graph = rectangle(width=60., height=40.)
    constraint = dict(id='known', kind='horizontal', record_id=None, entities=['g0'], nodes=[], value=0., source='source_geometry')
    bindings = {'constraints': [constraint]}
    solution = solve_parametric(graph, bindings['constraints'])
    candidate = deepcopy(solution)
    for entity in candidate['entities']:
        for end in ('start', 'end'):
            entity[end][0] += 1. if defect != 'worse' else -1.
    if defect == 'equation':
        candidate['constraints'][0]['passed'] = False
    if defect == 'dof':
        candidate['diagnostics']['remaining_shape_dof'] += 1
    frozen = {'source': 'frozen_original_pixels'}
    calls = []
    def observation(*args, **kwargs):
        calls.append('observe')
        return frozen
    def prior(actual, original, entities, scale):
        assert actual is frozen and original is graph and entities is graph['entities']
        return lambda shape: np.array([shape[0]['start'][0]-10.]), {'enabled': True}
    def solve(solver, original, constraints, baseline, output, **kwargs):
        calls.append('solve')
        assert original is graph and constraints is bindings['constraints']
        assert kwargs['seed_entities'] is solution['entities']
        assert kwargs['image_stroke_prior'] is frozen
        assert kwargs['budget_seconds'] <= 90.
        if defect == 'exception':
            raise RuntimeError('must-not-copy-sensitive-error-message')
        return candidate
    def validate(*args):
        calls.append('validate')
        return {'passed': not (defect == 'source' and calls.count('validate') == 2)}
    def export(image, baseline, actual, output):
        calls.append('export')
        assert actual['constraints'] == bindings['constraints']
        return {'validation': {key: {'passed': defect != 'native', 'dxf_readback_performed': defect != 'readback'}
            for key in ('exact_radius_validation', 'strict_relation_validation', 'native_measurement_validation',
                        'native_source_angle_validation', 'native_axis_relation_validation')}}
    monkeypatch.setattr(ink, 'source_stroke_observation', observation)
    monkeypatch.setattr(ink, 'stroke_prior', prior)
    monkeypatch.setattr(p, '_solve_with_source_observation', solve)
    return graph, bindings, solution, candidate, calls, dict(source_validator=validate, exporter=export)


def test_fixed_source_objective_improvement_preserves_all_equations_and_domain(tmp_path, monkeypatch):
    graph, bindings, solution, candidate, calls, options = setup_refinement(monkeypatch)
    before = deepcopy((graph, bindings, solution))
    result, receipt = refine_source_subset('image', {}, {}, graph, bindings, solution, tmp_path, **options)
    assert result is candidate and receipt['accepted']
    assert receipt['source_cost_after'] < receipt['source_cost_before']
    assert calls == ['validate', 'observe', 'solve', 'validate', 'export']
    assert (graph, bindings, solution) == before


@pytest.mark.parametrize('defect', ['worse', 'equation', 'source', 'dof', 'native', 'readback', 'exception'])
def test_failed_optional_refinement_preserves_certified_artifact(tmp_path, monkeypatch, defect):
    graph, bindings, solution, _, _, options = setup_refinement(monkeypatch, defect=defect)
    result, receipt = refine_source_subset('image', {}, {}, graph, bindings, solution, tmp_path, **options)
    assert result is solution and not receipt['accepted']
    assert 'must-not-copy-sensitive' not in (tmp_path/'refinement.json').read_text('utf8')


@pytest.mark.parametrize('budget', [float('nan'), float('inf'), 0., -1.])
def test_invalid_budget_is_rejected_before_work(tmp_path, budget):
    with pytest.raises(ValueError):
        refine_source_subset('image', {}, {}, {}, {}, {}, tmp_path, budget_seconds=budget)
