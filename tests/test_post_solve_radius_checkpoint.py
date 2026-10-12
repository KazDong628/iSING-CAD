"""A feasible checkpoint supplies initial values, never a new optimization domain."""
from copy import deepcopy

import pytest

from contour_agent import parametric_pipeline as p
from contour_agent import source_radius_hypotheses as h
from contour_agent.parametric_solver import solve_parametric
from test_source_radius_hypotheses import fixture


def stages(tmp_path, monkeypatch, *, source_pass=True, certificate_pass=True):
    image, document, graph, known, inventory = fixture(tmp_path)
    bindings = {'constraints': [known]}
    solution = solve_parametric(graph, [known])
    radius = {'id': 'new-R', 'kind': 'radius', 'record_id': 'r000', 'entities': ['g001'],
        'nodes': [], 'value': 3., 'source': 'ocr_local_binding', 'required': True,
        'enforcement': 'exact', 'nominal_source': 'source_ocr'}
    new_bindings = {'constraints': [known, radius]}
    new_solution = solve_parametric(graph, new_bindings['constraints'])
    candidate = h.graph_from_solution(graph, {}, new_solution)
    calls = []
    def refine(*args, **kwargs):
        calls.append('hypothesis')
        assert args[3] is graph
        assert kwargs['seed_entities'] == solution['entities']
        assert kwargs['max_attempts'] == 1
        return candidate, {'status': 'source_verified_seed'}
    monkeypatch.setattr(h, 'refine_source_owned_radius_hypotheses', refine)
    def certify(*args, **kwargs):
        calls.append('online_binding')
        assert args[3] is graph and args[4] is bindings
        return candidate, new_bindings, {'status': 'accepted_seed' if certificate_pass else 'rejected'}
    monkeypatch.setattr(h, 'certify_refined_radius_seed', certify)
    def solve(*args, **kwargs):
        calls.append('final_solve')
        assert args[1] is graph
        assert kwargs['seed_entities'] == candidate['entities']
        assert kwargs['budget_seconds'] == 120.
        return new_solution
    monkeypatch.setattr(p, '_solve_with_source_observation', solve)
    def source(*args):
        calls.append('source_check')
        return {'passed': True if calls.count('source_check') == 1 else source_pass}
    monkeypatch.setattr(p, '_solved_source_validation', source)
    return image, document, graph, bindings, solution, inventory, calls


def test_retry_keeps_original_domain_and_certifies_new_radius_after_joint_solve(tmp_path, monkeypatch):
    image, document, graph, bindings, solution, inventory, calls = stages(tmp_path, monkeypatch)
    before = deepcopy((graph, bindings, solution))
    new_graph, new_bindings, new_solution, receipt = p._retry_source_radius_from_checkpoint(
        image, document, {}, graph, bindings, solution, inventory, tmp_path/'retry', use_api=True)
    assert receipt['accepted'] and not receipt['publication_approved']
    assert not receipt['optimization_domain_recentered']
    assert len(new_bindings['constraints']) == 2 and new_solution['accepted']
    assert (graph, bindings, solution) == before
    assert calls == ['source_check', 'hypothesis', 'online_binding', 'final_solve', 'source_check']


@pytest.mark.parametrize('defect', ['source', 'certificate', 'exception'])
def test_optional_retry_failure_preserves_all_prior_artifacts_and_constraints(tmp_path, monkeypatch, defect):
    image, document, graph, bindings, solution, inventory, calls = stages(tmp_path, monkeypatch,
        source_pass=defect!='source', certificate_pass=defect!='certificate')
    if defect == 'exception':
        def failed(*args, **kwargs): raise RuntimeError('synthetic retry failure')
        monkeypatch.setattr(h, 'refine_source_owned_radius_hypotheses', failed)
    result = p._retry_source_radius_from_checkpoint(image,document,{},graph,bindings,solution,
        inventory,tmp_path/'retry',use_api=True)
    assert result[:3] == (graph, bindings, solution)
    assert result[3]['accepted'] is False
    assert (tmp_path/'retry/radius-checkpoint-refinement.json').exists()


def test_unaccepted_subset_never_triggers_an_optional_provider_request(tmp_path, monkeypatch):
    image, document, graph, bindings, solution, inventory, calls = stages(tmp_path, monkeypatch)
    solution['accepted'] = False
    result = p._retry_source_radius_from_checkpoint(image,document,{},graph,bindings,solution,
        inventory,tmp_path/'retry',use_api=True)
    assert not result[3]['accepted'] and calls == []
