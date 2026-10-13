"""Cooperative limits leave default work and certified artifacts unchanged."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json

import pytest

from contour_agent.execution_control import (
    ExecutionInterrupted, execution_checkpoint, execution_scope)
from contour_agent.source_constraint_completion import complete_source_constraints
from test_source_constraint_completion import setup_completion


class Clock:
    def __init__(self, now=0.):
        self.now = now

    def __call__(self):
        return self.now


def test_default_and_empty_scope_do_not_call_a_clock():
    def unavailable():
        raise AssertionError('an absent deadline must not read a clock')
    execution_checkpoint('default')
    with execution_scope(clock=unavailable):
        execution_checkpoint('unlimited')
    execution_checkpoint('restored')


def test_mock_clock_deadline_is_absolute_and_inclusive():
    clock = Clock(7.)
    with execution_scope(deadline=8., clock=clock):
        execution_checkpoint()
        clock.now = 8.
        with pytest.raises(ExecutionInterrupted) as caught:
            execution_checkpoint('caller-secret-must-not-appear')
        assert caught.value.reason == 'deadline'
        assert isinstance(caught.value, InterruptedError)
        assert 'caller-secret' not in str(caught.value)
        assert 'caller-secret' not in repr(caught.value)
    execution_checkpoint()


@pytest.mark.parametrize('inner', [6., 12.])
def test_nested_scopes_take_earliest_same_clock_deadline(inner):
    clock = Clock(5.)
    with execution_scope(deadline=10., clock=clock):
        with execution_scope(deadline=inner, clock=clock):
            clock.now = min(10., inner)
            with pytest.raises(ExecutionInterrupted, match='deadline'):
                execution_checkpoint()
        clock.now = 9.
        execution_checkpoint()
        clock.now = 10.
        with pytest.raises(ExecutionInterrupted, match='deadline'):
            execution_checkpoint()


def test_distinct_clock_domains_are_not_compared_as_numeric_timestamps():
    parent, child = Clock(100.), Clock(1.)
    with execution_scope(deadline=110., clock=parent):
        with execution_scope(deadline=5., clock=child):
            execution_checkpoint()
            parent.now = 110.
            with pytest.raises(ExecutionInterrupted, match='deadline'):
                execution_checkpoint()


def test_nested_scopes_inherit_cancellation_and_restore_after_error():
    flags = {'outer': False, 'inner': False}
    with execution_scope(cancelled=lambda: flags['outer']):
        with pytest.raises(ExecutionInterrupted, match='cancelled'):
            with execution_scope(cancelled=lambda: flags['inner']):
                flags['outer'] = True
                execution_checkpoint()
        flags['outer'] = False
        execution_checkpoint()
    execution_checkpoint()


def test_inner_cancellation_does_not_leak_and_cancellation_precedes_deadline():
    clock = Clock(5.)
    with execution_scope(deadline=10., clock=clock):
        with execution_scope(cancelled=lambda: True, deadline=1., clock=clock):
            with pytest.raises(ExecutionInterrupted) as caught:
                execution_checkpoint()
            assert caught.value.reason == 'cancelled'
        execution_checkpoint()


def test_execution_scope_is_thread_isolated():
    def unlimited_worker():
        execution_checkpoint()
        return 'unlimited'

    def cancelled_worker():
        with execution_scope(cancelled=lambda: True):
            with pytest.raises(ExecutionInterrupted, match='cancelled'):
                execution_checkpoint()
        execution_checkpoint()
        return 'restored'

    with ThreadPoolExecutor(max_workers=2) as pool:
        with execution_scope(cancelled=lambda: True):
            assert pool.submit(unlimited_worker).result(timeout=5) == 'unlimited'
            assert pool.submit(cancelled_worker).result(timeout=5) == 'restored'
            with pytest.raises(ExecutionInterrupted, match='cancelled'):
                execution_checkpoint()


@pytest.mark.parametrize('deadline', [True, float('nan'), float('inf'), '5'])
def test_invalid_deadline_is_rejected(deadline):
    with pytest.raises(ValueError):
        with execution_scope(deadline=deadline):
            pass


def test_invalid_callback_and_clock_are_rejected():
    with pytest.raises(TypeError):
        with execution_scope(cancelled=True):
            pass
    with pytest.raises(TypeError):
        with execution_scope(clock=None):
            pass
    with execution_scope(deadline=5., clock=lambda: float('nan')):
        with pytest.raises(ValueError, match='finite time'):
            execution_checkpoint()


def test_pre_cancelled_completion_never_calls_provider_or_solver(tmp_path, monkeypatch):
    graph, bindings, solution, inventory, calls, options = setup_completion(tmp_path, monkeypatch)
    before = deepcopy((graph, bindings, solution, inventory))
    output = tmp_path/'cancelled'
    with execution_scope(cancelled=lambda: True):
        with pytest.raises(ExecutionInterrupted, match='cancelled'):
            complete_source_constraints('image', {}, {}, graph, bindings, solution,
                inventory, output, max_rounds=1, use_api=True, **options)
    assert calls == []
    assert not output.exists()
    assert (graph, bindings, solution, inventory) == before


@pytest.mark.parametrize('reason', ['cancelled', 'deadline'])
def test_interruption_after_provider_keeps_incumbent_and_prevents_solve(
        tmp_path, monkeypatch, reason):
    graph, bindings, solution, inventory, calls, options = setup_completion(tmp_path, monkeypatch)
    before = deepcopy((graph, bindings, solution, inventory))
    output = tmp_path/'completion'
    output.mkdir()
    drawing = output/'drawing.dxf'
    drawing.write_bytes(b'previous-certified-dxf')
    flags, clock = {'cancelled': False}, Clock()
    original = options['binding_analyzer']

    def analyzer(*args, **kwargs):
        fresh = original(*args, **kwargs)
        flags['cancelled'] = reason == 'cancelled'
        clock.now = 10.
        return fresh

    options['binding_analyzer'] = analyzer
    with execution_scope(cancelled=lambda: flags['cancelled'], deadline=10., clock=clock):
        with pytest.raises(ExecutionInterrupted) as caught:
            complete_source_constraints('image', {}, {}, graph, bindings, solution,
                inventory, output, max_rounds=1, use_api=True, **options)
    assert caught.value.reason == reason
    assert calls == ['source', 'binding']
    assert (graph, bindings, solution, inventory) == before
    assert drawing.read_bytes() == b'previous-certified-dxf'
    receipt = json.loads((output/'completion.json').read_text('utf8'))
    assert receipt['rounds'][0]['status'] == 'interrupted'
    assert receipt['rounds'][0]['execution_stop_reason'] == reason


def test_interruption_after_solver_does_not_publish_its_candidate(tmp_path, monkeypatch):
    from contour_agent import parametric_pipeline
    graph, bindings, solution, inventory, calls, options = setup_completion(tmp_path, monkeypatch)
    before = deepcopy((graph, bindings, solution, inventory))
    original = parametric_pipeline._solve_with_source_observation
    flags = {'cancelled': False}

    def solve(*args, **kwargs):
        candidate = original(*args, **kwargs)
        flags['cancelled'] = True
        return candidate

    monkeypatch.setattr(parametric_pipeline, '_solve_with_source_observation', solve)
    output = tmp_path/'completion'
    with execution_scope(cancelled=lambda: flags['cancelled']):
        with pytest.raises(ExecutionInterrupted, match='cancelled'):
            complete_source_constraints('image', {}, {}, graph, bindings, solution,
                inventory, output, max_rounds=1, **options)
    assert calls == ['source', 'binding', 'certificate', 'solve']
    assert (graph, bindings, solution, inventory) == before
    assert not (output/'round-01/parametric-solution.json').exists()


def test_interruption_after_refinement_preserves_previous_subset(tmp_path, monkeypatch):
    graph, bindings, solution, inventory, calls, options = setup_completion(
        tmp_path, monkeypatch, growth=False)
    solution['diagnostics']['optimizer_converged'] = False
    solution['underconstrained'] = True
    before = deepcopy((graph, bindings, solution, inventory))
    flags = {'cancelled': False}

    def refine(*args, **kwargs):
        calls.append('refine')
        flags['cancelled'] = True
        return deepcopy(solution), {'accepted': True}

    monkeypatch.setattr('contour_agent.source_subset_refinement.refine_source_subset', refine)
    with execution_scope(cancelled=lambda: flags['cancelled']):
        with pytest.raises(ExecutionInterrupted, match='cancelled'):
            complete_source_constraints('image', {}, {}, graph, bindings, solution,
                inventory, tmp_path/'completion', max_rounds=1, **options)
    assert calls == ['source', 'binding', 'certificate', 'refine']
    assert (graph, bindings, solution, inventory) == before
