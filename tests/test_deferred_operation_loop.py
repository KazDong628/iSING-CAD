"""Real f76d loop/stage/executor scheduling; synthetic evidence, zero network."""
from copy import deepcopy
import hashlib

import cv2
import numpy as np
import pytest

from contour_agent.parametric_pipeline import _topology_edit_loop
from contour_agent.reconstruction_feedback import geometry_fingerprint


def _setup(tmp_path, monkeypatch, operations, *, invalid_first_receipt=False):
    source = tmp_path / 'source.png'
    assert cv2.imwrite(str(source), np.full((64, 64, 3), 255, dtype=np.uint8))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    vertices = [[10., 10.], [30., 10.], [50., 20.], [50., 40.], [30., 50.], [10., 40.]]
    graph = {'candidate_id': 'base', 'units': 'pixel', 'source_sha256': source_sha,
             'nodes': [{'id': f'v{i:03d}', 'x': p[0], 'y': p[1], 'source_px': p}
                       for i, p in enumerate(vertices)],
             'entities': [{'id': f'g{i:03d}', 'type': 'LINE', 'start': p,
                           'end': vertices[(i + 1) % 6], 'start_node': f'v{i:03d}',
                           'end_node': f'v{(i + 1) % 6:03d}'} for i, p in enumerate(vertices)]}
    raw = geometry_fingerprint(graph)
    selected = {'id': 'base', 'graph': graph, 'overlay_path': str(source.resolve())}
    bundle = {'source_sha256': source_sha, 'base_graph_sha256': raw,
              'annotation_inventory': [], 'candidates': [selected]}
    feedback = {'geometry_sha256': raw, 'solver_accepted': True,
                'solver_status': 'accepted', 'constraint_count': 0,
                'radius_binding_coverage': {'unresolved': []}}

    class Editor:
        calls = 0

        def propose(self, image, overlay, parent, inventory, **kwargs):
            self.calls += 1
            invalid = invalid_first_receipt and self.calls == 1
            return {'status': 'failed' if invalid else 'completed',
                    'schema_success': not invalid, 'network_requests': 0,
                    'operations': deepcopy(operations)}

    editor = Editor()
    monkeypatch.setattr('contour_agent.parametric_pipeline._topology_source_validation',
                        lambda *args: {'passed': True})
    monkeypatch.setattr('contour_agent.parametric_pipeline._preflight_input_identity',
                        lambda *args: {'schema_version': 'synthetic', 'reference_geometry_used': False})
    monkeypatch.setattr('contour_agent.constraint_binding.analyze_constraint_bindings',
                        lambda *args, **kwargs: {'constraints': []})
    monkeypatch.setattr('contour_agent.parametric_pipeline._solve_with_source_observation',
                        lambda *args, **kwargs: {'accepted': True, 'entities': []})
    monkeypatch.setattr('contour_agent.parametric_pipeline.reconstruction_feedback',
                        lambda *args, **kwargs: deepcopy(feedback))
    monkeypatch.setattr('contour_agent.topology_editing.propose_annotation_arc_edits',
                        lambda *args, **kwargs: [])
    monkeypatch.setattr('contour_agent.planning_provider.evaluate_candidates',
                        lambda pool, **kwargs: {'admissible_candidate_ids': ['base'], 'evaluated': []})
    monkeypatch.setattr('contour_agent.topology_editing._affines',
                        lambda *args: (lambda p: np.asarray(p), lambda p: np.asarray(p), 1.))
    monkeypatch.setattr('contour_agent.topology_editing._annotation_inventory',
                        lambda *args: ([], {}))
    executed = []

    def reject_action(graph, baseline, operation, inventory, **kwargs):
        # Rejection is intentional: it keeps the parent identical so dedup must
        # distinguish a truly attempted failure from a merely deferred proposal.
        executed.append(deepcopy(operation))
        raise ValueError('synthetic_source_interval_not_supported')

    monkeypatch.setattr('contour_agent.topology_editing._apply_one', reject_action)
    baseline = {'extraction': {'raw_polyline_px': [*vertices, vertices[0]]}}
    final, report = _topology_edit_loop(source, {}, baseline, selected, bundle, tmp_path / 'run',
        editor_provider=editor, use_api=True, max_rounds=2)
    return editor, executed, final, report


def _operations(count):
    return [{'action': 'refit_entity_as_line', 'entity_ids': [f'g{i:03d}'], 'record_id': None}
            for i in range(count)]


def _assert_bounds(editor, final, report):
    assert editor.calls == 2
    assert len(report['rounds']) == 2
    assert final['id'] == 'base' and report['accepted_round_count'] == 0
    assert report['max_rounds'] == 2
    assert report['budget']['provider_calls_reserved'] == 2
    assert report['budget']['max_provider_calls'] == 12
    assert report['budget']['max_preflights'] == 18
    assert report['budget']['max_seconds'] == 600.
    assert all(round_['executed_operation_count'] <= 5 for round_ in report['rounds'])


def test_cap_deferred_sixth_proposal_reaches_real_executor_in_round_two(tmp_path, monkeypatch):
    operations = _operations(6)
    editor, executed, final, report = _setup(tmp_path, monkeypatch, operations)
    _assert_bounds(editor, final, report)
    assert executed == operations
    first, second = report['rounds']
    assert first['executed_operation_count'] == 5
    assert all(row['status'] == 'rejected' for row in first['execution']['operations'])
    assert second['executed_operation_count'] == 1
    attempted = [row for row in second['execution']['operations'] if row['status'] != 'skipped']
    assert [row['operation'] for row in attempted] == [operations[5]]
    assert second['visited_operations_skipped'] == 5
    assert len(report['visited_operations']) == 6
    assert all(row['status'] != 'pending' for row in report['visited_operations'])


def test_schema_rejected_proposal_not_entering_filter_can_execute_next_round(tmp_path, monkeypatch):
    operations = _operations(1)
    editor, executed, final, report = _setup(tmp_path, monkeypatch, operations,
                                            invalid_first_receipt=True)
    _assert_bounds(editor, final, report)
    assert report['rounds'][0]['executed_operation_count'] == 0
    assert report['rounds'][0]['execution']['operations'] == []
    assert executed == operations
    assert report['rounds'][1]['executed_operation_count'] == 1
    assert len(report['visited_operations']) == 1
    assert report['visited_operations'][0]['status'] == 'rejected'


@pytest.mark.parametrize('count', [1, 5])
def test_actual_failed_executions_stay_deduplicated(tmp_path, monkeypatch, count):
    operations = _operations(count)
    editor, executed, final, report = _setup(tmp_path, monkeypatch, operations)
    _assert_bounds(editor, final, report)
    assert executed == operations
    assert report['rounds'][1]['executed_operation_count'] == 0
    assert report['rounds'][1]['visited_operations_skipped'] == count
    assert len(report['visited_operations']) == count
