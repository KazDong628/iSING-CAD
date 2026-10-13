"""A round's source proposal may recover ownership, never old coordinates."""
from copy import deepcopy
import hashlib
import math

import cv2
import numpy as np
import pytest

from contour_agent import constraint_binding as binding
from contour_agent.measurement_anchors import support_intersection, virtual_corner_locations


def fixture(tmp_path):
    center, radius = np.array([200., 280.]), 20.
    def contact(degrees):
        theta = math.radians(degrees)
        return (center+radius*np.array([math.cos(theta), math.sin(theta)])).tolist()
    def tangent(degrees):
        theta = math.radians(degrees)
        return np.array([math.sin(theta), -math.cos(theta)])
    first = {'id': 'horizontal', 'type': 'LINE', 'start_node': 'v0', 'end_node': 'v1',
             'start': [50., 300.], 'end': [200., 300.]}
    arc = {'id': 'fillet', 'type': 'ARC', 'start_node': 'v1', 'end_node': 'v2',
           'start': [200., 300.], 'end': contact(15.), 'center': center.tolist(),
           'radius': radius, 'clockwise': True}
    sloped = {'id': 'sloped', 'type': 'LINE', 'start_node': 'v2', 'end_node': 'v3',
              'start': contact(15.), 'end': (np.asarray(contact(15.))+80*tangent(15.)).tolist()}
    source = {'units': 'mm', 'ground_truth_used': False,
              'entities': [first, arc, sloped]}
    node_points = {'v0': first['start'], 'v1': first['end'],
                   'v2': sloped['start'], 'v3': sloped['end']}
    source['nodes'] = [{'id': identifier, 'x': point[0], 'y': point[1],
                        'source_px': list(point)} for identifier, point in node_points.items()]
    gray = np.full((400, 400), 255, np.uint8)
    for entity in source['entities']:
        if entity['type'] == 'LINE':
            points = np.asarray([entity['start'], entity['end']])
        else:
            angles = np.linspace(math.pi/2, math.radians(15.), 100)
            points = center+radius*np.c_[np.cos(angles), np.sin(angles)]
        cv2.polylines(gray, [np.rint(points).astype(np.int32)], False, 0, 2)
    path = tmp_path/'round-source.png'
    assert cv2.imwrite(str(path), gray)
    source['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    current = deepcopy(source)
    current['entities'][1]['end'] = contact(17.)
    current['entities'][2].update(start=contact(17.),
        end=(np.asarray(contact(17.))+19.6*tangent(17.)).tolist())
    for node in current['nodes']:
        if node['id'] in {'v2', 'v3'}:
            point = current['entities'][2]['start' if node['id'] == 'v2' else 'end']
            node.update(x=point[0], y=point[1], source_px=list(point))
    transform = lambda values: np.asarray(values, float)
    return gray, source, current, transform


def observe(gray, source, current, transform, *, records=(), angle_observations=()):
    return binding._source_virtual_corners(gray, records, current, transform, 4.,
        angle_observations=angle_observations, source_support_graph=source)


def test_round_source_proposal_reobserves_long_ink_support_when_current_line_is_short(tmp_path):
    gray, source, current, transform = fixture(tmp_path)
    assert len(source['entities'][2]['end']) == 2
    assert np.linalg.norm(np.subtract(current['entities'][2]['end'], current['entities'][2]['start'])) < 24.
    # This is a real source observation gap, rather than a mocked detector.
    assert virtual_corner_locations(gray, [], current, transform, 4.) == []
    old_observation = virtual_corner_locations(gray, [], source, transform, 4.)
    assert len(old_observation) == 1
    before = deepcopy((source, current))
    result = observe(gray, source, current, transform)
    assert len(result) == 1
    row = result[0]
    expected = support_intersection(current['entities'][0], current['entities'][2])
    assert row['anchor'] == {'type': 'line_intersection', 'entities': ['horizontal', 'sloped']}
    assert [row['x'], row['y']] == pytest.approx(expected)
    assert row['source_px'] == pytest.approx(old_observation[0]['source_px'], abs=1e-9)
    assert np.linalg.norm(expected-np.asarray(row['source_px'])) > .1
    assert all(item['verified'] for item in row['evidence']['source_line_support'])
    assert row['evidence']['coordinates_locked'] is False
    assert row['evidence']['nominal_used'] is False
    assert (source, current) == before


def test_source_seed_is_not_needed_when_current_source_proof_exists_and_is_deduplicated(tmp_path):
    gray, source, _, transform = fixture(tmp_path)
    current = deepcopy(source)
    fresh = binding._source_virtual_corners(gray, [], current, transform, 4.)
    merged = observe(gray, source, current, transform)
    assert len(fresh) == len(merged) == 1
    assert merged[0]['anchor'] == fresh[0]['anchor']
    assert [merged[0]['x'], merged[0]['y']] == [fresh[0]['x'], fresh[0]['y']]


def test_missing_original_ink_cannot_be_revived_from_previous_proposal(tmp_path):
    gray, source, current, transform = fixture(tmp_path)
    assert observe(np.full_like(gray, 255), source, current, transform) == []


def test_occluded_original_line_cannot_be_revived_from_old_geometry_or_old_certificate(tmp_path):
    gray, source, current, transform = fixture(tmp_path)
    # Old geometry or a cached Boolean is not permission to ignore current ink.
    source['entities'][2]['angle_support_evidence'] = {'verified': True, 'nominal': 24.,
        'source_interval_endpoints_px': [source['entities'][2]['start'], source['entities'][2]['end']]}
    records = [{'id': 'occlusion', 'box': [[0., 0.], [399., 0.], [399., 399.], [0., 399.]]}]
    assert observe(gray, source, current, transform, records=records) == []


def test_more_distant_old_domain_without_ink_ownership_is_not_a_fallback(tmp_path):
    gray, source, current, transform = fixture(tmp_path)
    old_domain = deepcopy(source)
    for entity in old_domain['entities']:
        for key in ('start', 'end', 'center'):
            if key in entity:
                entity[key][0] += 30.
    for node in old_domain['nodes']:
        node['x'] += 30.
        node['source_px'][0] += 30.
    assert observe(gray, old_domain, current, transform) == []


def test_fresh_angle_proposal_for_neighbor_arc_does_not_replace_line_ownership(tmp_path):
    gray, source, current, transform = fixture(tmp_path)
    observation = {'record_id': 'angle', 'nominal': 24., 'verified': True,
        'evidence': {'verified': True, 'nominal_used_to_rank': False},
        'source_line': {'start_px': [220., 285.], 'end_px': [225., 266.]},
        'target_candidates': [{'entity_id': 'fillet', 'entity_type': 'ARC',
                               'whole_line_supported': False}]}
    result = observe(gray, source, current, transform, angle_observations=[observation])
    observation['nominal'] = 999.
    changed = observe(gray, source, current, transform, angle_observations=[observation])
    assert len(result) == 1
    assert changed == result


def test_seed_affine_cannot_reinterpret_source_pixels_in_a_different_frame(tmp_path):
    gray, source, current, transform = fixture(tmp_path)
    for node in source['nodes']:
        node['source_px'][0] += 5.
    with pytest.raises(ValueError):
        observe(gray, source, current, transform)


@pytest.mark.parametrize('defect', ['hash', 'missing_hash', 'units', 'gt', 'missing_gt',
    'missing_entity', 'extra_entity', 'duplicate_id', 'reordered', 'type', 'start_node',
    'end_node', 'arc_traversal'])
def test_incompatible_seed_provenance_or_typed_traversal_is_rejected(tmp_path, defect):
    gray, source, current, transform = fixture(tmp_path)
    if defect == 'hash':
        source['source_sha256'] = '0'*64
    elif defect == 'missing_hash':
        del source['source_sha256']
    elif defect == 'units':
        source['units'] = 'px'
    elif defect == 'gt':
        source['ground_truth_used'] = True
    elif defect == 'missing_gt':
        del source['ground_truth_used']
    elif defect == 'missing_entity':
        source['entities'].pop()
    elif defect == 'extra_entity':
        source['entities'].append({'id': 'extra', 'type': 'LINE', 'start_node': 'v3',
            'end_node': 'v4', 'start': [10., 10.], 'end': [20., 10.]})
    elif defect == 'duplicate_id':
        source['entities'][2]['id'] = 'horizontal'
    elif defect == 'reordered':
        source['entities'].reverse()
    elif defect == 'type':
        source['entities'][2]['type'] = 'ARC'
    elif defect == 'start_node':
        source['entities'][2]['start_node'] = 'different'
    elif defect == 'end_node':
        source['entities'][2]['end_node'] = 'different'
    else:
        source['entities'][1]['clockwise'] = False
    with pytest.raises(ValueError):
        observe(gray, source, current, transform)
