"""One finite radius tail and a declared dimension can own a circle center."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from contour_agent.measurement_anchors import source_curve_center_locations


def fixture():
    gray = np.full((600, 650), 255, np.uint8)
    tip, center = np.array([100., 100.]), np.array([330., 420.])
    cv2.line(gray, tuple(tip.astype(int)), tuple(center.astype(int)), 0, 2)
    cv2.line(gray, (310, 420), (570, 420), 0, 1)
    direction = (tip-center)/np.linalg.norm(tip-center)
    row = {'record_id': 'radius', 'nominal': 50., 'arrowhead_verified': True,
        'arrowhead': {'tip_px': tip.tolist(), 'direction_px': direction.tolist(), 'verified': True},
        'shaft_evidence': {'verified': True}, 'segment_px': [tip.tolist(), (tip-60*direction).tolist()],
        'source_arrow_ownership': {'physical_arrow_id': 'a0', 'status': 'globally_unique_source_claim'},
        'target_candidates': [{'entity_id': 'g0', 'entity_type': 'ARC'}]}
    graph = {'entities': [{'id': 'g0', 'type': 'ARC', 'center': [5., 0.],
        'start': [15., 0.], 'end': [5., 10.], 'radius': 10., 'clockwise': False}]}
    witness = {'record_id': 'height', 'nominal': 60., 'axis': 'y',
        'binding_evidence': {'verified': True}, 'axis_association': {'ambiguous': False},
        'line': {'lo': 150., 'hi': 420., 'cross': 550., 'extension_stations': {
            'hi': {'lo': 310., 'hi': 570., 'cross': 420., 'thickness': 1.}}}}
    return gray, graph, [row], [witness], center


def test_single_radius_tail_yields_symbolic_unknown_center_not_pixel_lock():
    gray, graph, rows, witnesses, center = fixture()
    found = source_curve_center_locations(gray, graph, rows, 'y', dimension_witnesses=witnesses)
    assert len(found) == 1
    item = found[0]
    # Transverse ray position is an approximate ownership observation. Only
    # the dimension's observed station ordinate identifies this measurement.
    assert item['source_px'][1] == center[1]
    assert abs(item['source_px'][0]-center[0]) < 3.
    assert item['anchor'] == {'type': 'curve_center', 'entity_id': 'g0', 'axis': 'y'}
    assert item['y'] == 0.  # The unknown stays a symbolic CAD center, not y=420.
    assert item['evidence']['center_semantics_verified']
    assert item['evidence']['nominal_used'] is False
    assert item['evidence']['coordinates_locked'] is False
    rows[0]['nominal'] = 999.
    witnesses[0]['nominal'] = .1
    changed = deepcopy(graph)
    changed['entities'][0]['center'] = [900., 200.]
    more = source_curve_center_locations(gray, changed, rows, 'y', dimension_witnesses=witnesses)
    assert more[0]['source_px'] == item['source_px']
    assert more[0]['anchor'] == item['anchor']
    assert more[0]['y'] == 200.


@pytest.mark.parametrize('defect', ['continues', 'gap', 'ambiguous_owner', 'no_arrows',
    'ambiguous_station', 'extension_too_short', 'wrong_extension'])
def test_infinite_ray_or_unproved_dimension_does_not_create_center(defect):
    gray, graph, rows, witnesses, center = fixture()
    if defect == 'continues':
        backward = -np.asarray(rows[0]['arrowhead']['direction_px'])
        cv2.line(gray, tuple(center.astype(int)), tuple((center+65*backward).astype(int)), 0, 2)
    elif defect == 'gap':
        cv2.circle(gray, (260, 322), 18, 255, -1)
    elif defect == 'ambiguous_owner':
        rows[0]['target_candidates'].append({'entity_id': 'other', 'entity_type': 'ARC'})
    elif defect == 'no_arrows':
        witnesses[0]['binding_evidence']['verified'] = False
    elif defect == 'ambiguous_station':
        witnesses[0]['axis_association']['ambiguous'] = True
    elif defect == 'extension_too_short':
        witnesses[0]['line']['extension_stations']['hi']['lo'] = 400.
    else:
        witnesses[0]['line']['extension_stations']['hi']['cross'] = 480.
    assert not source_curve_center_locations(gray, graph, rows, 'y', dimension_witnesses=witnesses)
