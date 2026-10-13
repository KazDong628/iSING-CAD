"""Finite source strokes, rather than nominal proximity, own section stations."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from contour_agent.section_binding import source_section_locations


def fixture(tmp_path, *, clipped_section=False):
    gray = np.full((500, 500), 255, np.uint8)
    entities = [
        {'id': 'datum_h', 'type': 'LINE', 'start': [70., 300.], 'end': [130., 300.]},
        {'id': 'datum_v', 'type': 'LINE', 'start': [100., 280.], 'end': [100., 320.]},
        {'id': 'upper', 'type': 'LINE', 'start': [150., 185.], 'end': [210., 215.]},
        {'id': 'lower', 'type': 'LINE', 'start': [150., 245.], 'end': [210., 275.]},
    ]
    for entity in entities:
        cv2.line(gray, tuple(np.rint(entity['start']).astype(int)),
                 tuple(np.rint(entity['end']).astype(int)), 0, 2)
    strokes = [((100, 300), (100, 410)), ((180, 200), (180, 410)),
               ((100, 400), (180, 400)), ((180, 200), (310, 200)),
               ((180, 260), (310, 260)), ((300, 200), (300, 260))]
    for first, last in strokes:
        cv2.line(gray, first, last, 0, 1)
    for tip, triangle in [
        ((100, 400), [(100, 400), (112, 396), (112, 404)]),
        ((180, 400), [(180, 400), (168, 396), (168, 404)]),
        ((300, 200), [(300, 200), (296, 212), (304, 212)]),
        ((300, 260), [(300, 260), (296, 248), (304, 248)]),
    ]:
        cv2.fillPoly(gray, [np.asarray(triangle, np.int32)], 0)
    path = tmp_path/'section-source.png'
    assert cv2.imwrite(str(path), gray)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    reference = {'record_id': 'offset', 'axis': 'x', 'nominal': 80.,
        'binding_evidence': {'verified': True, 'ambiguous': False, 'search_complete': True,
            'endpoint_arrows': [{'verified': True, 'tip_px': [100., 400.], 'direction_px': [-1., 0.]},
                                {'verified': True, 'tip_px': [180., 400.], 'direction_px': [1., 0.]}]},
        'axis_association': {'ambiguous': False},
        'line': {'lo': 100., 'hi': 180., 'cross': 400., 'extension_stations': {
            'lo': {'lo': 300., 'hi': 410., 'cross': 100., 'thickness': 1},
            'hi': {'lo': 350. if clipped_section else 200., 'hi': 410., 'cross': 180., 'thickness': 1}}}}
    measured = {'record_id': 'height', 'axis': 'y', 'nominal': 60.,
        'binding_evidence': {'verified': True, 'ambiguous': False, 'search_complete': True,
            'endpoint_arrows': [{'verified': True, 'tip_px': [300., 200.], 'direction_px': [0., -1.]},
                                {'verified': True, 'tip_px': [300., 260.], 'direction_px': [0., 1.]}]},
        'axis_association': {'ambiguous': False},
        'line': {'lo': 200., 'hi': 260., 'cross': 300., 'extension_stations': {
            'lo': {'lo': 180., 'hi': 310., 'cross': 200., 'thickness': 1},
            'hi': {'lo': 180., 'hi': 310., 'cross': 260., 'thickness': 1}}}}
    records = [
        {'id': 'offset', 'text': '(80)', 'box': [[120., 370.], [150., 370.], [150., 390.], [120., 390.]],
         'parsed': {'kind': 'length', 'reference': True, 'nominal': 80.}},
        {'id': 'height', 'text': '60', 'box': [[315., 210.], [335., 210.], [335., 245.], [315., 245.]],
         'parsed': {'kind': 'length', 'reference': False, 'nominal': 60.}},
    ]
    virtual = [{'id': 'datum_corner', 'x': 100., 'y': 300., 'source_px': [100., 300.],
        'anchor': {'type': 'line_intersection', 'entities': ['datum_h', 'datum_v']},
        'evidence': {'method': 'source_supported_theoretical_sharp_corner_v1',
            'source_line_support': [{'verified': True}, {'verified': True}],
            'nominal_used': False, 'coordinates_locked': False}}]
    graph = {'entities': entities, 'nodes': []}
    transform = lambda values: np.asarray(values, float)
    return gray, records, graph, transform, [reference, measured], virtual


def observe(fixture):
    gray, records, graph, transform, witnesses, virtual = fixture
    return source_section_locations(gray, records, graph, transform, 4., 'y', witnesses, virtual)


def test_png_finite_reference_station_defines_line_sections_without_extra_vertices(tmp_path):
    inputs = fixture(tmp_path)
    before = deepcopy((inputs[1], inputs[2], inputs[4], inputs[5]))
    result = observe(inputs)
    assert {row['anchor']['entity_id'] for row in result} == {'upper', 'lower'}
    for row in result:
        anchor = row['anchor']
        assert anchor['type'] == 'section_curve_intersection' and anchor['branch'] == 'linear'
        assert anchor['station'] == {'type': 'source_offset_from_line_intersection',
            'entities': ['datum_h', 'datum_v'], 'axis': 'x', 'direction': 1,
            'source_record_id': 'offset', 'source_offset': 80., 'reference': True}
        assert row['source_px'] == [180., 200. if anchor['entity_id'] == 'upper' else 260.]
        assert row['x'] == 180.
        assert row['evidence']['nominal_used_for_station_ownership'] is False
        assert row['evidence']['coordinates_locked'] is False
        assert row['evidence']['reference_offset_is_station_definition_not_independent_equation']
    assert (inputs[1], inputs[2], inputs[4], inputs[5]) == before


def test_reference_nominal_changes_symbolic_station_but_not_source_owner_or_observation(tmp_path):
    inputs = fixture(tmp_path)
    original = {row['anchor']['entity_id']: row for row in observe(inputs)}
    inputs[1][0]['parsed']['nominal'] = 90.
    inputs[4][0]['nominal'] = 90.
    changed = {row['anchor']['entity_id']: row for row in observe(inputs)}
    assert set(changed) == set(original) == {'upper', 'lower'}
    for identifier, row in changed.items():
        assert row['source_px'] == original[identifier]['source_px']
        assert row['anchor']['station']['source_offset'] == 90.
        assert row['x'] == 190.
    assert changed['upper']['y'] == 205.
    assert changed['lower']['y'] == 265.


def test_measured_nominal_cannot_select_different_section_owners(tmp_path):
    inputs = fixture(tmp_path)
    original = observe(inputs)
    inputs[1][1]['parsed']['nominal'] = 999.
    inputs[4][1]['nominal'] = .01
    assert observe(inputs) == original


def test_clipped_reference_extension_is_remeasured_from_one_continuous_real_stroke(tmp_path):
    assert len(observe(fixture(tmp_path, clipped_section=True))) == 2


@pytest.mark.parametrize('clipped', [False, True])
def test_section_extension_cannot_bridge_white_gap_even_with_claimed_finite_span(tmp_path, clipped):
    inputs = fixture(tmp_path, clipped_section=clipped)
    inputs[0][305:325, 176:185] = 255
    assert observe(inputs) == []


def test_base_extension_to_line_intersection_cannot_bridge_white_gap(tmp_path):
    inputs = fixture(tmp_path)
    inputs[0][340:360, 96:105] = 255
    assert observe(inputs) == []


def test_measurement_extension_cannot_bridge_white_gap_to_source_contact(tmp_path):
    inputs = fixture(tmp_path)
    inputs[0][196:205, 230:250] = 255
    result = observe(inputs)
    assert {row['anchor']['entity_id'] for row in result} == {'lower'}


def test_extension_crossing_station_interior_is_not_section_termination(tmp_path):
    inputs = fixture(tmp_path)
    inputs[4][1]['line']['extension_stations']['lo']['lo'] = 140.
    cv2.line(inputs[0], (140, 200), (310, 200), 0, 1)
    result = observe(inputs)
    assert {row['anchor']['entity_id'] for row in result} == {'lower'}


def test_nearest_vertex_does_not_replace_required_symbolic_line_intersection(tmp_path):
    inputs = fixture(tmp_path)
    inputs[2]['nodes'] = [{'id': 'near', 'x': 100., 'y': 300., 'source_px': [100., 300.]}]
    inputs[5][0]['anchor'] = {'type': 'node', 'node_id': 'near'}
    assert observe(inputs) == []


def test_two_reference_origins_remain_ambiguous_instead_of_ranked_by_nominal(tmp_path):
    inputs = fixture(tmp_path)
    other = deepcopy(inputs[5][0])
    other['id'] = 'another_corner'
    other['source_px'] = [101., 300.]
    inputs[5].append(other)
    assert observe(inputs) == []


def test_two_finite_curve_owners_are_not_resolved_using_nominal_or_nearest_projection(tmp_path):
    inputs = fixture(tmp_path)
    extra = deepcopy(inputs[2]['entities'][2])
    extra['id'] = 'another_upper'
    for key in ('start', 'end'):
        extra[key][1] += 2.
    inputs[2]['entities'].append(extra)
    for nominal in (80., 90.):
        inputs[1][0]['parsed']['nominal'] = nominal
        assert {row['anchor']['entity_id'] for row in observe(inputs)} == {'lower'}


def test_infinite_extension_of_target_line_is_not_a_finite_curve_owner(tmp_path):
    inputs = fixture(tmp_path)
    inputs[2]['entities'][2]['end'] = [170., 195.]
    assert {row['anchor']['entity_id'] for row in observe(inputs)} == {'lower'}


@pytest.mark.parametrize('defect', ['reference_unverified', 'reference_ambiguous',
    'measurement_unverified', 'measurement_ambiguous', 'not_reference'])
def test_unverified_or_ambiguous_dimension_cannot_define_section(tmp_path, defect):
    inputs = fixture(tmp_path)
    if defect == 'reference_unverified':
        inputs[4][0]['binding_evidence']['verified'] = False
    elif defect == 'reference_ambiguous':
        inputs[4][0]['axis_association']['ambiguous'] = True
    elif defect == 'measurement_unverified':
        inputs[4][1]['binding_evidence']['verified'] = False
    elif defect == 'measurement_ambiguous':
        inputs[4][1]['binding_evidence']['ambiguous'] = True
    else:
        inputs[1][0]['parsed']['reference'] = False
    assert observe(inputs) == []
