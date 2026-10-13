"""Dimension ownership supplements a real source turn, never a fitted point."""
from copy import deepcopy
import math

import cv2
import numpy as np
import pytest

from contour_agent.extremum_dimension_support import dimension_supported_extrema


def fixture(tmp_path, *, curve=True, hatch=False, extension=True):
    gray = np.full((600, 600), 255, np.uint8)
    center, radius = np.array([300., 180.]), 150.
    angles = np.linspace(math.radians(30.), math.radians(150.), 400)
    points = center+radius*np.c_[np.cos(angles), np.sin(angles)]
    if hatch:
        material = np.zeros_like(gray)
        polygon = np.vstack([points, [points[-1, 0], 550.], [points[0, 0], 550.]])
        cv2.fillPoly(material, [np.rint(polygon).astype(np.int32)], 1)
        hatches = np.full_like(gray, 255)
        for offset in range(-600, 1200, 13):
            cv2.line(hatches, (offset, 550), (offset+550, 0), 0, 1)
        gray[(material == 1) & (hatches == 0)] = 0
    if curve:
        cv2.polylines(gray, [np.rint(points).astype(np.int32)], False, 0, 3)
    cv2.line(gray, (245, 330), (245, 430), 0, 1)
    if extension:
        cv2.line(gray, (205, 330), (302, 330), 0, 1)
        cv2.line(gray, (245, 430), (355, 430), 0, 1)
    for tip, sign in ((330, 1), (430, -1)):
        cv2.fillPoly(gray, [np.array([[245, tip], [241, tip+12*sign], [249, tip+12*sign]])], 0)
    path = tmp_path/'source.png'
    assert cv2.imwrite(str(path), gray)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    transform = lambda values: np.asarray(values, float)*[1., -1.]+[0., 600.]
    cad_center = np.array([300., 420.])
    def point(angle):
        return (cad_center+radius*np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])).tolist()
    graph = {'entities': [{'id': 'arc', 'type': 'ARC', 'center': cad_center.tolist(),
        'radius': radius, 'start': point(210), 'end': point(330), 'clockwise': False}]}
    records = [{'id': 'height', 'text': '100', 'parsed': {'nominal': 100.},
        'box': [[215., 365.], [233., 365.], [233., 390.], [215., 390.]]}]
    witness = {'record_id': 'height', 'axis': 'y', 'nominal': 100.,
        'binding_evidence': {'verified': True, 'ambiguous': False, 'search_complete': True,
            'endpoint_arrows': [{'verified': True, 'tip_px': [245., 330.], 'direction_px': [0., -1.]},
                                {'verified': True, 'tip_px': [245., 430.], 'direction_px': [0., 1.]}]},
        'axis_association': {'ambiguous': False},
        'line': {'lo': 330., 'hi': 430., 'cross': 245., 'extension_stations': {
            'lo': {'lo': 205., 'hi': 355., 'cross': 330., 'thickness': 1},
            'hi': {'lo': 245., 'hi': 355., 'cross': 430., 'thickness': 1}}}}
    return gray, records, graph, transform, witness


@pytest.mark.parametrize('hatch', [False, True])
def test_real_png_turn_with_finite_extension_is_symbolic(tmp_path, hatch):
    gray, records, graph, transform, witness = fixture(tmp_path, hatch=hatch)
    before = deepcopy((records, graph, witness))
    diagnostics = []
    result = dimension_supported_extrema(gray, records, graph, transform, 6., 'y',
        [witness], diagnostics=diagnostics)
    assert len(result) == 1, diagnostics
    location = result[0]
    assert location['anchor'] == {'type': 'curve_extremum', 'entity_id': 'arc', 'axis': 'y', 'sense': 'minimum'}
    assert [location['x'], location['y']] == [300., 270.]
    assert location['source_px'] == pytest.approx([300., 330.], abs=3.)
    evidence = location['evidence']
    assert evidence['verified'] and evidence['arrow_pair_verified'] and evidence['finite_arc_domain_checked']
    assert evidence['source_inlier_count'] >= 32
    assert evidence['source_turn_depth_px'] >= 2.
    assert evidence['source_extension_support_fraction'] >= .9
    assert not evidence['coordinates_locked'] and not evidence['nominal_used']
    assert not evidence['reference_geometry_used']
    assert not evidence['section_interpretation_not_resolved']
    assert (records, graph, witness) == before


def test_nominal_changes_do_not_change_source_observation(tmp_path):
    gray, records, graph, transform, witness = fixture(tmp_path, hatch=True)
    result = dimension_supported_extrema(gray, records, graph, transform, 6., 'y', [witness])
    records[0]['text'] = '999'
    records[0]['parsed']['nominal'] = 999.
    witness['nominal'] = .01
    assert dimension_supported_extrema(gray, records, graph, transform, 6., 'y', [witness]) == result


@pytest.mark.parametrize('hatch', [False, True])
def test_short_extension_does_not_truncate_turn_context_when_proposal_band_is_four(tmp_path, hatch):
    gray, records, graph, transform, witness = fixture(tmp_path, hatch=hatch)
    # The extension owns the turn at x=300 but ends before its right curved
    # shoulder. Proposal uncertainty must not become a shoulder-length bound.
    witness['line']['extension_stations']['lo']['hi'] = 315.
    observed = {}
    for band in (4., 8.):
        diagnostics = []
        result = dimension_supported_extrema(gray, records, graph, transform, band,
            'y', [witness], diagnostics=diagnostics)
        assert len(result) == 1, diagnostics
        location = result[0]
        assert location['evidence']['source_inlier_count'] >= 32
        assert location['evidence']['source_turn_depth_px'] >= 2.
        assert location['evidence']['source_extension_support_fraction'] >= .9
        observed[band] = location
    assert observed[4.]['anchor'] == observed[8.]['anchor']
    assert observed[4.]['source_px'] == pytest.approx(observed[8.]['source_px'], abs=1.)
    records[0]['text'] = '999'
    records[0]['parsed']['nominal'] = 999.
    witness['nominal'] = .01
    for band in (4., 8.):
        assert dimension_supported_extrema(gray, records, graph, transform, band,
            'y', [witness]) == [observed[band]]


def test_cad_anchor_moves_with_unknowns_and_source_pixels_are_not_coordinate_locks(tmp_path):
    gray, records, graph, transform, witness = fixture(tmp_path)
    entity = graph['entities'][0]
    for key in ('center', 'start', 'end'):
        entity[key][0] += 2.
        entity[key][1] += 1.
    result = dimension_supported_extrema(gray, records, graph, transform, 6., 'y', [witness])
    assert len(result) == 1
    assert [result[0]['x'], result[0]['y']] == [302., 271.]
    assert result[0]['source_px'] == pytest.approx([300., 330.], abs=3.)


@pytest.mark.parametrize('defect', ['arrow', 'ambiguous', 'one_extension', 'wrong_station',
    'wrong_finite_attachment', 'missing_extension_ink', 'duplicate_arc_owner', 'one_sided_curve'])
def test_unverified_or_wrong_dimension_support_does_not_create_anchor(tmp_path, defect):
    gray, records, graph, transform, witness = fixture(tmp_path, extension=defect != 'missing_extension_ink')
    if defect == 'arrow':
        witness['binding_evidence']['endpoint_arrows'][0]['verified'] = False
    elif defect == 'ambiguous':
        witness['axis_association']['ambiguous'] = True
    elif defect == 'one_extension':
        del witness['line']['extension_stations']['hi']
    elif defect == 'wrong_station':
        witness['line']['lo'] = 340.
        witness['line']['extension_stations']['lo']['cross'] = 340.
        witness['binding_evidence']['endpoint_arrows'][0]['tip_px'][1] = 340.
    elif defect == 'wrong_finite_attachment':
        witness['line']['extension_stations']['lo']['hi'] = 280.
    elif defect == 'duplicate_arc_owner':
        other = deepcopy(graph['entities'][0]); other['id'] = 'other'
        graph['entities'].append(other)
    elif defect == 'one_sided_curve':
        gray[:, :290] = 255
    diagnostics = []
    assert dimension_supported_extrema(gray, records, graph, transform, 6., 'y',
        [witness], diagnostics=diagnostics) == []
    assert diagnostics


@pytest.mark.parametrize('hatch', [False, True])
def test_only_dimension_horizontal_line_or_hatches_cannot_prove_a_curve(tmp_path, hatch):
    gray, records, graph, transform, witness = fixture(tmp_path, curve=False, hatch=hatch)
    assert dimension_supported_extrema(gray, records, graph, transform, 6., 'y', [witness]) == []


def test_axis_x_uses_same_finite_turn_proof(tmp_path):
    gray, records, graph, _, witness = fixture(tmp_path, hatch=True)
    gray = np.ascontiguousarray(gray.T)
    for entity in graph['entities']:
        for key in ('center', 'start', 'end'):
            entity[key] = entity[key][::-1]
        entity['clockwise'] = not entity['clockwise']
    for record in records:
        record['box'] = [p[::-1] for p in record['box']]
    for arrow in witness['binding_evidence']['endpoint_arrows']:
        arrow['tip_px'] = arrow['tip_px'][::-1]
        arrow['direction_px'] = arrow['direction_px'][::-1]
    witness['axis'] = 'x'
    transform = lambda values: np.asarray(values, float)*[-1., 1.]+[600., 0.]
    diagnostics = []
    result = dimension_supported_extrema(gray, records, graph, transform, 6., 'x',
        [witness], diagnostics=diagnostics)
    assert len(result) == 1, diagnostics
    assert result[0]['anchor']['axis'] == 'x'
    assert [result[0]['x'], result[0]['y']] == [270., 300.]
    assert result[0]['source_px'] == pytest.approx([330., 300.], abs=3.)


@pytest.mark.parametrize('band', [0., -1., float('nan'), float('inf')])
def test_invalid_band_is_rejected(tmp_path, band):
    gray, records, graph, transform, witness = fixture(tmp_path)
    with pytest.raises(ValueError):
        dimension_supported_extrema(gray, records, graph, transform, band, 'y', [witness])
