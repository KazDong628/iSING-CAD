"""One measured extension must not become ambiguous after local LINE repair."""
from copy import deepcopy

import pytest

from contour_agent.radial_dimension_binding import _coincident_virtual_station_options


def options():
    return [{'location': {'source_px': [100.+offset, y],
                'anchor': {'type': 'line_intersection', 'entities': [f'side-{index}', f'cap-{index}']}},
            'proof': {'support_kind': 'source_supported_virtual_corner',
                'extension': {'lo': 20., 'hi': 800., 'cross': 100., 'thickness': 2.}}}
            for index, (offset, y) in enumerate(((.2, 100.), (.3, 700.)))]


def test_same_narrow_extension_retains_both_aliases_and_chooses_its_nearest_attachment():
    rows = options(); before = deepcopy(rows)
    selected = _coincident_virtual_station_options(rows, 0, 20., 4.)
    assert len(selected) == 1
    assert selected[0]['location']['anchor']['entities'] == ['side-0', 'cap-0']
    assert len(selected[0]['proof']['coincident_source_station_members']) == 2
    assert selected[0]['proof']['coordinate_equality_enforced'] is False
    assert rows == before
    selected = _coincident_virtual_station_options(rows, 0, 800., 4.)
    assert selected[0]['location']['anchor']['entities'] == ['side-1', 'cap-1']


@pytest.mark.parametrize('defect', ['different_stroke', 'wide_station', 'tied_attachment',
    'face', 'missing_stroke', 'broad_stroke', 'nonfinite'])
def test_actual_competing_stations_remain_unresolved(defect):
    rows = options()
    if defect == 'different_stroke': rows[1]['proof']['extension']['lo'] = 30.
    if defect == 'wide_station': rows[1]['location']['source_px'][0] = 103.
    if defect == 'tied_attachment': rows[1]['location']['source_px'][1] = 102.
    if defect == 'face': rows[1]['proof']['support_kind'] = 'source_supported_derived_location'
    if defect == 'missing_stroke': rows[1]['proof'].pop('extension')
    if defect == 'broad_stroke':
        for row in rows: row['proof']['extension']['thickness'] = 14.
    if defect == 'nonfinite': rows[1]['location']['source_px'][0] = float('nan')
    assert len(_coincident_virtual_station_options(rows, 0, 20., 4.)) == 2


def test_rotated_station_axes_use_the_corresponding_source_ordinate():
    rows = options()
    for row in rows: row['location']['source_px'].reverse()
    selected = _coincident_virtual_station_options(rows, 1, 20., 4.)
    assert len(selected) == 1
    assert selected[0]['location']['anchor']['entities'] == ['side-0', 'cap-0']
