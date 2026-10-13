"""The online selector sees symbolic measurement semantics, never reference CAD."""
from contour_agent.binding_provider import bounded_inventory


def test_bounded_source_packet_preserves_section_station_and_reference_semantics():
    anchor=dict(type='section_curve_intersection',entity_id='curve',axis='y',branch='positive',
        station=dict(type='source_offset_from_line_intersection',entities=['h','v'],axis='x',
                     direction=1,source_record_id='ref',source_offset=41.,reference=True))
    records=[dict(id='measure',text='26 +3/0',parsed=dict(kind='length',nominal=26.,
        upper_deviation=3.,lower_deviation=0.),box=[]),
        dict(id='ref',text='(41)',parsed=dict(kind='length',nominal=41.,reference=True),box=[])]
    candidate=dict(id='c',record_id='measure',kind='distance_y',entities=['curve','h','v'],nodes=[],
        measurement_policy='source_tolerance_interval',anchors=[anchor,dict(type='curve_extremum',
        entity_id='upper',axis='y',sense='minimum')],local_reliable=True,
        evidence=dict(observed_station_groups=[dict(anchor=anchor,construction=dict(
            source_section_semantics_verified=True,reference_record_id='ref',dimension_record_id='measure',
            reference_offset_is_station_definition_not_independent_equation=True))],
            evaluation_coordinates=[[999.,999.]],secret='never-send'))
    packet=bounded_inventory(dict(units='mm',records=records,candidates=[candidate],relations=[],
                                  ground_truth_secret='never-send'))
    assert packet['records'][1]['parsed']['reference'] is True
    assert packet['candidates'][0]['anchors'][0]==anchor
    assert packet['candidates'][0]['measurement_policy']=='source_tolerance_interval'
    proof=packet['candidates'][0]['evidence']['observed_station_groups'][0]['construction']
    assert proof['source_section_semantics_verified'] is True
    assert proof['reference_offset_is_station_definition_not_independent_equation'] is True
    assert 'evaluation_coordinates' not in repr(packet)
    assert 'never-send' not in repr(packet)
