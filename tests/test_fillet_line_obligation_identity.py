"""Finite supports keep their labels through a two-parent fillet insertion."""
from copy import deepcopy

from contour_agent.source_topology_initial_guess import required_constraints_preserved


def case():
    old={'entities':[
        dict(id='a',type='LINE',start=[0.,0.],end=[10.,0.],start_node='p',end_node='q'),
        dict(id='b',type='LINE',start=[10.,0.],end=[10.,10.],start_node='q',end_node='r')]}
    new={'entities':[
        dict(id='x',type='LINE',start=[0.,0.],end=[8.,0.],start_node='s',end_node='t'),
        dict(id='f',type='ARC',start=[8.,0.],end=[10.,2.],start_node='t',end_node='u'),
        dict(id='y',type='LINE',start=[10.,2.],end=[10.,10.],start_node='u',end_node='v')],
        'entity_identity':{'parent_to_current':{'a':['x','f','y'],'b':['x','f','y']}}}
    known=[dict(id='k1',kind='horizontal',entities=['a'],nodes=[],source='source_geometry'),
           dict(id='k2',kind='angle',record_id='angle',value=24.,reference_axis='horizontal',
                entities=['b'],nodes=[],required=True,enforcement='exact',source='ocr_api_binding'),
           dict(id='k3',kind='distance_y',record_id='height',value=10.,entities=['a','b'],nodes=[],
                anchors=[dict(type='support_level',entity_id='a',axis='y'),
                         dict(type='support_level',entity_id='b',axis='y')],source='ocr_local_binding')]
    current=deepcopy(known)
    for row in current:
        row['entities']=[{'a':'x','b':'y'}[eid] for eid in row['entities']]
        for anchor in row.get('anchors',[]):anchor['entity_id']={'a':'x','b':'y'}[anchor['entity_id']]
    return old,new,known,current


def test_both_trimmed_lines_retain_typed_angle_and_support_level_dimensions():
    old,new,known,current=case()
    receipt=required_constraints_preserved(old,new,known,current)
    assert receipt['passed']
    assert {e:r['entity_id'] for e,r in receipt['retained_finite_line_identity_mapping'].items()}=={'a':'x','b':'y'}


def test_changed_angle_value_or_a_new_trimmed_endpoint_is_not_certified():
    old,new,known,current=case()
    current[1]['value']=23.
    assert not required_constraints_preserved(old,new,known,current)['passed']
    old,new,known,current=case()
    known.append(dict(id='endpoint',kind='distance_y',record_id='end',value=0.,
                      entities=['a'],nodes=['q']))
    current.append(dict(id='endpoint',kind='distance_y',record_id='end',value=0.,
                        entities=['x'],nodes=['t']))
    assert not required_constraints_preserved(old,new,known,current)['passed']


def test_arbitrary_line_split_with_two_collinear_descendants_stays_ambiguous():
    old,new,known,current=case()
    new['entities'][1]=dict(id='f',type='LINE',start=[8.,0.],end=[10.,0.])
    result=required_constraints_preserved(old,new,known,current)
    assert not result['passed']
    assert 'a' not in result['retained_finite_line_identity_mapping']
