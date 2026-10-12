from copy import deepcopy

import pytest

from contour_agent.source_contact_evidence import contact_construction_relations
from contour_agent.source_topology_initial_guess import required_constraints_preserved
from test_source_contact_construction import fixture


def inputs(tmp_path):
    args=fixture(tmp_path);graph=deepcopy(args[4]);relations=contact_construction_relations(*args)
    assert len(relations)==2
    current=[]
    for rid,eid,value,kind in [('r15','g0',15.,'radius'),('r65','g2',65.,'radius'),('angle','g1',30.,'angle')]:
        c=dict(id=rid,kind=kind,record_id=rid,entities=[eid],nodes=[],value=value,source='ocr_local_binding',
            required=True,enforcement='exact',nominal_source='source_ocr',source_arrow_verified=True)
        if kind=='angle':c.update(reference_axis='horizontal',angle_mode='unsigned')
        current.append(c)
    for relation in relations:
        current.append(dict(id=relation['id'],kind='tangent',record_id=None,entities=relation['entities'],
            nodes=relation['nodes'],value=None,source='source_geometry',required=False,
            source_contact_path_evidence=relation['evidence']))
    left,right=deepcopy(graph['entities'][0]),deepcopy(graph['entities'][2])
    left.update(id='old-a',end_node='old-contact');right.update(id='old-b',start_node='old-contact')
    parent=dict(entities=[left,right]);required=[]
    for c,eid in zip(current[:2],['old-a','old-b']):
        r=deepcopy(c);r['entities']=[eid];required.append(r)
    required.append(dict(id='old-tangent',kind='tangent',record_id=None,entities=['old-a','old-b'],
        nodes=['old-contact'],value=None,source='source_geometry',required=False))
    graph['entity_identity']={'parent_to_current':{'old-a':['g0','g1','g2'],'old-b':['g0','g1','g2']}}
    return parent,graph,required,current


def test_fresh_exact_source_path_replaces_inferred_old_contact_with_audit(tmp_path):
    args=inputs(tmp_path);before=deepcopy(args)
    result=required_constraints_preserved(*args)
    assert result['passed']
    replacement=result['inferred_relation_replacements'][0]
    assert replacement['path_entities']==['g0','g1','g2']
    assert replacement['old_constraint_id']=='old-tangent'
    assert len(replacement['new_constraint_ids'])==2
    assert not replacement['old_explicit_annotation_removed']
    assert args==before


@pytest.mark.parametrize('defect',['explicit','record','radius','angle','contact','evidence','missing','outer'])
def test_source_path_does_not_erase_explicit_or_unverified_obligations(tmp_path,defect):
    parent,graph,required,current=inputs(tmp_path)
    if defect=='explicit':required[-1]['required']=True
    elif defect=='record':required[-1]['record_id']='explicit-contact'
    elif defect=='radius':current[0]['source_arrow_verified']=False
    elif defect=='angle':current[2]['value']=31.
    elif defect=='contact':graph['entities'][1]['end'][1]+=1
    elif defect=='evidence':current[-1]['source_contact_path_evidence']['original_source_image_rechecked']=False
    elif defect=='missing':current.pop()
    elif defect=='outer':graph['entities'][0]['start'][0]+=1
    assert not required_constraints_preserved(parent,graph,required,current)['passed']
