"""Missing design contacts must become equations, not a cosmetic overlay fix."""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import pytest

from contour_agent.joint_review_provider import validate_joint_review
from contour_agent.source_joint_completion import analytic_fillet_seed, complete_source_joints
from contour_agent import source_joint_completion as completion
from contour_agent import parametric_pipeline as pipeline
from contour_agent.parametric_solver import solve_parametric
from contour_agent.relation_contract import _geometry, _joint_check
from test_parametric_solver import rounded_graph, dimensions_and_relations


@pytest.mark.parametrize('defect', ['extra', 'missing', 'duplicate', 'foreign', 'number', 'unknown_record'])
def test_joint_response_cannot_introduce_values_or_unknown_identities(defect):
    value={'joints':[{'joint_id':'j0','relationship':'tangent','source_record_ids':['r0']}]}
    if defect=='extra':value['coordinates']=[0,0]
    elif defect=='missing':value['joints']=[]
    elif defect=='duplicate':value['joints']*=2
    elif defect=='foreign':value['joints'][0]['joint_id']='other'
    elif defect=='number':value['joints'][0]['radius']=15
    elif defect=='unknown_record':value['joints'][0]['source_record_ids']=['other']
    with pytest.raises(ValueError):validate_joint_review(value,{'j0'},{'r0'})


def test_fillet_initialization_corrects_contacts_without_changing_known_directions_or_radius():
    graph=rounded_graph(noise=.01);constraints=dimensions_and_relations()
    solved=solve_parametric(graph,[c for c in constraints if c['kind']!='tangent'])
    assert solved['accepted']
    before=deepcopy(solved['entities'])
    seed,receipt=analytic_fillet_seed(before,constraints)
    assert before==solved['entities'] and receipt['entities']
    assert receipt['initial_guess_only'] and not receipt['endpoints_fixed_in_solver']
    for a,b in zip(seed,seed[1:]+seed[:1]):
        check=_joint_check(_geometry(a),_geometry(b),a['end_node'])
        assert check['passed'],check
    assert all(e['radius']==3. for e in seed if e['type']=='ARC')


def case(tmp_path,monkeypatch):
    image=tmp_path/'source.bin';image.write_bytes(b'original source fixture')
    graph=rounded_graph(noise=.01);graph['source_sha256']=hashlib.sha256(image.read_bytes()).hexdigest()
    constraints=[c for c in dimensions_and_relations() if c['kind']!='tangent']
    bindings={'constraints':constraints,'counts':{}}
    solution=solve_parametric(graph,constraints)
    inv={'relations':[]};calls=[]
    rows=[dict(joint_id='j0',entities=['g000','g001'],node_id='v001',source_radius_record_ids=['r002'])]
    monkeypatch.setattr(completion,'joint_review_inventory',lambda *a:(deepcopy(rows),[{'id':'r002'}]))
    monkeypatch.setattr(pipeline,'_constraint_receipts_complete',lambda b,s,g:all(x['passed'] for x in s['constraints']))
    class Review:
        disagreement=False
        def review(self,*args,view,**kwargs):
            calls.append('api')
            return dict(http_success=True,schema_success=True,network_requests=1,joints=[dict(
                joint_id='j0',relationship='corner' if self.disagreement and view else 'tangent',
                source_record_ids=['r002'])],extra_untrusted='SECRET-MUST-NOT-PERSIST')
    def refine(image,doc,baseline,original,cs,output,**kwargs):
        calls.append('solve');assert original is graph
        assert cs[:len(constraints)]==constraints
        from contour_agent.parametric_solver import _sample_entity
        points=np.concatenate([_sample_entity(e,25) for e in solution['entities']]).tolist()
        points.append(points[0])
        return solve_parametric(graph,cs,seed_entities=kwargs['initial_seed_entities'],
            source_observation={'units':graph['units'],'provenance':'input_mask_boundary',
                'reference_dxf_read':False,'points':points,
                'optimization_budget':{'max_wall_seconds':10.,'max_objective_evaluations':10000}})
    def validate(*args):calls.append('source');return {'passed':True}
    def export(*args):
        calls.append('native')
        return {'validation':{k:dict(passed=True,dxf_readback_performed=True) for k in
            ('exact_radius_validation','strict_relation_validation','native_measurement_validation',
             'native_source_angle_validation','native_axis_relation_validation')}}
    return image,graph,bindings,solution,inv,calls,Review(),dict(refiner=refine,source_validator=validate,exporter=export)


def test_source_consensus_enlarges_problem_and_keeps_every_old_equation(tmp_path,monkeypatch):
    image,g,b,s,inv,calls,provider,options=case(tmp_path,monkeypatch)
    original=deepcopy((g,b,s,inv))
    fresh,solved,new_inventory,receipt=complete_source_joints(image,{}, {},g,b,s,inv,tmp_path/'out',
        provider=provider,use_api=True,**options)
    assert receipt['accepted'] and len(fresh['constraints'])==len(b['constraints'])+1
    assert receipt['all_prior_equations_retained']
    assert new_inventory['relations'][0]['evidence']['independent_source_tangent_measurement'] is False
    assert calls==['api','api','solve','source','native']
    assert (g,b,s,inv)==original
    assert 'SECRET-MUST-NOT-PERSIST' not in (tmp_path/'out/completion.json').read_text('utf8')


@pytest.mark.parametrize('defect',['disagreement','unavailable','solver','source','native','identity'])
def test_failed_contact_completion_preserves_previous_artifacts_and_reason(tmp_path,monkeypatch,defect):
    image,g,b,s,inv,calls,provider,options=case(tmp_path,monkeypatch)
    if defect=='disagreement':provider.disagreement=True
    if defect=='unavailable':
        def fail(*a,**kw):raise RuntimeError('SECRET FAILURE')
        provider.review=fail
    if defect=='solver':options['refiner']=lambda *a,**kw:{'accepted':False}
    if defect=='source':options['source_validator']=lambda *a:{'passed':False}
    if defect=='native':options['exporter']=lambda *a:{'validation':{}}
    if defect=='identity':image.write_bytes(b'changed image')
    result=complete_source_joints(image,{}, {},g,b,s,inv,tmp_path/'out',provider=provider,use_api=True,**options)
    assert result[:3]==(b,s,inv) and not result[-1]['accepted']
    if defect in ('disagreement','unavailable','identity'):assert 'solve' not in calls
    assert 'SECRET FAILURE' not in (tmp_path/'out/completion.json').read_text('utf8')


def test_offline_mode_does_not_invent_design_contacts(tmp_path,monkeypatch):
    image,g,b,s,inv,calls,provider,options=case(tmp_path,monkeypatch)
    result=complete_source_joints(image,{}, {},g,b,s,inv,tmp_path/'out',provider=provider,use_api=False,**options)
    assert not result[-1]['accepted'] and calls==[]


def test_ocr_vertical_context_contains_uncropped_glyphs_without_drawn_overlay():
    from PIL import Image
    from contour_agent.source_ocr_review import _crop_payload
    import base64
    from io import BytesIO
    source=Image.new('RGB',(300,700),'white')
    source.putpixel((22,180),(0,0,0))
    row={'id':'r0','box':[[10,300],[35,300],[35,380],[10,380]]}
    url,audit=_crop_payload(source,row,0)
    decoded=Image.open(BytesIO(base64.b64decode(url.split(',')[1])))
    assert audit['source_crop_box'][1] <= 180
    assert audit['reading_axis_context_expanded'] and not audit['source_pixels_annotated']
    assert decoded.getpixel((22-audit['source_crop_box'][0],180-audit['source_crop_box'][1]))==(0,0,0)
