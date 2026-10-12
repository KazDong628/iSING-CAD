import copy,hashlib,json
import pytest
from contour_agent.ocr import canonical_records
from contour_agent.constraint_binding import _constraint
from contour_agent.parametric_solver import solve_parametric
from contour_agent.source_state import resume_source_state
from contour_agent.unchanged_source_obligations import attach_unchanged_parent,retain_unchanged_obligations
from test_source_priority import source_fixture


def build_checkpoint(tmp_path):
    image,graph,model=source_fixture(tmp_path)
    for e in graph['entities']:e['stable_id']='stable_'+e['id']
    document={'records':[{'text':'20.5','box':[[1,1],[3,1],[3,3],[1,3]]}]}
    candidate=dict(id='c0',kind='distance_x',record_id='r000',entities=['g0'],nodes=['v0','v1'],
        value=20.5,local_reliable=True,evidence={'source_dimension_line_verified':True})
    constraints=[_constraint(candidate,'ocr_local_binding')]
    solution=solve_parametric(graph,constraints)
    inventory={'source_image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
        'all_records':canonical_records(document),'all_candidates':[candidate]}
    parent=resume_source_state(image,document,model,graph,{'constraints':constraints},solution,inventory,tmp_path/'resume')
    child=copy.deepcopy(parent);child['source_evidence'].pop('source_state_resume')
    for i,e in enumerate(child['entities']):
        e.update(id='new'+str(i),parent_entity_ids=['g'+str(i)],
            start_node='new'+e['start_node'],end_node='new'+e['end_node'])
    for n in child['nodes']:n['id']='new'+n['id']
    attach_unchanged_parent(child,parent)
    fresh={'constraints':[],'bindings':[],'counts':{'recognized_dimensions':1}}
    return image,document,model,child,fresh,{'relations':[]}


def test_unchanged_dimension_survives_object_renumbering_without_new_API_claim(tmp_path):
    image,doc,model,graph,bindings,inventory=build_checkpoint(tmp_path)
    result,receipt=retain_unchanged_obligations(image,doc,model,graph,bindings,inventory)
    assert len(receipt['carried'])==1 and not receipt['rejected']
    row=result['constraints'][0]
    assert row['value']==20.5 and row['nodes']==['newv0','newv1'] and row['entities']==['new0']
    assert row['current_API_reconfirmed'] is False and row['repeat_detector_reconfirmed'] is False
    assert bindings['constraints']==[]
    assert inventory['all_candidates'][0]['fresh_detector_reconfirmed'] is False


def test_changed_object_does_not_inherit_old_obligation(tmp_path):
    image,doc,model,graph,bindings,inventory=build_checkpoint(tmp_path)
    node=next(n for n in graph['nodes'] if n['id']=='newv1');node['x']+=.001
    for e in graph['entities']:
        for side in ('start','end'):
            if e[side+'_node']=='newv1':e[side]=[node['x'],node['y']]
    result,receipt=retain_unchanged_obligations(image,doc,model,graph,bindings,inventory)
    assert not result['constraints'] and not receipt['carried']


def test_fresh_positive_dimension_contradiction_revokes_retention(tmp_path):
    image,doc,model,graph,bindings,inventory=build_checkpoint(tmp_path)
    bindings['constraints']=[dict(id='fresh',kind='distance_x',record_id='r000',entities=['new0'],
        nodes=['newv0','newv1'],value=21.,source='ocr_api_binding',required=True,
        enforcement='exact',nominal_source='source_ocr')]
    result,receipt=retain_unchanged_obligations(image,doc,model,graph,bindings,inventory)
    assert len(result['constraints'])==1 and not receipt['carried']
    assert receipt['rejected'][0]['reason']=='fresh_positive_annotation_conflict'


@pytest.mark.parametrize('tamper',['source','ocr','parent','solution'])
def test_changed_source_or_checkpoint_is_rejected(tmp_path,tamper):
    image,doc,model,graph,bindings,inventory=build_checkpoint(tmp_path)
    if tamper=='source':image.write_bytes(b'changed')
    elif tamper=='ocr':doc['records'][0]['text']='21'
    elif tamper=='parent':
        path=tmp_path/'resume/observed-topology.json';path.write_text('{}',encoding='utf8')
    else:
        path=tmp_path/'resume/solution.json';path.write_text('{}',encoding='utf8')
    with pytest.raises(ValueError):retain_unchanged_obligations(image,doc,model,graph,bindings,inventory)
