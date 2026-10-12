from copy import deepcopy
import json

import pytest

from contour_agent.parametric_solver import solve_parametric
from contour_agent.source_state import resume_source_state
from contour_agent.source_topology_initial_guess import line_seed_problem_graphs
from contour_agent.reconstruction_priority import source_first_model
from test_source_priority import source_fixture


def setup_state(tmp_path):
    image,graph,model=source_fixture(tmp_path)
    constraint=dict(id='width',kind='distance_x',record_id='r0',entities=[],nodes=['v0','v1'],
        value=20.5,source='ocr_local_binding',required=True,enforcement='exact',nominal_source='source_ocr')
    bindings=dict(constraints=[constraint])
    solution=solve_parametric(graph,[constraint])
    assert solution['accepted']
    return image,graph,source_first_model(model),bindings,solution


def test_resume_uses_solved_geometry_and_retains_original_domain(tmp_path):
    image,graph,model,bindings,solution=setup_state(tmp_path)
    original=deepcopy((graph,bindings,solution))
    observed=resume_source_state(image,{},model,graph,bindings,solution,{},tmp_path/'resume')
    assert observed['entities']==solution['entities']
    assert observed['entities']!=graph['entities']
    feedback=json.loads((tmp_path/'resume/reconstruction-feedback.json').read_text('utf8'))
    from contour_agent.reconstruction_feedback import geometry_fingerprint
    assert feedback['geometry_sha256']==geometry_fingerprint(observed)
    assert feedback['solver_accepted'] and feedback['source_validation']['passed']
    assert json.loads((tmp_path/'resume/parametric-solve.json').read_text('utf8'))==solution
    domain,parent=line_seed_problem_graphs(observed)
    assert domain==graph and parent==graph
    second=resume_source_state(image,{},model,observed,bindings,solution,{},tmp_path/'second')
    assert line_seed_problem_graphs(second)[0]==graph
    assert (graph,bindings,solution)==original


@pytest.mark.parametrize('defect',['rejected','receipt','native','identity','hash'])
def test_resume_rejects_mismatched_or_tampered_source_state(tmp_path,defect):
    image,graph,model,bindings,solution=setup_state(tmp_path)
    if defect=='rejected':solution['accepted']=False
    elif defect=='receipt':solution['constraints']=[]
    elif defect=='native':solution['entities'][0]['end'][0]+=2
    elif defect=='identity':solution['entities'][0]['id']='wrong'
    elif defect=='hash':graph['source_sha256']='changed'
    with pytest.raises(ValueError):
        resume_source_state(image,{},model,graph,bindings,solution,{},tmp_path/'resume')


def test_resumed_original_domain_is_hash_checked(tmp_path):
    image,graph,model,bindings,solution=setup_state(tmp_path)
    observed=resume_source_state(image,{},model,graph,bindings,solution,{},tmp_path/'resume')
    path=tmp_path/'resume/domain.json';d=json.loads(path.read_text('utf8'))
    d['nodes'][0]['x']+=4;path.write_text(json.dumps(d),'utf8')
    with pytest.raises(ValueError,match='snapshot_changed'):line_seed_problem_graphs(observed)


@pytest.mark.parametrize('original_support,parent_support,expected',[
    (.88,.88,True),(.90,.88,False),(.88,.90,False)])
def test_resume_ink_gate_does_not_ratchet_to_numerical_initialization(
        tmp_path,monkeypatch,original_support,parent_support,expected):
    from contour_agent.parametric_pipeline import _solved_source_validation
    image,graph,model,bindings,solution=setup_state(tmp_path)
    observed=resume_source_state(image,{},model,graph,bindings,solution,{},tmp_path/'resume')
    # The new fixed-angle solution may fit ink less closely than the previous
    # free-angle iterate. The ORIGINAL domain and parent must still both pass
    # the same .035 drop threshold. A resume cannot tighten or loosen them.
    supports=iter([.95,.86,original_support,.86,parent_support,.86])
    def measure(self,points):
        return {'stroke_supported_fraction':next(supports),
            'edge_supported_fraction':.8,'p90_edge_distance_px':1.,
            'support_measurement_version':'source-stroke-support-v2'}
    monkeypatch.setattr('contour_agent.topology._StrokeEvidence.summarize',measure)
    check=_solved_source_validation(image,{},model,observed,solution['entities'])
    assert check['passed'] is expected
    assert check['maximum_support_drop']==.035
    assert check['before']['stroke_supported_fraction']==original_support
    assert check['resumed_initialization_source_comparison']['passed'] is False
    assert check['resumed_initialization_source_comparison']['acceptance_gate'] is False
    assert check['acceptance_thresholds_changed'] is False
    assert check['source_comparison_basis']=='immutable_resume_domain_and_parent'


def test_resume_fixed_ink_comparator_cannot_bypass_mask_gate(tmp_path,monkeypatch):
    from contour_agent.parametric_pipeline import _solved_source_validation
    image,graph,model,bindings,solution=setup_state(tmp_path)
    observed=resume_source_state(image,{},model,graph,bindings,solution,{},tmp_path/'resume')
    monkeypatch.setattr('contour_agent.parametric_pipeline._oracle_mask_validation',
        lambda *args:dict(applicable=True,passed=False,reason='original_mask_gate_failed'))
    # Without source-first opt-in the mask remains a hard acceptance condition.
    hard_model=deepcopy(model);hard_model.pop('reconstruction_priority',None)
    check=_solved_source_validation(image,{},hard_model,observed,solution['entities'])
    assert not check['passed'] and 'original_mask_gate_failed' in check['reasons']


def test_resumed_solved_subset_is_a_recertifiable_native_checkpoint(tmp_path,monkeypatch):
    image,graph,model,bindings,solution=setup_state(tmp_path)
    checkpoint=tmp_path/'resume'
    observed=resume_source_state(image,{},model,graph,bindings,solution,{},checkpoint)
    calls=[]
    def fresh_binding(image_arg,document,baseline,current,output,*,use_api):
        calls.append((current['entities'],use_api))
        output.mkdir(parents=True,exist_ok=True)
        inventory={'all_records':[],'all_candidates':[],'relations':[]}
        (output/'binding-candidates.json').write_text(json.dumps(inventory),'utf8')
        return deepcopy(bindings)
    monkeypatch.setattr('contour_agent.constraint_binding.analyze_constraint_bindings',fresh_binding)
    from contour_agent.parametric_pipeline import _certify_preflight_checkpoint
    certified,receipt=_certify_preflight_checkpoint(image,{},model,observed,checkpoint,tmp_path/'certify')
    assert receipt['status']=='certified' and certified is not None
    assert calls==[(observed['entities'],False)]
    assert certified['bindings']['constraints']==bindings['constraints']
    assert certified['solution']['entities']==solution['entities']
    assert not certified['model']['validation']['source_mask_fidelity_used_as_acceptance_gate']
    assert (tmp_path/'certify/export/drawing.dxf').is_file()


def test_new_arrow_proposals_schedule_fresh_resume_recheck_without_erasing_subset(tmp_path):
    image,graph,model,bindings,solution=setup_state(tmp_path)
    observed=resume_source_state(image,{},model,graph,bindings,solution,{},tmp_path/'resume')
    from contour_agent.source_state import retain_checkpoint_for_fresh_source_recheck
    assert retain_checkpoint_for_fresh_source_recheck(model,observed,frozen_replay=True)
    assert not retain_checkpoint_for_fresh_source_recheck(model,observed,frozen_replay=False)
    assert not retain_checkpoint_for_fresh_source_recheck({},observed,frozen_replay=True)
    assert not retain_checkpoint_for_fresh_source_recheck(model,graph,frozen_replay=True)
    stored=tmp_path/'resume/solution.json';stored.write_text('{}','utf8')
    with pytest.raises(ValueError,match='snapshot_changed'):
        retain_checkpoint_for_fresh_source_recheck(model,observed,frozen_replay=True)


@pytest.mark.parametrize('allow_partial',[False,True])
def test_one_failed_resume_obligation_cannot_revoke_other_source_certificates(tmp_path,monkeypatch,allow_partial):
    from contour_agent.source_topology_initial_guess import retain_line_seed_bindings
    image,graph,model,bindings,solution=setup_state(tmp_path)
    passed=bindings['constraints'][0]
    pending={**deepcopy(passed),'id':'pending','record_id':'r_missing'}
    # Isolate routing from the independently tested source/native audits.
    saved={'domain.json':graph,'inventory.json':{},'solution.json':solution,
        'bindings.json':{'constraints':[passed,pending]}}
    monkeypatch.setattr('contour_agent.source_state.resumed_files',lambda _:saved)
    calls=[]
    def certificate(*args):
        required=args[6];calls.append([r['id'] for r in required])
        checks=[{'constraint_id':r['id'],'record_id':r['record_id'],
            'passed':r['id']!='pending'} for r in required]
        return {'passed':all(c['passed'] for c in checks),'checks':checks,
            'constraints':[deepcopy(r) for r in required if r['id']!='pending']}
    monkeypatch.setattr('contour_agent.source_radius_hypotheses.carried_binding_certificate',certificate)
    def merge(old,inv,fresh,cert):
        assert cert['passed'] and all(r['id']!='pending' for r in cert['constraints'])
        return {**deepcopy(fresh),'constraints':cert['constraints']},deepcopy(inv)
    monkeypatch.setattr('contour_agent.source_radius_hypotheses.merge_carried_bindings',merge)
    fresh={'constraints':[],'bindings':[{'record_id':'r_missing','accepted':False}],
        'counts':{'recognized_dimensions':2}}
    if not allow_partial:
        with pytest.raises(ValueError,match='not_retained'):
            retain_line_seed_bindings(image,{},model,graph,fresh,{},tmp_path/'retain')
    else:
        updated,_,receipt=retain_line_seed_bindings(image,{},model,graph,fresh,{},tmp_path/'retain',allow_partial_resume=True)
        assert receipt['status']=='resumed_source_subset_retained_pending_repair'
        assert receipt['pending_previous_constraints']==[pending]
        assert not receipt['complete_checkpoint_reuse_permitted']
        assert updated['constraints']==[passed] and updated['bindings'][0]['accepted'] is False
        assert updated['counts']['bound_source_records']==1
        assert not json.loads((tmp_path/'retain/resumed-known-source-certificate.json').read_text('utf8'))['passed']
        assert calls==[[passed['id'],'pending'],[passed['id']]]
