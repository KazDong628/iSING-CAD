"""Exact provisional radii cannot silently become final source certificates."""
import copy
import hashlib
import json

import pytest

from contour_agent import source_radius_hypotheses as h
from contour_agent.ocr import canonical_records
from contour_agent.parametric_solver import solve_parametric
from contour_agent.topology_candidates import _affines
from test_parametric_solver import rounded_graph


def fixture(tmp_path):
    image = tmp_path / 'source.png'
    image.write_bytes(b'only-source-image')
    graph = rounded_graph(radius=3., noise=0.)
    document = {'records': [{'id': 'r000', 'text': 'R3', 'box': [[1,1],[2,1],[2,2],[1,2]]}]}
    known = {'id':'kh0','kind':'horizontal','nodes':[],'entities':['g000'],
             'record_id':None,'value':None,'source':'source_geometry'}
    leader = {'arrowhead_verified':True,'arrowhead':{'verified':True,'tip_px':[0.,0.]},
        'shaft_evidence':{'verified':True},'contour_visibility':{'verified':True},
        'source_text_shaft_attachment':{'strong_text_adjacency':True},
        'source_arrow_ownership':{'status':'globally_unique_source_claim'},
        'target_candidates':[{'entity_id':'g001'}]}
    candidate = {'id':'c000','record_id':'r000','kind':'radius','nodes':[],
        'entities':['g001'],'value':3.,'local_reliable':False,
        'evidence':{'whole_primitive_radius':{'checked':True,'passed':False,'status':'requires_topology_repartition'},
                    'uniquely_supported_leader':True,'directed_source_target_count':1,'leader':leader}}
    inventory = {'source_image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
        'all_records':canonical_records(document),'all_candidates':[candidate],
        'relations':[{'id':'h0','type':'horizontal','entities':['g000'],
                      'local_reliable':True,'evidence':{'verified':True}}]}
    return image,document,graph,known,inventory


def test_hypothesis_needs_unique_proven_arrow_and_unbound_whole_interval(tmp_path):
    _,_,graph,known,inventory = fixture(tmp_path)
    assert len(h.radius_hypotheses(inventory,graph,[known])) == 1
    for change in ('unverified_arrow','ambiguous_owner','already_admitted','fitted_value_only'):
        altered = copy.deepcopy(inventory)
        c = altered['all_candidates'][0]
        if change == 'unverified_arrow': c['evidence']['leader']['arrowhead']['verified'] = False
        if change == 'ambiguous_owner': c['evidence']['leader']['source_arrow_ownership']['duplicate_target_ambiguity_preserved'] = True
        if change == 'already_admitted': c['evidence']['whole_primitive_radius']['passed'] = True
        if change == 'fitted_value_only': c['value'] = 2.999
        assert h.radius_hypotheses(altered,graph,[known]) == []


def test_source_owned_fillet_ink_failure_can_retry_fixed_radius_but_missing_evidence_cannot(tmp_path):
    _,_,graph,known,inventory=fixture(tmp_path)
    entity=graph['entities'][1]
    entity['fillet_construction']='original_image_two_line_exact_radius_seed'
    entity['source_refinement']={'ground_truth_used':False}
    interval=inventory['all_candidates'][0]['evidence']['whole_primitive_radius']
    interval.update(status='constructed_fillet_original_image_not_verified',
        method='original_image_exact_two_line_fillet_admission_v1',
        observation='fresh_original_image_ink',source_ink_checks=[{}, {}, {}])
    assert len(h.radius_hypotheses(inventory,graph,[known]))==1
    interval['source_ink_checks']=[]
    assert h.radius_hypotheses(inventory,graph,[known])==[]


def test_solution_graph_preserves_original_affine_and_input_identity(tmp_path):
    _,_,graph,known,_ = fixture(tmp_path)
    before = copy.deepcopy(graph)
    solution = solve_parametric(graph,[known])
    moved = h.graph_from_solution(graph,{},solution)
    transform,_,_ = _affines(graph,{})
    for node in moved['nodes']:
        assert node['source_px'] == pytest.approx(transform([[node['x'],node['y']]])[0])
    assert graph == before
    solution['entities'][0]['id'] = 'replaced'
    with pytest.raises(ValueError,match='count, type or traversal'):
        h.graph_from_solution(graph,{},solution)


def run_loop(tmp_path,monkeypatch,*,source_pass=True,new_binding=True,source_first=False):
    from contour_agent.reconstruction_priority import source_first_model
    image,document,graph,known,inventory = fixture(tmp_path)
    ink={'source':'original_image_direction_compatible_ink','reference_dxf_read':False}
    if source_first:
        def observe(im,doc,base,original,**kwargs):
            assert original is graph and kwargs['probe_entities']==graph['entities']
            return ink
        monkeypatch.setattr('contour_agent.source_stroke_prior.source_stroke_observation',observe)
    observation=({'optimization_budget':{'max_wall_seconds':60.,'max_objective_evaluations':30000},
                  'mask_representation_budget_diagnostic':{'maximum_deviation':.25,'units':'mm'}}
                 if source_first else {'boundary_error_budget':{'maximum_deviation':.25,'units':'mm'}})
    monkeypatch.setattr('contour_agent.parametric_pipeline._solver_source_observation',lambda *_:observation)
    required = []
    def solver(g,cs,*,output_dir,source_observation):
        assert cs[0] == known
        if source_first:
            assert source_observation['image_stroke_prior'] is ink
            assert 'boundary_error_budget' not in source_observation
            assert source_observation['optimization_budget']['max_wall_seconds'] <= 60.
            assert source_observation['mask_representation_budget_diagnostic']['maximum_deviation'] == .25
        required[:] = copy.deepcopy(cs)
        return solve_parametric(g,cs,output_dir=output_dir)
    def analyzer(_im,_doc,_base,_graph,directory,**_options):
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'binding-candidates.json').write_text(json.dumps(inventory),'utf8')
        return {'constraints':copy.deepcopy(required if new_binding else [known]),'bindings':[]}
    result,receipt = h.refine_source_owned_radius_hypotheses(image,document,source_first_model({}) if source_first else {},graph,
        {'constraints':[known]},inventory,tmp_path/'loop',solver=solver,binding_analyzer=analyzer,
        source_validator=lambda *_:{'passed':source_pass},topology_validator=lambda *_:{'passed':True})
    return (image,document,graph,known,inventory),result,receipt


def test_source_first_radius_hypothesis_still_runs_bounded_joint_solver(tmp_path, monkeypatch):
    data, result, receipt = run_loop(tmp_path, monkeypatch, source_first=True)
    assert receipt['status'] == 'source_verified_seed'
    assert receipt['accepted_record_ids'] == ['r000']
    assert receipt['admitted_constraints'][0] == data[3]
    assert receipt['numerical_seed_only'] and not receipt['native_DXF_export_performed']


@pytest.mark.parametrize('source_pass,new_binding,reason',[(False,True,'original_source_or_topology_check_failed'),
    (True,False,'new_radius_not_independently_rebound')])
def test_failed_source_or_binding_never_replaces_original_graph(tmp_path,monkeypatch,source_pass,new_binding,reason):
    data,result,receipt = run_loop(tmp_path,monkeypatch,source_pass=source_pass,new_binding=new_binding)
    assert result == data[2] and receipt['status'] == 'no_admitted_hypothesis'
    assert receipt['attempts'][0]['reason'] == reason
    assert not receipt['attempts'][0]['binding_approved']
    assert (tmp_path/'loop/source-radius-hypotheses.json').exists()


def test_successful_local_trial_is_only_seed_and_known_constraints_stay(tmp_path,monkeypatch):
    data,result,receipt = run_loop(tmp_path,monkeypatch)
    assert receipt['status'] == 'source_verified_seed'
    assert receipt['numerical_seed_only'] and receipt['final_online_binding_required']
    assert not receipt['native_DXF_export_performed']
    assert receipt['admitted_constraints'][0] == data[3]
    assert receipt['accepted_record_ids'] == ['r000']
    assert receipt['maximum_attempts'] == 2 and receipt['maximum_wall_seconds'] == 90
    assert not receipt['attempts'][0]['binding_approved']
    assert result['source_evidence']['source_radius_hypothesis']['topology_changed'] is False


@pytest.mark.parametrize('mutate', ['abstain','artifact','source'])
def test_formal_binding_failure_falls_back_without_promoting_provisional(tmp_path,monkeypatch,mutate):
    data,candidate,receipt = run_loop(tmp_path,monkeypatch)
    image,document,graph,known,inventory = data
    if mutate == 'artifact':
        path = receipt['seed_solution_artifact']
        altered = json.loads(open(path,encoding='utf8').read());altered['status']='tampered'
        open(path,'w',encoding='utf8').write(json.dumps(altered))
    if mutate == 'source': image.write_bytes(b'changed-source')
    def analyzer(_im,_doc,_base,_graph,directory,**_options):
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'binding-candidates.json').write_text(json.dumps(inventory),'utf8')
        return {'constraints':[known],'bindings':[], 'provider':{'schema_success':True,'network_requests':1}}
    original = {'constraints':[known]}
    final,bound,audit = h.certify_refined_radius_seed(image,document,{},graph,original,inventory,
        candidate,receipt,tmp_path/'formal',use_api=True,binding_analyzer=analyzer)
    assert final == graph and bound == original and audit['status'] == 'rejected'
    assert not audit['all_known_constraints_preserved']


def test_formal_binding_preserves_known_metadata_and_new_r_requires_fresh_selection(tmp_path,monkeypatch):
    data,candidate,receipt = run_loop(tmp_path,monkeypatch)
    image,document,graph,known,inventory = data
    required = copy.deepcopy(receipt['admitted_constraints'])
    def analyzer(_im,_doc,_base,_graph,directory,**_options):
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'binding-candidates.json').write_text(json.dumps(inventory),'utf8')
        return {'constraints':required,'bindings':[], 'counts':{}, 'provider':{'schema_success':True,'network_requests':1}}
    final,bound,audit = h.certify_refined_radius_seed(image,document,{},graph,{'constraints':[known]},inventory,
        candidate,receipt,tmp_path/'formal',use_api=True,binding_analyzer=analyzer)
    assert audit['status'] == 'accepted_seed'
    assert audit['all_known_constraints_preserved'] and audit['native_solve_and_export_pending']
    assert final == candidate and len(bound['constraints']) == 2


def test_certificate_cannot_be_applied_to_other_solved_graph(tmp_path):
    image,document,graph,known,inventory=fixture(tmp_path)
    solution=solve_parametric(graph,[known])
    altered=h.graph_from_solution(graph,{},solution)
    altered['nodes'][0]['x']+=.1
    audit=h.carried_binding_certificate(image,document,{},graph,altered,inventory,[known],inventory,
                                        {'constraints':[known]},solution)
    assert not audit['passed'] and audit['reason']=='supplied_graph_differs_from_exact_solution'


@pytest.mark.parametrize('route',['carried','scoped'])
@pytest.mark.parametrize('source_pass',[True,False])
def test_radius_rechecks_receive_original_pixels_records_and_current_geometry(tmp_path,monkeypatch,route,source_pass):
    from PIL import Image
    import numpy as np
    image,document,graph,_,inventory=fixture(tmp_path)
    Image.fromarray(np.full((24,32),173,np.uint8)).save(image)
    inventory['source_image_sha256']=hashlib.sha256(image.read_bytes()).hexdigest()
    candidate=inventory['all_candidates'][0];candidate['local_reliable']=True
    radius={'id':'kr0','kind':'radius','nodes':[],'entities':['g001'],
        'record_id':'r000','value':3.,'source':'ocr_api_binding',
        'required':True,'enforcement':'exact','nominal_source':'source_ocr',
        'source_arrow_verified':True}
    solution=solve_parametric(graph,[radius])
    current=h.graph_from_solution(graph,{},solution)
    # Arrow exactly owns the current finite primitive; only the fresh source
    # feasibility result is varied. Existing rejection remains authoritative.
    transform,_,_=_affines(graph,{})
    candidate['evidence']['leader']['arrowhead']['tip_px']=transform([current['entities'][1]['start']])[0].tolist()
    calls=[]
    def feasibility(entity,nominal,leader,baseline,matched,affine,band,*,gray,records):
        assert matched is current and entity is current['entities'][1]
        assert gray.shape==(24,32) and np.all(gray==173)
        assert records==canonical_records(document) and nominal==3.
        calls.append(True)
        return {'checked':True,'passed':source_pass,'status':'fresh_original_ink_checked'}
    monkeypatch.setattr('contour_agent.constraint_binding._radius_primitive_feasibility',feasibility)
    monkeypatch.setattr(h,'_carried_radius_finite_ownership',lambda *a,**k:{'passed':True})
    if route=='carried':
        audit=h.carried_binding_certificate(image,document,{},graph,current,inventory,[radius],
            {'all_candidates':[]},{'constraints':[]},solution)
        assert audit['passed'] is source_pass, audit
    else:
        audit=h._scoped_radius_rebinding(image,document,{},graph,current,inventory,[radius],tmp_path/'scoped')
        fresh=json.loads((tmp_path/'scoped/binding-candidates.json').read_text('utf8'))
        assert fresh['provisional_radius_checks'][0]['passed'] is source_pass
    assert calls==[True]


@pytest.mark.parametrize('change', ['pixel_roundoff', 'pixel_move', 'cad_move', 'arc_move', 'nonfinite'])
def test_solved_identity_distinguishes_affine_cache_roundoff_from_geometry_changes(tmp_path, change):
    image,document,graph,known,inventory=fixture(tmp_path)
    solution=solve_parametric(graph,[known])
    observed=h.graph_from_solution(graph,{},solution)
    if change=='pixel_roundoff': observed['nodes'][0]['source_px'][0]+=3e-11
    if change=='pixel_move': observed['nodes'][0]['source_px'][0]+=.01
    if change=='cad_move': observed['nodes'][0]['x']+=3e-11
    if change=='arc_move': observed['entities'][1]['radius']+=3e-11
    if change=='nonfinite': observed['nodes'][0]['source_px'][0]=float('nan')
    audit=h.carried_binding_certificate(image,document,{},graph,observed,inventory,[known],inventory,
        {'constraints':[known]},solution)
    assert audit['passed'] is (change=='pixel_roundoff')


def test_radius_checkpoint_is_passed_as_initial_values_without_moving_original_graph(tmp_path, monkeypatch):
    image,document,graph,known,inventory=fixture(tmp_path)
    original=copy.deepcopy(graph)
    seed=solve_parametric(graph,[known])['entities']
    monkeypatch.setattr('contour_agent.parametric_pipeline._solver_source_observation',lambda *_:{
        'boundary_error_budget':{'maximum_deviation':.25,'units':'mm'}})
    required=[]
    def solver(g,cs,*,output_dir,source_observation,seed_entities):
        assert g is graph and g==original
        assert seed_entities==seed
        assert source_observation['boundary_error_budget']['maximum_deviation']==.25
        assert source_observation['boundary_error_budget']['max_wall_seconds']<=60.
        required[:]=copy.deepcopy(cs)
        # This adapter test checks immutable domain/seed routing. Its synthetic
        # source observation lacks boundary points; numerical seed validation
        # is exercised separately by the production source-budget solver tests.
        return solve_parametric(g,cs,output_dir=output_dir)
    def analyzer(_im,_doc,_base,_graph,directory,**_options):
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'binding-candidates.json').write_text(json.dumps(inventory),'utf8')
        return {'constraints':copy.deepcopy(required),'bindings':[]}
    _,receipt=h.refine_source_owned_radius_hypotheses(image,document,{},graph,{'constraints':[known]},
        inventory,tmp_path/'loop',solver=solver,binding_analyzer=analyzer,seed_entities=seed,
        source_validator=lambda *_:{'passed':True},topology_validator=lambda *_:{'passed':True})
    assert receipt['status']=='source_verified_seed' and graph==original
    assert receipt['certified_checkpoint_used_as_initial_values']
    assert not receipt['optimization_domain_recentered']
