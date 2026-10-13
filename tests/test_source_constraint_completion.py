from copy import deepcopy
import json

import pytest

from contour_agent import parametric_pipeline as p, source_radius_hypotheses as h
from contour_agent.parametric_solver import solve_parametric
from contour_agent.source_constraint_completion import complete_source_constraints
from test_parametric_solver import rectangle


def setup_completion(tmp_path, monkeypatch, *, source_ok=True, certificate_ok=True, growth=True):
    graph=rectangle(width=60.,height=40.)
    known=dict(id='known',kind='horizontal',record_id=None,entities=['g0'],nodes=[],value=0.,source='source_geometry')
    added=dict(id='height',kind='distance_y',record_id='r1',entities=[],nodes=['v0','v3'],value=40.,source='ocr_api_binding')
    bindings={'constraints':[known]}
    solution=solve_parametric(graph,[known])
    inventory={'all_records':[],'all_candidates':[],'relations':[
        dict(id='rel0',type='horizontal',entities=['g0'],nodes=[],local_reliable=True,evidence={'verified':True})]}
    calls=[]
    def analyzer(image,document,baseline,observed,output,**kwargs):
        calls.append('binding')
        assert observed is not graph
        path=output/'binding-candidates.json'
        path.write_text(json.dumps({'all_records':[],'all_candidates':[],'relations':[]}),encoding='utf8')
        fresh=[added] if growth else []
        return dict(constraints=deepcopy(fresh),bindings=[],counts={'recognized_dimensions':2},
                    inventory_artifact=str(path),provider={'http_success':True,'network_requests':1})
    def certify(*args):
        calls.append('certificate')
        return dict(passed=certificate_ok,constraints=[known],checks=[])
    monkeypatch.setattr(h,'carried_binding_certificate',certify)
    def solver_wrapper(solver,original,constraints,baseline,output,**kwargs):
        calls.append('solve')
        assert original is graph
        assert kwargs['seed_entities']==solution['entities']
        assert kwargs['budget_seconds']==45.
        return solve_parametric(original,constraints)
    monkeypatch.setattr(p,'_solve_with_source_observation',solver_wrapper)
    def validate(image,document,baseline,original,entities):
        calls.append('source')
        assert original is graph
        return {'passed':True if calls.count('source')==1 else source_ok}
    def no_recovery(image,document,baseline,original,constraints,output,**kwargs):
        return solve_parametric(original,constraints)
    return graph,bindings,solution,inventory,calls,dict(binding_analyzer=analyzer,
        source_validator=validate,numerical_refiner=no_recovery)


def test_new_source_constraint_uses_original_domain_and_preserves_known_equation(tmp_path,monkeypatch):
    graph,bindings,solution,inventory,calls,options=setup_completion(tmp_path,monkeypatch)
    before=deepcopy((graph,bindings,solution,inventory))
    result=complete_source_constraints('image',{}, {},graph,bindings,solution,inventory,tmp_path/'completion',max_rounds=1,use_api=True,**options)
    assert result[3]['accepted']
    assert result[3]['rounds'][0]['all_known_constraints_preserved']
    assert not result[3]['optimization_domain_recentered']
    assert {c['kind'] for c in result[0]['constraints']}=={'horizontal','distance_y'}
    assert (graph,bindings,solution,inventory)==before
    assert calls==['source','binding','certificate','solve','source']


def test_source_first_completion_keeps_original_ink_in_the_enlarged_solve(tmp_path,monkeypatch):
    from contour_agent.reconstruction_priority import source_first_model
    graph,bindings,solution,inventory,calls,options=setup_completion(tmp_path,monkeypatch)
    ink={'source':'original_image_direction_compatible_ink','reference_dxf_read':False}
    def observe(image,document,baseline,original,**kwargs):
        assert original is graph and kwargs['probe_entities']==solution['entities']
        return ink
    monkeypatch.setattr('contour_agent.source_stroke_prior.source_stroke_observation',observe)
    def solve(solver,original,constraints,baseline,output,**kwargs):
        assert original is graph and kwargs['image_stroke_prior'] is ink
        assert kwargs['seed_entities']==solution['entities']
        return solve_parametric(original,constraints)
    monkeypatch.setattr(p,'_solve_with_source_observation',solve)
    result=complete_source_constraints('image',{},source_first_model({}),graph,bindings,
        solution,inventory,tmp_path/'completion',max_rounds=1,**options)
    assert result[3]['accepted']
    assert result[3]['rounds'][0]['original_image_observation_supplied_to_solver'] is True


@pytest.mark.parametrize('defect',['source','certificate','no_growth','provider','regression'])
def test_unaccepted_completion_keeps_previous_solution_and_error_receipt(tmp_path,monkeypatch,defect):
    graph,bindings,solution,inventory,calls,options=setup_completion(tmp_path,monkeypatch,
        source_ok=defect!='source',certificate_ok=defect!='certificate',growth=defect!='no_growth')
    if defect=='provider':
        def fail(*args,**kwargs): raise RuntimeError('sensitive-provider-diagnostic-must-not-be-logged')
        options['binding_analyzer']=fail
    if defect=='regression':
        monkeypatch.setattr(h,'merge_carried_bindings',lambda old,new,fresh,carry:(fresh,new))
    result=complete_source_constraints('image',{}, {},graph,bindings,solution,inventory,tmp_path/'completion',max_rounds=1,**options)
    assert result[:3]==(bindings,solution,inventory)
    assert not result[3]['accepted']
    log=(tmp_path/'completion/completion.json').read_text('utf8')
    assert 'sensitive-provider-diagnostic' not in log


def test_uncertified_previous_solution_never_calls_provider(tmp_path,monkeypatch):
    graph,bindings,solution,inventory,calls,options=setup_completion(tmp_path,monkeypatch)
    solution['accepted']=False
    result=complete_source_constraints('image',{}, {},graph,bindings,solution,inventory,tmp_path/'completion',**options)
    assert not result[3]['accepted'] and calls==[]


def test_provider_interruption_is_persisted_and_propagated(tmp_path,monkeypatch):
    graph,bindings,solution,inventory,calls,options=setup_completion(tmp_path,monkeypatch)
    def interrupted(*args,**kwargs):raise InterruptedError('cancel')
    options['binding_analyzer']=interrupted
    with pytest.raises(InterruptedError):
        complete_source_constraints('image',{}, {},graph,bindings,solution,inventory,tmp_path/'completion',**options)
    receipt=json.loads((tmp_path/'completion/completion.json').read_text('utf8'))
    assert receipt['rounds'][0]['status']=='interrupted'


def test_failed_source_iterate_refines_unknowns_without_unbinding_knowns(tmp_path,monkeypatch):
    graph,bindings,solution,inventory,calls,options=setup_completion(tmp_path,monkeypatch)
    def validate(*args):
        calls.append('source')
        return {'passed':calls.count('source')!=2}
    def recover(image,document,baseline,original,constraints,output,**kwargs):
        calls.append('recover')
        assert original is graph
        assert {(r['kind'],tuple(r['entities']),r['value']) for r in constraints}=={
            ('horizontal',('g0',),0.),('distance_y',(),40.)}
        assert 0 < kwargs['budget_seconds'] <= 90.
        return solve_parametric(original,constraints)
    options.update(source_validator=validate,numerical_refiner=recover)
    result=complete_source_constraints('image',{}, {},graph,bindings,solution,inventory,
        tmp_path/'completion',max_rounds=1,**options)
    assert result[3]['accepted']
    assert calls.count('binding')==1
    assert calls.count('recover')==1
    assert result[3]['rounds'][0]['fixed_annotation_recovery']['source_gate_passed']
    assert result[3]['rounds'][0]['all_known_constraints_preserved']


def test_no_new_binding_still_refines_nonconverged_known_subset(tmp_path, monkeypatch):
    graph, bindings, solution, inventory, calls, options = setup_completion(tmp_path, monkeypatch, growth=False)
    solution['diagnostics']['optimizer_converged'] = False
    solution['underconstrained'] = True
    def refine(image, document, baseline, original, known, current, output, **kwargs):
        calls.append('polish')
        assert original is graph and current is solution and known is bindings
        assert kwargs['budget_seconds'] == 90.
        return current, {'accepted': True, 'ground_truth_used': False}
    monkeypatch.setattr('contour_agent.source_subset_refinement.refine_source_subset', refine)
    result = complete_source_constraints('image', {}, {}, graph, bindings, solution, inventory,
        tmp_path/'completion', max_rounds=1, **options)
    assert result[3]['accepted']
    assert result[3]['reason'] == 'known_equations_preserved_unknown_attributes_refined'
    assert calls == ['source', 'binding', 'certificate', 'polish']
