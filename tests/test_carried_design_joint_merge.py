from copy import deepcopy
from contour_agent.source_radius_hypotheses import merge_carried_bindings
from contour_agent.relation_contract import relation_checks
from contour_agent.parametric_solver import solve_parametric
from test_parametric_solver import rounded_graph,dimensions_and_relations


def test_reobserved_optional_tangent_cannot_duplicate_or_weaken_carried_design_equation():
    constraints=dimensions_and_relations()
    solved=solve_parametric(rounded_graph(),constraints)
    old=deepcopy(next(c for c in constraints if c['kind']=='tangent'))
    old.update(source='source_geometry',required=True,enforcement='exact',evidence_class='source_design_review')
    proof={'verified':True,'evidence_class':'source_design_review','independent_source_tangent_measurement':False}
    relation=dict(id=old['id'][1:],type='tangent',entities=old['entities'],nodes=old['nodes'],
                  local_reliable=True,source='source_design_review',evidence=proof)
    inventory={'relations':[relation]}
    weaker={**old,'id':'kfresh','required':False,'value':None,'evidence_class':None}
    fresh={'constraints':[weaker],'bindings':[{'relation_id':'fresh','accepted':True,'source':'source_geometry'}]}
    carry={'passed':True,'constraints':[old]}
    original=deepcopy((inventory,fresh,carry))
    result,inv=merge_carried_bindings(inventory,{'relations':[]},fresh,carry)
    assert (inventory,fresh,carry)==original
    assert len(result['constraints'])==1 and result['constraints'][0]['required']
    assert result['constraints'][0]['evidence_class']=='source_design_review'
    assert result['bindings'][0]['equation_status']=='consolidated_with_carried_known_obligation'
    assert result['bindings'][-1]['source']=='source_design_review'
    assert not result['bindings'][-1]['evidence']['independent_source_tangent_measurement']
    assert relation_checks(solved['entities'],result['constraints'])['passed']
    assert inv['relations'][0]['evidence']==proof
