"""A local fillet proposal cannot discard a previously verified dimension."""
from copy import deepcopy

from scripts.run_source_obligation_replay import check_edit_obligations


def test_all_edit_actions_recheck_known_subset_without_api(tmp_path, monkeypatch):
    parent={'entities':[{'id':'arc','type':'ARC'}]}
    required=[{'id':'k3','kind':'radius','record_id':'r3','value':3.,'entities':['arc'],
        'nodes':[],'required':True,'enforcement':'exact','source':'ocr_local_binding',
        'nominal_source':'source_ocr'}]
    candidates=[{'id':name,'graph':deepcopy(parent)} for name in ['correct-fillet','lost-radius','soft-radius']]
    calls=[]
    def binding(image, document, baseline, graph, output, *, use_api):
        assert use_api is False
        calls.append(output.name)
        rows=deepcopy(required)
        if output.name=='lost-radius':rows=[]
        if output.name=='soft-radius':rows[0]['enforcement']='soft'
        return {'constraints':rows}
    monkeypatch.setattr('contour_agent.constraint_binding.analyze_constraint_bindings',binding)
    receipts=check_edit_obligations('source',{}, {},parent,required,candidates,tmp_path)
    assert calls==['correct-fillet','lost-radius','soft-radius']
    assert receipts['correct-fillet']['passed']
    assert not receipts['lost-radius']['passed'] and not receipts['soft-radius']['passed']
    assert required[0]['enforcement']=='exact'


def test_failed_local_binding_keeps_explicit_failure_receipt(tmp_path, monkeypatch):
    def unavailable(*args,**kwargs):raise ValueError('private diagnostic')
    monkeypatch.setattr('contour_agent.constraint_binding.analyze_constraint_bindings',unavailable)
    receipt=check_edit_obligations('source',{}, {},{},[],[{'id':'edit','graph':{}}],tmp_path)['edit']
    assert not receipt['passed'] and receipt['error_type']=='ValueError'
    assert 'private diagnostic' not in str(receipt)
