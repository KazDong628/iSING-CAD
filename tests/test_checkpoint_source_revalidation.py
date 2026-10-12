"""Fresh source revalidation, not inventory byte equality, protects a subset."""
from copy import deepcopy

import pytest

from contour_agent.parametric_pipeline import _final_checkpoint_replacement_gate
from contour_agent.parametric_solver import solve_parametric
from contour_agent.reconstruction_feedback import reconstruction_feedback
from contour_agent.source_state import revalidate_checkpoint_binding_subset
from test_source_radius_hypotheses import fixture


def rechecked_case(tmp_path):
    image,document,graph,known,inventory=fixture(tmp_path)
    bindings={'constraints':[known],'bindings':[]}
    solution=solve_parametric(graph,[known])
    trusted=dict(graph=graph,bindings=bindings,solution=solution,inventory=inventory,
        feedback=reconstruction_feedback(graph,bindings,solution),
        certificate={'source_evidence_sha256':'old-evidence'})
    fresh=deepcopy(inventory);fresh['relations']=[]
    updated,fresh,receipt=revalidate_checkpoint_binding_subset(image,document,{},trusted,
        {'constraints':[],'bindings':[]},fresh,tmp_path/'fresh-recheck')
    assert receipt['passed']
    receipt['input_identity_verified_against_checkpoint']=True
    rejected={'accepted':False,'status':'optimization_budget_exhausted','constraints':[]}
    attempt=dict(bindings=updated,solution=rejected,source_validation={'passed':False},
        feedback=reconstruction_feedback(graph,updated,rejected),
        source_binding_revalidation_certificate=receipt)
    return trusted,attempt


def test_nondetection_with_fresh_native_source_proof_keeps_checkpoint_after_failed_solve(tmp_path):
    trusted,attempt=rechecked_case(tmp_path)
    gate=_final_checkpoint_replacement_gate(trusted,attempt,evidence_sha256='different-inventory')
    assert gate['known_source_subset_independently_revalidated']
    assert gate['retain_checkpoint'] and not gate['replace_checkpoint']
    assert 'final_numerical_solution_not_accepted' in gate['reasons']
    assert not gate['same_independent_source_evidence']


@pytest.mark.parametrize('defect',['input','typed_identity','count','carry','check'])
def test_incomplete_revalidation_cannot_override_changed_source_evidence(tmp_path,defect):
    trusted,attempt=rechecked_case(tmp_path)
    proof=attempt['source_binding_revalidation_certificate']
    if defect=='input':proof['input_identity_verified_against_checkpoint']=False
    elif defect=='typed_identity':proof['previous_typed_constraints_sha256']='different-obligations'
    elif defect=='count':proof['previous_constraint_count']+=1
    elif defect=='carry':proof['carried_binding_certificate']['passed']=False
    elif defect=='check':proof['carried_binding_certificate']['checks'][0]['passed']=False
    gate=_final_checkpoint_replacement_gate(trusted,attempt,evidence_sha256='different-inventory')
    assert not gate['retain_checkpoint'] and not gate['replace_checkpoint']


def test_changed_source_file_fails_revalidation_and_does_not_merge_old_bindings(tmp_path):
    image,document,graph,known,inventory=fixture(tmp_path)
    trusted=dict(graph=graph,bindings={'constraints':[known]},solution=solve_parametric(graph,[known]),inventory=inventory)
    image.write_bytes(b'changed-original-image')
    fresh={'constraints':[],'bindings':[]}
    updated,_,receipt=revalidate_checkpoint_binding_subset(image,document,{},trusted,fresh,inventory,tmp_path/'failed')
    assert not receipt['passed'] and updated==fresh
    assert receipt['carried_binding_certificate']['reason']=='original_binding_image_changed'
