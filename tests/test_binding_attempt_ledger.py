"""Every actual binding call survives reporting, including failed repair passes."""
import json
from pathlib import Path

from contour_agent.binding_attempts import collect_binding_attempts, binding_attempt_summary, published_binding_sources
from contour_agent.model_transcript import build_model_transcript
from scripts.report_controlled_iteration import transport_receipts


def write(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')


def provider(n,success=True):
    return {'status':'completed' if success else 'failed','network_requests':n,
            'http_success':success,'schema_success':success,'bindings':[],'relations':[]}


def test_failed_formal_radius_binding_not_hidden_by_original_selected_receipt(tmp_path):
    original=provider(2);formal=provider(3,False)
    write(tmp_path,'parametric-stage.json',{'provider':original,
        'source_radius_seed_certification':{'status':'rejected','provider':formal}})
    write(tmp_path,'constraint-bindings.json',{'provider':original})
    write(tmp_path,'source-radius-certification/full-source-binding/constraint-bindings.json',{'provider':formal})
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    rows=collect_binding_attempts(tmp_path)
    assert [row['phase'] for row in rows]==['initial','radius_rebinding']
    assert binding_attempt_summary(rows)['known_network_requests']==5
    assert binding_attempt_summary(rows)['http_failed_attempts']==1
    assert all(row['recovered_from_artifacts'] for row in rows)
    report=transport_receipts(tmp_path)
    assert sum(row.get('network_requests') or 0 for row in report)==5
    transcript=build_model_transcript({'artifact_directory':str(tmp_path)},tmp_path)
    assert sum(row['transport']['network_requests'] or 0 for row in transcript['stages'])==5
    assert [row['label'] for row in transcript['stages']]==['初始约束绑定','半径候选重新绑定']
    assert all(p.read_bytes()==data for p,data in before.items())


def test_successful_formal_receipt_alias_is_not_counted_twice(tmp_path):
    original=provider(2);formal=provider(3)
    write(tmp_path,'parametric-stage.json',{'provider':formal,
        'source_radius_seed_certification':{'status':'accepted_seed','provider':formal}})
    write(tmp_path,'initial-radius-constraint-bindings.json',{'provider':original})
    write(tmp_path,'constraint-bindings.json',{'provider':formal})
    write(tmp_path,'source-radius-certification/full-source-binding/constraint-bindings.json',{'provider':formal})
    rows=collect_binding_attempts(tmp_path)
    assert len(rows)==2 and binding_attempt_summary(rows)['known_network_requests']==5


def test_trusted_failed_rebinding_then_fallback_preserves_two_identical_calls(tmp_path):
    response=provider(2)
    write(tmp_path,'parametric-stage.json',{'provider':response,
        'source_radius_seed_certification':{'status':'rejected','provider':response}})
    write(tmp_path,'constraint-bindings.json',{'provider':provider(9)}) # reused parent, not new API work
    write(tmp_path,'final-binding-attempt/constraint-bindings.json',{'provider':response})
    write(tmp_path,'final-binding-attempt/source-radius-certification/full-source-binding/constraint-bindings.json',{'provider':response})
    rows=collect_binding_attempts(tmp_path)
    assert [row['phase'] for row in rows]==['radius_rebinding','fallback']
    assert binding_attempt_summary(rows)['known_network_requests']==4


def test_trusted_success_does_not_count_reused_parent_or_selected_alias(tmp_path):
    response=provider(3)
    write(tmp_path,'parametric-stage.json',{'provider':response,
        'source_radius_seed_certification':{'status':'accepted_seed','provider':response}})
    write(tmp_path,'constraint-bindings.json',{'provider':response})
    write(tmp_path,'final-binding-attempt/constraint-bindings.json',{'provider':response})
    write(tmp_path,'final-binding-attempt/initial-radius-constraint-bindings.json',{'provider':provider(9)})
    write(tmp_path,'final-binding-attempt/source-radius-certification/full-source-binding/constraint-bindings.json',{'provider':response})
    rows=collect_binding_attempts(tmp_path)
    assert len(rows)==1 and rows[0]['phase']=='radius_rebinding'
    assert binding_attempt_summary(rows)['known_network_requests']==3


def test_carried_source_constraint_is_never_a_current_API_confirmation():
    constraints=[{'id':'k0','kind':'radius','record_id':'r0','source':'ocr_api_binding'},
        {'id':'k1','kind':'angle','record_id':'r1','source':'ocr_local_binding',
         'admission_method':'carried_source_binding_certificate','prior_binding_source':'ocr_api_binding'}]
    result=published_binding_sources({'constraints':constraints,'dimensions_updated_by_api':True})
    assert result['current_API_confirmed_constraint_count']==1
    assert result['carried_constraint_count']==1
    assert result['constraints'][1]['current_API_reconfirmed'] is False
    assert published_binding_sources({'constraints':constraints})['current_API_confirmed_constraint_count']==0


def test_pending_and_disabled_are_not_actual_network_attempts(tmp_path):
    write(tmp_path,'parametric-stage.json',{'provider':{'status':'pending','network_requests':None}})
    assert collect_binding_attempts(tmp_path)==[]


def test_structural_repair_bindings_and_editor_are_counted_once(tmp_path):
    write(tmp_path,'source-topology-completion/round-01/candidate-01/constraint-bindings.json',{'provider':provider(2)})
    write(tmp_path,'source-topology-completion/round-01/candidate-02/constraint-bindings.json',{'provider':provider(2,False)})
    write(tmp_path,'source-topology-completion/completion.json',{'rounds':[{'round':1,'provider':provider(1)}]})
    # Root publication alias has no provider: actual invocations are in the
    # two independently attempted candidate directories.
    rows=collect_binding_attempts(tmp_path)
    assert len(rows)==2 and binding_attempt_summary(rows)['known_network_requests']==4
    assert sum(r.get('network_requests') or 0 for r in transport_receipts(tmp_path))==5


def test_selected_binding_receipt_does_not_claim_to_be_last_actual_invocation():
    original=provider(2)
    failed=provider(3,False)
    from scripts.report_strict_joint_iteration import _binding_receipt
    selected=_binding_receipt({'provider':original},[
        {'attempt_id':'binding_initial','phase':'initial','provider':original},
        {'attempt_id':'binding_radius_rebinding','phase':'radius_rebinding','provider':failed}])
    assert selected['scope']=='pipeline_selected_binding_receipt'
    assert selected['network_requests']==2
    assert selected['last_actual_binding_attempt']['network_requests']==3
    assert selected['last_actual_binding_attempt']['http_success'] is False
    unknown=_binding_receipt({'provider':original})
    assert unknown['last_actual_binding_attempt'] is None
    assert unknown['last_actual_binding_attempt_known'] is False


def test_post_solve_rebinding_is_an_actual_call_even_when_identical_to_selected_response(tmp_path):
    response = provider(2)
    write(tmp_path,'binding-initial-attempt.json',{'provider':response})
    write(tmp_path,'constraint-bindings.json',{'provider':response})
    write(tmp_path,'post-solve-radius-retry/source-radius-certification/full-source-binding/constraint-bindings.json',
        {'provider':response})
    rows=collect_binding_attempts(tmp_path)
    assert [row['phase'] for row in rows]==['initial','post_solve_radius_rebinding']
    assert binding_attempt_summary(rows)['known_network_requests']==4
    transcript=build_model_transcript({'artifact_directory':str(tmp_path)},tmp_path)
    assert [row['label'] for row in transcript['stages']]==['初始约束绑定','有效解半径候选重新绑定']


def test_source_completion_calls_remain_in_ledger_including_failed_second_round(tmp_path):
    write(tmp_path,'binding-initial-attempt.json',{'provider':provider(1)})
    write(tmp_path,'source-constraint-completion/round-01/constraint-bindings.json',{'provider':provider(2)})
    write(tmp_path,'source-constraint-completion/round-02/constraint-bindings.json',{'provider':provider(1,False)})
    rows=collect_binding_attempts(tmp_path)
    assert [row['phase'] for row in rows]==['initial','source_completion_1','source_completion_2']
    summary=binding_attempt_summary(rows)
    assert summary['known_network_requests']==4 and summary['http_failed_attempts']==1
    transcript=build_model_transcript({'artifact_directory':str(tmp_path)},tmp_path)
    assert [row['label'] for row in transcript['stages']]==[
        '初始约束绑定','有效解源约束补全第一轮','有效解源约束补全第二轮']
