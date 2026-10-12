"""Multimodal proposals expand search without replacing source or numeric proof."""
from copy import deepcopy
import json

import httpx
import pytest

from contour_agent.angle_target_provider import (AngleTargetProvider,
    validate_angle_target_response, verified_angle_hints, locate_angle_targets)
from contour_agent.config import Settings
from contour_agent.vision_provider import _InspectionError
from contour_agent import constraint_binding as binding
from test_annotation_angle_binding import angle_source


def hint(rid="source_angle", horizontal=False):
    row={"record_id":rid,"observed_text":"15°","reference_axis":"vertical",
         "source_line_px":[[277,200],[334.2,420]],"reference_line_px":[[250,100],[250,620]],
         "reference_tip_px":[250,500],"target_tip_px":[354,500]}
    if horizontal:
        row["reference_axis"]="horizontal"
        for k in ("source_line_px","reference_line_px"):row[k]=[p[::-1] for p in row[k]]
        for k in ("reference_tip_px","target_tip_px"):row[k]=row[k][::-1]
    return row


@pytest.mark.parametrize("horizontal",[False,True])
def test_source_only_proposal_recovers_missing_hough_line_and_binds_exact_nominal(tmp_path,monkeypatch,horizontal):
    _,gray,record,model,graph=angle_source(tmp_path,horizontal=horizontal)
    record["source_angle_proposals"]=[hint(horizontal=horizontal)]
    monkeypatch.setattr(binding,"_leaders",lambda *args:[])
    rows=binding._angle_source_observations(gray,[record],graph,binding._source_transform(model,graph),5.)
    candidate,=binding._angle_candidates(rows,graph)
    assert candidate["local_reliable"] and candidate["value"]==15.
    assert rows[0]["evidence"]["multimodal_source_hint_count"]==1
    assert rows[0]["evidence"]["reference_arrow"]["verified"]
    assert rows[0]["evidence"]["target_arrow"]["verified"]
    assert abs(rows[0]["source_line"]["observed_direction_deg_from_axis"]-15.)>.1


@pytest.mark.parametrize("change",["arrows","reference","wrong_value","wrong_line"])
def test_model_claim_never_substitutes_for_missing_source_evidence(tmp_path,monkeypatch,change):
    _,gray,record,model,graph=angle_source(tmp_path,arrows=change!="arrows",reference=change!="reference")
    row=hint()
    if change=="wrong_value":row["observed_text"]="12°"
    if change=="wrong_line":row["source_line_px"]=[[600,200],[660,420]]
    record["source_angle_proposals"]=[row]
    monkeypatch.setattr(binding,"_leaders",lambda *args:[])
    assert binding._angle_source_observations(gray,[record],graph,binding._source_transform(model,graph),5.)==[]


@pytest.mark.parametrize("mutation",[
    lambda r:r.update(radius=15),lambda r:r.update(reference_axis="guessed"),
    lambda r:r.update(target_tip_px=[True,500]),lambda r:r.update(target_tip_px=[float("nan"),500]),
    lambda r:r.update(source_line_px=[[0,0],[0,0]]),lambda r:r.update(record_id="unknown"),
])
def test_pixel_schema_rejects_coordinates_numbers_and_unknown_ids(mutation):
    row=hint();mutation(row)
    with pytest.raises(_InspectionError):
        validate_angle_target_response(json.dumps({"proposals":[row],"unknown_record_ids":[]}),{"source_angle"},[800,700])


def test_one_bounded_call_sends_only_source_images_and_preserves_failed_artifacts(tmp_path,monkeypatch):
    path,_,record,_,_=angle_source(tmp_path)
    doc={"records":[record],"ground_truth":{"never_send":"SECRET_GT"}}
    doc["records"][0]["irrelevant"]="SECRET_GT"
    before=deepcopy(doc);sent=[]
    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200,json={"choices":[{"finish_reason":"stop","message":{"content":json.dumps(
            {"proposals":[hint("r000")],"unknown_record_ids":[]})}}]})
    client=httpx.AsyncClient
    monkeypatch.setattr("contour_agent.radius_target_provider.httpx.AsyncClient",
        lambda **kwargs:client(**kwargs,transport=httpx.MockTransport(handler)))
    enriched,receipt=locate_angle_targets(path,doc,tmp_path/"receipt",AngleTargetProvider(Settings(api_key="test-private")))
    assert receipt["network_requests"]==1 and receipt["http_success"] and receipt["schema_success"]
    assert receipt["ground_truth_sent"] is False and receipt["dimensions_verified"] is False
    assert receipt["total_angle_records"]==1 and "total_radius_records" not in receipt
    assert "SECRET_GT" not in json.dumps(sent) and "test-private" not in json.dumps(receipt)
    assert receipt["input_images"][1]["kind"]=="angle_source_crop"
    detail=receipt['input_images'][2]
    assert detail['source_view']=='magnified_angle_detail'
    assert detail['pixel_mapping']['display_per_source_pixel'][0]>1.
    assert enriched["records"][0]["source_angle_proposals"]==[hint("r000")]
    assert doc==before and (tmp_path/"receipt/angle-targets.json").is_file()


def test_invalid_or_failed_online_response_preserves_original_hints(tmp_path):
    _,_,record,_,_=angle_source(tmp_path)
    record["source_angle_proposals"]=[hint()];doc={"records":[record]}
    class Failed:
        def locate(self,*args,**kwargs):
            return {"schema_success":False,"http_success":True,"proposals":[],"network_requests":1}
    enriched,receipt=locate_angle_targets("unused",doc,tmp_path/"failed",Failed())
    assert enriched==doc and receipt["schema_success"] is False
    assert (tmp_path/"failed/effective-angle-source-document.json").is_file()


def test_multimodal_pixel_inaccuracy_is_snapped_to_original_ink_not_to_nominal(tmp_path):
    _,gray,record,_,_=angle_source(tmp_path)
    row=hint()
    row["source_line_px"]=[[p[0]+7.,p[1]] for p in row["source_line_px"]]
    record["source_angle_proposals"]=[row]
    verified,=verified_angle_hints(gray,record,5.)
    assert verified["source_line_search"]["nominal_used_to_rank"] is False
    assert verified["source_line_px"]!=row["source_line_px"]
    assert verified["source_line_search"]["original_proposal_px"]==row["source_line_px"]


def test_priority_pages_keep_all_obligations_and_complete_candidate_groups():
    from contour_agent.binding_provider import _binding_pages
    records=[{"id":f"r{i:03d}","parsed":{"kind":"length","nominal":10.}} for i in range(30)]
    records[-2]["parsed"]["kind"]="angle";records[-1]["parsed"]["kind"]="radius"
    candidates=[{"id":f"c{i:03d}","record_id":r["id"],"local_reliable":i<28} for i,r in enumerate(records)]
    inv={"all_records":records,"all_candidates":candidates,"relations":[],"binding_priority":"radius_and_angle_first"}
    pages,coverage=_binding_pages(inv)
    assert [r["id"] for r in pages[0]["records"][:2]]==["r028","r029"]
    assert {r["id"] for page in pages for r in page["records"]}=={r["id"] for r in records}
    assert coverage["record_exclusions"]==[]


def test_no_angles_never_touches_provider_settings_or_network(tmp_path):
    from types import SimpleNamespace
    doc={"records":[]}
    result,receipt=locate_angle_targets("unused",doc,tmp_path,SimpleNamespace(settings=object()))
    assert result==doc and receipt["network_requests"]==0 and receipt["status"]=="skipped"


def test_connection_retry_has_one_total_budget_and_persists_each_attempt(tmp_path,monkeypatch):
    calls=[]
    def locate(self,*args,**kwargs):
        calls.append(self.settings.api_timeout)
        if len(calls)==1:
            return {"status":"failed","error_code":"timeout","network_requests":1,"http_success":False,
                "schema_success":False,"proposals":[],"unknown_record_ids":["r000"]}
        return {"status":"succeeded","network_requests":1,"http_success":True,"schema_success":True,
            "proposals":[hint("r000")],"unknown_record_ids":[]}
    monkeypatch.setattr(AngleTargetProvider,"locate",locate)
    doc={"records":[{"text":"15°"}]}
    result,receipt=locate_angle_targets("unused",doc,tmp_path,AngleTargetProvider(Settings(api_key="private",api_timeout=40)))
    assert len(calls)==2 and 0<calls[1]<=calls[0]<=40.
    assert receipt["network_requests"]==2 and receipt["http_success"] and not receipt["all_attempts_http_success"]
    assert len(receipt["attempts"])==2 and receipt["total_timeout_seconds"]==40.
    assert (tmp_path/"angle-targets-attempt-01.json").is_file()
    assert result["records"][0]["source_angle_proposals"]==[hint("r000")]


def test_short_line_new_proof_still_rejects_other_records_and_unrelated_ink(tmp_path):
    import numpy as np
    from contour_agent.angle_target_provider import fresh_short_line_supported
    _,gray,record,_,_=angle_source(tmp_path)
    record["source_angle_proposals"]=[hint()];hints=verified_angle_hints(gray,record,5.)
    samples=np.linspace([277,200],[282.2,220],16)
    entity={"type":"LINE","angle_support_evidence":{"record_id":"source_angle",
        "ground_truth_used":False,"requires_angle_binding_and_solve":True,
        "method":"verified_two_radius_joint_line_topology_restore"}}
    line=np.asarray(hints[0]["source_line_px"])
    assert fresh_short_line_supported(gray,entity,"source_angle",hints,samples,line,5.)
    assert not fresh_short_line_supported(gray,entity,"other",hints,samples,line,5.)
    assert not fresh_short_line_supported(gray,entity,"source_angle",[],samples,line,5.)
    assert not fresh_short_line_supported(gray,entity,"source_angle",hints,samples+[40,0],line,5.)


def test_source_ink_failure_gets_bounded_multimodal_feedback_and_keeps_bad_attempt(tmp_path,monkeypatch):
    path,_,record,_,_=angle_source(tmp_path)
    sent=[]
    def handler(request):
        packet=json.loads(request.content);sent.append(packet)
        row=hint('r000')
        if len(sent)==1:row['source_line_px']=[[600,200],[660,420]]
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(
            {'proposals':[row],'unknown_record_ids':[]})}}]})
    client=httpx.AsyncClient
    monkeypatch.setattr('contour_agent.radius_target_provider.httpx.AsyncClient',
        lambda **kwargs:client(**kwargs,transport=httpx.MockTransport(handler)))
    result,receipt=locate_angle_targets(path,{'records':[record]},tmp_path/'retry',
        AngleTargetProvider(Settings(api_key='private',api_timeout=40)))
    assert len(sent)==receipt['network_requests']==2
    assert receipt['attempts'][0]['local_source_hint_checks'][0]['straight_source_ink_verified'] is False
    assert receipt['attempts'][1]['local_source_hint_checks'][0]['straight_source_ink_verified'] is True
    assert 'previous_source_hint_check' in json.dumps(sent[1])
    assert receipt['retained_verified_hint_record_ids']==['r000']
    assert result['records'][0]['source_angle_proposals']==[hint('r000')]
    assert (tmp_path/'retry/angle-targets-attempt-01.json').is_file()


def test_withheld_numeric_record_still_masks_glyphs_during_angle_source_detection(tmp_path,monkeypatch):
    _,gray,record,model,graph=angle_source(tmp_path)
    record['source_angle_proposals']=[hint()]
    withheld={'id':'uncertain_text','text':'41','parsed':{'kind':'length','nominal':41.},
        'box':[[30,30],[60,30],[60,60],[30,60]],'source_text_admission':{'status':'withheld'}}
    seen=[]
    def leaders(_gray,rows):
        seen.extend(r['id'] for r in rows);return []
    monkeypatch.setattr(binding,'_leaders',leaders)
    observations=binding._angle_source_observations(gray,[record,withheld],graph,
        binding._source_transform(model,graph),5.)
    assert 'uncertain_text' in seen
    assert any(r['record_id']=='source_angle' for r in observations)
    record['source_text_admission']={'status':'withheld'}
    assert binding._angle_source_observations(gray,[record,withheld],graph,
        binding._source_transform(model,graph),5.)==[]


def test_binding_builder_passes_all_text_boxes_without_admitting_withheld_values(tmp_path,monkeypatch):
    path,_,record,model,graph=angle_source(tmp_path)
    withheld={'text':'41','box':[[30,30],[60,30],[60,60],[30,60]],
        'source_text_admission':{'status':'withheld'}}
    seen=[]
    def observe(_gray,rows,*args):
        seen.extend(rows);return []
    monkeypatch.setattr(binding,'_angle_source_observations',observe)
    inventory=binding.build_binding_candidates(path,{'records':[record,withheld]},model,graph,tmp_path/'inventory')
    assert next(r for r in seen if r['id']=='r001')['source_text_admission']['status']=='withheld'
    assert not any(r.get('record_id')=='r001' and r.get('local_reliable') for r in inventory['all_candidates'])


def test_new_bad_source_pixels_do_not_erase_previously_verified_same_source_hint(tmp_path):
    path,_,record,_,_=angle_source(tmp_path)
    record['source_angle_proposals']=[hint('r000')]
    bad=hint('r000');bad['source_line_px']=[[600,200],[660,420]]
    class BadPixels:
        def locate(self,*args,**kwargs):
            return {'schema_success':True,'http_success':True,'proposals':[bad],
                'unknown_record_ids':[],'network_requests':1}
    result,receipt=locate_angle_targets(path,{'records':[record]},tmp_path/'bad-new',BadPixels())
    assert result['records'][0]['source_angle_proposals']==[hint('r000')]
    assert receipt['proposals']==[bad]
    assert receipt['retained_existing_verified_hint_record_ids']==['r000']
    assert receipt['retained_verified_hint_record_ids']==[]
def test_missing_angle_selection_matches_provider_list_contract():
    from contour_agent.angle_target_provider import unresolved_angle_record_ids
    document={'records':[{'id':'r1','text':'12°'},{'id':'r2','text':'15°'},
        {'id':'r3','text':'R40'}]}
    result=unresolved_angle_record_ids(document,{'constraints':[{'kind':'angle','record_id':'r001'}]})
    assert type(result) is list and result==['r000']


def test_missing_angle_search_uses_binding_profile_when_structural_editing_disabled():
    from types import SimpleNamespace
    from contour_agent.angle_target_provider import angle_localization_provider
    from contour_agent.config import Settings
    binding=SimpleNamespace(settings=Settings(api_key='test-private',model='source-model'))
    result=angle_localization_provider(binding_provider=binding,editor_provider=None)
    assert isinstance(result,AngleTargetProvider) and result.settings is binding.settings
    editor=SimpleNamespace(settings=Settings(api_key='other-test-private',model='editor-model'))
    assert angle_localization_provider(binding,editor).settings is editor.settings
    assert angle_localization_provider(object(),None) is None
