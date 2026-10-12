from copy import deepcopy
import json
from pathlib import Path
import httpx
from PIL import Image
import pytest
from contour_agent.config import Settings
from contour_agent.dataset import read_ocr
from contour_agent.provider import ProviderError
from contour_agent.source_ocr_review import (SourceOCRReviewProvider, review_source_ocr,
    load_effective_ocr, _object_hash, _hash, _valid_transcriptions)


def source(tmp_path,texts):
    image=tmp_path/"source.png";Image.new("RGB",(600,600),"white").save(image)
    raw=tmp_path/"raw.json"
    raw.write_text(json.dumps({"meta":{"original_size":{"width":600,"height":600}},
        "records":[{"text":t,"box":[[20+40*i,80],[45+40*i,80],[45+40*i,170],[20+40*i,170]]} for i,t in enumerate(texts)]}),"utf8")
    return image,raw,read_ocr(raw)


class Fake:
    def __init__(self,values):self.values=values;self.calls=[]
    def transcribe(self,image,rows,*,view,timeout):
        self.calls.append(deepcopy(rows))
        assert all(set(r)=={"id","box"} for r in rows)
        return {"status":"succeeded","http_success":True,"schema_success":True,"network_requests":1,
                "transcriptions":[{"id":r["id"],"text":self.values[view][r["id"]]} for r in rows]}


def test_consensus_changes_only_source_text_and_restores_exact_document(tmp_path):
    image,raw,doc=source(tmp_path,["1135","66.85","+18","4606.038","864"])
    before=deepcopy(doc);provider=Fake([{f"r{i:03}":v for i,v in enumerate(["135±2","69.85","181±1","46.038"])}]*2)
    model={"scale":{"bindings":[{"record_id":"r004"}]},"entities":[{"radius":99999,"gt":"NEVER"}]}
    effective,audit=review_source_ocr(image,doc,model,tmp_path/"out",provider=provider,use_api=True,raw_ocr_path=raw)
    assert doc==before and read_ocr(raw)==before
    assert [r["text"] for r in effective["records"]]==["135±2","69.85","181±1","46.038","864"]
    assert len(provider.calls)==2 and audit["changed_record_ids"]==["r000","r001","r002","r003"]
    assert audit["decisions"][4]["status"]=="scale_record_locked"
    assert audit["consensus_is_ground_truth"] is False
    assert load_effective_ocr(image,raw,tmp_path/"out")==effective


@pytest.mark.parametrize("failure",["conflict","exception","unknown","null"])
def test_disagreement_failure_unreadable_preserve_original_and_draft(tmp_path,failure):
    image,raw,doc=source(tmp_path,["1135"]);out=tmp_path/"out";out.mkdir();(out/"drawing.dxf").write_bytes(b"draft")
    values={"conflict":["135±2","136±2"],"unknown":["()","()"],"null":[None,None]}.get(failure,["135","135"])
    provider=Fake([{ "r000":v} for v in values])
    if failure=="exception":
        def broken(*a,**k):raise RuntimeError("SECRET BODY")
        provider.transcribe=broken
    effective,audit=review_source_ocr(image,doc,{},out,provider=provider,use_api=True,raw_ocr_path=raw)
    assert effective["records"][0]["text"]==doc["records"][0]["text"] and not audit["changed_record_ids"]
    assert bool(effective["records"][0].get("source_text_admission")) == (failure=="conflict")
    assert (out/"drawing.dxf").read_bytes()==b"draft"
    assert "SECRET BODY" not in json.dumps(audit)
    assert load_effective_ocr(image,raw,out)==effective


def test_effective_ocr_tamper_coordinate_or_receipt_is_rejected(tmp_path):
    image,raw,doc=source(tmp_path,["1135"]);out=tmp_path/"out"
    provider=Fake([{"r000":"135±2"}]*2)
    review_source_ocr(image,doc,{},out,provider=provider,use_api=True,raw_ocr_path=raw)
    effective=out/"effective-source-ocr.json";auditfile=out/"source-ocr-review.json"
    mutated=json.loads(effective.read_text("utf8"));mutated["records"][0]["box"][0][0]+=5
    effective.write_text(json.dumps(mutated),"utf8")
    audit=json.loads(auditfile.read_text("utf8"));audit["effective_file_sha256"]=_hash(effective);audit["effective_document_sha256"]=_object_hash(mutated)
    auditfile.write_text(json.dumps(audit),"utf8")
    with pytest.raises(ValueError,match="不一致"):load_effective_ocr(image,raw,out)


def test_raw_source_changes_cannot_reuse_review(tmp_path):
    image,raw,doc=source(tmp_path,["1135"]);out=tmp_path/"out"
    review_source_ocr(image,doc,{},out,use_api=False,raw_ocr_path=raw)
    Image.new("RGB",(600,600),"black").save(image)
    with pytest.raises(ValueError,match="不一致"):load_effective_ocr(image,raw,out)


def test_page_limit_and_unreviewed_records_are_explicit(tmp_path):
    image,raw,doc=source(tmp_path,[str(i+10) for i in range(19)])
    provider=Fake([{f"r{i:03}":str(i+10) for i in range(19)}]*2)
    effective,audit=review_source_ocr(image,doc,{},tmp_path/"out",provider=provider,use_api=True,raw_ocr_path=raw)
    assert len(provider.calls)==4 and max(map(len,provider.calls))==8
    assert sum(d["status"]=="record_budget_exhausted" for d in audit["decisions"])==3
    assert effective==doc


def test_http_protocol_only_sends_pixels_ids_never_original_text_or_cad(tmp_path,monkeypatch):
    image,raw,doc=source(tmp_path,["1135"]);captured=[]
    def handler(request):
        payload=json.loads(request.content);captured.append(payload)
        return httpx.Response(200,json={"choices":[{"finish_reason":"stop","message":{"content":json.dumps({"transcriptions":[{"id":"r000","text":"135±2"}]})}}]})
    real=httpx.AsyncClient
    monkeypatch.setattr("contour_agent.source_ocr_review.httpx.AsyncClient",lambda *a,**k:real(*a,**k,transport=httpx.MockTransport(handler)))
    provider=SourceOCRReviewProvider(Settings(api_key="PRIVATE TOKEN",base_url="https://example.com/v1",api_timeout=5))
    effective,audit=review_source_ocr(image,doc,{"entities":[{"radius":987654,"secret":"GT_SENTINEL"}]},tmp_path/"out",provider=provider,use_api=True,raw_ocr_path=raw)
    assert len(captured)==2 and effective["records"][0]["text"]=="135±2"
    assert "1135" not in str(captured) and "987654" not in str(captured) and "GT_SENTINEL" not in str(captured)
    assert "PRIVATE TOKEN" not in str(audit)
    assert all(p["provider"]["cad_values_sent"] is False for p in audit["pages"])


def test_http_failure_does_not_leak_body_or_credentials(tmp_path,monkeypatch):
    image,raw,doc=source(tmp_path,["1135"])
    real=httpx.AsyncClient
    monkeypatch.setattr("contour_agent.source_ocr_review.httpx.AsyncClient",lambda *a,**k:real(*a,**k,transport=httpx.MockTransport(lambda r:httpx.Response(500,text="SECRET BODY"))))
    provider=SourceOCRReviewProvider(Settings(api_key="PRIVATE TOKEN",base_url="https://example.com/v1",api_timeout=5))
    effective,audit=review_source_ocr(image,doc,{},tmp_path/"out",provider=provider,use_api=True,raw_ocr_path=raw)
    assert effective==doc and audit["status"]=="completed_with_unresolved"
    assert "SECRET BODY" not in str(audit) and "PRIVATE TOKEN" not in str(audit)
    assert all(p["provider"]["http_status"]==500 for p in audit["pages"])


@pytest.mark.parametrize("row",[{"id":[],"text":"5"},{"id":"r000","text":"5","coordinate":[1,2]}])
def test_response_schema_forbids_untrusted_fields_or_non_dimensions(row):
    with pytest.raises(ProviderError):_valid_transcriptions({"transcriptions":[row]},{"r000"})


def test_all_nested_scale_witnesses_are_locked_without_api_use(tmp_path):
    image,raw,doc=source(tmp_path,["1135","66.85","+18"])
    model={"scale":{"bindings":[{"record_id":"r000"}],
        "linear_witnesses":[{"record_id":"r001"}],
        "alternatives":[{"cross_evidence_consistency":{"record_id":"r002"}}]}}
    provider=Fake([{},{}])
    effective,audit=review_source_ocr(image,doc,model,tmp_path/"out",provider=provider,use_api=True,raw_ocr_path=raw)
    assert effective==doc and provider.calls==[]
    assert audit["protected_scale_record_ids"]==["r000","r001","r002"]
    assert all(r["status"]=="scale_record_locked" for r in audit["decisions"])


def test_service_passes_effective_ocr_to_dimension_and_parametric_stages(tmp_path,monkeypatch):
    from contour_agent.service import AgentService
    image,raw,doc=source(tmp_path,["1135"])
    checkpoint=tmp_path/"checkpoint.pt";checkpoint.write_bytes(b"unused mock")
    (tmp_path/"dataset"/"origin").mkdir(parents=True)
    service=AgentService(Settings(runtime_root=tmp_path/"runtime",dataset_root=tmp_path/"dataset",api_key="",segmentation_checkpoint=str(checkpoint)))
    def build(image,document,directory,**kwargs):
        directory.mkdir(parents=True,exist_ok=True)
        for name in ("drawing.dxf","model.json","validation.json","preview.svg","overlay.png","dimension-evidence.json"):
            (directory/name).write_text("draft","utf8")
        assert document["records"][0]["text"]=="1135"
        return {"scale":{"status":"resolved","pixels_per_mm":2},"extraction":{"evidence":{}},
            "validation":{"passed":True,"scaled_mm":True,"dimensions_verified":False},"entities":[],
            "bounds":{},"coordinate_system":{"units":"mm"},"automatic_completion":True,
            "issues":[],"polyline_px":[[0,0],[1,0],[1,1],[0,0]]}
    seen=[]
    def refine(image,document,result,output,**kwargs):
        seen.append(document["records"][0]["text"])
        assert (output/"drawing.dxf").read_text("utf8")=="draft"
        return result,{"status":"disabled","accepted":False}
    provider=Fake([{"r000":"135±2"}]*2)
    monkeypatch.setattr(service.source_ocr_review_provider,"transcribe",provider.transcribe)
    monkeypatch.setattr(service.automatic_dimension_provider,"normalize",lambda rows:{"status":"succeeded","dimensions":[],"http_success":True,"schema_success":True})
    monkeypatch.setattr(service.vision_provider,"inspect",lambda *a,**k:{"status":"failed","http_success":False})
    monkeypatch.setattr("contour_agent.service.build_automatic",build)
    monkeypatch.setattr("contour_agent.service.refine_parametric",refine)
    try:
        job=service.create_auto_source(image,raw,use_api=True,asynchronous=False,use_segmentation=True)
        assert seen==["135±2"]
        assert job["dimension_analysis"]["decisions"][0]["local"]["nominal"]==135
        assert job["source_ocr_review"]["changed_record_ids"]==["r000"]
        assert "effective_source_ocr" in job["artifacts"]
        assert read_ocr(raw)["records"][0]["text"]=="1135"
    finally:service.close()


def test_quarantine_one_bad_text_row_keeps_valid_unique_id_peers():
    quarantines=[]
    rows=_valid_transcriptions({"transcriptions":[{"id":"r000","text":"135±2"},
        {"id":"r001","text":"B1"},{"id":"r002","text":"SECRET TOKEN"},
        {"id":"r003","text":"4606.038.7"}]},{"r000","r001","r002","r003"},quarantines=quarantines)
    assert rows[0]["text"]=="135±2" and all(r["text"] is None for r in rows[1:])
    assert len(quarantines)==3 and "SECRET TOKEN" not in repr(quarantines)
    assert "B1" not in repr(quarantines)


def test_length_records_have_budget_priority_over_numeric_unknowns(tmp_path):
    image,raw,doc=source(tmp_path,["B1"]*16+["1135"])
    provider=Fake([{f"r{i:03}":("135±2" if i==16 else "B1") for i in range(17)}]*2)
    effective,audit=review_source_ocr(image,doc,{},tmp_path/"out",provider=provider,use_api=True,raw_ocr_path=raw)
    assert provider.calls[0][0]["id"]=="r016"
    assert effective["records"][16]["text"]=="135±2"
    assert audit["changed_record_ids"]==["r016"]
    assert "SECRET" not in repr(audit)


def test_duplicate_id_or_extra_response_fields_still_reject_page():
    for value,code in [({"transcriptions":[{"id":"r000","text":"5"},{"id":"r000","text":"6"}]},"ocr_schema_duplicate_record_id"),
                       ({"transcriptions":[],"reasoning":"SECRET"},"ocr_schema_envelope")]:
        with pytest.raises(ProviderError) as caught:_valid_transcriptions(value,{"r000","r001"})
        assert caught.value.code==code


def test_priority_review_changes_replay_in_original_record_order(tmp_path):
    image,raw,doc=source(tmp_path,["B1","1135"])
    provider=Fake([{"r000":"15","r001":"135±2"}]*2)
    out=tmp_path/"out"
    effective,audit=review_source_ocr(image,doc,{},out,provider=provider,use_api=True,raw_ocr_path=raw)
    assert [r["id"] for r in provider.calls[0]]==["r001","r000"]
    assert audit["changed_record_ids"]==["r000","r001"]
    assert load_effective_ocr(image,raw,out)==effective
