"""A disputed transcription stays visible but cannot become a hard attribute."""
import base64
from copy import deepcopy
from io import BytesIO
import json

import httpx
from PIL import Image, ImageChops, ImageDraw
import pytest

from contour_agent.config import Settings
from contour_agent.constraint_binding import analyze_constraint_bindings, build_binding_candidates
from contour_agent.dataset import read_ocr
from contour_agent.source_ocr_review import (SourceOCRReviewProvider, review_source_ocr,
    load_effective_ocr, canonical_document_sha256, _object_hash, _hash)
from test_constraint_binding import source as dimension_source
from test_source_ocr_review import source, Fake


@pytest.mark.parametrize("texts,withheld",[
    (["69.85",None],True), ([None,"69.85"],True),
    (["69.85","58.69"],True), (["66.85","69.85"],True),
    (["66.85",None],False), ([None,None],False),
    (["66.85","66.85"],False), (["69.85","69.85"],False),
])
def test_disputed_numeric_text_is_withheld_but_consensus_and_unavailable_only_are_not(tmp_path,texts,withheld):
    image,raw,doc=source(tmp_path,["66.85"]);out=tmp_path/"out"
    effective,audit=review_source_ocr(image,doc,{},out,
        provider=Fake([{"r000":text} for text in texts]),use_api=True,raw_ocr_path=raw)
    assert audit["withheld_record_ids"]==(["r000"] if withheld else [])
    assert bool(effective["records"][0].get("source_text_admission")) is withheld
    if withheld:assert effective["records"][0]["text"]=="66.85"
    assert read_ocr(raw)==doc
    restored=load_effective_ocr(image,raw,out)
    assert restored==effective
    assert canonical_document_sha256(restored)==_object_hash(effective)


def test_one_readable_counterevidence_and_failed_peer_still_withholds(tmp_path):
    image,raw,doc=source(tmp_path,["66.85"]);out=tmp_path/"out"
    class Partial(Fake):
        def transcribe(self,*args,view,**kwargs):
            if view==1:raise RuntimeError("SECRET FAILURE")
            return super().transcribe(*args,view=view,**kwargs)
    effective,audit=review_source_ocr(image,doc,{},out,
        provider=Partial([{"r000":"69.85"},{}]),use_api=True,raw_ocr_path=raw)
    assert audit["decisions"][0]["status"]=="review_unavailable"
    assert audit["withheld_record_ids"]==["r000"]
    assert effective["records"][0]["source_text_admission"]["successful_views"]==[True,False]
    assert "SECRET FAILURE" not in json.dumps(audit)
    assert load_effective_ocr(image,raw,out)==effective


def test_unknown_symbol_does_not_become_a_false_numeric_obligation(tmp_path):
    image,raw,doc=source(tmp_path,["B1"])
    effective,audit=review_source_ocr(image,doc,{},tmp_path/"out",
        provider=Fake([{"r000":"15"},{"r000":None}]),use_api=True,raw_ocr_path=raw)
    assert effective==doc and audit["withheld_record_ids"]==[]


def suspect_dimension_fixture(tmp_path):
    image,document,model,graph=dimension_source(tmp_path)
    document["records"][0]["text"]="66.85"
    factor=66.85/150.
    for node in graph["nodes"]:node.update(x=node["x"]*factor,y=node["y"]*factor)
    for entity in graph["entities"]:
        for key in ("start","end"):entity[key]=[x*factor for x in entity[key]]
    model["scale"]["pixels_per_mm"]=2./factor
    raw=tmp_path/"raw.json";raw.write_text(json.dumps(document),"utf8");document=read_ocr(raw)
    # Establish the defect would be real: old OCR can otherwise make a source
    # supported hard constraint; the following absence is not a blank test.
    before=analyze_constraint_bindings(image,document,model,graph,tmp_path/"before")
    assert any(row.get("record_id")=="r000" and row["value"]==66.85 for row in before["constraints"])
    stale=json.loads((tmp_path/"before"/"binding-candidates.json").read_text("utf8"))
    effective,audit=review_source_ocr(image,document,model,tmp_path/"review",
        provider=Fake([{"r000":"69.85"},{"r000":None}]),use_api=True,raw_ocr_path=raw)
    return image,effective,model,graph,stale


def test_source_review_to_inventory_to_binding_preserves_denominator_and_withholds_old_value(tmp_path):
    image,effective,model,graph,_=suspect_dimension_fixture(tmp_path)
    inventory=build_binding_candidates(image,effective,model,graph,tmp_path/"inventory")
    assert inventory["counts"]["ocr_records"]==1
    assert inventory["counts"]["recognized_dimensions"]==1
    assert inventory["counts"]["withheld_source_dimensions"]==1
    assert inventory["all_records"][0]["parsed"]["nominal"]==66.85
    assert inventory["all_records"][0]["source_text_admission"]["status"]=="withheld"
    assert not any(row["record_id"]=="r000" for row in inventory["all_candidates"])
    result=analyze_constraint_bindings(image,effective,model,graph,tmp_path/"after")
    assert result["counts"]["unbound_dimensions"]==1
    assert not any(row.get("record_id")=="r000" for row in result["constraints"])
    assert "unresolved_source_ocr_semantics" in result["issues"]


@pytest.mark.parametrize("api",[False,True])
def test_stale_or_api_selected_old_candidate_cannot_bypass_current_review(tmp_path,monkeypatch,api):
    image,effective,model,graph,stale=suspect_dimension_fixture(tmp_path)
    (tmp_path/"after").mkdir()
    monkeypatch.setattr("contour_agent.constraint_binding.build_binding_candidates",lambda *a,**k:deepcopy(stale))
    class Selector:
        def select(self,*args):
            candidate=next(row for row in stale["all_candidates"] if row["record_id"]=="r000")
            return {"status":"succeeded","http_success":True,"schema_success":True,"network_requests":1,
                "bindings":[{"record_id":"r000","candidate_id":candidate["id"],"observed_text":"66.85"}],
                "relations":[],"input_record_ids":[r["id"] for r in stale["all_records"]],
                "input_candidate_ids":[r["id"] for r in stale["all_candidates"]],
                "input_relation_ids":[r["id"] for r in stale["relations"]]}
    result=analyze_constraint_bindings(image,effective,model,graph,tmp_path/"after",
        provider=Selector(),use_api=api)
    assert not any(row.get("record_id")=="r000" for row in result["constraints"])
    rejected=[row for row in result["bindings"] if row.get("record_id")=="r000"]
    assert rejected and all(not row["accepted"] and row["reason"]=="unresolved_source_ocr_semantics" for row in rejected)


def test_removing_quarantine_metadata_even_with_updated_effective_hash_is_rejected(tmp_path):
    image,raw,doc=source(tmp_path,["66.85"]);out=tmp_path/"out"
    review_source_ocr(image,doc,{},out,provider=Fake([{"r000":"69.85"},{"r000":None}]),use_api=True,raw_ocr_path=raw)
    path=out/"effective-source-ocr.json";stored=json.loads(path.read_text("utf8"))
    stored["records"][0].pop("source_text_admission");path.write_text(json.dumps(stored),"utf8")
    audit_path=out/"source-ocr-review.json";audit=json.loads(audit_path.read_text("utf8"))
    audit.update(effective_file_sha256=_hash(path),effective_document_sha256=_object_hash(stored),withheld_record_ids=[])
    audit_path.write_text(json.dumps(audit),"utf8")
    with pytest.raises(ValueError,match="不一致"):load_effective_ocr(image,raw,out)


def test_vertical_second_view_sends_both_pixel_rotations_under_one_record(tmp_path,monkeypatch):
    image,raw,doc=source(tmp_path,["66.85"])
    pixels=Image.open(image);ImageDraw.Draw(pixels).rectangle((25,90,35,105),fill="black");pixels.save(image)
    captured=[]
    def handler(request):
        payload=json.loads(request.content);captured.append(payload)
        return httpx.Response(200,json={"choices":[{"finish_reason":"stop","message":{"content":json.dumps(
            {"transcriptions":[{"id":"r000","text":None}]})}}]})
    real=httpx.AsyncClient
    monkeypatch.setattr("contour_agent.source_ocr_review.httpx.AsyncClient",
        lambda *a,**k:real(*a,**k,transport=httpx.MockTransport(handler)))
    provider=SourceOCRReviewProvider(Settings(api_key="PRIVATE",base_url="https://example.com/v1",api_timeout=5))
    _,audit=review_source_ocr(image,doc,{},tmp_path/"out",provider=provider,use_api=True,raw_ocr_path=raw)
    assert len(captured)==2
    for kind in ('tight_target_glyphs','context'):
        assert [c["rotation_degrees"] for c in audit["pages"][0]["provider"]["crops"] if c['source_view']==kind]==[0]
        assert [c["rotation_degrees"] for c in audit["pages"][1]["provider"]["crops"] if c['source_view']==kind]==[90,-90]
    urls=[row["image_url"]["url"] for row in captured[1]["messages"][1]["content"] if row["type"]=="image_url"]
    crops=[Image.open(BytesIO(base64.b64decode(url.split(",",1)[1]))) for url in urls]
    assert len(crops)==4 and all(max(crop.size)<=1024 for crop in crops)
    for first,second in ((crops[0],crops[2]),(crops[1],crops[3])):
        assert ImageChops.difference(first.rotate(180),second).getbbox() is None
        assert ImageChops.difference(first,second).getbbox() is not None
    assert "66.85" not in str(captured)
