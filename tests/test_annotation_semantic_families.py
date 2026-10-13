from copy import deepcopy
import json

import httpx
from PIL import Image
import pytest

from contour_agent.annotation_semantics import (inclination_ratio, source_annotation_inventory)
from contour_agent.config import Settings
from contour_agent.dataset import read_ocr
from contour_agent.ocr import parse_dimension
from contour_agent.source_ocr_review import (SourceOCRReviewProvider, review_source_ocr,
    load_effective_ocr, _crop_payload, _semantic)


def source(tmp_path,texts):
    image=tmp_path/'image.png';Image.new('RGB',(800,700),'white').save(image)
    raw=tmp_path/'ocr.json'
    raw.write_text(json.dumps({'records':[{'text':t,'box':[[80,70+10*i],
        [240,70+10*i],[240,110+10*i],[80,110+10*i]]} for i,t in enumerate(texts)]}),'utf8')
    return image,raw,read_ocr(raw)


class Reader:
    def __init__(self,values):self.values=values;self.calls=[]
    def transcribe(self,image,rows,*,view,timeout):
        self.calls.append(deepcopy(rows))
        assert all(set(r)=={'id','box'} for r in rows)
        return dict(status='succeeded',http_success=True,schema_success=True,network_requests=1,
            transcriptions=[{'id':r['id'],'text':self.values.get(r['id'])} for r in rows])


@pytest.mark.parametrize('text,n,d',[('1 IN 20',1,20),('INCLINE 1 IN 2.5',1,2.5),
    ('slope 2 in 35',2,35)])
def test_ratios_preserve_literal_data_without_inventing_an_axis(text,n,d):
    r=inclination_ratio(text)
    assert (r['numerator'],r['denominator'])==(n,d)
    assert r['reference_axis'] is None and r['direction'] is None
    assert r['numeric_constraint_applied'] is False
    assert parse_dimension(text)['kind']=='unknown'


@pytest.mark.parametrize('text',['1 IN 0','0 IN 20','INCLINE IN25','SECRET 1 IN 20',
    '1 IN 20 extra','1/20','1 IN inf'])
def test_invalid_or_incomplete_ratios_are_never_repaired(text):
    assert inclination_ratio(text) is None


def test_holes_and_manufacturing_stages_are_visible_separate_feature_families(tmp_path):
    _,_,doc=source(tmp_path,['2-⌀70 HOLES','WHEN ROUGH TURNED ⌀140^{+0}_{-2}',
        '1 IN 20','-R15','⌀148 FORCE FIT','RaR1','R45°'])
    audit=source_annotation_inventory(doc)
    assert audit['non_exterior_feature_record_ids']==['r000','r001']
    assert audit['unresolved_profile_record_ids']==['r002','r003','r004','r005','r006']
    assert audit['source_coverage_complete'] is False
    assert audit['gt_geometry_used'] is False


def test_radius_and_angle_risks_receive_slots_before_ordinary_lengths(tmp_path):
    texts=[str(10+i) for i in range(17)]+['-R15','R45°','WHEN ROUGH TURNED ⌀140','2-⌀70 HOLES']
    image,raw,doc=source(tmp_path,texts);before=deepcopy(doc)
    reader=Reader({f'r{i:03}':t for i,t in enumerate(texts)}|{'r017':'R15','r018':'45°'})
    effective,audit=review_source_ocr(image,doc,{},tmp_path/'after',provider=reader,use_api=True,raw_ocr_path=raw)
    assert [r['id'] for r in reader.calls[0]][:2]==['r017','r018']
    assert sum(len(c) for c in reader.calls)==32  # 16 bounded records, two views
    assert not any(r['id'] in {'r019','r020'} for call in reader.calls for r in call)
    assert effective['records'][17]['text']=='R15' and effective['records'][18]['text']=='45°'
    assert doc==before and load_effective_ocr(image,raw,tmp_path/'after')==effective
    assert audit['changed_record_ids']==['r017','r018']


def test_ratio_transcription_survives_evidence_reload_and_remains_unbound(tmp_path):
    image,raw,doc=source(tmp_path,['INCLINE IN25','1 IN 20'])
    reader=Reader({'r000':'1 IN 2.5','r001':'1 IN 20'})
    effective,_=review_source_ocr(image,doc,{},tmp_path/'after',provider=reader,use_api=True,raw_ocr_path=raw)
    assert load_effective_ocr(image,raw,tmp_path/'after')==effective
    assert _semantic('1 IN 20')!=_semantic('1 IN 2.5')
    assert source_annotation_inventory(effective)['unresolved_profile_record_ids']==['r000','r001']


def test_tight_crop_is_source_box_anchored_and_context_stays_separate(tmp_path):
    image,_,doc=source(tmp_path,['unknown'])
    row={'id':'r000','box':doc['records'][0]['box']}
    with Image.open(image) as source_image:
        _,tight=_crop_payload(source_image,row,0,focus=True)
        _,context=_crop_payload(source_image,row,0)
    a,b=tight['source_crop_box'],context['source_crop_box']
    assert b[0]<=a[0]<a[2]<=b[2] and b[1]<=a[1]<a[3]<=b[3]
    assert tight['source_view']=='tight_target_glyphs' and context['source_view']=='context'
    assert tight['source_pixels_annotated'] is False


def test_wire_sends_tight_and_context_pixels_but_no_input_values(tmp_path,monkeypatch):
    image,_,doc=source(tmp_path,['1135']);captured=[]
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{
            'content':json.dumps({'transcriptions':[{'id':'r000','text':'135±2'}]})}}]})
    real=httpx.AsyncClient
    monkeypatch.setattr('contour_agent.source_ocr_review.httpx.AsyncClient',
        lambda *a,**k:real(*a,**k,transport=httpx.MockTransport(handler)))
    provider=SourceOCRReviewProvider(Settings(api_key='PRIVATE',base_url='https://example.com/v1',api_timeout=5))
    receipt=provider.transcribe(image,[{'id':'r000','box':doc['records'][0]['box']}],view=0,timeout=5)
    content=captured[0]['messages'][1]['content']
    assert len([r for r in content if r['type']=='image_url'])==2
    assert 'TIGHT TARGET GLYPHS' in str(content) and 'CONTEXT ONLY' in str(content)
    assert '1135' not in str(captured) and 'PRIVATE' not in str(receipt)
    assert [r['source_view'] for r in receipt['crops']]==['tight_target_glyphs','context']


def test_thin_slanted_text_box_expands_short_axis_without_expected_values(tmp_path):
    image,_,_=source(tmp_path,['unused'])
    row={'id':'r000','box':[[100,200],[455,200],[455,209],[100,209]]}
    with Image.open(image) as original:
        _,crop=_crop_payload(original,row,0,focus=True)
    box=crop['source_crop_box']
    assert box[3]-box[1]>150 and crop['thin_box_focus_expanded'] is True
    assert crop['source_pixels_annotated'] is False
    assert max(crop['input_image_size'])<=1024


def test_unresolved_semantic_families_block_complete_reconstruction():
    from contour_agent.reconstruction_contract import reconstruction_contract
    audit=reconstruction_contract([],{'binding_counts':{'unresolved_profile_annotation_records':2}}, {})
    assert audit['satisfied'] is False
    assert 'source_annotation_semantic_families_unresolved' in audit['reasons']
