from copy import deepcopy
import json
import pytest
from test_source_ocr_review import source,Fake
from contour_agent.source_ocr_review import review_source_ocr,load_effective_ocr
from contour_agent.source_ocr_retry import retry_withheld_source_ocr


def disputed(tmp_path):
    image,raw,doc=source(tmp_path,['+18','65'])
    out=tmp_path/'out'
    effective,_=review_source_ocr(image,doc,{},out,provider=Fake([
        {'r000':'18±1','r001':'65'},{'r000':'181±1','r001':'65'}]),use_api=True,raw_ocr_path=raw)
    return image,raw,effective,out


def test_single_record_retry_preserves_history_and_unrelated_annotations(tmp_path):
    image,raw,doc,out=disputed(tmp_path);before=deepcopy(doc)
    old=(out/'source-ocr-review.json').read_bytes()
    provider=Fake([{'r000':'181±1'}]*2)
    result,receipt=retry_withheld_source_ocr(image,doc,out,provider=provider,use_api=True)
    assert doc==before and (out/'source-ocr-review.json').read_bytes()==old
    assert result['records'][0]['text']=='181±1'
    assert 'source_text_admission' not in result['records'][0]
    assert result['records'][1]==doc['records'][1]
    assert [len(c) for c in provider.calls]==[1,1]
    assert receipt['changed_record_ids']==['r000']
    assert load_effective_ocr(image,raw,out)==result


@pytest.mark.parametrize('texts',[['18±1','181±1'],[None,None],['181±1',None]])
def test_retry_disagreement_or_unreadable_keeps_withholding(tmp_path,texts):
    image,raw,doc,out=disputed(tmp_path)
    result,_=retry_withheld_source_ocr(image,doc,out,provider=Fake([{'r000':v} for v in texts]),use_api=True)
    assert result==doc and load_effective_ocr(image,raw,out)==doc


def test_retry_does_not_request_uncontested_or_offline_records(tmp_path):
    image,raw,doc,out=disputed(tmp_path);provider=Fake([{},{}])
    result,_=retry_withheld_source_ocr(image,doc,out,provider=provider,use_api=False)
    assert result==doc and not provider.calls


@pytest.mark.parametrize('tamper',['document','parent','consensus','foreign_record'])
def test_restart_rejects_tampered_retry_chain(tmp_path,tamper):
    image,raw,doc,out=disputed(tmp_path)
    retry_withheld_source_ocr(image,doc,out,provider=Fake([{'r000':'181±1'}]*2),use_api=True)
    file=out/('effective-source-ocr-retry.json' if tamper=='document' else 'source-ocr-retry.json')
    value=json.loads(file.read_text('utf8'))
    if tamper=='document':value['records'][0]['text']='999'
    elif tamper=='parent':value['parent_document_sha256']='wrong'
    elif tamper=='consensus':value['attempts'][0]['views'][0]['transcriptions'][0]['text']='18±1'
    elif tamper=='foreign_record':value['attempts'][0]['record_id']='r001'
    file.write_text(json.dumps(value),'utf8')
    with pytest.raises(ValueError):load_effective_ocr(image,raw,out)


def test_isolated_provider_sends_upright_vertical_pixels_in_both_distinct_views(tmp_path,monkeypatch):
    import httpx
    from contour_agent.config import Settings
    from contour_agent.source_ocr_review import SourceOCRReviewProvider
    image,raw,doc,out=disputed(tmp_path);requests=[]
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(
            {'transcriptions':[{'id':'r000','text':'181±1'}]})}}]})
    real=httpx.AsyncClient
    monkeypatch.setattr('contour_agent.source_ocr_review.httpx.AsyncClient',
        lambda *a,**k:real(*a,**k,transport=httpx.MockTransport(handler)))
    provider=SourceOCRReviewProvider(Settings(api_key='PRIVATE',base_url='https://example.com/v1',api_timeout=5))
    _,audit=retry_withheld_source_ocr(image,doc,out,provider=provider,use_api=True)
    assert len(requests)==2
    views=audit['attempts'][0]['views']
    for kind in ('tight_target_glyphs','context'):
        assert [[c['rotation_degrees'] for c in v['crops'] if c['source_view']==kind]
                for v in views]==[[90,-90],[90,-90]]
    context=[[c for c in v['crops'] if c['source_view']=='context'] for v in views]
    assert context[0][0]['source_crop_box']!=context[1][0]['source_crop_box']
    assert all('18' not in item['text'] for request in requests
               for item in request['messages'][-1]['content'] if item['type']=='text')
