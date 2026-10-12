"""All page glyphs remain readable without manufacturing annotation pixels."""
import base64
import copy
import hashlib
from io import BytesIO
import json

import httpx
from PIL import Image

from contour_agent.binding_provider import BindingProvider, _source_glyph_panel
from contour_agent.config import Settings


def test_atlas_covers_every_record_and_does_not_render_supplied_nominals(tmp_path):
    source=tmp_path/'source.png'
    image=Image.new('RGB',(400,400),'white')
    records=[]
    for i in range(16):
        x=20+(i%4)*90;y=20+(i//4)*90
        image.paste((20+i,40+i,60+i),(x,y,x+30,y+24))
        records.append({'id':f'r{i:03d}','text':'R15','parsed':{'kind':'radius','nominal':15},
                        'box':[[x,y],[x+30,y],[x+30,y+24],[x,y+24]]})
    image.save(source)
    packets,tiles=_source_glyph_panel(source,{'records':records})
    changed=copy.deepcopy(records)
    for row in changed:row.update(text='R999999',parsed={'kind':'radius','nominal':999999})
    assert packets==_source_glyph_panel(source,{'records':changed})[0]
    assert len(packets)==1 and [t['record_id'] for t in tiles]==[r['id'] for r in records]
    raw=base64.b64decode(packets[0]['image_url']['url'].split(',',1)[1])
    atlas=Image.open(BytesIO(raw))
    for i,tile in enumerate(tiles):
        left,top,right,bottom=tile['atlas_box_px']
        assert atlas.getpixel(((left+right)//2,(top+bottom)//2))==(20+i,40+i,60+i)
        assert tile['input_image_sha256']==hashlib.sha256(raw).hexdigest()
        assert tile['nominal_drawn_in_caption'] is False


def test_empty_and_invalid_glyph_boxes_do_not_create_invented_tiles(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(20,20),'white').save(source)
    records=[{'id':'r000','box':[]},{'id':'r001','box':[[100,100],[120,120]]},
             {'id':'r002','box':[[float('nan'),0],[1,2]]}]
    assert _source_glyph_panel(source,{'records':records})==([],[])


def test_binding_transmits_original_glyph_atlas_and_accounts_for_image(tmp_path,monkeypatch):
    source=tmp_path/'source.png';Image.new('RGB',(160,90),'white').save(source)
    requests=[]
    def handler(request):
        wire=json.loads(request.content);requests.append(wire)
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{
            'content':'{"bindings":[],"relations":[]}'}}]})
    original=httpx.AsyncClient
    monkeypatch.setattr('contour_agent.binding_provider.httpx.AsyncClient',
        lambda *a,**kw:original(*a,**kw,transport=httpx.MockTransport(handler)))
    inventory={'units':'mm','records':[{'id':'r000','text':'R15',
        'parsed':{'kind':'radius','nominal':15},'box':[[10,10],[40,40]]}],
        'candidates':[{'id':'c000','record_id':'r000','kind':'radius','entities':['g000'],
                       'nodes':[],'value':15,'evidence':{}}],'relations':[]}
    receipt=BindingProvider(Settings(api_key='test')).select(source,source,inventory)
    images=[p for p in requests[0]['messages'][1]['content'] if p['type']=='image_url']
    assert receipt['input_image_count']==len(images)==4
    assert receipt['network_requests']==1 and receipt['source_glyph_tiles'][0]['record_id']=='r000'
    assert receipt['source_glyph_tiles'][0]['page_index']==1
    assert images[-1]['image_url']['url'].startswith('data:image/png;base64,')
    assert receipt['ground_truth_sent'] is False
