import json

import pytest
from PIL import Image

from contour_agent.config import Settings
from contour_agent.service import AgentService


@pytest.mark.parametrize('interrupted_stage',['solving','auditing'])
def test_restart_recovers_saved_valid_export_before_journal_update(tmp_path,monkeypatch,interrupted_stage):
    image=tmp_path/'source.png'
    Image.new('RGB',(128,96),'white').save(image)
    ocr=tmp_path/'source.json'
    ocr.write_text(json.dumps({'meta':{'original_size':{'width':128,'height':96}},'records':[]}))
    monkeypatch.setattr('contour_agent.automatic.extract_main_profile',lambda *a,**k:{
        'status':'needs_review','polyline_px':[[20,20],[100,20],[100,75],[20,75],[20,20]],
        'image_size':{'width':128,'height':96},'issues':[],'evidence':{}})
    monkeypatch.setattr('contour_agent.automatic.estimate_scale',lambda *a,**k:{'status':'unresolved','pixels_per_mm':None,'issues':[]})
    settings=Settings(runtime_root=tmp_path/'runtime',api_key='')
    first=AgentService(settings)
    try:
        job=first.create_auto_source(image,ocr,use_api=False,asynchronous=False)
        assert job['automatic_completion']
        job.update(status=interrupted_stage,artifacts={},automatic_completion=False,validation=None,
                   provider={'status':'pending','network_requests':0})
        job['dimension_analysis']['provider'].update(status='pending',network_requests=0)
        first.store.save(job)
    finally:first.close()
    monkeypatch.setattr('contour_agent.provider.DimensionProvider.normalize',lambda *a,**k:pytest.fail('Recovery retried API'))
    second=AgentService(settings,recover_running=True)
    try:
        recovered=second.store.get(job['id'])
        assert recovered['status']=='completed' and recovered['automatic_completion']
        assert recovered['artifacts']['dxf'] and recovered['artifacts']['overlay']
        assert not recovered['validation']['dimensions_verified']
        assert recovered['dimension_analysis']['provider']['network_requests'] is None
        from pathlib import Path
        saved=json.loads((Path(recovered['artifact_directory'])/'dimension-analysis.json').read_text('utf8'))
        assert saved['provider']['status']=='interrupted'
        if interrupted_stage=='auditing':
            assert recovered['provider']['status']=='interrupted'
            assert recovered['provider']['network_requests'] is None
    finally:second.close()
