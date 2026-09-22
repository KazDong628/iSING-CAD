import threading
import pytest
from contour_agent.config import Settings
from contour_agent.service import AgentService


@pytest.fixture
def service(tmp_path,monkeypatch):
    s=AgentService(Settings(runtime_root=tmp_path,api_key=""))
    def build(image,document,directory,progress):
        directory.mkdir(parents=True,exist_ok=True)
        for name in ('drawing.dxf','preview.svg','model.json','validation.json','overlay.png','dimension-evidence.json'):
            (directory/name).write_text('test artifact',encoding='utf8')
        progress('export','test export complete')
        return {'scale':{'status':'resolved','pixels_per_mm':2},'extraction':{'evidence':{}},
                'validation':{'passed':True,'scaled_mm':True,'dimensions_verified':False},'entities':[],
                'bounds':{},'coordinate_system':{'units':'mm'},'automatic_completion':True,
                'issues':[],'polyline_px':[[0,0],[1,0],[1,1],[0,0]]}
    monkeypatch.setattr('contour_agent.service.build_automatic',build)
    yield s
    s.close()


def test_unregistered_case_generates_without_manual_confirmation(service):
    result=service.create_auto_case('044-main',use_api=False,asynchronous=False)
    assert result['status']=='completed'
    assert result['automatic_completion']
    assert result['template_id'] is None
    assert result['manual_confirmation'] is None and not result['manual_intervention']
    assert result['artifacts']['dxf'] and result['artifacts']['overlay']
    assert not result['engineering_accepted'] and not result['validation']['dimensions_verified']


def test_job_persists_selected_provider_without_exposing_a_key(service):
    result=service.create_auto_case('044-main',use_api=False,asynchronous=False,
                                    provider_id='h800-qwen3.8-27b')
    assert result['provider_id']=='h800-qwen3.8-27b'
    assert result['provider_profile']['model']=='Qwen3.8-27B-FP8'
    assert 'api_key' not in result['provider_profile']
    with pytest.raises(ValueError,match='未知在线模型配置'):
        service.create_auto_case('044-main',use_api=False,provider_id='invented-provider')


def test_api_failure_preserves_automatic_artifacts_without_manual_gate(service,monkeypatch):
    monkeypatch.setattr(service.vision_provider,'inspect',lambda *a,**k:{'status':'failed','error_code':'timeout','network_requests':1,'http_success':False})
    result=service.create_auto_case('044-main',use_api=True,asynchronous=False)
    assert result['status']=='completed' and result['artifacts'] and result['automatic_completion']
    assert result['provider']['http_success'] is False
    assert result['parameters']==[] and result['assumptions']==[]


def test_drawing_available_while_visual_audit_is_pending(service,monkeypatch):
    entered,release=threading.Event(),threading.Event()
    def pending(*args,**kwargs):
        entered.set()
        assert release.wait(5)
        return {'status':'failed','error_code':'timeout'}
    monkeypatch.setattr(service.vision_provider,'inspect',pending)
    job=service.create_auto_case('044-main',use_api=True)
    try:
        assert entered.wait(5)
        current=service.store.get(job['id'])
        assert current['status']=='auditing' and current['artifacts']['dxf']
        service.cancel(job['id'])
    finally:
        release.set()
    service.executor.shutdown(wait=True)
    assert service.store.get(job['id'])['status']=='cancelled'


def test_dimension_api_failure_keeps_drawing_and_still_runs_visual_review(service,monkeypatch):
    from contour_agent.provider import ProviderError
    def timeout(rows):
        raise ProviderError('timeout','sensitive-provider-error',network_requests=1)
    monkeypatch.setattr(service.automatic_dimension_provider,'normalize',timeout)
    monkeypatch.setattr(service.vision_provider,'inspect',lambda *a,**k:{'status':'succeeded','verdict':'match','network_requests':1})
    result=service.create_auto_case('CL60-main',use_api=True,asynchronous=False)
    assert result['status']=='completed' and result['artifacts']['dxf']
    assert result['dimension_analysis']['provider']['error_code']=='timeout'
    assert result['provider']['verdict']=='match'
    assert 'sensitive-provider-error' not in str(result)
    assert not result['validation']['dimensions_verified']


def test_failed_cad_preserves_generated_segmentation_artifacts(service,monkeypatch):
    def failed_build(image,document,directory,progress):
        evidence=directory/'learned-evidence'
        evidence.mkdir(parents=True)
        for name in ('prediction-mask.png','prediction-overlay.png','segmentation.json'):
            (evidence/name).write_bytes(b'saved-segmentation-evidence')
        progress('measure','segmentation finished')
        current=service.store.list(1)[0]
        assert current['artifacts']['segmentation_mask']
        raise ValueError('CAD fitting failed')
    monkeypatch.setattr('contour_agent.service.build_automatic',failed_build)
    result=service.create_auto_case('CL60-main',use_api=False,asynchronous=False)
    assert result['status']=='failed' and not result['automatic_completion']
    assert result['artifacts']['segmentation_mask'] and result['artifacts']['segmentation_overlay']
    assert 'dxf' not in result['artifacts']


def test_exception_after_file_export_still_publishes_candidate_dxf(service,monkeypatch):
    def failed_build(image,document,directory,progress):
        directory.mkdir(parents=True)
        for name in ('drawing.dxf','preview.svg','model.json','validation.json','overlay.png','dimension-evidence.json'):
            (directory/name).write_bytes(b'candidate artifact')
        raise ValueError('Post-export validation failed')
    monkeypatch.setattr('contour_agent.service.build_automatic',failed_build)
    result=service.create_auto_case('CL60-main',use_api=False,asynchronous=False)
    assert result['status']=='failed' and not result['automatic_completion']
    assert all(result['artifacts'].get(k) for k in ('dxf','svg','model','validation','overlay','dimensions'))
