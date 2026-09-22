import json

from fastapi.testclient import TestClient

from contour_agent.config import Settings
from contour_agent.server import create_app


def test_comparison_serves_report_and_downloads_without_exposing_private_files(tmp_path):
    root=tmp_path/'runtime'
    reports=root/'dxf-comparison'
    run='20260921T010000123456Z-abc12345'
    target=reports/run
    target.mkdir(parents=True)
    (target/'index.html').write_text('<p>DXF comparison</p>',encoding='utf8')
    (target/'comparison.json').write_text('{"development_only":true}',encoding='utf8')
    (target/'comparison.csv').write_text('case,entities\na,10\n',encoding='utf8')
    (target/'private.txt').write_text('private',encoding='utf8')
    pointer=reports/'latest.json'
    pointer.write_text(json.dumps({'run_id':run}),encoding='utf8')
    with TestClient(create_app(Settings(runtime_root=root,api_key=''))) as client:
        response=client.get('/dxf-comparison')
        assert response.status_code==200 and 'DXF comparison' in response.text
        response=client.get(f'/api/dxf-comparison/{run}/comparison.json')
        assert response.json()['development_only'] is True
        assert response.headers['content-type'].startswith('application/json')
        assert 'attachment' in response.headers['content-disposition']
        assert client.get(f'/api/dxf-comparison/{run}/comparison.csv').status_code==200
        assert client.get(f'/api/dxf-comparison/{run}/private.txt').status_code==404
        assert client.post('/dxf-comparison').status_code==405
        for invalid in ('../outside',None,4):
            pointer.write_text(json.dumps({'run_id':invalid}),encoding='utf8')
            assert client.get('/dxf-comparison').status_code==404


def test_missing_comparison_never_uses_an_unrelated_report(tmp_path):
    with TestClient(create_app(Settings(runtime_root=tmp_path,api_key=''))) as client:
        assert client.get('/dxf-comparison').status_code==404
