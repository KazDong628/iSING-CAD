import json

from fastapi.testclient import TestClient

from contour_agent.config import Settings
from contour_agent.server import create_app


def test_pilot_reports_only_serve_declared_files_under_runtime(tmp_path):
    root=tmp_path/'runtime'
    folder=root/'segmentation/pilot'
    run='20260920T180000123456Z-1234abcd'
    run_dir=folder/run
    run_dir.mkdir(parents=True)
    (run_dir/'index.html').write_text('<p>Selected four-case development pilot</p>',encoding='utf8')
    (run_dir/'pilot_summary.json').write_text('{"selected_count":4,"catalog_count":50}',encoding='utf8')
    (run_dir/'private-file.txt').write_text('not a public artifact',encoding='utf8')
    (folder/'latest.json').write_text(json.dumps({'run_id':run}),encoding='utf8')
    with TestClient(create_app(Settings(runtime_root=root,api_key=''))) as client:
        response=client.get('/segmentation-pilot')
        assert response.status_code==200 and 'Selected four-case' in response.text
        assert client.get(f'/api/segmentation/pilot/{run}/pilot_summary.json').json()['catalog_count']==50
        assert client.get(f'/api/segmentation/pilot/{run}/private-file.txt').status_code==404
        (folder/'latest.json').write_text(json.dumps({'run_id':'../outside'}),encoding='utf8')
        assert client.get('/segmentation-pilot').status_code==404


def test_missing_pilot_does_not_publish_old_or_unrelated_results(tmp_path):
    with TestClient(create_app(Settings(runtime_root=tmp_path,api_key=''))) as client:
        assert client.get('/segmentation-pilot').status_code==404
