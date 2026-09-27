import json

from fastapi.testclient import TestClient

from contour_agent.config import Settings
from contour_agent.server import create_app


def test_iteration_and_feedback_downloads_match_published_artifacts(tmp_path):
    output=tmp_path/"jobs"/"reconstruction"/"automatic-001"
    output.mkdir(parents=True)
    names=("topology-edit-proposals.json","topology-iterations.json","reconstruction-feedback.json")
    for name in names:(output/name).write_text(json.dumps({"ground_truth_used":False,"file":name}),encoding="utf8")
    (output/"private.json").write_text('{"not_public":true}',encoding="utf8")
    with TestClient(create_app(Settings(runtime_root=tmp_path,api_key=""))) as client:
        client.app.state.service.store.save({"id":"reconstruction","status":"completed","updated_at":"2026-09-22",
                                             "artifact_directory":str(output)})
        for name in names:
            response=client.get(f"/api/jobs/reconstruction/artifacts/{name}")
            assert response.status_code==200 and response.json()["file"]==name
        assert client.get("/api/jobs/reconstruction/artifacts/private.json").status_code==404
