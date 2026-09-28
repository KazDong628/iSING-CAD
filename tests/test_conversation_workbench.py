import hashlib
import shutil
import subprocess
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from contour_agent.config import ROOT, Settings
from contour_agent.conversation_store import ConversationStore
from contour_agent.server import create_app


def test_conversation_store_persists_messages_and_job_memory(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    conversation = store.create("轮廓修订")
    store.add_message(conversation["id"], "user", "右上角区域不正确", attachments=[{"kind": "feedback_screenshot", "name": "crop.png"}])
    store.remember_job(conversation["id"], "a" * 32)

    reopened = ConversationStore(tmp_path / "conversations").get(conversation["id"])
    assert reopened["messages"][0]["text"] == "右上角区域不正确"
    assert reopened["memory"]["last_job_id"] == "a" * 32
    assert reopened["memory"]["turn_count"] == 1


def test_agent_page_and_conversation_api_are_local_and_persistent(tmp_path):
    app = create_app(Settings(runtime_root=tmp_path, api_key=""))
    with TestClient(app) as client:
        page = client.get("/agent")
        assert page.status_code == 200
        assert "用对话完成主轮廓重建" in page.text
        created = client.post("/api/conversations", json={"title": "测试任务"})
        assert created.status_code == 201
        conversation_id = created.json()["id"]
        turn = client.post(
            f"/api/conversations/{conversation_id}/turns",
            data={"prompt": "先记住这个要求"},
            headers={"Origin": "http://testserver"},
        )
        assert turn.status_code == 202
        body = turn.json()
        assert [row["role"] for row in body["messages"]] == ["user", "assistant"]
        assert "source_image" not in str(body) and "source_ocr" not in str(body)
        assert client.get(f"/api/conversations/{conversation_id}").json()["memory"]["turn_count"] == 1


def test_agent_assets_use_content_versions_and_disable_browser_cache(tmp_path):
    app = create_app(Settings(runtime_root=tmp_path, api_key=""))
    with TestClient(app) as client:
        page = client.get("/agent")
        assert page.status_code == 200
        assert page.headers["cache-control"] == "no-store"
        for asset in ("agent.css", "agent.js"):
            digest = hashlib.sha256((ROOT / "web" / asset).read_bytes()).hexdigest()[:12]
            assert f'/static/{asset}?v={digest}' in page.text
            response = client.get(f"/static/{asset}?v={digest}")
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"


def test_default_conversation_title_tracks_first_turn(tmp_path):
    app = create_app(Settings(runtime_root=tmp_path, api_key=""))
    with TestClient(app) as client:
        conversation_id = client.post("/api/conversations", json={}).json()["id"]
        result = client.post(
            f"/api/conversations/{conversation_id}/turns",
            data={"prompt": "分析当前图纸并记住主轮廓要求"},
        )
        assert result.status_code == 202
        assert result.json()["title"] == "分析当前图纸并记住主轮廓要求"
        summary = client.get("/api/conversations").json()["conversations"][0]
        assert summary["title"] == "分析当前图纸并记住主轮廓要求"
        assert summary["memory"]["turn_count"] == 1


def test_conversation_upload_requires_image_and_ocr_pair(tmp_path):
    app = create_app(Settings(runtime_root=tmp_path, api_key=""))
    with TestClient(app) as client:
        conversation_id = client.post("/api/conversations", json={}).json()["id"]
        result = client.post(
            f"/api/conversations/{conversation_id}/turns",
            data={"prompt": "绘制轮廓"},
            files={"image": ("drawing.png", b"not-an-image", "image/png")},
        )
        assert result.status_code == 400
        assert "OCR" in result.json()["detail"]


def test_running_conversation_must_be_stopped_before_history_is_deleted(tmp_path):
    app = create_app(Settings(runtime_root=tmp_path, api_key=""))
    conversation = app.state.conversations.create("可停止任务")
    job_id = "b" * 32
    now = datetime.now(timezone.utc).isoformat()
    app.state.service.store.save({
        "id": job_id, "status": "solving", "mode": "autonomous_image",
        "conversation_id": conversation["id"], "created_at": now, "updated_at": now,
        "events": [], "artifacts": {}, "provider": {"status": "disabled"},
    })
    app.state.conversations.remember_job(conversation["id"], job_id)
    artifact = tmp_path / "jobs" / job_id / "automatic-001" / "partial.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")

    with TestClient(app) as client:
        blocked = client.delete(f"/api/conversations/{conversation['id']}")
        assert blocked.status_code == 409
        assert artifact.is_file()
        stopped = client.post(f"/api/jobs/{job_id}/cancel")
        assert stopped.status_code == 200
        assert stopped.json()["status"] == "cancelled"
        removed = client.delete(f"/api/conversations/{conversation['id']}")
        assert removed.status_code == 200
        assert removed.json()["deleted_job_count"] == 1
        assert client.get(f"/api/jobs/{job_id}").status_code == 404
        assert client.get(f"/api/conversations/{conversation['id']}").status_code == 404
        assert not (tmp_path / "jobs" / job_id).exists()


def test_delete_conversation_removes_upload_copy_but_never_source_dataset_file(tmp_path):
    app = create_app(Settings(runtime_root=tmp_path / "runtime", api_key=""))
    conversation = app.state.conversations.create("已完成任务")
    job_id = "c" * 32
    now = datetime.now(timezone.utc).isoformat()
    source = tmp_path / "source-dataset.png"
    source.write_bytes(b"source")
    upload = tmp_path / "runtime" / "conversation-uploads" / conversation["id"] / "turn" / "source.png"
    upload.parent.mkdir(parents=True)
    upload.write_bytes(b"copy")
    app.state.service.store.save({
        "id": job_id, "status": "completed", "mode": "autonomous_image",
        "conversation_id": conversation["id"], "created_at": now, "updated_at": now,
        "source_image": str(source), "events": [], "artifacts": {},
        "provider": {"status": "disabled"},
    })
    app.state.conversations.remember_job(conversation["id"], job_id)

    with TestClient(app) as client:
        response = client.delete(f"/api/conversations/{conversation['id']}")
        assert response.status_code == 200
    assert source.is_file()
    assert not (tmp_path / "runtime" / "conversation-uploads" / conversation["id"]).exists()


def test_agent_javascript_is_syntax_valid():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for frontend syntax validation")
    result = subprocess.run([node, "--check", "web/agent.js"], capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    source = open("web/agent.js", encoding="utf-8").read()
    assert 'addEventListener("paste",acceptPastedImage)' in source
    assert 'state.currentJob?"feedback":"sourceImage"' in source
    assert '/segmentation-review' in source
    assert 'globalCompositeOperation=review.tool==="erase"?"destination-out"' in source
    assert 'event.key!=="Tab"' in source
    assert 'zoomAtPointer("preview",event)' in source
    assert 'zoomAtPointer("review",event)' in source
    assert 'busy.has(state.currentJob?.status)' in source
    assert 'classList.toggle("running",running)' in source
    assert '/cancel' in source
    assert 'method:"DELETE"' in source
    assert 'openDeleteDialog(row)' in source
    html = open("web/agent.html", encoding="utf-8").read()
    assert 'id="review-canvas"' in html
    assert 'id="review-confirm"' in html
    assert 'id="provider-select"' in html
    assert 'id="preview-zoom"' in html and 'id="review-zoom"' in html
    assert 'class="send-spinner"' in html
    assert 'id="stop-job"' in html
    assert 'id="delete-dialog"' in html
    assert 'data.append("provider_id",state.selectedProvider)' in source
