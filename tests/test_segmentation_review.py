import hashlib
import io
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from contour_agent.segmentation_review import create_segmentation_review_router


def png(size=(32, 24), value=255, mode="L"):
    stream = io.BytesIO()
    Image.new(mode, size, value).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def review_client(tmp_path):
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "masks").mkdir()
    (data / "images/044-main.png").write_bytes(png(value=0))
    (data / "masks/044-main.png").write_bytes(png(value=0))
    rows = [{"id": "044-main", "image": "images/044-main.png", "mask": "masks/044-main.png",
             "split": "train", "source_image_sha256": "a" * 64, "label_source": "source_heuristic"}]
    manifest = data / "manifest.json"
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf8")
    app = FastAPI()
    app.include_router(create_segmentation_review_router(manifest))
    with TestClient(app) as client:
        yield client, data, manifest, rows


def save(client, payload=None, case_id="044-main", **kwargs):
    return client.put(f"/api/segmentation-review/cases/{case_id}/review", content=payload if payload is not None else png(),
                      headers={"Content-Type": "image/png", **kwargs})


def test_loading_labels_does_not_review_or_change_manifest(review_client):
    client, data, manifest, _ = review_client
    before = manifest.read_bytes()
    assert client.get("/segmentation-review").status_code == 200
    listing = client.get("/api/segmentation-review/cases").json()
    assert listing["cases"][0]["reviewed"] is False
    assert listing["cases"][0]["width"] == 32
    assert "image" not in listing["cases"][0]
    for kind in ("image", "mask"):
        response = client.get(f"/api/segmentation-review/cases/044-main/{kind}")
        assert response.status_code == 200 and response.headers["content-type"] == "image/png"
    assert manifest.read_bytes() == before
    assert not (data / "reviews.json").exists()
    assert not (data / "reviewed_masks").exists()


def test_explicit_save_records_review_preserves_split_and_weak_mask(review_client):
    client, data, manifest, _ = review_client
    before, initial = manifest.read_bytes(), (data / "masks/044-main.png").read_bytes()
    response = save(client)
    assert response.status_code == 200
    record = json.loads((data / "reviews.json").read_text())["044-main"]
    assert record["reviewed"] is True and record["label_source"] == "local_user_review"
    assert record["source_image_sha256"] == "a" * 64
    assert record["mask"] == "reviewed_masks/044-main.png"
    assert record["timestamp"] and response.json()["split"] == "train"
    assert manifest.read_bytes() == before
    assert (data / "masks/044-main.png").read_bytes() == initial
    reviewed = client.get("/api/segmentation-review/cases/044-main/mask")
    assert hashlib.sha256(reviewed.content).hexdigest() == record["mask_sha256"]
    assert client.get("/api/segmentation-review/cases").json()["cases"][0]["reviewed"] is True


def test_multiple_review_records_are_preserved_on_resave(review_client):
    client, data, _, _ = review_client
    (data / "reviews.json").write_text(json.dumps({"other": {"reviewed": False}}))
    assert save(client).status_code == 200
    assert save(client, png(value=0)).status_code == 200
    assert json.loads((data / "reviews.json").read_text())["other"] == {"reviewed": False}
    with Image.open(data / "reviewed_masks/044-main.png") as mask:
        assert mask.mode == "L" and mask.getextrema() == (0, 0)


@pytest.mark.parametrize("payload", [b"not a PNG", png(size=(31, 24)), png(value=127), png(mode="RGBA", value=(255, 255, 255, 0))])
def test_invalid_save_is_rejected_without_review_record(review_client, payload):
    client, data, _, _ = review_client
    assert save(client, payload).status_code == 422
    assert not (data / "reviews.json").exists()


def test_request_size_content_type_and_origin_are_bounded(review_client):
    client, data, _, _ = review_client
    assert save(client, b"x" * 12_000_001).status_code == 413
    assert save(client, **{"Content-Type": "text/plain"}).status_code == 415
    assert save(client, Origin="https://external.invalid").status_code == 403
    assert not (data / "reviews.json").exists()


def test_registered_image_dimension_limit_is_enforced(review_client):
    client, data, _, _ = review_client
    (data / "images/044-main.png").write_bytes(png(size=(1537, 1)))
    assert save(client, png(size=(1537, 1))).status_code == 422
    assert not client.get("/api/segmentation-review/cases").json()["cases"][0]["available"]


def test_initial_mask_size_mismatch_is_visible_before_selection(review_client):
    client, data, _, _ = review_client
    (data / "masks/044-main.png").write_bytes(png(size=(31, 24)))
    case = client.get("/api/segmentation-review/cases").json()["cases"][0]
    assert not case["available"] and case["issue"]
    assert client.get("/api/segmentation-review/cases/044-main/mask").status_code == 422


@pytest.mark.parametrize("absolute", [False, True])
def test_manifest_cannot_register_paths_outside_data_root(review_client, absolute):
    client, data, manifest, rows = review_client
    outside = data.parent / "outside.png"
    outside.write_bytes(png())
    rows[0]["image"] = str(outside) if absolute else "../outside.png"
    manifest.write_text(json.dumps({"rows": rows}))
    assert client.get("/api/segmentation-review/cases/044-main/image").status_code == 404
    assert save(client).status_code == 404


def test_absolute_registered_paths_inside_data_are_supported(review_client):
    client, data, manifest, rows = review_client
    rows[0]["image"] = str(data / "images/044-main.png")
    rows[0]["sourcehash"] = rows[0].pop("source_image_sha256")
    manifest.write_text(json.dumps({"samples": rows}))
    assert client.get("/api/segmentation-review/cases/044-main/image").status_code == 200
    assert save(client).status_code == 200


def test_unregistered_ids_and_file_paths_are_not_served(review_client):
    client, data, _, _ = review_client
    (data / "private.png").write_bytes(png())
    for case_id in ("unknown", "private.png", "..%5Cprivate.png"):
        assert client.get(f"/api/segmentation-review/cases/{case_id}/image").status_code == 404
        assert save(client, case_id=case_id).status_code == 404


def test_malicious_review_path_and_changed_mask_are_rejected(review_client):
    client, data, _, _ = review_client
    assert save(client).status_code == 200
    review_path = data / "reviews.json"
    records = json.loads(review_path.read_text())
    records["044-main"]["mask"] = "images/044-main.png"
    review_path.write_text(json.dumps(records))
    assert client.get("/api/segmentation-review/cases/044-main/mask").status_code == 404
    records["044-main"]["mask"] = "reviewed_masks/044-main.png"
    review_path.write_text(json.dumps(records))
    (data / "reviewed_masks/044-main.png").write_bytes(png(value=0))
    assert client.get("/api/segmentation-review/cases/044-main/mask").status_code == 409


def test_save_rejects_reparse_path_escape_without_symlink_privileges(review_client, monkeypatch):
    client, data, _, _ = review_client
    original = Path.resolve

    def redirect(path, *args, **kwargs):
        if path == data / "reviewed_masks":
            return data.parent / "outside"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", redirect)
    assert save(client).status_code == 404
    assert not (data / "reviews.json").exists()


def test_corrupt_review_log_is_never_overwritten(review_client):
    client, data, _, _ = review_client
    path = data / "reviews.json"
    path.write_text("invalid retained bytes")
    assert save(client).status_code == 503
    assert path.read_text() == "invalid retained bytes"
