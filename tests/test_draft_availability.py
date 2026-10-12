import json

import ezdxf
import numpy as np
from PIL import Image
import pytest

from contour_agent.automatic import build_automatic
from contour_agent.config import Settings
from contour_agent.draft_artifacts import dxf_availability
from contour_agent.mask_geometry import extract_mask_profile
from contour_agent.server import create_app
from contour_agent.service import AgentService
from fastapi.testclient import TestClient


@pytest.fixture
def source(tmp_path, monkeypatch):
    mask = np.zeros((180, 240), np.uint8)
    mask[20:160, 30:210] = 1
    image = tmp_path / "source.png"
    Image.fromarray(255 - mask * 255).save(image)
    extraction = extract_mask_profile(mask, (240, 180))
    monkeypatch.setattr("contour_agent.automatic.extract_main_profile", lambda *a, **k: extraction)
    document = {"meta": {"original_size": {"width": 240, "height": 180}}, "records": []}
    ocr = tmp_path / "source.json"
    ocr.write_text(json.dumps(document), encoding="utf8")
    return image, ocr, document


@pytest.mark.parametrize("stage", ["estimate_scale", "fit_polyline_with_diagnostics"])
def test_initial_failure_preserves_real_readable_unaccepted_source_dxf(tmp_path, monkeypatch, source, stage):
    def fail(*a, **k):
        raise RuntimeError("sensitive-provider-message-must-not-be-saved")
    monkeypatch.setattr("contour_agent.automatic." + stage, fail)
    image, _, document = source
    output = tmp_path / "after"
    result = build_automatic(image, document, output)
    assert result["initialization_recovery"]["source_checkpoint_verified"]
    assert not result["validation"]["passed"] and not result["automatic_completion"]
    assert not result["validation"]["dimensions_verified"]
    assert result["coordinate_system"]["units"] == "pixel"
    available = dxf_availability(output)
    assert available["available"] and available["entity_count"] >= 4
    doc = ezdxf.readfile(output / "drawing.dxf")
    assert doc.units == 0
    points = np.array([list(e.dxf.start)[:2] for e in doc.modelspace()])
    # Real mask boundary dimensions, not a canned replacement or empty file.
    assert np.ptp(points[:, 0]) == pytest.approx(179, abs=1)
    assert np.ptp(points[:, 1]) == pytest.approx(139, abs=1)
    assert "sensitive-provider-message" not in (output / "model.json").read_text(encoding="utf8")


def test_cancellation_is_not_converted_into_fallback(tmp_path, source):
    def stop(stage, message):
        if stage == "source_draft":
            raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        build_automatic(source[0], source[2], tmp_path / "after", progress=stop)
    assert dxf_availability(tmp_path / "after")["available"]


def test_empty_source_cannot_generate_fake_dxf(tmp_path, monkeypatch, source):
    monkeypatch.setattr("contour_agent.automatic.extract_main_profile", lambda *a, **k: {"polyline_px": [], "status": "missing"})
    monkeypatch.setattr("contour_agent.automatic.extract_unhatched", lambda *a, **k: {"polyline_px": [], "status": "missing"})
    with pytest.raises(ValueError):
        build_automatic(source[0], source[2], tmp_path / "after")
    assert not dxf_availability(tmp_path / "after")["available"]


def test_empty_or_corrupt_dxf_does_not_count_as_output(tmp_path):
    ezdxf.new().saveas(tmp_path / "drawing.dxf")
    assert dxf_availability(tmp_path)["reason"] == "empty_dxf"
    (tmp_path / "drawing.dxf").write_bytes(b"not a drawing")
    assert not dxf_availability(tmp_path)["available"]


def test_provider_setup_failure_still_has_downloadable_verified_draft(tmp_path, monkeypatch, source):
    image, ocr, _ = source
    service = AgentService(Settings(runtime_root=tmp_path / "runtime", api_key=""))
    def fail(*args):
        raise RuntimeError("configuration failed")
    monkeypatch.setattr(service, "_provider_bundle", fail)
    try:
        job = service.create_auto_source(image, ocr, use_api=False, asynchronous=False)
        assert job["status"] == "failed"
        assert job["dxf_availability"]["available"]
        assert "dxf" in job["artifacts"] and "source_draft_dxf" in job["artifacts"]
        assert not job["engineering_accepted"]
        # Exercise the real artifact allowlist, including the early checkpoint.
        app = create_app(Settings(runtime_root=tmp_path / "runtime", api_key=""))
        with TestClient(app) as client:
            response = client.get(job["artifacts"]["source_draft_dxf"])
            assert response.status_code == 200
            downloaded = tmp_path / "downloaded.dxf"
            downloaded.write_bytes(response.content)
            assert len(ezdxf.readfile(downloaded).modelspace()) > 0
    finally:
        service.close()
