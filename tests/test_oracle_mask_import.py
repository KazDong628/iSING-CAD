"""Uploaded DXF must become pixels without leaking reference coordinates."""
import io
import json

import ezdxf
import numpy as np
from PIL import Image, ImageDraw

from contour_agent.oracle_mask_import import generate_oracle_mask


def test_uploaded_dxf_is_registered_and_rasterized(tmp_path):
    image_path = tmp_path / "drawing.png"
    image = Image.new("RGB", (160, 120), "white")
    drawer = ImageDraw.Draw(image)
    outline = [(28, 24), (132, 24), (132, 90), (88, 90), (88, 105), (28, 105)]
    drawer.line(outline + [outline[0]], fill="black", width=2)
    image.save(image_path)
    hint = Image.new("L", image.size, 0)
    ImageDraw.Draw(hint).polygon(outline, fill=255)
    hint_path = tmp_path / "hint.png"
    hint.save(hint_path)
    ocr_path = tmp_path / "ocr.json"
    ocr_path.write_text(json.dumps({"meta": {"original_size": {"width": 160, "height": 120}},
                                    "records": []}), encoding="utf8")
    document = ezdxf.new("R2010")
    document.units = 4
    document.modelspace().add_lwpolyline([(x - 28, y - 24) for x, y in outline], close=True)
    dxf_path = tmp_path / "reference.dxf"
    document.saveas(dxf_path)

    pixels, receipt = generate_oracle_mask(dxf_path.read_bytes(), image_path, ocr_path,
                                           hint_path, image.size, temporary_root=tmp_path)
    with Image.open(io.BytesIO(pixels)) as generated:
        actual = np.asarray(generated)
    expected = np.asarray(hint)
    intersection = np.count_nonzero((actual > 0) & (expected > 0))
    union = np.count_nonzero((actual > 0) | (expected > 0))
    assert intersection / union > .9
    assert receipt["mask_size"] == {"width": 160, "height": 120}
    assert receipt["ground_truth_dxf_coordinates_sent_to_provider"] is False
    assert not list(tmp_path.glob("gt-dxf-import-*"))
    assert not any("polyline" in key or "transform" in key for key in receipt)


def test_uploaded_dxf_rejects_open_geometry(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (32, 32), "white").save(image)
    hint = tmp_path / "hint.png"
    Image.new("L", (32, 32), 0).save(hint)
    ocr = tmp_path / "ocr.json"
    ocr.write_text('{"meta":{"original_size":{"width":32,"height":32}},"records":[]}', encoding="utf8")
    document = ezdxf.new("R2010")
    document.modelspace().add_line((0, 0), (10, 10))
    dxf = tmp_path / "open.dxf"
    document.saveas(dxf)
    import pytest
    with pytest.raises(ValueError, match="唯一、无孔、未修补"):
        generate_oracle_mask(dxf.read_bytes(), image, ocr, hint, (32, 32), temporary_root=tmp_path)
