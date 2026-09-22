"""Synthetic source-image checks; no dataset/GT/calibration template is used."""
import math

import cv2
import ezdxf
import numpy as np
import pytest

from contour_agent.automatic import build_automatic
from contour_agent.dimension_evidence import estimate_scale
from contour_agent.raster import extract_main_profile
from contour_agent.outline_fallback import extract_unhatched
from contour_agent.vectorize import _arc, fit_polyline


def _save(path, image):
    okay, encoded = cv2.imencode(".png", image)
    assert okay
    path.write_bytes(encoded.tobytes())
    return path


def _drawing(shape="rectangle", *, rotated=False, hatched=True, multi=False, dimensions=False):
    height, width = (800, 1100) if dimensions else (650, 900)
    image = np.full((height, width), 255, np.uint8)
    mask = np.zeros_like(image)
    if dimensions:
        cv2.rectangle(mask, (240, 360), (900, 700), 255, -1)
    elif multi:
        cv2.rectangle(mask, (120, 140), (370, 480), 255, -1)
        cv2.rectangle(mask, (550, 140), (800, 480), 255, -1)
    elif shape == "ellipse":
        cv2.ellipse(mask, (450, 325), (230, 160), 0, 0, 360, 255, -1)
    else:
        cv2.rectangle(mask, (200, 150), (700, 500), 255, -1)
    if hatched:
        hatch = np.zeros_like(image)
        for offset in range(-height, width + height, 31):
            cv2.line(hatch, (offset, height-1), (offset+height, 0), 255, 1)
        image[(hatch > 0) & (mask > 0)] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, 0, 3)
    # A large external dimension rectangle is deliberately not hatched.
    if not dimensions:
        cv2.rectangle(image, (30, 35), (860, 610), 0, 1)
        cv2.putText(image, "120 +/- 1", (330, 75), cv2.FONT_HERSHEY_SIMPLEX, .8, 0, 1)
    records = []
    if dimensions:
        for index, nominal in enumerate((200, 400, 600)):
            y = 80 + index * 80
            x_end = 80 + nominal
            cv2.line(image, (80, y), (x_end, y), 0, 2)
            cv2.line(image, (x_end, y-25), (x_end, y+25), 0, 2)
            cv2.putText(image, f"D{nominal}", (120, y-12), cv2.FONT_HERSHEY_SIMPLEX, .55, 0, 1)
            records.append({"text": f"Ø{nominal}", "box": [[120,y-30],[185,y-30],[185,y-10],[120,y-10]], "score": 1.0})
        # Independent full linear span establishes the half-section scale.
        cv2.line(image, (80, 320), (780, 320), 0, 2)
        records.append({"text": "350", "box": [[390,290],[455,290],[455,310],[390,310]], "score": 1.0})
    if rotated:
        transform = cv2.getRotationMatrix2D((width/2, height/2), 17, 1)
        image = cv2.warpAffine(image, transform, (width, height), borderValue=255)
        mask = cv2.warpAffine(mask, transform, (width, height), flags=cv2.INTER_NEAREST)
    return image, mask, {"meta": {"original_size": {"width": width, "height": height}}, "records": records}


def _polygon_mask(polyline, shape):
    result = np.zeros(shape, np.uint8)
    cv2.fillPoly(result, [np.rint(polyline).astype(np.int32)], 255)
    return result > 0


@pytest.mark.parametrize("rotated", [False, True])
def test_phase_free_fallback_handles_irregular_paired_hatches(tmp_path, rotated):
    image, expected, document = _drawing(hatched=False)
    hatch = np.zeros_like(image)
    # No single periodic phase: deliberately alternate paired and broad gaps.
    offset = -image.shape[0]
    for index in range(150):
        offset += (9, 31, 16, 37, 12)[index % 5]
        cv2.line(hatch, (offset, image.shape[0]-1), (offset+image.shape[0], 0), 255, 1)
    image[(hatch > 0) & (expected > 0)] = 0
    if rotated:
        transform = cv2.getRotationMatrix2D((450, 325), 17, 1)
        image = cv2.warpAffine(image, transform, (900, 650), borderValue=255)
        expected = cv2.warpAffine(expected, transform, (900, 650), flags=cv2.INTER_NEAREST)
    result = extract_unhatched(_save(tmp_path / "nonperiodic.png", image), document)
    assert result["status"] == "needs_review"
    assert result["polyline_px"][0] == result["polyline_px"][-1]
    assert result["evidence"]["global_hatch_phase_required"] is False
    predicted = _polygon_mask(result["polyline_px"], expected.shape)
    target = expected > 0
    assert np.count_nonzero(predicted & target) / np.count_nonzero(predicted | target) > .94


@pytest.mark.parametrize("blank", [False, True])
def test_phase_free_fallback_rejects_unhatched_and_blank_sources(tmp_path, blank):
    image, _, document = _drawing(hatched=False)
    if blank:
        image[:] = 255
    result = extract_unhatched(_save(tmp_path / "unresolved.png", image), document)
    assert result["polyline_px"] == []
    assert result["status"] == "needs_review"


def test_phase_free_fallback_retains_comparable_islands(tmp_path):
    image, expected, document = _drawing(multi=True)
    result = extract_unhatched(_save(tmp_path / "islands.png", image), document)
    assert len(result["candidates"]) >= 2
    merged = np.zeros(expected.shape, bool)
    for candidate in result["candidates"]:
        assert candidate["polyline_px"][0] == candidate["polyline_px"][-1]
        merged |= _polygon_mask(candidate["polyline_px"], expected.shape)
    assert np.count_nonzero(merged & (expected > 0)) / np.count_nonzero(expected) > .95


@pytest.mark.parametrize("shape,rotated", [("rectangle", False), ("ellipse", False), ("rectangle", True)])
def test_generic_raster_follows_hatched_shape_not_external_dimension_box(tmp_path, shape, rotated):
    image, expected, _ = _drawing(shape, rotated=rotated)
    path = _save(tmp_path / "synthetic.png", image)
    result = extract_main_profile(path)
    assert result["polyline_px"]
    assert result["polyline_px"][0] == result["polyline_px"][-1]
    predicted = _polygon_mask(result["polyline_px"], expected.shape)
    target = expected > 0
    iou = np.count_nonzero(predicted & target) / np.count_nonzero(predicted | target)
    assert iou > .94, result["evidence"]
    assert result["evidence"]["hatch_lattice_coherence"] > .5


def test_separated_comparable_material_islands_are_retained(tmp_path):
    image, expected, _ = _drawing(multi=True)
    result = extract_main_profile(_save(tmp_path / "islands.png", image))
    assert len(result["candidates"]) >= 2
    assert result["status"] == "needs_review"
    assert any("islands" in issue or "regions" in issue for issue in result["issues"])
    merged = np.zeros(expected.shape, bool)
    for candidate in result["candidates"]:
        merged |= _polygon_mask(candidate["polyline_px"], expected.shape)
    assert np.count_nonzero(merged & (expected > 0)) / np.count_nonzero(expected) > .95


@pytest.mark.parametrize("blank", [True, False])
def test_no_hatch_evidence_does_not_invent_a_material_region(tmp_path, blank):
    image = np.full((500, 700), 255, np.uint8) if blank else _drawing(hatched=False)[0]
    result = extract_main_profile(_save(tmp_path / "no-hatches.png", image))
    assert result["status"] == "needs_review"
    assert not result["polyline_px"]


def test_rectangle_corners_cannot_be_replaced_by_a_concyclic_arc():
    entities = fit_polyline([[0,0],[200,0],[200,120],[0,120],[0,0]], tolerance_px=1)
    assert len(entities) == 4
    assert all(e["type"] == "LINE" for e in entities)
    assert all(math.dist(e["end"], entities[(i+1)%len(entities)]["start"]) < 1e-9 for i,e in enumerate(entities))


@pytest.mark.parametrize("direction", [1, -1])
def test_arc_fit_preserves_exact_endpoints_radius_and_direction(direction):
    angles = np.linspace(.2, .2 + direction*1.8, 81)
    center, radius = np.array([30., 50.]), 80.
    points = center + radius * np.column_stack([np.cos(angles),np.sin(angles)])
    arc = _arc(points, .03)
    assert arc is not None
    assert arc["center"] == pytest.approx(center, abs=1e-8)
    assert arc["radius"] == pytest.approx(radius, abs=1e-8)
    assert arc["start"] == pytest.approx(points[0], abs=1e-8)
    assert arc["end"] == pytest.approx(points[-1], abs=1e-8)
    assert arc["clockwise"] == (direction < 0)


def test_source_dimension_consensus_drives_mm_export_without_reference(tmp_path):
    image, _, document = _drawing(dimensions=True)
    path = _save(tmp_path / "dimensioned.png", image)
    scale = estimate_scale(path, document)
    assert scale["status"] == "resolved", scale
    assert scale["pixels_per_mm"] == pytest.approx(2, abs=.02)
    result = build_automatic(path, document, tmp_path / "output")
    assert result["validation"]["passed"]
    assert result["coordinate_system"]["units"] == "mm"
    assert result["template_used"] is False and result["ground_truth_used"] is False
    assert result["manual_intervention"] is False
    doc = ezdxf.readfile(tmp_path / "output" / "drawing.dxf")
    assert doc.units == 4
    assert all(entity.dxftype() in {"LINE", "ARC"} for entity in doc.modelspace())
    # The known synthetic width is 660 px / 2 px per mm. Raster-edge tolerance
    # is explicitly separate from exact geometric connectivity.
    assert result["bounds"]["max_x"] - result["bounds"]["min_x"] == pytest.approx(330, abs=3)


def test_unresolved_scale_produces_explicit_pixel_units(tmp_path):
    image, _, document = _drawing()
    path = _save(tmp_path / "unscaled.png", image)
    result = build_automatic(path, document, tmp_path / "output")
    assert result["coordinate_system"]["units"] == "pixel"
    assert not result["validation"]["scaled_mm"]
    assert not result["validation"]["dimensions_verified"]
    assert ezdxf.readfile(tmp_path / "output" / "drawing.dxf").units == 0


@pytest.mark.parametrize("fragmented",[False,True])
def test_valid_dxf_does_not_hide_disconnected_material(tmp_path,monkeypatch,fragmented):
    from contour_agent.mask_geometry import extract_mask_profile
    mask=np.zeros((300,400),np.uint8)
    mask[30:200,30:280]=1
    if fragmented:mask[260:270,350:360]=1
    extraction=extract_mask_profile(mask,(400,300))
    monkeypatch.setattr("contour_agent.automatic.extract_main_profile",lambda *args,**kwargs:extraction)
    path=_save(tmp_path/"source.png",255-mask*255)
    result=build_automatic(path,{"records":[]},tmp_path/"output")
    assert result["validation"]["passed"]
    assert result["automatic_completion"] is (not fragmented)
    assert result["complete_material_exterior"] is (not fragmented)
    assert (tmp_path/"output"/"drawing.dxf").is_file()


def test_diameter_consensus_alone_cannot_resolve_half_vs_full_scale(tmp_path):
    image, _, document = _drawing(dimensions=True)
    document["records"] = document["records"][:-1]
    result = estimate_scale(_save(tmp_path / "ambiguous.png", image), document)
    assert result["status"] == "ambiguous"
    assert result["pixels_per_mm"] is None
    assert result["diameter_semantics"] == "unresolved_half_or_full"


def test_full_diameter_interpretation_uses_independent_linear_witness(tmp_path):
    image, _, document = _drawing(dimensions=True)
    document["records"][-1]["text"] = "700"
    result = estimate_scale(_save(tmp_path / "full.png", image), document)
    assert result["status"] == "resolved", result
    assert result["pixels_per_mm"] == pytest.approx(1, abs=.02)
    assert result["diameter_semantics"] == "full_diameter_span"
