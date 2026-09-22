"""Source-only cross-family scale checks; no dataset or reference geometry."""
import copy
import cv2
import numpy as np
import pytest

from contour_agent.dimension_evidence import estimate_scale
from contour_agent.ocr import parse_dimension


@pytest.mark.parametrize("text,kind,nominal,upper,lower", [
    ("⌀840+4", "diameter", 840, 4, 0),
    ("Ø240+4", "diameter", 240, 4, 0),
    ("⌀730-4", "diameter", 730, 0, -4),
    ("φ25+.2", "diameter", 25, .2, 0),
    ("R2.5-0.1", "radius", 2.5, 0, -.1),
])
def test_complete_marked_unilateral_dimensions(text, kind, nominal, upper, lower):
    parsed = parse_dimension(text)
    assert (parsed["kind"], parsed["nominal"], parsed["upper_deviation"], parsed["lower_deviation"]) == (kind, nominal, upper, lower)


@pytest.mark.parametrize("text", ["⌀782±20BD", "⌀840+4A", "⌀240+", "⌀240++4", "⌀240+4-2", "20-25", "185+3", "R80Ra6.3"])
def test_partial_merged_or_unmarked_range_text_is_not_repaired(text):
    assert parse_dimension(text)["kind"] == "unknown"


def _drawing(tmp_path, diameter_scale, *, extra_mode=False):
    image = np.full((1200, 1300), 255, np.uint8)
    records = []
    lengths = [(100, 2), (160, 2), (230, 2)]
    if extra_mode:
        lengths += [(110, 4), (170, 4), (210, 4)]
    for i, (nominal, scale) in enumerate(lengths):
        y = 80 + i * 100
        lo, hi = 120, 120 + round(nominal * scale)
        cv2.line(image, (lo, y), (hi, y), 0, 1)
        cv2.line(image, (lo, y-15), (lo, y+15), 0, 1)
        cv2.line(image, (hi, y-15), (hi, y+15), 0, 1)
        records.append({"text": str(nominal), "box": [[150,y-28],[220,y-28],[220,y-5],[150,y-5]]})
    for i, nominal in enumerate([200, 500, 800]):
        y = 750 + i * 120
        hi = round(120 + diameter_scale * nominal / 2)
        cv2.line(image, (20, y), (hi, y), 0, 1)
        cv2.line(image, (hi, y-15), (hi, y+15), 0, 1)
        records.append({"text": f"Ø{nominal}+4", "box": [[60,y-28],[140,y-28],[140,y-5],[60,y-5]]})
    path = tmp_path / "source.png"
    path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
    return path, {"records": records}


def test_biased_diameter_endpoints_cannot_override_three_independent_lengths(tmp_path):
    path, document = _drawing(tmp_path, 2.066)
    before = copy.deepcopy(document)
    result = estimate_scale(path, document)
    assert document == before
    assert result["status"] == "resolved"
    assert result["pixels_per_mm"] == pytest.approx(2, abs=.01)
    assert result["scale_kind"] == "linear_dimension_consensus"
    assert result["axis_origin_px"] is None
    assert result["cross_evidence_consistency"]["status"] == "conflict"
    assert result["cross_evidence_consistency"]["combined_ratio_spread"] > .03
    assert result["cross_evidence_consistency"]["maximum_ratio_spread"] == .03
    assert result["diameter_evidence"]["pixels_per_mm"] > 2.06
    assert result["discarded_diameter_reason"] and result["issues"]


def test_consistent_diameter_and_linear_evidence_remain_available(tmp_path):
    path, document = _drawing(tmp_path, 2.02)
    result = estimate_scale(path, document)
    assert result["status"] == "resolved"
    assert result["pixels_per_mm"] == pytest.approx(2.02, abs=.01)
    assert result["diameter_semantics"] == "half_section_radius_station"
    assert result["cross_evidence_consistency"]["status"] == "consistent"
    assert result["cross_evidence_consistency"]["combined_ratio_spread"] <= .03


def test_verified_diameter_does_not_hide_multiple_linear_scale_modes(tmp_path):
    path, document = _drawing(tmp_path, 2, extra_mode=True)
    result = estimate_scale(path, document)
    assert result["status"] == "ambiguous"
    assert result["pixels_per_mm"] is None
    assert len(result["alternatives"]) == 2
    assert result["cross_evidence_consistency"]["status"] == "multiple_linear_modes"
