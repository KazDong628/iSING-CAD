"""Synthetic linear-only scale consensus checks without dataset or GT access."""
import cv2
import numpy as np
import pytest

from contour_agent.dimension_evidence import _linear_scale_consensus, estimate_scale


def _source(tmp_path, dimensions, *, diameter=False):
    image = np.full((850, 1150), 255, np.uint8)
    records = []
    for index, (nominal, pixels_per_mm) in enumerate(dimensions):
        y = 80 + index*105
        lo, hi = 120, 120+round(nominal*pixels_per_mm)
        cv2.line(image, (lo, y), (hi, y), 0, 1)
        cv2.line(image, (lo, y-15), (lo, y+15), 0, 1)
        cv2.line(image, (hi, y-15), (hi, y+15), 0, 1)
        text = str(nominal)
        cv2.putText(image, text, (150, y-8), cv2.FONT_HERSHEY_SIMPLEX, .6, 0, 1)
        records.append({"text": text, "box": [[148,y-29],[225,y-29],[225,y-5],[148,y-5]]})
    if diameter:
        records.append({"text": "Ø400", "box": [[800,730],[875,730],[875,750],[800,750]]})
    path = tmp_path/"source.png"
    success, encoded = cv2.imencode(".png", image)
    assert success
    path.write_bytes(encoded.tobytes())
    return path, {"records": records}


@pytest.mark.parametrize("diameter", [False, True])
def test_three_linear_dimensions_resolve_scale_without_sufficient_diameters(tmp_path, diameter):
    path, document = _source(tmp_path, [(100,2), (160,2), (230,2)], diameter=diameter)
    result = estimate_scale(path, document)
    assert result["status"] == "resolved"
    assert result["pixels_per_mm"] == pytest.approx(2, abs=.025)
    assert result["scale_kind"] == "linear_dimension_consensus"
    assert result["method"].startswith("linear_dimension_consensus")
    assert result["axis_origin_px"] is None
    assert result["axis"] is None
    assert result["rotation_axis_resolved"] is False
    assert result["distinct_dimensions"] == 3
    assert result["distinct_physical_lines"] == 3
    assert result["ratio_spread"] <= .03
    assert result["measurement_axes"] == ["x"]
    assert "isotropic_image_scaling" in result["assumptions"]
    assert result["diameter_evidence"]["status"] == "unresolved"


def test_duplicate_physical_line_does_not_supply_three_independent_lengths(tmp_path):
    path, document = _source(tmp_path, [(100,3)])
    original = document["records"][0]
    document["records"] = [{**original, "text": str(nominal)} for nominal in (100,101,102)]
    result = estimate_scale(path, document)
    assert result["status"] == "unresolved"
    assert result["pixels_per_mm"] is None
    assert len(result["linear_witnesses"]) >= 3


def test_three_lines_with_repeated_nominal_do_not_resolve(tmp_path):
    path, document = _source(tmp_path, [(100,2), (100,2), (100,2)])
    result = estimate_scale(path, document)
    assert result["status"] == "unresolved"
    assert result["pixels_per_mm"] is None


def test_competing_consistent_scales_remain_ambiguous(tmp_path):
    dimensions = [(100,2), (160,2), (200,2), (100,4), (160,4), (200,4)]
    path, document = _source(tmp_path, dimensions)
    result = estimate_scale(path, document)
    assert result["status"] == "ambiguous"
    assert result["pixels_per_mm"] is None
    assert result["axis_origin_px"] is None
    assert len(result["alternatives"]) == 2
    assert sorted(c["pixels_per_mm"] for c in result["alternatives"]) == pytest.approx([2,4], abs=.025)


def _witness(index, nominal, ratio):
    return {"record_id": f"r{index}", "nominal": nominal, "pixels_per_mm": ratio,
            "axis": "x", "line": {"lo": 10., "hi": 10+nominal*ratio, "cross": index*50.}}


def test_three_percent_rule_is_maximum_minimum_spread_not_plus_minus():
    witnesses = [_witness(i, nominal, ratio) for i, (nominal, ratio) in
                 enumerate([(100,1.97), (150,2), (200,2.03)])]
    assert _linear_scale_consensus(witnesses) == []


def test_overlapping_windows_with_incompatible_combined_range_are_ambiguous():
    witnesses = [_witness(i, 100+i*30, ratio) for i, ratio in enumerate([2,2.025,2.05,2.075])]
    clusters = _linear_scale_consensus(witnesses)
    assert len(clusters) == 2
    assert all(c["ratio_spread"] <= .03 for c in clusters)


def test_more_support_for_one_scale_does_not_hide_competing_scale():
    first = [_witness(i, 100+i*20, 2) for i in range(5)]
    second = [_witness(10+i, 110+i*25, 4) for i in range(3)]
    clusters = _linear_scale_consensus(first+second)
    assert len(clusters) == 2
    assert {c["distinct_dimensions"] for c in clusters} == {3,5}
