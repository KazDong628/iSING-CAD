"""Known synthetic mask checks; no source dataset, reference or model required."""
import json

import cv2
import numpy as np
import pytest
from shapely import contains_xy
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from contour_agent.mask_geometry import extract_mask_profile


def _rectangle(shape=(120, 180), *, value=1):
    mask = np.zeros(shape, dtype=np.float32)
    mask[25:91, 40:141] = value
    return mask


def test_empty_mask_yields_no_invented_outline():
    result = extract_mask_profile(np.zeros((120, 180), np.uint8), (180, 120))
    assert result["status"] == "needs_review"
    assert result["polyline_px"] == []
    assert result["candidates"] == []
    assert result["confidence"] == 0
    assert result["evidence"]["foreground_coverage"] == 0
    assert result["evidence"]["retained_foreground_fraction"] == 0


@pytest.mark.parametrize("foreground", [True, 1, 255])
def test_binary_input_forms_valid_closed_exterior_without_probability_claim(foreground):
    mask = _rectangle().astype(bool if foreground is True else np.uint8)*foreground
    result = extract_mask_profile(mask, {"width": 180, "height": 120})
    points = result["polyline_px"]
    assert points[0] == points[-1]
    assert Polygon(points).is_valid
    assert Polygon(points).bounds == pytest.approx((40,25,140,90))
    assert result["confidence"] == 0
    assert result["evidence"]["input_kind"] == "binary"
    assert result["candidates"][0]["foreground_probability_mean"] is None
    assert result["evidence"]["engineering_certified"] is False


def test_probability_statistics_are_not_accuracy_acceptance():
    probability = np.full((120,180), .05, np.float32)
    probability[25:91,40:141] = .92
    probability[25,40:141] = .55
    result = extract_mask_profile(probability, (180,120))
    expected = probability[probability >= .5].mean()
    assert result["confidence"] == pytest.approx(expected)
    assert result["candidates"][0]["boundary_probability_mean"] < result["confidence"]
    assert result["evidence"]["ambiguous_probability_fraction"] > 0
    assert "uncalibrated" in result["evidence"]["confidence_kind"]
    assert result["status"] == "needs_review"
    assert not result["evidence"]["dimensions_verified"]


def test_significant_separate_islands_are_retained_and_coverage_disclosed():
    mask = np.zeros((150,240), np.uint8)
    mask[20:120,20:100] = 1
    mask[25:105,140:220] = 1
    mask[140,230] = 1
    result = extract_mask_profile(mask, (240,150))
    assert len(result["candidates"]) == 2
    assert result["primary_candidate_id"] == "material-0"
    assert all(Polygon(c["polyline_px"]).is_valid for c in result["candidates"])
    assert result["evidence"]["discarded_foreground_pixels"] == 1
    assert result["evidence"]["retained_foreground_fraction"] == pytest.approx(14400/14401)
    assert result["evidence"]["primary_foreground_fraction"] == pytest.approx(8000/14401)
    assert any("Multiple" in issue for issue in result["issues"])


def test_hole_is_diagnosed_but_only_exterior_is_exported():
    mask = _rectangle()
    mask[45:70,65:100] = 0
    result = extract_mask_profile(mask, (180,120))
    assert result["evidence"]["holes_omitted"] == 1
    assert result["evidence"]["interior_holes_exported"] is False
    assert result["candidates"][0]["omitted_hole_contour_area_px"] > 0
    assert Polygon(result["polyline_px"]).contains(Point(80,55))
    assert any("Interior holes" in issue for issue in result["issues"])


def test_resized_coordinates_use_declared_pixel_center_transform():
    mask = np.zeros((200,300), np.uint8)
    mask[20:81,30:91] = 1
    result = extract_mask_profile(mask, (600,400))
    assert Polygon(result["polyline_px"]).bounds == pytest.approx((60.5,40.5,180.5,160.5))
    mapping = result["evidence"]["coordinate_mapping"]
    assert mapping["scale_x"] == mapping["scale_y"] == 2
    assert mapping["kind"] == "pixel_center_resize"


def test_axis_resize_ratios_are_kept_separate():
    mask = _rectangle()
    result = extract_mask_profile(mask, (360,360))
    assert Polygon(result["polyline_px"]).bounds == pytest.approx((80.5,76,280.5,271))
    assert result["evidence"]["coordinate_mapping"]["scale_x"] == 2
    assert result["evidence"]["coordinate_mapping"]["scale_y"] == 3


def test_self_touching_centre_trace_uses_exact_cells_without_buffer_repair():
    mask = np.zeros((120,120), np.uint8)
    cv2.fillPoly(mask, [np.array([[20,20],[100,100],[20,100],[100,20]], np.int32)], 1)
    result = extract_mask_profile(mask, (120,120))
    assert Polygon(result["polyline_px"]).is_valid
    raw=Polygon(result["raw_polyline_px"])
    assert raw.area == int(mask.sum())
    yy,xx=np.indices(mask.shape)
    assert np.array_equal(contains_xy(raw,xx,yy),mask.astype(bool))
    assert not result["evidence"]["rejected_components"]
    assert result["evidence"]["boundary_representation_fallback"]
    assert result["evidence"]["self_intersections_checked"] is True
    assert result["evidence"]["topological_repair_applied"] is False


def test_one_pixel_spur_preserves_all_occupied_cells_and_original_mask(tmp_path):
    mask=np.zeros((120,120),np.uint8)
    mask[20:80,20:80]=1;mask[50,80:105]=1
    before=mask.copy()
    result=extract_mask_profile(mask,(120,120),tmp_path)
    raw=Polygon(result["raw_polyline_px"])
    assert raw.is_valid and raw.contains(Point(103,50))
    assert raw.area == int(mask.sum()) == 3625
    yy,xx=np.indices(mask.shape)
    assert np.array_equal(contains_xy(raw,xx,yy),before.astype(bool))
    assert np.array_equal(mask,before)
    for name in ("threshold-mask.png","material-mask.png"):
        stored=cv2.imdecode(np.fromfile(tmp_path/name,np.uint8),cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(stored,before*255)
    assert result["evidence"]["retained_foreground_fraction"] == 1
    conversion=result["evidence"]["boundary_conversions"][0]
    assert conversion["foreground_pixels_changed"] == conversion["bridges_added"] == 0
    assert conversion["pixel_area_before"] == conversion["pixel_cell_area_after"]
    assert conversion["mask_morphology_applied"] is False


def test_corner_touching_pixel_islands_remain_separate_without_invented_bridge():
    mask=np.zeros((100,100),np.uint8)
    mask[10:40,10:40]=1;mask[40:70,40:70]=1
    result=extract_mask_profile(mask,(100,100),simplify_tolerance_px=0)
    candidates=result["candidates"]
    assert len(candidates) == 2
    polygons=[Polygon(candidate["raw_polyline_px"]) for candidate in candidates]
    assert all(polygon.is_valid for polygon in polygons)
    assert polygons[0].intersection(polygons[1]).geom_type == "Point"
    assert sum(polygon.area for polygon in polygons) == int(mask.sum())
    yy,xx=np.indices(mask.shape)
    assert np.array_equal(contains_xy(unary_union(polygons),xx,yy),mask.astype(bool))
    assert result["evidence"]["boundary_conversions"][0]["part_count"] == 2
    assert result["evidence"]["retained_foreground_fraction"] == 1


def test_cell_fallback_preserves_hole_pixels_and_declared_resize_transform(tmp_path):
    mask=np.zeros((120,120),np.uint8)
    mask[20:80,20:80]=1;mask[50,80:105]=1;mask[30:40,30:40]=0
    result=extract_mask_profile(mask,(240,360),tmp_path,simplify_tolerance_px=0)
    candidate=result["candidates"][0]
    assert candidate["bounds_px"] == pytest.approx((39.5,59.5,209.5,239.5))
    assert candidate["hole_count"] == 1
    assert candidate["omitted_hole_contour_area_px"] == 600
    assert candidate["material_pixel_area_px"] == int(mask.sum())*6
    stored=cv2.imdecode(np.fromfile(tmp_path/"material-mask.png",np.uint8),cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(stored,mask*255)
    assert result["evidence"]["foreground_mask_modified"] is False


def test_degenerate_simplification_reverts_to_valid_raw_ring():
    result = extract_mask_profile(_rectangle(), (180,120), simplify_tolerance_px=1000)
    assert result["candidates"][0]["simplification_reverted"] is True
    assert result["polyline_px"] == result["raw_polyline_px"]
    assert Polygon(result["polyline_px"]).is_valid


def test_border_contact_is_disclosed_as_potential_clipped_view():
    mask = np.zeros((100,100), np.uint8)
    mask[:70,:60] = 1
    result = extract_mask_profile(mask, (100,100))
    assert result["polyline_px"]
    assert result["candidates"][0]["touches_image_border"]
    assert any("clipped" in issue for issue in result["issues"])


def test_diagnostics_are_serializable_and_artifacts_exist(tmp_path):
    result = extract_mask_profile(_rectangle(), (180,120), tmp_path)
    saved = json.loads((tmp_path/"result.json").read_text(encoding="utf-8"))
    assert saved["polyline_px"] == result["polyline_px"]
    assert all((tmp_path/name).exists() for name in result["artifacts"].values())


def test_small_material_fragment_cannot_hide_behind_candidate_area_filter():
    mask = np.zeros((600,800),np.uint8)
    mask[30:430,30:430] = 1
    mask[500:516,600:624] = 1  # 384 pixels, below the former 4% primary filter.
    result = extract_mask_profile(mask,(800,600))
    assert len(result["candidates"]) == 1
    connectivity = result["evidence"]["connectivity"]
    assert connectivity["meaningful_components_4"] == 2
    assert connectivity["component_areas_4_mask_px"] == [160000,384]
    assert not connectivity["complete_exterior_candidate"]
    assert not connectivity["single_material_region"]


def test_corner_touching_regions_fail_complete_material_gate():
    mask = np.zeros((100,100),np.uint8)
    mask[10:40,10:40] = 1
    mask[40:70,40:70] = 1
    connectivity = extract_mask_profile(mask,(100,100))["evidence"]["connectivity"]
    assert connectivity["components_8"] == 1
    assert connectivity["components_4"] == 2
    assert connectivity["corner_only_connections"] == 1
    assert not connectivity["complete_exterior_candidate"]


def test_tiny_noise_is_disclosed_separately_and_real_bridge_can_pass():
    mask = _rectangle()
    mask[110,170] = 1
    connectivity = extract_mask_profile(mask,(180,120))["evidence"]["connectivity"]
    assert connectivity["components_4"] == 2
    assert connectivity["tiny_noise_pixels"] == 1
    assert connectivity["complete_exterior_candidate"]
    mask[109:111,169:171] = 1
    connectivity = extract_mask_profile(mask,(180,120))["evidence"]["connectivity"]
    assert not connectivity["complete_exterior_candidate"]



@pytest.mark.parametrize("mask", [np.full((10,10), np.nan), np.full((10,10), 2), np.zeros((2,3,4)), np.zeros((0,10))])
def test_invalid_probability_or_shape_is_not_silently_normalized(mask):
    with pytest.raises(ValueError):
        extract_mask_profile(mask, (100,100))


@pytest.mark.parametrize("size", [(100,0), (100.5,100), (True,100), (100,), None])
def test_invalid_original_size_is_rejected(size):
    with pytest.raises(ValueError):
        extract_mask_profile(_rectangle(), size)
