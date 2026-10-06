"""Stroke interiors must be real, bounded ink, not a wider distance gate."""
import cv2
import numpy as np
import pytest

from contour_agent.topology import _StrokeEvidence, stroke_support_fraction


def horizontal_stroke():
    gray = np.full((160,180),255,np.uint8)
    cv2.rectangle(gray,(20,58),(160,62),0,-1)
    points = np.c_[np.linspace(35.,145.,111),np.full(111,60.)]
    return gray, points


def test_thick_black_stroke_center_is_supported_without_changing_legacy_metrics():
    gray,points = horizontal_stroke()
    evidence = _StrokeEvidence(gray,[],2.)
    measured = evidence.summarize(points)
    assert evidence.near == 1.5
    assert evidence.metadata["orientation_tolerance_degrees"] == 45.
    assert measured["edge_supported_fraction"] == 0.
    assert measured["legacy_edge_supported_fraction"] == 0.
    assert measured["legacy_edge_only_metrics"]["edge_supported_fraction"] == 0.
    assert measured["stroke_supported_fraction"] == 1.
    assert measured["ink_interior_supported_fraction"] == 1.
    assert measured["support_measurement_version"] == "source-stroke-support-v2"
    distances,supported = evidence.query(points)
    assert np.all(distances > evidence.near) and np.all(supported)


def test_white_gap_between_parallel_strokes_is_not_an_ink_interior():
    gray = np.full((160,180),255,np.uint8)
    cv2.rectangle(gray,(20,50),(160,54),0,-1)
    cv2.rectangle(gray,(20,66),(160,70),0,-1)
    points = np.c_[np.linspace(35.,145.,111),np.full(111,60.)]
    measured = _StrokeEvidence(gray,[],2.).summarize(points)
    assert measured["stroke_supported_fraction"] == 0.
    assert measured["ink_interior_supported_fraction"] == 0.


def test_black_filled_region_is_not_promoted_to_a_thin_stroke():
    gray = np.full((160,180),255,np.uint8)
    cv2.rectangle(gray,(20,30),(160,90),0,-1)
    points = np.c_[np.linspace(35.,145.,111),np.full(111,60.)]
    measured = _StrokeEvidence(gray,[],2.).summarize(points)
    assert measured["stroke_supported_fraction"] == 0.
    assert measured["ink_interior_supported_fraction"] == 0.


def test_candidate_crossing_a_hatch_stroke_in_wrong_direction_is_not_supported():
    gray = np.full((160,180),255,np.uint8)
    cv2.rectangle(gray,(78,20),(82,140),0,-1)
    points = np.c_[np.linspace(35.,145.,111),np.full(111,60.)]
    evidence = _StrokeEvidence(gray,[],2.)
    measured = evidence.summarize(points)
    assert measured["stroke_supported_fraction"] == 0.
    assert not evidence.query(points)[1].any()


def test_ocr_glyph_interiors_cannot_supply_added_stroke_evidence():
    gray,points = horizontal_stroke()
    records = [{"box":[[10.,45.],[170.,45.],[170.,75.],[10.,75.]]}]
    measured = _StrokeEvidence(gray,records,2.).summarize(points)
    assert measured["ink_interior_supported_fraction"] == 0.
    assert measured["stroke_supported_fraction"] == measured["edge_supported_fraction"] == 0.


def test_one_sided_ink_edge_or_wrongly_oriented_path_is_not_interior_evidence():
    gray, _ = horizontal_stroke()
    points = np.c_[np.full(101,80.),np.linspace(40.,80.,101)]
    evidence = _StrokeEvidence(gray,[],2.)
    _,legacy,interior = evidence._query_evidence(points)
    assert not interior.any() and not legacy.any()
    # A wide dark block has an edge on only one side within the narrow band.
    cv2.rectangle(gray,(20,62),(160,150),0,-1)
    center = np.c_[np.linspace(35.,145.,111),np.full(111,60.)]
    measured = _StrokeEvidence(gray,[],2.).summarize(center)
    assert measured["ink_interior_supported_fraction"] == 0.


def test_pairwise_support_comparison_never_compares_v2_to_legacy():
    old = {"edge_supported_fraction":.8}
    new = {"edge_supported_fraction":.72,"stroke_supported_fraction":.91,
           "support_measurement_version":"source-stroke-support-v2"}
    newer = {"edge_supported_fraction":.7,"stroke_supported_fraction":.9,
             "support_measurement_version":"source-stroke-support-v2"}
    assert stroke_support_fraction(new) == .91
    assert stroke_support_fraction(new,against=newer) == .91
    assert stroke_support_fraction(newer,against=new) == .9
    assert stroke_support_fraction(old,against=new) == .8
    assert stroke_support_fraction(new,against=old) == .72
    with pytest.raises(ValueError):stroke_support_fraction({"stroke_supported_fraction":True})
