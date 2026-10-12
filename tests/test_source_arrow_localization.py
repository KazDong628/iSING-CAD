from copy import deepcopy

import cv2
import numpy as np
import pytest

from contour_agent.constraint_binding import verify_source_arrow_proposal
from contour_agent.source_arrow_localization import (localize_source_arrow_proposal, source_arrow_hypotheses,
    native_radius_leader_segments, verify_source_hough_leader, source_label_shaft_ownership)


def source():
    gray = np.full((160, 310), 255, np.uint8)
    cv2.line(gray, (100, 75), (229, 75), 0, 2)
    cv2.fillConvexPoly(gray, np.array([[100,75],[116,69],[116,81]], np.int32), 0)
    record = {"id": "r000", "parsed": {"kind": "radius", "nominal": 40.},
              "box": [[230., 50.], [275., 50.], [275., 90.], [230., 90.]]}
    boundary = np.c_[np.full(11, 100.), np.linspace(70, 80, 11)]
    return gray, record, boundary


def test_local_ink_snaps_text_center_to_actual_shaft_without_weakening_verifier():
    gray, record, boundary = source()
    proposal = {"record_id": "r000", "tip_px": [101., 74.], "shaft_px": [245., 59.]}
    assert verify_source_arrow_proposal(gray, record, proposal, boundary, 5.) is None
    result = localize_source_arrow_proposal(gray, record, proposal, boundary, 5.)
    assert result is not None
    assert result["arrowhead_verified"] and result["shaft_evidence"]["verified"]
    assert result["shaft_evidence"]["supported_fraction"] >= .88
    assert result["source_pixel_localization"]["method"] == "native_source_hough_shaft_snap"
    assert result["source_pixel_localization"]["nominal_used_to_rank"] is False
    # The corrected segment independently passes the unmodified verifier.
    snapped = {"shaft_px": result["segment_px"][0], "tip_px": result["segment_px"][1]}
    assert verify_source_arrow_proposal(gray, record, snapped, boundary, 5.) is not None


def test_snapping_never_uses_radius_to_choose_a_target():
    gray, record, _ = source()
    proposal = {"tip_px": [101., 74.], "shaft_px": [245., 59.]}
    original = source_arrow_hypotheses(gray, record, proposal)
    other = deepcopy(record); other["parsed"]["nominal"] = 99999.
    assert source_arrow_hypotheses(gray, other, proposal) == original
    assert 0 < len(original) <= 24


def test_source_snapping_cannot_repair_missing_arrow_or_broken_ink():
    gray, record, boundary = source()
    proposal = {"record_id": "r000", "tip_px": [101., 74.], "shaft_px": [245., 59.]}
    no_arrow = np.full_like(gray, 255)
    cv2.line(no_arrow, (100,75), (229,75), 0, 2)
    assert localize_source_arrow_proposal(no_arrow, record, proposal, boundary, 5.) is None
    gray[67:85, 155:190] = 255
    assert localize_source_arrow_proposal(gray, record, proposal, boundary, 5.) is None


def test_wrong_record_or_unbounded_pixel_shift_is_not_snapped():
    gray, record, boundary = source()
    wrong = {"record_id": "r999", "tip_px": [101., 74.], "shaft_px": [245., 59.]}
    assert localize_source_arrow_proposal(gray, record, wrong, boundary, 5.) is None
    distant = {"record_id": "r000", "tip_px": [20., 25.], "shaft_px": [245., 59.]}
    assert localize_source_arrow_proposal(gray, record, distant, boundary, 5.) is None


def test_competing_verified_arrowheads_remain_ambiguous(monkeypatch):
    gray, record, boundary = source()
    original = {"tip_px": [101., 74.], "shaft_px": [245., 59.]}
    candidates = [{"proposal": {"tip_px": [x, 75.], "shaft_px": [229., 75.]},
                   "localization": {"tip_shift_px": 1.}} for x in [100., 170.]]
    monkeypatch.setattr("contour_agent.source_arrow_localization.source_arrow_hypotheses", lambda *args, **kwargs: candidates)

    def verifier(gray, record, proposal, *args):
        if proposal is original:
            return None
        return {"score": 1., "arrowhead": {"tip_px": proposal["tip_px"], "length_px": 16.}}

    assert localize_source_arrow_proposal(gray, record, original, boundary, 5., verifier=verifier) is None


def test_native_short_leader_survives_full_image_downsampling_without_weaker_checks():
    from contour_agent.constraint_binding import _leaders
    gray = np.full((300, 3266), 255, np.uint8)
    cv2.line(gray, (100, 75), (140, 75), 0, 1)
    cv2.fillConvexPoly(gray, np.array([[100,75],[108,71],[108,79]], np.int32), 0)
    record = {"id": "r000", "parsed": {"kind": "radius", "nominal": 40.},
              "box": [[141.,50.],[176.,50.],[176.,90.],[141.,90.]]}
    boundary = np.c_[np.full(11, 100.), np.linspace(70, 80, 11)]
    assert _leaders(gray, [record]) == []
    preserved = gray.copy()
    segments = native_radius_leader_segments(gray, record, [record])
    assert 0 < len(segments) <= 96
    assert np.array_equal(gray, preserved)
    changed = deepcopy(record); changed["parsed"]["nominal"] = 99999.
    assert all(np.array_equal(a,b) for a,b in zip(segments,
               native_radius_leader_segments(gray, changed, [changed])))
    accepted = [verify_source_hough_leader(gray, record, segment, boundary, 5.)
                for line in segments for segment in (line, line[::-1])]
    accepted = [row for row in accepted if row is not None]
    assert accepted
    assert all(row["shaft_evidence"]["verified"] and row["shaft_evidence"]["supported_fraction"] >= .88
               and row["proposal_origin"] == "source_hough" and not row["model_proposal_used"] for row in accepted)


def test_hough_crossing_requires_full_arrow_shaft_and_label_ray():
    gray, record, boundary = source()
    crossing = np.array([[180.,40.],[180.,100.]])
    cv2.line(gray, (180,40), (180,100), 0, 2)
    segment = np.array([[229.,75.],[100.,75.]])
    accepted = verify_source_hough_leader(gray, record, segment, boundary, 5., [boundary, crossing])
    assert accepted is not None and accepted["shaft_evidence"]["verified"]
    assert accepted["crossing_source_contour"] and not accepted["contour_visibility"]["verified"]
    assert accepted["proposal_origin"] == "source_hough"
    broken = gray.copy(); broken[68:84, 145:170] = 255
    assert verify_source_hough_leader(broken, record, segment, boundary, 5., [boundary,crossing]) is None
    hatch = np.full_like(gray, 255)
    cv2.line(hatch, (100,75), (229,75), 0, 2)
    cv2.line(hatch, (180,40), (180,100), 0, 2)
    assert verify_source_hough_leader(hatch, record, segment, boundary, 5., [boundary,crossing]) is None
    wrong_box = deepcopy(record); wrong_box["box"] = [[230.,110.],[275.,110.],[275.,145.],[230.,145.]]
    assert verify_source_hough_leader(gray, wrong_box, segment, boundary, 5., [boundary,crossing]) is None
    assert verify_source_hough_leader(None, record, segment, boundary, 5., [boundary,crossing]) is None


def repetitive_label_source(*, continues_through_label=True, family=True):
    gray, record, boundary = source()
    # Keep a source-observed taper at the material contact. Its local shape
    # alone cannot prove that a repetitive stroke belongs to this OCR label.
    if continues_through_label:
        cv2.line(gray, (100,75), (309,75), 0, 2)
    if family:
        for y in (105,135):
            cv2.line(gray, (100,y), (309,y), 0, 2)
    return gray, record, boundary


def test_taper_and_full_ink_do_not_assign_hatch_crossing_to_radius_label():
    from contour_agent.constraint_binding import _arrowhead_evidence, _leader_evidence
    gray, record, boundary = repetitive_label_source()
    direction = np.array([-1., 0.])
    assert _arrowhead_evidence(gray, np.array([100.,75.]), direction, 60., 5., boundary) is not None
    audit = source_label_shaft_ownership(gray, record, [100.,75.], direction)
    assert audit["repetitive_label_crossing"]
    assert audit["label_and_extension_support"] == 1.
    assert audit["parallel_family"]["pitch_px"] == pytest.approx(30., abs=3.)
    proposal = {"tip_px": [100.,75.], "shaft_px": [229.,75.]}
    assert verify_source_arrow_proposal(gray, record, proposal, boundary, 5.) is None
    rejected = []
    assert _leader_evidence(np.asarray(record["box"]), boundary, np.array([60.,75.]),
                            [np.array([[229.,75.],[100.,75.]])], 5., gray, rejections=rejected) is None
    assert rejected[0]["reason"] == "repetitive_source_stroke_crosses_label_without_unique_leader_ownership"


def test_long_true_radius_leader_is_not_rejected_for_passing_its_label():
    gray, record, boundary = repetitive_label_source(family=False)
    audit = source_label_shaft_ownership(gray, record, [100.,75.], [-1.,0.])
    assert audit["label_and_extension_support"] == 1.
    assert not audit["repetitive_label_crossing"]
    assert verify_source_arrow_proposal(gray, record, {"tip_px": [100.,75.], "shaft_px": [229.,75.]}, boundary, 5.)


def test_parallel_background_does_not_reject_a_leader_ending_at_its_text():
    gray, record, boundary = repetitive_label_source(continues_through_label=False)
    audit = source_label_shaft_ownership(gray, record, [100.,75.], [-1.,0.])
    assert not audit["repetitive_label_crossing"]
    assert verify_source_arrow_proposal(gray, record, {"tip_px": [100.,75.], "shaft_px": [229.,75.]}, boundary, 5.)


def test_label_shaft_ownership_uses_no_nominal_radius_or_primitive():
    gray, record, _ = repetitive_label_source()
    before = source_label_shaft_ownership(gray, record, [100.,75.], [-1.,0.])
    changed = deepcopy(record)
    changed["parsed"]["nominal"] = 99999.
    assert source_label_shaft_ownership(gray, changed, [100.,75.], [-1.,0.]) == before


def test_two_parallel_strokes_do_not_establish_repetitive_hatch_family():
    gray, record, _ = repetitive_label_source(family=False)
    cv2.line(gray, (100,105), (309,105), 0, 2)
    assert not source_label_shaft_ownership(gray, record, [100.,75.], [-1.,0.])["repetitive_label_crossing"]


def test_twenty_fifth_real_ink_competitor_cannot_be_discarded_as_unique_arrow():
    gray = np.full((170, 310), 255, np.uint8)
    for y in range(42, 117, 4):
        if abs(y-75) > 8 and abs(y-105) > 8:
            cv2.line(gray, (100, y), (229, y), 0, 1)
    for y in (75, 105):
        cv2.line(gray, (100, y), (229, y), 0, 2)
        cv2.fillConvexPoly(gray, np.array([[100, y], [116, y-6], [116, y+6]], np.int32), 0)
    row = {"id": "r000", "parsed": {"kind": "radius", "nominal": 40.},
           "box": [[230., 30.], [285., 30.], [285., 115.], [230., 115.]]}
    boundary = np.array([[100., y] for y in range(40, 120)])
    proposal = {"record_id": "r000", "tip_px": [101., 74.], "shaft_px": [245., 59.]}
    audit = {}
    assert localize_source_arrow_proposal(gray, row, proposal, boundary, 5., audit=audit) is None
    assert audit["status"] == "hypothesis_budget_exhausted" and audit["search_complete"] is False
    assert audit["enumerated_hypothesis_count"] == 25
    early = [verify_source_arrow_proposal(gray, row, item["proposal"], boundary, 5.)
             for item in audit["diagnostic_only_hypotheses"]]
    early = [item for item in early if item]
    assert early and all(item["arrowhead"]["tip_px"][1] < 85 for item in early)
    # The previously omitted second arrow is real source ink and independently
    # passes the unchanged complete shaft/arrow verifier; no mocked validator.
    late = verify_source_arrow_proposal(gray, row, {"tip_px": [100., 105.],
                                        "shaft_px": [229., 105.]}, boundary, 5.)
    assert late is not None and late["shaft_evidence"]["verified"]
    assert late["arrowhead"]["tip_px"][1] > 95


def test_exactly_full_unique_hypothesis_inventory_is_complete():
    gray, row, _ = source()
    proposal = {"tip_px": [101., 74.], "shaft_px": [245., 59.]}
    full = source_arrow_hypotheses(gray, row, proposal)
    assert 1 < len(full) < 24
    audit = {}
    assert source_arrow_hypotheses(gray, row, proposal, limit=len(full), audit=audit) == full
    assert audit["search_complete"] is True and audit["status"] == "completed"
    assert source_arrow_hypotheses(gray, row, proposal, limit=len(full)-1, audit=audit) == []
    assert audit["search_complete"] is False


def test_unrelated_raw_fragments_do_not_hide_a_late_label_adjacent_segment(monkeypatch):
    gray = np.full((200, 300), 255, np.uint8)
    row = {"id": "r", "parsed": {"kind": "radius"},
           "box": [[200., 60.], [230., 60.], [230., 90.], [200., 90.]]}
    # Crop origin is (152, 12). Six hundred vertical fragments cannot reach
    # this text rectangle; the final horizontal stroke can. No raw[:500].
    raw = np.array([[[10, 20, 10, 50]]] * 600 + [[[10, 63, 47, 63]]], np.int32)
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *args, **kwargs: raw)
    audit = {}
    result = native_radius_leader_segments(gray, row, audit=audit)
    assert len(result) == 1 and audit["search_complete"] is True
    assert audit["raw_segment_counts"] == [601, 601]
    assert audit["observed_native_segment_count"] == 1


def test_native_budget_counts_only_deduplicated_label_competitors(monkeypatch):
    gray = np.full((200, 300), 255, np.uint8)
    row = {"id": "r", "parsed": {"kind": "radius"},
           "box": [[200., 60.], [230., 60.], [230., 90.], [200., 90.]]}
    raw = np.array([[[10, y, 47, y]] for y in (52, 63, 74)], np.int32)
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *args, **kwargs: raw)
    audit = {}
    assert native_radius_leader_segments(gray, row, limit=2, audit=audit) == []
    assert audit["status"] == "native_segment_budget_exhausted"
    assert audit["observed_native_segment_count"] == 3
    assert audit["search_complete"] is False and len(audit["diagnostic_only_segments"]) == 2
