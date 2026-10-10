"""Missing or wrong API locations must not disable original-source snapping."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from contour_agent.constraint_binding import verify_source_arrow_proposal
from contour_agent.source_arrow_localization import localize_ocr_radius_arrows
from contour_agent import source_arrow_localization as localization


def source():
    gray = np.full((170, 310), 255, np.uint8)
    cv2.line(gray, (100, 75), (140, 75), 0, 1)
    cv2.fillConvexPoly(gray, np.asarray([[100, 75], [108, 71], [108, 79]], np.int32), 0)
    cv2.putText(gray, "R40", (144, 70), cv2.FONT_HERSHEY_SIMPLEX, .4, 0, 1, cv2.LINE_AA)
    record = {"id": "source_radius", "text": "R40", "parsed": {"kind": "radius", "nominal": 40.},
              "box": [[141., 50.], [176., 50.], [176., 90.], [141., 90.]]}
    boundary = np.c_[np.full(11, 100.), np.linspace(70, 80, 11)]
    return gray, record, boundary


def signature(rows):
    return [(row["segment_px"], row["arrowhead"]["tip_px"]) for row in rows]


def test_missing_and_wrong_api_seed_recover_same_independently_verified_arrow():
    gray, row, boundary = source()
    original = deepcopy(row)
    pixels = gray.copy()
    recovered, receipt = localize_ocr_radius_arrows(gray, row, [row], boundary, 5.)
    assert recovered and receipt["accepted_observation_count"] == len(recovered)
    assert receipt["api_proposal_used"] is False and receipt["nominal_used_to_rank"] is False
    wrong = {**row, "source_arrow_proposals": [{"record_id": row["id"],
             "tip_px": [285., 140.], "shaft_px": [250., 120.]}]}
    other, _ = localize_ocr_radius_arrows(gray, wrong, [wrong], boundary, 5.)
    assert signature(recovered) == signature(other)
    for evidence in recovered:
        assert evidence["model_proposal_used"] is False
        assert evidence["source_text_shaft_attachment"]["strong_text_adjacency"] is True
        assert evidence["shaft_evidence"]["verified"] is True
        assert evidence["shaft_evidence"]["supported_fraction"] >= .88
        assert verify_source_arrow_proposal(gray, row, {
            "shaft_px": evidence["segment_px"][0], "tip_px": evidence["segment_px"][1]}, boundary, 5.)
    assert row == original and np.array_equal(pixels, gray)


def test_ocr_nominal_does_not_rank_source_seed_or_arrow():
    gray, row, boundary = source()
    first, _ = localize_ocr_radius_arrows(gray, row, [row], boundary, 5.)
    changed = deepcopy(row)
    changed["parsed"]["nominal"] = 999999.
    second, _ = localize_ocr_radius_arrows(gray, changed, [changed], boundary, 5.)
    assert first and signature(first) == signature(second)


@pytest.mark.parametrize("defect", ["missing_arrow", "broken_shaft", "missing_text", "wrong_box", "blank"])
def test_local_search_never_substitutes_for_missing_source_proof(defect):
    gray, row, boundary = source()
    if defect == "missing_arrow":
        gray[65:85, 95:120] = 255
        cv2.line(gray, (100, 75), (120, 75), 0, 2)
    elif defect == "broken_shaft":
        gray[65:85, 118:139] = 255
    elif defect == "missing_text":
        gray[50:90, 141:176] = 255
    elif defect == "wrong_box":
        row["box"] = [[150., 110.], [215., 110.], [215., 145.], [150., 145.]]
    elif defect == "blank":
        gray[:] = 255
    recovered, receipt = localize_ocr_radius_arrows(gray, row, [row], boundary, 5.)
    assert recovered == [] and receipt["accepted_observation_count"] == 0


@pytest.mark.parametrize("budget", [{"seed_limit": 0}, {"time_limit_seconds": 0.}])
def test_exhausted_search_budget_is_explicit_and_creates_no_certificate(budget):
    gray, row, boundary = source()
    recovered, receipt = localize_ocr_radius_arrows(gray, row, [row], boundary, 5., **budget)
    assert not recovered and receipt["attempted_seed_count"] == 0
    assert receipt["status"] in {"seed_budget_exhausted", "time_budget_exhausted"}
    assert receipt["candidate_seed_count"] > 0


def test_seed_budget_is_bounded_even_when_caller_requests_more():
    gray, row, boundary = source()
    _, receipt = localize_ocr_radius_arrows(gray, row, [row], boundary, 5., seed_limit=9999)
    assert receipt["maximum_seed_count"] == 24 and receipt["attempted_seed_count"] <= 24
    assert receipt["maximum_hypotheses_per_seed"] == 24
    left, top, right, bottom = receipt["seed_roi_px"]
    assert (right-left)*(bottom-top) <= receipt["maximum_roi_pixels"]


@pytest.mark.parametrize("budget_kind", ["seed", "time"])
def test_verified_partial_result_is_not_admitted_with_uninspected_competing_seeds(monkeypatch, budget_kind):
    gray, row, boundary = source()
    verified, original_receipt = localize_ocr_radius_arrows(gray, row, [row], boundary, 5.)
    assert verified and original_receipt["candidate_seed_count"] > 1
    clock = [0.]
    if budget_kind == "time":
        monkeypatch.setattr(localization.time, "monotonic", lambda: clock[0])
    def first_seed_succeeds(*args, **kwargs):
        clock[0] = 9.
        return deepcopy(verified[0])
    monkeypatch.setattr(localization, "localize_source_arrow_proposal", first_seed_succeeds)
    budget = {"seed_limit": 1} if budget_kind == "seed" else {"time_limit_seconds": 8.}
    admitted, receipt = localize_ocr_radius_arrows(gray, row, [row], boundary, 5., **budget)
    assert admitted == []
    assert receipt["status"] == f"{budget_kind}_budget_exhausted"
    assert receipt["attempted_seed_count"] == receipt["verified_observation_count"] == 1
    assert receipt["accepted_observation_count"] == 0 and receipt["uninspected_seed_count"] > 0
    assert receipt["diagnostic_only_observations"][0]["arrowhead_verified"] is True
    assert receipt["acceptance_withheld_reason"] == "uninspected_source_seed_competitors"
