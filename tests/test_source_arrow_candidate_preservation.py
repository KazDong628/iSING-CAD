"""Original-ink hypotheses survive local ambiguity until global ownership."""
from copy import deepcopy

import cv2
import numpy as np

import contour_agent.constraint_binding as binding
import contour_agent.source_arrow_localization as localization
from test_source_arrow_ownership import image, observation, record


def two_arrow_source():
    gray = np.full((180, 420), 255, np.uint8)
    for y in (65, 83):
        cv2.line(gray, (100, y), (229, y), 0, 2)
        cv2.fillConvexPoly(gray, np.asarray([[100, y], [116, y-4], [116, y+4]], np.int32), 0)
    cv2.putText(gray, "R40", (233, 82), cv2.FONT_HERSHEY_SIMPLEX, .55, 0, 1, cv2.LINE_AA)
    row = record(box=((230, 45), (300, 45), (300, 100), (230, 100)))
    proposal = {"record_id": row["id"], "shaft_px": [235., 75.], "tip_px": [105., 74.]}
    row["source_arrow_proposals"] = [proposal]
    boundary = np.c_[np.full(31, 100.), np.linspace(60, 90, 31)]
    return gray, row, boundary, proposal


def test_completed_local_search_preserves_both_real_arrow_shafts():
    gray, row, boundary, proposal = two_arrow_source()
    assert binding.verify_source_arrow_proposal(gray, row, proposal, boundary, 4.) is None
    audit = {}
    found = localization.localize_source_arrow_candidates(gray, row, proposal, boundary, 4., audit=audit)
    assert len(found) == audit["physical_arrow_candidate_count"] == 2
    assert audit["search_complete"] and audit["distinct_source_candidates_preserved"]
    assert min(item["arrowhead"]["tip_px"][1] for item in found) < 70.
    assert max(item["arrowhead"]["tip_px"][1] for item in found) > 78.
    for item in found:
        assert item["arrowhead_verified"] and item["shaft_evidence"]["verified"]
        assert localization.source_arrow_label_attachment(gray, row, item)["strong_text_adjacency"]
    # A consumer requiring one unique arrow still abstains, never picks by R.
    assert localization.localize_source_arrow_proposal(gray, row, proposal, boundary, 4.) is None
    changed = deepcopy(row); changed["parsed"]["nominal"] = 99999.
    other = localization.localize_source_arrow_candidates(gray, changed, proposal, boundary, 4.)
    assert [item["arrowhead"] for item in found] == [item["arrowhead"] for item in other]


def test_full_observation_pipeline_preserves_api_local_physical_alternatives():
    gray, row, _, _ = two_arrow_source()
    graph = {"entities": [
        {"id": "upper", "type": "LINE", "start": [100., 60.], "end": [100., 70.]},
        {"id": "lower", "type": "LINE", "start": [100., 78.], "end": [100., 88.]},
    ]}
    audit = {}
    found = binding._radius_source_observations(gray, [row], graph, lambda p: np.asarray(p), [], 4.,
                                               ownership_audit=audit)
    assert found
    assert any(item["arrowhead"]["tip_px"][1] < 70. for item in found)
    assert any(item["arrowhead"]["tip_px"][1] > 78. for item in found)
    assert audit["provider_proposal_source_searches"][0]["physical_arrow_candidate_count"] == 2
    assert all(item["source_text_shaft_attachment"]["strong_text_adjacency"] for item in found)


def test_independent_scope_retains_distinct_proposal_arrows_before_cad_targeting():
    gray, row, _, _ = two_arrow_source()
    # The scope is a deliberately broad immutable source-mask loop. Neither
    # physical arrow disappears simply because local search found two tips.
    boundary = np.array([[20., 10.], [390., 10.], [390., 150.], [20., 150.], [20., 10.]])
    result = localization.observe_source_arrow_scope(gray, row, boundary, 4.)
    verified = [item for item in result["observations"] if item.get("independent_source_verified")]
    assert len(verified) >= 2
    assert all(not item["source_arrow"]["target_conditioned_detection"] for item in verified)


def test_unopposed_real_taper_does_not_prove_a_distant_glyph_owns_it():
    gray = image()
    wide = record(box=((230, 20), (300, 20), (300, 90), (230, 90)))
    weak = observation(tip=(100., 25.), target="wrong")
    weak["method"] = "source_hough_arrow_with_full_shaft_locally_verified"
    # Real physical ink and a box-intersecting ray can belong to a dimension
    # extension. An unopposed claim must still demonstrate glyph attachment.
    cv2.line(gray, (100, 25), (229, 25), 0, 2)
    cv2.fillConvexPoly(gray, np.array([[100, 25], [116, 19], [116, 31]], np.int32), 0)
    assert not localization.source_arrow_label_attachment(gray, wide, weak)["strong_text_adjacency"]
    kept, rejected, _ = localization.resolve_source_arrow_ownership(gray, [wide], [weak])
    assert kept == []
    assert rejected[0]["reason"] == "source_arrow_label_attachment_insufficient"
    actual = observation(target="actual")
    kept, rejected, _ = localization.resolve_source_arrow_ownership(gray, [wide], [weak, actual])
    assert len(kept) == 1 and kept[0]["target_candidates"][0]["entity_id"] == "actual"


def test_api_provenance_and_nominal_cannot_override_missing_glyph_ownership():
    gray = image()
    wide = record(box=((230, 20), (300, 20), (300, 90), (230, 90)))
    weak = observation(tip=(100., 25.))
    weak["model_proposal_used"] = True
    wide["parsed"]["nominal"] = .001
    kept, rejected, _ = localization.resolve_source_arrow_ownership(gray, [wide], [weak])
    assert kept == [] and rejected[0]["reason"] == "source_arrow_label_attachment_insufficient"


def test_incomplete_hypothesis_search_never_promotes_partial_candidates(monkeypatch):
    gray, row, boundary, proposal = two_arrow_source()
    def exhausted(*args, audit=None, **kwargs):
        audit.update(search_complete=False, status="hypothesis_budget_exhausted")
        return [{"proposal": {"shaft_px": [229., 65.], "tip_px": [100., 65.]},
                 "localization": {"tip_shift_px": 1.}}]
    monkeypatch.setattr(localization, "source_arrow_hypotheses", exhausted)
    audit = {}
    assert localization.localize_source_arrow_candidates(gray, row, proposal, boundary, 4., audit=audit) == []
    assert audit["search_complete"] is False
