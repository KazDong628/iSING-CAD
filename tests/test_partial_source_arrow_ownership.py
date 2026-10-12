"""A partial seed search retains ink, but cannot invent exclusive ownership."""
from copy import deepcopy

import cv2
import numpy as np

from contour_agent.constraint_binding import verify_source_arrow_proposal
from contour_agent.source_arrow_localization import resolve_source_arrow_ownership
from test_source_arrow_ownership import image, record


def partial_claim(gray, row):
    boundary = np.c_[np.full(11, 100.), np.linspace(70., 80., 11)]
    evidence = verify_source_arrow_proposal(gray, row, {
        "record_id": row["id"], "tip_px": [100., 75.], "shaft_px": [229., 75.]}, boundary, 4.)
    assert evidence is not None and evidence["shaft_evidence"]["verified"]
    return {**evidence, "record_id": row["id"],
            "target_candidates": [{"entity_id": "target", "entity_type": "ARC", "tip_gap_px": 0.}],
            "source_candidate_search": {"search_complete": False, "status": "inner_search_incomplete",
                                        "uninspected_seed_count": 3, "uniqueness_established": False}}


def test_partial_positive_with_original_glyph_and_shaft_survives_complete_label_audit():
    gray = image(); row = record(); claim = partial_claim(gray, row)
    original = deepcopy(claim)
    kept, rejected, audit = resolve_source_arrow_ownership(gray, [row], [claim])
    assert len(kept) == 1 and not rejected
    proof = audit["partial_search_label_ownership_rechecks"][0]
    assert proof["search_complete"] and proof["checked_record_ids"] == [row["id"]]
    assert kept[0]["source_candidate_search"]["search_complete"] is False
    assert claim == original


def test_neighboring_label_without_generated_claim_is_checked_on_same_real_arrow():
    gray = image(); row = record()
    cv2.line(gray, (229, 75), (319, 75), 0, 2)
    cv2.putText(gray, "R65", (323, 70), cv2.FONT_HERSHEY_SIMPLEX, .55, 0, 1, cv2.LINE_AA)
    other = record("unsearched", ((320, 50), (390, 50), (390, 90), (320, 90)))
    # An OCR value withheld for semantic reread still names visible source
    # text; it must remain a possible owner of this physical arrow.
    other["source_text_admission"] = {"status": "withheld"}
    claim = partial_claim(gray, row)
    kept, rejected, audit = resolve_source_arrow_ownership(gray, [row, other], [claim])
    assert not kept
    proof = audit["partial_search_label_ownership_rechecks"][0]
    assert proof["search_complete"] and proof["verified_competing_record_ids"] == [other["id"]]
    assert audit["shared_arrow_conflicts"][0]["status"] == "source_ownership_ambiguous"
    assert {item["record_id"] for item in rejected} == {row["id"], other["id"]}


def test_a_neighboring_ocr_box_or_glyph_without_full_shaft_is_not_a_competing_owner():
    gray = image(); row = record()
    cv2.putText(gray, "R65", (323, 70), cv2.FONT_HERSHEY_SIMPLEX, .55, 0, 1, cv2.LINE_AA)
    other = record("unsearched", ((320, 50), (390, 50), (390, 90), (320, 90)))
    kept, rejected, audit = resolve_source_arrow_ownership(gray, [row, other], [partial_claim(gray, row)])
    assert len(kept) == 1 and not rejected
    proof = audit["partial_search_label_ownership_rechecks"][0]
    assert proof["glyph_adjacent_record_ids"] == [other["id"]]
    assert proof["verified_competing_record_ids"] == []


def test_partial_label_audit_cannot_be_skipped_by_an_excessive_inventory():
    gray = image(); row = record(); claim = partial_claim(gray, row)
    records = [row]+[record(f"r{index}") for index in range(256)]
    kept, rejected, audit = resolve_source_arrow_ownership(gray, records, [claim])
    assert not kept
    assert rejected[0]["reason"] == "partial_source_search_ownership_not_rechecked"
    assert not audit["partial_search_label_ownership_rechecks"][0]["search_complete"]


def test_nominal_values_do_not_resolve_an_unsearched_label_conflict():
    gray = image(); row = record()
    cv2.line(gray, (229, 75), (319, 75), 0, 2)
    cv2.putText(gray, "R65", (323, 70), cv2.FONT_HERSHEY_SIMPLEX, .55, 0, 1, cv2.LINE_AA)
    other = record("unsearched", ((320, 50), (390, 50), (390, 90), (320, 90)))
    claim = partial_claim(gray, row)
    before = resolve_source_arrow_ownership(gray, [row, other], [claim])
    row["parsed"]["nominal"] = 1.e6; other["parsed"]["nominal"] = .001
    after = resolve_source_arrow_ownership(gray, [row, other], [claim])
    assert before == after and not after[0]


def test_tip_refinement_keeps_partial_search_provenance_and_owner_recheck(monkeypatch):
    import contour_agent.constraint_binding as binding
    import contour_agent.source_arrow_localization as localization
    gray = image(); row = record(); claim = partial_claim(gray, row)
    monkeypatch.setattr(localization, "native_radius_leader_segments", lambda *args, **kwargs: [])
    monkeypatch.setattr(localization, "localize_ocr_radius_arrows",
                        lambda *args, **kwargs: ([deepcopy(claim)], {"record_id": row["id"]}))
    recovered = deepcopy(claim); recovered.pop("source_candidate_search")
    recovered["method"] = "source_ocr_edge_arrow_with_full_shaft_locally_verified"
    monkeypatch.setattr(localization, "recover_source_attached_radius_arrow",
                        lambda *args, **kwargs: deepcopy(recovered))
    graph = {"entities": [{"id": "target", "type": "LINE", "start": [100., 70.], "end": [100., 80.]}]}
    audit = {}
    owned = binding._radius_source_observations(gray, [row], graph, lambda p: np.asarray(p), [], 4.,
                                                ownership_audit=audit)
    assert owned and all(item["source_candidate_search"]["search_complete"] is False for item in owned)
    assert audit["partial_search_label_ownership_rechecks"]
