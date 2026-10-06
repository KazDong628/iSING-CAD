"""Physical-arrow ownership tests: neither CAD radius nor GT selects a label."""
from copy import deepcopy

import cv2
import numpy as np

from contour_agent.source_arrow_localization import resolve_source_arrow_ownership
from contour_agent.constraint_binding import radius_binding_coverage


def record(rid="r000", box=((230, 50), (300, 50), (300, 90), (230, 90))):
    return {"id": rid, "text": "R40", "parsed": {"kind": "radius", "nominal": 40.},
            "box": [list(p) for p in box]}


def observation(rid="r000", tip=(100., 75.), direction=(-1., 0.), target="g000"):
    # These inputs already passed the complete source shaft verifier; the
    # resolver cannot invent that certificate from text distance alone.
    return {"record_id": rid, "method": "agent_proposed_source_arrow_locally_verified",
            "arrowhead_verified": True,
            "arrowhead": {"tip_px": list(tip), "direction_px": list(direction), "length_px": 20.},
            "shaft_evidence": {"verified": True}, "source_label_association": {},
            "target_candidates": [{"entity_id": target, "entity_type": "ARC", "tip_gap_px": 0.}]}


def image():
    gray = np.full((180, 420), 255, np.uint8)
    cv2.line(gray, (100, 75), (229, 75), 0, 2)
    cv2.fillConvexPoly(gray, np.array([[100, 75], [116, 69], [116, 81]], np.int32), 0)
    cv2.putText(gray, "R40", (233, 70), cv2.FONT_HERSHEY_SIMPLEX, .55, 0, 1, cv2.LINE_AA)
    return gray


def test_shared_arrow_missing_text_abstains_and_keeps_both_radius_records():
    gray = np.full((180, 420), 255, np.uint8)
    first = record(); second = record("r001")
    original = [observation(), observation("r001")]
    kept, rejected, audit = resolve_source_arrow_ownership(gray, [first, second], original)
    assert kept == []
    assert {r["record_id"] for r in rejected} == {"r000", "r001"}
    assert all(r["reason"] == "shared_source_arrow_ownership_ambiguous" for r in rejected)
    inventory = {"all_records": [first, second], "all_candidates": [],
                 "radius_source_observations": kept, "radius_source_rejections": rejected}
    coverage = radius_binding_coverage(inventory, {"entities": []})
    assert coverage["recognized_count"] == coverage["unresolved_count"] == 2
    assert coverage["bound_count"] == 0
    assert audit["shared_arrow_conflicts"][0]["selected_record_id"] is None
    assert original[0] == observation()  # Never rewrite the input evidence.


def test_two_real_texts_on_same_shaft_are_not_forced_to_one_label():
    gray = image()
    cv2.putText(gray, "R65", (323, 70), cv2.FONT_HERSHEY_SIMPLEX, .55, 0, 1, cv2.LINE_AA)
    second = record("r001", ((320, 50), (390, 50), (390, 90), (320, 90)))
    kept, rejected, audit = resolve_source_arrow_ownership(gray, [record(), second],
                                                          [observation(), observation("r001")])
    assert not kept and len(rejected) == 2
    assert audit["shared_arrow_conflicts"][0]["status"] == "source_ownership_ambiguous"
    assert all(a["strong_text_adjacency"] for a in audit["shared_arrow_conflicts"][0]["attachments"])


def test_shared_arrow_owner_requires_complete_shaft_text_and_poor_competing_attachment():
    gray = image()
    # A diagonal-label bounding rectangle can intersect a different leader
    # even when its actual glyphs are far from that leader's shaft.
    cv2.putText(gray, "R114", (215, 137), cv2.FONT_HERSHEY_SIMPLEX, .9, 0, 2, cv2.LINE_AA)
    second = record("r001", ((200, 40), (310, 40), (310, 145), (200, 145)))
    kept, rejected, audit = resolve_source_arrow_ownership(gray, [record(), second],
                                                          [observation(), observation("r001")])
    assert len(kept) == len(rejected) == 1
    assert kept[0]["record_id"] == "r000"
    assert rejected[0]["reason"] == "source_arrow_claim_owned_by_other_label"
    conflict = audit["shared_arrow_conflicts"][0]
    assert conflict["observed_normalized_margin"] > conflict["required_normalized_margin"]
    broken = observation(); broken["shaft_evidence"]["verified"] = False
    assert not resolve_source_arrow_ownership(gray, [record()], [broken])[0]


def test_true_distinct_arrows_and_objects_of_one_annotation_remain_visible():
    gray = image()
    original = [observation(), observation(tip=(100., 65.), target="g001")]
    kept, rejected, audit = resolve_source_arrow_ownership(gray, [record()], original)
    assert len(kept) == audit["physical_arrow_count"] == 2
    assert not rejected
    assert {r["target_candidates"][0]["entity_id"] for r in kept} == {"g000", "g001"}


def test_same_object_duplicate_detections_are_one_arrow_without_radius_ranking():
    gray = image()
    original = [observation(), observation(tip=(102., 76.))]
    changed = deepcopy(original); changed[0]["nominal"] = 999999.
    kept, rejected, audit = resolve_source_arrow_ownership(gray, [record()], original)
    assert len(kept) == audit["physical_arrow_count"] == 1 and not rejected
    assert kept[0]["source_arrow_ownership"]["duplicate_observation_count"] == 1
    other = record(); other["parsed"]["nominal"] = 0.0001
    selected = resolve_source_arrow_ownership(gray, [other], changed)[0]
    assert kept[0]["arrowhead"] == selected[0]["arrowhead"]


def test_neighboring_parallel_arrowheads_are_not_merged_by_tip_proximity():
    gray = image()
    original = [observation(), observation(tip=(101., 81.), target="g001")]
    kept, _, audit = resolve_source_arrow_ownership(gray, [record()], original)
    assert len(kept) == audit["physical_arrow_count"] == 2


def test_duplicate_physical_arrow_retains_disagreement_about_target_objects():
    gray = image()
    kept, rejected, audit = resolve_source_arrow_ownership(gray, [record()],
        [observation(), observation(tip=(102., 76.), target="g001")])
    assert len(kept) == audit["physical_arrow_count"] == 1 and not rejected
    assert {target["entity_id"] for target in kept[0]["target_candidates"]} == {"g000", "g001"}
    assert kept[0]["source_arrow_ownership"]["duplicate_target_ambiguity_preserved"]


def test_unavailable_source_pixels_cannot_establish_text_direction_or_ownership():
    kept, rejected, _ = resolve_source_arrow_ownership(None, [record()], [observation()])
    assert not kept
    assert rejected[0]["reason"] == "complete_source_shaft_not_verified"


def test_repetitive_hatch_claim_far_from_glyphs_does_not_hide_actual_leader():
    gray = image()
    actual = observation()
    hatch = observation(tip=(100., 25.), target="g001")
    hatch["source_label_association"] = {"parallel_family": {"pitch_px": 30.}}
    wide = record(box=((230, 20), (300, 20), (300, 90), (230, 90)))
    kept, rejected, _ = resolve_source_arrow_ownership(gray, [wide], [actual, hatch])
    assert len(kept) == 1 and kept[0]["target_candidates"][0]["entity_id"] == "g000"
    assert rejected[0]["reason"] == "repetitive_shaft_without_source_glyph_adjacency"
    # Parallel nearby background alone does not invalidate the genuine leader.
    actual["source_label_association"] = hatch["source_label_association"]
    assert len(resolve_source_arrow_ownership(gray, [record()], [actual])[0]) == 1


def test_reversed_shaft_and_arrow_inside_text_remain_review_items():
    gray = image()
    reversed_arrow = observation(direction=(1., 0.))
    inside = observation(tip=(248., 70.))
    kept, rejected, _ = resolve_source_arrow_ownership(gray, [record()], [reversed_arrow, inside])
    assert not kept
    assert {row["reason"] for row in rejected} == {
        "source_arrow_direction_does_not_reach_label", "source_arrow_tip_inside_text_requires_review"}


def test_different_arrows_may_bind_different_labels_to_the_same_object():
    gray = image()
    second = record("r001", ((230, 105), (300, 105), (300, 145), (230, 145)))
    cv2.putText(gray, "R40", (233, 125), cv2.FONT_HERSHEY_SIMPLEX, .55, 0, 1, cv2.LINE_AA)
    kept, rejected, _ = resolve_source_arrow_ownership(gray, [record(), second],
                                                       [observation(), observation("r001", tip=(100., 130.))])
    assert len(kept) == 2 and not rejected
    assert {r["record_id"] for r in kept} == {"r000", "r001"}


def test_observation_pipeline_does_not_erase_neighboring_arrows_before_global_audit(monkeypatch):
    # Stub only the already tested pixel verifier to isolate the observation
    # collection contract. Nearby tips on distinct parallel shafts must both
    # reach the global resolver, including the detector/proposal entry path.
    import contour_agent.constraint_binding as binding
    import contour_agent.source_arrow_localization as localization
    boundary = np.array([[100., 75.], [101., 81.]])
    monkeypatch.setattr(binding, "_samples", lambda entity, transform: boundary)
    monkeypatch.setattr(localization, "native_radius_leader_segments", lambda *args, **kwargs: [])
    def arrow(gray, target, direction, *args):
        return {"tip_px": list(target), "direction_px": list(direction), "length_px": 20., "score": 1.}
    monkeypatch.setattr(binding, "_arrowhead_evidence", arrow)
    def verified(gray, row, proposal, *args):
        tip = np.asarray(proposal["tip_px"], float)
        unit = tip-np.asarray(proposal["shaft_px"], float); unit /= np.linalg.norm(unit)
        return {"arrowhead": arrow(gray, tip, unit), "arrowhead_verified": True,
                "shaft_evidence": {"verified": True}, "contour_visibility": {"verified": True},
                "label_gap_px": 1., "label_ray_intersection_gap_px": 1.,
                "crossing_source_contour": False, "crossing_admission": "source_only_fixture"}
    monkeypatch.setattr(binding, "verify_source_arrow_proposal", verified)
    segments = [np.array([[229., 75.], [100., 75.]]), np.array([[229., 81.], [101., 81.]])]
    audit = {}
    found = binding._radius_source_observations(image(), [record()],
        {"entities": [{"id": "g000", "type": "ARC"}]}, lambda points: points,
        segments, 4., ownership_audit=audit)
    assert audit["input_observation_count"] == audit["physical_arrow_count"] == len(found) == 2
