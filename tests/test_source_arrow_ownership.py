"""Physical-arrow ownership tests: neither CAD radius nor GT selects a label."""
from copy import deepcopy

import cv2
import numpy as np

from contour_agent.source_arrow_localization import (
    resolve_source_arrow_ownership, same_original_ink_arrow_shaft,
)
from contour_agent.constraint_binding import (
    _admit_joint_radius_conflicts, _radius_joint_conflict_preflight,
    _radius_line_joint_preflight, _radius_one_arrow_incompatible_claim_preflight,
    _radius_ocr_owned_target_preflight,
    _radius_joint_conflict_resolution, radius_binding_coverage,
)


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


def test_wide_original_ink_shaft_proves_duplicate_edge_localizations():
    gray = np.full((120, 180), 255, np.uint8)
    cv2.rectangle(gray, (35, 49), (110, 56), 0, -1)
    first = observation(tip=(105., 50.), direction=(1., 0.))
    second = observation(tip=(105., 55.), direction=(1., 0.))
    for row in (first, second):
        row["arrowhead"].update(length_px=32., cross_section_widths_px=[8., 10.])
    assert same_original_ink_arrow_shaft(gray, first, second)
    # Proximity and measured widths alone cannot join two white-separated
    # strokes, even when their tips aim at the same contour object.
    separate = np.full_like(gray, 255)
    cv2.line(separate, (35, 50), (110, 50), 0, 1)
    cv2.line(separate, (35, 55), (110, 55), 0, 1)
    assert not same_original_ink_arrow_shaft(separate, first, second)


def test_one_pixel_antialias_gap_on_common_shaft_does_not_split_arrow():
    gray = np.full((120, 180), 255, np.uint8)
    cv2.rectangle(gray, (35, 49), (110, 56), 0, -1)
    # An isolated light raster sample must not act as a longitudinal white
    # slit. Source ink still connects around it within the one-pixel band.
    gray[53, 94] = 255
    first = observation(tip=(105., 50.), direction=(1., 0.))
    second = observation(tip=(105., 55.), direction=(1., 0.))
    for row in (first, second):
        row["arrowhead"].update(length_px=32., cross_section_widths_px=[8., 10.])
    assert same_original_ink_arrow_shaft(gray, first, second)


def test_two_edge_detections_of_one_taper_join_at_measured_arrow_body():
    gray = np.full((120, 180), 255, np.uint8)
    cv2.rectangle(gray, (35, 50), (90, 55), 0, -1)
    cv2.fillConvexPoly(gray, np.array([[90, 50], [105, 52],
                                       [105, 53], [90, 55]], np.int32), 0)
    first = observation(tip=(105., 50.), direction=(1., 0.))
    second = observation(tip=(105., 55.), direction=(1., 0.))
    for row in (first, second):
        row["arrowhead"].update(length_px=32., cross_section_widths_px=[8., 10.])
    # At .30 of arrow length the taper has not widened to span the two edge
    # hypotheses, even with axial +/-1 px support. That is not two arrows.
    near_tip = np.linspace((105.-.30*32, 50.), (105.-.30*32, 55.), 9)
    band = np.rint(near_tip[None, :, :] +
                   np.asarray([-1., 0., 1.])[:, None, None]*np.array([1., 0.])).astype(int)
    assert np.mean(np.any(gray[band[:, :, 1], band[:, :, 0]] < 170, axis=0)) < .9
    assert same_original_ink_arrow_shaft(gray, first, second)


def test_parallel_arrows_with_one_crossing_hatch_remain_distinct():
    gray = np.full((120, 180), 255, np.uint8)
    cv2.line(gray, (35, 50), (110, 50), 0, 1)
    cv2.line(gray, (35, 55), (110, 55), 0, 1)
    # A single diagonal crossing can join the strokes locally, but does not
    # establish one continuous common shaft along multiple axial sections.
    cv2.line(gray, (91, 50), (97, 55), 0, 1)
    first = observation(tip=(105., 50.), direction=(1., 0.))
    second = observation(tip=(105., 55.), direction=(1., 0.))
    for row in (first, second):
        row["arrowhead"].update(length_px=32., cross_section_widths_px=[8., 10.])
    assert not same_original_ink_arrow_shaft(gray, first, second)


def test_global_arrow_grouping_uses_full_ink_bridge_for_wide_shaft_edges():
    gray = np.full((180, 420), 255, np.uint8)
    cv2.rectangle(gray, (100, 70), (229, 78), 0, -1)
    cv2.putText(gray, "R40", (233, 70), cv2.FONT_HERSHEY_SIMPLEX, .55, 0, 1, cv2.LINE_AA)
    first = observation(tip=(100., 72.), target="arc")
    second = observation(tip=(100., 77.), target="line")
    for row in (first, second):
        row["arrowhead"].update(length_px=32., cross_section_widths_px=[8., 10.])
    assert same_original_ink_arrow_shaft(gray, first, second)
    kept, rejected, audit = resolve_source_arrow_ownership(gray, [record()], [first, second])
    assert len(kept) == audit["physical_arrow_count"] == 1
    assert not rejected
    assert {target["entity_id"] for target in kept[0]["target_candidates"]} == {
        "arc", "line"}
    assert kept[0]["source_arrow_ownership"]["duplicate_target_ambiguity_preserved"]

    separate = np.full_like(gray, 255)
    cv2.line(separate, (100, 72), (229, 72), 0, 1)
    cv2.line(separate, (100, 77), (229, 77), 0, 1)
    cv2.putText(separate, "R40", (233, 70), cv2.FONT_HERSHEY_SIMPLEX, .55, 0, 1, cv2.LINE_AA)
    assert not same_original_ink_arrow_shaft(separate, first, second)
    distinct, _, audit = resolve_source_arrow_ownership(separate, [record()], [first, second])
    assert len(distinct) == audit["physical_arrow_count"] == 2


def test_wide_shaft_never_joins_distinct_arrow_directions():
    gray = np.zeros((120, 180), np.uint8)
    first = observation(tip=(105., 50.), direction=(1., 0.))
    second = observation(tip=(105., 55.), direction=(0., 1.))
    for row in (first, second):
        row["arrowhead"].update(length_px=32., cross_section_widths_px=[12.])
    assert not same_original_ink_arrow_shaft(gray, first, second)


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


def test_hough_body_endpoint_may_be_farther_than_tip_band_before_pixel_localization(monkeypatch):
    """Only the freshly verified tip, not a short Hough shaft end, targets CAD."""
    import contour_agent.constraint_binding as binding
    import contour_agent.source_arrow_localization as localization
    boundary = np.array([[100., 70.], [100., 80.]])
    monkeypatch.setattr(binding, "_samples", lambda entity, transform: boundary)
    monkeypatch.setattr(localization, "native_radius_leader_segments", lambda *args, **kwargs: [])
    def arrow(gray, target, direction, *args):
        return {"tip_px": [100., 75.], "direction_px": list(direction),
                "length_px": 60., "score": 1., "verified": True}
    monkeypatch.setattr(binding, "_arrowhead_evidence", arrow)
    def verified(gray, row, proposal, *args):
        unit = np.asarray(proposal["tip_px"], float)-np.asarray(proposal["shaft_px"], float)
        unit /= np.linalg.norm(unit)
        return {"arrowhead": arrow(gray, proposal["tip_px"], unit), "arrowhead_verified": True,
                "shaft_evidence": {"verified": True}, "contour_visibility": {"verified": True},
                "label_gap_px": 1., "label_ray_intersection_gap_px": 1.,
                "crossing_source_contour": False, "crossing_admission": "source_only_fixture"}
    monkeypatch.setattr(binding, "verify_source_arrow_proposal", verified)
    segment = np.array([[229., 75.], [130., 75.]])
    assert min(np.linalg.norm(boundary-segment[1], axis=1)) > 10.
    found = binding._radius_source_observations(image(), [record()],
        {"entities": [{"id": "g000", "type": "ARC"}]}, lambda points: points, [segment], 4.)
    assert len(found) == 1
    assert found[0]["target_candidates"][0]["entity_id"] == "g000"
    assert found[0]["target_candidates"][0]["tip_gap_px"] <= 1.


def _radius_claim(record_id, target, nominal, *, reliable=True, feasible=True, unique=True):
    return {"kind": "radius", "record_id": record_id, "entities": [target],
            "value": nominal, "local_reliable": reliable,
            "evidence": {"uniquely_supported_leader": unique,
                         "leader": {"arrowhead_verified": True, "radial_alignment": .99},
                         "whole_primitive_radius": {"passed": feasible},
                         "source_text": {"symbol_confusion": False}}}


def _joint_observation(record_id, *targets):
    return {"record_id": record_id, "arrowhead_verified": True,
            "shaft_evidence": {"verified": True},
            "source_text_shaft_attachment": {"strong_text_adjacency": True},
            "source_arrow_ownership": {"status": "globally_unique_source_claim"},
            "target_candidates": [{"entity_id": target, "entity_type": "ARC",
                                   "tip_gap_px": 3.} for target in targets]}


def test_shared_joint_radius_resolves_only_with_independent_incompatible_label():
    primary = _radius_claim("r176", "g016", 176., reliable=False, unique=False)
    neighboring = _radius_claim("r040", "g017", 40.)
    observations = [_joint_observation("r176", "g016", "g017"),
                    _joint_observation("r176", "g016")]
    resolution = _radius_joint_conflict_resolution(primary, observations,
                                                   [primary, neighboring])
    assert resolution["entity_id"] == "g016"
    assert resolution["independent_incompatible_claims"] == [
        {"entity_id": "g017", "record_ids": ["r040"]}]
    assert resolution["ground_truth_used"] is False
    assert resolution["nominal_used_to_resolve_constraint_conflict"] is True
    assert resolution["nominal_used_to_rank_arrow_geometry"] is False
    # The ordinary 5px leader-margin gate may not run interval feasibility.
    # Source conflict preflight can safely select the unique observed arrow
    # before the unchanged fixed-radius feasibility check runs.
    pending = deepcopy(primary)
    pending["evidence"].pop("whole_primitive_radius")
    preflight = _radius_joint_conflict_preflight(pending, observations,
                                                 [pending, neighboring])
    assert preflight is not None
    assert preflight[1] is observations[1]
    assert _radius_joint_conflict_resolution(pending, observations,
                                             [pending, neighboring]) is None
    # An unverified competing claim, a nominal-equal claim, or an infeasible
    # fixed-radius interval cannot turn ambiguous arrows into an exact radius.
    for changed in ({"local_reliable": False}, {"value": 176.},
                    {"evidence": {**neighboring["evidence"],
                                  "whole_primitive_radius": {"passed": False}}}):
        rival = {**neighboring, **changed}
        assert _radius_joint_conflict_resolution(primary, observations,
                                                 [primary, rival]) is None
    infeasible = _radius_claim("r176", "g016", 176., reliable=False, feasible=False)
    assert _radius_joint_conflict_resolution(infeasible, observations,
                                             [infeasible, neighboring]) is None


def test_second_unique_radius_arrow_target_is_not_erased_by_conflict_propagation():
    primary = _radius_claim("r176", "g016", 176., reliable=False)
    neighboring = _radius_claim("r040", "g017", 40.)
    two_real_targets = [_joint_observation("r176", "g016"),
                        _joint_observation("r176", "g017")]
    assert _radius_joint_conflict_resolution(primary, two_real_targets,
                                             [primary, neighboring]) is None
    # No unique arrow to the proposed primitive also requires abstention.
    both_ambiguous = [_joint_observation("r176", "g016", "g017"),
                      _joint_observation("r176", "g016", "g017")]
    assert _radius_joint_conflict_resolution(primary, both_ambiguous,
                                             [primary, neighboring]) is None
    unresolved_fillet = _joint_observation("r176", "g016", "g017")
    unresolved_fillet["target_candidates"].append(
        {"entity_id": "g018", "entity_type": "LINE", "tip_gap_px": 2.})
    assert _radius_joint_conflict_resolution(primary,
        [unresolved_fillet, _joint_observation("r176", "g016")],
        [primary, neighboring]) is None


def test_shared_joint_preflight_needs_independent_source_arrow_chain():
    primary = _radius_claim("r176", "g016", 176., reliable=False, unique=False)
    neighboring = _radius_claim("r040", "g017", 40.)
    observations = [_joint_observation("r176", "g016", "g017"),
                    _joint_observation("r176", "g016")]
    for path, value in (("shaft_evidence", {"verified": False}),
                        ("source_text_shaft_attachment", {"strong_text_adjacency": False}),
                        ("source_arrow_ownership", {"status": "competing_source_claim"})):
        unsupported = deepcopy(observations)
        unsupported[1][path] = value
        assert _radius_joint_conflict_preflight(primary, unsupported,
                                                [primary, neighboring]) is None
    weak = deepcopy(primary)
    weak["evidence"]["leader"]["radial_alignment"] = .87
    assert _radius_joint_conflict_preflight(weak, observations,
                                            [weak, neighboring]) is None
    weak = deepcopy(primary)
    weak["evidence"]["source_text"]["symbol_confusion"] = True
    assert _radius_joint_conflict_preflight(weak, observations,
                                            [weak, neighboring]) is None


def test_joint_radius_runs_unchanged_interval_gate_only_after_source_preflight(monkeypatch):
    import contour_agent.constraint_binding as binding
    primary = _radius_claim("r176", "g016", 176., reliable=False, unique=False)
    primary["evidence"].pop("whole_primitive_radius")
    primary["evidence"]["multiple_directed_source_targets_require_review"] = True
    rival = _radius_claim("r040", "g017", 40.)
    observations = [_joint_observation("r176", "g016", "g017"),
                    _joint_observation("r176", "g016")]
    seen = []
    def fixed_radius(entity, nominal, leader, model, graph, transform, band):
        seen.append((entity["id"], nominal, leader))
        return {"passed": True, "source_deviation_budget_px": 3.79}
    monkeypatch.setattr(binding, "_radius_primitive_feasibility", fixed_radius)
    graph = {"entities": [{"id": "g016", "type": "ARC"}]}
    _admit_joint_radius_conflicts([primary, rival], observations, {}, graph, None, 4.)
    assert seen == [("g016", 176., observations[1])]
    assert primary["local_reliable"]
    assert primary["evidence"]["whole_primitive_radius"]["source_deviation_budget_px"] == 3.79
    # If the same original-interval gate fails, the source conflict alone is
    # insufficient to admit an exact radius.
    primary["local_reliable"] = False
    primary["evidence"].pop("whole_primitive_radius")
    primary["evidence"].pop("source_joint_radius_conflict_resolution")
    primary["evidence"]["multiple_directed_source_targets_require_review"] = True
    monkeypatch.setattr(binding, "_radius_primitive_feasibility",
                        lambda *args: {"passed": False, "status": "requires_topology_repartition"})
    _admit_joint_radius_conflicts([primary, rival], observations, {}, graph, None, 4.)
    assert not primary["local_reliable"]
    assert primary["evidence"]["multiple_directed_source_targets_require_review"]
    # A second unique source tip aimed at the rival must not even invoke the
    # costly interval feasibility search.
    primary["evidence"].pop("whole_primitive_radius")
    monkeypatch.setattr(binding, "_radius_primitive_feasibility",
                        lambda *args: (_ for _ in ()).throw(AssertionError("must abstain")))
    two_real_arrows = [_joint_observation("r176", "g016"),
                       _joint_observation("r176", "g017")]
    _admit_joint_radius_conflicts([primary, rival], two_real_arrows, {}, graph, None, 4.)
    assert not primary["local_reliable"]


def test_radius_arrow_at_line_arc_joint_keeps_verified_arc_despite_coarse_fit(monkeypatch):
    import contour_agent.constraint_binding as binding

    primary = _radius_claim("r003", "arc", 3., reliable=False, unique=False)
    primary["evidence"].pop("whole_primitive_radius")
    graph = {"entities": [
        {"id": "line", "type": "LINE", "start_node": "a", "end_node": "b",
         "start": [-12., 3.], "end": [0., 3.]},
        {"id": "arc", "type": "ARC", "start_node": "b", "end_node": "c",
         "start": [0., 3.], "end": [3., 0.], "center": [0., 0.]},
    ]}
    arc_only = _joint_observation("r003", "arc")
    at_join = deepcopy(arc_only)
    at_join["target_candidates"].append(
        {"entity_id": "line", "entity_type": "LINE", "tip_gap_px": 4.})
    observations = [arc_only, at_join]
    preflight = _radius_line_joint_preflight(primary, observations, graph)
    assert preflight is not None
    assert preflight[0]["neighboring_line_id"] == "line"
    assert preflight[0]["ground_truth_used"] is False
    assert preflight[1] is arc_only

    calls = []
    monkeypatch.setattr(binding, "_radius_primitive_feasibility",
        lambda entity, nominal, *args: calls.append((entity["id"], nominal)) or {"passed": True})
    _admit_joint_radius_conflicts([primary], observations, {}, graph, None, 4.)
    assert calls == [("arc", 3.)]
    assert primary["local_reliable"]

    # A separate LINE cannot be excluded as an adjacent source object.
    disconnected = deepcopy(graph)
    disconnected["entities"][0]["end_node"] = "other"
    assert _radius_line_joint_preflight(primary, observations, disconnected) is None
    # The fitted angle may be wrong before the fixed-R numerical solve. The
    # full source arrows and shared topology still identify the ARC target.
    corner = deepcopy(graph)
    corner["entities"][0]["start"] = [0., -12.]
    assert _radius_line_joint_preflight(primary, observations, corner) is not None
    both_ambiguous = [at_join, deepcopy(at_join)]
    assert _radius_line_joint_preflight(primary, both_ambiguous, graph) is None
    weak = deepcopy(observations)
    weak[0]["shaft_evidence"]["verified"] = False
    assert _radius_line_joint_preflight(primary, weak, graph) is None


def test_single_arrow_two_adjacent_arcs_needs_other_radius_source_claim(monkeypatch):
    import contour_agent.constraint_binding as binding

    primary = _radius_claim("r176", "left", 176., reliable=False, unique=False)
    primary["evidence"].pop("whole_primitive_radius")
    rival = _radius_claim("r040", "right", 40.)
    graph = {"entities": [
        {"id": "left", "type": "ARC", "start_node": "a", "end_node": "b"},
        {"id": "right", "type": "ARC", "start_node": "b", "end_node": "c"},
    ]}
    observation = _joint_observation("r176", "left", "right")
    proof = _radius_one_arrow_incompatible_claim_preflight(
        primary, [observation], [primary, rival], graph)
    assert proof is not None
    assert proof[0]["independent_incompatible_claims"] == [
        {"entity_id": "right", "record_ids": ["r040"]}]
    assert proof[0]["nominal_used_to_rank_arrow_geometry"] is False
    assert proof[1] is observation

    calls = []
    monkeypatch.setattr(binding, "_radius_primitive_feasibility",
        lambda entity, nominal, *args: calls.append((entity["id"], nominal)) or {"passed": True})
    _admit_joint_radius_conflicts([primary, rival], [observation], {}, graph, None, 4.)
    assert calls == [("left", 176.)]
    assert primary["local_reliable"]
    assert primary["evidence"]["source_joint_radius_conflict_resolution"][
        "fixed_nominal_source_interval_feasible"]

    primary["local_reliable"] = False
    primary["evidence"].pop("whole_primitive_radius")
    primary["evidence"].pop("source_joint_radius_conflict_resolution")
    monkeypatch.setattr(binding, "_radius_primitive_feasibility",
                        lambda *args: {"passed": False})
    _admit_joint_radius_conflicts([primary, rival], [observation], {}, graph, None, 4.)
    assert not primary["local_reliable"]

    for changed in ({"local_reliable": False}, {"value": 176.},
                    {"evidence": {**rival["evidence"],
                                  "whole_primitive_radius": {"passed": False}}}):
        unsupported = {**rival, **changed}
        assert _radius_one_arrow_incompatible_claim_preflight(
            primary, [observation], [primary, unsupported], graph) is None
    unrelated = deepcopy(graph)
    unrelated["entities"][1]["start_node"] = "other"
    assert _radius_one_arrow_incompatible_claim_preflight(
        primary, [observation], [primary, rival], unrelated) is None
    weak = deepcopy(observation)
    weak["source_text_shaft_attachment"]["strong_text_adjacency"] = False
    assert _radius_one_arrow_incompatible_claim_preflight(
        primary, [weak], [primary, rival], graph) is None


def test_one_strong_glyph_attached_arrow_excludes_only_occupied_weak_target(monkeypatch):
    import contour_agent.constraint_binding as binding

    primary = _radius_claim("first", "near", 65., reliable=False, unique=False)
    primary["evidence"].pop("whole_primitive_radius")
    rival = _radius_claim("second", "far", 35.)
    strong = _joint_observation("first", "near")
    strong["source_text_shaft_attachment"].update(checked=True)
    weak = _joint_observation("first", "far")
    weak["source_text_shaft_attachment"].update(checked=True, strong_text_adjacency=False)
    observations = [strong, weak]
    proof = _radius_ocr_owned_target_preflight(primary, observations, [primary, rival])
    assert proof is not None
    assert proof[1] is strong
    assert proof[0]["independent_incompatible_claims"] == [
        {"entity_id": "far", "record_ids": ["second"]}]
    assert proof[0]["ground_truth_used"] is False

    graph = {"entities": [{"id": "near", "type": "ARC"}]}
    seen = []
    monkeypatch.setattr(binding, "_radius_primitive_feasibility",
        lambda entity, nominal, leader, *args:
            seen.append((entity["id"], nominal, leader)) or {"passed": True})
    _admit_joint_radius_conflicts([primary, rival], observations, {}, graph, None, 4.)
    assert seen == [("near", 65., strong)]
    assert primary["local_reliable"]

    # An unoccupied rival, a second strongly attached arrow, a weak-only
    # pair, or a strong arrow at a LINE has no unique radius ownership proof.
    unclaimed = {**rival, "local_reliable": False}
    assert _radius_ocr_owned_target_preflight(primary, observations,
                                               [primary, unclaimed]) is None
    both_strong = deepcopy(observations)
    both_strong[1]["source_text_shaft_attachment"]["strong_text_adjacency"] = True
    assert _radius_ocr_owned_target_preflight(primary, both_strong,
                                               [primary, rival]) is None
    weak_only = deepcopy(observations)
    weak_only[0]["source_text_shaft_attachment"]["strong_text_adjacency"] = False
    assert _radius_ocr_owned_target_preflight(primary, weak_only,
                                               [primary, rival]) is None
    line_tip = deepcopy(observations)
    line_tip[0]["target_candidates"][0]["entity_type"] = "LINE"
    assert _radius_ocr_owned_target_preflight(primary, line_tip,
                                               [primary, rival]) is None
    equal_nominal = {**rival, "value": 65.}
    assert _radius_ocr_owned_target_preflight(primary, observations,
                                               [primary, equal_nominal]) is None

    # Original-source fixed-radius feasibility is mandatory even after the
    # source arrow and competing-label exclusions succeed.
    primary["local_reliable"] = False
    primary["evidence"].pop("whole_primitive_radius")
    primary["evidence"].pop("source_joint_radius_conflict_resolution")
    monkeypatch.setattr(binding, "_radius_primitive_feasibility",
                        lambda *args: {"passed": False})
    _admit_joint_radius_conflicts([primary, rival], observations, {}, graph, None, 4.)
    assert not primary["local_reliable"]
