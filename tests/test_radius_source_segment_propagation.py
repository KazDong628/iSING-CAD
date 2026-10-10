"""Current preflight radius evidence may seed a fresh topology pixel check."""

from contour_agent.parametric_pipeline import _radius_segment_hypotheses


def test_preflight_carries_only_source_segments_without_old_targets_or_verdicts():
    bindings = {"radius_binding_coverage": {"required_mappings": [
        {"record_id": "r009", "entity_id": "g002", "source_arrow_verified": True,
         "source_evidence": [
             {"segment_px": [[10, 20], [30, 40]], "arrowhead_verified": True,
              "target_candidates": [{"entity_id": "g002"}]},
             {"segment_px": [[10, 20], [30, 40]]},
             {"segment_px": [[float("nan"), 20], [30, 40]]},
         ]},
        {"record_id": "r024", "entity_id": None, "source_arrow_verified": True,
         "source_evidence": [{"segment_px": [[50, 60], [70, 80]],
                              "arrowhead_verified": True}]},
    ]}}

    assert _radius_segment_hypotheses(bindings) == [
        {"record_id": "r009", "kind": "radius",
         "source_evidence": {"segment_px": [[10., 20.], [30., 40.]]}},
        {"record_id": "r024", "kind": "radius",
         "source_evidence": {"segment_px": [[50., 60.], [70., 80.]]}},
    ]


def test_preflight_keeps_coordinate_hypothesis_until_a_new_observation_replaces_it():
    inherited = [
        {"record_id": "r015", "kind": "radius", "arrowhead_verified": True,
         "candidate_entity_id": "g014", "source_evidence": {
             "segment_px": [[1395, 805], [1555, 1207]], "arrowhead": {"verified": True}}},
        {"record_id": "r024", "kind": "radius", "arrowhead_verified": True,
         "source_evidence": {"segment_px": [[50, 60], [70, 80]]}},
    ]
    fresh = {"radius_binding_coverage": {"required_mappings": [
        {"record_id": "r024", "entity_id": "g123", "source_evidence": [
            {"segment_px": [[51, 61], [71, 81]], "arrowhead_verified": True}]},
    ]}}

    assert _radius_segment_hypotheses(fresh, inherited) == [
        {"record_id": "r024", "kind": "radius",
         "source_evidence": {"segment_px": [[51., 61.], [71., 81.]]}},
        {"record_id": "r015", "kind": "radius",
         "source_evidence": {"segment_px": [[1395., 805.], [1555., 1207.]]}},
    ]
