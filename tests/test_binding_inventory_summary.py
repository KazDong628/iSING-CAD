"""Source evidence summaries keep failures visible without copying local audits."""
from copy import deepcopy
import json

from contour_agent.binding_provider import bounded_inventory


def inventory():
    return {"units": "mm", "records": [{"id": "r000", "text": "R176", "box": [[10., 10.], [20., 20.]],
        "parsed": {"kind": "radius", "nominal": 176., "calibration_coordinates": "PRIVATE_GT"}}],
        "all_records": [{"id": "r000"}, {"id": "r001"}],
        "candidates": [{"id": "c000", "record_id": "r000", "kind": "radius", "entities": ["g019"],
            "nodes": ["v019", "v020"], "evidence": {"method": "source_label_proximity",
                "multiple_directed_source_targets_require_review": True, "directed_source_target_count": 2,
                "source_arrow_ownership_rejection_reasons": ["shared_source_arrow_ownership_ambiguous"],
                "leader": {"arrowhead_verified": True, "arrowhead": {"verified": True, "tip_px": [11., 12.],
                    "direction_px": [0., 1.], "fitted_radius": 140.555, "gt": "PRIVATE_GT"},
                    "shaft_evidence": {"verified": True, "supported_fraction": .95, "raw_pixels": ["SECRET"]*10000},
                    "source_arrow_ownership": {"status": "source_ownership_ambiguous",
                        "competing_record_ids": ["r000", "r001"], "duplicate_target_ambiguity_preserved": True},
                    "construction_history": {"radius": 176., "origin": "FREE_HISTORY_MUST_NOT_SEND"}},
                "whole_primitive_radius": {"checked": True, "passed": False, "status": "requires_topology_repartition",
                    "reason": "source_interval_crosses_multiple_curvatures", "conservative_max_residual_px": 30.9,
                    "source_deviation_budget_px": 3.79, "center_source_px": [999., 999.], "radius_source_px": 694.,
                    "model": {"center": "PRIVATE_GT"}},
                "extension_lines": [{"verified": False, "reason": "source_extension_missing"}],
                "source_span_scale_compatible": False, "fitted_radius": 140.555,
                "nominal_difference": -35.445, "unknown_private_audit": ["SECRET"]*10000,
                "occluded_leader_hypotheses": [{"reason": "crosses_other_contour", "coordinates": ["SECRET"]*1000}]}}],
        "relations": [{"id": "rel000", "type": "tangent", "entities": ["g019", "g020"], "nodes": ["v020"],
            "evidence": {"verified": False, "reason": "source_tangent_evidence_insufficient",
                "sides": [{"verified": False, "reason": "source_junction_strokes_ambiguous"}, {"verified": True}]}}],
        "radius_binding_coverage": {"recognized_count": 12}, "constructed_radius_priors": [{"value": "FREE_HISTORY_MUST_NOT_SEND"}]}


def test_summary_keeps_source_failures_competing_ids_and_denominator_without_fitted_or_gt_payloads():
    original = inventory(); before = deepcopy(original)
    packet = bounded_inventory(original); text = json.dumps(packet, ensure_ascii=False, allow_nan=False)
    assert original == before
    assert all(value not in text for value in ("PRIVATE_GT", "SECRET", "FREE_HISTORY_MUST_NOT_SEND",
                                                "fitted_radius", "nominal_difference", "center_source_px", "radius_source_px"))
    row = packet["candidates"][0]; evidence = row["evidence"]
    assert row["entities"] == ["g019"] and row["nodes"] == ["v019", "v020"]
    assert evidence["whole_primitive_radius"]["passed"] is False
    assert evidence["whole_primitive_radius"]["status"] == "requires_topology_repartition"
    assert evidence["whole_primitive_radius"]["reason"] == "source_interval_crosses_multiple_curvatures"
    assert evidence["leader"]["source_arrow_ownership"]["competing_record_ids"] == ["r000", "r001"]
    assert evidence["multiple_directed_source_targets_require_review"] is True
    assert evidence["extension_lines"][0] == {"verified": False, "reason": "source_extension_missing"}
    assert evidence["source_arrow_ownership_rejection_reasons"] == ["shared_source_arrow_ownership_ambiguous"]
    assert evidence["occluded_leader_hypotheses_summary"]["reason_counts"] == {"crosses_other_contour": 1}
    assert packet["relations"][0]["evidence"]["sides"][0]["verified"] is False
    assert packet["inventory_coverage"]["radius_record_denominator"] == 12
    assert packet["inventory_coverage"]["record_ids_not_sent"] == ["r001"]
    assert packet["source_evidence_summary"]["full_audit_retained_locally"]


def test_known_fields_are_recursively_bounded_and_limits_are_disclosed():
    raw = inventory(); evidence = raw["candidates"][0]["evidence"]
    evidence["method"] = "x"*10000
    evidence["extension_lines"] = [{"verified": False, "reason": "source_extension_missing"}]*1000
    evidence["leader"]["shaft_evidence"]["verified"] = {"nested": {"secret": "SECRET"}}
    packet = bounded_inventory(raw); audit = packet["source_evidence_summary"]
    assert len(packet["candidates"][0]["evidence"]["method"]) == 160
    assert len(packet["candidates"][0]["evidence"]["extension_lines"]) == 2
    assert packet["candidates"][0]["evidence"]["leader"]["shaft_evidence"]["verified"] is None
    assert audit["truncated_strings"] == 1 and audit["truncated_collection_items"] == 998
    assert audit["invalid_or_depth_limited_values"] == 1
    assert len(json.dumps(packet, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) < 60000


def test_summary_does_not_drop_sent_rows_or_identity_arrays_to_fit_byte_budget():
    raw = inventory(); competitors = [f"r{i:03d}" for i in range(100)]
    raw["candidates"][0]["evidence"]["leader"]["source_arrow_ownership"]["competing_record_ids"] = competitors
    raw["relations"] += [{"id": "rel001", "type": "horizontal", "entities": ["g001"], "nodes": []}]
    packet = bounded_inventory(raw, relation_limit=1)
    assert packet["candidates"][0]["evidence"]["leader"]["source_arrow_ownership"]["competing_record_ids"] == competitors
    assert [r["id"] for r in packet["records"]] == ["r000"]
    assert [r["id"] for r in packet["candidates"]] == ["c000"]
    assert packet["inventory_coverage"]["all_relation_count"] == 2
    assert packet["inventory_coverage"]["relation_ids_not_sent"] == ["rel001"]


def test_provider_enforces_utf8_byte_cap_before_network_for_multibyte_inventory(tmp_path, monkeypatch):
    from PIL import Image
    from contour_agent.binding_provider import BindingProvider
    from contour_agent.config import Settings
    path = tmp_path/"source.png"; Image.new("RGB", (32, 32), "white").save(path)
    def unexpected_network(*args, **kwargs):
        raise AssertionError("oversized byte packet must not reach the transport")
    monkeypatch.setattr("contour_agent.binding_provider.httpx.AsyncClient", unexpected_network)
    raw = inventory(); raw["units"] = "字"*21000
    receipt = BindingProvider(Settings(api_key="test")).select(path, path, raw)
    assert receipt["error_code"] == "inventory_size_limit" and receipt["network_requests"] == 0
    assert receipt["inventory_text_chars"] < 60000 < receipt["inventory_text_bytes"]
    assert receipt["inventory_byte_limit"] == 60000
