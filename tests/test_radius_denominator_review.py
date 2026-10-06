"""Unresolved radius text stays in the source denominator, without guessing R."""
from copy import deepcopy

import pytest

from contour_agent.constraint_binding import radius_binding_coverage
from contour_agent.ocr import canonical_records, parse_dimension
from contour_agent.radius_contract import annotation_radius_contract


def coverage_with_additional_record(text, *, parsed=None):
    records = canonical_records({"records": [{"text":"R5"}, {"text":text}]})
    if parsed is not None:
        # A stale or externally assembled inventory must not erase the raw
        # source radius marker by assigning a different kind or invalid value.
        records[1]["parsed"] = deepcopy(parsed)
    graph = {"entities":[{"id":"g000", "type":"ARC", "radius":5.}]}
    constraints = [{"id":"k000", "kind":"radius", "record_id":"r000",
                    "entities":["g000"], "value":5.}]
    inventory = {"all_records":records, "all_candidates":[{
        "id":"c000", "record_id":"r000", "kind":"radius", "entities":["g000"],
        "local_reliable":True, "evidence":{"leader":{"arrowhead_verified":True}}}]}
    coverage = radius_binding_coverage(inventory, graph, constraints)
    contract = annotation_radius_contract({"radius_binding_coverage":coverage, "constraints":constraints},
                                         {"accepted":True, "entities":graph["entities"]})
    return coverage, contract


@pytest.mark.parametrize("text", ["R1O", "R?", "R", "R0", "R-3", "Ｒ５Ｏ", "(R?)"])
def test_unparsed_or_invalid_radius_text_cannot_vanish_behind_an_exact_subset(text):
    coverage, contract = coverage_with_additional_record(text)
    # The existing valid R5 really is solved. Failure must come from keeping
    # the other source label in the denominator, not from rejecting the subset.
    assert coverage["bound_count"] == 1
    assert contract["exact_radius_validation"]["passed"] is True
    assert coverage["recognized_count"] == 2
    assert "r001" in coverage["recognized_radius_records"]
    assert any(row["record_id"] == "r001" and row["reason"] == "unresolved_radius_text"
               for row in coverage["unresolved"])
    assert not coverage["all_radius_records_resolved"]
    assert not contract["satisfied"] and not contract["all_annotated_radii_verified"]
    assert all(row["record_id"] != "r001" for row in coverage["bound_mappings"])


@pytest.mark.parametrize("parsed", [
    {"kind":"length", "nominal":5.},
    {"kind":"unknown", "nominal":None},
    {"kind":"radius", "nominal":None},
    {"kind":"radius", "nominal":True},
    {"kind":"radius", "nominal":float("nan")},
    {"kind":"radius", "nominal":float("inf")},
    {"kind":"radius", "nominal":6.},
])
def test_source_radius_marker_survives_wrong_or_invalid_parsed_metadata(parsed):
    coverage, contract = coverage_with_additional_record("R5", parsed=parsed)
    assert coverage["recognized_count"] == 2
    assert any(row["record_id"] == "r001" and row["reason"] == "unresolved_radius_text"
               for row in coverage["unresolved"])
    assert not contract["satisfied"]


@pytest.mark.parametrize("text", ["Ra3.2", "25", "REV", "REFERENCE", "RIGHT"])
def test_non_radius_text_does_not_create_false_radius_obligations(text):
    coverage, contract = coverage_with_additional_record(text)
    assert coverage["recognized_radius_records"] == ["r000"]
    assert coverage["all_radius_records_resolved"] and contract["satisfied"]


def test_valid_radius_without_verified_arrow_remains_unknown_not_exempt():
    coverage, contract = coverage_with_additional_record("R8")
    assert parse_dimension("R8")["kind"] == "radius"
    assert coverage["recognized_count"] == 2
    assert coverage["unknown_arrow_records"] == ["r001"]
    assert coverage["verified_absent_arrow_records"] == []
    assert not coverage["all_radius_records_resolved"] and not contract["satisfied"]
