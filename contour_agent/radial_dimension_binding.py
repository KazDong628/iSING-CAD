"""Source-owned half-section measurement stations; the common datum is free."""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from .dimension_evidence import _axis_lines, _dimension_station_arrow, binding_linear_witnesses
from .dimension_policy import source_dimension_target
from .measurement_anchors import (source_derived_locations, source_curve_center_locations,
                                  virtual_corner_locations, anchor_entity_ids)


def _coincident_virtual_station_options(options, coordinate, dimension_cross, band):
    """Keep one attachment to one observed extension, retaining all aliases.

    A restored lower fillet can expose a second theoretical corner on the
    same diameter extension as an upper corner. Two attachments to the same
    narrow source stroke are not two different dimensional stations. Prefer
    its uniquely closest finite attachment to this dimension line; never
    merge different strokes, wider station spreads or equally near corners.
    This chooses an anchor, not a coordinate equality between the aliases.
    """
    if len(options) < 2:
        return options
    try:
        if any(row['proof'].get('support_kind') != 'source_supported_virtual_corner' for row in options):
            return options
        strokes = [row['proof']['extension'] for row in options]
        signatures = {(stroke['lo'], stroke['hi'], stroke['cross'], stroke.get('thickness')) for stroke in strokes}
        if len(signatures) != 1:
            return options
        width = float(strokes[0]['thickness'])
        points = np.asarray([row['location']['source_px'] for row in options], float)
        if (points.shape != (len(options), 2) or not np.isfinite(points).all()
                or not math.isfinite(width) or not 0 < width <= 12
                or np.ptp(points[:, coordinate]) > width):
            return options
        order = sorted(range(len(options)), key=lambda index: abs(points[index, 1-coordinate]-dimension_cross))
        distances = [abs(points[index, 1-coordinate]-dimension_cross) for index in order]
        if distances[1]-distances[0] <= band:
            return options
        selected = deepcopy(options[order[0]])
        selected['proof'].update(
            coincident_source_station_members=[{'anchor': deepcopy(row['location']['anchor']),
                'source_px': points[index].tolist()} for index, row in enumerate(options)],
            source_station_spread_px=float(np.ptp(points[:, coordinate])),
            observed_station_width_px=width,
            representative_rule='unique_nearest_finite_attachment_on_same_observed_dimension_extension',
            coordinate_equality_enforced=False)
        return [selected]
    except (ValueError, TypeError, KeyError, IndexError):
        return options


def radial_dimension_candidates(gray, records, graph, model, band, *, radius_observations=(), angle_observations=()):
    """Eliminate one source-proven diameter datum without a fixed CAD axis.

    Every primary station must have one finite source association. The base
    diameter remains an explicit *dependency*, not another solved annotation.
    This function does not claim absolute datum recovery or full coverage.
    """
    from .constraint_binding import (_source_transform, _derived_station_groups,
        _extension_supports, _interval_gap, _stroke_interval)
    scale = model.get("scale") or {}
    if (scale.get("status") != "resolved" or scale.get("diameter_semantics") != "half_section_radius_station"
            or scale.get("axis") not in {"x", "y"} or scale.get("direction") not in {-1, 1}
            or scale.get("endpoint") not in {"lo", "hi"}):
        return [], []
    axis = scale["axis"]; coordinate = 0 if axis == "x" else 1
    # Source-image y points downward while the CAD ordinate points upward.
    direction = int(scale["direction"])*(1 if axis == "x" else -1)
    source_direction = int(scale["direction"])
    transform = _source_transform(model, graph)
    basis = transform([[0., 0.], [1., 0.], [0., 1.]])
    if abs(basis[1, 1]-basis[0, 1]) > 1e-5 or abs(basis[2, 0]-basis[0, 0]) > 1e-5:
        return [], []
    mapping = {row["id"]: row for row in records}
    extensions = _axis_lines(gray, "y" if axis == "x" else "x")
    levels = source_derived_locations(gray, records, graph, transform, band, axis)
    # A broad component that joins a datum leader and material edge cannot
    # establish which finite face owns a diameter. Keep it out of hard pairs.
    levels = [row for row in levels if row["anchor"]["type"] == "support_level" and
        all(stroke.get("thickness", math.inf) <= 12. for stroke in row["evidence"].get("supporting_strokes", []))]
    corners = virtual_corner_locations(gray, records, graph, transform, band,
                                       angle_observations=angle_observations)
    centers = source_curve_center_locations(gray, graph, radius_observations, axis, records=records)
    sections = _axis_lines(gray, axis)

    def station_options(station, cross, *, center_only=False):
        groups = _derived_station_groups(centers if center_only else levels, extensions,
                                        coordinate, station, cross, band, section_lines=sections)
        answer = [{"location": option[2], "proof": proof} for option, proof in groups
                  if proof.get("station_interpretation_verified")]
        if center_only:
            return answer
        for location in corners:
            point = location["source_px"]
            if abs(point[coordinate]-station) > 2*band:
                continue
            for support in _extension_supports(extensions, station, cross, point[1-coordinate], band):
                if _interval_gap(point[coordinate], _stroke_interval(support)) > band:
                    continue
                # If a finite level already denotes this same measured face,
                # use the face rather than duplicate it as its corner.
                if any(item["location"]["anchor"].get("entity_id") in location["anchor"]["entities"] for item in answer):
                    continue
                answer.append({"location": location, "proof": {
                    "support_kind": "source_supported_virtual_corner", "extension": support,
                    "anchor": location["anchor"], "construction": location["evidence"],
                    "observed_attachment_px": point, "station_interpretation_verified": True}})
        return _coincident_virtual_station_options(answer, coordinate, cross, band)

    owned = []
    diagnostics = []
    for binding in scale.get("bindings") or []:
        record = mapping.get(binding.get("record_id"))
        if (not record or record["parsed"].get("kind") != "diameter"
                or record["parsed"].get("nominal") != binding.get("nominal")
                or record.get("box") != binding.get("box")):
            continue
        line = binding["line"]; station = float(line[scale["endpoint"]])
        label_size = float(max(np.ptp(np.asarray(record["box"], float), axis=0)))
        arrow = _dimension_station_arrow(gray, axis, line["cross"], station,
                                          1 if scale["endpoint"] == "hi" else -1, label_size)
        options = station_options(station, line["cross"]) if arrow else []
        diagnostics.append({"record_id": record["id"], "source_station_px": station,
            "directed_dimension_arrow_verified": bool(arrow), "finite_station_candidate_count": len(options),
            "status": "unique_source_station" if len(options) == 1 else "source_station_unresolved",
            "absolute_diameter_verified": False})
        if len(options) == 1:
            owned.append({"record": record, "station": station, "line": line,
                          "arrow": arrow, **options[0]})
    bases = [row for row in owned if row["location"]["anchor"]["type"] == "support_level"]
    if not bases:
        return [], diagnostics
    # This is an algebraic reference choice among source-owned finite faces;
    # neither nominal closeness nor reference CAD selects it.
    reference = min(bases, key=lambda row: row["record"]["id"])
    reference_parsed = {key: reference["record"]["parsed"].get(key) for key in
                        ("kind", "nominal", "lower_deviation", "upper_deviation")}
    reference_target, _ = source_dimension_target(reference_parsed)
    candidates = []

    def make(record, primary, station, *, axis_evidence=None):
        parsed = record["parsed"]
        primary_target, _ = source_dimension_target(parsed)
        multiplier = 2. if parsed["kind"] == "diameter" else 1.
        expected = (primary_target/multiplier-reference_target/2)*scale["pixels_per_mm"]*source_direction
        observed = station-reference["station"]
        limit = max(2*band, abs(expected)*.035)
        compatible = abs(observed-expected) <= limit
        anchors = [reference["location"]["anchor"], primary["location"]["anchor"]]
        if anchors[0] == anchors[1]:
            return
        relation = {"type": "half_section_common_axis_eliminated", "axis": axis, "direction": direction,
            "primary_kind": parsed["kind"], "reference_record_id": reference["record"]["id"],
            "reference_dimension": deepcopy(reference_parsed)}
        candidates.append({"record_id": record["id"], "kind": "distance_"+axis,
            "value": parsed["nominal"], "nodes": [], "anchors": deepcopy(anchors),
            "entities": sorted({identifier for anchor in anchors for identifier in
                anchor_entity_ids(anchor)}),
            "source_station_relation": relation, "local_reliable": compatible,
            "_score": 0., "evidence": {
                "method": "source_owned_half_section_station_difference_v1", "axis": axis,
                "reference_record_id": reference["record"]["id"],
                "reference_station_arrow": reference["arrow"],
                "observed_station_groups": [reference["proof"], primary["proof"]],
                "source_axis_evidence": axis_evidence,
                "source_half_section_semantics": scale["diameter_semantics"],
                "source_span_scale_compatible": compatible, "observed_signed_span_px": observed,
                "expected_signed_span_px": expected, "source_span_tolerance_px": limit,
                "source_reference_selection": "stable_identifier_among_verified_finite_material_faces",
                "reference_diameter_is_dependency_not_solved_record": True,
                "absolute_axis_position_verified": False, "coordinates_locked": False,
                "nominal_used_for_station_ownership": False}})

    for row in owned:
        if row is reference:
            continue
        make(row["record"], row, row["station"])
    # A dimension from the visible axis to an explicitly observed circle
    # center is the same algebraic construction, with diameter/2 as its datum.
    for witness in binding_linear_witnesses(gray, records):
        if witness["axis"] != axis:
            continue
        line = witness["line"]
        axis_station = line["lo" if source_direction > 0 else "hi"]
        target_station = line["hi" if source_direction > 0 else "lo"]
        if abs(axis_station-float(scale.get("axis_origin_px", math.inf))) > 2*band:
            continue
        axis_lines = [stroke for stroke in extensions
            if _interval_gap(axis_station, _stroke_interval(stroke)) <= band
            and stroke["lo"]-band <= line["cross"] <= stroke["hi"]+band
            and stroke["span"] >= .1*gray.shape[1-coordinate]]
        if len(axis_lines) != 1:
            continue
        options = station_options(target_station, line["cross"], center_only=True)
        if len(options) != 1:
            continue
        make(mapping[witness["record_id"]], options[0], target_station, axis_evidence={
            "method": "visible_axis_stroke_and_independent_half_section_dimension_consensus",
            "observed_axis_stroke": axis_lines[0], "axis_coordinate_not_locked": True})
    return candidates, diagnostics
