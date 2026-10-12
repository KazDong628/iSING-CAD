"""Source-supported measurement locations which need not be contour vertices.

The theoretical sharp corner of a LINE--ARC--LINE chain is a derived location,
not a new CAD vertex. Its coordinates must be recomputed during the solve.
This module reads source pixels and topology only; nominal dimensions never
select a support, an intersection, or a dimension endpoint.
"""
from __future__ import annotations

import math
import re
from itertools import combinations
import cv2
import numpy as np

from .dimension_evidence import _axis_lines
from .structural_evidence import axis_stroke_evidence, source_ink


DERIVED_ANCHOR_LINEAR_TOLERANCE = 1e-7
DERIVED_ANCHOR_ANGLE_TOLERANCE = 1e-9
_DERIVED_TYPES = {"support_level", "curve_extremum", "curve_center", "common_curve_center"}


def anchor_entity_ids(anchor):
    """Exact support union of a typed measurement location."""
    if anchor.get("type") == "common_curve_center":
        return [anchor["entity_id"], anchor["other_entity_id"]]
    return anchor.get("entities", []) or ([anchor["entity_id"]] if "entity_id" in anchor else [])


def _derived_entity(anchor, entities):
    identifier = anchor.get("entity_id")
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,119}", identifier):
        raise ValueError("Derived measurement anchor needs a bounded entity identifier")
    if identifier not in entities:
        raise ValueError("Derived measurement anchor references an unknown entity")
    entity = entities[identifier]
    expected = "LINE" if anchor.get("type") == "support_level" else "ARC"
    if entity.get("type") != expected:
        raise ValueError("Derived measurement anchor references non-" + expected + " support")
    return entity


def _arc_parameters(entity):
    center = np.asarray(entity["center"], float)
    start, end = np.asarray(entity["start"], float), np.asarray(entity["end"], float)
    radius = float(entity["radius"])
    if (center.shape != (2,) or start.shape != (2,) or end.shape != (2,)
            or not np.isfinite(np.r_[center, start, end, radius]).all()):
        raise ValueError("Derived measurement support must have finite coordinates")
    first = math.atan2(*(start-center)[::-1])
    last = math.atan2(*(end-center)[::-1])
    sign = -1. if entity.get("clockwise") else 1.
    sweep = (sign*(last-first)) % (2*math.pi)
    return center, radius, first, sign, sweep


def _cardinal_angle(anchor):
    axis = anchor.get("axis")
    if axis not in {"x", "y"} or anchor.get("sense") not in {"minimum", "maximum"}:
        raise ValueError("Curve extremum requires a valid axis and sense")
    return (0. if axis == "x" else math.pi/2) + (math.pi if anchor["sense"] == "minimum" else 0.)


def validate_derived_anchor(anchor, entities, measurement_axis):
    """Validate a symbolic reference, returning its identity and support union.

    Validation permits a geometrically imperfect initialization. Orientation
    equalities and finite-domain inequalities must also enter the optimization
    and its final certificate. No coordinate literal, free station, or numeric
    fitting target is allowed in these anchor schemas.
    """
    if not isinstance(anchor, dict) or anchor.get("type") not in _DERIVED_TYPES:
        raise ValueError("Unsupported derived measurement anchor type")
    keys = {"type", "entity_id", "axis"}
    if anchor["type"] == "common_curve_center":
        keys.add("other_entity_id")
    if anchor["type"] == "curve_extremum":
        keys.add("sense")
    if set(anchor) != keys:
        raise ValueError("Derived measurement anchors cannot contain literal coordinates or extra fields")
    if measurement_axis not in {"x", "y"} or anchor.get("axis") != measurement_axis:
        raise ValueError("Derived measurement anchor axis must match its projected distance")
    entity = _derived_entity(anchor, entities)
    supports = {anchor["entity_id"]}
    if anchor["type"] == "common_curve_center":
        other = _derived_entity({**anchor, "entity_id": anchor["other_entity_id"]}, entities)
        if anchor["other_entity_id"] == anchor["entity_id"] or _arc_parameters(other)[1] <= 0:
            raise ValueError("Common center requires two distinct positive-radius ARCs")
        supports.add(anchor["other_entity_id"])
    if anchor["type"] == "curve_extremum":
        _cardinal_angle(anchor)
    if anchor["type"] in {"curve_extremum", "curve_center", "common_curve_center"} and _arc_parameters(entity)[1] <= 0:
        raise ValueError("Curve measurement anchor requires a positive radius")
    derived_anchor_point(anchor, entities, allow_infeasible=True)
    identity = (anchor["type"], anchor["entity_id"], anchor["axis"])
    if anchor["type"] == "curve_extremum":
        identity += (anchor["sense"],)
    if anchor["type"] == "common_curve_center":
        identity += (anchor["other_entity_id"],)
    return identity, supports


def derived_anchor_equalities(anchor, entities):
    """Additional geometry equations required to give a level its meaning."""
    entity = _derived_entity(anchor, entities)
    if anchor.get("axis") not in {"x", "y"}:
        raise ValueError("Derived measurement anchor requires a valid axis")
    if anchor["type"] == "support_level":
        axis = 0 if anchor["axis"] == "x" else 1
        return np.asarray([float(entity["end"][axis])-float(entity["start"][axis])])
    if anchor["type"] == "common_curve_center":
        other = _derived_entity({**anchor, "entity_id": anchor["other_entity_id"]}, entities)
        return _arc_parameters(entity)[0]-_arc_parameters(other)[0]
    return np.empty(0, float)


def derived_anchor_domain_conditions(anchor, entities):
    """Nonnegative conditions for a finite geometric support, not its extension.

    The cardinal-angle margin is measured around the middle of the declared
    directed ARC branch. This handles clockwise, major, and zero-crossing arcs
    without accepting the opposite part of the circle. Endpoint extrema are
    valid, while an extremum of the full circle outside this arc is not.
    """
    entity = _derived_entity(anchor, entities)
    if anchor["type"] == "support_level":
        start, end = np.asarray(entity["start"], float), np.asarray(entity["end"], float)
        return np.asarray([float(np.linalg.norm(end-start))-1e-9])
    if anchor["type"] in {"curve_center", "common_curve_center"}:
        if anchor["type"] == "common_curve_center":
            other = _derived_entity({**anchor, "entity_id": anchor["other_entity_id"]}, entities)
            return np.asarray([min(_arc_parameters(entity)[1], _arc_parameters(other)[1])-1e-9])
        return np.asarray([_arc_parameters(entity)[1]-1e-9])
    _, radius, start, sign, sweep = _arc_parameters(entity)
    midpoint = start + sign*sweep/2
    delta = (_cardinal_angle(anchor)-midpoint+math.pi) % (2*math.pi)-math.pi
    return np.asarray([radius-1e-9, sweep-1e-9, sweep/2-abs(delta)])


def derived_anchor_point(anchor, entities, *, allow_infeasible=False):
    """Recompute from current unknowns; only the requested axis is measurable."""
    if anchor.get("type") not in _DERIVED_TYPES:
        raise ValueError("Unsupported derived measurement anchor type")
    entity = _derived_entity(anchor, entities)
    if anchor.get("axis") not in {"x", "y"}:
        raise ValueError("Derived measurement anchor requires a valid axis")
    if anchor["type"] == "support_level":
        start, end = np.asarray(entity["start"], float), np.asarray(entity["end"], float)
        if start.shape != (2,) or end.shape != (2,):
            raise ValueError("Derived measurement support must have two coordinates")
        point = (start+end)/2
    elif anchor["type"] in {"curve_center", "common_curve_center"}:
        point = _arc_parameters(entity)[0].copy()
    else:
        center, radius, _, _, _ = _arc_parameters(entity)
        # Exact cardinal vectors avoid trigonometric residue on the other axis.
        direction = np.zeros(2)
        direction[0 if anchor["axis"] == "x" else 1] = 1. if anchor.get("sense") == "maximum" else -1.
        _cardinal_angle(anchor)
        point = center + radius*direction
    if not np.isfinite(point).all() or np.max(np.abs(point)) > 1e9:
        raise ValueError("Derived measurement support exceeds the bounded coordinate range")
    if not allow_infeasible:
        if np.any(np.abs(derived_anchor_equalities(anchor, entities)) > DERIVED_ANCHOR_LINEAR_TOLERANCE):
            if anchor["type"] == "common_curve_center":
                raise ValueError("Source-declared ARC centers do not coincide")
            raise ValueError("Support level LINE is not constant on the measured axis")
        conditions = derived_anchor_domain_conditions(anchor, entities)
        if conditions[0] <= 0 or (len(conditions) > 1 and conditions[1] <= 0) or np.any(conditions < -DERIVED_ANCHOR_ANGLE_TOLERANCE):
            raise ValueError("Derived measurement location is outside its finite support domain")
    return point


def _source_extremum_support(ink, entity, anchor, transform, band):
    """Observe a turn in original ink, never infer it from a fitted radius."""
    center, radius, start, sign, sweep = _arc_parameters(entity)
    cardinal = _cardinal_angle(anchor)
    travelled = (sign*(cardinal-start)) % (2*math.pi)
    if abs(travelled-2*math.pi) < 1e-8:
        travelled = 0.
    half_angle = min(.5, travelled*.85, (sweep-travelled)*.85)
    result = {"method": "source_ink_finite_curve_extremum_v1", "verified": False,
              "nominal_used": False, "coordinates_locked": False,
              "reason": "insufficient_source_turn_on_both_sides"}
    if half_angle <= 0:
        return result
    angles = np.linspace(cardinal-half_angle, cardinal+half_angle, 41)
    material_points = center+radius*np.c_[np.cos(angles), np.sin(angles)]
    proposed = transform(material_points)
    center_px = transform([center])[0]
    normals = proposed-center_px
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    offsets = np.arange(-max(4, math.ceil(band)), max(4, math.ceil(band))+1)
    measured, kept = [], []
    for index, (point, normal) in enumerate(zip(proposed, normals)):
        pixels = np.rint(point+offsets[:, None]*normal).astype(int)
        if ((pixels < 0).any() or (pixels[:, 0] >= ink.shape[1]).any()
                or (pixels[:, 1] >= ink.shape[0]).any()):
            continue
        occupied = ink[pixels[:, 1], pixels[:, 0]]
        begins = np.flatnonzero(occupied & ~np.r_[False, occupied[:-1]])
        ends = np.flatnonzero(occupied & ~np.r_[occupied[1:], False])
        if (len(begins) != 1 or begins[0] == 0 or ends[0] == len(offsets)-1
                or ends[0]-begins[0]+1 > max(7., band)):
            continue
        measured.append(point+float(offsets[begins[0]:ends[0]+1].mean())*normal)
        kept.append(index)
    result.update(sample_count=len(angles), unambiguous_sample_count=len(measured))
    if len(measured) < 32 or min(kept, default=41) > 3 or max(kept, default=0) < 37:
        result["reason"] = "ambiguous_or_missing_source_curve_ink"
        return result
    points = np.asarray(measured)
    axis = 0 if anchor["axis"] == "x" else 1
    proposed_extreme = transform([derived_anchor_point(anchor, {entity["id"]: entity})])[0]
    cross = points[:, 1-axis]-proposed_extreme[1-axis]
    half_span = float(np.ptp(cross))/2
    if half_span < max(12., 2*band):
        return result
    parameter = cross/half_span
    coefficients = np.polynomial.polynomial.polyfit(parameter, points[:, axis], 2)
    fitted = np.polynomial.polynomial.polyval(parameter, coefficients)
    residual = float(np.quantile(np.abs(fitted-points[:, axis]), .9))
    curvature = float(coefficients[2])
    expected_sign = math.copysign(1., proposed_extreme[axis]-center_px[axis])
    result.update(source_half_span_px=half_span, source_fit_p90_residual_px=residual,
                  source_turn_depth_px=abs(curvature), finite_arc_domain_checked=True)
    if (abs(curvature) < max(1.5, 3*residual) or curvature*expected_sign >= 0):
        result["reason"] = "source_turn_direction_or_magnitude_unconfirmed"
        return result
    vertex_parameter = -float(coefficients[1])/(2*curvature)
    vertex = proposed_extreme.copy()
    vertex[1-axis] += vertex_parameter*half_span
    vertex[axis] = np.polynomial.polynomial.polyval(vertex_parameter, coefficients)
    # A turn must be bracketed by observed source points. A parabola whose
    # stationary point is only reached by extrapolation is not finite evidence.
    if (abs(vertex_parameter) > .55 or residual > 2.
            or float(np.linalg.norm(vertex-proposed_extreme)) > max(2., band)):
        result["reason"] = "source_extremum_not_uniquely_attached_to_finite_arc"
        return result
    result.update(verified=True, reason=None, observed_source_extremum_px=vertex.tolist(),
                  source_position_uncertainty_px=max(1., residual),
                  observation_only_not_coordinate_lock=True)
    return result


def source_derived_locations(gray, records, graph, transform, band, axis):
    """Propose finite LINE levels / ARC extrema for observed dimension stations.

    These are source-supported *locations*, not admitted dimensions. The caller
    must establish matching dimension arrows/extensions, finite attachment and
    a unique interpretation, preserving all competing associations. In
    particular a section through a curve may not be replaced by its extremum
    merely because their raster positions are close.
    """
    if axis not in {"x", "y"} or not math.isfinite(float(band)) or band <= 0:
        raise ValueError("Source derived locations require an axis and positive finite band")
    origin, x_axis, y_axis = transform([[0., 0.], [1., 0.], [0., 1.]])
    basis = np.asarray([x_axis-origin, y_axis-origin])
    # The existing dimension witness axes are source-image x/y. Do not silently
    # reinterpret them through a rotated or sheared proposal transform.
    if (np.any(np.linalg.norm(basis, axis=1) <= 1e-12)
            or abs(basis[0, 1]) > 1e-4*abs(basis[0, 0])
            or abs(basis[1, 0]) > 1e-4*abs(basis[1, 1])):
        return []
    cleaned, ink = source_ink(gray, records)
    along = "y" if axis == "x" else "x"
    source_lines = _axis_lines(cleaned, along)
    axis_index = 0 if axis == "x" else 1
    result = []
    for entity in graph.get("entities") or []:
        entity_map = {entity["id"]: entity}
        if entity.get("type") == "LINE":
            anchor = {"type": "support_level", "entity_id": entity["id"], "axis": axis}
            points = transform(np.linspace(entity["start"], entity["end"], 24))
            evidence = axis_stroke_evidence(points, "vertical" if axis == "x" else "horizontal", source_lines, band)
            if not evidence["verified"]:
                continue
            point = derived_anchor_point(anchor, entity_map, allow_infeasible=True)
            source_point = transform([point])[0]
            proposal_interval = [float(points[:, 1-axis_index].min()), float(points[:, 1-axis_index].max())]
            observed_stroke = evidence["supporting_strokes"][0]
            observed_interval = [float(observed_stroke["lo"]), float(observed_stroke["hi"])]
            attachment_interval = [max(proposal_interval[0], observed_interval[0]),
                                   min(proposal_interval[1], observed_interval[1])]
            if attachment_interval[0] >= attachment_interval[1]:
                continue
            evidence.update(method="source_supported_finite_line_level_v1",
                            source_axis_evidence_method="unique_source_axis_stroke_v1",
                            source_attachment_interval_px=attachment_interval,
                            proposal_finite_interval_px=proposal_interval,
                            observed_source_stroke_interval_px=observed_interval,
                            attachment_interval_rule="intersection_of_finite_proposal_and_observed_source_stroke",
                            observed_source_level_px=evidence["observed_station_px"],
                            required_axis_equality=True, nominal_used=False, coordinates_locked=False,
                            measurement_axis=axis, source_supported_entities=[entity["id"]])
            result.append({"id": "level_"+entity["id"]+"_"+axis, "x": float(point[0]), "y": float(point[1]),
                           "source_px": source_point.tolist(), "anchor": anchor, "evidence": evidence})
        elif entity.get("type") == "ARC":
            for sense in ("minimum", "maximum"):
                anchor = {"type": "curve_extremum", "entity_id": entity["id"], "axis": axis, "sense": sense}
                try:
                    point = derived_anchor_point(anchor, entity_map)
                except (ValueError, KeyError, TypeError, ArithmeticError):
                    continue
                evidence = _source_extremum_support(ink, entity, anchor, transform, band)
                if not evidence["verified"]:
                    continue
                evidence.update(measurement_axis=axis, source_supported_entities=[entity["id"]],
                                section_interpretation_not_resolved=True)
                result.append({"id": "extremum_"+entity["id"]+"_"+axis+"_"+sense,
                               "x": float(point[0]), "y": float(point[1]),
                               "source_px": transform([point])[0].tolist(), "anchor": anchor, "evidence": evidence})
    return result


def source_curve_center_locations(gray, graph, radius_observations, axis, *, records=()):
    """Observe source centers, also remeasuring long shafts when needed.

    Short arrow-local directions are not reliable remote center coordinates.
    Remeasure their original finite ink independently; do not extrapolate a
    fitted CAD radius or reuse the predicted center as source evidence.
    """
    from copy import deepcopy
    from .source_measurement_tracks import radial_shaft_track
    result = _curve_centers_from_rays(gray, graph, radius_observations, axis)
    refined = []
    for row in radius_observations:
        if (row.get("arrowhead_verified") is not True or
                (row.get("shaft_evidence") or {}).get("verified") is not True or
                (row.get("source_arrow_ownership") or {}).get("status") != "globally_unique_source_claim"):
            continue
        track = radial_shaft_track(gray, row, records)
        if track is None:
            continue
        updated = deepcopy(row)
        updated["source_ray_origin_px"] = track["origin_px"]
        updated["source_ray_direction_px"] = track["direction_px"]
        updated["source_long_shaft_remeasurement"] = track["evidence"]
        refined.append(updated)
    extra = _curve_centers_from_rays(gray, graph, refined, axis, records=records)
    # Preserve conflicting observations as alternatives. Deduplicate only the
    # same source-owned ARC center at the same observed position.
    for location in extra:
        same = [old for old in result if old["anchor"] == location["anchor"] and
                np.linalg.norm(np.asarray(old["source_px"])-location["source_px"]) <= 2.]
        if not same:
            result.append(location)
    return result


def _curve_centers_from_rays(gray, graph, radius_observations, axis, *, records=()):
    """Locate explicit circle centers where two verified radius shafts terminate.

    Intersecting infinite leader lines are insufficient: both *observed* shafts
    must continue behind their OCR labels to the intersection, then stop. The
    source point only associates a dimension station. The solver evaluates the
    current ARC center symbolically, so no fitted or measured coordinate is a
    dimensional target. Ambiguous arrow-to-ARC ownership remains unadmitted.
    """
    if axis not in {"x", "y"}:
        raise ValueError("Source curve center requires a projected axis")
    entities = {row["id"]: row for row in graph.get("entities", [])}
    observations = [row for row in radius_observations
        if row.get("arrowhead_verified") is True
        and (row.get("arrowhead") or {}).get("verified") is True
        and (row.get("shaft_evidence") or {}).get("verified") is True
        and (row.get("source_arrow_ownership") or {}).get("status") == "globally_unique_source_claim"]
    if len(observations) < 2:
        return []
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    distance = cv2.distanceTransform((1-ink).astype(np.uint8), cv2.DIST_L2, 3)
    height, width = gray.shape

    def evidence_to_center(row, center):
        tip = np.asarray(row["arrowhead"]["tip_px"], float)
        direction = np.asarray(row.get("source_ray_direction_px", row["arrowhead"]["direction_px"]), float)
        direction /= np.linalg.norm(direction)
        ends = np.asarray(row["segment_px"], float)
        tail = ends[np.argmax((ends-tip) @ -direction)]
        extension = float((center-tail) @ -direction)
        if extension < 24.:
            return None
        samples = np.linspace(tail, center, min(4096, int(np.linalg.norm(center-tail))+1))
        pixels = np.rint(samples).astype(int)
        if (pixels < 0).any() or (pixels[:, 0] >= width).any() or (pixels[:, 1] >= height).any():
            return None
        occupied = distance[pixels[:, 1], pixels[:, 0]] <= 2.
        visible = np.ones(len(samples), bool)
        for record in records:
            if record.get("id") != row["record_id"]:
                continue
            box = np.asarray(record.get("box"), float)
            if box.ndim == 2 and box.shape[1] == 2 and np.isfinite(box).all():
                visible &= ~np.all((samples >= box.min(axis=0)-3.) &
                                   (samples <= box.max(axis=0)+3.), axis=1)
        if np.mean(visible) < .65 or np.count_nonzero(visible) < 24:
            return None
        longest = current = 0
        for supported, observed in zip(occupied, visible):
            current = 0 if supported or not observed else current+1
            longest = max(longest, current)
        step = float(np.linalg.norm(center-tail))/max(1, len(samples)-1)
        maximum_gap = longest*step
        if float(occupied[visible].mean()) < .9 or maximum_gap > 12.:
            return None
        # A crossing along two continuing shafts is not a declared center.
        beyond = np.rint(center-np.arange(12., 42.)[:, None]*direction).astype(int)
        if ((beyond < 0).any() or (beyond[:, 0] >= width).any()
                or (beyond[:, 1] >= height).any()):
            return None
        continuation = float(np.mean(distance[beyond[:, 1], beyond[:, 0]] <= 2.))
        if continuation > .25:
            return None
        return {"record_id": row["record_id"], "source_tail_px": tail.tolist(),
                "full_tail_supported_fraction": float(occupied[visible].mean()),
                "visible_tail_fraction": float(visible.mean()),
                "own_ocr_occlusion_is_not_ink_support": True,
                "maximum_unobserved_run_px": maximum_gap,
                "beyond_center_continuation_fraction": continuation,
                "tail_extension_px": extension, "verified": True,
                "long_shaft_remeasurement": row.get("source_long_shaft_remeasurement")}

    result = []
    for first, second in combinations(observations, 2):
        first_id = (first.get("source_arrow_ownership") or {}).get("physical_arrow_id")
        second_id = (second.get("source_arrow_ownership") or {}).get("physical_arrow_id")
        if not first_id or not second_id or first_id == second_id or first["record_id"] == second["record_id"]:
            continue
        a, b = (np.asarray(row.get("source_ray_origin_px", row["arrowhead"]["tip_px"]), float) for row in (first, second))
        u, v = (np.asarray(row.get("source_ray_direction_px", row["arrowhead"]["direction_px"]), float) for row in (first, second))
        if (a.shape != (2,) or b.shape != (2,) or u.shape != (2,) or v.shape != (2,)
                or not np.isfinite(np.r_[a, b, u, v]).all() or min(np.linalg.norm(u), np.linalg.norm(v)) <= 0):
            continue
        u, v = u/np.linalg.norm(u), v/np.linalg.norm(v)
        if abs(float(np.linalg.det(np.column_stack((u, v))))) < math.sin(math.radians(5.)):
            continue
        travel = np.linalg.solve(np.column_stack((u, -v)), b-a)
        if max(travel) >= -24.:
            continue
        center = a+travel[0]*u
        if not (0 <= center[0] < width and 0 <= center[1] < height):
            continue
        support = [evidence_to_center(row, center) for row in (first, second)]
        if not all(support):
            continue
        # Two independently owned radii whose full observed shafts end at one
        # point declare a common center. A dimension to that point must retain
        # BOTH ARC references and their center equalities, not select a fitted
        # center by proximity or silently discard the ambiguity.
        owned = [row.get("target_candidates") or [] for row in (first, second)]
        identifiers = [targets[0].get("entity_id") for targets in owned if len(targets) == 1]
        if (len(identifiers) == 2 and len(set(identifiers)) == 2 and
                all(entities.get(identifier, {}).get("type") == "ARC" for identifier in identifiers)):
            identifiers.sort()
            anchor = {"type": "common_curve_center", "entity_id": identifiers[0],
                      "other_entity_id": identifiers[1], "axis": axis}
            point = derived_anchor_point(anchor, entities, allow_infeasible=True)
            result.append({"id": "common_center_"+"_".join(identifiers)+"_"+axis,
                "x": float(point[0]), "y": float(point[1]), "source_px": center.tolist(),
                "anchor": anchor, "evidence": {
                    "method": "two_source_owned_radius_shafts_common_finite_termination_v2",
                    "verified": True, "center_semantics_verified": True,
                    "observed_source_center_px": center.tolist(),
                    "source_radius_record_ids": [first["record_id"], second["record_id"]],
                    "source_arrow_ids": [first_id, second_id], "tail_support": support,
                    "source_supported_entities": identifiers, "measurement_axis": axis,
                    "required_center_equalities": ["x", "y"],
                    "nominal_used": False, "coordinates_locked": False,
                    "other_arc_concentricity_not_implicitly_asserted": False}})
            continue
        for row in (first, second):
            targets = row.get("target_candidates") or []
            if len(targets) != 1:
                continue
            identifier = targets[0].get("entity_id")
            if identifier not in entities or entities[identifier].get("type") != "ARC":
                continue
            anchor = {"type": "curve_center", "entity_id": identifier, "axis": axis}
            point = derived_anchor_point(anchor, entities)
            result.append({"id": "center_"+identifier+"_"+first_id+"_"+second_id+"_"+axis,
                "x": float(point[0]), "y": float(point[1]), "source_px": center.tolist(),
                "anchor": anchor, "evidence": {
                    "method": "two_source_radius_shafts_with_common_observed_termination_v1",
                    "verified": True, "center_semantics_verified": True,
                    "observed_source_center_px": center.tolist(),
                    "source_radius_record_ids": [first["record_id"], second["record_id"]],
                    "source_arrow_ids": [first_id, second_id], "tail_support": support,
                    "owner_record_id": row["record_id"], "unique_arrow_target": identifier,
                    "source_supported_entities": [identifier], "measurement_axis": axis,
                    "nominal_used": False, "coordinates_locked": False,
                    "other_arc_concentricity_not_implicitly_asserted": True}})
    return result


def support_intersection(first, second):
    a, b = np.asarray(first["start"], float), np.asarray(first["end"], float)
    c, d = np.asarray(second["start"], float), np.asarray(second["end"], float)
    u, v = b-a, d-c
    lengths = np.linalg.norm(u)*np.linalg.norm(v)
    determinant = float(u[0]*v[1]-u[1]*v[0])
    if lengths <= 1e-12 or abs(determinant)/lengths < math.sin(math.radians(1.)):
        return None
    q = c-a
    return a + float(q[0]*v[1]-q[1]*v[0])/determinant*u


def _straight_source_support(ink, points, band):
    """Sample the whole proposed LINE; parallel strokes remain ambiguous."""
    start, end = np.asarray(points[0]), np.asarray(points[-1])
    delta = end-start
    length = float(np.linalg.norm(delta))
    result = {"verified": False, "method": "whole_line_single_source_ink_run_v1",
              "length_px": length, "nominal_used": False}
    if length < max(24., 4*band):
        return result
    direction = delta/length
    normal = np.array([-direction[1], direction[0]])
    offsets = np.arange(-math.ceil(band), math.ceil(band)+1)
    observations = []
    for point in np.linspace(start, end, 48)[2:-2]:
        pixels = np.rint(point+offsets[:, None]*normal).astype(int)
        if ((pixels < 0).any() or (pixels[:, 0] >= ink.shape[1]).any()
                or (pixels[:, 1] >= ink.shape[0]).any()):
            continue
        occupied = ink[pixels[:, 1], pixels[:, 0]]
        begins = np.flatnonzero(occupied & ~np.r_[False, occupied[:-1]])
        ends = np.flatnonzero(occupied & ~np.r_[occupied[1:], False])
        if len(begins) != 1 or begins[0] == 0 or ends[0] == len(offsets)-1:
            continue
        if ends[0]-begins[0]+1 > max(7., band):
            continue
        observations.append(point+float(offsets[begins[0]:ends[0]+1].mean())*normal)
    result["unambiguous_fraction"] = len(observations)/44
    if len(observations) < 34:
        return result
    observed = np.asarray(observations)
    center = observed.mean(axis=0)
    _, _, axes = np.linalg.svd(observed-center, full_matrices=False)
    actual = axes[0]
    perpendicular = np.array([-actual[1], actual[0]])
    residual = float(np.max(np.abs((observed-center)@perpendicular)))
    angle = math.degrees(math.acos(float(np.clip(abs(actual@direction), -1, 1))))
    result.update(max_line_residual_px=residual, direction_error_degrees=angle,
                  verified=bool(residual <= 2. and angle <= 2.))
    return result


def virtual_corner_locations(gray, records, graph, transform, band, *, angle_observations=()):
    """Bounded local candidates, independently checked against original ink."""
    entities = graph.get("entities") or []
    _, ink = source_ink(gray, records)
    result, cache = [], {}
    source_support = {}
    # Reuse complete, independently observed finite support certificates. A
    # thick material edge intersected by hatching can defeat the generic
    # single-ink-run measurement while still have an unambiguous axial stroke
    # or a verified angle's finite straight support. Neither proof uses nominal
    # dimensions to choose a boundary, and their admission gates stay intact.
    for axis in ("x", "y"):
        for location in source_derived_locations(gray, records, graph, transform, band, axis):
            if location["anchor"]["type"] != "support_level":
                continue
            evidence = location["evidence"]
            source_support.setdefault(location["anchor"]["entity_id"], []).append({
                "verified": True, "method": "reused_finite_source_axis_support",
                "source_certificate": evidence, "nominal_used": False})
    for observation in angle_observations:
        evidence = observation.get("evidence") or {}
        if (observation.get("verified") is not True or evidence.get("verified") is not True
                or evidence.get("nominal_used_to_rank") is not False):
            continue
        for target in observation.get("target_candidates") or []:
            if target.get("entity_type") != "LINE" or target.get("whole_line_supported") is not True:
                continue
            source_support.setdefault(target["entity_id"], []).append({
                "verified": True, "method": "reused_finite_source_angular_line_support",
                "record_id": observation["record_id"], "source_line": observation["source_line"],
                "source_target": target, "nominal_used": False})
    for index, arc in enumerate(entities):
        if arc.get("type") != "ARC" or len(entities) < 3:
            continue
        left, right = entities[index-1], entities[(index+1) % len(entities)]
        if (left.get("type") != "LINE" or right.get("type") != "LINE"
                or left.get("end_node") != arc.get("start_node")
                or arc.get("end_node") != right.get("start_node")):
            continue
        point = support_intersection(left, right)
        if point is None or not np.isfinite(point).all():
            continue
        source_point = transform([point])[0]
        arc_ends = transform([arc["start"], arc["end"]])
        chord = float(np.linalg.norm(arc_ends[1]-arc_ends[0]))
        # Limit the construction to the local removed corner, not a remote
        # intersection between arbitrary supporting infinite lines.
        if max(np.linalg.norm(arc_ends-source_point, axis=1)) > max(4*band, 2*chord):
            continue
        support = []
        for line in (left, right):
            if line["id"] not in cache:
                cache[line["id"]] = _straight_source_support(
                    ink, transform([line["start"], line["end"]]), band)
                if not cache[line["id"]]["verified"] and source_support.get(line["id"]):
                    cache[line["id"]] = {"verified": True,
                        "method": "reused_independent_finite_source_line_certificates",
                        "certificates": source_support[line["id"]], "nominal_used": False,
                        "generic_ink_run_diagnostic": cache[line["id"]]}
            support.append(cache[line["id"]])
        if not all(row["verified"] for row in support):
            continue
        result.append({"id": "intersection_"+left["id"]+"_"+right["id"],
            "x": float(point[0]), "y": float(point[1]), "source_px": source_point.tolist(),
            "anchor": {"type": "line_intersection", "entities": [left["id"], right["id"]]},
            "evidence": {"method": "source_supported_theoretical_sharp_corner_v1",
                         "intervening_arc": arc["id"], "source_line_support": support,
                         "nominal_used": False, "coordinates_locked": False}})
    return result
