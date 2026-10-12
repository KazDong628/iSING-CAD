"""Conservative source-pixel checks for optional geometric relations.

A fitted graph suggests where to inspect, never supplies its own proof. These
checks neither read reference CAD nor use nominal dimensions. Missing/ambiguous
strokes stay unresolved instead of becoming hard solver constraints.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .dimension_evidence import _axis_lines


def source_ink(gray, records):
    cleaned = gray.copy()
    text_mask = np.zeros_like(gray, np.uint8)
    for row in records:
        try:
            box = np.asarray(row.get("box"), float)
            if box.ndim == 2 and box.shape[1] == 2 and len(box) >= 3 and np.isfinite(box).all():
                cv2.fillPoly(cleaned, [np.rint(box).astype(np.int32)], 255)
                cv2.fillPoly(text_mask, [np.rint(box).astype(np.int32)], 255)
        except (TypeError, ValueError):
            continue
    # Loose OCR polygons sometimes cover a real long boundary. Retain its
    # actually observed ink only when most of that same continuous axial stroke
    # lies outside text. A glyph wholly inside an OCR box is never restored.
    for axis in ("x", "y"):
        for line in _axis_lines(gray, axis):
            if line["span"] < max(32., max(gray.shape) * .04):
                continue
            half = line["thickness"] / 2
            limits = ([line["lo"], line["cross"] - half, line["hi"] + 1, line["cross"] + half + 1]
                      if axis == "x" else
                      [line["cross"] - half, line["lo"], line["cross"] + half + 1, line["hi"] + 1])
            x0, y0, x1, y1 = [int(round(v)) for v in limits]
            x0, y0, x1, y1 = max(0,x0), max(0,y0), min(gray.shape[1],x1), min(gray.shape[0],y1)
            observed = gray[y0:y1,x0:x1] < 170
            outside = text_mask[y0:y1,x0:x1] == 0
            if observed.any() and float(np.count_nonzero(observed & outside)) / np.count_nonzero(observed) >= .7:
                region = cleaned[y0:y1,x0:x1]
                region[observed] = gray[y0:y1,x0:x1][observed]
    _, ink = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return cleaned, ink > 0


def axis_stroke_evidence(samples, axis, lines, band):
    """Require one measured axial ink strip covering the material-edge proposal."""
    along, across = (0, 1) if axis == "horizontal" else (1, 0)
    low, high = float(samples[:, along].min()), float(samples[:, along].max())
    span = high - low
    result = {"method": "unique_source_axis_stroke_v1", "verified": False,
              "span_px": span, "proposal_band_px": band,
              "reason": "source_axis_stroke_unavailable"}
    if span < max(24., 4 * band):
        result["reason"] = "source_axis_span_too_short"
        return result
    options = []
    for line in lines:
        half = max(1., float(line["thickness"])) / 2
        gap = np.maximum(np.abs(samples[:, across] - line["cross"]) - half, 0.)
        coverage = max(0., min(high, line["hi"]) - max(low, line["lo"])) / span
        # Endpoint-sized raster omissions are allowed, but an isolated dimension
        # crossing or short hatch fragment cannot certify an entire boundary.
        if coverage >= .94 and float(gap.max()) <= band:
            options.append({**line, "covered_fraction": coverage,
                            "maximum_projection_gap_px": float(gap.max())})
    result["supporting_strokes"] = options
    if len(options) != 1:
        if options:
            result["reason"] = "multiple_parallel_source_strokes"
        return result
    result.update(verified=True, reason=None, observed_station_px=options[0]["cross"])
    return result


def _ink_trace(ink, samples, band, *, span_fraction=.45):
    """Measure single connected ink runs across normals, reject local crossings."""
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(samples, axis=0), axis=1))]
    span = min(96., float(distances[-1]) * span_fraction)
    if span < max(24., 4 * band):
        return {"verified": False, "reason": "insufficient_local_source_span", "span_px": span}
    stations = np.linspace(1.5, span, 32)
    seed = np.c_[np.interp(stations, distances, samples[:, 0]),
                 np.interp(stations, distances, samples[:, 1])]
    tangent = np.gradient(seed, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-9)
    normals = np.c_[-tangent[:, 1], tangent[:, 0]]
    extent = max(4, int(math.ceil(band)))
    offsets = np.arange(-extent, extent + 1, dtype=float)
    measured, kept, widths = [], [], []
    for index, (point, normal) in enumerate(zip(seed, normals)):
        cross = np.rint(point + offsets[:, None] * normal).astype(int)
        valid = ((cross[:, 0] >= 0) & (cross[:, 0] < ink.shape[1]) &
                 (cross[:, 1] >= 0) & (cross[:, 1] < ink.shape[0]))
        if not valid.all():
            continue
        occupied = ink[cross[:, 1], cross[:, 0]]
        starts = np.flatnonzero(occupied & ~np.r_[False, occupied[:-1]])
        ends = np.flatnonzero(occupied & ~np.r_[occupied[1:], False])
        # Never choose the run nearest the fitted curve from several alternatives:
        # that would make the prior decide which hatch/dimension stroke is truth.
        if len(starts) != 1 or starts[0] == 0 or ends[0] == len(offsets) - 1:
            continue
        width = int(ends[0] - starts[0] + 1)
        if width > max(7., band):
            continue
        measured.append(point + float(offsets[starts[0]:ends[0] + 1].mean()) * normal)
        kept.append(index)
        widths.append(width)
    result = {"method": "unambiguous_normal_ink_runs_quadratic_tangent_v1", "verified": False,
              "span_px": span, "sample_count": len(stations), "unambiguous_samples": len(kept),
              "reason": "source_junction_strokes_ambiguous"}
    if len(kept) < 24 or min(kept, default=32) > 3 or max(kept, default=0) < 28:
        return result
    points = np.asarray(measured)
    parameter = stations[kept] / span
    # Estimate the derivative at the joint from source ink on two independently
    # chosen scales. A corner hidden by one broad fit must not pass as tangency.
    def fit(keep):
        coefficients = np.polynomial.polynomial.polyfit(parameter[keep], points[keep], 2)
        fitted = np.polynomial.polynomial.polyval(parameter[keep], coefficients).T
        residual = np.linalg.norm(points[keep] - fitted, axis=1)
        direction = coefficients[1]
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        return direction, float(np.percentile(residual, 90))
    full, residual = fit(np.ones(len(points), bool))
    short_mask = parameter <= .75
    if int(short_mask.sum()) < 16:
        return result
    short, short_residual = fit(short_mask)
    stability = math.degrees(math.acos(float(np.clip(full @ short, -1., 1.))))
    result.update(tangent_direction_px=full.tolist(), fit_residual_p90_px=residual,
                  short_fit_residual_p90_px=short_residual, scale_disagreement_degrees=stability)
    if max(residual, short_residual) > max(1.25, float(np.median(widths)) / 2):
        result["reason"] = "source_local_curve_fit_unstable"
    elif stability > 2.:
        result["reason"] = "source_tangent_scale_disagreement"
    else:
        result.update(verified=True, reason=None)
    return result


def _bounded_source_ink_tangent_trace(ink, samples, band, *, strip_association=False, full_finite_window=False,
                                    circle_model=False):
    """Retry an incomplete measurement with bounded, unclipped ink sections.

    The search aperture is wider than the allowed centre displacement: observing
    both sides of a thick stroke does not authorize moving the source boundary.
    Window choice is made per side, before comparing the two joint directions.
    No CAD tangent or annotation value participates in the measurement.
    """
    previous = _ink_trace(ink, samples, band,span_fraction=.85 if full_finite_window else .45)
    if previous.get("verified") and not circle_model:
        return previous
    samples = np.asarray(samples, dtype=float)
    if (samples.ndim != 2 or samples.shape[1] != 2 or len(samples) < 3
            or not np.isfinite(samples).all() or not math.isfinite(band) or band <= 0):
        return previous
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(samples, axis=0), axis=1))]
    maximum = min(96., float(distances[-1]) * .85)
    minimum = max(24., 4 * band)
    windows = sorted({maximum, 24., 32., 48., 64., 96.}, reverse=True)
    windows = [span for span in windows if minimum <= span <= maximum]
    attempts = []
    # A strip retry must observe both sides before measuring its width. This
    # aperture is not an acceptance distance: association still uses `band`.
    extent = max(8, int(math.ceil(2 * band + (max(7., band) / 2 if strip_association else 0.))))
    offsets = np.arange(-extent, extent + 1, dtype=float)
    for span in windows:
        stations = np.linspace(1.5, span, 96)
        seed = np.c_[np.interp(stations, distances, samples[:, 0]),
                     np.interp(stations, distances, samples[:, 1])]
        # Consecutive subpixel samples of a raster mask inherit its horizontal
        # and vertical stair steps. Their one-step normals can run ALONG the
        # real drawn stroke and falsely report a clipped thick run. Use a
        # finite local chord only to orient the inspection sections; the
        # measured centres and the final tangent still come from original ink.
        spacing=max(1e-8,float(np.median(np.linalg.norm(np.diff(seed,axis=0),axis=1))))
        window=max(2,min(len(seed)//6,int(math.ceil(max(3.,band)/spacing))))
        locations=np.arange(len(seed))
        direction=seed[np.minimum(len(seed)-1,locations+window)]-seed[np.maximum(0,locations-window)]
        direction /= np.maximum(np.linalg.norm(direction, axis=1)[:, None], 1e-9)
        normals = np.c_[-direction[:, 1], direction[:, 0]]
        measured, kept, widths, lateral_offsets = [], [], [], []
        rejected = {key: 0 for key in ("outside_image", "no_ink", "multiple_runs",
                                      "clipped_run", "wide_run", "centre_outside_band")}
        for index, (point, normal) in enumerate(zip(seed, normals)):
            cross = np.rint(point + offsets[:, None] * normal).astype(int)
            if not ((cross[:, 0] >= 0) & (cross[:, 0] < ink.shape[1]) &
                    (cross[:, 1] >= 0) & (cross[:, 1] < ink.shape[0])).all():
                rejected["outside_image"] += 1
                continue
            occupied = ink[cross[:, 1], cross[:, 0]]
            starts = np.flatnonzero(occupied & ~np.r_[False, occupied[:-1]])
            ends = np.flatnonzero(occupied & ~np.r_[occupied[1:], False])
            if len(starts) != 1:
                rejected["no_ink" if not len(starts) else "multiple_runs"] += 1
                continue
            if starts[0] == 0 or ends[0] == len(offsets) - 1:
                rejected["clipped_run"] += 1
                continue
            width = int(ends[0] - starts[0] + 1)
            centre = float(offsets[starts[0]:ends[0] + 1].mean())
            if width > max(7., band):
                rejected["wide_run"] += 1
                continue
            association_gap = (max(abs(centre) - (width - 1) / 2., 0.)
                               if strip_association else abs(centre))
            if association_gap > band:
                rejected["centre_outside_band"] += 1
                continue
            measured.append(point + centre * normal)
            kept.append(index)
            widths.append(width)
            lateral_offsets.append(centre)
        row = {"span_px": span, "sample_count": len(stations),
               "unambiguous_samples": len(kept), "rejected_samples": rejected,
               "verified": False, "reason": "source_junction_strokes_ambiguous"}
        attempts.append(row)
        # Same 75% support and endpoint coverage as the original 24/32 test.
        if len(kept) < 72 or min(kept, default=96) > 9 or max(kept, default=0) < 86:
            continue
        if strip_association:
            # Width explains a mask following one side of a stroke. It cannot
            # explain a path walking across and then away from a nearby stroke.
            # Bound robust signed-offset drift by the observed strip width,
            # rather than consulting the mask derivative or a CAD tangent.
            low, high = np.percentile(lateral_offsets, [10., 90.])
            observed_width = float(np.median(widths))
            row.update(source_strip_lateral_offset_span_px=float(high - low),
                       source_strip_observed_width_px=observed_width)
            if high - low > observed_width:
                row["reason"] = "source_strip_association_drifts_across_stroke"
                continue
        points = np.asarray(measured)
        parameter = stations[kept] / span
        if circle_model:
            from .source_circle_tangent import circular_ink_derivative
            measured_circle = circular_ink_derivative(points, stations[kept], widths)
            row['circular_source_measurement'] = measured_circle
            if measured_circle['verified']:
                row.update(measured_circle)
                row.update(verified=True, reason=None)
                break
            # Do not pick a quadratic derivative to rescue this circular fit.
            # The caller retains the independently measured old alternative.
            continue
        def fit(selected):
            coefficients = np.polynomial.polynomial.polyfit(parameter[selected], points[selected], 2)
            fitted = np.polynomial.polynomial.polyval(parameter[selected], coefficients).T
            vector = coefficients[1]
            vector /= max(float(np.linalg.norm(vector)), 1e-9)
            return vector, float(np.percentile(np.linalg.norm(points[selected] - fitted, axis=1), 90))
        short = parameter <= .75
        if int(short.sum()) < 48:
            continue
        full, residual = fit(np.ones(len(points), bool))
        near, near_residual = fit(short)
        stability = math.degrees(math.acos(float(np.clip(full @ near, -1., 1.))))
        row.update(tangent_direction_px=full.tolist(), fit_residual_p90_px=residual,
                   short_fit_residual_p90_px=near_residual, scale_disagreement_degrees=stability)
        if max(residual, near_residual) > max(1.25, float(np.median(widths)) / 2):
            row["reason"] = "source_local_curve_fit_unstable"
        elif stability > 2.:
            row["reason"] = "source_tangent_scale_disagreement"
        else:
            row.update(verified=True, reason=None)
            break
    selected = next((row for row in attempts if row["verified"]), None)
    return {**(selected or {"verified": False, "reason": "source_tangent_multiscale_evidence_insufficient"}),
            "method": ("bounded_multiscale_unique_source_strip_tangent_v1" if strip_association
                       else "bounded_multiscale_unique_source_ink_tangent_v1"),
            "measurement_aperture_half_width_px": extent,
            **({"source_ink_strip_gap_budget_px": band,
                "association_distance": "proposal_to_observed_ink_strip_edge",
                "stroke_center_displacement_is_not_geometry_permission": True}
               if strip_association else {"centre_displacement_budget_px": band}),
            "section_normal_basis":"finite_local_source_path_chord; original_ink_centres_only",
            "window_selection": "longest_independently_verified_finite_source_window",
            "window_attempts": attempts, "previous_measurement": previous,
            "independent_source_tangent_measurement": True,
            "fitted_tangency_used_as_evidence": False}


def _connected_source_ink_tangent_trace(ink, samples, band):
    """Resolve intermittent hatch runs by source continuity, never proximity.

    A normal section may contain both a boundary and a disconnected hatch.
    Discarding the whole section loses real boundary evidence. Here every
    bounded narrow run is retained, and only one source-continuous track with
    the original coverage, endpoint, residual and multiscale gates can pass.
    Parallel tracks and branching sections remain ambiguous. No CAD tangent,
    nominal radius or reference geometry participates in this measurement.
    """
    samples = np.asarray(samples, float)
    failure = {"method": "unique_connected_source_ink_track_tangent_v1",
               "verified": False, "reason": "source_connected_track_unresolved",
               "independent_source_tangent_measurement": True,
               "fitted_tangency_used_as_evidence": False,
               "track_selection": "unique_source_continuity_component"}
    if (samples.ndim != 2 or samples.shape[1] != 2 or len(samples) < 3
            or not np.isfinite(samples).all() or not math.isfinite(band) or band <= 0):
        return failure
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(samples, axis=0), axis=1))]
    maximum = min(96., float(distances[-1]) * .85)
    windows = sorted({maximum, 24., 32., 48., 64., 96.}, reverse=True)
    extent = max(8, int(math.ceil(2 * band + max(7., band) / 2)))
    offsets = np.arange(-extent, extent + 1, dtype=float)
    attempts = []
    for span in (s for s in windows if max(24., 4*band) <= s <= maximum):
        stations = np.linspace(1.5, span, 96)
        seed = np.c_[np.interp(stations, distances, samples[:, 0]),
                     np.interp(stations, distances, samples[:, 1])]
        spacing = max(1e-8, float(np.median(np.linalg.norm(np.diff(seed, axis=0), axis=1))))
        window = max(2, min(len(seed)//6, int(math.ceil(max(3., band)/spacing))))
        indices = np.arange(len(seed))
        directions = seed[np.minimum(95, indices+window)]-seed[np.maximum(0, indices-window)]
        directions /= np.maximum(np.linalg.norm(directions, axis=1)[:, None], 1e-9)
        normals = np.c_[-directions[:, 1], directions[:, 0]]
        runs = []
        for index, (point, normal) in enumerate(zip(seed, normals)):
            cross = np.rint(point + offsets[:, None]*normal).astype(int)
            if not ((cross[:, 0] >= 0) & (cross[:, 0] < ink.shape[1]) &
                    (cross[:, 1] >= 0) & (cross[:, 1] < ink.shape[0])).all():
                continue
            occupied = ink[cross[:, 1], cross[:, 0]]
            starts = np.flatnonzero(occupied & ~np.r_[False, occupied[:-1]])
            ends = np.flatnonzero(occupied & ~np.r_[occupied[1:], False])
            for start, end in zip(starts, ends):
                width = int(end-start+1)
                if start == 0 or end == len(offsets)-1 or width > max(7., band):
                    continue
                centre = float(offsets[start:end+1].mean())
                # Retain competing tracks throughout the inspection aperture.
                # Filtering them by fitted-path distance would let the prior
                # choose ownership before source uniqueness is established.
                runs.append((index, centre, width, point+centre*normal))
        parents = list(range(len(runs)))
        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index
        # The gap is bounded in observed source arclength. A disconnected
        # branch cannot bridge across the entire inspection window.
        for index, run in enumerate(runs):
            for earlier in range(index-1, -1, -1):
                prior = runs[earlier]
                gap = run[0]-prior[0]
                if gap > 8:
                    break
                if gap <= 0:
                    continue
                travel = stations[run[0]]-stations[prior[0]]
                if abs(run[1]-prior[1]) <= max(1., travel*.25):
                    parents[find(index)] = find(earlier)
        components = {}
        for index, run in enumerate(runs):
            components.setdefault(find(index), []).append(run)
        tracks = []
        for component in components.values():
            grouped = {}
            for run in component:
                grouped.setdefault(run[0], []).append(run)
            # A source branch is not resolved by choosing its nearest run.
            unique = [members[0] for members in grouped.values() if len(members) == 1]
            unique.sort(key=lambda run: run[0])
            if len(unique) >= 72 and unique[0][0] <= 9 and unique[-1][0] >= 86:
                tracks.append(unique)
        row = {"span_px": span, "sample_count": 96, "component_count": len(components),
               "eligible_track_count": len(tracks), "verified": False,
               "reason": "source_connected_track_ambiguous" if len(tracks)>1 else
                         "source_connected_track_coverage_insufficient"}
        attempts.append(row)
        if len(tracks) != 1:
            continue
        track = tracks[0]
        positions = np.asarray([run[0] for run in track], int)
        centres = np.asarray([run[1] for run in track])
        widths = np.asarray([run[2] for run in track])
        points = np.asarray([run[3] for run in track])
        drift = float(np.ptp(np.percentile(centres, [10., 90.])))
        row.update(unambiguous_samples=len(track), source_strip_lateral_offset_span_px=drift,
                   source_strip_observed_width_px=float(np.median(widths)))
        if np.any(np.maximum(np.abs(centres)-(widths-1)/2., 0.) > band):
            row["reason"] = "source_connected_track_outside_association_band"
            continue
        if drift > float(np.median(widths)):
            row["reason"] = "source_strip_association_drifts_across_stroke"
            continue
        parameter = stations[positions]/span
        near = parameter <= .75
        if int(near.sum()) < 48:
            continue
        def fit(selected):
            coefficients = np.polynomial.polynomial.polyfit(parameter[selected], points[selected], 2)
            fitted = np.polynomial.polynomial.polyval(parameter[selected], coefficients).T
            vector = coefficients[1]
            vector /= max(float(np.linalg.norm(vector)), 1e-9)
            return vector, float(np.percentile(np.linalg.norm(points[selected]-fitted, axis=1), 90))
        full, residual = fit(np.ones(len(points), bool))
        short, short_residual = fit(near)
        stability = math.degrees(math.acos(float(np.clip(full@short, -1., 1.))))
        row.update(tangent_direction_px=full.tolist(), fit_residual_p90_px=residual,
                   short_fit_residual_p90_px=short_residual, scale_disagreement_degrees=stability)
        if max(residual, short_residual) > max(1.25, float(np.median(widths))/2):
            row["reason"] = "source_local_curve_fit_unstable"
        elif stability > 2.:
            row["reason"] = "source_tangent_scale_disagreement"
        else:
            row.update(verified=True, reason=None)
            return {**failure, **row, "window_attempts": attempts,
                    "source_ink_strip_gap_budget_px": band,
                    "measurement_aperture_half_width_px": extent}
    return {**failure, "window_attempts": attempts}


def _source_ink_tangent_trace(ink, samples, band, *, full_finite_window=False):
    """Measure source tangents, with width-aware association only as a retry.

    Material masks can follow a side of a thick boundary stroke rather than its
    centre. Requiring the centre inside the proposal band rejects an observed
    strip that passes the axis-stroke check above. The retry uses the same
    proposal-to-ink-strip distance, while keeping finite width, unique run,
    support coverage, residual and multiscale derivative gates unchanged.
    It does not move graph nodes, enlarge solver bounds, or force tangency.
    """
    previous = _bounded_source_ink_tangent_trace(ink, samples, band,full_finite_window=full_finite_window)
    if previous.get("verified"):
        return previous
    retry = _bounded_source_ink_tangent_trace(ink, samples, band, strip_association=True,
        full_finite_window=full_finite_window)
    if retry.get("verified"):
        return {**retry, "previous_center_band_measurement": previous}
    connected = _connected_source_ink_tangent_trace(ink, samples, band)
    if connected.get("verified"):
        return {**connected, "previous_center_band_measurement": previous,
                "width_aware_strip_retry": retry}
    return {**previous, "width_aware_strip_retry": retry, "connected_track_retry": connected}


def remeasure_finite_source_joint(ink, paths, band):
    """Two independently stable finite-window derivatives, never a G1 fit.

    This retry observes at most 85% of each supplied finite side and 96 pixels.
    Choosing the window does not depend on the measured junction angle. It
    repairs a short-window noise bias without widening the 3-degree gate.
    """
    traces=[]
    for path in paths:
        previous=_source_ink_tangent_trace(ink,path,band,full_finite_window=True)
        circle=_bounded_source_ink_tangent_trace(ink,path,band,full_finite_window=True,circle_model=True)
        traces.append({**circle,'previous_polynomial_measurement':previous}
                      if circle.get('verified') else {**previous,'circular_model_diagnostic':circle})
    angle=None
    if all(t.get('verified') for t in traces):
        angle=math.degrees(math.acos(float(np.clip(-np.dot(
            traces[0]['tangent_direction_px'],traces[1]['tangent_direction_px']),-1.,1.))))
    return dict(method='independent_finite_window_source_joint_v1',sides=traces,
        observed_deviation_degrees=angle,verified=angle is not None and angle<=3.,
        tolerance_degrees=3.,independent_source_tangent_measurement=True,
        fitted_tangency_used_as_evidence=False,nominal_dimensions_used=False,
        reason=None if angle is not None and angle<=3. else
            'source_junction_is_not_tangent' if angle is not None else 'source_tangent_evidence_insufficient')


def structural_evidence(gray, records, graph, transform, sample, band):
    """Return source-validated copies of the graph's structural hypotheses."""
    cleaned, ink = source_ink(gray, records)
    lines = {"horizontal": _axis_lines(cleaned, "x"), "vertical": _axis_lines(cleaned, "y")}
    entities = {e["id"]: e for e in graph.get("entities", [])}
    origin, x_point, y_point = transform([[0., 0.], [1., 0.], [0., 1.]])
    x_axis, y_axis = x_point - origin, y_point - origin
    aligned = (abs(x_axis[1]) <= 1e-6 * max(np.linalg.norm(x_axis), 1e-9) and
               abs(y_axis[0]) <= 1e-6 * max(np.linalg.norm(y_axis), 1e-9))
    result = []
    trace_cache = {}
    for index, original in enumerate(graph.get("relations", [])):
        kind, entity_ids = original.get("type"), original.get("entities", [])
        if kind not in {"horizontal", "vertical", "tangent"} or not all(e in entities for e in entity_ids):
            continue
        item = {"id": original.get("id", f"rel{index:03d}"), "type": kind,
                "entities": entity_ids, "nodes": original.get("nodes", []),
                "source": "geometry_hypothesis", "required": False, "local_reliable": False}
        evidence = {"verified": False, "reason": "relation_entity_type_mismatch"}
        if kind in lines and len(entity_ids) == 1 and entities[entity_ids[0]]["type"] == "LINE":
            evidence = (axis_stroke_evidence(sample(entities[entity_ids[0]], transform), kind, lines[kind], band)
                        if aligned else {"verified": False, "reason": "source_coordinate_axes_not_aligned"})
        elif kind == "tangent" and len(entity_ids) == 2 and len(set(entity_ids)) == 2:
            first, second = [entities[e] for e in entity_ids]
            shared = ({first.get("start_node"), first.get("end_node")} &
                      {second.get("start_node"), second.get("end_node")}) - {None}
            evidence = {"verified": False, "reason": "tangent_requires_unique_shared_node"}
            if len(shared) == 1:
                node = next(iter(shared))
                traces = []
                for entity in (first, second):
                    at_end = entity.get("end_node") == node
                    key = entity["id"], at_end
                    if key not in trace_cache:
                        points = sample(entity, transform)
                        trace_cache[key] = _source_ink_tangent_trace(ink, points[::-1] if at_end else points, band)
                    traces.append(trace_cache[key])
                evidence = {"method": "two_sided_source_tangent_v1", "verified": False,
                            "shared_node": node, "sides": traces, "tolerance_degrees": 3.,
                            "reason": "source_tangent_evidence_insufficient"}
                if all(t["verified"] for t in traces):
                    cosine = float(np.dot(traces[0]["tangent_direction_px"], traces[1]["tangent_direction_px"]))
                    # Both traces run AWAY from a shared node, so a smooth
                    # continuation must be opposite, not merely parallel.
                    angle = math.degrees(math.acos(float(np.clip(-cosine, -1., 1.))))
                    evidence.update(observed_deviation_degrees=angle, verified=angle <= 3.,
                                    reason=None if angle <= 3. else "source_junction_is_not_tangent")
                    if angle>3.:
                        paths=[]
                        for entity in (first,second):
                            points=sample(entity,transform)
                            paths.append(points[::-1] if entity.get('end_node')==node else points)
                        retry=remeasure_finite_source_joint(ink,paths,band)
                        if retry['verified']:
                            evidence={**retry,'shared_node':node,'short_window_measurement':evidence}
                        else:evidence['finite_window_retry']=retry
                item["nodes"] = [node]
        item.update(local_reliable=bool(evidence["verified"]), evidence=evidence)
        result.append(item)
    return result
