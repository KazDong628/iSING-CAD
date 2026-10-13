"""Finite dimension-extension ownership of an independently observed turn.

An extension narrows the source inspection region; it cannot manufacture a
curve extremum. Observed coordinates establish ownership only. The returned
CAD anchor remains symbolic and is evaluated from the current unknown curve.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from .execution_control import execution_checkpoint
from .measurement_anchors import (
    _arc_parameters, _cardinal_angle, derived_anchor_point)


def _witness_valid(witness, axis):
    proof = witness.get('binding_evidence') or {}
    arrows = proof.get('endpoint_arrows') or []
    line = witness.get('line') or {}
    if (witness.get('axis') != axis or proof.get('verified') is not True
            or proof.get('ambiguous') or proof.get('search_complete') is False
            or (witness.get('axis_association') or {}).get('ambiguous')
            or len(arrows) != 2 or not all(a.get('verified') is True for a in arrows)
            or set(line.get('extension_stations') or {}) != {'lo', 'hi'}):
        return False
    k = 0 if axis == 'x' else 1
    try:
        tips = np.asarray([a['tip_px'] for a in arrows], float)
        directions = np.asarray([a['direction_px'] for a in arrows], float)
        if (tips.shape != (2, 2) or directions.shape != (2, 2)
                or not np.isfinite(np.r_[tips.ravel(), directions.ravel()]).all()
                or float(line['hi'])-float(line['lo']) < 12.):
            return False
        directions /= np.maximum(np.linalg.norm(directions, axis=1)[:, None], 1e-12)
        for index, end in enumerate(('lo', 'hi')):
            extension = line['extension_stations'][end]
            if (abs(tips[index, k]-float(line[end])) > 3.
                    or abs(tips[index, 1-k]-float(line['cross'])) > 3.
                    or abs(directions[index, k]) < .9
                    or abs(float(extension['cross'])-float(line[end])) > 2.
                    or float(extension['hi'])-float(extension['lo']) < 12.
                    or not float(extension['lo'])-2 <= float(line['cross']) <= float(extension['hi'])+2):
                return False
        return float(directions[0]@directions[1]) < -.9
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return False


def _text_exclusion(gray, records):
    excluded = np.zeros(gray.shape, np.uint8)
    for record in records:
        box = np.asarray(record.get('box') or [], float)
        if box.ndim == 2 and box.shape[1:] == (2,) and np.isfinite(box).all():
            cv2.fillPoly(excluded, [np.rint(box).astype(np.int32)], 1)
    return excluded.astype(bool)


def _exclusion(gray, records, witness, axis):
    excluded = _text_exclusion(gray, records).astype(np.uint8)
    k = 0 if axis == 'x' else 1
    for extension in witness['line']['extension_stations'].values():
        width = min(6., max(1.5, float(extension.get('thickness', 1))/2+1.))
        corners = np.zeros((2, 2), float)
        corners[:, k] = [extension['cross']-width, extension['cross']+width]
        corners[:, 1-k] = [extension['lo']-1., extension['hi']+1.]
        cv2.rectangle(excluded, tuple(np.floor(corners[0]).astype(int)),
                      tuple(np.ceil(corners[1]).astype(int)), 1, -1)
    return excluded.astype(bool)


def _finite_attachment(distance, text, observed, extension, line, axis_index):
    start, stop = float(line['cross']), float(observed[1-axis_index])
    samples = np.empty((max(2, int(abs(stop-start))+1), 2), float)
    samples[:, 1-axis_index] = np.linspace(start, stop, len(samples))
    samples[:, axis_index] = float(extension['cross'])
    pixels = np.rint(samples).astype(int)
    h, w = distance.shape
    if ((pixels < 0).any() or (pixels[:, 0] >= w).any() or (pixels[:, 1] >= h).any()):
        return False, {}
    visible = ~text[pixels[:, 1], pixels[:, 0]]
    supported = distance[pixels[:, 1], pixels[:, 0]] <= 2.
    longest = current = 0
    for seen, occupied in zip(visible, supported):
        current = current+1 if seen and not occupied else 0
        longest = max(longest, current)
    fraction = float(supported[visible].mean()) if visible.any() else 0.
    audit = {'source_extension_visible_sample_count': int(visible.sum()),
             'source_extension_support_fraction': fraction,
             'source_extension_maximum_unobserved_gap_px': longest}
    return bool(visible.sum() >= 8 and visible.mean() >= .4 and fraction >= .9 and longest <= 4), audit


def _track(ink, excluded, crosses, expected, axis_index, direction, radius):
    """Observe the first dark edge from a clean white side, not a nearest run.

    Excluded text/extension pixels are barriers. A hidden curve cannot be
    replaced by a hatch farther down the same scan. The other side of a thick
    stroke may be merged with hatching, so complete dark-run uniqueness is not
    required by this separate, finite-turn proof.
    """
    points = []
    h, w = ink.shape
    offsets = np.arange(-radius, radius+1., 1.)[::direction]
    for cross, prediction in zip(crosses, expected):
        pixels = np.empty((len(offsets), 2), int)
        pixels[:, axis_index] = np.rint(prediction+offsets).astype(int)
        pixels[:, 1-axis_index] = int(round(cross))
        if ((pixels < 0).any() or (pixels[:, 0] >= w).any() or (pixels[:, 1] >= h).any()):
            continue
        white = 0
        for point in pixels:
            x, y = point
            if excluded[y, x]:
                break
            if ink[y, x]:
                if white >= 4:
                    points.append(point.astype(float))
                break
            white += 1
    return np.asarray(points, float).reshape((-1, 2))


def _turn(points, axis_index, expected_sign, extension, band):
    result = {'verified': False, 'reason': 'insufficient_finite_source_turn_samples',
              'source_sample_count': len(points)}
    if len(points) < 32:
        return None, result
    cross = points[:, 1-axis_index]
    midpoint = float(np.mean(cross))
    span = float(np.ptp(cross))/2
    if span < max(12., 2*band):
        return None, result
    parameter = (cross-midpoint)/span
    keep = np.ones(len(points), bool)
    for _ in range(3):
        coefficients = np.polynomial.polynomial.polyfit(parameter[keep], points[keep, axis_index], 2)
        residuals = np.abs(np.polynomial.polynomial.polyval(parameter, coefficients)-points[:, axis_index])
        median = float(np.median(residuals[keep]))
        keep = residuals <= max(1.5, 4*median)
    residual = float(np.quantile(residuals[keep], .9))
    curvature = float(coefficients[2])
    result.update(source_inlier_count=int(keep.sum()), source_half_span_px=span,
                  source_fit_p90_residual_px=residual, source_turn_depth_px=abs(curvature))
    if (keep.sum() < 32 or keep.mean() < .8 or residual > 1.5
            or curvature*expected_sign >= 0 or abs(curvature) < max(2., 4*residual)):
        result['reason'] = 'continuous_source_turn_direction_or_fit_unconfirmed'
        return None, result
    parameter_vertex = -float(coefficients[1])/(2*curvature)
    vertex_cross = midpoint+parameter_vertex*span
    vertex_axis = float(np.polynomial.polynomial.polyval(parameter_vertex, coefficients))
    inliers = points[keep]
    left = inliers[inliers[:, 1-axis_index] < vertex_cross]
    right = inliers[inliers[:, 1-axis_index] > vertex_cross]
    if (min(len(left), len(right)) < 12 or
            min(vertex_cross-float(cross[keep].min()), float(cross[keep].max())-vertex_cross) < max(12., 2*band)):
        result['reason'] = 'source_turn_not_observed_on_both_sides'
        return None, result
    # A hatch envelope is not a single ordered curved boundary.
    ordered = inliers[np.argsort(inliers[:, 1-axis_index])]
    slopes = np.abs(np.diff(ordered[:, axis_index])/np.diff(ordered[:, 1-axis_index]))
    if np.quantile(slopes, .95) > 1.25:
        result['reason'] = 'source_track_not_a_continuous_local_curve'
        return None, result
    uncertainty = max(2., residual*3+1.)
    if (abs(vertex_axis-float(extension['cross'])) > max(3., uncertainty)
            or not float(extension['lo'])+1 <= vertex_cross <= float(extension['hi'])-1):
        result['reason'] = 'finite_extension_does_not_attach_to_actual_turn'
        return None, result
    vertex = np.empty(2, float)
    vertex[axis_index], vertex[1-axis_index] = vertex_axis, vertex_cross
    result.update(verified=True, reason='finite_extension_and_two_sided_source_turn_verified',
                  source_position_uncertainty_px=uncertainty,
                  source_transverse_interval_px=[float(cross[keep].min()), float(cross[keep].max())])
    return vertex, result


def dimension_supported_extrema(gray, records, graph, transform, band, axis, witnesses, *, diagnostics=None):
    """Return source-supported symbolic ARC extrema at finite dimension ends.

    Optional ``diagnostics`` receives bounded rejection reasons. No nominal,
    reference geometry, or nearest-prediction ranking selects an observation.
    """
    if axis not in {'x', 'y'} or not math.isfinite(float(band)) or band <= 0:
        raise ValueError('Dimension extremum support requires an axis and positive finite band')
    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise ValueError('Dimension extremum support requires a grayscale source image')
    k = 0 if axis == 'x' else 1
    basis = np.asarray(transform([[0., 0.], [1., 0.], [0., 1.]]), float)
    vectors = basis[1:]-basis[0]
    if (vectors.shape != (2, 2) or not np.isfinite(vectors).all()
            or min(abs(vectors[0, 0]), abs(vectors[1, 1])) <= 1e-12
            or abs(vectors[0, 1])+abs(vectors[1, 0]) > 1e-4):
        return []
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    distance = cv2.distanceTransform((1-ink).astype(np.uint8), cv2.DIST_L2, 3)
    text = _text_exclusion(gray, records)
    output = []
    for witness in witnesses:
        execution_checkpoint('dimension_extremum.witness')
        if not _witness_valid(witness, axis):
            if diagnostics is not None:
                diagnostics.append({'record_id': witness.get('record_id'), 'reason': 'dimension_arrow_pair_or_finite_extensions_unverified'})
            continue
        excluded = _exclusion(gray, records, witness, axis)
        for end, extension in witness['line']['extension_stations'].items():
            candidates = []
            for entity in graph.get('entities') or []:
                if entity.get('type') != 'ARC':
                    continue
                for sense in ('minimum', 'maximum'):
                    execution_checkpoint('dimension_extremum.arc')
                    anchor = {'type': 'curve_extremum', 'entity_id': entity['id'], 'axis': axis, 'sense': sense}
                    try:
                        point = derived_anchor_point(anchor, {entity['id']: entity})
                        center, radius, start, sign, sweep = _arc_parameters(entity)
                        cardinal = _cardinal_angle(anchor)
                        travel = (sign*(cardinal-start)) % (2*math.pi)
                        half_angle = min(.5, travel*.95, (sweep-travel)*.95)
                        proposed = np.asarray(transform([point]), float)[0]
                        if half_angle <= 0 or abs(proposed[k]-extension['cross']) > max(4., 2*band):
                            continue
                        samples = center+radius*np.c_[np.cos(np.linspace(cardinal-half_angle, cardinal+half_angle, 65)),
                                                    np.sin(np.linspace(cardinal-half_angle, cardinal+half_angle, 65))]
                        pixels = np.asarray(transform(samples), float)
                        # The finite extension must own the turn itself. Its
                        # segment need not cover both curved shoulders: a
                        # short extension often stops at the tangent point.
                        # Inspect bounded adjoining curve context, still
                        # clipped to the declared finite ARC branch.
                        # Proposal uncertainty is a normal-direction ownership
                        # corridor, not the length needed to observe curvature.
                        # Use the finite curve's two-sided source span, bounded
                        # to 128 pixels of adjoining context. The minimum
                        # 32-pixel context supports the unchanged 32-sample
                        # proof; it neither changes dimensional tolerances nor
                        # allows an extension to own a turn outside its ends.
                        context = min(128., max(32., float(np.ptp(pixels[:, 1-k]))/2))
                        low = max(float(pixels[:, 1-k].min()), float(extension['lo'])-context)
                        high = min(float(pixels[:, 1-k].max()), float(extension['hi'])+context)
                        if high-low < max(32., 4*band):
                            continue
                        first, last = math.ceil(low), math.floor(high)
                        crosses = np.linspace(first, last, min(257, last-first+1))
                        center_px = np.asarray(transform([center]), float)[0]
                        radii = np.abs(np.diag(vectors))*radius
                        radial_sign = math.copysign(1., proposed[k]-center_px[k])
                        expected = center_px[k]+radial_sign*radii[k]*np.sqrt(np.maximum(0., 1.-((crosses-center_px[1-k])/radii[1-k])**2))
                        tracks = []
                        # Observing a white-to-ink transition needs four
                        # genuinely white pixels before the edge. This is an
                        # observation aperture, separate from the unchanged
                        # station/ownership and native-constraint tolerances.
                        aperture = max(8., math.ceil(2*band))+4.
                        for direction in (1, -1):
                            points = _track(ink, excluded, crosses, expected, k, direction, aperture)
                            observed, proof = _turn(points, k, radial_sign, extension, band)
                            if observed is not None:
                                tracks.append((observed, proof))
                            elif diagnostics is not None:
                                diagnostics.append({'record_id': witness.get('record_id'), 'endpoint': end,
                                    'entity_id': entity['id'], 'sense': sense, **proof})
                        if not tracks:
                            continue
                        if len(tracks) == 2 and np.linalg.norm(tracks[0][0]-tracks[1][0]) > 5.:
                            if diagnostics is not None:
                                diagnostics.append({'record_id': witness.get('record_id'), 'endpoint': end,
                                    'entity_id': entity['id'], 'reason': 'competing_source_curve_tracks'})
                            continue
                        observed = np.mean([row[0] for row in tracks], axis=0)
                        proof = tracks[0][1]
                        attached, attachment_proof = _finite_attachment(
                            distance, text, observed, extension, witness['line'], k)
                        if not attached:
                            if diagnostics is not None:
                                diagnostics.append({'record_id': witness.get('record_id'), 'endpoint': end,
                                    'entity_id': entity['id'], 'reason': 'finite_extension_ink_attachment_unconfirmed',
                                    **attachment_proof})
                            continue
                        candidates.append({'id': 'extremum_'+entity['id']+'_'+axis+'_'+sense,
                            'x': float(point[0]), 'y': float(point[1]), 'source_px': observed.tolist(),
                            'anchor': anchor, 'evidence': {**proof, **attachment_proof,
                                'method': 'finite_dimension_extension_two_sided_source_turn_v1',
                                'dimension_record_id': witness['record_id'], 'dimension_endpoint': end,
                                'measurement_axis': axis, 'source_supported_entities': [entity['id']],
                                'source_observation_aperture_px': aperture,
                                'source_attachment_interval_px': [float(extension['lo']), float(extension['hi'])],
                                'observed_source_level_px': float(observed[k]),
                                'finite_arc_domain_checked': True, 'arrow_pair_verified': True,
                                'section_interpretation_not_resolved': False, 'nominal_used': False,
                                'coordinates_locked': False, 'reference_geometry_used': False}})
                    except (ValueError, KeyError, TypeError, ArithmeticError):
                        continue
            if len(candidates) == 1:
                candidate = candidates[0]
                if not any(old['anchor'] == candidate['anchor'] for old in output):
                    output.append(candidate)
            elif candidates and diagnostics is not None:
                diagnostics.append({'record_id': witness.get('record_id'), 'endpoint': end,
                                    'reason': 'multiple_finite_curve_extremum_owners'})
            elif diagnostics is not None:
                diagnostics.append({'record_id': witness.get('record_id'), 'endpoint': end,
                                    'reason': 'no_unique_source_supported_extremum_at_dimension_endpoint'})
    return output
