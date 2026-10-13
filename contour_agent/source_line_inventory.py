"""Finite original-ink LINE witnesses, independent of dimension binding.

These witnesses protect object type during discrete topology search. They do
not freeze a fitted angle, length or endpoint: numerical attributes still come
from source annotations and the subsequent constrained solve.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.spatial import cKDTree


def _overlap(original, proposed, band, minimum_fraction):
    a, b = np.asarray(original, float)
    p, q = np.asarray(proposed, float)
    length = float(np.linalg.norm(b-a))
    other_length = float(np.linalg.norm(q-p))
    if min(length, other_length) < 1e-8:
        return None
    unit = (b-a)/length
    if abs(float(unit @ ((q-p)/other_length))) < math.cos(math.radians(3.)):
        return None
    lateral = np.abs((np.asarray([p, q])-a) @ [-unit[1], unit[0]])
    if float(lateral.max()) > band:
        return None
    along = (np.asarray([p, q])-a) @ unit
    first, last = max(0., float(along.min())), min(length, float(along.max()))
    return (first, last) if last-first >= minimum_fraction*length else None


def source_line_witnesses(gray, source_entities, boundary, grid, stroke):
    """Require a finite Hough stroke AND immutable boundary AND original ink.

    A CAD chord by itself is insufficient. Hatch/extension lines away from the
    material boundary, unsupported fitted LINEs and short raster steps cannot
    establish witnesses. Hough searches and sampling are explicitly bounded.
    """
    minimum_length = max(16., 8.*grid)
    # Hough observes a drawn stroke's Canny edge, whereas the mask stores its
    # centre. Accommodate stroke width and one raster cell on either side.
    band = max(3., 2.*grid)
    edges = cv2.Canny(gray, 60, 160)
    # Immutable polylines may store a long straight edge with only its two
    # endpoints. Vertex-to-point distance would incorrectly erase that edge's
    # support. Densify without moving any source segment, with a bounded size.
    boundary = np.asarray(boundary, float)
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(boundary, axis=0), axis=1))]
    count = min(100000, max(2, int(math.ceil(distances[-1]/max(.5, .5*grid)))+1))
    positions = np.linspace(0., distances[-1], count)
    dense = np.c_[np.interp(positions, distances, boundary[:, 0]),
                  np.interp(positions, distances, boundary[:, 1])]
    tree = cKDTree(dense)
    witnesses = []
    for index, entity in enumerate(source_entities):
        if entity['type'] != 'LINE':
            continue
        segment = np.asarray([entity['start'], entity['end']], float)
        length = float(np.linalg.norm(segment[1]-segment[0]))
        if length < minimum_length:
            continue
        # A full-page Hough accumulator lets unrelated hatching/extension
        # strokes starve a short real boundary. Inspect each finite proposal's
        # original-pixel window instead; geometry still needs BOTH independent
        # ink and immutable boundary support below.
        low = np.maximum(0, np.floor(segment.min(axis=0)-max(8., 3.*grid))).astype(int)
        high = np.minimum([gray.shape[1], gray.shape[0]],
                          np.ceil(segment.max(axis=0)+max(8., 3.*grid))).astype(int)
        if np.prod(high-low) > 2_000_000:
            continue
        crop = edges[low[1]:high[1], low[0]:high[0]]
        detected = cv2.HoughLinesP(crop, 1., np.pi/720., 12,
                                  minLineLength=max(10., .65*length),
                                  maxLineGap=max(4., 2.*grid))
        segments = ([] if detected is None else
                    detected[:128, 0].reshape(-1, 2, 2)+low)
        ink_segments = [s for s in segments if _overlap(segment, s, band, .7) is not None]
        if not ink_segments:
            continue
        samples = np.linspace(*segment, min(2048, max(8, int(length)+1)))
        boundary_distance = tree.query(samples)[0]
        support = stroke.summarize(samples)
        if (float(np.quantile(boundary_distance, .9)) > max(1.5, grid)
                or support['stroke_supported_fraction'] < .85):
            continue
        witnesses.append({'id': f'source-line-{index:03d}',
                          'source_segment_px': segment.tolist(),
                          'length_px': length,
                          'hough_segment_px': np.asarray(ink_segments[0], float).tolist(),
                          'boundary_p90_distance_px': float(np.quantile(boundary_distance, .9)),
                          'stroke_supported_fraction': support['stroke_supported_fraction']})
    return witnesses


def line_preservation(witnesses, source_entities, grid):
    """Allow splitting/merging collinear LINEs, but reject replacing one by ARC.

    Coverage uses the union of finite projected intervals, so two copies of one
    fragment cannot masquerade as retaining the whole original straight run.
    """
    rows = []
    for witness in witnesses:
        intervals = []
        for entity in source_entities:
            if entity['type'] == 'LINE':
                interval = _overlap(witness['source_segment_px'],
                                    [entity['start'], entity['end']], max(3., 2.*grid), 0.)
                if interval is not None and interval[1] > interval[0]:
                    intervals.append(interval)
        covered, end = 0., -math.inf
        for first, last in sorted(intervals):
            covered += max(0., last-max(first, end))
            end = max(end, last)
        fraction = min(1., covered/witness['length_px'])
        rows.append({'witness_id': witness['id'], 'line_coverage_fraction': fraction,
                     'preserved': fraction >= .7})
    return {'passed': all(row['preserved'] for row in rows),
            'witnesses': witnesses, 'measurements': rows,
            'lost_witness_ids': [row['witness_id'] for row in rows if not row['preserved']],
            'basis': 'finite_original_ink_hough_stroke_and_immutable_boundary',
            'attributes_fixed': False, 'ground_truth_used': False}
