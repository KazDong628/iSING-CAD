"""Independent circular stroke derivatives; observed radii are never dimensions."""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import least_squares


def circular_ink_derivative(points, stations, widths):
    """Select a circle from one side's ink alone, before comparing joint tangents.

    Full and near windows must independently support curvature, radial fit and
    the derivative. No CAD radius, nominal value or other side is available.
    Nearly straight strokes and noncircular transitions remain unresolved.
    """
    failure = {"verified": False, "method": "independent_two_scale_source_circle_v1",
               "reason": "source_circle_not_identifiable", "nominal_dimensions_used": False,
               "observed_radius_is_dimensional_evidence": False}
    points, stations = np.asarray(points, float), np.asarray(stations, float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 24 or
            stations.shape != (len(points),) or not np.isfinite(points).all() or
            not np.isfinite(stations).all() or np.ptp(stations) <= 0):
        return failure
    near = stations <= stations.max() * .75
    if near.sum() < 16:
        return failure
    limit = max(1.25, float(np.median(widths)) / 2)

    def fit(selected):
        xy = points[selected]; t = stations[selected]
        origin = xy.mean(axis=0); scale = max(float(np.ptp(xy, axis=0).max()), 1.)
        z = (xy-origin)/scale
        matrix = np.c_[2*z, np.ones(len(z))]
        if np.linalg.cond(matrix) > 1e5:
            return None
        estimate = np.linalg.lstsq(matrix, np.sum(z*z, axis=1), rcond=None)[0]
        r2 = estimate[2]+estimate[:2]@estimate[:2]
        if r2 <= 0:
            return None
        solved = least_squares(lambda v: np.linalg.norm(z-v[:2], axis=1)-v[2],
            np.r_[estimate[:2], math.sqrt(r2)], max_nfev=30,
            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if not solved.success or solved.x[2] <= 0:
            return None
        center = origin + scale*solved.x[:2]; radius = float(scale*solved.x[2])
        radial = np.linalg.norm(xy-center, axis=1)-radius
        residual = float(np.percentile(np.abs(radial), 90))
        angles = np.unwrap(np.arctan2(*(xy-center).T[::-1]))
        phase = np.polynomial.polynomial.polyfit(t, angles, 1)
        turn = abs(float(angles[-1]-angles[0]))
        sagitta = radius*(1-math.cos(turn/2))
        phase_error = float(np.percentile(np.abs(angles-(phase[0]+phase[1]*t))*radius, 90))
        if (not math.radians(8.) <= turn <= math.radians(125.) or
                sagitta < max(1.5, 3*residual) or residual > limit or phase_error > limit):
            return None
        tangent = math.copysign(1., phase[1])*np.array([-math.sin(phase[0]), math.cos(phase[0])])
        return dict(tangent=tangent, radius=radius, residual=residual, phase_error=phase_error,
                    turn_degrees=math.degrees(turn), sagitta=sagitta)

    full, short = fit(np.ones(len(points), bool)), fit(near)
    if full is None or short is None:
        return failure
    stability = math.degrees(math.acos(float(np.clip(full['tangent']@short['tangent'], -1., 1.))))
    radius_disagreement = abs(full['radius']-short['radius'])/max(full['radius'], short['radius'])
    result = {**failure, "scale_disagreement_degrees": stability,
              "radius_scale_disagreement_fraction": radius_disagreement,
              "observed_radius_px": full['radius'], "source_turn_degrees": full['turn_degrees'],
              "fit_residual_p90_px": full['residual'], "short_fit_residual_p90_px": short['residual'],
              "phase_residual_p90_px": full['phase_error'],
              "window_selection": "independent_source_radial_fit_before_joint_comparison"}
    if stability > 2. or radius_disagreement > .15:
        result['reason'] = 'source_circle_multiscale_disagreement'
        return result
    return {**result, "verified": True, "reason": None,
            "tangent_direction_px": full['tangent'].tolist()}
