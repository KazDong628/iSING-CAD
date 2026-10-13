"""Bounded null-space image refinement on an already feasible CAD problem."""
from __future__ import annotations

import time

import numpy as np


def refine_feasible_point(initial, residuals, equalities, equality_jacobian,
                          objective_jacobian, inequalities, lower, upper,
                          certify_and_retain, *, seconds, guard=lambda: None):
    """Move unknowns along the equality manifold, then reproject and certify.

    This proposes initial values for the existing optimizer. The caller must
    retain a trial only after checking ALL native contracts, bounds, inequalities
    and geometry. No soft objective may decide whether a known equation holds.
    """
    started = time.monotonic()
    point = np.asarray(initial, float).copy()
    accepted = 0
    trials = 0

    class Subbudget(Exception):
        pass

    def check():
        if time.monotonic()-started >= seconds:
            raise Subbudget()
        guard()

    def project(value):
        # All CAD equality Jacobians are already available. A generic finite
        # difference least-squares projection repeatedly differentiates the
        # complete ink-free problem, wasting the refinement budget. Newton
        # corrections use ALL rows (including dependent ones); every remaining
        # inequality is checked before the caller's complete native certificate.
        value = value.copy()
        for _ in range(12):
            check()
            error = equalities(value)
            if np.max(np.abs(error)) < 1e-11:
                break
            correction = np.linalg.lstsq(equality_jacobian(value), -error, rcond=1e-10)[0]
            value = np.clip(value+correction, lower, upper)
        return value

    try:
        for _ in range(6):
            check()
            matrix = equality_jacobian(point)
            _, singular, right = np.linalg.svd(matrix, full_matrices=True)
            tolerance = max(1e-12, (float(singular[0]) if len(singular) else 0.)*1e-10)
            rank = int(np.count_nonzero(singular > tolerance))
            null = right[rank:].T
            if not null.shape[1]:
                break
            current = residuals(point)
            jacobian = objective_jacobian(point)
            check()
            step = null @ np.linalg.lstsq(jacobian @ null, -current, rcond=1e-8)[0]
            magnitude = float(np.linalg.norm(step))
            if magnitude < 1e-10:
                break
            step *= min(1., .005/magnitude)
            improved = False
            for factor in (1., .5, .25):
                check()
                trial = np.clip(point+factor*step, lower, upper)
                proposed = project(trial)
                trials += 1
                check()
                if any(np.any(np.atleast_1d(fun(proposed)) < -1e-10) for fun in inequalities):
                    continue
                value = residuals(proposed)
                if float(value @ value) < float(current @ current) and certify_and_retain(proposed):
                    point = proposed
                    accepted += 1
                    improved = True
                    break
            if not improved:
                break
    except Subbudget:
        pass
    except (ValueError, ArithmeticError, np.linalg.LinAlgError):
        # A bad branch is an ordinary failed proposal, never replacement of a
        # previous certified checkpoint or permission to weaken a constraint.
        pass
    return point, {'method': 'bounded_null_space_objective_steps_with_all_equation_projection',
                   'projection_trials': trials, 'certified_improvements': accepted,
                   'maximum_seconds': seconds, 'elapsed_seconds': time.monotonic()-started,
                   'known_equations_changed': False, 'original_domain_recentered': False,
                   'publication_approved': False, 'reference_geometry_used': False}
