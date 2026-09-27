from collections import Counter

import numpy as np

from contour_agent.vectorize import assess_fit_quality, fit_polyline


def rounded_rectangle():
    points = []
    for center, angles in [((90, 10), (-90, 0)), ((90, 50), (0, 90)),
                           ((10, 50), (90, 180)), ((10, 10), (180, 270))]:
        theta = np.radians(np.linspace(*angles, 41))
        points.extend((np.asarray(center) + 10 * np.c_[np.cos(theta), np.sin(theta)]).tolist())
    # A small raster wobble along the left side is not an extra CAD feature.
    points.insert(3 * 41, [.15, 30.])
    return np.asarray(points)


def test_closed_profile_does_not_split_a_side_at_array_seam():
    source = rounded_rectangle()
    for offset in (0, 27, 100):
        points = np.roll(source, offset, axis=0)
        entities = fit_polyline(points, tolerance_px=.3)
        assert Counter(e["type"] for e in entities) == {"LINE": 4, "ARC": 4}
        quality = assess_fit_quality(points, entities, max_step_px=.1)
        assert quality["sampled_topology_valid"]
        assert quality["max_endpoint_gap_px"] < 1e-9
        assert quality["source_boundary_deviation_px"]["conservative_upper_bound_px"] < .4


def test_seam_join_does_not_remove_a_source_corner():
    source = [[0., 0.], [100., 0.], [100., 60.], [0., 60.]]
    entities = fit_polyline(source, tolerance_px=.3)
    assert Counter(e["type"] for e in entities) == {"LINE": 4}
    assert entities[0]["fitting_optimization"]["cyclic_seam_merges"] == 0
    assert assess_fit_quality(source, entities)["area_iou"] == 1.
