"""Analytic source-generated fixtures, never evaluation GT coordinates."""
import copy
import math

import numpy as np
import pytest

from contour_agent.topology_editing import _annotated_line_fillet
from contour_agent.vectorize import _sample_entities


def _parts(left_arc=False):
    prefix = ({"type": "ARC", "start": [-25., 25.], "end": [0., 0.],
               "center": [0., 25.], "radius": 25., "clockwise": False} if left_arc else
              {"type": "LINE", "start": [-40., 0.], "end": [0., 0.]})
    return [prefix,
            {"type": "ARC", "start": [0., 0.], "end": [5., 5.],
             "center": [0., 5.], "radius": 5., "clockwise": False},
            {"type": "ARC", "start": [5., 5.], "end": [25., 25.],
             "center": [25., 5.], "radius": 20., "clockwise": True}]


def _binding():
    return {"record_id": "r0", "nominal": 5., "arrowhead_verified": True,
            "target_source_px": [5/math.sqrt(2), 5-5/math.sqrt(2)]}


def _tangent(entity, endpoint):
    if entity["type"] == "LINE":
        delta = np.array(entity["end"])-entity["start"]
    else:
        radial = np.array(entity[endpoint])-entity["center"]
        delta = np.array([-radial[1], radial[0]])*(-1 if entity["clockwise"] else 1)
    return delta/np.linalg.norm(delta)


@pytest.mark.parametrize("left_arc,reverse", [(False, False), (False, True), (True, False), (True, True)])
def test_exact_fillet_with_circular_supports_preserves_outer_endpoints_and_tangency(left_arc, reverse):
    originals = _parts(left_arc)
    if reverse:
        originals.reverse()
        for entity in originals:
            entity["start"], entity["end"] = entity["end"], entity["start"]
            if entity["type"] == "ARC":
                entity["clockwise"] = not entity["clockwise"]
    source = _sample_entities(originals, max_step_px=.1)[0]
    before = source.copy()
    parts = _annotated_line_fillet(source, 5., .12, _binding())
    expected = [p["type"] for p in originals]
    assert [p["type"] for p in parts] == expected
    assert parts[0]["start"] == source[0].tolist()
    assert parts[-1]["end"] == source[-1].tolist()
    assert np.array_equal(source, before)
    assert parts[1]["radius"] == 5.
    assert parts[1]["radius_binding_status"] == "applied"
    assert parts[1]["fillet_construction"] == "source_support_offset_loci"
    for first, second in zip(parts, parts[1:]):
        assert first["end"] == second["start"]
        assert _tangent(first, "end") @ _tangent(second, "start") > 1-1e-10
    for entity in parts:
        if entity["type"] == "ARC":
            for endpoint in ("start", "end"):
                assert math.dist(entity["center"], entity[endpoint]) == pytest.approx(entity["radius"], abs=1e-9)
    # Validate actual output against source samples without relying on its
    # recorded fit-error field or merely on construction success.
    from scipy.spatial import cKDTree
    sampled = _sample_entities(parts, max_step_px=.1)[0]
    assert cKDTree(source).query(sampled)[0].max() < .13
    assert cKDTree(sampled).query(source)[0].max() < .13


@pytest.mark.parametrize("change", ["no_arrow", "wrong_target", "wrong_radius"])
def test_curved_fillet_does_not_bypass_missing_evidence_or_wrong_radius(change):
    source = _sample_entities(_parts(), max_step_px=.1)[0]
    binding = copy.deepcopy(_binding())
    radius = 5.
    if change == "no_arrow":
        binding["arrowhead_verified"] = False
    elif change == "wrong_target":
        binding["target_source_px"] = [-30., 0.]
    else:
        radius = 1.
    with pytest.raises(ValueError):
        _annotated_line_fillet(source, radius, .12, binding)


def test_two_smooth_cocircular_supports_do_not_invent_a_small_fillet():
    theta = np.linspace(-1., 1., 401)
    source = 30*np.column_stack([np.cos(theta), np.sin(theta)])
    binding = {**_binding(), "target_source_px": [30., 0.]}
    with pytest.raises(ValueError, match="source_does_not_support_exact_annotated_fillet"):
        _annotated_line_fillet(source, 5., .12, binding)
