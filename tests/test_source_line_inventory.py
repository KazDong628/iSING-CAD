import cv2
import numpy as np

from contour_agent.source_line_inventory import line_preservation, source_line_witnesses
from contour_agent.topology import _StrokeEvidence


def line(a, b):
    return {'type': 'LINE', 'start': a, 'end': b}


def test_finite_ink_line_survives_as_object_without_locking_its_attributes():
    gray = np.full((140, 150), 255, np.uint8)
    a, b = [30., 20.], [60., 110.]
    cv2.line(gray, (30, 20), (60, 110), 0, 2, cv2.LINE_AA)
    boundary = np.asarray([a, b])  # Sparse immutable polyline, not dense raster vertices.
    stroke = _StrokeEvidence(gray, [], 2.)
    witnesses = source_line_witnesses(gray, [line(a, b)], boundary, 2., stroke)
    assert len(witnesses) == 1
    assert line_preservation(witnesses, [line(a, b)], 2.)['passed']
    curved = [{'type': 'ARC', 'start': a, 'end': b}]
    assert not line_preservation(witnesses, curved, 2.)['passed']
    # Splitting and endpoint relocation are allowed; fitted attributes are not
    # equalities. The source annotations and solver determine exact values.
    split = [line([31., 21.], [46., 66.]), line([46., 66.], [61., 111.])]
    result = line_preservation(witnesses, split, 2.)
    assert result['passed'] and not result['attributes_fixed']
    assert not result['ground_truth_used']


def test_dark_hatch_away_from_mask_and_unsupported_cad_chord_are_not_witnesses():
    gray = np.full((160, 150), 255, np.uint8)
    cv2.line(gray, (20, 20), (50, 110), 0, 2, cv2.LINE_AA)
    boundary = np.linspace([50, 20], [80, 110], 150)
    stroke = _StrokeEvidence(gray, [], 2.)
    assert not source_line_witnesses(gray, [line([20, 20], [50, 110])], boundary, 2., stroke)
    assert not source_line_witnesses(gray, [line([50, 20], [80, 110])], boundary, 2., stroke)


def test_duplicate_fragments_cannot_fake_whole_line_coverage():
    witnesses = [{'id': 's0', 'source_segment_px': [[0, 0], [100, 0]], 'length_px': 100.}]
    result = line_preservation(witnesses, [line([0, 0], [40, 0])]*4, 1.)
    assert result['measurements'][0]['line_coverage_fraction'] == .4
    assert not result['passed']
