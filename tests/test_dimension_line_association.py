"""Source-only label-axis and chain association; no reference CAD input."""
import cv2
import numpy as np
import pytest

from contour_agent.dimension_evidence import _linear_witnesses, estimate_scale
from contour_agent.ocr import canonical_records


def test_near_vertical_dimension_rejects_distant_horizontal_material_edges(tmp_path):
    image = np.full((1000, 1200), 255, np.uint8)
    records = []
    for index, nominal in enumerate([10, 20, 30]):
        x, lo = 200 + index * 300, 60 + index * 220
        hi = lo + nominal * 8
        middle = (lo + hi) // 2
        box = [[x-46, middle-15], [x-6, middle-15], [x-6, middle+15], [x-46, middle+15]]
        cv2.line(image, (x, lo), (x, hi), 0, 1)
        # A remote orthogonal edge would create a second, false 35 px/mm mode.
        cv2.line(image, (20, middle+45), (20+nominal*35, middle+45), 0, 1)
        records.append({"text": str(nominal), "box": box})
    document = {"records": records}
    witnesses = _linear_witnesses(image, canonical_records(document))
    assert len(witnesses) == 3
    assert {w["axis"] for w in witnesses} == {"y"}
    assert all(w["pixels_per_mm"] == pytest.approx(8) for w in witnesses)
    assert all(not w["axis_association"]["ambiguous"] for w in witnesses)
    assert all(any(not a["retained"] for a in w["axis_association"]["alternatives"]) for w in witnesses)
    path = tmp_path / "source.png"
    path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
    result = estimate_scale(path, document)
    assert result["status"] == "resolved"
    assert result["pixels_per_mm"] == pytest.approx(8)
    assert result["ratio_spread"] <= .03


def test_equal_label_distances_preserve_both_axes_for_review():
    image = np.full((500, 500), 255, np.uint8)
    cv2.line(image, (50, 205), (450, 205), 0, 1)
    cv2.line(image, (205, 50), (205, 450), 0, 1)
    records = canonical_records({"records": [{"text": "100", "box": [[160,160], [200,160], [200,200], [160,200]]}]})
    witnesses = _linear_witnesses(image, records)
    assert {w["axis"] for w in witnesses} == {"x", "y"}
    assert all(w["axis_association"]["ambiguous"] for w in witnesses)


def test_chained_stroke_splits_only_for_node_binding_and_retains_source_evidence():
    image = np.full((500, 950), 255, np.uint8)
    cv2.line(image, (50, 300), (850, 300), 0, 1)
    for station in [50, 350, 550, 850]:
        cv2.line(image, (station, 230), (station, 380), 0, 1)
    records = canonical_records({"records": [
        {"text": text, "box": [[x,270], [x+60,270], [x+60,295], [x,295]]}
        for text, x in [("30", 170), ("20", 420), ("30", 670)]
    ]})
    scale_witnesses = _linear_witnesses(image, records)
    assert len(scale_witnesses) == 3
    assert [w["span_px"] for w in scale_witnesses] == [800, 800, 800]
    binding_witnesses = _linear_witnesses(image, records, split_chains=True)
    assert [w["span_px"] for w in binding_witnesses] == [300, 200, 300]
    assert all(w["line"]["unsegmented_stroke"]["span"] == 801 for w in binding_witnesses)
    assert all(w["line"]["segmentation_method"] == "nearest_observed_extensions_outside_label" for w in binding_witnesses)
