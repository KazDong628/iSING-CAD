"""Source-only label-axis and chain association; no reference CAD input."""
import cv2
import numpy as np
import pytest

from contour_agent.dimension_evidence import _linear_witnesses, binding_linear_witnesses, estimate_scale
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


def _short_external_dimension(*, axis="y", missing_arrow=False, crossing_only=False, second_shaft=False):
    image = np.full((2000, 2200), 255, np.uint8)
    for x in ([1000, 1020] if second_shaft else [1000]):
        cv2.line(image, (x, 530), (x, 710), 0, 1)
        for y in (600, 640):
            cv2.line(image, (925, y), (1050, y), 0, 1)
        if not crossing_only:
            cv2.fillConvexPoly(image, np.asarray([[x, 600], [x-5, 575], [x+5, 575]]), 0)
            if not missing_arrow:
                cv2.fillConvexPoly(image, np.asarray([[x, 640], [x-5, 665], [x+5, 665]]), 0)
    # The OCR box straddles both observed stations. Requiring extensions to
    # lie outside this box would miss the actual short measured interval.
    box = np.asarray([[950, 585], [985, 585], [985, 650], [950, 650]])
    if axis == "x":
        image = image.T.copy()
        box = box[:, ::-1]
    records = canonical_records({"records": [{"text": "10", "box": box.tolist()}]})
    return image, records


@pytest.mark.parametrize("axis", ["x", "y"])
def test_binding_recovers_external_arrow_short_span_without_changing_scale(axis):
    image, records = _short_external_dimension(axis=axis)
    before = _linear_witnesses(image, records)
    binding = binding_linear_witnesses(image, records)
    recovered = [w for w in binding if w.get("binding_evidence")]
    assert len(recovered) == 1
    witness = recovered[0]
    assert witness["axis"] == axis
    assert witness["span_px"] == pytest.approx(40)
    assert witness["binding_evidence"]["verified"] is True
    assert witness["binding_evidence"]["ambiguous"] is False
    assert witness["binding_evidence"]["scale_estimation_eligible"] is False
    assert witness["binding_evidence"]["nominal_used_to_rank"] is False
    assert before == _linear_witnesses(image, records)
    assert all(w["span_px"] != pytest.approx(40) for w in before)


def test_short_dimension_stations_do_not_depend_on_nominal_or_scale():
    image, records = _short_external_dimension()
    first = binding_linear_witnesses(image, records)
    records[0]["parsed"]["nominal"] = 999.
    second = binding_linear_witnesses(image, records)
    assert len(first) == len(second) == 1
    assert first[0]["line"] == second[0]["line"]
    assert first[0]["binding_evidence"] == second[0]["binding_evidence"]
    assert second[0]["nominal"] == 999.


@pytest.mark.parametrize("failure", ["missing_arrow", "crossing_only"])
def test_short_dimension_requires_both_arrowheads_not_plain_intersections(failure):
    image, records = _short_external_dimension(**{failure: True})
    binding = binding_linear_witnesses(image, records)
    assert not any(w.get("binding_evidence", {}).get("verified") for w in binding)


def test_short_dimension_competing_arrow_pairs_stay_ambiguous():
    image, records = _short_external_dimension(second_shaft=True)
    binding = binding_linear_witnesses(image, records)
    recovered = [w for w in binding if w.get("binding_evidence")]
    assert len(recovered) == 2
    assert all(w["binding_evidence"]["ambiguous"] for w in recovered)
    assert all(w["axis_association"]["ambiguous"] for w in recovered)


def test_adjacent_text_stroke_does_not_replace_actual_extension_station():
    image, records = _short_external_dimension()
    # A glyph-like stroke stops before the dimension shaft. It cannot be a
    # measured endpoint even though it is inside the old broad opening band.
    cv2.line(image, (952, 615), (994, 615), 0, 1)
    binding = binding_linear_witnesses(image, records)
    assert len(binding) == 1
    assert binding[0]["span_px"] == pytest.approx(40)
    assert binding[0]["line"]["lo"] == pytest.approx(600)
    assert binding[0]["line"]["hi"] == pytest.approx(640)
