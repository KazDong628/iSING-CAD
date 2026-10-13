"""Native-pixel synthetic two-arrow dimensions, without CAD or GT input."""
import copy
import math

import cv2
import numpy as np
import pytest

from contour_agent.oblique_dimension_evidence import oblique_dimension_witnesses, refine_oblique_arrow_contacts


def _drawing(tmp_path, *, angle=37., reflection=1., defect=None, inward=False):
    image = np.full((440, 440), 255, np.uint8)
    low, high, cross = 130., 280., 220.
    cv2.line(image, (90, 220), (320, 220), 0, 1, cv2.LINE_AA)
    signs = (1, -1) if inward else (-1, 1)
    for index, (tip, sign) in enumerate(zip((low, high), signs)):
        if defect == "single_arrow" and index == 1:
            continue
        if defect == "plain_shaft":
            continue
        y = cross+16 if defect == "different_shafts" and index == 1 else cross
        triangle = np.array([[tip, y], [tip-sign*18, y-6], [tip-sign*18, y+6]])
        cv2.fillConvexPoly(image, triangle.astype(np.int32), 0, cv2.LINE_AA)
    if defect == "shaft_gap":
        image[215:226, 190:220] = 255
    if defect == "two_pairs":
        cv2.line(image, (90, 140), (320, 140), 0, 1, cv2.LINE_AA)
        for tip, sign in zip((low, high), signs):
            triangle = np.array([[tip, 140], [tip-sign*18, 134], [tip-sign*18, 146]])
            cv2.fillConvexPoly(image, triangle.astype(np.int32), 0, cv2.LINE_AA)
    if defect == "hatch":
        image[:] = 255
        for offset in range(-80, 90, 14):
            cv2.line(image, (70, 220+offset), (350, 220+offset), 0, 1, cv2.LINE_AA)
    cv2.putText(image, "20", (183, 190), cv2.FONT_HERSHEY_SIMPLEX, .8, 0, 1, cv2.LINE_AA)
    box = np.array([[180., 168.], [216., 168.], [216., 194.], [180., 194.]])
    theta = math.radians(angle)
    basis = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]]) @ np.diag([reflection, 1.])
    affine = np.column_stack([basis, np.array([220., 220.])-basis @ [220., 220.]])
    def transform(points):
        return np.asarray(points) @ basis.T+affine[:, 2]
    image = cv2.warpAffine(image, affine, (440, 440), flags=cv2.INTER_LINEAR, borderValue=255)
    # Axis-aligned OCR boxes are valid after an arbitrary rigid rotation too;
    # their conservative extent contains the actual source glyphs.
    transformed = transform(box)
    lo, hi = transformed.min(axis=0), transformed.max(axis=0)
    source_box = [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]
    row = dict(id="r001", text="20", box=source_box,
               parsed=dict(kind="length", nominal=20., reference=False))
    path = tmp_path/"source.png"
    assert cv2.imwrite(str(path), image)
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), [row], transform([[low, cross], [high, cross]])


@pytest.mark.parametrize("angle,reflection", [(23., 1.), (37., 1.), (65., 1.), (37., -1.), (127., 1.)])
@pytest.mark.parametrize("inward", [False, True])
def test_oblique_opposing_filled_arrows_are_observed_at_original_resolution(tmp_path, angle, reflection, inward):
    image, records, endpoints = _drawing(tmp_path, angle=angle, reflection=reflection, inward=inward)
    result = oblique_dimension_witnesses(image, records)
    assert len(result) == 1
    witness = result[0]
    observed = np.asarray(witness["endpoints_source_px"])
    error = min(np.linalg.norm(observed-endpoints, axis=1).max(),
                np.linalg.norm(observed-endpoints[::-1], axis=1).max())
    assert error <= 5.
    assert witness["record_id"] == "r001"
    assert witness["arrow_pair_verified"] is True
    assert np.linalg.norm(witness["direction_px"]) == pytest.approx(1.)
    proof = witness["evidence"]
    assert proof["same_source_shaft"] and proof["arrow_directions_opposed"]
    assert proof["label_association_unique"]
    assert proof["rigid_rotation_only"] and proof["original_pixel_resolution"]
    assert proof["numeric_binding_certified"] is False
    assert proof["contour_contacts_verified"] is False
    assert proof["requires_contact_tip_recheck"] is True
    assert proof["nominal_used_to_rank"] is proof["nominal_used_to_set_span"] is False
    arrows = proof["endpoint_arrows"]
    assert np.dot(arrows[0]["direction_px"], arrows[1]["direction_px"]) == pytest.approx(-1.)


def test_nominal_never_selects_a_shaft_or_sets_the_observed_span(tmp_path):
    image, records, _ = _drawing(tmp_path)
    original = copy.deepcopy(records)
    before = oblique_dimension_witnesses(image, records)
    changed = copy.deepcopy(records)
    changed[0]["parsed"]["nominal"] = 20000.
    changed[0]["text"] = "unrelated nominal metadata"
    assert before == oblique_dimension_witnesses(image, changed)
    assert records == original


@pytest.mark.parametrize("defect", ["single_arrow", "different_shafts", "plain_shaft", "shaft_gap", "hatch"])
def test_incomplete_pair_or_hatch_cannot_become_a_dimension_witness(tmp_path, defect):
    image, records, _ = _drawing(tmp_path, defect=defect)
    assert oblique_dimension_witnesses(image, records) == []


@pytest.mark.parametrize("kind,reference", [("radius", False), ("diameter", False), ("length", True)])
def test_radius_leaders_and_reference_lengths_are_outside_scope(tmp_path, kind, reference):
    image, records, _ = _drawing(tmp_path)
    records[0]["parsed"].update(kind=kind, reference=reference)
    assert oblique_dimension_witnesses(image, records) == []


def test_one_physical_pair_with_competing_source_labels_remains_unknown(tmp_path):
    image, records, _ = _drawing(tmp_path)
    second = copy.deepcopy(records[0])
    second["id"] = "r002"
    records.append(second)
    assert oblique_dimension_witnesses(image, records) == []


def test_single_radius_label_can_make_aligned_arrow_ownership_ambiguous(tmp_path):
    image, records, _ = _drawing(tmp_path)
    radius = copy.deepcopy(records[0])
    radius.update(id="r002", text="R40", parsed=dict(kind="radius", nominal=40., reference=False))
    # A radius label cannot emit a two-arrow length witness, but its competing
    # ownership must still prevent two aligned leaders being claimed by length.
    records.append(radius)
    assert oblique_dimension_witnesses(image, records) == []


def test_one_label_with_two_independently_verified_parallel_pairs_stays_unknown(tmp_path):
    image, records, _ = _drawing(tmp_path, defect="two_pairs")
    assert oblique_dimension_witnesses(image, records) == []


@pytest.mark.parametrize("angle,reflection", [(23., 1.), (37., 1.), (65., 1.), (37., -1.), (127., 1.)])
def test_independent_source_contacts_require_a_fresh_pixel_taper_recheck(tmp_path, angle, reflection):
    image, records, contacts = _drawing(tmp_path, angle=angle, reflection=reflection)
    witness = oblique_dimension_witnesses(image, records)[0]
    original = copy.deepcopy(witness)
    refined = refine_oblique_arrow_contacts(image, witness, contacts)
    assert refined is not None
    actual = np.asarray(refined["endpoints_source_px"])
    assert min(np.linalg.norm(actual-contacts), np.linalg.norm(actual-contacts[::-1])) < 1e-9
    assert refined["evidence"]["contact_pixel_tapers_verified"] is True
    assert refined["evidence"]["source_object_ownership_requires_independent_check"] is True
    assert refined["evidence"]["numeric_binding_certified"] is False
    assert witness == original


@pytest.mark.parametrize("defect", ["far_contact", "wrong_shaft", "broken_taper", "unverified_pair"])
def test_contact_refinement_cannot_move_the_shaft_or_jump_to_an_unrelated_target(tmp_path, defect):
    image, records, contacts = _drawing(tmp_path)
    witness = oblique_dimension_witnesses(image, records)[0]
    if defect == "far_contact":
        contacts += np.asarray(witness["direction_px"])*80.
    elif defect == "wrong_shaft":
        u = witness["direction_px"]
        contacts += np.array([-u[1], u[0]])*8.
    elif defect == "broken_taper":
        image[:] = 255
    else:
        witness["arrow_pair_verified"] = False
    assert refine_oblique_arrow_contacts(image, witness, contacts) is None


def test_busy_record_budget_and_invalid_image_remain_unknown(tmp_path):
    image, records, _ = _drawing(tmp_path)
    many = [{**records[0], "id": f"r{i:03d}"} for i in range(65)]
    assert oblique_dimension_witnesses(image, many) == []
    assert oblique_dimension_witnesses(image.astype(float), records) == []
