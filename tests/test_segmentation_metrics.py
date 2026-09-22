"""Synthetic metric and augmentation checks independent of project datasets."""
import json

import numpy as np
import pytest

from contour_agent.segmentation_metrics import add_annotation_noise, segmentation_metrics


def _mask():
    mask = np.zeros((100,140), bool)
    mask[25:70,35:100] = True
    return mask


def test_identical_masks_have_perfect_region_and_boundary_agreement():
    result = segmentation_metrics(_mask(),_mask())
    for name in ("iou","dice","pixel_precision","pixel_recall","boundary_precision","boundary_recall","boundary_f1"):
        assert result[name] == 1
    assert result["average_symmetric_boundary_distance_px"] == 0
    assert result["hausdorff95_px"] == 0
    assert result["empty_case"] == "none"
    assert "not independent truth" in result["scope"]


def test_region_metrics_match_hand_counted_intersection():
    prediction = np.array([[1,1,0],[1,0,0]],np.uint8)
    target = np.array([[1,0,0],[1,1,0]],np.uint8)
    result = segmentation_metrics(prediction,target,0)
    assert result["intersection_pixels"] == 2
    assert result["union_pixels"] == 4
    assert result["iou"] == .5
    assert result["dice"] == pytest.approx(2/3)
    assert result["pixel_precision"] == result["pixel_recall"] == pytest.approx(2/3)


def test_boundary_tolerance_does_not_change_region_overlap():
    target = _mask()
    prediction = np.roll(target,1,axis=1)
    exact = segmentation_metrics(prediction,target,tolerance_px=0)
    tolerant = segmentation_metrics(prediction,target,tolerance_px=1)
    assert exact["iou"] == tolerant["iou"] < 1
    assert exact["boundary_f1"] < 1
    assert tolerant["boundary_f1"] == 1
    assert tolerant["hausdorff_distance_px"] == 1
    assert tolerant["hausdorff95_px"] == 1


@pytest.mark.parametrize("prediction_empty,target_empty", [(True,True),(True,False),(False,True)])
def test_empty_cases_are_explicit_and_json_safe(prediction_empty,target_empty):
    prediction = np.zeros((100,140),bool) if prediction_empty else _mask()
    target = np.zeros((100,140),bool) if target_empty else _mask()
    result = segmentation_metrics(prediction,target)
    both = prediction_empty and target_empty
    expected = 1 if both else 0
    assert result["iou"] == result["dice"] == result["boundary_f1"] == expected
    assert result["average_symmetric_boundary_distance_px"] == (0 if both else None)
    assert result["hausdorff95_px"] == (0 if both else None)
    assert result["empty_case"] == ("both_empty" if both else "empty_prediction" if prediction_empty else "empty_target")
    json.dumps(result,allow_nan=False)


def test_image_border_is_a_boundary_and_uint8_255_is_supported():
    mask = np.ones((10,20),np.uint8)*255
    result = segmentation_metrics(mask,mask)
    assert result["predicted_boundary_pixels"] == 2*10+2*20-4
    assert result["boundary_f1"] == 1


def test_hole_boundary_affects_boundary_recall():
    prediction = _mask()
    target = prediction.copy()
    target[40:55,55:75] = False
    result = segmentation_metrics(prediction,target,tolerance_px=0)
    assert result["boundary_precision"] == 1
    assert result["boundary_recall"] < 1
    assert result["boundary_f1"] < 1


@pytest.mark.parametrize("bad", [np.ones((10,10))*.5,np.ones((10,10))*np.nan,np.zeros((2,3,4)),np.zeros((0,10))])
def test_probability_and_invalid_mask_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        segmentation_metrics(bad,bad)


def test_shape_mismatch_and_negative_tolerance_are_rejected():
    with pytest.raises(ValueError):
        segmentation_metrics(np.zeros((10,10)),np.zeros((10,11)))
    with pytest.raises(ValueError):
        segmentation_metrics(_mask(),_mask(),-1)


def _image_and_target():
    image = np.full((256,384,3),255,np.uint8)
    image[60:63,65:305] = [10,20,30]
    image[60:190,65:68] = [10,20,30]
    mask = np.zeros((256,384),np.uint8)
    mask[65:185,70:300] = 1
    return image,mask


def test_annotation_noise_is_reproducible_and_does_not_mutate_inputs():
    image,mask = _image_and_target()
    original_image,original_mask = image.copy(),mask.copy()
    first = add_annotation_noise(image,17,mask)
    second = add_annotation_noise(image,17,mask)
    different = add_annotation_noise(image,18,mask)
    assert first.shape == image.shape and first.dtype == np.uint8
    assert np.array_equal(first,second)
    assert not np.array_equal(first,different)
    assert not np.array_equal(first,image)
    assert np.array_equal(image,original_image)
    assert np.array_equal(mask,original_mask)
    assert not np.shares_memory(first,image)


@pytest.mark.parametrize("strength", [0,.25,1,2])
def test_annotation_darkening_respects_pixel_budget_and_preserves_existing_ink(strength):
    image,mask = _image_and_target()
    result = add_annotation_noise(image,32,mask,strength)
    changed = np.any(result != image,axis=2)
    assert changed.sum() <= int(mask.size*min(.12,.06*strength))
    assert np.all(result <= image)
    if strength == 0:
        assert np.array_equal(result,image)
        assert not np.shares_memory(result,image)


def test_optional_hatch_changes_only_mask_interior_relative_to_same_annotation_seed():
    image,mask = _image_and_target()
    without_hatch = add_annotation_noise(image,52)
    with_hatch = add_annotation_noise(image,52,mask)
    changed = np.any(with_hatch != without_hatch,axis=2)
    assert changed.any()
    assert not np.any(changed & (mask == 0))
    assert np.all(with_hatch <= without_hatch)


def test_local_rng_does_not_modify_global_numpy_rng_state():
    image,_ = _image_and_target()
    state = np.random.get_state()
    add_annotation_noise(image,77)
    after = np.random.get_state()
    assert state[0] == after[0]
    assert np.array_equal(state[1],after[1])
    assert state[2:] == after[2:]


def test_tiny_rgb_image_is_safe_and_bounded():
    image = np.full((1,1,3),255,np.uint8)
    assert np.array_equal(add_annotation_noise(image,0),image)


@pytest.mark.parametrize("strength", [-1,2.01,float("nan")])
def test_invalid_augmentation_strength_is_rejected(strength):
    image,_ = _image_and_target()
    with pytest.raises(ValueError):
        add_annotation_noise(image,1,strength=strength)


def test_invalid_augmentation_image_mask_and_seed_are_rejected():
    image,mask = _image_and_target()
    with pytest.raises(ValueError):
        add_annotation_noise(image.astype(float),1)
    with pytest.raises(ValueError):
        add_annotation_noise(image,1,mask[:20])
    with pytest.raises(ValueError):
        add_annotation_noise(image,-1)
