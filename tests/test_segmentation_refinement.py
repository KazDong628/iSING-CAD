import numpy as np
import pytest

from contour_agent.segmentation_refinement import connectivity, refine_probabilities


def image_for(mask):
    image = np.full((*mask.shape, 3), 255, np.uint8)
    image[mask] = 0
    return image


def test_supported_broad_gap_is_joined_without_deleting_island():
    coarse = np.zeros((60, 80), np.float32)
    coarse[15:45, 5:40] = .95
    coarse[15:45, 42:60] = .95
    coarse[15:45, 40:42] = .4
    detail = coarse.copy(); detail[15:45, 40:42] = .95
    expected = detail >= .5
    refined, record = refine_probabilities(image_for(expected), coarse, detail, model_size=128)
    assert record['status'] == 'accepted'
    assert record['before']['components_4'] == 2
    assert record['after']['components_4'] == 1
    assert record['existing_components_deleted'] == 0
    assert record['joined_components'][0]['survives_one_pixel_erosion']
    assert np.all((refined >= .5)[coarse >= .5])


def test_one_pixel_bridge_is_rejected():
    coarse = np.zeros((60, 80), np.float32)
    coarse[15:45, 10:40] = .95; coarse[15:45, 42:60] = .95
    coarse[30, 40:42] = .4
    detail = coarse.copy(); detail[30, 40:42] = .95
    refined, record = refine_probabilities(image_for(detail >= .5), coarse, detail, model_size=128)
    assert record['status'] == 'rejected'
    assert any('erosion' in reason for reason in record['reasons'])
    assert np.array_equal(refined, coarse)


def test_uncertain_real_slot_is_not_closed_by_fusion():
    coarse = np.zeros((60, 80), np.float32)
    coarse[10:50, 10:65] = .9; coarse[25:29, 30:34] = .4
    detail = coarse.copy(); detail[25:29, 30:34] = .95
    refined, record = refine_probabilities(image_for(detail >= .5), coarse, detail, model_size=128)
    assert record['status'] == 'rejected'
    assert any('Hole count' in reason for reason in record['reasons'])
    assert np.array_equal(refined, coarse)


def test_new_fragment_cannot_be_accepted_even_with_source_ink():
    coarse = np.zeros((60, 80), np.float32)
    coarse[10:50, 10:40] = .95; coarse[25:28, 41:44] = .4
    detail = coarse.copy(); detail[25:28, 41:44] = .95
    refined, record = refine_probabilities(image_for(detail >= .5), coarse, detail, model_size=128)
    assert record['status'] == 'rejected'
    assert np.array_equal(refined, coarse)


def test_existing_speck_is_preserved_not_hidden_as_connectivity_fix():
    coarse = np.zeros((60, 80), np.float32); coarse[10:40, 10:40] = .9; coarse[42, 42] = .9
    refined, record = refine_probabilities(image_for(coarse >= .5), coarse, coarse, model_size=128)
    assert connectivity(refined >= .5)['components_4'] == 2
    assert refined[42, 42] >= .5


def test_corner_contacts_are_reported_separately():
    mask = np.eye(4, dtype=bool)
    result = connectivity(mask)
    assert result['components_4'] == 4 and result['components_8'] == 1
    assert result['corner_only_connections'] == 3


def test_low_confidence_existing_island_cannot_vanish_as_false_repair():
    coarse = np.zeros((60, 80), np.float32); coarse[10:40, 10:40] = .95
    coarse[42:45, 42:45] = .55
    detail = coarse.copy(); detail[42:45, 42:45] = .1
    image = image_for(detail >= .5)
    refined, record = refine_probabilities(image, coarse, detail, model_size=128)
    assert record['status'] == 'rejected'
    assert any('deletes existing' in reason for reason in record['reasons'])
    assert np.array_equal(refined, coarse)


def test_cancelled_split_and_join_cannot_hide_single_pixel_bridge():
    coarse = np.zeros((100, 100), np.float32)
    coarse[20:80, 10:18] = .9
    coarse[20:80, 20:40] = .9
    coarse[20:80, 14] = .55
    coarse[50, 18:20] = .4
    detail = coarse.copy()
    detail[20:80, 14] = .1
    detail[50, 18:20] = .95
    refined, record = refine_probabilities(image_for(detail >= .5), coarse, detail, model_size=128)
    assert record['before']['components_4'] == record['proposed']['components_4'] == 2
    assert record['changed_fraction'] <= .04
    assert record['status'] == 'rejected'
    assert record['proposed_split_component_ids']
    assert any('splits existing' in reason for reason in record['reasons'])
    assert record['joined_components']
    assert record['joined_components'][0]['survives_one_pixel_erosion'] is False
    assert np.array_equal(refined, coarse)


@pytest.mark.parametrize('value', [np.nan, -1., 1.1])
def test_invalid_probability_rejected(value):
    p = np.zeros((10, 10), np.float32); p[0, 0] = value
    with pytest.raises(ValueError):
        refine_probabilities(np.zeros((10, 10, 3), np.uint8), p, np.zeros_like(p))
