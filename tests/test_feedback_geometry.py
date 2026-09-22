import cv2
import numpy as np
import pytest

from contour_agent.feedback_geometry import apply_line_operations, locate_feedback_crop


def _write(path, image):
    assert cv2.imencode(path.suffix, image)[0]
    cv2.imencode(path.suffix, image)[1].tofile(str(path))


def test_pasted_crop_is_registered_back_to_source_coordinates(tmp_path):
    source = np.full((500, 800, 3), 255, np.uint8)
    cv2.rectangle(source, (180, 110), (650, 420), (0, 0, 0), 4)
    cv2.line(source, (220, 170), (600, 330), (0, 0, 0), 3)
    overlay = cv2.resize(source, (400, 250), interpolation=cv2.INTER_AREA)
    cv2.rectangle(overlay, (90, 55), (325, 210), (0, 0, 220), 2)
    crop = overlay[40:225, 70:345]
    source_path, overlay_path, crop_path = tmp_path / "source.png", tmp_path / "overlay.png", tmp_path / "crop.png"
    _write(source_path, source); _write(overlay_path, overlay); _write(crop_path, crop)

    result = locate_feedback_crop(crop_path, overlay_path, source_path)

    assert result["score"] > .8
    assert np.allclose(result["source_bbox_px"], [140, 80, 690, 450], atol=6)
    assert result["ground_truth_used"] is False


def test_top_and_right_arc_chains_are_replaced_by_single_lines():
    # Ordered clockwise cycle: top and right each contain an unwanted arc.
    entities = [
        {"type": "LINE", "start": [10, 10], "end": [45, 10]},
        {"type": "ARC", "start": [45, 10], "end": [90, 12], "center": [68, 40], "radius": 38, "clockwise": False},
        {"type": "ARC", "start": [90, 12], "end": [92, 50], "center": [70, 30], "radius": 30, "clockwise": False},
        {"type": "LINE", "start": [92, 50], "end": [90, 90]},
        {"type": "LINE", "start": [90, 90], "end": [10, 90]},
        {"type": "LINE", "start": [10, 90], "end": [10, 10]},
    ]
    operations = [
        {"action": "replace_boundary_chain_with_line", "side": "top", "basis": "顶部是一条线段"},
        {"action": "replace_boundary_chain_with_line", "side": "right", "basis": "右侧是一条线段"},
        {"action": "exclude_hatching_from_boundary", "side": "crop", "basis": "剖面线不是主轮廓"},
    ]

    revised, audit = apply_line_operations(entities, [0, 0, 100, 100], operations)

    assert len(revised) < len(entities)
    assert sum(row["type"] == "ARC" for row in revised) == 0
    assert [row["status"] for row in audit] == ["applied", "applied", "classification_rule_applied"]
    assert all(row["type"] == "LINE" for row in revised)
    assert audit[0]["shared_corner_snapped"] and audit[1]["shared_corner_snapped"]
    assert all(np.allclose(row["end"], revised[(index + 1) % len(revised)]["start"])
               for index, row in enumerate(revised))


def test_radius_annotated_arc_cannot_be_flattened_by_screenshot_line_edit():
    entities = [
        {"type": "LINE", "start": [10, 10], "end": [45, 10], "_source_entity_ids": ["g000"]},
        {"type": "ARC", "start": [45, 10], "end": [90, 12], "center": [68, 40],
         "radius": 38, "clockwise": False, "_source_entity_ids": ["g023"]},
        {"type": "LINE", "start": [90, 12], "end": [90, 90], "_source_entity_ids": ["g024"]},
        {"type": "LINE", "start": [90, 90], "end": [10, 90], "_source_entity_ids": ["g025"]},
        {"type": "LINE", "start": [10, 90], "end": [10, 10], "_source_entity_ids": ["g026"]},
    ]
    operation = [{"action": "replace_boundary_chain_with_line", "side": "top",
                  "basis": "顶部是一条线段"}]

    with pytest.raises(ValueError, match="半径标注保护的圆弧冲突.*g023"):
        apply_line_operations(
            entities, [0, 0, 100, 100], operation,
            protected_radius_entities={"g023"},
        )


def test_two_collinear_line_fragments_are_merged_by_feedback_instruction():
    entities = [
        {"type": "LINE", "start": [10, 10], "end": [10, 45], "_source_entity_ids": ["g017"]},
        {"type": "LINE", "start": [10, 45], "end": [10, 90], "_source_entity_ids": ["g018"]},
        {"type": "LINE", "start": [10, 90], "end": [90, 90], "_source_entity_ids": ["g019"]},
        {"type": "LINE", "start": [90, 90], "end": [90, 10], "_source_entity_ids": ["g020"]},
        {"type": "LINE", "start": [90, 10], "end": [10, 10], "_source_entity_ids": ["g021"]},
    ]

    revised, audit = apply_line_operations(
        entities, [0, 0, 100, 100],
        [{"action": "replace_boundary_chain_with_line", "side": "left", "basis": "17和18是一条直线"}],
    )

    assert len(revised) == 4
    assert audit[0]["status"] == "applied"
    assert audit[0]["removed_entity_count"] == 2
    assert revised[0]["_source_entity_ids"] == ["g017", "g018"]


def test_single_existing_line_is_an_accepted_noop():
    entities = [
        {"type": "LINE", "start": [10, 10], "end": [10, 90], "_source_entity_ids": ["g007"]},
        {"type": "LINE", "start": [10, 90], "end": [90, 90], "_source_entity_ids": ["g008"]},
        {"type": "LINE", "start": [90, 90], "end": [90, 10], "_source_entity_ids": ["g009"]},
        {"type": "LINE", "start": [90, 10], "end": [10, 10], "_source_entity_ids": ["g010"]},
    ]

    revised, audit = apply_line_operations(
        entities, [0, 0, 100, 100],
        [{"action": "replace_boundary_chain_with_line", "side": "left", "basis": "左边界是一条直线"}],
    )

    assert revised == entities
    assert audit[0]["status"] == "already_satisfied"
    assert audit[0]["source_entity_ids"] == ["g007"]
