import pytest

from src.geometry import box_area, box_center, box_iou, is_left_of, is_right_of, xywh_to_xyxy


def test_xywh_to_xyxy_converts_coco_box():
    assert xywh_to_xyxy([10, 20, 30, 40]) == (10.0, 20.0, 40.0, 60.0)


def test_xywh_to_xyxy_allows_zero_area():
    assert xywh_to_xyxy([10, 20, 0, 0]) == (10.0, 20.0, 10.0, 20.0)


def test_xywh_to_xyxy_rejects_negative_extent():
    with pytest.raises(ValueError):
        xywh_to_xyxy([10, 20, -1, 5])


def test_box_area():
    assert box_area([0, 0, 10, 20]) == 200.0


def test_box_area_zero_area():
    assert box_area([0, 0, 0, 20]) == 0.0
    assert box_area([0, 0, 20, 0]) == 0.0


def test_box_iou_no_overlap():
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_box_iou_partial_overlap():
    assert box_iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)


def test_box_iou_containment():
    assert box_iou([0, 0, 10, 10], [2, 2, 8, 8]) == pytest.approx(36 / 100)


def test_box_iou_identical_boxes():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_box_iou_zero_area_box():
    assert box_iou([0, 0, 0, 10], [0, 0, 10, 10]) == 0.0
    assert box_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0


def test_box_iou_rejects_invalid_xyxy_order():
    with pytest.raises(ValueError):
        box_iou([10, 0, 0, 10], [0, 0, 10, 10])


def test_box_center():
    assert box_center([0, 2, 10, 20]) == (5.0, 11.0)


def test_left_right_logic_uses_box_centers():
    left = [0, 0, 10, 10]
    right = [20, 0, 30, 10]
    assert is_left_of(left, right)
    assert is_right_of(right, left)
    assert not is_left_of(right, left)
    assert not is_right_of(left, right)
