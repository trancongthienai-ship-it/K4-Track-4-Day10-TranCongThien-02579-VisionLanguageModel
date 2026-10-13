"""Unit test kiểm thử toàn diện hàm reward IoU trong src/reward.py."""
import pytest
from src.reward import compute_iou_reward


def test_reward_malformed_json():
    # JSON hỏng -> trả 0.0
    gt = [{"label": "pothole", "bbox": (100, 100, 200, 200)}]
    assert compute_iou_reward("not a json", gt) == 0.0
    assert compute_iou_reward("[{'label': 'pothole'}]", gt) == 0.0
    assert compute_iou_reward("[{\"label\": \"pothole\", \"bbox_2d\": [100, 100]}]", gt) == 0.0


def test_reward_both_empty():
    # Cả GT và Pred đều rỗng -> 1.0
    assert compute_iou_reward("[]", []) == 1.0


def test_reward_one_empty():
    gt = [{"label": "pothole", "bbox": (100, 100, 200, 200)}]
    # Model đoán rỗng khi có GT -> 0.2
    assert compute_iou_reward("[]", gt) == 0.2
    # Model đoán có box khi GT rỗng -> 0.2
    pred_str = '[{"label": "pothole", "bbox_2d": [100, 100, 200, 200]}]'
    assert compute_iou_reward(pred_str, []) == 0.2


def test_reward_perfect_match():
    # Khớp hoàn hảo 1 box -> 0.2 + 0.8 * 1.0 = 1.0
    gt = [{"label": "pothole", "bbox": (100, 100, 200, 200)}]
    pred_str = '[{"label": "pothole", "bbox_2d": [100, 100, 200, 200]}]'
    assert compute_iou_reward(pred_str, gt) == pytest.approx(1.0, 1e-4)


def test_reward_class_mismatch():
    # Đúng toạ độ nhưng sai nhãn -> không ghép đôi -> reward = 0.2
    gt = [{"label": "pothole", "bbox": (100, 100, 200, 200)}]
    pred_str = '[{"label": "crack", "bbox_2d": [100, 100, 200, 200]}]'
    assert compute_iou_reward(pred_str, gt) == pytest.approx(0.2, 1e-4)


def test_reward_extra_box():
    # Có 1 box đúng, 1 box bịa -> n_pred = 2, n_gt = 1
    # total_iou = 1.0 -> 0.2 + 0.8 * (1.0 / 2) = 0.6
    gt = [{"label": "pothole", "bbox": (100, 100, 200, 200)}]
    pred_str = (
        '[{"label": "pothole", "bbox_2d": [100, 100, 200, 200]},'
        '{"label": "manhole", "bbox_2d": [500, 500, 600, 600]}]'
    )
    assert compute_iou_reward(pred_str, gt) == pytest.approx(0.6, 1e-4)


def test_reward_low_iou_below_threshold():
    # IoU < 0.5 không được chấp nhận
    # a: [100, 100, 200, 200] (area 10000)
    # b: [150, 150, 250, 250] (inter 50x50 = 2500, union 17500 -> IoU = 0.14 < 0.5)
    gt = [{"label": "pothole", "bbox": (100, 100, 200, 200)}]
    pred_str = '[{"label": "pothole", "bbox_2d": [150, 150, 250, 250]}]'
    assert compute_iou_reward(pred_str, gt) == pytest.approx(0.2, 1e-4)
