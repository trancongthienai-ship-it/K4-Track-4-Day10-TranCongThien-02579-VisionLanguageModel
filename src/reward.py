"""Hàm reward tính điểm IoU cho GRPO (E5)."""
from typing import Any, Dict, List, Optional
from scripts.labcommon import CLASSES
from scripts.score import iou_xyxy, parse_output

IOU_THR = 0.5


def compute_iou_reward(completion: str, gt_boxes: List[Dict[str, Any]]) -> float:
    """Tính reward IoU cho một chuỗi sinh ra bởi model đối chiếu với ground-truth boxes.

    - Nếu completion không phải JSON hợp lệ: trả 0.0
    - Nếu không có gt và không có pred: trả 1.0
    - Ghép đôi tham lam theo IoU giảm dần, cùng nhãn lớp, chỉ nhận IoU >= 0.5
    - Công thức thưởng: 0.2 + 0.8 * (tổng IoU các cặp khớp) / max(n_gt, n_pred, 1)
    """
    pred = parse_output(completion)
    if pred is None:
        return 0.0

    n_gt = len(gt_boxes)
    n_pred = len(pred)

    if n_gt == 0 and n_pred == 0:
        return 1.0

    if n_gt == 0 or n_pred == 0:
        # Có box bịa hoặc bỏ sót hoàn toàn
        return 0.2

    # Ghép đôi tham lam theo nhãn lớp
    matched_ious = []
    for label in set(CLASSES) | {p["label"] for p in pred}:
        g = [x["bbox"] for x in gt_boxes if x["label"] == label]
        p = [x["bbox"] for x in pred if x["label"] == label]
        pairs = sorted(
            ((iou_xyxy(a, b), i, j) for i, a in enumerate(g) for j, b in enumerate(p)),
            reverse=True,
        )
        used_g, used_p = set(), set()
        for iou_val, i, j in pairs:
            if iou_val < IOU_THR:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            matched_ious.append(iou_val)

    total_matched_iou = sum(matched_ious)
    reward_score = 0.2 + 0.8 * (total_matched_iou / max(n_gt, n_pred, 1))
    return float(max(0.0, min(1.0, reward_score)))
