"""Tiện ích chuyển đổi nhãn giữa YOLO và định dạng Grounding (0-1000) cho Qwen3-VL."""
import json
from typing import Any, Dict, List, Tuple

from scripts.labcommon import CLASSES

PROMPT_NORM1000 = (
    "Locate all road damage objects in this image. "
    "Classes to detect: pothole, crack, manhole. "
    "Output a JSON list of detected objects where each object has 'label' and 'bbox_2d' [x1, y1, x2, y2] "
    "with coordinates scaled to [0, 1000] relative to image width and height. "
    "Return only the JSON list."
)

PROMPT_PIXEL = (
    "Locate all road damage objects in this image. "
    "Classes to detect: pothole, crack, manhole. "
    "Output a JSON list of detected objects where each object has 'label' and 'bbox_2d' [x1, y1, x2, y2] "
    "with absolute pixel coordinates. "
    "Return only the JSON list."
)


def yolo_to_box(label: str, yolo: List[float], coord: str = "norm1000", width: int = 640, height: int = 360) -> Dict[str, Any]:
    """Chuyển đổi toạ độ YOLO [xc, yc, w, h] chuẩn hoá [0, 1] sang box [x1, y1, x2, y2].

    - Nếu coord == 'norm1000': thang [0, 1000].
    - Nếu coord == 'pixel': thang pixel tuyệt đối dựa trên width, height.
    """
    xc, yc, w, h = yolo
    if coord == "pixel":
        scale_x = float(width)
        scale_y = float(height)
    else:
        scale_x = 1000.0
        scale_y = 1000.0

    x1 = int(round((xc - w / 2.0) * scale_x))
    y1 = int(round((yc - h / 2.0) * scale_y))
    x2 = int(round((xc + w / 2.0) * scale_x))
    y2 = int(round((yc + h / 2.0) * scale_y))

    max_x = int(round(scale_x))
    max_y = int(round(scale_y))

    x1 = max(0, min(max_x, x1))
    y1 = max(0, min(max_y, y1))
    x2 = max(0, min(max_x, x2))
    y2 = max(0, min(max_y, y2))

    # Đảm bảo x2 > x1 và y2 > y1 theo yêu cầu score.py
    if x2 <= x1:
        if x2 < max_x:
            x2 = x1 + 1
        else:
            x1 = max(0, x2 - 1)
    if y2 <= y1:
        if y2 < max_y:
            y2 = y1 + 1
        else:
            y1 = max(0, y2 - 1)

    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects: List[Dict[str, Any]], coord: str = "norm1000", width: int = 640, height: int = 360) -> str:
    """Chuyển danh sách đối tượng sang JSON string chuẩn hoá.

    Sắp xếp các box theo thứ tự (y1, x1) từ trên xuống dưới, từ trái sang phải
    để chuỗi mục tiêu ổn định cho mô hình học.
    """
    boxes = [yolo_to_box(obj["label"], obj["yolo"], coord=coord, width=width, height=height) for obj in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def box_to_pixel(bbox_1000: List[float], width: int, height: int) -> List[float]:
    """Quy đổi toạ độ từ thang 0-1000 về pixel thực tế của ảnh."""
    x1, y1, x2, y2 = bbox_1000
    return [
        (x1 / 1000.0) * width,
        (y1 / 1000.0) * height,
        (x2 / 1000.0) * width,
        (y2 / 1000.0) * height,
    ]
