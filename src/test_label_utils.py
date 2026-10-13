"""Unit tests cho src/label_utils.py."""
import json
import pytest

from scripts.score import parse_output
from src.label_utils import (
    PROMPT_NORM1000,
    box_to_pixel,
    objects_to_answer,
    yolo_to_box,
)


def test_yolo_to_box_norm1000():
    # xc=0.5, yc=0.5, w=0.2, h=0.2 -> x1=400, y1=400, x2=600, y2=600
    res = yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.2], coord="norm1000")
    assert res["label"] == "pothole"
    assert res["bbox_2d"] == [400, 400, 600, 600]


def test_yolo_to_box_pixel():
    res = yolo_to_box("crack", [0.5, 0.5, 0.2, 0.2], coord="pixel", width=640, height=360)
    assert res["label"] == "crack"
    assert res["bbox_2d"] == [256, 144, 384, 216]


def test_objects_to_answer_empty():
    ans = objects_to_answer([])
    assert ans == "[]"
    parsed = parse_output(ans)
    assert parsed == []


def test_objects_to_answer_ordering_and_validity():
    objs = [
        {"label": "manhole", "yolo": [0.8, 0.8, 0.1, 0.1]}, # y1 ≈ 750
        {"label": "pothole", "yolo": [0.2, 0.2, 0.1, 0.1]}, # y1 ≈ 150
        {"label": "crack", "yolo": [0.5, 0.2, 0.1, 0.1]},   # y1 ≈ 150, but x1 > pothole
    ]
    ans = objects_to_answer(objs)
    parsed = parse_output(ans)
    assert parsed is not None
    assert len(parsed) == 3
    # Check order: top-to-bottom, left-to-right
    labels_order = [p["label"] for p in parsed]
    assert labels_order == ["pothole", "crack", "manhole"]


def test_box_to_pixel():
    px = box_to_pixel([400, 400, 600, 600], 640, 360)
    assert px == [256.0, 144.0, 384.0, 216.0]
