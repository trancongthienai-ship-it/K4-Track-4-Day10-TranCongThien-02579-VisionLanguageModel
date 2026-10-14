"""Script trực quan hoá 6 ảnh dự đoán đúng/sai có chú thích cho báo cáo Lab Day 10."""
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.labcommon import read_jsonl
from scripts.score import iou_xyxy, parse_output

COLOR_GT = "#00FF00"      # Xanh lá: Ground Truth
COLOR_TP = "#00BFFF"      # Xanh dương: Dự đoán đúng (True Positive, IoU >= 0.5)
COLOR_FP = "#FF3030"      # Đỏ: Dự đoán sai / thừa (False Positive)


def denorm_bbox(bbox_1000, width, height):
    """Chuyển toạ độ 0-1000 sang toạ độ pixel [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox_1000
    return [
        int(x1 * width / 1000.0),
        int(y1 * height / 1000.0),
        int(x2 * width / 1000.0),
        int(y2 * height / 1000.0),
    ]


def visualize_predictions(
    data_root: str = "data",
    pred_path: str = "work/e1.jsonl",
    output_dir: str = "report/figures",
    num_samples: int = 6,
):
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_entries = read_jsonl(data_root / "splits" / "test.jsonl")
    preds = {x["image_id"]: x["raw"] for x in read_jsonl(pred_path)}

    print(f"Đang phân tích {len(test_entries)} ảnh test...")
    analyzed = []

    for entry in test_entries:
        img_id = entry["image_id"]
        raw = preds.get(img_id, "[]")
        pred_boxes = parse_output(raw) or []
        gt_boxes = entry.get("boxes", [])

        # Ghép đôi kiểm tra TP, FP
        tp_count = 0
        fp_count = 0
        fn_count = len(gt_boxes)

        used_gt = set()
        for p in pred_boxes:
            p_label = p["label"]
            p_box = p["bbox"]
            matched = False
            for idx, g in enumerate(gt_boxes):
                if idx in used_gt:
                    continue
                if g["label"] == p_label:
                    iou = iou_xyxy(p_box, g["bbox"])
                    if iou >= 0.5:
                        used_gt.add(idx)
                        tp_count += 1
                        matched = True
                        break
            if not matched:
                fp_count += 1
        fn_count -= len(used_gt)

        analyzed.append({
            "entry": entry,
            "pred_boxes": pred_boxes,
            "gt_boxes": gt_boxes,
            "tp": tp_count,
            "fp": fp_count,
            "fn": fn_count,
        })

    # Chọn 3 ca thành công (nhiều TP, ít FP/FN) và 3 ca thất bại (có FP hoặc FN)
    success_cases = sorted(analyzed, key=lambda x: (x["tp"], -x["fp"]), reverse=True)[:3]
    error_cases = sorted(analyzed, key=lambda x: (x["fp"] + x["fn"], -x["tp"]), reverse=True)[:3]
    selected = [("success", c) for c in success_cases] + [("error", c) for c in error_cases]

    print(f"Bắt đầu vẽ {len(selected)} ảnh minh hoạ vào {output_dir}...")
    font = None
    try:
        font = ImageFont.load_default()
    except Exception:
        pass

    results_info = []

    for idx, (case_type, item) in enumerate(selected, 1):
        entry = item["entry"]
        img_path = data_root / entry["image_path"]
        if not img_path.exists():
            print(f"[CẢNH BÁO] Không tìm thấy ảnh: {img_path}")
            continue

        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        draw = ImageDraw.Draw(im)

        # 1. Vẽ Ground Truth (Xanh lá, viền nét đứt hoặc nét dày)
        for g in item["gt_boxes"]:
            box_px = denorm_bbox(g["bbox"], w, h)
            draw.rectangle(box_px, outline=COLOR_GT, width=3)
            draw.text((box_px[0], max(0, box_px[1] - 12)), f"GT: {g['label']}", fill=COLOR_GT, font=font)

        # 2. Vẽ Predictions
        used_gt = set()
        for p in item["pred_boxes"]:
            p_label = p["label"]
            p_box = p["bbox"]
            is_tp = False
            for g_idx, g in enumerate(item["gt_boxes"]):
                if g_idx not in used_gt and g["label"] == p_label:
                    if iou_xyxy(p_box, g["bbox"]) >= 0.5:
                        used_gt.add(g_idx)
                        is_tp = True
                        break

            box_px = denorm_bbox(p_box, w, h)
            color = COLOR_TP if is_tp else COLOR_FP
            tag = "TP" if is_tp else "FP"
            draw.rectangle(box_px, outline=color, width=2)
            draw.text((box_px[0], min(h - 12, box_px[3] + 2)), f"Pred ({tag}): {p_label}", fill=color, font=font)

        out_name = f"pred_sample_{idx}_{case_type}_{entry['image_id']}.jpg"
        save_file = output_dir / out_name
        im.save(save_file, quality=92)
        desc = (
            f"Ảnh {idx} (`{out_name}`): {case_type.upper()} | "
            f"GT: {[g['label'] for g in item['gt_boxes']]} | "
            f"Pred: {[p['label'] for p in item['pred_boxes']]} (TP={item['tp']}, FP={item['fp']}, FN={item['fn']})"
        )
        results_info.append(desc)
        print(f" -> Đã lưu: {save_file}")

    info_txt = output_dir / "samples_info.txt"
    info_txt.write_text("\n".join(results_info), encoding="utf-8")
    print(f"Hoàn thành! Đã lưu thông tin tại {info_txt}")


if __name__ == "__main__":
    visualize_predictions()
