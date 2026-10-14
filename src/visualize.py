"""Script trực quan hoá 6 ảnh dự đoán đúng/sai có chú thích cho báo cáo Lab Day 10."""
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, iou_xyxy, parse_output

COLOR_GT = "#00FF00"         # Xanh lá: Ground Truth
COLOR_TP = "#00BFFF"         # Xanh dương sáng: True Positive (Dự đoán đúng, IoU >= 0.5)
COLOR_FP = "#FF3030"         # Đỏ: False Positive (Dự đoán sai / báo động giả)
COLOR_FN = "#FFA500"         # Cam: False Negative (Bỏ sót Ground Truth)


def denorm_bbox(bbox_1000, width, height):
    """Chuyển toạ độ 0-1000 sang toạ độ pixel [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox_1000
    return [
        max(0, min(width - 1, int(x1 * width / 1000.0))),
        max(0, min(height - 1, int(y1 * height / 1000.0))),
        max(0, min(width - 1, int(x2 * width / 1000.0))),
        max(0, min(height - 1, int(y2 * height / 1000.0))),
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

    print(f"Đang phân tích Ground Truth và Dự đoán trên {len(test_entries)} ảnh test...")
    analyzed = []

    for entry in test_entries:
        img_id = entry["image_id"]
        raw = preds.get(img_id, "[]")
        pred_boxes = parse_output(raw) or []
        gt_boxes = _gt_boxes(entry)  # Lấy đúng key 'objects' và chuyển sang norm1000

        # Ghép đôi kiểm tra TP, FP, FN
        tp_count = 0
        fp_count = 0
        used_gt = set()
        matched_preds = {}

        for p_idx, p in enumerate(pred_boxes):
            p_label = p["label"]
            p_box = p["bbox"]
            best_iou = 0.0
            best_g_idx = -1

            for g_idx, g in enumerate(gt_boxes):
                if g_idx in used_gt:
                    continue
                if g["label"] == p_label:
                    iou = iou_xyxy(p_box, g["bbox"])
                    if iou >= 0.5 and iou > best_iou:
                        best_iou = iou
                        best_g_idx = g_idx

            if best_g_idx >= 0:
                used_gt.add(best_g_idx)
                tp_count += 1
                matched_preds[p_idx] = (True, best_iou)
            else:
                fp_count += 1
                matched_preds[p_idx] = (False, 0.0)

        fn_count = len(gt_boxes) - len(used_gt)

        analyzed.append({
            "entry": entry,
            "pred_boxes": pred_boxes,
            "gt_boxes": gt_boxes,
            "matched_preds": matched_preds,
            "used_gt": used_gt,
            "tp": tp_count,
            "fp": fp_count,
            "fn": fn_count,
        })

    # Lọc chỉ xét các ảnh thật sự có hư hại trong Ground Truth
    with_gt = [c for c in analyzed if len(c["gt_boxes"]) > 0]

    # Chọn 3 ca thành công HOÀN HẢO: TP >= 1, FP == 0, FN == 0 (hoặc ít FN nhất)
    # Tìm các ca hoàn hảo 100% (không có bất kỳ box thừa hay box bỏ sót nào)
    perfect_cases = [c for c in with_gt if c["tp"] >= 1 and c["fp"] == 0 and c["fn"] == 0]
    perfect_cases.sort(key=lambda x: x["tp"], reverse=True)

    if len(perfect_cases) >= 3:
        success_cases = perfect_cases[:3]
    else:
        # Nếu chưa đủ 3 ca hoàn hảo tuyệt đối, lấy thêm các ca có FP == 0 và FN nhỏ nhất
        fallback = [c for c in with_gt if c not in perfect_cases and c["tp"] >= 1 and c["fp"] == 0]
        fallback.sort(key=lambda x: (x["tp"], -x["fn"]), reverse=True)
        success_cases = (perfect_cases + fallback)[:3]

    # Chọn 3 ca thất bại: có lỗi FP hoặc FN rõ rệt nhất để phục vụ phân tích lỗi
    error_candidates = [c for c in with_gt if (c["fp"] > 0 or c["fn"] > 0) and c not in success_cases]
    error_candidates.sort(key=lambda x: (x["fp"] + x["fn"], x["fp"]), reverse=True)
    error_cases = error_candidates[:3]

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

        # 1. Vẽ Ground Truth (Xanh lá)
        for g_idx, g in enumerate(item["gt_boxes"]):
            box_px = denorm_bbox(g["bbox"], w, h)
            is_detected = g_idx in item["used_gt"]
            gt_tag = "GT" if is_detected else "GT (bỏ sót)"
            gt_color = COLOR_GT if is_detected else COLOR_FN
            draw.rectangle(box_px, outline=gt_color, width=3)
            draw.text((box_px[0], max(0, box_px[1] - 13)), f"{gt_tag}: {g['label']}", fill=gt_color, font=font)

        # 2. Vẽ Predictions (Xanh dương = TP, Đỏ = FP)
        for p_idx, p in enumerate(item["pred_boxes"]):
            p_label = p["label"]
            p_box = p["bbox"]
            is_tp, iou_val = item["matched_preds"].get(p_idx, (False, 0.0))

            box_px = denorm_bbox(p_box, w, h)
            color = COLOR_TP if is_tp else COLOR_FP
            tag = f"Pred TP (IoU={iou_val:.2f}): {p_label}" if is_tp else f"Pred FP: {p_label}"
            draw.rectangle(box_px, outline=color, width=2)
            draw.text((box_px[0], min(h - 13, box_px[3] + 2)), tag, fill=color, font=font)

        out_name = f"pred_sample_{idx}_{case_type}_{entry['image_id']}.jpg"
        save_file = output_dir / out_name
        im.save(save_file, quality=95)

        gt_labels = [g["label"] for g in item["gt_boxes"]]
        pred_labels = [p["label"] for p in item["pred_boxes"]]
        desc = (
            f"Ảnh {idx} (`{out_name}`): {case_type.upper()} | "
            f"GT: {gt_labels} | Pred: {pred_labels} (TP={item['tp']}, FP={item['fp']}, FN={item['fn']})"
        )
        results_info.append(desc)
        print(f" -> Đã lưu: {save_file} [{desc}]")

    info_txt = output_dir / "samples_info.txt"
    info_txt.write_text("\n".join(results_info), encoding="utf-8")
    print(f"\nHoàn thành! Đã lưu thông tin chi tiết tại {info_txt}")


if __name__ == "__main__":
    visualize_predictions()
