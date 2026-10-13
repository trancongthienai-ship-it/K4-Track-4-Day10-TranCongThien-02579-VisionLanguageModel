"""Script suy luận và tự động chấm điểm cho E0..E4."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from PIL import Image
import torch
import yaml
from peft import PeftModel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.labcommon import read_jsonl, write_jsonl
from src.label_utils import PROMPT_NORM1000, PROMPT_PIXEL
from src.model_utils import (
    load_base_model_and_processor,
    load_projector_weights,
)


def run_inference(
    model: Any,
    processor: Any,
    entries: List[Dict[str, Any]],
    data_root: Path,
    prompt: str,
    output_pred_path: Path,
    batch_size: int = 4,
    max_new_tokens: int = 256,
):
    """Suy luận từng batch ảnh và ghi ra file jsonl."""
    model.eval()
    results = []

    print(f"Bắt đầu suy luận trên {len(entries)} ảnh với batch size = {batch_size}...")
    for i in tqdm(range(0, len(entries), batch_size), desc="Inference"):
        batch_entries = entries[i : i + batch_size]
        images = []
        for e in batch_entries:
            img_p = data_root / e["image_path"]
            images.append(Image.open(img_p).convert("RGB"))

        messages_list = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for img in images
        ]

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]

        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        gen_tokens = outputs[:, prompt_len:]
        decoded = processor.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for entry, raw_text in zip(batch_entries, decoded):
            results.append({"image_id": entry["image_id"], "raw": raw_text.strip()})

    output_pred_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_pred_path, results)
    print(f"Đã lưu kết quả suy luận vào {output_pred_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Inference and Scoring")
    parser.add_argument("--config", help="Đường dẫn file config yaml (vd configs/e0.yaml)")
    parser.add_argument("--name", default=None, help="Tên thí nghiệm (vd E0, E1)")
    parser.add_argument("--checkpoint-dir", default=None, help="Thư mục adapter đã lưu")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--pred-out", default=None)
    parser.add_argument("--evidence-out", default=None)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    # Nạp config nếu có
    cfg = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    exp_name = args.name or cfg.get("name", "E0")
    model_id = cfg.get("model_id", "Qwen/Qwen3-VL-4B-Instruct")
    coord = args.coord or cfg.get("coord", "norm1000")
    prompt = PROMPT_PIXEL if coord == "pixel" else PROMPT_NORM1000

    splits_dir = Path(args.splits_dir)
    data_root = Path(args.data_root)
    entries = read_jsonl(splits_dir / f"{args.split}.jsonl")

    pred_out = Path(args.pred_out or f"work/{exp_name.lower()}.jsonl")
    evidence_out = Path(args.evidence_out or f"report/evidence/{exp_name.lower()}.json")

    print(f"=== CHẠY SUY LUẬN {exp_name} TRÊN SPLIT {args.split} ===")
    print(f"Model ID: {model_id} | Coord: {coord} | Output: {pred_out}")

    model, processor = load_base_model_and_processor(
        model_id=model_id,
        device_map=args.device_map,
        is_train=False,
    )

    ckpt_dir = args.checkpoint_dir or f"work/checkpoints/{exp_name.lower()}"
    ckpt_path = Path(ckpt_dir)

    if exp_name != "E0" and ckpt_path.exists():
        print(f"Nạp adapter LoRA từ {ckpt_path}...")
        model = PeftModel.from_pretrained(model, str(ckpt_path))
        # Nạp projector weights nếu có
        load_projector_weights(model, ckpt_path)

    # Chạy suy luận
    run_inference(
        model=model,
        processor=processor,
        entries=entries,
        data_root=data_root,
        prompt=prompt,
        output_pred_path=pred_out,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # Tự động chấm điểm
    score_cmd = [
        sys.executable,
        "scripts/score.py",
        "--pred",
        str(pred_out),
        "--name",
        exp_name,
        "--split",
        args.split,
        "--coord",
        coord,
        "--out",
        str(evidence_out),
    ]

    # Kiểm tra metadata huấn luyện nếu có (từ work/<name>_run.json)
    run_meta_path = Path(f"work/{exp_name.lower()}_run.json")
    if run_meta_path.exists():
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        for k in ("trainable_params", "peak_vram_gb", "train_minutes"):
            if k in run_meta:
                score_cmd.extend(["--meta", f"{k}={run_meta[k]}"])

    print(f"\n--- Đang chấm điểm: {' '.join(score_cmd)} ---")
    subprocess.run(score_cmd, check=True)
    print(f"Đã tạo file bằng chứng: {evidence_out}")


if __name__ == "__main__":
    main()
