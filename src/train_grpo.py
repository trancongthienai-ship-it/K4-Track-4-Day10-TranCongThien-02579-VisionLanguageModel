"""Huấn luyện GRPO (Group Relative Policy Optimization) với IoU Reward cho E5 (Bonus)."""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from PIL import Image
import torch
import yaml
from peft import PeftModel
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.labcommon import read_jsonl
from src.label_utils import PROMPT_NORM1000
from src.model_utils import load_base_model_and_processor
from src.reward import compute_iou_reward


def train_grpo(cfg: Dict[str, Any], output_dir: Path, data_root: Path):
    """Vòng lặp huấn luyện GRPO tối ưu hoá hàm phần thưởng IoU."""
    grpo_cfg = cfg.get("grpo", {})
    num_generations = grpo_cfg.get("num_generations", 4)
    max_steps = grpo_cfg.get("max_steps", 60)
    lr = float(grpo_cfg.get("learning_rate", 5.0e-6))
    max_completion_length = grpo_cfg.get("max_completion_length", 128)
    model_id = cfg.get("model_id", "Qwen/Qwen3-VL-4B-Instruct")
    init_ckpt = Path("work/checkpoints/e1")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path("work/e5_grpo.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print("==================== BẮT ĐẦU HUẤN LUYỆN GRPO E5 ====================")
    print(f"Cấu hình: num_generations={num_generations}, max_steps={max_steps}, lr={lr}")
    print(f"Adapter xuất phát: {init_ckpt} | Log: {log_file}")

    # Nạp base model & processor
    model, processor = load_base_model_and_processor(
        model_id=model_id,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        is_train=True,
    )

    if init_ckpt.exists():
        print(f"Nạp adapter tiền huấn luyện từ {init_ckpt}...")
        model = PeftModel.from_pretrained(model, str(init_ckpt), is_trainable=True)
    else:
        print("[CẢNH BÁO] Không tìm thấy work/checkpoints/e1, khởi tạo LoRA mới cho E5...")
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=16,
            alpha=32,
            dropout=0.05,
            target_modules=r".*q_proj|.*k_proj|.*v_proj|.*o_proj",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    # Đảm bảo trainable params là float32
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        foreach=False,
    )

    # Nạp tập dữ liệu huấn luyện
    train_entries = read_jsonl(data_root / "splits" / "train.jsonl")
    log_lines = []

    model.train()
    start_time = time.time()
    pbar = tqdm(total=max_steps, desc="GRPO Steps")

    for step in range(1, max_steps + 1):
        # Lấy một mẫu ngẫu nhiên từ train split
        entry = train_entries[step % len(train_entries)]
        img_path = data_root / entry["image_path"]
        gt_boxes = entry.get("boxes", [])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            pbar.update(1)
            continue

        prompt = PROMPT_NORM1000
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        # 1. Sinh nhóm G completions với sampling
        model.eval()
        with torch.no_grad():
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=max_completion_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=num_generations,
            )

        gen_tokens = gen_outputs[:, prompt_len:]
        decoded_texts = processor.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        # 2. Tính phần thưởng IoU cho từng câu trả lời trong nhóm
        rewards = []
        for completion in decoded_texts:
            r = compute_iou_reward(completion.strip(), gt_boxes)
            rewards.append(r)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=model.device)
        mean_r = rewards_tensor.mean().item()
        std_r = rewards_tensor.std().item()

        # Tính Relative Advantage trong nhóm
        if std_r > 1e-6:
            advantages = (rewards_tensor - mean_r) / (std_r + 1e-8)
        else:
            advantages = rewards_tensor - mean_r

        # 3. Tính Policy Gradient Loss cho nhóm
        model.train()
        optimizer.zero_grad()

        # Nhân bản inputs để tính log-likelihood cho G câu trả lời
        expanded_input_ids = gen_outputs
        attention_mask = (expanded_input_ids != processor.tokenizer.pad_token_id).long()
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            expanded_pixel_values = pixel_values.repeat(num_generations, 1, 1, 1)
        else:
            expanded_pixel_values = None

        model_kwargs = {"input_ids": expanded_input_ids, "attention_mask": attention_mask}
        if expanded_pixel_values is not None:
            model_kwargs["pixel_values"] = expanded_pixel_values
        if "image_grid_thw" in inputs:
            model_kwargs["image_grid_thw"] = inputs["image_grid_thw"].repeat(num_generations, 1)

        outputs = model(**model_kwargs)
        logits = outputs.logits[:, prompt_len - 1 : -1, :]
        targets = expanded_input_ids[:, prompt_len:]

        # Token-level log-probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
        seq_log_probs = token_log_probs.mean(dim=-1)

        # Objective: - advantage * seq_log_prob
        loss = -(advantages.detach() * seq_log_probs).mean()

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        log_entry = {
            "step": step,
            "loss": round(loss.item(), 4) if torch.isfinite(loss) else 0.0,
            "mean_reward": round(mean_r, 4),
            "std_reward": round(std_r, 4),
            "best_reward": round(max(rewards), 4),
            "sample_completion": decoded_texts[0][:100],
        }
        log_lines.append(json.dumps(log_entry, ensure_ascii=False))

        pbar.set_postfix(mean_r=f"{mean_r:.3f}", loss=f"{log_entry['loss']:.4f}")
        pbar.update(1)

    pbar.close()
    elapsed_min = (time.time() - start_time) / 60.0

    # Lưu log và checkpoint
    log_file.write_text("\n".join(log_lines), encoding="utf-8")
    model.save_pretrained(output_dir)

    run_meta = {
        "name": "E5_GRPO",
        "steps": max_steps,
        "num_generations": num_generations,
        "train_minutes": round(elapsed_min, 2),
        "final_mean_reward": log_lines[-1] if log_lines else {},
    }
    Path("work/e5_run.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã hoàn thành GRPO sau {elapsed_min:.2f} phút!")
    print(f"Log lưu tại {log_file} | Checkpoint lưu tại {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train GRPO with IoU Reward")
    parser.add_argument("--config", default="configs/e5.yaml")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="work/checkpoints/e5")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train_grpo(cfg, Path(args.output_dir), Path(args.data_root))


if __name__ == "__main__":
    main()
