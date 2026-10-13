"""Script huấn luyện QLoRA cho Qwen3-VL-4B (E1, E2, E3, E4)."""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import DataCollatorForQwenVL, RoadDamageDataset, encode_example
from src.model_utils import (
    apply_setup,
    load_base_model_and_processor,
    save_projector_weights,
)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine LR schedule có warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train(cfg: Dict[str, Any], output_dir: Path, data_root: Path = Path("data")):
    exp_name = cfg.get("name", "E1")
    model_id = cfg.get("model_id", "Qwen/Qwen3-VL-4B-Instruct")
    lr = float(cfg.get("lr", 2e-4))
    batch_size = int(cfg.get("batch_size", 2))
    grad_accum = int(cfg.get("grad_accum", 4))
    max_steps = int(cfg.get("max_steps", 300))
    train_projector = bool(cfg.get("train_projector", False))
    coord = cfg.get("coord", "norm1000")
    fraction = float(cfg.get("fraction", 1.0))
    seed = int(cfg.get("seed", 42))

    lora_cfg = cfg.get("lora", {})
    lora_r = int(lora_cfg.get("r", 16))
    lora_alpha = int(lora_cfg.get("alpha", 32))
    lora_dropout = float(lora_cfg.get("dropout", 0.05))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    print(f"\n==================== BẮT ĐẦU HUẤN LUYỆN {exp_name} ====================", flush=True)
    print(f"Cấu hình: lr={lr}, batch_size={batch_size}, grad_accum={grad_accum}, max_steps={max_steps}")
    print(f"LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout} | Train Projector: {train_projector}")

    # 1. Nạp Base Model & Processor
    model, processor = load_base_model_and_processor(
        model_id=model_id,
        device_map="auto",
        is_train=True,
    )

    # 2. Gradient Checkpointing & Setup LoRA / Projector
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    peft_model, setup_info = apply_setup(
        model=model,
        setup_name=exp_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        train_projector=train_projector,
    )

    # 3. Chuẩn bị Dataset & DataLoader
    train_dataset = RoadDamageDataset(
        jsonl_path=str(data_root / "splits" / "train.jsonl"),
        data_root=str(data_root),
        coord=coord,
        fraction=fraction,
        seed=seed,
    )

    def collate_wrapper(items):
        encoded_items = [
            encode_example(processor, it["image"], it["prompt"], it["answer"])
            for it in items
        ]
        collator = DataCollatorForQwenVL(pad_token_id=processor.tokenizer.pad_token_id)
        return collator(encoded_items)

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_wrapper,
        drop_last=True,
    )

    # 4. Optimizer, Scheduler, Scaler
    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.0)

    warmup_steps = int(0.05 * max_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    scaler = GradScaler()

    # 5. Training Loop
    peft_model.train()
    start_time = time.time()
    step = 0
    accum_count = 0
    running_loss = 0.0

    pbar = tqdm(total=max_steps, desc=f"Train {exp_name}")
    epoch = 0

    while step < max_steps:
        epoch += 1
        for batch in dataloader:
            batch = {k: v.to(peft_model.device) for k, v in batch.items()}

            with autocast(dtype=torch.float16):
                outputs = peft_model(**batch)
                loss = outputs.loss / grad_accum

            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss không hữu hạn: {loss.item()} (nguy cơ tràn số fp16)")

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum
            accum_count += 1

            if accum_count == grad_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                step += 1
                accum_count = 0
                avg_loss = running_loss / grad_accum
                running_loss = 0.0

                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}", "ep": epoch})
                pbar.update(1)

                if step % 100 == 0 or step == max_steps:
                    ckpt_step_dir = output_dir / f"checkpoint-{step}"
                    peft_model.save_pretrained(ckpt_step_dir)
                    if train_projector:
                        save_projector_weights(peft_model, ckpt_step_dir)

                if step >= max_steps:
                    break

    pbar.close()
    elapsed_minutes = (time.time() - start_time) / 60.0

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024.0**3)

    print(f"\nHuấn luyện hoàn thành trong {elapsed_minutes:.2f} phút!")
    print(f"VRAM đỉnh: {peak_vram_gb:.2f} GB | Trainable params: {setup_info['trainable_params']:,}")

    # 6. Lưu kết quả cuối cùng
    output_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(output_dir)
    if train_projector:
        save_projector_weights(peft_model, output_dir)

    run_meta = {
        "name": exp_name,
        "trainable_params": setup_info["trainable_params"],
        "peak_vram_gb": round(peak_vram_gb, 2),
        "train_minutes": round(elapsed_minutes, 2),
        "steps": max_steps,
        "lr": lr,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
    }

    meta_file = Path(f"work/{exp_name.lower()}_run.json")
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu run metadata vào {meta_file}")
    return run_meta


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL-4B LoRA")
    parser.add_argument("--config", required=True, help="File cấu hình yaml (configs/e1.yaml, ...)")
    parser.add_argument("--output-dir", default=None, help="Thư mục lưu adapter")
    parser.add_argument("--data-root", default="data")
    # Tùy chọn override cho ablation E4
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--coord", default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.name:
        cfg["name"] = args.name
    if args.rank is not None:
        cfg.setdefault("lora", {})["r"] = args.rank
        cfg["lora"]["alpha"] = 2 * args.rank
    if args.fraction is not None:
        cfg["fraction"] = args.fraction
    if args.coord is not None:
        cfg["coord"] = args.coord

    exp_name = cfg.get("name", "E1")
    output_dir = Path(args.output_dir or f"work/checkpoints/{exp_name.lower()}")

    train(cfg, output_dir, data_root=Path(args.data_root))


if __name__ == "__main__":
    main()
