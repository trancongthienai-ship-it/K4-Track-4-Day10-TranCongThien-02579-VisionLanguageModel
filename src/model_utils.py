"""Quản lý khởi tạo mô hình Qwen3-VL-4B, QLoRA, Projector và checkpoints."""
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration


PROJECTOR_PARAM_PATTERNS = ["visual.merger", "visual.deepstack_merger_list"]


def get_quantization_config() -> BitsAndBytesConfig:
    """Cấu hình lượng tử hoá NF4 4-bit.

    Bắt buộc skip 'visual' và 'lm_head' để giữ ViT & Projector ở fp16,
    cho phép huấn luyện đầy đủ Projector ở float32 trong E2 và E3.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )


def load_base_model_and_processor(
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    device_map: Any = None,
    is_train: bool = False,
) -> Tuple[Any, Any]:
    """Nạp base model lượng tử hoá 4-bit và processor."""
    if device_map is None:
        device_map = {"": 0} if torch.cuda.is_available() else "auto"

    quant_cfg = get_quantization_config()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_cfg,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    if is_train:
        processor.tokenizer.padding_side = "right"
    else:
        processor.tokenizer.padding_side = "left"

    return model, processor


def apply_setup(
    model: Any,
    setup_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    train_projector: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Cấu hình LoRA và mở/đóng băng các khối theo setup (E1, E2, E3, E4)."""
    # 1. Đóng băng toàn bộ tham số
    for param in model.parameters():
        param.requires_grad = False

    # Cần cho gradient checkpointing
    model.enable_input_require_grads()

    # 2. Xác định target modules theo regex
    # LLM (36 layers * 7 modules = 252 modules)
    llm_pattern = r".*language_model.*layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
    # ViT (24 blocks * 4 modules = 96 modules)
    vit_pattern = r".*visual.*blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))"

    if setup_name == "E3" or "vit" in setup_name.lower():
        target_pattern = rf"({llm_pattern}|{vit_pattern})"
    else:
        target_pattern = llm_pattern

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_pattern,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_config)

    # 3. Mở projector cho E2, E3 (nếu train_projector=True)
    projector_param_count = 0
    if train_projector or setup_name in ("E2", "E3"):
        for mod_name, module in peft_model.named_modules():
            if any(p in mod_name for p in PROJECTOR_PARAM_PATTERNS):
                module.to(dtype=torch.float32)
                for p_name, param in list(module._parameters.items()):
                    if param is not None:
                        module._parameters[p_name] = torch.nn.Parameter(
                            param.data.to(dtype=torch.float32),
                            requires_grad=True,
                        )
                        projector_param_count += param.numel()

        for name, param in peft_model.named_parameters():
            if any(p in name for p in PROJECTOR_PARAM_PATTERNS):
                param.requires_grad = True

    # Đếm tham số huấn luyện
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in peft_model.parameters())

    info = {
        "setup": setup_name,
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": (trainable_params / all_params) * 100.0,
        "projector_params": projector_param_count,
    }
    print(
        f"[{setup_name}] Trainable params: {trainable_params:,} / {all_params:,} "
        f"({info['trainable_percent']:.2f}%) | Projector: {projector_param_count:,}"
    )
    return peft_model, info


def save_projector_weights(model: Any, output_dir: Path):
    """Lưu riêng trọng số Projector (visual.merger và visual.deepstack_merger_list)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    projector_state = {}
    for name, param in model.named_parameters():
        if any(p in name for p in PROJECTOR_PARAM_PATTERNS) and param.requires_grad:
            projector_state[name] = param.detach().cpu()
    if projector_state:
        torch.save(projector_state, output_dir / "projector_weights.pt")
        print(f"Đã lưu {len(projector_state)} tensors projector vào {output_dir / 'projector_weights.pt'}")


def load_projector_weights(model: Any, checkpoint_dir: Path):
    """Nạp lại trọng số Projector đã huấn luyện."""
    weights_path = Path(checkpoint_dir) / "projector_weights.pt"
    if not weights_path.exists():
        return False
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    with torch.no_grad():
        model_dict = dict(model.named_parameters())
        for name, tensor in state_dict.items():
            if name in model_dict:
                model_dict[name].copy_(tensor.to(device=model_dict[name].device, dtype=model_dict[name].dtype))
    print(f"Đã nạp thành công {len(state_dict)} tensors projector từ {weights_path}")
    return True
