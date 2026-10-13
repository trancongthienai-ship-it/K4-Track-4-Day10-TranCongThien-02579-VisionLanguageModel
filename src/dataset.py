"""Dataset và Collate cho bài toán Grounding Qwen3-VL."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset

from scripts.labcommon import read_jsonl
from src.label_utils import PROMPT_NORM1000, PROMPT_PIXEL, objects_to_answer


class RoadDamageDataset(Dataset):
    """Dataset đọc dữ liệu từ data/splits/{split}.jsonl và ảnh từ data/."""

    def __init__(
        self,
        jsonl_path: str,
        data_root: str = "data",
        coord: str = "norm1000",
        fraction: float = 1.0,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.entries = read_jsonl(jsonl_path)
        self.coord = coord
        self.prompt = PROMPT_PIXEL if coord == "pixel" else PROMPT_NORM1000

        if fraction < 1.0:
            import random
            rng = random.Random(seed)
            sample_size = max(1, int(round(len(self.entries) * fraction)))
            self.entries = rng.sample(self.entries, sample_size)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        image_path = self.data_root / entry["image_path"]
        if not image_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

        image = Image.open(image_path).convert("RGB")
        answer_json = objects_to_answer(
            entry["objects"],
            coord=self.coord,
            width=entry["width"],
            height=entry["height"],
        )
        return {
            "image_id": entry["image_id"],
            "image": image,
            "prompt": self.prompt,
            "answer": answer_json,
            "width": entry["width"],
            "height": entry["height"],
            "objects": entry["objects"],
        }


def encode_example(processor, image: Image.Image, prompt: str, answer: Optional[str] = None):
    """Mã hoá một mẫu gồm ảnh, prompt và câu trả lời.

    - Nếu answer is not None: tính nhãn và mask loss (labels = -100 trên prompt và image tokens).
    - Nếu answer is None: chỉ mã hoá prompt cho suy luận.
    """
    user_msg = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }

    if answer is None:
        # Dùng cho suy luận (inference)
        inputs = processor.apply_chat_template(
            [user_msg],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs

    assistant_msg = {
        "role": "assistant",
        "content": [{"type": "text", "text": answer}],
    }

    prompt_enc = processor.apply_chat_template(
        [user_msg],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    full_enc = processor.apply_chat_template(
        [user_msg, assistant_msg],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    prompt_len = prompt_enc["input_ids"].shape[1]
    labels = full_enc["input_ids"].clone()
    # Mask loss: không tính loss trên prompt và image tokens
    labels[:, :prompt_len] = -100

    full_enc["labels"] = labels
    return full_enc


class DataCollatorForQwenVL:
    """Collate batch cho Qwen3-VL:

    - Pad input_ids, attention_mask, mm_token_type_ids, labels về độ dài lớn nhất trong batch.
    - Concatenate pixel_values và image_grid_thw theo chiều 0 (không pad).
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[1] for item in batch)
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        mm_token_type_ids_list = []

        has_mm_type = "mm_token_type_ids" in batch[0]
        has_labels = "labels" in batch[0]

        pixel_values_list = []
        image_grid_thw_list = []

        for item in batch:
            cur_len = item["input_ids"].shape[1]
            pad_len = max_len - cur_len

            # Right pad input_ids
            pad_input = torch.full((1, pad_len), self.pad_token_id, dtype=item["input_ids"].dtype)
            input_ids_list.append(torch.cat([item["input_ids"], pad_input], dim=1))

            # Right pad attention_mask
            pad_attn = torch.zeros((1, pad_len), dtype=item["attention_mask"].dtype)
            attention_mask_list.append(torch.cat([item["attention_mask"], pad_attn], dim=1))

            # Right pad mm_token_type_ids nếu có
            if has_mm_type:
                pad_mm = torch.zeros((1, pad_len), dtype=item["mm_token_type_ids"].dtype)
                mm_token_type_ids_list.append(torch.cat([item["mm_token_type_ids"], pad_mm], dim=1))

            # Right pad labels với -100 nếu có
            if has_labels:
                pad_label = torch.full((1, pad_len), -100, dtype=item["labels"].dtype)
                labels_list.append(torch.cat([item["labels"], pad_label], dim=1))

            if "pixel_values" in item:
                pixel_values_list.append(item["pixel_values"])
            if "image_grid_thw" in item:
                image_grid_thw_list.append(item["image_grid_thw"])

        batch_out = {
            "input_ids": torch.cat(input_ids_list, dim=0),
            "attention_mask": torch.cat(attention_mask_list, dim=0),
        }

        if has_mm_type:
            batch_out["mm_token_type_ids"] = torch.cat(mm_token_type_ids_list, dim=0)
        if has_labels:
            batch_out["labels"] = torch.cat(labels_list, dim=0)
        if pixel_values_list:
            batch_out["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        if image_grid_thw_list:
            batch_out["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)

        return batch_out
