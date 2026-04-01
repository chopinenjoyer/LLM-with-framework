from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizer import ByteTokenizer


def prompt_from_instruction(instruction: str) -> str:
    return f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"


def load_text_files(paths: list[str]) -> list[str]:
    texts: list[str] = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_dir():
            for child in sorted(path.rglob("*.txt")):
                texts.append(child.read_text(encoding="utf-8"))
        else:
            texts.append(path.read_text(encoding="utf-8"))
    return texts


def load_shard_manifest(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class InstructionExample:
    instruction: str
    response: str


def load_instruction_jsonl(path: str | Path) -> list[InstructionExample]:
    examples: list[InstructionExample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        examples.append(InstructionExample(instruction=item["instruction"], response=item["response"]))
    return examples


class PackedTokenDataset(Dataset):
    def __init__(self, tokens: np.ndarray, block_size: int) -> None:
        self.tokens = tokens
        self.block_size = block_size
        self.num_items = max(1, (max(len(tokens) - 1, 1) + block_size - 1) // block_size)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        end = start + self.block_size + 1
        chunk = self.tokens[start:end]
        if len(chunk) < self.block_size + 1:
            pad = np.full(self.block_size + 1 - len(chunk), fill_value=0, dtype=np.int64)
            chunk = np.concatenate([chunk, pad])
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[:-1], dtype=torch.long)
        return x, y


class ShardedPackedTokenDataset(Dataset):
    def __init__(self, shard_paths: list[str], block_size: int) -> None:
        if not shard_paths:
            raise ValueError("At least one shard is required")
        self.block_size = block_size
        self.shard_paths = [str(path) for path in shard_paths]
        self.shards = [np.load(path, mmap_mode="r") for path in self.shard_paths]
        self.items_per_shard: list[int] = []
        for shard in self.shards:
            num_items = max(1, (max(len(shard) - 1, 1) + block_size - 1) // block_size)
            self.items_per_shard.append(num_items)
        self.cumulative_items = np.cumsum(self.items_per_shard)

    def __len__(self) -> int:
        return int(self.cumulative_items[-1])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_idx = int(np.searchsorted(self.cumulative_items, idx, side="right"))
        previous = 0 if shard_idx == 0 else int(self.cumulative_items[shard_idx - 1])
        local_idx = idx - previous
        shard = self.shards[shard_idx]
        start = local_idx * self.block_size
        end = start + self.block_size + 1
        chunk = shard[start:end]
        if len(chunk) < self.block_size + 1:
            pad = np.full(self.block_size + 1 - len(chunk), fill_value=0, dtype=np.int64)
            chunk = np.concatenate([chunk, pad])
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[:-1], dtype=torch.long)
        return x, y


class SFTDataset(Dataset):
    def __init__(self, items: list[dict[str, list[int]]], block_size: int) -> None:
        self.items = items
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.items[idx]["input_ids"]
        labels = self.items[idx]["labels"]
        if len(input_ids) > self.block_size:
            input_ids = input_ids[-self.block_size :]
            labels = labels[-self.block_size :]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_sft(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(item[0].size(0) for item in batch)
    input_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    for input_ids, labels in batch:
        input_pad = torch.full((max_len - input_ids.size(0),), pad_token_id, dtype=torch.long)
        label_pad = torch.full((max_len - labels.size(0),), -100, dtype=torch.long)
        input_rows.append(torch.cat([input_ids, input_pad], dim=0))
        label_rows.append(torch.cat([labels, label_pad], dim=0))
    return torch.stack(input_rows), torch.stack(label_rows)


def build_sft_item(tokenizer: ByteTokenizer, instruction: str, response: str) -> dict[str, list[int]]:
    prompt = prompt_from_instruction(instruction)
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    response_ids = tokenizer.encode(response, add_bos=False, add_eos=True)
    input_ids = prompt_ids + response_ids
    labels = ([-100] * len(prompt_ids)) + response_ids
    return {"input_ids": input_ids, "labels": labels}
