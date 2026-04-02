from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from llm.dataio import load_instruction_jsonl, load_shard_manifest
from llm.datasets import PackedTokenDataset, SFTDataset, ShardedPackedTokenDataset, collate_sft
from llm.generation import generate_response
from llm.models import DecoderOnlyLM
from llm.text import normalize_eval_text
from tokenizer import ByteTokenizer


def evaluate_loss(model: DecoderOnlyLM, dataloader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for input_ids, labels in dataloader:
            _, loss = model(input_ids, labels)
            total_loss += float(loss.item())
            total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


def evaluate_perplexity(model: DecoderOnlyLM, data_dir: Path, batch_size: int) -> float:
    manifest_path = data_dir / "pretrain_manifest.json"
    if manifest_path.exists():
        manifest = load_shard_manifest(manifest_path)
        dataset = ShardedPackedTokenDataset(manifest["val_shards"], block_size=model.config.block_size)
    else:
        val_tokens = np.load(data_dir / "pretrain_val.npy")
        dataset = PackedTokenDataset(val_tokens, block_size=model.config.block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    losses: list[float] = []
    with torch.no_grad():
        for input_ids, labels in dataloader:
            _, loss = model(input_ids, labels)
            losses.append(float(loss.item()))
    return math.exp(sum(losses) / max(len(losses), 1))


def evaluate_sft_loss(model: DecoderOnlyLM, tokenizer: ByteTokenizer, data_dir: Path, batch_size: int) -> float:
    items = torch.load(data_dir / "sft_val.pt", map_location="cpu")
    dataset = SFTDataset(items, block_size=model.config.block_size)
    collate = partial(collate_sft, pad_token_id=tokenizer.spec.pad_token_id)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    losses: list[float] = []
    with torch.no_grad():
        for input_ids, labels in dataloader:
            _, loss = model(input_ids, labels)
            losses.append(float(loss.item()))
    return sum(losses) / max(len(losses), 1)


def evaluate_exact_match(
    model: DecoderOnlyLM,
    tokenizer: ByteTokenizer,
    validation_path: Path,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> float:
    examples = load_instruction_jsonl(validation_path)
    correct = 0
    for example in examples:
        prediction = generate_response(
            model,
            tokenizer,
            example.instruction,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        )
        if normalize_eval_text(prediction) == normalize_eval_text(example.response):
            correct += 1
    return correct / max(len(examples), 1)
