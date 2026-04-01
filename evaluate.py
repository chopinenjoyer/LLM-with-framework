from __future__ import annotations

import argparse
import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import (
    PackedTokenDataset,
    SFTDataset,
    ShardedPackedTokenDataset,
    collate_sft,
    load_instruction_jsonl,
    load_shard_manifest,
    prompt_from_instruction,
)
from model import DecoderOnlyLM, LLMConfig
from tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretraining and fine-tuning checkpoints.")
    parser.add_argument("--data-dir", default="artifacts/datasets")
    parser.add_argument("--checkpoint", default="artifacts/sft_model.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    return parser.parse_args()


def load_model(checkpoint_path: str) -> tuple[DecoderOnlyLM, ByteTokenizer]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    tokenizer = ByteTokenizer.load(checkpoint["tokenizer_path"])
    config = LLMConfig(**checkpoint["model_config"])
    model = DecoderOnlyLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


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


def generate_response(
    model: DecoderOnlyLM,
    tokenizer: ByteTokenizer,
    instruction: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> str:
    prompt = prompt_from_instruction(instruction)
    input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True, add_eos=False)], dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None if temperature <= 0 else top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.spec.eos_token_id,
        )
    response_ids = generated[0].tolist()[input_ids.size(1) :]
    response = tokenizer.decode(response_ids).strip()
    compact = response.replace(" ", "")
    if compact and len(compact) >= 12 and len(set(compact)) <= 2:
        return ""
    return response


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
        if normalize_text(prediction) == normalize_text(example.response):
            correct += 1
    return correct / max(len(examples), 1)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    model, tokenizer = load_model(args.checkpoint)
    ppl = evaluate_perplexity(model, data_dir, args.batch_size)
    sft_loss = evaluate_sft_loss(model, tokenizer, data_dir, args.batch_size)
    exact_match = evaluate_exact_match(
        model,
        tokenizer,
        Path("data/instructions_val.jsonl"),
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.repetition_penalty,
    )
    print(f"pretrain_perplexity={ppl:.4f}")
    print(f"sft_val_loss={sft_loss:.4f}")
    print(f"sft_exact_match={exact_match:.2%}")


if __name__ == "__main__":
    main()
