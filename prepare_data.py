from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_utils import build_sft_item, load_instruction_jsonl, load_text_files
from tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data for pretraining and fine-tuning.")
    parser.add_argument("--corpus", nargs="+", default=["data/raw_corpus.txt"])
    parser.add_argument("--sft-train", default="data/instructions_train.jsonl")
    parser.add_argument("--sft-val", default="data/instructions_val.jsonl")
    parser.add_argument("--output-dir", default="artifacts/datasets")
    parser.add_argument("--pretrain-val-ratio", type=float, default=0.05)
    parser.add_argument("--train-shard-size", type=int, default=200000)
    parser.add_argument("--val-shard-size", type=int, default=50000)
    return parser.parse_args()


def write_shards(tokens: np.ndarray, shard_dir: Path, prefix: str, shard_size: int) -> list[str]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[str] = []
    for shard_idx, start in enumerate(range(0, len(tokens), shard_size)):
        end = min(start + shard_size, len(tokens))
        shard_path = shard_dir / f"{prefix}_{shard_idx:05d}.npy"
        np.save(shard_path, tokens[start:end])
        shard_paths.append(str(shard_path))
    if not shard_paths:
        shard_path = shard_dir / f"{prefix}_00000.npy"
        np.save(shard_path, tokens)
        shard_paths.append(str(shard_path))
    return shard_paths


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteTokenizer()
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    corpus_texts = load_text_files(args.corpus)
    pretrain_tokens: list[int] = []
    for text in corpus_texts:
        pretrain_tokens.extend(tokenizer.encode(text, add_bos=True, add_eos=True))

    pretrain_array = np.asarray(pretrain_tokens, dtype=np.int64)
    split_index = max(1, int(len(pretrain_array) * (1.0 - args.pretrain_val_ratio)))
    train_array = pretrain_array[:split_index]
    val_array = pretrain_array[split_index:]
    if len(val_array) < 2:
        val_array = train_array[-max(2, len(train_array) // 20) :]
    train_shards = write_shards(train_array, output_dir / "pretrain_train_shards", "train", args.train_shard_size)
    val_shards = write_shards(val_array, output_dir / "pretrain_val_shards", "val", args.val_shard_size)
    manifest = {
        "train_shards": train_shards,
        "val_shards": val_shards,
        "train_tokens": int(len(train_array)),
        "val_tokens": int(len(val_array)),
    }
    (output_dir / "pretrain_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sft_train_examples = load_instruction_jsonl(args.sft_train)
    sft_val_examples = load_instruction_jsonl(args.sft_val)
    sft_train_items = [build_sft_item(tokenizer, item.instruction, item.response) for item in sft_train_examples]
    sft_val_items = [build_sft_item(tokenizer, item.instruction, item.response) for item in sft_val_examples]

    torch.save(sft_train_items, output_dir / "sft_train.pt")
    torch.save(sft_val_items, output_dir / "sft_val.pt")

    summary = (
        f"Prepared pretrain tokens: train={len(train_array)} val={len(val_array)}\n"
        f"Pretrain shards: train={len(train_shards)} val={len(val_shards)}\n"
        f"Prepared SFT examples: train={len(sft_train_items)} val={len(sft_val_items)}\n"
        f"Tokenizer saved to {tokenizer_path}"
    )
    print(summary)


if __name__ == "__main__":
    main()
