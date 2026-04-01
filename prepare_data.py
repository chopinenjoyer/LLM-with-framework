from __future__ import annotations

import argparse
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
    return parser.parse_args()


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

    np.save(output_dir / "pretrain_train.npy", train_array)
    np.save(output_dir / "pretrain_val.npy", val_array)

    sft_train_examples = load_instruction_jsonl(args.sft_train)
    sft_val_examples = load_instruction_jsonl(args.sft_val)
    sft_train_items = [build_sft_item(tokenizer, item.instruction, item.response) for item in sft_train_examples]
    sft_val_items = [build_sft_item(tokenizer, item.instruction, item.response) for item in sft_val_examples]

    torch.save(sft_train_items, output_dir / "sft_train.pt")
    torch.save(sft_val_items, output_dir / "sft_val.pt")

    summary = (
        f"Prepared pretrain tokens: train={len(train_array)} val={len(val_array)}\n"
        f"Prepared SFT examples: train={len(sft_train_items)} val={len(sft_val_items)}\n"
        f"Tokenizer saved to {tokenizer_path}"
    )
    print(summary)


if __name__ == "__main__":
    main()
