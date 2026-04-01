from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import PackedTokenDataset
from model import DecoderOnlyLM, LLMConfig
from training_utils import ManualAdamW, save_checkpoint, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a decoder-only language model from scratch.")
    parser.add_argument("--data-dir", default="artifacts/datasets")
    parser.add_argument("--output", default="artifacts/pretrained_model.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_tokens = np.load(data_dir / "pretrain_train.npy")
    val_tokens = np.load(data_dir / "pretrain_val.npy")

    train_dataset = PackedTokenDataset(train_tokens, block_size=args.block_size)
    val_dataset = PackedTokenDataset(val_tokens, block_size=args.block_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    config = LLMConfig(
        block_size=args.block_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = DecoderOnlyLM(config)
    optimizer = ManualAdamW(list(model.parameters()), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        batch_count = 0
        for input_ids, labels in train_loader:
            optimizer.zero_grad()
            _, loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batch_count += 1

        train_loss = running_loss / max(batch_count, 1)
        val_loss = evaluate_loss(model, val_loader)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = str(data_dir / "tokenizer.json")
    save_checkpoint(
        output_path,
        model=model,
        model_config=config.to_dict(),
        tokenizer_path=tokenizer_path,
        extra={"stage": "pretrain"},
    )
    save_json(
        output_path.with_suffix(".json"),
        {
            "stage": "pretrain",
            "train_tokens": int(len(train_tokens)),
            "val_tokens": int(len(val_tokens)),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
        },
    )
    print(f"Checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()
