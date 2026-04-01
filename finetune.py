from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_utils import SFTDataset, collate_sft
from model import DecoderOnlyLM, LLMConfig
from tokenizer import ByteTokenizer
from training_utils import ManualAdamW, save_checkpoint, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for the language model.")
    parser.add_argument("--data-dir", default="artifacts/datasets")
    parser.add_argument("--checkpoint", default="artifacts/pretrained_model.pt")
    parser.add_argument("--output", default="artifacts/sft_model.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    tokenizer = ByteTokenizer.load(checkpoint["tokenizer_path"])

    config = LLMConfig(**checkpoint["model_config"])
    model = DecoderOnlyLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_items = torch.load(data_dir / "sft_train.pt", map_location="cpu")
    val_items = torch.load(data_dir / "sft_val.pt", map_location="cpu")
    train_dataset = SFTDataset(train_items, block_size=config.block_size)
    val_dataset = SFTDataset(val_items, block_size=config.block_size)

    collate = partial(collate_sft, pad_token_id=tokenizer.spec.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

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
    save_checkpoint(
        output_path,
        model=model,
        model_config=config.to_dict(),
        tokenizer_path=checkpoint["tokenizer_path"],
        extra={"stage": "sft"},
    )
    save_json(
        output_path.with_suffix(".json"),
        {
            "stage": "sft",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "source_checkpoint": args.checkpoint,
        },
    )
    print(f"Checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()
