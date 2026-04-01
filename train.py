from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import QATransformer, Vocabulary, load_dataset


class QADataset(Dataset):
    def __init__(self, encoded_questions: list[list[int]], labels: list[int]) -> None:
        self.encoded_questions = encoded_questions
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.encoded_questions[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny QA model with PyTorch.")
    parser.add_argument("--dataset", default="data/qa_dataset.json")
    parser.add_argument("--output", default="artifacts/qa_model.pt")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples, answers = load_dataset(args.dataset)
    vocab = Vocabulary()
    for example in examples:
        vocab.add_sentence(example.question)

    encoded_questions = [vocab.encode(example.question, args.max_length) for example in examples]
    labels = [example.answer_id for example in examples]
    dataset = QADataset(encoded_questions, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = QATransformer(
        vocab_size=vocab.size,
        num_answers=len(answers),
        pad_token_id=vocab.token_to_id[Vocabulary.PAD],
    )

    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_questions, batch_labels in dataloader:
            model.zero_grad(set_to_none=True)
            logits = model(batch_questions)
            loss = criterion(logits, batch_labels)
            loss.backward()

            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter -= args.lr * parameter.grad

            running_loss += loss.item() * batch_labels.size(0)
            predictions = logits.argmax(dim=1)
            correct += predictions.eq(batch_labels).sum().item()
            total += batch_labels.size(0)

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            avg_loss = running_loss / total
            accuracy = correct / total
            print(f"epoch={epoch:03d} loss={avg_loss:.4f} accuracy={accuracy:.2%}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "answers": answers,
            "vocab": vocab.token_to_id,
            "max_length": args.max_length,
            "examples": [{"question": example.question, "answer_id": example.answer_id} for example in examples],
        },
        output_path,
    )

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "num_examples": len(examples),
                "num_answers": len(answers),
                "vocab_size": vocab.size,
                "epochs": args.epochs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
