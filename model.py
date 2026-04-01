from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("'", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class Vocabulary:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self) -> None:
        self.token_to_id = {self.PAD: 0, self.UNK: 1}
        self.id_to_token = [self.PAD, self.UNK]

    def add_sentence(self, sentence: str) -> None:
        for token in normalize_text(sentence).split():
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.id_to_token)
                self.id_to_token.append(token)

    def encode(self, sentence: str, max_length: int) -> list[int]:
        tokens = normalize_text(sentence).split()[:max_length]
        ids = [self.token_to_id.get(token, self.token_to_id[self.UNK]) for token in tokens]
        padding = [self.token_to_id[self.PAD]] * (max_length - len(ids))
        return ids + padding

    @property
    def size(self) -> int:
        return len(self.id_to_token)


class QATransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_answers: int,
        pad_token_id: int,
        d_model: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_answers),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        padding_mask = token_ids.eq(self.pad_token_id)
        x = self.embedding(token_ids)

        valid_tokens = (~padding_mask).unsqueeze(-1)
        pooled = (x * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


@dataclass
class TrainingExample:
    question: str
    answer_id: int


def load_dataset(dataset_path: str | Path) -> tuple[list[TrainingExample], list[str]]:
    raw_items = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    examples: list[TrainingExample] = []
    answers: list[str] = []

    for answer_id, item in enumerate(raw_items):
        answers.append(item["answer"])
        for question in item["questions"]:
            examples.append(TrainingExample(question=question, answer_id=answer_id))

    return examples, answers
