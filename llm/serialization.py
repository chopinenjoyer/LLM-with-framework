from __future__ import annotations

import json
from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    model_config: dict[str, int | float],
    tokenizer_path: str,
    extra: dict[str, object] | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "tokenizer_path": tokenizer_path,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def save_json(path: str | Path, data: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
