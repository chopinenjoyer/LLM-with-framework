from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class LLMConfig:
    vocab_size: int = 259
    block_size: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
