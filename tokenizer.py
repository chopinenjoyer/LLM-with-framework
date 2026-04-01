from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenizerSpec:
    pad_token_id: int = 256
    bos_token_id: int = 257
    eos_token_id: int = 258
    vocab_size: int = 259


class ByteTokenizer:
    def __init__(self, spec: TokenizerSpec | None = None) -> None:
        self.spec = spec or TokenizerSpec()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.spec.bos_token_id)
        token_ids.extend(text.encode("utf-8"))
        if add_eos:
            token_ids.append(self.spec.eos_token_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        data = bytearray()
        for token_id in token_ids:
            if token_id in {
                self.spec.pad_token_id,
                self.spec.bos_token_id,
                self.spec.eos_token_id,
            }:
                continue
            if 0 <= token_id <= 255:
                data.append(token_id)
        return data.decode("utf-8", errors="ignore")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.spec.__dict__, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(spec=TokenizerSpec(**data))
