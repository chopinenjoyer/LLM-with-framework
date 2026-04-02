from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class InstructionExample:
    instruction: str
    response: str


def load_text_files(paths: list[str]) -> list[str]:
    texts: list[str] = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_dir():
            for child in sorted(path.rglob("*.txt")):
                texts.append(child.read_text(encoding="utf-8"))
        else:
            texts.append(path.read_text(encoding="utf-8"))
    return texts


def load_shard_manifest(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_instruction_jsonl(path: str | Path) -> list[InstructionExample]:
    examples: list[InstructionExample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        examples.append(InstructionExample(instruction=item["instruction"], response=item["response"]))
    return examples
