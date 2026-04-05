from __future__ import annotations


def prompt_from_instruction(instruction: str) -> str:
    return f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"
