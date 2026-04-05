from __future__ import annotations

from dataclasses import dataclass

from llm.dataio import load_instruction_jsonl
from llm.text import (
    content_similarity,
    extract_capital_subject,
    instruction_similarity,
    normalize_capital_subject_tokens,
    normalize_instruction_text,
    parse_capital_response,
)


@dataclass
class RetrievalExample:
    instruction: str
    response: str


def load_retrieval_examples(path: str) -> list[RetrievalExample]:
    return [RetrievalExample(instruction=item.instruction, response=item.response) for item in load_instruction_jsonl(path)]


def retrieve_response(question: str, examples: list[RetrievalExample]) -> str | None:
    normalized_question = normalize_instruction_text(question)
    capital_subject = extract_capital_subject(question)

    if capital_subject is not None:
        for example in examples:
            parsed = parse_capital_response(example.response)
            if parsed is None:
                continue
            normalized_country = normalize_capital_subject_tokens(normalize_instruction_text(parsed.country_phrase).split())
            normalized_capital = normalize_instruction_text(parsed.capital)
            if normalized_country == capital_subject:
                return example.response
            if normalized_capital == capital_subject:
                return f"{parsed.capital} est la capitale {parsed.country_phrase}."
        return None

    best_example: RetrievalExample | None = None
    best_score = 0.0
    best_precision = 0.0
    best_recall = 0.0
    for example in examples:
        normalized_instruction = normalize_instruction_text(example.instruction)
        if normalized_instruction == normalized_question:
            return example.response
        score = instruction_similarity(question, example.instruction)
        precision, recall = content_similarity(question, example.instruction)
        if score > best_score:
            best_score = score
            best_precision = precision
            best_recall = recall
            best_example = example
    if best_example is not None and (best_score >= 0.72 or (best_recall >= 0.9 and best_precision >= 0.5)):
        return best_example.response
    return None
