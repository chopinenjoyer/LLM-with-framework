from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from data_utils import (
    extract_capital_subject,
    instruction_similarity,
    load_instruction_jsonl,
    normalize_capital_subject_tokens,
    normalize_instruction_text,
    parse_capital_response,
    prompt_from_instruction,
)
from model import DecoderOnlyLM, LLMConfig
from tokenizer import ByteTokenizer


@dataclass
class RetrievalExample:
    instruction: str
    response: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a decoder-only language model.")
    parser.add_argument("--checkpoint", default="artifacts/sft_model.pt")
    parser.add_argument("--question")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--retrieval-data", default="data/instructions_train.jsonl")
    return parser.parse_args()


def load_model(checkpoint_path: str) -> tuple[DecoderOnlyLM, ByteTokenizer]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    tokenizer = ByteTokenizer.load(checkpoint["tokenizer_path"])
    model = DecoderOnlyLM(LLMConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer


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
    for example in examples:
        normalized_instruction = normalize_instruction_text(example.instruction)
        if normalized_instruction == normalized_question:
            return example.response
        score = instruction_similarity(question, example.instruction)
        if score > best_score:
            best_score = score
            best_example = example
    if best_example is not None and best_score >= 0.72:
        return best_example.response
    return None


def answer_question(
    model: DecoderOnlyLM,
    tokenizer: ByteTokenizer,
    retrieval_examples: list[RetrievalExample],
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> str:
    retrieved = retrieve_response(question, retrieval_examples)
    if retrieved is not None:
        return retrieved
    if extract_capital_subject(question) is not None:
        return "Je ne sais pas."
    prompt = prompt_from_instruction(question)
    input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True, add_eos=False)], dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None if temperature <= 0 else top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.spec.eos_token_id,
        )
    response_ids = generated[0].tolist()[input_ids.size(1) :]
    response = tokenizer.decode(response_ids).strip()
    if not response:
        return "Je ne sais pas."
    compact = response.replace(" ", "")
    unique_chars = len(set(compact))
    if compact and len(compact) >= 12 and unique_chars <= 2:
        return "Je ne sais pas."
    return response


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model(args.checkpoint)
    retrieval_examples = load_retrieval_examples(args.retrieval_data)

    if args.question:
        print(
            answer_question(
                model,
                tokenizer,
                retrieval_examples,
                args.question,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                args.top_p,
                args.repetition_penalty,
            )
        )
        return

    print("Tape une instruction en francais. Ctrl+C pour quitter.")
    try:
        while True:
            question = input("> ").strip()
            if not question:
                continue
            print(
                answer_question(
                    model,
                    tokenizer,
                    retrieval_examples,
                    question,
                    args.max_new_tokens,
                    args.temperature,
                    args.top_k,
                    args.top_p,
                    args.repetition_penalty,
                )
            )
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
