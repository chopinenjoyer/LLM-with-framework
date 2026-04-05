from __future__ import annotations

import argparse

from llm.checkpoints import load_checkpoint
from llm.generation import generate_response
from llm.retrieval import load_retrieval_examples, retrieve_response
from llm.text import extract_capital_subject, normalize_instruction_text


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

def answer_question(
    model,
    tokenizer,
    retrieval_examples,
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
    if len(normalize_instruction_text(question).split()) >= 3:
        return "Je ne sais pas."
    response = generate_response(model, tokenizer, question, max_new_tokens, temperature, top_k, top_p, repetition_penalty)
    return response or "Je ne sais pas."


def main() -> None:
    args = parse_args()
    model, tokenizer, _ = load_checkpoint(args.checkpoint)
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
