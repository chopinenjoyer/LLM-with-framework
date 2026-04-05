from __future__ import annotations

import argparse

from llm.checkpoints import load_checkpoint
from llm.inference import INFERENCE_MODES, answer_question
from llm.retrieval import load_retrieval_examples


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
    parser.add_argument("--mode", choices=INFERENCE_MODES, default="hybrid")
    return parser.parse_args()


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
                args.mode,
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
                    args.mode,
                )
            )
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
