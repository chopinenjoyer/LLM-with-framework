from __future__ import annotations

import argparse

import torch

from data_utils import prompt_from_instruction
from model import DecoderOnlyLM, LLMConfig
from tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a decoder-only language model.")
    parser.add_argument("--checkpoint", default="artifacts/sft_model.pt")
    parser.add_argument("--question")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    return parser.parse_args()


def load_model(checkpoint_path: str) -> tuple[DecoderOnlyLM, ByteTokenizer]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    tokenizer = ByteTokenizer.load(checkpoint["tokenizer_path"])
    model = DecoderOnlyLM(LLMConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer


def answer_question(
    model: DecoderOnlyLM,
    tokenizer: ByteTokenizer,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> str:
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

    if args.question:
        print(
            answer_question(
                model,
                tokenizer,
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
