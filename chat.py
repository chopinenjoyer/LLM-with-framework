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
) -> str:
    prompt = prompt_from_instruction(question)
    input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True, add_eos=False)], dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None if temperature <= 0 else 50,
            eos_token_id=tokenizer.spec.eos_token_id,
        )
    response_ids = generated[0].tolist()[input_ids.size(1) :]
    response = tokenizer.decode(response_ids).strip()
    return response or "Je ne sais pas."


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model(args.checkpoint)

    if args.question:
        print(answer_question(model, tokenizer, args.question, args.max_new_tokens, args.temperature))
        return

    print("Tape une instruction en francais. Ctrl+C pour quitter.")
    try:
        while True:
            question = input("> ").strip()
            if not question:
                continue
            print(answer_question(model, tokenizer, question, args.max_new_tokens, args.temperature))
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
