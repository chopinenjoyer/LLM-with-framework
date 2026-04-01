from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from model import QATransformer, Vocabulary, normalize_text


@dataclass
class Example:
    question: str
    answer_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question to the tiny QA model.")
    parser.add_argument("--model", default="artifacts/qa_model.pt")
    parser.add_argument("--question", help="Question asked to the model.")
    return parser.parse_args()


def load_model(model_path: str) -> tuple[QATransformer, dict[str, int], list[str], int, list[Example]]:
    checkpoint = torch.load(model_path, map_location="cpu")
    vocab = checkpoint["vocab"]
    answers = checkpoint["answers"]
    max_length = checkpoint["max_length"]
    examples = [Example(question=item["question"], answer_id=item["answer_id"]) for item in checkpoint.get("examples", [])]

    model = QATransformer(
        vocab_size=len(vocab),
        num_answers=len(answers),
        pad_token_id=vocab[Vocabulary.PAD],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab, answers, max_length, examples


def answer_question(
    model: QATransformer,
    vocab: dict[str, int],
    answers: list[str],
    max_length: int,
    examples: list[Example],
    question: str,
) -> str:
    normalized_question = normalize_text(question)

    for example in examples:
        if normalize_text(example.question) == normalized_question:
            return answers[example.answer_id]

    question_tokens = set(normalized_question.split())
    best_example: Example | None = None
    best_score = 0.0
    for example in examples:
        example_tokens = set(normalize_text(example.question).split())
        if not question_tokens or not example_tokens:
            continue
        score = len(question_tokens & example_tokens) / len(question_tokens | example_tokens)
        if score > best_score:
            best_score = score
            best_example = example

    if best_example is not None and best_score >= 0.6:
        return answers[best_example.answer_id]

    vocabulary = Vocabulary()
    vocabulary.token_to_id = vocab
    vocabulary.id_to_token = [""] * len(vocab)
    for token, idx in vocab.items():
        vocabulary.id_to_token[idx] = token

    encoded = vocabulary.encode(question, max_length)
    inputs = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        logits = model(inputs)
        answer_id = logits.argmax(dim=1).item()
    return answers[answer_id]


def main() -> None:
    args = parse_args()
    model, vocab, answers, max_length, examples = load_model(args.model)

    if args.question:
        print(answer_question(model, vocab, answers, max_length, examples, args.question))
        return

    print("Tape une question en francais. Ctrl+C pour quitter.")
    while True:
        question = input("> ").strip()
        if not question:
            continue
        print(answer_question(model, vocab, answers, max_length, examples, question))


if __name__ == "__main__":
    main()
