from __future__ import annotations

import argparse
from pathlib import Path

from llm.checkpoints import load_checkpoint
from llm.evaluation import evaluate_exact_match, evaluate_perplexity, evaluate_sft_loss
from llm.retrieval import load_retrieval_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretraining and fine-tuning checkpoints.")
    parser.add_argument("--data-dir", default="artifacts/datasets")
    parser.add_argument("--checkpoint", default="artifacts/sft_model.pt")
    parser.add_argument("--eval-data", default="data/instructions_eval.jsonl")
    parser.add_argument("--retrieval-data", default="data/instructions_train.jsonl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    model, tokenizer, _ = load_checkpoint(args.checkpoint)
    retrieval_examples = load_retrieval_examples(args.retrieval_data)
    ppl = evaluate_perplexity(model, data_dir, args.batch_size)
    sft_loss = evaluate_sft_loss(model, tokenizer, data_dir, args.batch_size)
    model_exact_match = evaluate_exact_match(
        model,
        tokenizer,
        retrieval_examples,
        Path(args.eval_data),
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.repetition_penalty,
        mode="model",
    )
    retrieval_exact_match = evaluate_exact_match(
        model,
        tokenizer,
        retrieval_examples,
        Path(args.eval_data),
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.repetition_penalty,
        mode="retrieval",
    )
    hybrid_exact_match = evaluate_exact_match(
        model,
        tokenizer,
        retrieval_examples,
        Path(args.eval_data),
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.repetition_penalty,
        mode="hybrid",
    )
    print(f"pretrain_perplexity={ppl:.4f}")
    print(f"sft_val_loss={sft_loss:.4f}")
    print(f"eval_model_exact_match={model_exact_match:.2%}")
    print(f"eval_retrieval_exact_match={retrieval_exact_match:.2%}")
    print(f"eval_hybrid_exact_match={hybrid_exact_match:.2%}")


if __name__ == "__main__":
    main()
