from __future__ import annotations

from llm.generation import generate_response
from llm.models import DecoderOnlyLM
from llm.retrieval import RetrievalExample, retrieve_response
from tokenizer import ByteTokenizer


INFERENCE_MODES = ("model", "retrieval", "hybrid")


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
    mode: str = "hybrid",
    fallback_response: str = "Je ne sais pas.",
) -> str:
    if mode not in INFERENCE_MODES:
        raise ValueError(f"Unsupported inference mode: {mode}")

    if mode in {"retrieval", "hybrid"}:
        retrieved = retrieve_response(question, retrieval_examples)
        if retrieved is not None:
            return retrieved
        if mode == "retrieval":
            return fallback_response

    response = generate_response(
        model,
        tokenizer,
        question,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
    )
    return response or fallback_response
