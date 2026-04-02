from __future__ import annotations

import torch

from llm.prompts import prompt_from_instruction
from llm.text import is_degenerate_text
from llm.models import DecoderOnlyLM
from tokenizer import ByteTokenizer


def generate_response(
    model: DecoderOnlyLM,
    tokenizer: ByteTokenizer,
    instruction: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> str:
    prompt = prompt_from_instruction(instruction)
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
    if not response or is_degenerate_text(response):
        return ""
    return response
