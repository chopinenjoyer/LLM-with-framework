from __future__ import annotations

import torch

from llm.models import DecoderOnlyLM, LLMConfig
from tokenizer import ByteTokenizer


def load_checkpoint(checkpoint_path: str) -> tuple[DecoderOnlyLM, ByteTokenizer, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    tokenizer = ByteTokenizer.load(checkpoint["tokenizer_path"])
    model = DecoderOnlyLM(LLMConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, checkpoint
