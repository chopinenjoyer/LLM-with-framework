from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LLMConfig:
    vocab_size: int = 259
    block_size: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, emb_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(emb_dim, dim=2)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(~self.mask[:, :, :seq_len, :seq_len], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x


class DecoderOnlyLM(nn.Module):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError("Sequence length exceeds block_size")

        positions = torch.arange(0, seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            flat_labels = shift_labels.reshape(-1)
            if bool(flat_labels.ne(-100).any()):
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    flat_labels,
                    ignore_index=-100,
                )
            else:
                loss = logits.new_zeros(())
        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        repetition_window: int = 128,
        max_consecutive_repeats: int = 24,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        repeat_counts = torch.ones((input_ids.size(0), 1), dtype=torch.long, device=input_ids.device)
        for _ in range(max_new_tokens):
            idx = input_ids[:, -self.config.block_size :]
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            if repetition_penalty > 1.0:
                window_tokens = idx[:, -min(idx.size(1), repetition_window) :]
                for batch_idx in range(window_tokens.size(0)):
                    seen_tokens = torch.unique(window_tokens[batch_idx])
                    logits[batch_idx, seen_tokens] = logits[batch_idx, seen_tokens] / repetition_penalty
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k < logits.size(-1):
                    values, _ = torch.topk(logits, top_k)
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_mask = cumulative_probs > top_p
                    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                    sorted_mask[:, 0] = False
                    removal_mask = torch.zeros_like(logits, dtype=torch.bool)
                    removal_mask.scatter_(1, sorted_indices, sorted_mask)
                    logits = logits.masked_fill(removal_mask, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            same_as_previous = next_token.eq(input_ids[:, -1:])
            repeat_counts = torch.where(same_as_previous, repeat_counts + 1, torch.ones_like(repeat_counts))
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
            if bool((repeat_counts >= max_consecutive_repeats).all()):
                break
        return input_ids
