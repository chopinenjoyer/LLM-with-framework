from __future__ import annotations

import math

import torch


class ManualAdamW:
    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        self.params = [param for param in params if param.requires_grad]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self.exp_avg = [torch.zeros_like(param) for param in self.params]
        self.exp_avg_sq = [torch.zeros_like(param) for param in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1 ** self.step_count
        bias_correction2 = 1.0 - self.beta2 ** self.step_count
        with torch.no_grad():
            for param, exp_avg, exp_avg_sq in zip(self.params, self.exp_avg, self.exp_avg_sq):
                if param.grad is None:
                    continue
                grad = param.grad
                exp_avg.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(self.eps)
                step_size = self.lr / bias_correction1
                if self.weight_decay and param.ndim >= 2:
                    param.mul_(1.0 - self.lr * self.weight_decay)
                param.addcdiv_(exp_avg, denom, value=-step_size)
