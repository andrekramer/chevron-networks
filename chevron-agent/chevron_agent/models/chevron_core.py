from __future__ import annotations

import torch
from torch import nn


class ChevronCore(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, leak: float, epsilon: float, use_tension_gate: bool) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            msg = "hidden_dim must be even for ChevronCore"
            raise ValueError(msg)
        self.d = hidden_dim // 2
        self.leak = leak
        self.epsilon = epsilon
        self.use_tension_gate = use_tension_gate

        self.in_a = nn.Linear(input_dim, self.d)
        self.in_n = nn.Linear(input_dim, self.d)
        self.aa = nn.Linear(self.d, self.d, bias=False)
        self.an = nn.Linear(self.d, self.d, bias=False)
        self.na = nn.Linear(self.d, self.d, bias=False)
        self.nn = nn.Linear(self.d, self.d, bias=False)

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, self.d, device=device),
            torch.zeros(batch_size, self.d, device=device),
        )

    def forward(
        self,
        z: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        done_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        a, n = state
        a = a * done_mask
        n = n * done_mask

        m_a = self.aa(a) + self.an(n) + self.in_a(z)
        m_n = self.na(a) + self.nn(n) + self.in_n(z)

        if self.use_tension_gate:
            pi = torch.sigmoid(a - n)
            tension = torch.sqrt(self.epsilon + pi * (1.0 - pi))
        else:
            tension = torch.ones_like(a)

        a_next = torch.tanh((1.0 - self.leak) * a + tension * m_a)
        n_next = torch.tanh((1.0 - self.leak) * n + tension * m_n)

        return {
            "A": a_next,
            "N": n_next,
            "state": (a_next, n_next),
            "tension": tension if self.use_tension_gate else None,
        }
