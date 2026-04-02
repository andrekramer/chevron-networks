from __future__ import annotations

import torch
from torch import nn

from chevron_agent.models.encoders import VisualEncoder


class BaselineGRUAgent(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, image_size: int = 9, in_channels: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = VisualEncoder(hidden_dim=hidden_dim, image_size=image_size, in_channels=in_channels)
        self.core = nn.GRUCell(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor, done_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(obs)
        hidden = hidden * done_mask
        next_hidden = self.core(z, hidden)
        logits = self.policy_head(next_hidden)
        value = self.value_head(next_hidden).squeeze(-1)
        return {
            "logits": logits,
            "value": value,
            "hidden": next_hidden,
        }
