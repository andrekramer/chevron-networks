from __future__ import annotations

import torch
from torch import nn

from chevron_agent.models.chevron_core import ChevronCore
from chevron_agent.models.encoders import VisualEncoder


class ChevronAgent(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        leak: float,
        epsilon: float,
        use_tension_gate: bool,
        image_size: int = 9,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = VisualEncoder(hidden_dim=hidden_dim, image_size=image_size, in_channels=in_channels)
        self.core = ChevronCore(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            leak=leak,
            epsilon=epsilon,
            use_tension_gate=use_tension_gate,
        )
        d = hidden_dim // 2
        self.policy_head = nn.Linear(d, action_dim)
        self.value_head = nn.Linear(d, 1)

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.initial_state(batch_size, device)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
        done_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z = self.encoder(obs)
        core_out = self.core(z, hidden, done_mask)
        logits = self.policy_head(core_out["A"])
        value = self.value_head(core_out["N"]).squeeze(-1)
        return {
            "logits": logits,
            "value": value,
            "hidden": core_out["state"],
            "A": core_out["A"],
            "N": core_out["N"],
            "tension": core_out["tension"],
        }
