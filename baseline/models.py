from typing import Literal

import torch
import torch.nn as nn


ChevronVariant = Literal["full", "diag_only", "offdiag_frozen"]


class ChevronLinear(nn.Module):
    def __init__(self, in_groups: int, out_groups: int, bias: bool = True, variant: ChevronVariant = "full"):
        super().__init__()
        self.in_groups = in_groups
        self.out_groups = out_groups
        self.variant = variant

        self.weight = nn.Parameter(torch.empty(out_groups, in_groups, 2, 2))
        nn.init.xavier_uniform_(self.weight.view(out_groups * 2, in_groups * 2))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_groups, 2))
        else:
            self.register_parameter("bias", None)

        if variant == "offdiag_frozen":
            mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
            self.register_buffer("offdiag_mask", mask)
            with torch.no_grad():
                self.weight[:, :, 0, 1].zero_()
                self.weight[:, :, 1, 0].zero_()
        else:
            self.register_buffer("offdiag_mask", torch.zeros(2, 2))

    def _effective_weight(self) -> torch.Tensor:
        if self.variant == "full":
            return self.weight
        if self.variant == "diag_only":
            w = self.weight.clone()
            w[:, :, 0, 1] = 0.0
            w[:, :, 1, 0] = 0.0
            return w
        if self.variant == "offdiag_frozen":
            # Keep off-diagonal fixed at zero while diagonals remain learnable.
            return self.weight * (1.0 - self.offdiag_mask)
        raise ValueError(f"Unknown variant: {self.variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_groups, 2], weight: [out_groups, in_groups, 2, 2]
        w = self._effective_weight()
        out = torch.einsum("bij,oikj->bok", x, w)
        if self.bias is not None:
            out = out + self.bias
        return out


class ChevronMLP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        hidden_groups: int = 64,
        variant: ChevronVariant = "full",
    ):
        super().__init__()
        if emb_dim % 2 != 0:
            raise ValueError("emb_dim must be even to split into 2-channel groups")

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        in_groups = emb_dim

        self.fc1 = ChevronLinear(in_groups=in_groups, out_groups=hidden_groups, variant=variant)
        self.fc2 = ChevronLinear(in_groups=hidden_groups, out_groups=hidden_groups, variant=variant)

        self.act = nn.GELU()
        self.match_head = nn.Linear(hidden_groups * 2, 1)
        self.polarity_head = nn.Linear(hidden_groups * 2, 1)

    def forward(self, w1: torch.Tensor, w2: torch.Tensor):
        e1 = self.embedding(w1)
        e2 = self.embedding(w2)

        # Pair words into 2-channel states per feature dimension: [e1_i, e2_i].
        x = torch.stack([e1, e2], dim=-1)
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))

        flat = h.reshape(h.size(0), -1)
        match_logit = self.match_head(flat).squeeze(-1)
        polarity_logit = self.polarity_head(flat).squeeze(-1)
        return match_logit, polarity_logit


class BaselineMLP(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.match_head = nn.Linear(hidden_dim, 1)
        self.polarity_head = nn.Linear(hidden_dim, 1)

    def forward(self, w1: torch.Tensor, w2: torch.Tensor):
        e1 = self.embedding(w1)
        e2 = self.embedding(w2)
        x = torch.cat([e1, e2], dim=-1)
        h = self.net(x)
        match_logit = self.match_head(h).squeeze(-1)
        polarity_logit = self.polarity_head(h).squeeze(-1)
        return match_logit, polarity_logit


class GraphBaseline(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # Two rounds of message passing on a 2-node graph: each node mixes self and neighbor.
        self.self_1 = nn.Linear(emb_dim, hidden_dim)
        self.msg_1 = nn.Linear(emb_dim, hidden_dim)
        self.self_2 = nn.Linear(hidden_dim, hidden_dim)
        self.msg_2 = nn.Linear(hidden_dim, hidden_dim)

        self.act = nn.GELU()
        self.match_head = nn.Linear(hidden_dim * 4, 1)
        self.polarity_head = nn.Linear(hidden_dim * 4, 1)

    def _mp_step(
        self,
        h_left: torch.Tensor,
        h_right: torch.Tensor,
        self_proj: nn.Linear,
        msg_proj: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_next = self.act(self_proj(h_left) + msg_proj(h_right))
        right_next = self.act(self_proj(h_right) + msg_proj(h_left))
        return left_next, right_next

    def forward(self, w1: torch.Tensor, w2: torch.Tensor):
        h1 = self.embedding(w1)
        h2 = self.embedding(w2)

        h1, h2 = self._mp_step(h1, h2, self.self_1, self.msg_1)
        h1, h2 = self._mp_step(h1, h2, self.self_2, self.msg_2)

        # Graph-style pair readout: node states plus symmetric interactions.
        pair = torch.cat([h1, h2, torch.abs(h1 - h2), h1 * h2], dim=-1)
        match_logit = self.match_head(pair).squeeze(-1)
        polarity_logit = self.polarity_head(pair).squeeze(-1)
        return match_logit, polarity_logit
