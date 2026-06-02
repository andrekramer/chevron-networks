from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelOutput:
    logits: torch.Tensor
    aux: dict[str, torch.Tensor]


class MLPBaseline(nn.Module):
    def __init__(self, context_length: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_length, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        return ModelOutput(self.net(x.float()), {})


class TinyTransformer(nn.Module):
    def __init__(
        self,
        context_length: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token = nn.Embedding(2, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, context_length, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, 2)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        tokens = x.long()
        h = self.token(tokens) + self.pos[:, : tokens.size(1)]
        mask = torch.triu(
            torch.full((tokens.size(1), tokens.size(1)), float("-inf"), device=x.device),
            diagonal=1,
        )
        h = self.encoder(h, mask=mask)
        return ModelOutput(self.head(h[:, -1]), {})


class MinimalCPA(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        rho: float,
        detach_a_to_n: bool = True,
        use_diff_to_n: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rho = rho
        self.detach_a_to_n = detach_a_to_n
        self.use_diff_to_n = use_diff_to_n

        self.bit_embed = nn.Embedding(2, hidden_dim)
        self.w_aa = nn.Linear(hidden_dim, hidden_dim)
        self.w_an = nn.Linear(hidden_dim, hidden_dim)
        self.w_xa = nn.Linear(hidden_dim, hidden_dim)
        self.w_na = nn.Linear(hidden_dim, hidden_dim)
        self.w_nn = nn.Linear(hidden_dim, hidden_dim)
        self.w_xn = nn.Linear(hidden_dim, hidden_dim)
        self.w_diffn = nn.Linear(hidden_dim, hidden_dim) if use_diff_to_n else None
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_n = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(batch_size, self.hidden_dim, device=device)
        return z, z.clone()

    def logits_from_state(self, A: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
        features = torch.cat([A, N, A - N, A * N], dim=-1)
        return self.head(features)

    def step(
        self,
        token: torch.Tensor,
        A: torch.Tensor,
        N: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        x_embed = self.bit_embed(token.long())
        n_source = A.detach() if self.detach_a_to_n else A
        A_prev = A
        N_prev = N
        A_update = self.w_aa(A) + self.w_an(N) + self.w_xa(x_embed)
        N_update = self.w_na(n_source) + self.w_nn(N) + self.w_xn(x_embed)
        if self.w_diffn is not None:
            N_update = N_update + self.w_diffn(A - N)
        A = self.norm_a(A + F.gelu(A_update))
        N_candidate = self.norm_n(N + F.gelu(N_update))
        N = (1.0 - self.rho) * N + self.rho * N_candidate
        aux = {
            "an_dist": (A - N).pow(2).mean(dim=-1),
            "an_cos": F.cosine_similarity(A, N, dim=-1, eps=1e-8),
            "a_move": (A - A_prev).pow(2).mean(dim=-1),
            "n_move": (N - N_prev).pow(2).mean(dim=-1),
        }
        return A, N, aux

    def forward(self, x: torch.Tensor) -> ModelOutput:
        tokens = x.long()
        A, N = self.initial_state(tokens.size(0), tokens.device)
        traces: dict[str, list[torch.Tensor]] = {
            "an_dist": [],
            "an_cos": [],
            "a_move": [],
            "n_move": [],
        }
        for t in range(tokens.size(1)):
            A, N, aux = self.step(tokens[:, t], A, N)
            for key, value in aux.items():
                traces[key].append(value)
        reduced = {key: torch.stack(values, dim=1).mean() for key, values in traces.items()}
        reduced["final_an_dist"] = (A - N).pow(2).mean()
        reduced["final_an_cos"] = F.cosine_similarity(A, N, dim=-1, eps=1e-8).mean()
        return ModelOutput(self.logits_from_state(A, N), reduced)


def build_model(args) -> nn.Module:
    if args.model == "mlp":
        return MLPBaseline(args.context_length, args.hidden_dim)
    if args.model == "transformer":
        return TinyTransformer(
            args.context_length,
            args.hidden_dim,
            args.num_layers,
            args.num_heads,
            args.dropout,
        )
    if args.model == "cpa":
        return MinimalCPA(
            args.hidden_dim,
            args.rho,
            detach_a_to_n=args.detach_a_to_n,
            use_diff_to_n=args.use_diff_to_n,
        )
    raise ValueError(f"unknown model: {args.model}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cpa_parameter_groups(model: nn.Module, lr: float, n_lr_mult: float, weight_decay: float):
    n_names = ("w_na", "w_nn", "w_xn", "w_diffn", "norm_n")
    n_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(prefix) for prefix in n_names):
            n_params.append(param)
        else:
            other_params.append(param)
    return [
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
        {"params": n_params, "lr": lr * n_lr_mult, "weight_decay": weight_decay},
    ]
