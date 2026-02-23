import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import PairDataset, build_wordnet_splits
from models import BaselineMLP, ChevronMLP, GraphBaseline


@dataclass
class Metrics:
    loss: float
    match_acc: float
    polarity_acc: float
    swap_match_consistency: float
    swap_polarity_flip_rate: float


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _compute_loss(
    match_logit: torch.Tensor,
    polarity_logit: torch.Tensor,
    match: torch.Tensor,
    polarity: torch.Tensor,
    polarity_weight: float,
) -> torch.Tensor:
    match_loss = F.binary_cross_entropy_with_logits(match_logit, match)

    antonym_mask = match > 0.5
    if antonym_mask.any():
        p_loss = F.binary_cross_entropy_with_logits(polarity_logit[antonym_mask], polarity[antonym_mask])
    else:
        p_loss = torch.tensor(0.0, device=match_logit.device)

    return match_loss + polarity_weight * p_loss


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, polarity_weight: float) -> Metrics:
    model.eval()

    total_loss = 0.0
    total_count = 0

    correct_match = 0
    correct_polarity = 0
    polarity_count = 0

    swap_match_consistent = 0
    swap_total = 0
    swap_pol_flip = 0
    swap_pol_total = 0

    for batch in loader:
        batch = _to_device(batch, device)
        w1, w2 = batch["w1"], batch["w2"]
        match, polarity = batch["match"], batch["polarity"]

        match_logit, polarity_logit = model(w1, w2)
        loss = _compute_loss(match_logit, polarity_logit, match, polarity, polarity_weight)

        bsz = w1.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz

        match_pred = (torch.sigmoid(match_logit) > 0.5).float()
        correct_match += (match_pred == match).sum().item()

        antonym_mask = match > 0.5
        if antonym_mask.any():
            pol_pred = (torch.sigmoid(polarity_logit[antonym_mask]) > 0.5).float()
            pol_true = polarity[antonym_mask]
            correct_polarity += (pol_pred == pol_true).sum().item()
            polarity_count += pol_true.numel()

        # Swap consistency checks
        swap_match_logit, swap_pol_logit = model(w2, w1)
        swap_match_pred = (torch.sigmoid(swap_match_logit) > 0.5).float()

        swap_match_consistent += (swap_match_pred == match_pred).sum().item()
        swap_total += bsz

        if antonym_mask.any():
            pol_pred = (torch.sigmoid(polarity_logit[antonym_mask]) > 0.5).float()
            swap_pol_pred = (torch.sigmoid(swap_pol_logit[antonym_mask]) > 0.5).float()
            swap_pol_flip += (swap_pol_pred == (1.0 - pol_pred)).sum().item()
            swap_pol_total += pol_pred.numel()

    return Metrics(
        loss=total_loss / max(total_count, 1),
        match_acc=correct_match / max(total_count, 1),
        polarity_acc=correct_polarity / max(polarity_count, 1),
        swap_match_consistency=swap_match_consistent / max(swap_total, 1),
        swap_polarity_flip_rate=swap_pol_flip / max(swap_pol_total, 1),
    )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    polarity_weight: float,
) -> Metrics:
    model.train()

    total_loss = 0.0
    total_count = 0

    correct_match = 0
    correct_polarity = 0
    polarity_count = 0

    for batch in loader:
        batch = _to_device(batch, device)
        w1, w2 = batch["w1"], batch["w2"]
        match, polarity = batch["match"], batch["polarity"]

        optimizer.zero_grad(set_to_none=True)
        match_logit, polarity_logit = model(w1, w2)
        loss = _compute_loss(match_logit, polarity_logit, match, polarity, polarity_weight)
        loss.backward()
        optimizer.step()

        bsz = w1.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz

        match_pred = (torch.sigmoid(match_logit) > 0.5).float()
        correct_match += (match_pred == match).sum().item()

        antonym_mask = match > 0.5
        if antonym_mask.any():
            pol_pred = (torch.sigmoid(polarity_logit[antonym_mask]) > 0.5).float()
            pol_true = polarity[antonym_mask]
            correct_polarity += (pol_pred == pol_true).sum().item()
            polarity_count += pol_true.numel()

    return Metrics(
        loss=total_loss / max(total_count, 1),
        match_acc=correct_match / max(total_count, 1),
        polarity_acc=correct_polarity / max(polarity_count, 1),
        swap_match_consistency=0.0,
        swap_polarity_flip_rate=0.0,
    )


def build_model(args, vocab_size: int) -> torch.nn.Module:
    if args.model == "baseline":
        return BaselineMLP(vocab_size=vocab_size, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim)
    if args.model == "graph":
        return GraphBaseline(vocab_size=vocab_size, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim)

    return ChevronMLP(
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        hidden_groups=args.hidden_groups,
        variant=args.chevron_variant,
    )


def format_metrics(prefix: str, m: Metrics) -> str:
    return (
        f"{prefix} loss={m.loss:.4f} "
        f"match_acc={m.match_acc:.4f} "
        f"polarity_acc={m.polarity_acc:.4f} "
        f"swap_match_consistency={m.swap_match_consistency:.4f} "
        f"swap_polarity_flip={m.swap_polarity_flip_rate:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chevron Networks WordNet antonym experiment")
    parser.add_argument("--model", choices=["baseline", "graph", "chevron"], default="chevron")
    parser.add_argument("--chevron-variant", choices=["full", "diag_only", "offdiag_frozen"], default="full")
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-groups", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--negative-multiplier", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--polarity-weight", type=float, default=0.3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    split = build_wordnet_splits(
        seed=args.seed,
        negative_multiplier=args.negative_multiplier,
        max_pairs=args.max_pairs,
    )

    train_loader = DataLoader(PairDataset(split.train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(PairDataset(split.val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(PairDataset(split.test), batch_size=args.batch_size, shuffle=False)

    model = build_model(args, vocab_size=len(split.vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    best_state = None

    print(f"vocab={len(split.vocab)} train={len(split.train)} val={len(split.val)} test={len(split.test)}")
    print(f"model={args.model} variant={args.chevron_variant if args.model == 'chevron' else 'n/a'}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.polarity_weight)
        val_metrics = evaluate(model, val_loader, device, args.polarity_weight)

        print(format_metrics(f"epoch={epoch:02d} train", train_metrics))
        print(format_metrics(f"epoch={epoch:02d} val  ", val_metrics))

        if val_metrics.match_acc > best_val:
            best_val = val_metrics.match_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device, args.polarity_weight)
    print(format_metrics("test", test_metrics))


if __name__ == "__main__":
    main()
