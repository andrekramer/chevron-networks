from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import RegimeSequenceDataset
from eval import evaluate, evaluate_cpa_stateful, print_metrics
from models import build_model, count_parameters, cpa_parameter_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chevron Predictive Architecture experiment")
    parser.add_argument("--model", choices=("mlp", "transformer", "cpa"), required=True)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--detach-a-to-n", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-diff-to-n", action="store_true")
    parser.add_argument("--lambda-band", type=float, default=0.01)
    parser.add_argument("--lambda-slow", type=float, default=0.001)
    parser.add_argument("--target-dist", type=float, default=1.0)
    parser.add_argument("--n-lr-mult", type=float, default=0.25)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--use-distractors", action="store_true")
    parser.add_argument("--distractor-prob", type=float, default=0.002)
    parser.add_argument("--min-distractor", type=int, default=5)
    parser.add_argument("--max-distractor", type=int, default=10)
    parser.add_argument("--regime-set", choices=("easy", "lagged"), default="easy")
    parser.add_argument("--train-sequences", type=int, default=64)
    parser.add_argument("--test-sequences", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1200)
    parser.add_argument("--min-regime", type=int, default=50)
    parser.add_argument("--max-regime", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--eval-noise", type=float, nargs="*", default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--stateful-eval", action="store_true")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_dataset(args: argparse.Namespace, *, train: bool, noise: float | None = None):
    return RegimeSequenceDataset(
        num_sequences=args.train_sequences if train else args.test_sequences,
        sequence_length=args.sequence_length,
        context_length=args.context_length,
        p_noise=args.noise if noise is None else noise,
        use_distractors=args.use_distractors,
        distractor_prob=args.distractor_prob,
        min_regime=args.min_regime,
        max_regime=args.max_regime,
        min_distractor=args.min_distractor,
        max_distractor=args.max_distractor,
        regime_set=args.regime_set,
        seed=args.seed + (0 if train else 1_000_000),
    )


def loss_for_batch(args: argparse.Namespace, model, batch, device: torch.device):
    x = batch["context"].to(device)
    y = batch["target"].to(device)
    output = model(x)
    loss_pred = F.cross_entropy(output.logits, y)
    losses = {"loss_pred": loss_pred}
    loss = loss_pred
    if args.model == "cpa":
        loss_band = (output.aux["final_an_dist"] - args.target_dist).pow(2)
        loss_slow = output.aux["n_move"]
        loss = loss + args.lambda_band * loss_band + args.lambda_slow * loss_slow
        losses.update(
            {
                "loss_band": loss_band,
                "loss_slow": loss_slow,
                "an_dist": output.aux["final_an_dist"],
                "an_cos": output.aux["final_an_cos"],
                "a_move": output.aux["a_move"],
                "n_move": output.aux["n_move"],
            }
        )
    losses["loss"] = loss
    return loss, losses


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    run_dir = args.out_dir / f"{args.model}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = make_dataset(args, train=True)
    test_ds = make_dataset(args, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = build_model(args).to(device)
    if args.model == "cpa":
        param_groups = cpa_parameter_groups(model, args.lr, args.n_lr_mult, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"device: {device}")
    print(f"parameters: {count_parameters(model)}")
    history_path = run_dir / "history.csv"
    fieldnames = [
        "epoch",
        "loss",
        "loss_pred",
        "loss_band",
        "loss_slow",
        "an_dist",
        "an_cos",
        "a_move",
        "n_move",
        "test_accuracy",
        "post_switch_recovery",
        "distractor_recovery",
    ]
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            totals: dict[str, float] = {}
            batches = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss, losses = loss_for_batch(args, model, batch, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
                batches += 1

            eval_result = evaluate(model, test_ds, batch_size=args.batch_size, device=device)
            row = {key: "" for key in fieldnames}
            row["epoch"] = epoch
            for key, value in totals.items():
                row[key] = value / batches
            row["test_accuracy"] = eval_result.metrics["accuracy"]
            row["post_switch_recovery"] = eval_result.metrics["post_switch_recovery"]
            row["distractor_recovery"] = eval_result.metrics["distractor_recovery"]
            writer.writerow(row)
            f.flush()
            print(
                f"epoch {epoch}: loss={row['loss']:.4f} "
                f"test_acc={row['test_accuracy']:.4f} "
                f"switch_recovery={row['post_switch_recovery']}"
            )

    final_metrics = evaluate(model, test_ds, batch_size=args.batch_size, device=device).metrics
    stateful_metrics = None
    if args.model == "cpa" and args.stateful_eval:
        stateful_metrics = evaluate_cpa_stateful(model, test_ds, device=device).metrics
    noise_metrics = {}
    for noise in args.eval_noise:
        ds = make_dataset(args, train=False, noise=noise)
        noise_metrics[str(noise)] = evaluate(model, ds, batch_size=args.batch_size, device=device).metrics

    torch.save({"model": model.state_dict(), "args": vars(args)}, run_dir / "model.pt")
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(
            {
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "parameters": count_parameters(model),
                "final": final_metrics,
                "stateful": stateful_metrics,
                "noise_sweep": noise_metrics,
            },
            f,
            indent=2,
        )
    print("final metrics:")
    print_metrics(final_metrics)
    if stateful_metrics is not None:
        print("stateful CPA metrics:")
        print_metrics(stateful_metrics)
    print(f"wrote: {run_dir}")


if __name__ == "__main__":
    main()
