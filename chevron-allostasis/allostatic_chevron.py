import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from torchvision import datasets, transforms


TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


class ChevronLinear(nn.Module):
    """Linear layer over paired A/N states shaped [batch, width, 2]."""

    def __init__(self, in_width: int, out_width: int, diagonal_only: bool = False):
        super().__init__()
        self.in_width = in_width
        self.out_width = out_width
        self.diagonal_only = diagonal_only
        self.weight = nn.Parameter(torch.empty(out_width, in_width, 2, 2))
        self.bias = nn.Parameter(torch.zeros(out_width, 2))
        nn.init.xavier_uniform_(self.weight.view(out_width * 2, in_width * 2))

    def effective_weight(self) -> torch.Tensor:
        if not self.diagonal_only:
            return self.weight
        weight = self.weight.clone()
        weight[:, :, 0, 1] = 0.0
        weight[:, :, 1, 0] = 0.0
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # weight indices are [out_width, in_width, input_channel, output_channel].
        return torch.einsum("bic,oicd->bod", x, self.effective_weight()) + self.bias


class AllostaticChevronNet(nn.Module):
    def __init__(
        self,
        width: int = 128,
        num_classes: int = 2,
        diagonal_only: bool = False,
        readout: str = "both",
    ):
        super().__init__()
        if readout not in {"both", "a_only"}:
            raise ValueError("readout must be 'both' or 'a_only'")

        self.width = width
        self.readout = readout
        self.input = nn.Linear(28 * 28, width * 2)
        self.chev1 = ChevronLinear(width, width, diagonal_only=diagonal_only)
        self.chev2 = ChevronLinear(width, width, diagonal_only=diagonal_only)
        self.norm1 = nn.LayerNorm((width, 2))
        self.norm2 = nn.LayerNorm((width, 2))
        self.head = nn.Linear(width * (2 if readout == "both" else 1), num_classes)

    def forward(self, x: torch.Tensor, return_stats: bool = False):
        x = x.flatten(1)
        h = self.input(x).view(x.size(0), self.width, 2)

        h = self.norm1(F.gelu(self.chev1(h)))
        d1_samples = channel_disagreement_per_sample(h)
        h = self.norm2(F.gelu(self.chev2(h)))
        d2_samples = channel_disagreement_per_sample(h)

        if self.readout == "a_only":
            readout = h[:, :, 0]
        else:
            readout = h.reshape(h.size(0), -1)
        logits = self.head(readout)

        if not return_stats:
            return logits
        return logits, {
            "disagree_1": d1_samples.mean(),
            "disagree_2": d2_samples.mean(),
            "priority_disagreement": 0.5 * (d1_samples + d2_samples),
        }

    def chevron_layers(self) -> list[ChevronLinear]:
        return [self.chev1, self.chev2]


class ScalarMLP(nn.Module):
    def __init__(self, width: int = 128, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.head = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor, return_stats: bool = False):
        h = self.net(x)
        logits = self.head(h)
        if not return_stats:
            return logits
        zero = logits.new_tensor(0.0)
        return logits, {
            "disagree_1": zero,
            "disagree_2": zero,
            "priority_disagreement": logits.new_zeros(logits.size(0)),
        }

    def chevron_layers(self) -> list[ChevronLinear]:
        return []


class RelabeledTaskDataset(Dataset):
    def __init__(self, base: Dataset, digits: tuple[int, int], max_examples: int = 0):
        self.base = base
        self.digits = digits
        self.indices = [
            idx for idx, (_, label) in enumerate(base) if int(label) in digits
        ]
        if max_examples > 0:
            self.indices = self.indices[:max_examples]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, label = self.base[self.indices[idx]]
        target = 0 if int(label) == self.digits[0] else 1
        return image, target


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    disagree_1: float
    disagree_2: float


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.images: list[torch.Tensor] = []
        self.labels: list[int] = []

    def __len__(self) -> int:
        return len(self.labels)

    def add_balanced(self, dataset: Dataset) -> None:
        if self.capacity <= 0:
            return

        per_task = max(1, self.capacity // len(TASKS))
        by_label: dict[int, list[int]] = {0: [], 1: []}
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            by_label[int(label)].append(idx)

        chosen: list[int] = []
        per_label = max(1, per_task // 2)
        for label_indices in by_label.values():
            self.rng.shuffle(label_indices)
            chosen.extend(label_indices[:per_label])

        for idx in chosen:
            image, label = dataset[idx]
            self.images.append(image.cpu())
            self.labels.append(int(label))

        overflow = max(0, len(self.labels) - self.capacity)
        if overflow:
            self.images = self.images[overflow:]
            self.labels = self.labels[overflow:]

    def tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.labels:
            return None
        images = torch.stack(self.images).to(device)
        labels = torch.tensor(self.labels, dtype=torch.long, device=device)
        return images, labels

    def loader(
        self,
        batch_size: int,
        device: torch.device,
        priorities: torch.Tensor | None = None,
    ) -> DataLoader | None:
        tensors = self.tensors(device)
        if tensors is None:
            return None
        images, labels = tensors
        dataset = TensorDataset(images, labels)
        if priorities is None:
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)

        weights = priorities.detach().float().cpu().clamp_min(1e-8)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def channel_disagreement(h: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(h[:, :, 0] - h[:, :, 1]))


def channel_disagreement_per_sample(h: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(h[:, :, 0] - h[:, :, 1]), dim=1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tasks(data_dir: Path, max_train: int, max_test: int):
    transform = transforms.Compose([transforms.ToTensor()])
    train_base = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    test_base = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)
    train_tasks = [RelabeledTaskDataset(train_base, task, max_train) for task in TASKS]
    test_tasks = [RelabeledTaskDataset(test_base, task, max_test) for task in TASKS]
    return train_tasks, test_tasks


def build_optimizer(
    model: nn.Module,
    phase: str,
    lr: float,
    slow_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if phase not in {"wake", "consolidate"}:
        raise ValueError("phase must be wake or consolidate")

    if not hasattr(model, "chevron_layers") or not model.chevron_layers():
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    non_chevron_params = list(model.input.parameters()) + list(model.head.parameters())
    non_chevron_params += list(model.norm1.parameters()) + list(model.norm2.parameters())
    chevron_params = []
    for layer in model.chevron_layers():
        chevron_params.extend([layer.weight, layer.bias])

    param_groups = [
        {"params": non_chevron_params, "lr": lr, "weight_decay": weight_decay},
        {"params": chevron_params, "lr": lr, "weight_decay": 0.0},
    ]
    return MaskedAdamW(param_groups, model, phase, lr, slow_lr)


class MaskedAdamW(torch.optim.AdamW):
    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict],
        model: AllostaticChevronNet,
        phase: str,
        lr: float,
        slow_lr: float,
    ):
        super().__init__(params, lr=lr)
        self.model = model
        self.phase = phase
        self.lr = lr
        self.slow_lr = slow_lr

    @torch.no_grad()
    def step(self, closure=None):
        self._scale_chevron_grads()
        return super().step(closure)

    def _scale_chevron_grads(self) -> None:
        for layer in self.model.chevron_layers():
            if layer.weight.grad is not None:
                scale = torch.ones_like(layer.weight.grad)
                if self.phase == "wake":
                    # Indices are [input_channel, output_channel].
                    scale[:, :, 0, 0] = 1.0
                    scale[:, :, 1, 0] = 1.0
                    scale[:, :, 0, 1] = self.slow_lr / self.lr if self.lr else 0.0
                    scale[:, :, 1, 1] = 0.0
                else:
                    fast = self.slow_lr / self.lr if self.lr else 0.0
                    scale[:, :, 0, 0] = fast
                    scale[:, :, 1, 0] = fast
                    scale[:, :, 0, 1] = 1.0
                    scale[:, :, 1, 1] = 1.0
                layer.weight.grad.mul_(scale)

            if layer.bias.grad is not None:
                scale_b = torch.ones_like(layer.bias.grad)
                if self.phase == "wake":
                    scale_b[:, 0] = 1.0
                    scale_b[:, 1] = 0.0
                else:
                    scale_b[:, 0] = self.slow_lr / self.lr if self.lr else 0.0
                    scale_b[:, 1] = 1.0
                layer.bias.grad.mul_(scale_b)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EvalResult:
    model.train()
    return run_epoch(model, loader, device, optimizer)


def build_replay_loader(
    model: nn.Module,
    buffer: ReplayBuffer,
    batch_size: int,
    device: torch.device,
    policy: str,
    disagreement_weight: float,
    loss_weight: float,
) -> DataLoader | None:
    priorities = replay_priorities(
        model,
        buffer,
        batch_size,
        device,
        policy,
        disagreement_weight,
        loss_weight,
    )
    return buffer.loader(batch_size, device, priorities)


def dream_phase(
    model: nn.Module,
    buffer: ReplayBuffer,
    device: torch.device,
    args: argparse.Namespace,
) -> list[EvalResult]:
    replay_loader = build_replay_loader(
        model,
        buffer,
        args.batch_size,
        device,
        args.replay_policy,
        args.replay_disagreement_weight,
        args.replay_loss_weight,
    )
    if replay_loader is None or args.consolidation_epochs <= 0:
        return []

    con_optim = build_optimizer(model, "consolidate", args.lr, args.slow_lr, args.weight_decay)
    return [train_epoch(model, replay_loader, con_optim, device) for _ in range(args.consolidation_epochs)]


@torch.no_grad()
def evaluate(model: AllostaticChevronNet, loader: DataLoader, device: torch.device) -> EvalResult:
    model.eval()
    return run_epoch(model, loader, device, None)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> EvalResult:
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    total_d1 = 0.0
    total_d2 = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        logits, stats = model(images, return_stats=True)
        loss = F.cross_entropy(logits, labels)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_size
        total_d1 += stats["disagree_1"].item() * batch_size
        total_d2 += stats["disagree_2"].item() * batch_size

    denom = max(1, total_count)
    return EvalResult(
        loss=total_loss / denom,
        accuracy=total_correct / denom,
        disagree_1=total_d1 / denom,
        disagree_2=total_d2 / denom,
    )


@torch.no_grad()
def replay_priorities(
    model: nn.Module,
    buffer: ReplayBuffer,
    batch_size: int,
    device: torch.device,
    policy: str,
    disagreement_weight: float,
    loss_weight: float,
) -> torch.Tensor | None:
    if policy == "uniform":
        return None

    tensors = buffer.tensors(device)
    if tensors is None:
        return None

    model.eval()
    images, labels = tensors
    disagreement_scores = []
    loss_scores = []
    for start in range(0, labels.size(0), batch_size):
        batch_images = images[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        logits, stats = model(batch_images, return_stats=True)
        disagreement_scores.append(stats["priority_disagreement"].detach().cpu())

        if policy == "loss_disagreement":
            loss_scores.append(F.cross_entropy(logits, batch_labels, reduction="none").detach().cpu())
        elif policy != "disagreement":
            raise ValueError(f"Unknown replay policy: {policy}")

    disagreement = normalize_priority(torch.cat(disagreement_scores).float())
    if policy == "disagreement":
        priorities = disagreement
    else:
        loss = normalize_priority(torch.cat(loss_scores).float())
        priorities = disagreement_weight * disagreement + loss_weight * loss
    return priorities + 1e-6


def normalize_priority(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    span = values.max() - values.min()
    if span <= 1e-12:
        return torch.ones_like(values)
    return (values - values.min()) / span


def coupling_norms(model: nn.Module) -> dict[str, float]:
    out: dict[str, float] = {}
    if not hasattr(model, "chevron_layers"):
        return out
    for i, layer in enumerate(model.chevron_layers(), start=1):
        w = layer.effective_weight().detach()
        out[f"layer{i}_AA"] = w[:, :, 0, 0].norm().item()
        out[f"layer{i}_AN"] = w[:, :, 1, 0].norm().item()
        out[f"layer{i}_NA"] = w[:, :, 0, 1].norm().item()
        out[f"layer{i}_NN"] = w[:, :, 1, 1].norm().item()
    return out


def forgetting_score(accuracy_matrix: list[list[float]]) -> float:
    if len(accuracy_matrix) <= 1:
        return 0.0
    final_row = accuracy_matrix[-1]
    scores = []
    for task_idx in range(len(final_row) - 1):
        best_seen = max(row[task_idx] for row in accuracy_matrix[task_idx:] if task_idx < len(row))
        scores.append(best_seen - final_row[task_idx])
    return sum(scores) / max(1, len(scores))


def chevron_self_test() -> None:
    layer = ChevronLinear(1, 1)
    x = torch.tensor([[[2.0, -3.0]]])

    with torch.no_grad():
        layer.bias.zero_()
        layer.weight.zero_()
        layer.weight[0, 0, 0, 0] = 1.0
        layer.weight[0, 0, 1, 1] = 1.0
    y = layer(x)
    assert torch.allclose(y, x), f"identity failed: {y}"

    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[0, 0, 0, 1] = 1.0
        layer.weight[0, 0, 1, 0] = 1.0
    y = layer(x)
    expected = torch.tensor([[[-3.0, 2.0]]])
    assert torch.allclose(y, expected), f"swap failed: {y}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Allostatic A/N Chevron Net on Split MNIST")
    parser.add_argument("--model", choices=["chevron", "mlp"], default="chevron")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--consolidation-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--slow-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=0)
    parser.add_argument("--replay-policy", choices=["uniform", "disagreement", "loss_disagreement"], default="uniform")
    parser.add_argument("--replay-disagreement-weight", type=float, default=1.0)
    parser.add_argument("--replay-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--dream-schedule",
        choices=["post_task", "after_epoch", "tension_gated", "persistent_tension"],
        default="post_task",
    )
    parser.add_argument("--dream-threshold", type=float, default=0.95)
    parser.add_argument("--dream-margin", type=float, default=0.03)
    parser.add_argument("--dream-ema", type=float, default=0.9)
    parser.add_argument("--dream-patience", type=int, default=2)
    parser.add_argument("--readout", choices=["both", "a_only"], default="both")
    parser.add_argument("--diagonal-only", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-per-task", type=int, default=0)
    parser.add_argument("--max-test-per-task", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    def log(message: str) -> None:
        if not args.quiet:
            print(message)

    if args.self_test:
        chevron_self_test()
        log("chevron_self_test=passed")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    log(f"device={device}")

    train_tasks, test_tasks = build_tasks(args.data_dir, args.max_train_per_task, args.max_test_per_task)
    if args.model == "chevron":
        model = AllostaticChevronNet(
            width=args.width,
            diagonal_only=args.diagonal_only,
            readout=args.readout,
        ).to(device)
    else:
        model = ScalarMLP(width=args.width).to(device)
    buffer = ReplayBuffer(args.buffer_size, seed=args.seed)

    run_name = (
        f"{args.model}_seed{args.seed}_w{args.width}_{args.readout}"
        f"_buf{args.buffer_size}_con{args.consolidation_epochs}_{args.replay_policy}"
        f"_{args.dream_schedule}"
    )
    if args.dream_schedule == "tension_gated":
        run_name += f"_thr{args.dream_threshold:g}"
    if args.dream_schedule == "persistent_tension":
        run_name += f"_m{args.dream_margin:g}_ema{args.dream_ema:g}_p{args.dream_patience}"
    if args.replay_policy == "loss_disagreement":
        run_name += f"_dw{args.replay_disagreement_weight:g}_lw{args.replay_loss_weight:g}"
    if args.model == "chevron" and args.diagonal_only:
        run_name += "_diag"
    run_dir = args.out_dir / run_name

    rows: list[dict[str, object]] = []
    accuracy_matrix: list[list[float]] = []
    tension_ema: float | None = None
    unresolved_epochs = 0

    for task_idx, task_digits in enumerate(TASKS):
        log(f"\ntrain_task={task_idx + 1} digits={task_digits}")
        train_loader = DataLoader(
            train_tasks[task_idx],
            batch_size=args.batch_size,
            shuffle=True,
        )
        wake_optim = build_optimizer(model, "wake", args.lr, args.slow_lr, args.weight_decay)

        for epoch in range(1, args.epochs + 1):
            result = train_epoch(model, train_loader, wake_optim, device)
            log(
                f"wake epoch={epoch} loss={result.loss:.4f} acc={result.accuracy:.4f} "
                f"d1={result.disagree_1:.4f} d2={result.disagree_2:.4f}"
            )

            wake_tension = 0.5 * (result.disagree_1 + result.disagree_2)
            should_dream = args.dream_schedule == "after_epoch"
            if args.dream_schedule == "tension_gated":
                should_dream = wake_tension >= args.dream_threshold
            elif args.dream_schedule == "persistent_tension":
                baseline = wake_tension if tension_ema is None else tension_ema
                if wake_tension > baseline + args.dream_margin:
                    unresolved_epochs += 1
                else:
                    unresolved_epochs = 0
                should_dream = unresolved_epochs >= args.dream_patience
                tension_ema = (
                    wake_tension
                    if tension_ema is None
                    else args.dream_ema * tension_ema + (1.0 - args.dream_ema) * wake_tension
                )
                if should_dream:
                    unresolved_epochs = 0
            if should_dream and len(buffer) > 0:
                for dream_idx, dream_result in enumerate(dream_phase(model, buffer, device, args), start=1):
                    log(
                        f"dream after_wake_epoch={epoch} dream_epoch={dream_idx} "
                        f"loss={dream_result.loss:.4f} acc={dream_result.accuracy:.4f} "
                        f"d1={dream_result.disagree_1:.4f} d2={dream_result.disagree_2:.4f}"
                    )

        buffer.add_balanced(train_tasks[task_idx])

        if args.dream_schedule == "post_task":
            for epoch, result in enumerate(dream_phase(model, buffer, device, args), start=1):
                log(
                    f"dream post_task_epoch={epoch} loss={result.loss:.4f} acc={result.accuracy:.4f} "
                    f"d1={result.disagree_1:.4f} d2={result.disagree_2:.4f}"
                )

        task_accs: list[float] = []
        norms = coupling_norms(model)
        for eval_idx in range(task_idx + 1):
            test_loader = DataLoader(test_tasks[eval_idx], batch_size=args.batch_size, shuffle=False)
            result = evaluate(model, test_loader, device)
            task_accs.append(result.accuracy)
            row = {
                "after_task": task_idx + 1,
                "eval_task": eval_idx + 1,
                "digits": f"{TASKS[eval_idx][0]}-{TASKS[eval_idx][1]}",
                "loss": f"{result.loss:.6f}",
                "accuracy": f"{result.accuracy:.6f}",
                "disagree_1": f"{result.disagree_1:.6f}",
                "disagree_2": f"{result.disagree_2:.6f}",
            }
            row.update({k: f"{v:.6f}" for k, v in norms.items()})
            rows.append(row)
            log(
                f"eval task={eval_idx + 1} acc={result.accuracy:.4f} "
                f"loss={result.loss:.4f} d1={result.disagree_1:.4f} d2={result.disagree_2:.4f}"
            )

        accuracy_matrix.append(task_accs)
        log("accuracy_matrix_row=" + " ".join(f"{acc:.4f}" for acc in task_accs))
        log(f"forgetting_so_far={forgetting_score(accuracy_matrix):.4f}")

    write_csv(run_dir / "metrics.csv", rows)
    print(f"final_forgetting={forgetting_score(accuracy_matrix):.4f}")
    print(f"metrics_csv={run_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
