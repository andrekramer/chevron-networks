"""Run the phase-one comparison over several independent random seeds."""

import argparse
import math
import random
import statistics
from typing import Dict, List

import torch

from chevron_attention import (
    ChevronAttention,
    RecallTask,
    TransformerBaseline,
    select_device,
    train_one,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--retrieval-weight", type=float, default=1.0)
    parser.add_argument("--permission-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-keys", type=int, default=16)
    parser.add_argument("--num-values", type=int, default=16)
    parser.add_argument("--num-facts", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    return parser.parse_args()


def summarize(values: List[float]) -> str:
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    return "%.4f +/- %.4f [%.4f, %.4f]" % (
        statistics.mean(values),
        spread,
        min(values),
        max(values),
    )


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    task = RecallTask(args.num_keys, args.num_values, args.num_facts)
    max_length = task.batch(1, random.Random(0)).n_tokens.size(1)
    all_results: Dict[str, List[Dict[str, float]]] = {
        "chevron": [],
        "baseline": [],
    }

    for seed in args.seeds:
        args.seed = seed
        print("SEED %d" % seed, flush=True)

        # Reset model initialization for a fair, independently reproducible run.
        torch.manual_seed(seed)
        chevron = ChevronAttention(
            task, max_length, args.d_model, args.heads, args.layers
        )
        all_results["chevron"].append(
            train_one("chevron-seed-%d" % seed, chevron, task, args, device)
        )

        torch.manual_seed(seed)
        baseline_width = int(round(args.d_model * math.sqrt(2) / args.heads)) * args.heads
        baseline = TransformerBaseline(
            task, max_length, baseline_width, args.heads, args.layers
        )
        all_results["baseline"].append(
            train_one("baseline-seed-%d" % seed, baseline, task, args, device)
        )

    print("MULTI-SEED RESULTS")
    print("seeds=%s" % ",".join(str(seed) for seed in args.seeds))
    for model_name, runs in all_results.items():
        print(model_name)
        for metric in runs[0]:
            values = [run[metric] for run in runs]
            print("  %s %s" % (metric, summarize(values)))
        print("  perfect_runs %d/%d" % (
            sum(run["answer_accuracy"] == 1.0 for run in runs), len(runs)
        ))


if __name__ == "__main__":
    main()
