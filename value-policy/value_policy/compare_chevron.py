from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from value_policy.config import EnvConfig, ModelConfig, PPOConfig
from value_policy.evaluate import evaluate_channel_interventions, evaluate_checkpoint
from value_policy.ppo import train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare ungated vs gated Structured Chevron across multiple seeds."
    )
    parser.add_argument("--output-dir", default="runs/compare_chevron")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=1024)
    parser.add_argument("--eval-num-envs", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--encoder-size", type=int, default=64)
    parser.add_argument("--cue-prob", type=float, default=0.60)
    parser.add_argument("--wait-penalty", type=float, default=-0.01)
    parser.add_argument("--reversal-cue-delay", type=int, default=1)
    parser.add_argument("--disable-uncertainty-scale", action="store_true")
    return parser


def flatten_seed_result(result: dict[str, Any]) -> dict[str, float | str | int]:
    row: dict[str, float | str | int] = {
        "variant": result["variant"],
        "seed": result["seed"],
        "checkpoint": result["checkpoint"],
    }
    train_final = result["train_final"] or {}
    sampled_eval = result["sampled_eval"]["eval"]
    greedy_eval = result["greedy_eval"]["eval"]
    interventions = result["channel_interventions"]
    for key, value in train_final.items():
        row[f"train_{key}"] = value
    for key, value in sampled_eval.items():
        row[f"sampled_{key}"] = value
    for key, value in greedy_eval.items():
        row[f"greedy_{key}"] = value
    for intervention_name, intervention_payload in interventions.items():
        if intervention_name == "base":
            for key, value in intervention_payload.items():
                row[f"intervention_base_{key}"] = value
            continue
        deltas = intervention_payload["delta_vs_base"]
        for key, value in deltas.items():
            row[f"{intervention_name}_delta_{key}"] = value
    return row


def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["variant"], []).append(result)

    summaries: list[dict[str, Any]] = []
    for variant, variant_results in grouped.items():
        rows = [flatten_seed_result(result) for result in variant_results]
        numeric_keys = sorted(
            key
            for key in rows[0]
            if key not in {"variant", "checkpoint"} and not isinstance(rows[0][key], str)
        )
        aggregate: dict[str, Any] = {
            "variant": variant,
            "num_seeds": len(variant_results),
            "seeds": [result["seed"] for result in variant_results],
        }
        for key in numeric_keys:
            aggregate[key] = mean(float(row[key]) for row in rows)
        summaries.append(aggregate)
    return summaries


def run_variant(
    variant: str,
    gated_policy: bool,
    args: argparse.Namespace,
    env_config: EnvConfig,
) -> list[dict[str, Any]]:
    variant_dir = Path(args.output_dir) / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        checkpoint_path = variant_dir / f"seed_{seed}.pt"
        model_config = ModelConfig(
            model_name="structured_chevron",
            hidden_size=args.hidden_size,
            encoder_size=args.encoder_size,
            uncertainty_scale=not args.disable_uncertainty_scale,
            gated_policy=gated_policy,
        )
        ppo_config = PPOConfig(
            num_envs=args.num_envs,
            updates=args.updates,
            seed=seed,
            device=args.device,
        )
        _, history = train_experiment(
            env_config,
            model_config,
            ppo_config,
            checkpoint_path=str(checkpoint_path),
        )
        sampled_eval = evaluate_checkpoint(
            checkpoint_path=str(checkpoint_path),
            device=args.device,
            num_envs=args.eval_num_envs,
            episodes=args.eval_episodes,
            seed=seed + 1000,
            greedy=False,
        )
        greedy_eval = evaluate_checkpoint(
            checkpoint_path=str(checkpoint_path),
            device=args.device,
            num_envs=args.eval_num_envs,
            episodes=args.eval_episodes,
            seed=seed + 2000,
            greedy=True,
        )
        interventions = evaluate_channel_interventions(
            checkpoint_path=str(checkpoint_path),
            device=args.device,
            num_envs=args.eval_num_envs,
            episodes=args.eval_episodes,
            seed=seed + 3000,
        )
        results.append(
            {
                "variant": variant,
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "train_final": history[-1] if history else None,
                "sampled_eval": sampled_eval,
                "greedy_eval": greedy_eval,
                "channel_interventions": interventions,
            }
        )
    return results


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_config = EnvConfig(
        cue_prob=args.cue_prob,
        wait_penalty=args.wait_penalty,
        reversal_cue_delay=args.reversal_cue_delay,
    )

    all_results = []
    all_results.extend(run_variant("ungated", gated_policy=False, args=args, env_config=env_config))
    all_results.extend(run_variant("gated", gated_policy=True, args=args, env_config=env_config))

    aggregate = aggregate_results(all_results)

    detail_json = output_dir / "detail.json"
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"

    detail_json.write_text(json.dumps(all_results, indent=2, sort_keys=True) + "\n")
    summary_json.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")

    rows = [flatten_seed_result(result) for result in all_results]
    fieldnames = sorted({key for row in rows for key in row})
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "seeds": args.seeds,
                "variants": ["ungated", "gated"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
