from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from value_policy.config import EnvConfig, ModelConfig, PPOConfig
from value_policy.evaluate import evaluate_checkpoint
from value_policy.ppo import train_experiment

PRIMARY_MODELS = ["mlp", "gru", "lstm", "free_chevron", "structured_chevron"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the primary baseline sweep.")
    parser.add_argument("--output-dir", default="runs/baseline_sweep")
    parser.add_argument("--models", nargs="+", default=PRIMARY_MODELS, choices=PRIMARY_MODELS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=1024)
    parser.add_argument("--eval-num-envs", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--encoder-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cue-prob", type=float, default=0.60)
    parser.add_argument("--wait-penalty", type=float, default=-0.01)
    parser.add_argument("--reversal-cue-delay", type=int, default=1)
    parser.add_argument("--gated-policy", action="store_true")
    parser.add_argument("--disable-uncertainty-scale", action="store_true")
    return parser


def flatten_result(result: dict) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "model": result["model"],
        "checkpoint": result["checkpoint"],
    }
    train_final = result["train_final"] or {}
    sampled_eval = result["sampled_eval"]["eval"]
    greedy_eval = result["greedy_eval"]["eval"]
    for key, value in train_final.items():
        row[f"train_{key}"] = value
    for key, value in sampled_eval.items():
        row[f"sampled_{key}"] = value
    for key, value in greedy_eval.items():
        row[f"greedy_{key}"] = value
    return row


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_config = EnvConfig(
        cue_prob=args.cue_prob,
        wait_penalty=args.wait_penalty,
        reversal_cue_delay=args.reversal_cue_delay,
    )
    ppo_config = PPOConfig(
        num_envs=args.num_envs,
        updates=args.updates,
        seed=args.seed,
        device=args.device,
    )

    results = []
    for model_name in args.models:
        checkpoint_path = output_dir / f"{model_name}.pt"
        model_config = ModelConfig(
            model_name=model_name,
            hidden_size=args.hidden_size,
            encoder_size=args.encoder_size,
            uncertainty_scale=not args.disable_uncertainty_scale,
            gated_policy=args.gated_policy and model_name == "structured_chevron",
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
            seed=args.seed + 1000,
            greedy=False,
        )
        greedy_eval = evaluate_checkpoint(
            checkpoint_path=str(checkpoint_path),
            device=args.device,
            num_envs=args.eval_num_envs,
            episodes=args.eval_episodes,
            seed=args.seed + 2000,
            greedy=True,
        )
        results.append(
            {
                "model": model_name,
                "checkpoint": str(checkpoint_path),
                "train_final": history[-1] if history else None,
                "sampled_eval": sampled_eval,
                "greedy_eval": greedy_eval,
            }
        )

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    summary_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    rows = [flatten_result(result) for result in results]
    fieldnames = sorted({key for row in rows for key in row})
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"output_dir": str(output_dir), "models": args.models}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
