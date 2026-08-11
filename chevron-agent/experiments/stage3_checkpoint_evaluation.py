"""Independent evaluation of the existing Stage 3 RL checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

import torch

from chevron_agent.config import Config
from chevron_agent.train import build_agent, evaluate


RUNS = {
    0: {"baseline": "baseline_stage3", "chevron": "chevron_stage3"},
    1: {"baseline": "baseline_stage3_seed1", "chevron": "chevron_stage3_seed1"},
    2: {"baseline": "baseline_stage3_seed2", "chevron": "chevron_stage3_seed2"},
}


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "population_sd": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _load_and_evaluate(
    checkpoint_path: Path,
    *,
    episodes: int,
    seed_offset: int,
) -> tuple[dict[str, float], int, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = Config(**checkpoint["config"])
    channels = 3 if config.observation_mode == "rgb" else 9
    agent = build_agent(
        config,
        action_dim=6,
        in_channels=channels,
    )
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    metrics = evaluate(
        agent,
        config,
        torch.device("cpu"),
        episodes,
        seed_offset=seed_offset,
    )
    parameters = sum(parameter.numel() for parameter in agent.parameters())
    return metrics, int(checkpoint["update"]), parameters


def _checkpoint_curve(
    run_dir: Path,
    *,
    episodes: int,
    seed_offset: int,
) -> list[dict[str, float]]:
    curve = []
    for checkpoint_path in sorted(run_dir.glob("update_*.pt")):
        metrics, update, _ = _load_and_evaluate(
            checkpoint_path,
            episodes=episodes,
            seed_offset=seed_offset,
        )
        curve.append({"update": update, **metrics})
    return curve


def run(
    repo_root: Path,
    *,
    episodes: int,
    seed_offset: int,
) -> dict[str, Any]:
    seed_results: list[dict[str, Any]] = []
    parameter_counts: dict[str, int] = {}
    for seed, model_runs in RUNS.items():
        model_results: dict[str, Any] = {}
        for model, run_name in model_runs.items():
            run_dir = repo_root / "runs" / run_name
            final_metrics, final_update, parameters = _load_and_evaluate(
                run_dir / "latest.pt",
                episodes=episodes,
                seed_offset=seed_offset,
            )
            selected_metrics, selected_update, _ = _load_and_evaluate(
                run_dir / "best_eval.pt",
                episodes=episodes,
                seed_offset=seed_offset,
            )
            curve = _checkpoint_curve(
                run_dir,
                episodes=episodes,
                seed_offset=seed_offset,
            )
            parameter_counts[model] = parameters
            model_results[model] = {
                "final_update": final_update,
                "final": final_metrics,
                "development_selected_update": selected_update,
                "development_selected": selected_metrics,
                "heldout_curve": curve,
                "curve_mean_success": statistics.fmean(
                    point["success_rate"] for point in curve
                ),
                "curve_mean_return": statistics.fmean(
                    point["return_mean"] for point in curve
                ),
            }
        seed_results.append({"seed": seed, "models": model_results})

    summary: dict[str, Any] = {}
    for model in ("baseline", "chevron"):
        summary[model] = {
            "final_success": _mean_sd(
                [result["models"][model]["final"]["success_rate"] for result in seed_results]
            ),
            "final_return": _mean_sd(
                [result["models"][model]["final"]["return_mean"] for result in seed_results]
            ),
            "curve_mean_success": _mean_sd(
                [result["models"][model]["curve_mean_success"] for result in seed_results]
            ),
            "selected_success": _mean_sd(
                [result["models"][model]["development_selected"]["success_rate"] for result in seed_results]
            ),
        }
    paired_final = [
        result["models"]["chevron"]["final"]["success_rate"]
        - result["models"]["baseline"]["final"]["success_rate"]
        for result in seed_results
    ]
    paired_curve = [
        result["models"]["chevron"]["curve_mean_success"]
        - result["models"]["baseline"]["curve_mean_success"]
        for result in seed_results
    ]
    summary["paired"] = {
        "final_success_difference": _mean_sd(paired_final),
        "final_chevron_wins": sum(value > 0 for value in paired_final),
        "curve_success_difference": _mean_sd(paired_curve),
        "curve_chevron_wins": sum(value > 0 for value in paired_curve),
    }
    return {
        "experiment": "stage3_existing_checkpoint_evaluation",
        "episodes_per_checkpoint": episodes,
        "evaluation_seed_offset": seed_offset,
        "strict_success": "positively rewarded terminal interaction",
        "parameter_counts": parameter_counts,
        "seed_results": seed_results,
        "summary": summary,
    }


def _pm(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f} +/- {metric['population_sd']:.3f}"


def write_results(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage3_existing_checkpoint_results.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage 3 independent checkpoint evaluation",
        "",
        f"Each saved checkpoint was evaluated on {result['episodes_per_checkpoint']} "
        "fresh, shared episodes. Success requires a positively rewarded terminal "
        "interaction; timeout shaping cannot count as success.",
        "",
        "| Model | Parameters | Final success | Final return | Curve mean success | Selected-checkpoint success |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, display in (("baseline", "GRU baseline"), ("chevron", "Plain Chevron")):
        metrics = result["summary"][model]
        lines.append(
            f"| {display} | {result['parameter_counts'][model]:,} | "
            f"{_pm(metrics['final_success'])} | "
            f"{_pm(metrics['final_return'])} | "
            f"{_pm(metrics['curve_mean_success'])} | "
            f"{_pm(metrics['selected_success'])} |"
        )
    paired = result["summary"]["paired"]
    lines.extend(
        [
            "",
            "## Paired differences",
            "",
            f"- Final Chevron-minus-GRU success: {_pm(paired['final_success_difference'])}",
            f"- Final Chevron wins: {paired['final_chevron_wins']}/3",
            f"- Learning-curve Chevron-minus-GRU success: {_pm(paired['curve_success_difference'])}",
            f"- Learning-curve Chevron wins: {paired['curve_chevron_wins']}/3",
            "",
            "This audit reuses existing trained checkpoints. It removes evaluation-set "
            "reuse and the permissive success metric, but remains a three-seed result.",
            "",
        ]
    )
    (output_dir / "stage3_existing_checkpoint_results.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-offset", type=int, default=1_000_000)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    result = run(
        repo_root,
        episodes=args.episodes,
        seed_offset=args.seed_offset,
    )
    write_results(result, repo_root / "experiments" / "results")
    for model in ("baseline", "chevron"):
        summary = result["summary"][model]
        print(
            f"{model:8s} final={_pm(summary['final_success'])} "
            f"curve={_pm(summary['curve_mean_success'])}"
        )
    print(
        "paired final ",
        _pm(result["summary"]["paired"]["final_success_difference"]),
    )


if __name__ == "__main__":
    main()
