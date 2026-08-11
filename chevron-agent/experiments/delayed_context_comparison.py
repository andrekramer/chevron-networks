from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from chevron_agent.envs.delayed_context import DelayedContextEnv, DelayedContextSpec
from chevron_agent.memory.delayed_context import ChevronMemory, MemoryConfig, StandardAttentionMemory


CONDITIONS = (
    "standard_attention",
    "standard_attention_buffer",
    "chevron_buffer",
    "chevron_immediate",
    "chevron_scalar_residual",
    "chevron_coupled_write",
)


@dataclass
class ComparisonResult:
    seed: int
    condition: str
    return_per_decision: float
    old_retention: float
    new_acquisition: float
    overall_accuracy: float
    retention_plasticity_score: float
    residual_calibration: float
    new_q_first: float
    new_q_late: float
    premature_write_rate: float
    established_overwrite_rate: float
    promotion_precision: float
    permanent_writes: int
    final_slots: int
    final_candidates: int
    max_conservation_error: float


def build_policy(condition: str, spec: DelayedContextSpec, seed: int, config: MemoryConfig):
    kwargs = dict(
        address_dim=spec.address_dim,
        diagnostic_dim=spec.diagnostic_dim,
        seed=seed + 50_000,
        config=config,
    )
    if condition == "standard_attention":
        return StandardAttentionMemory(**kwargs, use_buffer=False)
    if condition == "standard_attention_buffer":
        return StandardAttentionMemory(**kwargs, use_buffer=True)
    if condition == "chevron_buffer":
        return ChevronMemory(**kwargs, use_buffer=True)
    if condition == "chevron_immediate":
        return ChevronMemory(**kwargs, use_buffer=False, immediate_write=True)
    if condition == "chevron_scalar_residual":
        return ChevronMemory(**kwargs, use_buffer=True, per_slot_residual=False)
    if condition == "chevron_coupled_write":
        return ChevronMemory(**kwargs, use_buffer=True, coupled_write=True)
    raise ValueError(f"unknown condition: {condition}")


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def run_lifetime(
    condition: str,
    seed: int,
    spec: DelayedContextSpec,
    memory_config: MemoryConfig,
) -> ComparisonResult:
    env = DelayedContextEnv(spec)
    observation, info = env.reset(seed=seed)
    policy = build_policy(condition, spec, seed, memory_config)
    resolved_contexts: set[int] = set()
    context_exposures = np.zeros(spec.total_contexts, dtype=np.int64)
    decisions: list[dict[str, float | int | bool]] = []
    total_reward = 0.0
    truncated = False

    while not truncated:
        if bool(info["decision_active"]):
            context_id = int(info["context_id"])
            decision_index = int(info["decision_index"])
            address = observation[env.address_slice]
            diagnostic = observation[env.diagnostic_slice]
            action, trace = policy.act(address, diagnostic, decision_index, context_id)
            correct = action == int(env.correct_actions[context_id])
            policy.note_use(trace, context_id, correct)
            decisions.append(
                {
                    "index": decision_index,
                    "context_id": context_id,
                    "is_new": context_id >= spec.old_contexts,
                    "correct": correct,
                    "q": trace.q,
                    "was_unresolved": context_id not in resolved_contexts,
                    "context_exposure": int(context_exposures[context_id]),
                }
            )
            context_exposures[context_id] += 1
        else:
            action = 0

        observation, reward, _, truncated, next_info = env.step(action)
        total_reward += reward
        if next_info["outcome_decision_index"] is not None:
            eligible = policy.observe(reward)
            if eligible.decision_index != int(next_info["outcome_decision_index"]):
                raise AssertionError("agent eligibility queue lost alignment")
            # Audit metadata updates the metric label only; it is not passed to
            # the policy's outcome update.
            resolved_contexts.add(int(next_info["outcome_context_id"]))
        info = next_info

    if policy.pending:
        raise AssertionError("agent eligibility queue was not fully drained")

    final_start = spec.decision_steps - 200
    final = [row for row in decisions if int(row["index"]) >= final_start]
    old = [bool(row["correct"]) for row in final if not bool(row["is_new"])]
    new = [bool(row["correct"]) for row in final if bool(row["is_new"])]
    q_unresolved = [float(row["q"]) for row in decisions if bool(row["was_unresolved"])]
    q_resolved = [float(row["q"]) for row in decisions if not bool(row["was_unresolved"])]
    new_first = [
        float(row["q"])
        for row in decisions
        if bool(row["is_new"]) and int(row["context_exposure"]) < 2
    ]
    new_late = [
        float(row["q"])
        for row in decisions
        if bool(row["is_new"]) and int(row["context_exposure"]) >= 20
    ]
    premature_write_rate = policy.premature_writes / spec.decision_steps
    established_overwrite_rate = policy.established_overwrites / (
        spec.old_contexts * spec.decision_steps
    )
    old_retention = float(np.mean(old))
    new_acquisition = float(np.mean(new))
    return ComparisonResult(
        seed=seed,
        condition=condition,
        return_per_decision=total_reward / spec.decision_steps,
        old_retention=old_retention,
        new_acquisition=new_acquisition,
        overall_accuracy=float(np.mean([bool(row["correct"]) for row in decisions])),
        retention_plasticity_score=(
            old_retention
            + new_acquisition
            - 0.5 * established_overwrite_rate
            - 0.5 * premature_write_rate
        ),
        residual_calibration=finite_mean(q_unresolved) - finite_mean(q_resolved),
        new_q_first=finite_mean(new_first),
        new_q_late=finite_mean(new_late),
        premature_write_rate=premature_write_rate,
        established_overwrite_rate=established_overwrite_rate,
        promotion_precision=policy.promotion_precision,
        permanent_writes=policy.permanent_writes,
        final_slots=len(policy.slots),
        final_candidates=len(policy.candidates),
        max_conservation_error=policy.max_conservation_error,
    )


def summarize(results: list[ComparisonResult]) -> dict[str, dict[str, float]]:
    metrics = [field for field in ComparisonResult.__dataclass_fields__ if field not in {"seed", "condition"}]
    summary: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        rows = [row for row in results if row.condition == condition]
        summary[condition] = {}
        for metric in metrics:
            values = np.asarray([getattr(row, metric) for row in rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            summary[condition][f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            summary[condition][f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
    return summary


def paired_differences(results: list[ComparisonResult]) -> dict[str, dict[str, float]]:
    comparisons = (
        ("chevron_buffer", "standard_attention"),
        ("chevron_buffer", "standard_attention_buffer"),
        ("chevron_buffer", "chevron_immediate"),
        ("chevron_buffer", "chevron_scalar_residual"),
        ("chevron_buffer", "chevron_coupled_write"),
    )
    metrics = ("old_retention", "new_acquisition", "return_per_decision")
    output: dict[str, dict[str, float]] = {}
    by_key = {(row.condition, row.seed): row for row in results}
    seeds = sorted({row.seed for row in results})
    for left, right in comparisons:
        key = f"{left}_minus_{right}"
        output[key] = {}
        for metric in metrics:
            values = np.asarray(
                [getattr(by_key[(left, seed)], metric) - getattr(by_key[(right, seed)], metric) for seed in seeds]
            )
            output[key][f"{metric}_mean"] = float(values.mean())
            output[key][f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            output[key][f"{metric}_wins"] = int(np.sum(values > 0.0))
            standard_error = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            output[key][f"{metric}_approx_95ci_low"] = float(values.mean() - 1.96 * standard_error)
            output[key][f"{metric}_approx_95ci_high"] = float(values.mean() + 1.96 * standard_error)
    return output


def markdown(
    summary: dict[str, dict[str, float]],
    paired: dict[str, dict[str, float]],
    seed_start: int,
    seeds: int,
    label: str,
) -> str:
    lines = [
        f"# Delayed-context {label} results",
        "",
        f"Seeds {seed_start}-{seed_start + seeds - 1}.",
        "",
        "| Condition | Return/decision | Final old | Final new | q unresolved-resolved | Premature writes | Old overwrite rate | Promotion precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        values = summary[condition]
        lines.append(
            f"| {condition} | "
            f"{values['return_per_decision_mean']:.3f} +/- {values['return_per_decision_std']:.3f} | "
            f"{values['old_retention_mean']:.3f} +/- {values['old_retention_std']:.3f} | "
            f"{values['new_acquisition_mean']:.3f} +/- {values['new_acquisition_std']:.3f} | "
            f"{values['residual_calibration_mean']:.3f} +/- {values['residual_calibration_std']:.3f} | "
            f"{values['premature_write_rate_mean']:.4f} | "
            f"{values['established_overwrite_rate_mean']:.6f} | "
            f"{values['promotion_precision_mean']:.3f} |"
        )
    lines.extend(["", "## Paired diagnostics", "", "```json", json.dumps(paired, indent=2), "```", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--label", choices=("development", "confirmation"), default="development")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()

    if args.label == "confirmation" and args.seed_start < 100:
        raise ValueError("confirmation seeds must start at 100 or above")

    spec = DelayedContextSpec()
    memory_config = MemoryConfig()
    results = [
        run_lifetime(condition, seed, spec, memory_config)
        for condition in CONDITIONS
        for seed in range(args.seed_start, args.seed_start + args.seeds)
    ]
    summary = summarize(results)
    paired = paired_differences(results)
    payload = {
        "label": args.label,
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "task_spec": asdict(spec),
        "memory_config": asdict(memory_config),
        "summary": summary,
        "paired": paired,
        "individual": [asdict(result) for result in results],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"delayed_context_{args.label}"
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(json_safe(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (args.output_dir / f"{stem}.md").write_text(
        markdown(summary, paired, args.seed_start, args.seeds, args.label),
        encoding="utf-8",
    )
    print(json.dumps(json_safe({"summary": summary, "paired": paired}), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
