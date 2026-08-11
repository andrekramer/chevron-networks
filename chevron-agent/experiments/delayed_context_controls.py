from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from chevron_agent.envs.delayed_context import DelayedContextEnv, DelayedContextSpec


@dataclass
class LifetimeResult:
    seed: int
    condition: str
    delay: int
    return_per_decision: float
    old_retention: float
    new_acquisition: float
    overall_accuracy: float


class MemorylessPolicy:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, context_id: int) -> int:
        del context_id
        return int(self.rng.integers(0, 2))

    def observe(self, context_id: int, action: int, reward: float) -> None:
        del context_id, action, reward


class OracleContextMemory:
    """Task-validity control with privileged IDs but no correct-action access."""

    def __init__(self, contexts: int, seed: int) -> None:
        self.known_action = np.full(contexts, -1, dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def act(self, context_id: int) -> int:
        known = int(self.known_action[context_id])
        return known if known >= 0 else int(self.rng.integers(0, 2))

    def observe(self, context_id: int, action: int, reward: float) -> None:
        self.known_action[context_id] = action if reward > 0 else 1 - action


def run_lifetime(condition: str, seed: int, delay: int) -> LifetimeResult:
    spec = DelayedContextSpec(outcome_delay=delay)
    env = DelayedContextEnv(spec)
    _, info = env.reset(seed=seed)
    if condition == "memoryless":
        policy = MemorylessPolicy(seed=seed + 10_000)
    elif condition == "oracle_context_memory":
        policy = OracleContextMemory(contexts=spec.total_contexts, seed=seed + 10_000)
    else:
        raise ValueError(f"unknown condition: {condition}")

    eligibility: list[tuple[int, int]] = []
    decisions: list[tuple[int, bool, bool]] = []
    total_reward = 0.0
    truncated = False
    while not truncated:
        active = bool(info["decision_active"])
        if active:
            context_id = int(info["context_id"])
            action = policy.act(context_id)
            eligibility.append((context_id, action))
            correct = action == int(env.correct_actions[context_id])
            is_new = context_id >= spec.old_contexts
            decisions.append((int(info["decision_index"]), is_new, correct))
        else:
            action = 0

        _, reward, _, truncated, info = env.step(action)
        total_reward += reward
        if info["outcome_decision_index"] is not None:
            eligible_context, eligible_action = eligibility.pop(0)
            policy.observe(eligible_context, eligible_action, reward)

    if eligibility:
        raise AssertionError("eligibility queue was not fully drained")

    final_start = spec.decision_steps - 200
    final = [item for item in decisions if item[0] >= final_start]
    old = [correct for _, is_new, correct in final if not is_new]
    new = [correct for _, is_new, correct in final if is_new]
    return LifetimeResult(
        seed=seed,
        condition=condition,
        delay=delay,
        return_per_decision=total_reward / spec.decision_steps,
        old_retention=float(np.mean(old)),
        new_acquisition=float(np.mean(new)),
        overall_accuracy=float(np.mean([correct for _, _, correct in decisions])),
    )


def summarize(results: list[LifetimeResult]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    metrics = ("return_per_decision", "old_retention", "new_acquisition", "overall_accuracy")
    groups = sorted({(result.condition, result.delay) for result in results})
    for condition, delay in groups:
        rows = [result for result in results if result.condition == condition and result.delay == delay]
        key = f"{condition}/delay_{delay}"
        summary[key] = {}
        for metric in metrics:
            values = np.asarray([getattr(row, metric) for row in rows])
            summary[key][f"{metric}_mean"] = float(values.mean())
            summary[key][f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary


def markdown(summary: dict[str, dict[str, float]], seeds: int) -> str:
    lines = [
        "# Delayed-context task-validity controls",
        "",
        f"Results over {seeds} seeds. These are task controls, not model-comparison results.",
        "",
        "| Condition | Delay | Return/decision | Final old accuracy | Final new accuracy | Overall accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, values in summary.items():
        condition, delay_label = key.split("/")
        delay = delay_label.removeprefix("delay_")
        lines.append(
            f"| {condition} | {delay} | "
            f"{values['return_per_decision_mean']:.3f} +/- {values['return_per_decision_std']:.3f} | "
            f"{values['old_retention_mean']:.3f} +/- {values['old_retention_std']:.3f} | "
            f"{values['new_acquisition_mean']:.3f} +/- {values['new_acquisition_std']:.3f} | "
            f"{values['overall_accuracy_mean']:.3f} +/- {values['overall_accuracy_std']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The memoryless control should remain near chance. The oracle-context memory is",
            "allowed to use latent IDs only to verify that the delayed task is solvable when",
            "eligibility is handled correctly; experimental agents will not receive those IDs.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()

    results = [
        run_lifetime(condition, seed, delay)
        for condition in ("memoryless", "oracle_context_memory")
        for delay in (0, 3)
        for seed in range(args.seeds)
    ]
    summary = summarize(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec": asdict(DelayedContextSpec()),
        "seeds": args.seeds,
        "summary": summary,
        "individual": [asdict(result) for result in results],
    }
    json_path = args.output_dir / "delayed_context_controls.json"
    md_path = args.output_dir / "delayed_context_controls.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(markdown(summary, args.seeds), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
