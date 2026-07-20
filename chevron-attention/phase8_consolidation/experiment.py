"""Phase 8 development comparison: immediate versus provisional consolidation.

This module inherits the trained Phase 7 Soft Chevron and parameter-matched
joint write controller unchanged.  Both receive the same streams, retained
memory, candidate capacity, and provisional timing parameters.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from phase7_soft_chevron.continual_closure import (
    CHEVRON_LEARNED,
    JOINT_CONTROLLER,
    ClosureConfig,
    Prepared,
    decision,
    joint_features,
    prepare_models,
    probe,
    soft_components,
)
from phase7_soft_chevron.continual_memory import (
    ContinualCategories,
    ContinualConfig,
    PersistentMemory,
    forward_memory,
    memory_purity,
)
from phase7_soft_chevron.experiment import (
    JOINT_ATTENTION,
    SOFT_CHEVRON,
    Batch,
    CategoryMatchingTask,
    TaskConfig,
)
from phase8_consolidation.provisional_memory import (
    ProvisionalConfig,
    ProvisionalMemory,
    ResidualEvidence,
    residual_evidence,
)


CHEVRON_IMMEDIATE = "chevron_immediate"
CHEVRON_QUARANTINE = "chevron_quarantine"
JOINT_IMMEDIATE = "joint_immediate"
JOINT_QUARANTINE = "joint_quarantine"
ALPHA_QUARANTINE = "alpha_quarantine"
ORACLE = "oracle"
METHODS = (
    CHEVRON_IMMEDIATE,
    CHEVRON_QUARANTINE,
    JOINT_IMMEDIATE,
    JOINT_QUARANTINE,
    ALPHA_QUARANTINE,
    ORACLE,
)
QUARANTINE_METHODS = (
    CHEVRON_QUARANTINE,
    JOINT_QUARANTINE,
    ALPHA_QUARANTINE,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    event_kind: str
    event_repetitions: int
    novel_flips: int = 2
    warmup_observations: int = 60
    recovery_observations: int = 60
    interleaved: bool = False

    @property
    def should_consolidate(self) -> bool:
        return self.event_kind == "sustained"


SCENARIOS = (
    Scenario("isolated", "isolated", 1),
    Scenario("transient", "transient", 3),
    Scenario("sustained", "sustained", 8),
    Scenario("near", "sustained", 8, novel_flips=1),
)


def selected_development_config() -> ProvisionalConfig:
    """Configuration selected by the first development-seed screen."""

    return replace(
        ProvisionalConfig(),
        consolidation_threshold=0.20,
        minimum_support=5,
        minimum_eligible_mass=0.10,
        minimum_distinct_mismatch=0.04,
    )


@dataclass(frozen=True)
class DevelopmentConfig:
    observation_noise: float = 0.035
    final_per_category: int = 30
    memory_capacity: int = 9


def scenario_labels(stream: ContinualCategories, scenario: Scenario) -> List[int]:
    if scenario.event_kind != "sustained":
        return [stream.novel_labels[0]] * scenario.event_repetitions
    labels: List[int] = []
    if scenario.interleaved:
        for _ in range(scenario.event_repetitions):
            labels.extend(stream.novel_labels)
    else:
        for label in stream.novel_labels:
            labels.extend([label] * scenario.event_repetitions)
    return labels


def chevron_residual(
    prepared: Prepared,
    batch: Batch,
    active: Tensor,
    top_a_candidates: int,
) -> Tuple[ResidualEvidence, Tensor, Tensor]:
    alpha, assent, _fixed = soft_components(prepared.soft, batch, active)
    residual = residual_evidence(
        alpha[0],
        assent[0],
        top_a_candidates=top_a_candidates,
        active=active,
    )
    return residual, alpha[0], assent[0]


def joint_residual(
    prepared: Prepared,
    batch: Batch,
    active: Tensor,
    native_null_mass: float,
) -> ResidualEvidence:
    """Use the controller's directly supervised non-match estimate.

    Joint attention does not expose Chevron's slot-wise assent decomposition.
    Its parameter-matched controller therefore drives provisional eligibility
    directly.  Native null mass remains observable but is not multiplied into
    the controller score; doing so would penalize the baseline twice for the
    same no-match decision.
    """

    raw_novelty = float(
        prepared.controller(joint_features(prepared.joint, batch, active))
        .sigmoid()
        .item()
    )
    novelty = relative_threshold_evidence(
        raw_novelty, prepared.thresholds[JOINT_CONTROLLER]
    )
    q = min(max(native_null_mass, 0.0), 1.0)
    nu = min(max(novelty, 0.0), 1.0)
    return ResidualEvidence(
        remaining_mass=q,
        novelty=nu,
        eligible_mass=nu,
        best_assent=1.0 - nu,
        admitted_mass=1.0 - q,
    )


def center_probability(score: float, threshold: float, epsilon: float = 1e-6) -> float:
    """Map a calibrated decision threshold to 0.5 without changing ranking."""

    score = min(max(score, epsilon), 1.0 - epsilon)
    threshold = min(max(threshold, epsilon), 1.0 - epsilon)
    numerator = score * (1.0 - threshold)
    denominator = numerator + threshold * (1.0 - score)
    return numerator / denominator


def relative_threshold_evidence(
    score: float, threshold: float, epsilon: float = 1e-6
) -> float:
    """Express controller evidence as a clipped fraction of its threshold.

    The controller's raw score scale varies substantially by seed.  Its
    calibrated decision boundary is already frozen by Phase 7, so an odds
    ratio provides a parameter-free common scale: zero remains zero and the
    learned decision threshold maps to full provisional evidence.
    """

    score = min(max(score, epsilon), 1.0 - epsilon)
    threshold = min(max(threshold, epsilon), 1.0 - epsilon)
    score_odds = score / (1.0 - score)
    threshold_odds = threshold / (1.0 - threshold)
    return min(max(score_odds / threshold_odds, 0.0), 1.0)


def retained_templates(memory: PersistentMemory) -> List[Tensor]:
    return [slot.template_n for slot in memory.slots]


@torch.no_grad()
def run_method(
    method_name: str,
    prepared: Prepared,
    task: TaskConfig,
    scenario: Scenario,
    seed: int,
    provisional: ProvisionalConfig,
    development: DevelopmentConfig,
) -> Dict[str, object]:
    uses_joint = method_name in (JOINT_IMMEDIATE, JOINT_QUARANTINE)
    attention_method = JOINT_ATTENTION if uses_joint else SOFT_CHEVRON
    model = prepared.joint if uses_joint else prepared.soft
    device = next(model.parameters()).device
    stream = ContinualCategories(task, seed, scenario.novel_flips)
    memory = stream.initial_memory()
    memory.capacity = development.memory_capacity
    bank = ProvisionalMemory(provisional) if method_name in QUARANTINE_METHODS else None

    warmup = stream.shuffled_labels(
        stream.base_labels, scenario.warmup_observations
    )
    event = scenario_labels(stream, scenario)
    recovery = stream.shuffled_labels(
        stream.base_labels, scenario.recovery_observations
    )
    schedule = [("warmup", label) for label in warmup]
    schedule.extend(("event", label) for label in event)
    schedule.extend(("recovery", label) for label in recovery)

    observations_by_label: Dict[int, int] = {}
    acquisition_delays: Dict[int, int] = {}
    allocations: List[Dict[str, object]] = []
    cross_write_mass = 0.0
    total_write_mass = 0.0
    provisional_mass = 0.0
    provisional_observations = 0
    residual_by_phase: Dict[str, Dict[str, List[float]]] = {
        "familiar": {"q": [], "novelty": [], "eligible": []},
        "event": {"q": [], "novelty": [], "eligible": []},
    }

    for phase, label in schedule:
        query, match = stream.sample(label, development.observation_noise)
        cpu_batch, cpu_active = memory.batch(query, match)
        batch, active = cpu_batch.to(device), cpu_active.to(device)
        outputs = forward_memory(attention_method, model, batch, active)
        existing_before = label in memory.labels
        observations_by_label[label] = observations_by_label.get(label, 0) + 1

        allocate = False
        allocation_payload: Optional[Tuple[Tensor, Tensor, int]] = None
        write_mass = torch.zeros(task.num_slots, device=device)

        if method_name == ORACLE:
            matches = [
                index for index, stored in enumerate(memory.labels) if stored == label
            ]
            if matches:
                write_mass[matches[0]] = 1.0
            if (
                scenario.should_consolidate
                and label in stream.novel_labels
                and not existing_before
                and observations_by_label[label] >= provisional.minimum_support
            ):
                allocate = True
                allocation_payload = (query, match, label)
        elif method_name in (CHEVRON_IMMEDIATE, JOINT_IMMEDIATE):
            variant = CHEVRON_LEARNED if method_name == CHEVRON_IMMEDIATE else JOINT_CONTROLLER
            novelty, write_mass = decision(variant, prepared, batch, active, outputs)
            allocate = novelty >= prepared.thresholds[variant]
            if allocate:
                allocation_payload = (query, match, label)
        else:
            assert bank is not None
            if uses_joint:
                residual = joint_residual(
                    prepared,
                    batch,
                    active,
                    float(outputs["null_mass"][0].item()),
                )
                _novelty, write_mass = decision(
                    JOINT_CONTROLLER, prepared, batch, active, outputs
                )
            else:
                residual, alpha, assent = chevron_residual(
                    prepared,
                    batch,
                    active,
                    provisional.top_a_candidates,
                )
                write_mass = alpha if method_name == ALPHA_QUARANTINE else alpha * assent

            provisional_mass += residual.eligible_mass
            provisional_observations += int(
                residual.eligible_mass >= provisional.minimum_eligible_mass
            )
            residual_phase = "event" if phase == "event" else "familiar"
            residual_by_phase[residual_phase]["q"].append(residual.remaining_mass)
            residual_by_phase[residual_phase]["novelty"].append(residual.novelty)
            residual_by_phase[residual_phase]["eligible"].append(residual.eligible_mass)
            candidate_event = bank.observe(
                query,
                match,
                label,
                residual,
                retained_templates(memory),
            )
            if candidate_event.consolidation is not None:
                item = candidate_event.consolidation
                allocate = True
                allocation_payload = (item.key_a, item.template_n, item.value_id)

        active_write = write_mass[: len(memory.slots)]
        cross_write_mass += sum(
            float(active_write[index].item())
            for index, stored in enumerate(memory.labels)
            if stored != label
        )
        total_write_mass += float(active_write.sum().item())
        memory.write(query, match, write_mass, ContinualConfig())

        if allocate and allocation_payload is not None:
            allocation_key, allocation_template, allocation_label = allocation_payload
            memory.allocate(allocation_key, allocation_template, allocation_label)
            allocations.append(
                {
                    "phase": phase,
                    "stream_label": label,
                    "allocated_label": allocation_label,
                    "observation": observations_by_label.get(allocation_label, 0),
                    "existing_before": existing_before,
                }
            )
            if allocation_label in stream.novel_labels and allocation_label not in acquisition_delays:
                acquisition_delays[allocation_label] = observations_by_label[allocation_label]

    key_mse, template_mse = memory_purity(memory, stream)
    final_old = probe(
        attention_method,
        model,
        memory,
        stream,
        stream.base_labels,
        development.observation_noise,
        development.final_per_category,
    )
    final_new = probe(
        attention_method,
        model,
        memory,
        stream,
        stream.novel_labels,
        development.observation_noise,
        development.final_per_category,
    )
    novel_retained = len(set(memory.labels).intersection(stream.novel_labels))
    false_consolidations = len(
        {
            allocation["allocated_label"]
            for allocation in allocations
            if not scenario.should_consolidate
            and allocation["allocated_label"] in stream.novel_labels
        }
    )
    false_splits = sum(bool(allocation["existing_before"]) for allocation in allocations)

    def residual_mean(phase: str, key: str) -> Optional[float]:
        values = residual_by_phase[phase][key]
        return statistics.mean(values) if values else None

    return {
        "seed": seed,
        "scenario": scenario.name,
        "method": method_name,
        "should_consolidate": scenario.should_consolidate,
        "final_old_accuracy": final_old,
        "final_new_accuracy": final_new,
        "novel_categories_retained": novel_retained,
        "novel_categories_required": len(stream.novel_labels) if scenario.should_consolidate else 0,
        "false_consolidations": false_consolidations,
        "false_splits": false_splits,
        "allocations": len(allocations),
        "evictions": memory.evictions,
        "cross_write_mass": cross_write_mass / max(len(schedule), 1),
        "total_write_mass": total_write_mass / max(len(schedule), 1),
        "template_mse": template_mse,
        "key_mse": key_mse,
        "mean_acquisition_delay": (
            statistics.mean(acquisition_delays.values()) if acquisition_delays else None
        ),
        "provisional_mass": provisional_mass / max(len(schedule), 1),
        "provisional_observations": provisional_observations,
        "candidate_replacements": bank.replacements if bank is not None else 0,
        "candidate_consolidations": bank.consolidations if bank is not None else 0,
        "candidate_retained_match_rejections": (
            bank.retained_match_rejections if bank is not None else 0
        ),
        "candidate_active_at_end": bank.active_count if bank is not None else 0,
        "event_remaining_mass": residual_mean("event", "q"),
        "event_novelty": residual_mean("event", "novelty"),
        "event_eligible_mass": residual_mean("event", "eligible"),
        "familiar_remaining_mass": residual_mean("familiar", "q"),
        "familiar_novelty": residual_mean("familiar", "novelty"),
        "familiar_eligible_mass": residual_mean("familiar", "eligible"),
        "allocation_log": allocations,
    }


def numeric_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    keys = (
        "final_old_accuracy",
        "final_new_accuracy",
        "novel_categories_retained",
        "false_consolidations",
        "false_splits",
        "cross_write_mass",
        "template_mse",
        "mean_acquisition_delay",
        "candidate_consolidations",
    )
    result: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row[key] is not None]
        if values:
            result[key] = {
                "mean": statistics.mean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    return result


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for row in rows:
        grouped.setdefault(str(row["scenario"]), {}).setdefault(
            str(row["method"]), []
        ).append(row)
    return {
        scenario: {
            method: numeric_summary(method_rows)
            for method, method_rows in methods.items()
        }
        for scenario, methods in grouped.items()
    }


def print_summary(summary: Dict[str, object]) -> None:
    for scenario, methods_value in summary.items():
        print("\n" + scenario.upper())
        methods = methods_value
        assert isinstance(methods, dict)
        for method, metrics_value in methods.items():
            metrics = metrics_value
            assert isinstance(metrics, dict)
            old = metrics["final_old_accuracy"]["mean"]
            new = metrics["final_new_accuracy"]["mean"]
            false = metrics["false_consolidations"]["mean"]
            retained = metrics["novel_categories_retained"]["mean"]
            delay = metrics.get("mean_acquisition_delay", {}).get("mean")
            delay_text = "-" if delay is None else f"{delay:.2f}"
            print(
                f"{method:22s} old={old:.3f} new={new:.3f} "
                f"false_con={false:.2f} retained={retained:.2f} delay={delay_text}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[107, 117, 127])
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--controller-steps", type=int, default=350)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--scenarios", nargs="+", choices=[item.name for item in SCENARIOS],
        default=[item.name for item in SCENARIOS]
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phase8_consolidation/development-results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_config = TaskConfig()
    task = CategoryMatchingTask(task_config)
    provisional = selected_development_config()
    development = DevelopmentConfig()
    prepared = prepare_models(
        args.seeds,
        task,
        args.steps,
        ClosureConfig(controller_steps=args.controller_steps),
        torch.device(args.device),
    )
    selected = [item for item in SCENARIOS if item.name in args.scenarios]
    rows: List[Dict[str, object]] = []
    for seed in args.seeds:
        for scenario in selected:
            for method_name in args.methods:
                rows.append(
                    run_method(
                        method_name,
                        prepared[seed],
                        task_config,
                        scenario,
                        seed,
                        provisional,
                        development,
                    )
                )
    summary = summarize(rows)
    payload = {
        "development_seeds": args.seeds,
        "train_steps": args.steps,
        "controller_steps": args.controller_steps,
        "provisional_config": asdict(provisional),
        "development_config": asdict(development),
        "scenarios": [asdict(item) for item in selected],
        "rows": rows,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print_summary(summary)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
