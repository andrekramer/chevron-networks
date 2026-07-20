"""Transparent provisional memory for Phase 8.

The retained attention mechanism produces two different quantities:

``q``
    Conserved mass not admitted to retained values.
``nu``
    Non-assent among the candidates actually retrieved by A attention.

Only ``u = q * nu`` is eligible for provisional memory.  A small fixed-capacity
bank then accumulates coherent evidence before returning a consolidation
payload.  The persistence state is medium-term-memory inspired, but is an
accumulating eligibility trace rather than Grossberg's canonical habituative
transmitter-depletion variable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class ProvisionalConfig:
    capacity: int = 3
    top_a_candidates: int = 3
    coherence_threshold: float = 0.75
    coherence_mismatch_threshold: float = 0.15
    coherence_sharpness: float = 30.0
    candidate_update_rate: float = 0.25
    persistence_beta: float = 0.80
    consolidation_threshold: float = 0.65
    minimum_support: int = 5
    minimum_eligible_mass: float = 0.05
    minimum_distinct_mismatch: float = 0.08
    prune_threshold: float = 0.02
    reject_retained_matches_on_entry: bool = True

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.top_a_candidates <= 0:
            raise ValueError("top_a_candidates must be positive")
        for name in (
            "coherence_threshold",
            "candidate_update_rate",
            "persistence_beta",
            "consolidation_threshold",
            "minimum_eligible_mass",
            "minimum_distinct_mismatch",
            "prune_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie between zero and one")
        if not 0.0 < self.coherence_mismatch_threshold < 1.0:
            raise ValueError("coherence_mismatch_threshold must lie between zero and one")
        if self.coherence_sharpness <= 0.0:
            raise ValueError("coherence_sharpness must be positive")
        if self.minimum_support <= 0:
            raise ValueError("minimum_support must be positive")


@dataclass(frozen=True)
class ResidualEvidence:
    remaining_mass: float
    novelty: float
    eligible_mass: float
    best_assent: float
    admitted_mass: float


def residual_evidence(
    alpha: Tensor,
    assent: Tensor,
    *,
    top_a_candidates: int = 3,
    active: Optional[Tensor] = None,
) -> ResidualEvidence:
    """Separate unused attention mass from novelty/non-assent.

    ``alpha`` and ``assent`` describe one sequential observation.  Novelty is
    measured only among the strongest A-retrieved candidates, preventing mass
    rejected from A-plausible decoys from being mistaken for novelty when one
    of those candidates assents strongly.
    """

    if alpha.dim() != 1 or assent.dim() != 1 or alpha.shape != assent.shape:
        raise ValueError("alpha and assent must be equally shaped vectors")
    if top_a_candidates <= 0:
        raise ValueError("top_a_candidates must be positive")
    if active is None:
        active = torch.ones_like(alpha, dtype=torch.bool)
    if active.shape != alpha.shape or active.dtype != torch.bool:
        raise ValueError("active must be a boolean vector matching alpha")
    indices = active.nonzero().flatten()
    if indices.numel() == 0:
        return ResidualEvidence(1.0, 1.0, 1.0, 0.0, 0.0)

    active_alpha = alpha[indices]
    active_assent = assent[indices].clamp(0.0, 1.0)
    alpha_total = active_alpha.sum().clamp_min(1e-8)
    normalized_alpha = active_alpha / alpha_total
    admitted_mass = float((normalized_alpha * active_assent).sum().clamp(0.0, 1.0).item())
    remaining_mass = min(max(1.0 - admitted_mass, 0.0), 1.0)

    count = min(top_a_candidates, indices.numel())
    top_local = normalized_alpha.topk(count).indices
    best_assent = float(active_assent[top_local].max().item())
    novelty = min(max(1.0 - best_assent, 0.0), 1.0)
    eligible_mass = remaining_mass * novelty
    return ResidualEvidence(
        remaining_mass=remaining_mass,
        novelty=novelty,
        eligible_mass=eligible_mass,
        best_assent=best_assent,
        admitted_mass=admitted_mass,
    )


def normalized_mismatch(left: Tensor, right: Tensor, epsilon: float = 1e-6) -> float:
    if left.shape != right.shape:
        raise ValueError("mismatch inputs must have the same shape")
    numerator = (left - right).abs().sum()
    denominator = (left.abs() + right.abs()).sum() + epsilon
    return float((numerator / denominator).clamp(0.0, 1.0).item())


@dataclass
class Candidate:
    key_a: Tensor
    template_n: Tensor
    value_id: int
    persistence: float
    support: int
    last_updated: int
    label_counts: Dict[int, int] = field(default_factory=dict)

    @property
    def purity(self) -> float:
        if not self.label_counts:
            return 0.0
        return max(self.label_counts.values()) / sum(self.label_counts.values())


@dataclass(frozen=True)
class Consolidation:
    key_a: Tensor
    template_n: Tensor
    value_id: int
    persistence: float
    support: int
    purity: float


@dataclass(frozen=True)
class ProvisionalEvent:
    step: int
    kind: str
    candidate_index: Optional[int]
    remaining_mass: float
    novelty: float
    eligible_mass: float
    coherence: float
    evidence: float
    persistence: float
    support: int
    value_id: int
    consolidation: Optional[Consolidation] = None


class ProvisionalMemory:
    """Fixed-state candidate bank with decay and delayed consolidation."""

    def __init__(self, config: ProvisionalConfig = ProvisionalConfig()):
        self.config = config
        self.candidates: List[Optional[Candidate]] = [None] * config.capacity
        self.step = 0
        self.events: List[ProvisionalEvent] = []
        self.replacements = 0
        self.consolidations = 0
        self.retained_match_rejections = 0

    @property
    def active_count(self) -> int:
        return sum(candidate is not None for candidate in self.candidates)

    def _coherence(self, observation: Tensor, candidate: Candidate) -> float:
        mismatch = normalized_mismatch(observation, candidate.template_n)
        z = self.config.coherence_sharpness * (
            self.config.coherence_mismatch_threshold - mismatch
        )
        return 1.0 / (1.0 + math.exp(-z))

    def _decay(self) -> None:
        for index, candidate in enumerate(self.candidates):
            if candidate is None:
                continue
            candidate.persistence *= self.config.persistence_beta
            if candidate.persistence < self.config.prune_threshold:
                self.candidates[index] = None

    def tick(self) -> None:
        """Advance time without eligible evidence."""

        self.step += 1
        self._decay()

    def _is_distinct(self, template: Tensor, retained_templates: Sequence[Tensor]) -> bool:
        if isinstance(retained_templates, Tensor):
            retained_templates = list(retained_templates)
        if not retained_templates:
            return True
        closest = min(normalized_mismatch(template, retained) for retained in retained_templates)
        return closest >= self.config.minimum_distinct_mismatch

    def observe(
        self,
        key_a: Tensor,
        template_n: Tensor,
        value_id: int,
        residual: ResidualEvidence,
        retained_templates: Sequence[Tensor] = (),
    ) -> ProvisionalEvent:
        """Route one observation through provisional state.

        Labels are recorded only for the synthetic benchmark's value readout
        and purity metric.  Candidate selection and consolidation timing never
        inspect ``value_id``.
        """

        self.step += 1
        self._decay()
        eligible = residual.eligible_mass
        if eligible < self.config.minimum_eligible_mass:
            return self._record(
                "ignored", None, residual, 0.0, 0.0, 0.0, 0, value_id
            )

        if (
            self.config.reject_retained_matches_on_entry
            and not self._is_distinct(template_n, retained_templates)
        ):
            self.retained_match_rejections += 1
            return self._record(
                "rejected_retained_match",
                None,
                residual,
                0.0,
                0.0,
                0.0,
                0,
                value_id,
            )

        coherent: List[tuple[float, int]] = []
        for index, candidate in enumerate(self.candidates):
            if candidate is not None:
                coherent.append((self._coherence(template_n, candidate), index))
        best = max(coherent, default=(-1.0, -1))

        created = best[0] < self.config.coherence_threshold
        if created:
            empty = next(
                (index for index, candidate in enumerate(self.candidates) if candidate is None),
                None,
            )
            if empty is None:
                index = min(
                    range(len(self.candidates)),
                    key=lambda item: self.candidates[item].persistence,  # type: ignore[union-attr]
                )
                self.replacements += 1
                kind = "replaced"
            else:
                index = empty
                kind = "created"
            coherence = 1.0
            evidence = eligible
            candidate = Candidate(
                key_a=key_a.detach().cpu().clone(),
                template_n=template_n.detach().cpu().clone(),
                value_id=value_id,
                persistence=(1.0 - self.config.persistence_beta) * evidence,
                support=1,
                last_updated=self.step,
                label_counts={value_id: 1},
            )
            self.candidates[index] = candidate
        else:
            coherence, index = best
            candidate = self.candidates[index]
            assert candidate is not None
            evidence = eligible * coherence
            blend = min(max(self.config.candidate_update_rate * evidence, 0.0), 1.0)
            candidate.key_a = (1.0 - blend) * candidate.key_a + blend * key_a.detach().cpu()
            candidate.template_n = (
                (1.0 - blend) * candidate.template_n + blend * template_n.detach().cpu()
            )
            candidate.persistence += (1.0 - self.config.persistence_beta) * evidence
            candidate.persistence = min(candidate.persistence, 1.0)
            candidate.support += 1
            candidate.last_updated = self.step
            candidate.label_counts[value_id] = candidate.label_counts.get(value_id, 0) + 1
            kind = "updated"

        consolidation: Optional[Consolidation] = None
        if (
            candidate.persistence >= self.config.consolidation_threshold
            and candidate.support >= self.config.minimum_support
        ):
            if self._is_distinct(candidate.template_n, retained_templates):
                consolidation = Consolidation(
                    key_a=candidate.key_a.clone(),
                    template_n=candidate.template_n.clone(),
                    value_id=candidate.value_id,
                    persistence=candidate.persistence,
                    support=candidate.support,
                    purity=candidate.purity,
                )
                self.candidates[index] = None
                self.consolidations += 1
                kind = "consolidated"
            else:
                # The candidate has enough temporal support but represents
                # structure that retained memory already contains.  Phase 8
                # tests category allocation, not revision of an existing
                # template, so release the provisional slot instead of
                # allowing a mature non-distinct candidate to occupy it.
                self.candidates[index] = None
                self.retained_match_rejections += 1
                kind = "rejected_not_distinct"

        return self._record(
            kind,
            index,
            residual,
            coherence,
            evidence,
            candidate.persistence,
            candidate.support,
            value_id,
            consolidation,
        )

    def _record(
        self,
        kind: str,
        candidate_index: Optional[int],
        residual: ResidualEvidence,
        coherence: float,
        evidence: float,
        persistence: float,
        support: int,
        value_id: int,
        consolidation: Optional[Consolidation] = None,
    ) -> ProvisionalEvent:
        event = ProvisionalEvent(
            step=self.step,
            kind=kind,
            candidate_index=candidate_index,
            remaining_mass=residual.remaining_mass,
            novelty=residual.novelty,
            eligible_mass=residual.eligible_mass,
            coherence=coherence,
            evidence=evidence,
            persistence=persistence,
            support=support,
            value_id=value_id,
            consolidation=consolidation,
        )
        self.events.append(event)
        return event
