from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(right @ unit(left), dtype=np.float64)


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / temperature
    scaled = scaled - np.max(scaled)
    weights = np.exp(scaled)
    return weights / weights.sum()


def sigmoid(value: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class MemoryConfig:
    capacity: int = 12
    buffer_capacity: int = 12
    promotion_support: int = 2
    temperature: float = 0.12
    standard_match_similarity: float = 0.65
    candidate_match_similarity: float = 0.65
    theta_read: float = 0.22
    theta_write: float = 0.12
    gate_slope: float = 20.0
    allocation_q: float = 0.45
    allocation_max_assent: float = 0.55
    permanent_rate: float = 0.2
    exploration: float = 0.02
    action_confidence: float = 0.08

    def __post_init__(self) -> None:
        if not 0.0 <= self.exploration <= 1.0:
            raise ValueError("exploration must be in [0, 1]")
        if not self.theta_write < self.theta_read:
            raise ValueError("theta_write must be stricter (smaller) than theta_read")
        if self.buffer_capacity < 1:
            raise ValueError("buffer_capacity must be positive")


@dataclass
class Slot:
    address: np.ndarray
    diagnostic: np.ndarray
    value: float
    uses: int = 0
    established_old: bool = False
    audit_context_id: int | None = None
    promotion_id: int | None = None


@dataclass
class Candidate:
    address: np.ndarray
    diagnostic: np.ndarray
    value_sum: float
    support: int
    last_seen: int
    residual_signature: np.ndarray
    audit_context_counts: dict[int, int] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return float(np.clip(self.value_sum / max(1, self.support), -1.0, 1.0))

    @property
    def audit_context_id(self) -> int | None:
        if not self.audit_context_counts:
            return None
        return max(self.audit_context_counts, key=self.audit_context_counts.get)


@dataclass
class ReadTrace:
    z: float
    q: float
    admitted_mass: float
    max_assent: float
    conservation_error: float
    selected_slot: int | None
    allocation_trigger: bool
    mismatch: np.ndarray
    alpha: np.ndarray
    assent: np.ndarray
    residual: np.ndarray


@dataclass
class PendingDecision:
    decision_index: int
    address: np.ndarray
    diagnostic: np.ndarray
    action: int
    audit_context_id: int
    trace: ReadTrace
    immediate_slot: int | None = None


@dataclass
class Promotion:
    promotion_id: int
    audit_context_id: int | None
    later_useful: bool = False


class MemoryPolicy:
    def __init__(self, address_dim: int, diagnostic_dim: int, seed: int, config: MemoryConfig) -> None:
        self.address_dim = address_dim
        self.diagnostic_dim = diagnostic_dim
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.slots: list[Slot] = []
        self.candidates: list[Candidate] = []
        self.pending: list[PendingDecision] = []
        self.promotions: list[Promotion] = []
        self.premature_writes = 0
        self.established_overwrites = 0
        self.permanent_writes = 0
        self.read_admissions = 0
        self.incompatible_writes = 0
        self.max_conservation_error = 0.0

    def read(self, address: np.ndarray, diagnostic: np.ndarray) -> ReadTrace:
        raise NotImplementedError

    def act(
        self,
        address: np.ndarray,
        diagnostic: np.ndarray,
        decision_index: int,
        audit_context_id: int,
    ) -> tuple[int, ReadTrace]:
        trace = self.read(address, diagnostic)
        if self.rng.random() < self.config.exploration or abs(trace.z) < self.config.action_confidence:
            action = int(self.rng.integers(0, 2))
        else:
            action = int(trace.z > 0.0)
        pending = PendingDecision(
            decision_index=decision_index,
            address=address.copy(),
            diagnostic=diagnostic.copy(),
            action=action,
            audit_context_id=audit_context_id,
            trace=trace,
        )
        self._before_outcome(pending)
        self.pending.append(pending)
        self.max_conservation_error = max(self.max_conservation_error, trace.conservation_error)
        if trace.admitted_mass >= 0.5:
            self.read_admissions += 1
        return action, trace

    def observe(self, reward: float) -> PendingDecision:
        if not self.pending:
            raise RuntimeError("outcome arrived with no eligible decision")
        pending = self.pending.pop(0)
        target = 1.0 if (pending.action == 1) == (reward > 0.0) else -1.0
        self._eligible_write(pending, target)
        return pending

    def note_use(self, trace: ReadTrace, audit_context_id: int, correct: bool) -> None:
        if trace.selected_slot is None or not correct:
            return
        slot = self.slots[trace.selected_slot]
        if slot.promotion_id is None or slot.audit_context_id != audit_context_id:
            return
        self.promotions[slot.promotion_id].later_useful = True

    @property
    def promotion_precision(self) -> float:
        if not self.promotions:
            return float("nan")
        return float(np.mean([promotion.later_useful for promotion in self.promotions]))

    def _before_outcome(self, pending: PendingDecision) -> None:
        del pending

    def _eligible_write(self, pending: PendingDecision, target: float) -> None:
        raise NotImplementedError

    def _update_slot(
        self,
        index: int,
        address: np.ndarray,
        diagnostic: np.ndarray,
        target: float,
        gate: float,
    ) -> None:
        gate = float(np.clip(gate, 0.0, 1.0))
        rate = float(np.clip(self.config.permanent_rate * gate, 0.0, 1.0))
        slot = self.slots[index]
        slot.address = unit((1.0 - rate) * slot.address + rate * address)
        slot.diagnostic = unit((1.0 - rate) * slot.diagnostic + rate * diagnostic)
        slot.value = float(np.clip((1.0 - rate) * slot.value + rate * target, -1.0, 1.0))
        slot.uses += 1
        self.permanent_writes += 1

    def _allocate_slot(
        self,
        address: np.ndarray,
        diagnostic: np.ndarray,
        target: float,
        audit_context_id: int | None,
        *,
        promoted: bool,
    ) -> int:
        promotion_id = None
        if promoted:
            promotion_id = len(self.promotions)
            self.promotions.append(Promotion(promotion_id, audit_context_id))
        new_slot = Slot(
            address=unit(address.copy()),
            diagnostic=unit(diagnostic.copy()),
            value=float(np.clip(target, -1.0, 1.0)),
            uses=1,
            established_old=audit_context_id is not None and audit_context_id < 8,
            audit_context_id=audit_context_id,
            promotion_id=promotion_id,
        )
        if len(self.slots) < self.config.capacity:
            self.slots.append(new_slot)
            index = len(self.slots) - 1
        else:
            replaceable = [i for i, slot in enumerate(self.slots) if not slot.established_old]
            pool = replaceable or list(range(len(self.slots)))
            index = min(pool, key=lambda i: self.slots[i].uses)
            if self.slots[index].established_old:
                self.established_overwrites += 1
            self.slots[index] = new_slot
        self.permanent_writes += 1
        return index

    def _combined_similarity(self, address: np.ndarray, diagnostic: np.ndarray, slot: Slot) -> float:
        return 0.5 * float(address @ slot.address) + 0.5 * float(diagnostic @ slot.diagnostic)

    def _candidate_similarity(
        self,
        address: np.ndarray,
        diagnostic: np.ndarray,
        candidate: Candidate,
        residual: np.ndarray | None = None,
    ) -> float:
        del residual
        return 0.5 * float(address @ candidate.address) + 0.5 * float(diagnostic @ candidate.diagnostic)

    def _padded_residual(self, residual: np.ndarray) -> np.ndarray:
        padded = np.zeros(self.config.capacity, dtype=np.float64)
        padded[: min(len(residual), self.config.capacity)] = residual[: self.config.capacity]
        return padded

    def _add_candidate(
        self,
        pending: PendingDecision,
        target: float,
        promote_callback,
    ) -> None:
        matches = [
            self._candidate_similarity(
                pending.address,
                pending.diagnostic,
                candidate,
                pending.trace.residual,
            )
            for candidate in self.candidates
        ]
        if matches and max(matches) >= self.config.candidate_match_similarity:
            index = int(np.argmax(matches))
            candidate = self.candidates[index]
            support = candidate.support + 1
            candidate.address = unit((candidate.address * candidate.support + pending.address) / support)
            candidate.diagnostic = unit((candidate.diagnostic * candidate.support + pending.diagnostic) / support)
            candidate.value_sum += target
            candidate.support = support
            candidate.last_seen = pending.decision_index
            candidate.residual_signature = (
                candidate.residual_signature * (support - 1)
                + self._padded_residual(pending.trace.residual)
            ) / support
            candidate.audit_context_counts[pending.audit_context_id] = (
                candidate.audit_context_counts.get(pending.audit_context_id, 0) + 1
            )
        else:
            if len(self.candidates) >= self.config.buffer_capacity:
                evict = min(range(len(self.candidates)), key=lambda i: self.candidates[i].last_seen)
                self.candidates.pop(evict)
            candidate = Candidate(
                address=unit(pending.address.copy()),
                diagnostic=unit(pending.diagnostic.copy()),
                value_sum=target,
                support=1,
                last_seen=pending.decision_index,
                residual_signature=self._padded_residual(pending.trace.residual),
                audit_context_counts={pending.audit_context_id: 1},
            )
            self.candidates.append(candidate)
            index = len(self.candidates) - 1

        candidate = self.candidates[index]
        if candidate.support >= self.config.promotion_support:
            promote_callback(candidate)
            self.candidates.pop(index)


class StandardAttentionMemory(MemoryPolicy):
    def __init__(
        self,
        address_dim: int,
        diagnostic_dim: int,
        seed: int,
        config: MemoryConfig,
        *,
        use_buffer: bool,
    ) -> None:
        super().__init__(address_dim, diagnostic_dim, seed, config)
        self.use_buffer = use_buffer

    def read(self, address: np.ndarray, diagnostic: np.ndarray) -> ReadTrace:
        if not self.slots:
            return ReadTrace(
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                None,
                True,
                np.empty(0),
                np.empty(0),
                np.empty(0),
                np.empty(0),
            )
        similarities = np.asarray(
            [self._combined_similarity(address, diagnostic, slot) for slot in self.slots]
        )
        alpha = softmax(similarities, self.config.temperature)
        values = np.asarray([slot.value for slot in self.slots])
        z = float(alpha @ values)
        selected = int(np.argmax(alpha))
        self.slots[selected].uses += 1
        # A conventional confidence diagnostic, not a Chevron residual.
        q = float(np.clip(1.0 - max(similarities[selected], 0.0), 0.0, 1.0))
        return ReadTrace(
            z=z,
            q=q,
            admitted_mass=1.0,
            max_assent=1.0,
            conservation_error=0.0,
            selected_slot=selected,
            allocation_trigger=similarities[selected] < self.config.standard_match_similarity,
            mismatch=1.0 - similarities,
            alpha=alpha,
            assent=np.ones_like(alpha),
            residual=np.zeros_like(alpha),
        )

    def _eligible_write(self, pending: PendingDecision, target: float) -> None:
        if self.use_buffer and pending.trace.allocation_trigger:
            self._add_candidate(pending, target, self._promote)
            return
        self._assign_or_update(
            pending.address,
            pending.diagnostic,
            target,
            pending.audit_context_id,
            promoted=False,
        )

    def _promote(self, candidate: Candidate) -> None:
        self._assign_or_update(
            candidate.address,
            candidate.diagnostic,
            candidate.value,
            candidate.audit_context_id,
            promoted=True,
        )

    def _assign_or_update(
        self,
        address: np.ndarray,
        diagnostic: np.ndarray,
        target: float,
        audit_context_id: int | None,
        *,
        promoted: bool,
    ) -> int:
        if self.slots:
            similarities = [self._combined_similarity(address, diagnostic, slot) for slot in self.slots]
            best = int(np.argmax(similarities))
            if similarities[best] >= self.config.standard_match_similarity:
                self._update_slot(best, address, diagnostic, target, gate=1.0)
                return best
        return self._allocate_slot(
            address,
            diagnostic,
            target,
            audit_context_id,
            promoted=promoted,
        )


class ChevronMemory(MemoryPolicy):
    def __init__(
        self,
        address_dim: int,
        diagnostic_dim: int,
        seed: int,
        config: MemoryConfig,
        *,
        use_buffer: bool,
        immediate_write: bool = False,
        per_slot_residual: bool = True,
        coupled_write: bool = False,
    ) -> None:
        super().__init__(address_dim, diagnostic_dim, seed, config)
        self.use_buffer = use_buffer
        self.immediate_write = immediate_write
        self.per_slot_residual = per_slot_residual
        self.coupled_write = coupled_write
        if immediate_write and use_buffer:
            raise ValueError("immediate_write and use_buffer are mutually exclusive")

    def read(self, address: np.ndarray, diagnostic: np.ndarray) -> ReadTrace:
        if not self.slots:
            return ReadTrace(
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                None,
                True,
                np.empty(0),
                np.empty(0),
                np.empty(0),
                np.empty(0),
            )
        address_similarity = cosine(address, np.stack([slot.address for slot in self.slots]))
        alpha = softmax(address_similarity, self.config.temperature)
        diagnostic_similarity = cosine(diagnostic, np.stack([slot.diagnostic for slot in self.slots]))
        mismatch = np.clip((1.0 - diagnostic_similarity) / 2.0, 0.0, 1.0)
        assent = sigmoid(self.config.gate_slope * (self.config.theta_read - mismatch))
        admitted = alpha * assent
        residual = alpha * (1.0 - assent)
        admitted_mass = float(admitted.sum())
        q = float(residual.sum())
        conservation_error = abs(admitted_mass + q - 1.0)
        values = np.asarray([slot.value for slot in self.slots])
        z = float(admitted @ values)
        selected = int(np.argmax(admitted))
        self.slots[selected].uses += 1
        max_assent = float(np.max(assent))
        allocation = q > self.config.allocation_q and max_assent < self.config.allocation_max_assent
        return ReadTrace(
            z=z,
            q=q,
            admitted_mass=admitted_mass,
            max_assent=max_assent,
            conservation_error=conservation_error,
            selected_slot=selected,
            allocation_trigger=allocation,
            mismatch=mismatch,
            alpha=alpha,
            assent=assent,
            residual=residual,
        )

    def _candidate_similarity(
        self,
        address: np.ndarray,
        diagnostic: np.ndarray,
        candidate: Candidate,
        residual: np.ndarray | None = None,
    ) -> float:
        base = super()._candidate_similarity(address, diagnostic, candidate)
        if not self.per_slot_residual or residual is None:
            return base
        padded = self._padded_residual(residual)
        if np.linalg.norm(padded) < 1e-8 or np.linalg.norm(candidate.residual_signature) < 1e-8:
            return base
        residual_similarity = float(unit(padded) @ unit(candidate.residual_signature))
        return 0.8 * base + 0.2 * residual_similarity

    def _before_outcome(self, pending: PendingDecision) -> None:
        if not self.immediate_write or not pending.trace.allocation_trigger:
            return
        pending.immediate_slot = self._allocate_slot(
            pending.address,
            pending.diagnostic,
            target=0.0,
            audit_context_id=pending.audit_context_id,
            promoted=False,
        )
        self.premature_writes += 1

    def _eligible_write(self, pending: PendingDecision, target: float) -> None:
        if pending.immediate_slot is not None:
            self._update_slot(
                pending.immediate_slot,
                pending.address,
                pending.diagnostic,
                target,
                gate=1.0,
            )
            return

        if not self.slots or pending.trace.allocation_trigger:
            if self.use_buffer:
                self._add_candidate(pending, target, self._promote)
            else:
                self._allocate_slot(
                    pending.address,
                    pending.diagnostic,
                    target,
                    pending.audit_context_id,
                    promoted=False,
                )
            return

        selected = pending.trace.selected_slot
        if selected is None or selected >= len(self.slots):
            return
        mismatch = float(pending.trace.mismatch[selected])
        theta_write = self.config.theta_read if self.coupled_write else self.config.theta_write
        write_gate = float(sigmoid(self.config.gate_slope * (theta_write - mismatch)))
        if write_gate >= 0.5:
            self._update_slot(selected, pending.address, pending.diagnostic, target, write_gate)
        elif self.use_buffer:
            if pending.trace.admitted_mass >= 0.5:
                self.incompatible_writes += 1
            self._add_candidate(pending, target, self._promote)

    def _promote(self, candidate: Candidate) -> None:
        if self.slots:
            address_similarity = cosine(candidate.address, np.stack([slot.address for slot in self.slots]))
            diagnostic_similarity = cosine(
                candidate.diagnostic,
                np.stack([slot.diagnostic for slot in self.slots]),
            )
            mismatch = np.clip((1.0 - diagnostic_similarity) / 2.0, 0.0, 1.0)
            alpha = softmax(address_similarity, self.config.temperature)
            selected = int(np.argmax(alpha))
            theta_write = self.config.theta_read if self.coupled_write else self.config.theta_write
            write_gate = float(sigmoid(self.config.gate_slope * (theta_write - mismatch[selected])))
            if write_gate >= 0.5:
                self._update_slot(
                    selected,
                    candidate.address,
                    candidate.diagnostic,
                    candidate.value,
                    write_gate,
                )
                return
        self._allocate_slot(
            candidate.address,
            candidate.diagnostic,
            candidate.value,
            candidate.audit_context_id,
            promoted=True,
        )
