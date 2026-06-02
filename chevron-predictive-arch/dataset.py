from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset


REGIME_SETS = {
    "easy": ("copy", "not", "xor2", "majority3"),
    "lagged": ("copy_lag8", "not_lag8", "xor_lag4_8", "majority_lag1_4_8"),
}


@dataclass(frozen=True)
class SequenceData:
    bits: torch.Tensor
    regimes: torch.Tensor
    switches: torch.Tensor
    distractors: torch.Tensor
    noise: torch.Tensor
    sequence_id: int


def _rule_next(regime: str, bits: list[int]) -> int:
    x0 = bits[-1]
    x1 = bits[-2]
    x2 = bits[-3]
    if regime == "copy":
        return x0
    if regime == "not":
        return 1 - x0
    if regime == "xor2":
        return x0 ^ x1
    if regime == "majority3":
        return 1 if (x0 + x1 + x2) >= 2 else 0
    if regime == "copy_lag8":
        return bits[-8]
    if regime == "not_lag8":
        return 1 - bits[-8]
    if regime == "xor_lag4_8":
        return bits[-4] ^ bits[-8]
    if regime == "majority_lag1_4_8":
        return 1 if (bits[-1] + bits[-4] + bits[-8]) >= 2 else 0
    raise ValueError(f"unknown regime: {regime}")


def generate_sequence(
    length: int,
    *,
    sequence_id: int,
    p_noise: float,
    use_distractors: bool,
    distractor_prob: float,
    min_regime: int,
    max_regime: int,
    min_distractor: int,
    max_distractor: int,
    regime_names: tuple[str, ...],
    seed: int,
) -> SequenceData:
    if length < 8:
        raise ValueError("length must be at least 8")
    if min_regime <= 0 or max_regime < min_regime:
        raise ValueError("invalid regime duration range")

    gen = torch.Generator().manual_seed(seed)
    warmup = 8 if any("lag8" in name or "_8" in name for name in regime_names) else 3
    bits = [int(torch.randint(0, 2, (), generator=gen)) for _ in range(warmup)]
    regime_ids = [-1 for _ in range(length)]
    switches = [False for _ in range(length)]
    distractors = [False for _ in range(length)]
    noise = [False for _ in range(length)]

    regime_idx = int(torch.randint(0, len(regime_names), (), generator=gen))
    remaining = int(torch.randint(min_regime, max_regime + 1, (), generator=gen))
    distractor_remaining = 0

    for t in range(warmup, length):
        switched = False
        if remaining <= 0:
            old = regime_idx
            while regime_idx == old:
                regime_idx = int(torch.randint(0, len(regime_names), (), generator=gen))
            remaining = int(torch.randint(min_regime, max_regime + 1, (), generator=gen))
            switched = True

        if use_distractors and distractor_remaining <= 0:
            if float(torch.rand((), generator=gen)) < distractor_prob:
                distractor_remaining = int(
                    torch.randint(min_distractor, max_distractor + 1, (), generator=gen)
                )

        if distractor_remaining > 0:
            next_bit = int(torch.randint(0, 2, (), generator=gen))
            distractors[t] = True
            distractor_remaining -= 1
        else:
            next_bit = _rule_next(regime_names[regime_idx], bits)

        if float(torch.rand((), generator=gen)) < p_noise:
            next_bit = 1 - next_bit
            noise[t] = True

        bits.append(next_bit)
        regime_ids[t] = regime_idx
        switches[t] = switched
        remaining -= 1

    return SequenceData(
        bits=torch.tensor(bits, dtype=torch.long),
        regimes=torch.tensor(regime_ids, dtype=torch.long),
        switches=torch.tensor(switches, dtype=torch.bool),
        distractors=torch.tensor(distractors, dtype=torch.bool),
        noise=torch.tensor(noise, dtype=torch.bool),
        sequence_id=sequence_id,
    )


class RegimeSequenceDataset(Dataset):
    def __init__(
        self,
        *,
        num_sequences: int,
        sequence_length: int,
        context_length: int,
        p_noise: float = 0.0,
        use_distractors: bool = False,
        distractor_prob: float = 0.002,
        min_regime: int = 50,
        max_regime: int = 200,
        min_distractor: int = 5,
        max_distractor: int = 10,
        regime_set: str = "easy",
        seed: int = 0,
    ) -> None:
        if context_length < 3:
            raise ValueError("context_length must be at least 3")
        if sequence_length <= context_length + 1:
            raise ValueError("sequence_length must exceed context_length + 1")
        if regime_set not in REGIME_SETS:
            raise ValueError(f"unknown regime_set {regime_set!r}; choose one of {sorted(REGIME_SETS)}")
        if regime_set == "lagged" and context_length < 8:
            raise ValueError("lagged regime_set requires context_length >= 8")
        self.context_length = context_length
        self.regime_set = regime_set
        self.sequences = [
            generate_sequence(
                sequence_length,
                sequence_id=i,
                p_noise=p_noise,
                use_distractors=use_distractors,
                distractor_prob=distractor_prob,
                min_regime=min_regime,
                max_regime=max_regime,
                min_distractor=min_distractor,
                max_distractor=max_distractor,
                regime_names=REGIME_SETS[regime_set],
                seed=seed + i * 104729,
            )
            for i in range(num_sequences)
        ]
        self.index: list[tuple[int, int]] = []
        for seq_idx, seq in enumerate(self.sequences):
            for target_index in range(context_length, len(seq.bits)):
                self.index.append((seq_idx, target_index))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_idx, target_index = self.index[idx]
        seq = self.sequences[seq_idx]
        start = target_index - self.context_length
        context = seq.bits[start:target_index].float()
        return {
            "context": context,
            "target": seq.bits[target_index],
            "sequence_id": torch.tensor(seq.sequence_id, dtype=torch.long),
            "target_index": torch.tensor(target_index, dtype=torch.long),
            "regime": seq.regimes[target_index],
            "switch": seq.switches[target_index],
            "distractor": seq.distractors[target_index],
            "noise": seq.noise[target_index],
        }

    def sequence_ids(self) -> Iterable[int]:
        return (seq.sequence_id for seq in self.sequences)
