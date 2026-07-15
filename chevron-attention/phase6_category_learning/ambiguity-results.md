# Phase 6.2 results: contextual ambiguity

This experiment gives ART-style search a genuine alternative category. In each
episode, every bottom-up decoy is more similar to the A query than the target.
Only the target's N template matches the top-down context. All decoys carry the
same wrong value, so their combined softmax mass becomes consequential.

Five seeds were evaluated on 100 fresh episodes at each of five distractor
counts, for 500 episodes per condition. Memories and slot order were regenerated
for every episode.

## Accuracy

| Method | 1 decoy | 3 decoys | 7 decoys | 15 decoys | 31 decoys |
|---|---:|---:|---:|---:|---:|
| A-only softmax attention | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Joint A/N softmax attention | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9160 ± 0.0498 |
| Joint A/N top-1 retrieval | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Complementarity-only gate | 0.0040 ± 0.0055 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Soft vigilance, sharpness 80 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9120 ± 0.0449 |
| Soft vigilance, sharpness 160 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Vigilance-masked attention | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Hard ordered reset/search | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

At 31 decoys, the target's mean A rank was exactly 32. Hard search therefore
performed 31 resets before selecting it. Mean target N mismatch was about
`0.012`, while decoy mismatch was about `0.111`, on opposite sides of the
`0.060` vigilance threshold.

## What this establishes

The test validates the intended ART-like operation directly:

```text
strongest bottom-up candidate -> N mismatch -> reset
next candidate                -> N mismatch -> reset
...
32nd candidate                -> N match    -> resonate
```

Hard search is robust to correlated distractor multiplicity in this regime.
Ordinary softmax value averaging eventually loses because many individually
weak wrong values accumulate enough mass to outvote the target.

But ordered search is not uniquely necessary. Joint top-1 retrieval, a hard
vigilance mask, and sufficiently sharp differentiable vigilance all match its
100% accuracy. The useful ingredient is strong conditional filtering by the N
match, not the procedural search loop itself.

## Complementarity is not vigilance

The complementarity-only gate fails even with one decoy. This confirms the
mathematical correction made at the start of Phase 6:

```text
G = 2 sqrt(p(1-p))
```

is maximal when A and N agree and remains fairly high for a localized template
conflict. Target complementarity averaged `0.9997`, but the deliberately
mismatched decoys still averaged `0.9496`. It measures reciprocal engagement,
not mismatch. It cannot implement the reset condition `G > threshold` proposed
initially.

Vigilance should instead operate on normalized absolute contrast:

```text
M = sum(E * abs(C)) / sum(E)
```

with `G` optionally retained as a resonance or transformability signal.

## Strongest claim

> In a controlled associative-memory task where bottom-up similarity is
> systematically misleading, normalized A/N template mismatch can prevent
> correlated distractors from accumulating value mass. Hard ART-like search is
> one successful implementation, but top-1 joint retrieval and sharply gated
> standard attention are equally successful.

This supports A/N conditional matching. It still does not establish a general
Chevron advantage over standard attention, because the standard baselines can
express the same constraint compactly.
