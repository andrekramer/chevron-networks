# Phase 6 initial results: persistent category creation

Five seeded streams (`7`, `17`, `27`, `37`, and `47`) were run with the same
phase structure and seed-dependent prototypes/noise. Every method predicted
before seeing the label. The recovery and final probes were read-only.

The task first established three base categories, then presented eight
different coherent categories for five observations each. It finally presented
one new category for forty observations. A four-slot retained memory is exactly
large enough for the three base categories and the persistent new category.

## Behavioral result

| Method | Brief-category online accuracy | Old accuracy after brief categories | Persistent-new online accuracy | Final old accuracy | Final new accuracy | Final overall |
|---|---:|---:|---:|---:|---:|---:|
| Chevron ART-like | 0.8000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9750 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Persistent single-template attention | 0.8000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9750 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Immediate-write standard attention | 0.8000 ± 0.0000 | 0.3333 ± 0.0000 | 0.9750 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.2500 ± 0.0000 |
| Plain online MLP, no replay | 0.0100 ± 0.0137 | 0.5771 ± 0.3915 | 0.8300 ± 0.0209 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.2500 ± 0.0000 |

Both persistence-gated methods missed only the first prediction of each novel
block. Their fast candidate then supplied the current category label. The five
observations in a brief block raised persistence to `0.6723`, below the `0.75`
creation threshold. The persistent block crossed the threshold on its seventh
observation and became a retained category.

## Retained state

| Method | Base categories retained after brief blocks | Brief categories retained | Persistent category retained | Creations | Evictions |
|---|---:|---:|---:|---:|---:|
| Chevron ART-like | 3 | 0 | 1 | 4 | 0 |
| Persistent single-template attention | 3 | 0 | 1 | 4 | 0 |
| Immediate-write standard attention | 1 | 8 | 1 | 12 | 3 |

The immediate-write attention baseline received nine single-vector slots. This
matches the approximate feature-state budget of four Chevron slots, each with
an A key and N template, plus one fast A candidate. Immediate writes filled
memory with all eight brief categories and displaced the three base categories
over the remainder of the stream.

## What the result supports

The experiment validates the Phase 6 control flow:

```text
A-key attention -> N-template match -> vigilance veto -> search
                                              |
                                      no resonant category
                                              |
                                  fast coherent candidate
                                              |
                                  persistent -> create N
```

An explicit search test also confirms that vigilance can veto the
highest-attention slot and continue to a later resonant template.

The strongest claim from this first run is narrow: on this controlled
fixed-budget stream, persistence-gated category creation retains recurring
categories while reacting immediately to brief novel categories and
consolidating a sustained new one. Immediate-write attention and a no-replay
online MLP do not achieve that combination.

## The important negative result

The single-template persistent-attention ablation ties Chevron on every metric.
It uses the same vigilance, search, fast candidate, and creation threshold, but
has no distinct fast A key and slow N template.

Therefore this experiment does **not** yet show a Chevron-specific advantage.
It shows that persistent category allocation is sufficient for this task. The
A/N split will need a task in which a fast address must move while a retained
template or value must remain stable—for example gradual within-category drift,
recurring context-dependent boundaries, or noisy top-down expectations.

The MLP comparison is also preliminary. It is a deliberately plain online
learner with a fixed vocabulary and no replay; a replay-equipped or regularized
MLP is required before making a broader continual-learning comparison.

## Next experiments

1. Sweep brief duration, category separation, noise, vigilance, and memory
   capacity to locate the stability/plasticity frontier.
2. Add gradual category drift and recurrence so A-key adaptation can be tested
   separately from N-template retention.
3. Replace supervised allocation with label-free category formation, then add
   Fuzzy ART as the specialist reference baseline.
4. Add trainable Q/K projections and compare a replay-equipped MLP under a
   stated memory and update-compute budget.

