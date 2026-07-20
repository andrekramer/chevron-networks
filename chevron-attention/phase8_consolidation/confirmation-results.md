# Phase 8 locked confirmation result

## Outcome

The locked confirmation produced a positive result for Chevron quarantine, but
not for every predeclared Phase 8 hypothesis.

Chevron quarantine replicated the temporal-stability, ordinary-plasticity,
near-category, and assent-ablation results on completely new seeds. The broader
hypothesis that the same provisional mechanism would preserve sufficient
plasticity for the joint controller did not replicate. No parameters were
changed and no confirmation run was repeated.

Primary conditions used 20 paired seeds. Robustness conditions used the first
10 of those seeds. All paired intervals below are deterministic bootstrap 95
percent confidence intervals with 5,000 resamples.

## Primary result

| Condition | Method | Old accuracy | New accuracy | Novel categories retained | False splits |
|---|---|---:|---:|---:|---:|
| Default | Chevron immediate | 1.000 | 0.700 | 2.10 / 3 | 1.15 |
| Default | Chevron quarantine | **1.000** | **0.967** | **2.90 / 3** | **0.00** |
| Default | Joint immediate | 0.988 | 0.483 | 1.45 / 3 | 4.40 |
| Default | Joint quarantine | 0.992 | 0.733 | 2.20 / 3 | 0.45 |
| Near | Chevron immediate | 1.000 | 0.633 | 1.90 / 3 | 0.55 |
| Near | Chevron quarantine | **0.991** | **0.967** | **2.90 / 3** | **0.00** |
| Near | Joint immediate | 0.947 | 0.033 | 0.10 / 3 | 7.40 |
| Near | Joint quarantine | 0.991 | 0.267 | 0.85 / 3 | 0.65 |

On default sustained novelty, Chevron quarantine improved new-category
accuracy over immediate Chevron by 0.267, with paired interval [0.117, 0.417].
It improved over joint quarantine by 0.234 [0.117, 0.351].

On near-category novelty, Chevron quarantine improved over immediate Chevron by
0.333 [0.183, 0.500], winning 13 seeds, tying six, and losing one. It improved
over joint quarantine by 0.700 [0.569, 0.819], winning 19 seeds and tying one.

The near-category result therefore replicated strongly. The default result was
also positive in this benchmark, although it should not be generalized beyond
the tested synthetic continual-memory regime.

## H1: temporal quarantine protects stability

The strict cross-method hypothesis failed because joint quarantine produced one
false consolidation in the ten duration-four streams.

| Duration | Chevron immediate | Chevron quarantine | Joint immediate | Joint quarantine |
|---:|---:|---:|---:|---:|
| 1 | 0.90 | **0.00** | 0.70 | **0.00** |
| 3 | 1.00 | **0.00** | 0.70 | **0.00** |
| 4 | 1.00 | **0.00** | 0.70 | 0.10 |
| 5 | 1.00 | 1.00 | 0.70 | 0.80 |
| 6 | 1.00 | 1.00 | 0.70 | 0.80 |

Chevron quarantine met the intended boundary exactly: no consolidation below
five observations and consolidation at five. Relative to immediate Chevron,
the paired false-consolidation differences were -0.90 [-1.00, -0.70] at
duration one and -1.00 [-1.00, -1.00] at durations three and four.

The lone joint failure occurred on seed 1097 during recovery. A candidate
initialized by the transient novel pattern was subsequently updated by
coherent familiar observations and crossed the support threshold with mixed
category evidence. This shows that observation count is not always equivalent
to category-coherent support. It is a valid confirmation failure and is not
corrected in Phase 8.

## H2: quarantine preserves plasticity

The predeclared cross-method hypothesis failed:

- Chevron quarantine acquired 96.7 percent of blocked categories and 93.3
  percent of interleaved categories;
- joint quarantine acquired 73.3 percent blocked and 43.3 percent interleaved,
  below the required 75 and 60 percent thresholds.

Thus temporal quarantine preserved plasticity for Chevron but not reliably for
the learned joint controller. Provisional memory is not sufficient by itself;
its usefulness depends on the quality and temporal consistency of the upstream
novelty signal.

## H3: assent remains causal

This hypothesis passed strongly. In the near condition, alpha quarantine used
the same candidate system but removed `r` from retained writes.

- Chevron quarantine old-accuracy advantage: 0.573 [0.486, 0.664], winning all
  20 seeds;
- Chevron-minus-alpha cross-category write mass: -0.718 [-0.729, -0.707], lower
  on all 20 seeds;
- template MSE was 0.00064 for Chevron quarantine versus 0.02703 for alpha
  quarantine.

Temporal consolidation controls when new structure is admitted. Slot-specific
assent remains necessary to stop ordinary writes from contaminating retained
templates.

## H4: near-category advantage replicates

This hypothesis passed, with positive paired intervals relative to both
immediate Chevron and joint quarantine. Chevron quarantine learned 2.9 of three
near categories on average while preserving 0.991 old accuracy.

The result supports Chevron as a specific inductive bias for stability and
plasticity when new categories closely resemble retained categories.

## Robustness

| Condition | Chevron quarantine old/new | Joint quarantine old/new |
|---|---:|---:|
| Interleaved | 1.000 / 0.933 | 0.983 / 0.431 |
| Noise 0.06 | 1.000 / 0.867 | 0.983 / 0.427 |
| Noise 0.08 | 0.999 / 0.767 | 0.984 / 0.491 |
| Noise 0.10 | 0.997 / 0.598 | 0.985 / 0.333 |
| Three flipped components | 1.000 / 0.867 | 0.983 / 0.667 |
| Retained capacity 8 | 0.983 / 0.700 | 0.950 / 0.433 |
| Long, 16 observations | 1.000 / 1.000 | 1.000 / 0.500 |
| Two candidate slots, three interleaved categories | 1.000 / 0.233 | 0.969 / 0.167 |

Chevron quarantine protected old memory as noise increased, but plasticity
declined. The candidate-capacity result is also explicit: a two-slot bank
cannot reliably preserve three concurrently interleaved candidates.

## Strongest supported claim

The confirmation supports the following claim:

> In the locked synthetic continual-category benchmark, a five-observation
> provisional consolidation layer combined with learned slot-specific Chevron
> assent sharply reduced transient allocation and retained-memory contamination
> while improving acquisition of both ordinary and near-category novelty.
> The near-category advantage replicated across 20 new paired seeds. The same
> provisional layer did not preserve sufficient plasticity for the learned
> joint-attention controller, so the result supports the combined Chevron
> mechanism rather than provisional memory as a universally effective add-on.

## Limits

The evidence does not establish universal superiority over standard attention,
biological equivalence to ART or Grossberg MTM, realistic language-model
performance, agency, consciousness, or safety. Categories, values, and stream
structure remain synthetic and partially supervised.

The current synthetic programme should now stop tuning, as predeclared. The
next research programme should test the frozen mechanism on less synthetic
episodic or persistent-agent memory, including legitimate belief revision,
provenance, rollback, and adversarial evidence.

Raw rows, paired comparisons, confidence intervals, and per-seed differences
are stored in `confirmation-results.json`. The unmodified protocol is in
`confirmation-protocol.md`.

