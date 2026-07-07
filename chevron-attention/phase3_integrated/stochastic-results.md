# Phase 3 stochastic result

Command:

```bash
.venv/bin/python -m phase3_integrated.stochastic
```

This variant randomizes the controlled key, event duration, probe duration, and
query stream. Each key has its own retained permission state. Short override
episodes should affect current behavior without changing retained policy. Long
override episodes should consolidate into retained policy. After each episode,
no-context probes test what the retained state remembers.

## Summary over seeds 7, 17, 27, 37, 47

| method | answer accuracy | short context | short probe preserve | long context | long probe consolidate | short update gate | long update gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| integrated_idl | 0.9914 ± 0.0101 | 1.0000 ± 0.0000 | 0.9785 ± 0.0295 | 1.0000 ± 0.0000 | 0.9765 ± 0.0275 | 0.0118 ± 0.0137 | 0.3951 ± 0.0105 |
| always_update | 0.9426 ± 0.0418 | 1.0000 ± 0.0000 | 0.6938 ± 0.0780 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| fixed_slow | 0.9062 ± 0.0258 | 1.0000 ± 0.0000 | 0.7906 ± 0.0538 | 1.0000 ± 0.0000 | 0.7229 ± 0.0614 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| context_only | 0.8179 ± 0.0088 | 1.0000 ± 0.0000 | 0.6612 ± 0.0580 | 1.0000 ± 0.0000 | 0.3675 ± 0.0313 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

## Interpretation

The stochastic run preserves the core Phase 3 result under randomized event
timing and randomized keys.

- `integrated_idl` handles immediate contextual behavior in both short and long
  contexts: both context accuracies are 1.0000.
- It largely preserves retained policy after short overrides:
  short-probe accuracy is 0.9785.
- It largely consolidates retained policy after long overrides:
  long-probe accuracy is 0.9765.
- Its update gate remains low during short contexts, 0.0118, and opens during
  long contexts, 0.3951.

The baselines fail in the expected ways:

- `always_update` consolidates long changes, but it overwrites retained state
  during short events, so short-probe preservation drops to 0.6938.
- `context_only` handles immediate behavior but does not retain long changes,
  so long-probe consolidation drops to 0.3675.
- `fixed_slow` is caught between the two demands: it is worse than IDL on both
  short-probe preservation and long-probe consolidation under the randomized
  schedule.

This is the strongest Phase 3 result so far. It still remains a controlled
mechanism test: contextual gates are supplied by the experiment rather than
learned from tokens, and A retrieval is algorithmic. The next technical upgrade
would be to replace the supplied contextual gate with a learned Phase-1-style N
module while keeping this stochastic retention protocol.
