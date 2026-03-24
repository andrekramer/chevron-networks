Implemented and tested a small RL experiment for Value/Policy Chevron models on a contextual reversal bandit with noisy cues, optional W (wait), and a delayed reversal cue. The goal
  was to check whether locally coupled proposal and control channels help an agent remember, revise after reversals, and hold back under uncertainty.

  The main result is mixed. On task reward, gated and ungated Structured Chevron did not beat strong recurrent baselines like lstm, and gating did not produce a clear performance win
  over ungated Chevron. But channel-intervention tests showed something important: in the gated Chevron model, the V channel became causally meaningful. Zeroing V increased entropy and
  wait rate and reduced reward, while in the ungated model V was almost inert. So the current evidence supports a narrower claim: gating makes the proposal/control split more real
  internally, even though it has not yet translated into better overall task performance.

# Current Results

This note summarizes the strongest result currently in the repo.

The most informative comparison so far is the 3-seed ungated vs gated Structured Chevron run on the delayed-cue task:

- `cue_prob = 0.60`
- `wait_penalty = -0.01`
- `reversal_cue_delay = 1`
- seeds: `7, 17, 27`
- training budget: `100` PPO updates

Raw outputs live in:

- [runs/compare_chevron/summary.json](/Users/andrekramer/code/python/chevron-networks/value-policy/runs/compare_chevron/summary.json)
- [runs/compare_chevron/summary.csv](/Users/andrekramer/code/python/chevron-networks/value-policy/runs/compare_chevron/summary.csv)
- [runs/compare_chevron/detail.json](/Users/andrekramer/code/python/chevron-networks/value-policy/runs/compare_chevron/detail.json)

## Headline

Gating does **not** produce a clear task-performance win.

Gating **does** make the `V` channel causally matter.

That means the best positive result so far is about internal role separation, not reward improvement.

## Variant Summary

| Metric | Ungated | Gated |
| --- | ---: | ---: |
| Sampled reward | 1.753 | 1.733 |
| Sampled post-reversal bad-action rate | 0.619 | 0.634 |
| Sampled wait rate | 0.024 | 0.024 |
| Greedy reward | 1.857 | 1.899 |
| Greedy post-reversal bad-action rate | 0.662 | 0.634 |

Interpretation:

- The two variants are very close on task reward.
- Ungated is slightly better on sampled reward and sampled post-reversal bad-action rate.
- Gated is slightly better on greedy reward and greedy post-reversal bad-action rate.
- Neither variant establishes a decisive performance advantage.

## Channel Interventions

The strongest difference appears under intervention.

### `zero_p`

| Metric delta vs base | Ungated | Gated |
| --- | ---: | ---: |
| Reward | -1.718 | -1.857 |
| Action entropy | +0.813 | +0.830 |
| Wait rate | +0.323 | +0.307 |

Interpretation:

- In both variants, `P` carries most of the action-driving signal.
- Removing `P` is catastrophic in both models.

### `zero_v`

| Metric delta vs base | Ungated | Gated |
| --- | ---: | ---: |
| Reward | -0.008 | -0.194 |
| Action entropy | +0.000 | +0.260 |
| Wait rate | +0.001 | +0.052 |
| Premature commit rate | +0.000 | -0.061 |

Interpretation:

- In the ungated model, `V` is almost inert at evaluation time.
- In the gated model, removing `V` makes the policy much noisier, more hesitant, and somewhat worse in reward.
- This is the clearest current evidence that gating makes the proposal/control decomposition more real.

### `shuffle_v`

| Metric delta vs base | Ungated | Gated |
| --- | ---: | ---: |
| Reward | +0.041 | +0.075 |
| Post-reversal bad-action rate | +0.017 | +0.003 |
| Wait rate | -0.001 | +0.002 |

Interpretation:

- Shuffling `V` has a much smaller effect than touching `P` in both models.
- The `V` effect is still secondary to the `P` effect even in the gated variant.

## Current Conclusion

The evidence so far supports a narrow claim:

- Structured Chevron with gating learns a more causally meaningful `V` channel.

The evidence does **not** currently support the stronger claim:

- Structured Chevron with gating outperforms standard recurrent baselines or ungated Chevron on task reward.

## Recommended Next Step

The next useful experiment is not a larger blind sweep. It is a more targeted causal analysis or a task variant that makes control more valuable.

Reasonable next moves:

1. Compare gated Chevron channel interventions against `lstm` hidden-state interventions using analogous perturbations.
2. Increase reversal ambiguity further, for example with false-positive or noisy reset cues.
3. Save trajectory-level examples showing where `V` suppression changes action choice.
