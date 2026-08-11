# Delayed-context experiment: findings

## Outcome

Chevron memory learned a delayed context-action task reliably, retained old
contexts while acquiring new ones, and produced a well-calibrated unresolved
mass. It did not outperform the strongest conventional attention memory.

The useful result is narrower: within Chevron, provisional buffering and a
write gate stricter than the read gate both improved behavior under delayed
feedback. The extra per-slot residual signature did not materially improve on
scalar q in this capacity-sufficient task.

## Experiment

- Eight contexts were established first; four novel contexts appeared later.
- Each context had a randomly assigned binary action, so memory was required.
- Consequences arrived three decisions after the relevant action.
- Address and diagnostic evidence were independently noisy.
- Permanent memory had twelve slots, enough for all twelve true contexts.
- All systems received the same observations and memory capacity.
- Thresholds were checked on development seeds 0-4, then frozen.
- Confirmation used untouched seeds 100-119.

This was a fixed, online prototype-memory experiment in an RL environment. It
was not yet an end-to-end learned deep network.

## Task controls

Across 100 seeds, the memoryless condition remained at 50.1% overall accuracy.
An oracle-addressed memory that still had to learn actions from delayed reward
reached 99.3% overall and 100% final old/new accuracy. The task and eligibility
timing were therefore valid.

## Confirmation results

| Condition | Return/decision | Final old accuracy | Final new accuracy | Retention-plasticity score |
|---|---:|---:|---:|---:|
| Standard attention | 0.948 +/- 0.051 | 0.984 +/- 0.031 | 0.980 +/- 0.038 | 1.964 +/- 0.055 |
| Standard attention + buffer | 0.912 +/- 0.072 | 0.965 +/- 0.052 | 0.937 +/- 0.078 | 1.902 +/- 0.122 |
| Chevron + buffer | 0.944 +/- 0.048 | 0.991 +/- 0.008 | 0.979 +/- 0.051 | 1.971 +/- 0.054 |
| Chevron immediate write | 0.909 +/- 0.067 | 0.965 +/- 0.051 | 0.922 +/- 0.121 | 1.881 +/- 0.126 |
| Chevron scalar residual | 0.942 +/- 0.048 | 0.991 +/- 0.008 | 0.966 +/- 0.078 | 1.957 +/- 0.079 |
| Chevron coupled write | 0.890 +/- 0.083 | 0.960 +/- 0.053 | 0.939 +/- 0.082 | 1.899 +/- 0.109 |

Values are mean +/- sample standard deviation over twenty seeds.

## Paired comparisons

Approximate paired 95% confidence intervals:

- Chevron + buffer minus standard attention: return -0.0042
  [-0.0139, 0.0056]. There is no evidence of a performance advantage.
- Chevron + buffer minus standard attention + buffer: return +0.0315
  [0.0117, 0.0513]. The particular generic confirmation buffer was harmful;
  this is not evidence that buffers in general are harmful.
- Chevron + buffer minus Chevron immediate write: return +0.0353
  [0.0133, 0.0572]. Old retention improved by +0.0266
  [0.0040, 0.0493]; the new-acquisition interval still crossed zero.
- Chevron + buffer minus Chevron coupled write: return +0.0543
  [0.0256, 0.0831]. Old retention improved by +0.0317
  [0.0091, 0.0543], and new acquisition by +0.0402
  [0.0025, 0.0778].
- Full per-slot residual minus scalar residual: return +0.0021
  [-0.0020, 0.0062]. This task does not justify the added mechanism.

## Mechanism checks

For Chevron + buffer:

- mean q on the first two novel-context encounters was 0.869;
- mean q after twenty encounters fell to 0.209;
- unresolved-minus-resolved q was 0.753;
- promotion precision was 0.998;
- no permanent write occurred before feedback eligibility;
- read/residual conservation error stayed below 7e-16;
- causal unit tests showed that diagnostic changes alter assent without
  altering retrieval, and address changes alter retrieval without altering
  per-slot mismatch.

## Strongest defensible claim

In a small delayed contextual-bandit setting, an ART-inspired factorization of
attention into retrieval, assent, and unresolved mass is a viable online memory
mechanism. Its residual q tracks unresolved evidence and falls after useful
learning. Within that mechanism, provisional consolidation and stricter write
permission improve reliability relative to immediate or read-coupled writes.

The experiment does not show that Chevron outperforms standard attention. The
strongest conventional attention condition matched it. It also gives no reason
yet to prefer per-slot residual vectors over scalar q when capacity is ample.

## Next step

The next experiment should replace the fixed similarities and action readout
with small learned PyTorch components while retaining the tested online memory
rules. A matched attention policy should learn from the same delayed rewards.
Only after that should the task add capacity pressure or move back into a
spatial game, where q can drive a real information-gathering or cautious action.
