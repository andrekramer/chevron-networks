# Chevron Predictive Architecture: Experiment Trajectory

This note summarizes the experimental trajectory so far for the minimal Chevron Predictive Architecture (CPA) sequence-prediction benchmark.

The short version:

> A naive CPA did not beat simple baselines on the harder task. But after diagnosis, adding mismatch-driven N updates, increasing capacity, and training longer produced a CPA variant that reliably beats the original h64/5-epoch MLP baseline and reaches near 5-epoch Transformer parity on the current synthetic lagged/distractor task. A larger h128/10-epoch MLP, however, beats the h128/10-epoch CPA again.

## Setup

The benchmark is binary next-bit prediction from a fixed context window.

The harder task uses hidden lagged regimes:

- `copy_lag8`
- `not_lag8`
- `xor_lag4_8`
- `majority_lag1_4_8`

The active regime switches after random durations. The sequence also includes long distractor bursts where the rule output is replaced with random bits.

Models:

- MLP baseline
- Tiny Transformer baseline
- CPA with fast `A` and slow `N` states

Main metrics:

- overall accuracy
- post-switch recovery time
- distractor recovery time
- post-distractor accuracy

Lower recovery time is better.

## 1. Easy Task: Promising but Too Easy

On the initial easy task, all models reached very high accuracy:

| Model | Accuracy | Switch Recovery | Distractor Recovery | Post-Distractor Acc |
|---|---:|---:|---:|---:|
| MLP | 0.9830 | 0.1986 | 3.6667 | 0.8858 |
| Transformer | 0.9853 | 0.1849 | 3.0000 | 0.9180 |
| CPA | 0.9850 | 0.1507 | 2.6333 | 0.9348 |

CPA looked best on the stress metrics, but the task was too simple. Accuracy was near ceiling and recovery was almost immediate.

Interpretation:

> This was a useful sanity check, not evidence that CPA was better.

## 2. Harder Lagged Task: Naive CPA Failed

The lagged task made the local rule harder to infer and lengthened distractor bursts.

Single-seed 5-epoch result:

| Model | Accuracy | Switch Recovery | Distractor Recovery | Post-Distractor Acc |
|---|---:|---:|---:|---:|
| MLP | 0.8631 | 9.8889 | 35.0244 | 0.6738 |
| Transformer | 0.8767 | 7.0556 | 33.2195 | 0.6928 |
| CPA naive | 0.8165 | 15.3094 | 43.3718 | 0.6329 |

This was an important negative result. CPA did not merely need a harder benchmark to shine; the first minimal implementation was worse than both baselines.

Interpretation:

> The architecture was not automatically useful. The failure was diagnostic.

## 3. Diagnostic Sweep: A/N Coordination Mattered

Sweeping CPA time scale, band regularization, and detach behavior showed that the original `A.detach()` path was too restrictive.

Best compact sweep setting:

```text
rho = 0.10
lambda_band = 0.001
detach_a_to_n = False
```

That improved CPA, but not enough to beat the baselines.

Interpretation:

> A and N need enough gradient coordination to solve the lagged task. Preventing N updates from shaping A weakened the model.

## 4. Mismatch-Driven N Update Helped

The next CPA-specific change added explicit A/N mismatch into the N update:

```python
N_update += W_diffN(A - N)
```

Single-seed 10-epoch comparison:

| Model | Accuracy | Switch Recovery | Distractor Recovery | Post-Distractor Acc |
|---|---:|---:|---:|---:|
| CPA tuned | 0.8587 | 9.5694 | 37.3210 | 0.6363 |
| CPA diff | 0.8648 | 8.6364 | 35.5802 | 0.6518 |

This was the first clearly theory-shaped improvement. Updating retained structure from `A - N` improved the model in the expected direction.

Three-seed, 5-epoch comparison:

| Condition | Accuracy | Switch Recovery | Distractor Recovery | Post-Distractor Acc |
|---|---:|---:|---:|---:|
| CPA tuned | 0.8333 +/- 0.0065 | 14.54 +/- 1.40 | 41.57 +/- 2.79 | 0.6293 +/- 0.0172 |
| CPA diff | 0.8366 +/- 0.0085 | 14.32 +/- 2.86 | 40.95 +/- 2.83 | 0.6169 +/- 0.0155 |

Across seeds, the diff path gave a small repeatable accuracy gain, but did not yet solve the baseline gap.

Interpretation:

> Mismatch-driven retention helped, but the small CPA was still underpowered.

## 5. Capacity Sweep: Larger CPA Improved Smoothly

The next question was whether CPA needed more capacity.

Seed 0, 5-epoch CPA diff capacity sweep:

| Hidden Dim | Params | Accuracy | Switch Recovery | Distractor Recovery | Stateful Acc |
|---:|---:|---:|---:|---:|---:|
| 64 | 46,082 | 0.8431 | 11.20 | 38.23 | 0.6987 |
| 96 | 102,914 | 0.8568 | 9.03 | 38.22 | 0.7515 |
| 128 | 182,274 | 0.8633 | 10.08 | 37.31 | 0.7857 |

Scaling helped. The h128 CPA reached MLP-level raw accuracy on seed 0, though it still trailed the Transformer.

Three-seed h128 CPA diff, 5 epochs:

| Condition | Accuracy | Switch Recovery | Distractor Recovery | Post-Distractor Acc |
|---|---:|---:|---:|---:|
| CPA diff h64 | 0.8366 +/- 0.0085 | 14.32 +/- 2.86 | 40.95 +/- 2.83 | 0.6169 +/- 0.0155 |
| CPA diff h128 | 0.8579 +/- 0.0048 | 11.26 +/- 1.07 | 38.11 +/- 1.27 | 0.6416 +/- 0.0071 |
| MLP | 0.8615 +/- 0.0049 | 10.89 +/- 1.09 | 35.46 +/- 3.25 | 0.6648 +/- 0.0095 |
| Transformer | 0.8706 +/- 0.0106 | 8.91 +/- 1.66 | 34.51 +/- 1.55 | 0.6758 +/- 0.0169 |

Interpretation:

> Capacity was part of the problem. Larger CPA became much more competitive, but 5 epochs was still not enough to beat MLP across seeds.

## 6. h128 CPA Trained Longer Beat MLP

Finally, h128 CPA diff was trained for 10 epochs across three seeds.

Three-seed aggregate:

| Model | Accuracy | Switch Recovery | Distractor Recovery | Post-Distractor Acc |
|---|---:|---:|---:|---:|
| MLP 5e | 0.8615 +/- 0.0049 | 10.89 +/- 1.09 | 35.46 +/- 3.25 | 0.6648 +/- 0.0095 |
| Transformer 5e | 0.8706 +/- 0.0106 | 8.91 +/- 1.66 | 34.51 +/- 1.55 | 0.6758 +/- 0.0169 |
| CPA diff h128 10e | 0.8765 +/- 0.0019 | 7.92 +/- 0.74 | 34.21 +/- 0.50 | 0.6818 +/- 0.0115 |

Per-seed h128 CPA diff, 10 epochs:

| Seed | Accuracy | Switch Recovery | Distractor Recovery | Post-Distractor Acc | Stateful Acc |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.8780 | 7.39 | 33.69 | 0.6946 | 0.8093 |
| 1 | 0.8744 | 7.61 | 34.68 | 0.6783 | 0.7908 |
| 2 | 0.8770 | 8.77 | 34.26 | 0.6725 | 0.7842 |

This was the strongest result at that stage.

CPA h128 10e beats the original MLP h64 5e baseline on every reported aggregate metric:

- higher accuracy
- faster post-switch recovery
- faster distractor recovery
- higher post-distractor accuracy

It also reaches near 5-epoch Transformer parity, and slightly exceeds that Transformer aggregate on this run. However, this is not a fair Transformer comparison because the CPA was trained for 10 epochs while the Transformer baseline was trained for 5.

Interpretation:

> A larger mismatch-driven CPA trained longer reliably beats the original smaller MLP baseline on the harder lagged/distractor task. Transformer parity is plausible but not yet established under matched training budgets.

## 7. Larger MLP Beat CPA Again

The next sanity check was whether the MLP result was mostly a capacity/training-budget artifact. The MLP was scaled from h64 to h128 and trained for 10 epochs across the same three seeds and lagged/distractor setup.

Three-seed aggregate:

| Model | Params | Accuracy | Switch Recovery | Distractor Recovery | Post-Distractor Acc |
|---|---:|---:|---:|---:|---:|
| CPA diff h128 10e | 182,274 | 0.8765 +/- 0.0019 | 7.92 +/- 0.74 | 34.21 +/- 0.50 | 0.6818 +/- 0.0115 |
| MLP h128 10e | 25,090 | 0.8849 +/- 0.0060 | 7.08 +/- 0.81 | 32.40 +/- 2.71 | 0.7265 +/- 0.0234 |

Per-seed MLP h128, 10 epochs:

| Seed | Accuracy | Switch Recovery | Distractor Recovery | Post-Distractor Acc |
|---:|---:|---:|---:|---:|
| 0 | 0.8877 | 6.14 | 29.55 | 0.7512 |
| 1 | 0.8780 | 7.59 | 34.94 | 0.7045 |
| 2 | 0.8890 | 7.51 | 32.71 | 0.7236 |

MLP h128 10e beats CPA h128 10e on every reported aggregate metric except distractor-step accuracy, where they are effectively tied:

- higher accuracy
- faster post-switch recovery
- faster distractor recovery
- higher post-distractor accuracy

Interpretation:

> The previous CPA-over-MLP result does not survive a larger, equally longer-trained MLP baseline. The current evidence supports "CPA can be made competitive with a simple MLP on this task," not "CPA beats MLP."

## 8. Longer Distractor Bursts Did Not Reveal a CPA Advantage

The next stress test increased distractor burst length while keeping the distractor start probability fixed:

- current: 20-40 steps
- long: 40-80 steps
- very long: 80-160 steps

MLP h128, CPA diff h128, and Transformer h128 were trained for 10 epochs across the same three seeds. Note that h128 is the same hidden dimension, not the same parameter count: MLP h128 has 25,090 parameters, CPA h128 has 182,274, and Transformer h128 has 405,250.

Three-seed aggregate:

| Burst Length | Model | Accuracy | Switch Recovery | Distractor Recovery | Post-Distractor Acc | Non-Distractor Acc |
|---|---|---:|---:|---:|---:|---:|
| 20-40 | MLP h128 10e | 0.8849 +/- 0.0060 | 7.08 +/- 0.81 | 32.40 +/- 2.71 | 0.7265 +/- 0.0234 | 0.9480 +/- 0.0049 |
| 20-40 | CPA diff h128 10e | 0.8765 +/- 0.0019 | 7.92 +/- 0.74 | 34.21 +/- 0.50 | 0.6818 +/- 0.0115 | 0.9379 +/- 0.0036 |
| 20-40 | Transformer h128 10e | 0.8895 +/- 0.0048 | 6.27 +/- 0.13 | 33.16 +/- 1.14 | 0.7242 +/- 0.0363 | 0.9532 +/- 0.0073 |
| 40-80 | MLP h128 10e | 0.8246 +/- 0.0114 | 14.29 +/- 0.42 | 56.43 +/- 2.43 | 0.6879 +/- 0.0326 | 0.9409 +/- 0.0069 |
| 40-80 | CPA diff h128 10e | 0.8212 +/- 0.0058 | 15.27 +/- 1.20 | 58.47 +/- 1.22 | 0.6719 +/- 0.0077 | 0.9352 +/- 0.0018 |
| 40-80 | Transformer h128 10e | 0.8342 +/- 0.0087 | 12.35 +/- 1.24 | 55.58 +/- 1.87 | 0.6924 +/- 0.0408 | 0.9505 +/- 0.0057 |
| 80-160 | MLP h128 10e | 0.7702 +/- 0.0132 | 19.21 +/- 1.20 | 63.90 +/- 9.86 | 0.7112 +/- 0.0073 | 0.9446 +/- 0.0057 |
| 80-160 | CPA diff h128 10e | 0.7588 +/- 0.0103 | 19.77 +/- 4.64 | 52.01 +/- 1.56 | 0.6308 +/- 0.0111 | 0.9294 +/- 0.0104 |
| 80-160 | Transformer h128 10e | 0.7762 +/- 0.0172 | 17.52 +/- 0.70 | 64.56 +/- 10.42 | 0.6942 +/- 0.0556 | 0.9530 +/- 0.0072 |

Longer bursts made the task harder as intended: overall accuracy fell as more targets became random distractors. Non-distractor accuracy stayed high, so the models were still learning the underlying lagged regimes.

The h128 Transformer did not break under longer distractors. It had the best mean accuracy and non-distractor accuracy at every burst length, and the best switch recovery. Its weak point was very-long post-distractor accuracy: at 80-160 bursts, MLP did better there.

The only CPA-favorable signal was very-long distractor recovery: at 80-160 bursts, CPA recovered faster on average after distractors. But that came with much worse post-distractor accuracy and lower non-distractor accuracy, so it is not a convincing robustness win.

Interpretation:

> Lengthening distractor bursts does not rescue the CPA claim. The MLP remains stronger overall than CPA, and the h128 Transformer remains stronger still on accuracy and non-distractor rule use. The current CPA's faster recovery under very long bursts appears to be a narrow metric tradeoff rather than better retained rule use.

## Current Best Claim

The honest claim is:

> Minimal CPA did not work out of the box. But the trajectory is positive: A/N coordination, mismatch-driven N updates, larger hidden state, and longer training produced a CPA variant that beats the original h64/5-epoch MLP baseline across three seeds and approaches the 5-epoch Transformer baseline on this synthetic task. Once the MLP is also scaled to h128 and trained for 10 epochs, the MLP is ahead again.
> Longer distractor bursts do not change that conclusion. The h128 Transformer also survives the longer-burst stress test.

The stronger claims not yet justified:

> CPA beats a capacity-tuned MLP.
> CPA beats Transformer.

To make the Transformer claim, the next experiment should train the Transformer under the same 10-epoch budget and compare across the same three seeds. To make the MLP claim, CPA needs either a stronger architecture/training setup or a parameter-matched comparison against larger MLPs.

## Why This Is a Good Experimental Story

The trajectory is useful because it was not a straight confirmation.

1. The easy task gave a promising but weak signal.
2. The harder task falsified the naive CPA.
3. The failure identified specific architectural/training issues.
4. A CPA-specific mismatch mechanism improved results.
5. Scaling CPA improved results further.
6. Larger CPA trained longer beat the original MLP baseline across seeds.
7. Larger MLP trained longer beat CPA again.
8. Longer distractor bursts produced one narrow CPA-favorable recovery metric, but not an overall advantage.

That makes the result more credible than a single cherry-picked win.

The central lesson so far:

> Retained difference helps only when the architecture gives N a useful way to respond to persistent A/N mismatch, and only when the CPA has enough capacity and training time to exploit that signal.
