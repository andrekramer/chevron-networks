# Chevron Agent

Small PyTorch proof of concept for a Chevron proposal/norm reinforcement-learning agent.

## Experiment

Goal: show that a Chevron network can be built as a trainable proposal/norm agent and compared against a simple matched recurrent baseline.

Current setup:

- baseline: CNN encoder + GRU recurrent core
- Chevron: same encoder + Chevron proposal/norm recurrent core
- gated Chevron: Chevron core with tension gate
- trainer: PPO with ordinary backprop
- task: `CueGridEnv`, a small gridworld with cue-conditioned target choice

The current curriculum builds the task in stages so we can separate basic trainability from harder generalization.

## Difficulty Ladder

- Stage 1: `*_easy.yaml`
  fixed layout, auto-interact, no cue masking, no reversals
- Stage 2: `*_stage2.yaml`
  fixed layout, manual `INTERACT`, no cue masking, no reversals
- Stage 3: `*_stage3.yaml`
  symbolic object planes, small layout pool, manual `INTERACT`, cue+targets only, light distance shaping
- Stage 4: `*_stage4.yaml`
  larger grid and horizon, random layout, manual `INTERACT`
- Stage 5: `*_stage5.yaml`
  masked cues with `WAIT`
- Stage 6: `*_stage6.yaml`
  full 9x9 task with masked cues and reversals

## Current Results

Stable early stages:

- Stage 1: baseline, Chevron, and gated Chevron all reach `eval/success_rate = 1.0`
- Stage 2: baseline, Chevron, and gated Chevron all reach `eval/success_rate = 1.0`

Stage 3 is not solved. An independent audit found that the original evaluation
counted a positive progress-shaped reward on the last step of a timeout as
success. It also selected and reported checkpoints on the same 50 deterministic
episodes.

After correcting success to require a positively rewarded terminal interaction
and evaluating every saved checkpoint on 200 fresh shared episodes:

- final GRU success: `0.033 +/- 0.047`
- final Chevron success: `0.000 +/- 0.000`
- held-out curve-mean GRU success: `0.003 +/- 0.004`
- held-out curve-mean Chevron success: `0.000 +/- 0.000`

The earlier apparent Chevron advantage is therefore superseded. The saved
models learned progress shaping but almost never completed the task. See the
[independent audit](experiments/results/stage3_existing_checkpoint_results.md)
and its [predeclared protocol](experiments/stage3_confirmation_protocol.md).

Interpretation:

- the easy stages remain proof-of-trainability checks;
- Stage 3 provides no current Chevron-over-GRU result;
- increasing navigation difficulty is not justified until a task directly
  testing delayed memory and consolidation succeeds.

## Delayed-context memory experiment

A mechanism-first contextual-bandit experiment now isolates delayed
eligibility, old/new context learning, unresolved evidence, and protected
consolidation. Thresholds were frozen after five development seeds and tested
on twenty untouched confirmation seeds.

- Chevron + buffer: return `0.944 +/- 0.048`, final old accuracy
  `0.991 +/- 0.008`, final new accuracy `0.979 +/- 0.051`.
- Standard attention: return `0.948 +/- 0.051`, final old accuracy
  `0.984 +/- 0.031`, final new accuracy `0.980 +/- 0.038`.
- The paired Chevron-minus-standard return interval crosses zero; there is no
  evidence of Chevron superiority on this task.
- Within Chevron, buffering beats immediate write by `+0.035` mean return, and
  separate stricter write permission beats a read/write-coupled gate by
  `+0.054`; both paired approximate 95% intervals exclude zero.
- Novel-context q falls from `0.869` on early encounters to `0.209` after
  learning. Per-slot residual signatures add no clear benefit over scalar q.

See the [full findings](experiments/results/delayed_context_findings.md),
[frozen protocol](experiments/delayed_context_protocol.md), and generated
[confirmation table](experiments/results/delayed_context_confirmation.md).

This experiment uses fixed online prototype memories, not yet an end-to-end
learned deep network. The next step is to learn the projections and policy with
small matched PyTorch models while retaining the verified memory mechanics.

## Checkpoints

Training writes runs under `runs/<run_name>/`.

Saved checkpoints:

- `latest.pt`: most recent checkpoint
- `best_eval.pt`: best evaluation checkpoint

`best_eval.pt` is selected by evaluation success rate, with evaluation return mean used as the tiebreaker.

## Quick Start

```bash
python -m chevron_agent.train --config configs/baseline_easy.yaml
python -m chevron_agent.train --config configs/chevron_easy.yaml
python -m chevron_agent.train --config configs/baseline_stage2.yaml
python -m chevron_agent.train --config configs/chevron_stage3.yaml
```
