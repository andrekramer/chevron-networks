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

Initial stage-3 comparison:

- gated Chevron is not helping on the current stage-3 task
- plain Chevron is the strongest of the three on peak success
- baseline is still competitive, but weaker on average than plain Chevron

Stage-3 best-checkpoint results with `50` eval episodes:

- Seed 0
  baseline: success `0.44`, return `0.0712`
  Chevron: success `0.56`, return `0.0322`
- Seed 1
  baseline: success `0.0`, return `-0.0106`
  Chevron: success `0.26`, return `0.0258`
- Seed 2
  baseline: success `0.12`, return `0.0164`
  Chevron: success `0.46`, return `0.0610`

3-seed summary over best checkpoints:

- baseline mean best success: `0.187`
- Chevron mean best success: `0.427`
- baseline mean best return: `0.0257`
- Chevron mean best return: `0.0397`

Interpretation:

- the repo already supports the proof-of-existence claim
- the early ladder is solid
- the first nontrivial comparison currently favors plain Chevron over the matched GRU baseline on the shaped stage-3 task
- stage 3 is still not fully stable, so best-checkpoint comparison is more meaningful than final-update comparison

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
