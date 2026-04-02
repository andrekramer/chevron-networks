# Chevron Proposal/Norm Agent: Implementation Spec

## Goal

Build a small PyTorch reinforcement-learning repo that demonstrates:

1. a standard recurrent visual baseline can solve the task,
2. a Chevron recurrent core can replace the baseline core cleanly,
3. the Chevron core can be interpreted as a distributed proposal/norm agent,
4. the model trains with ordinary PPO and standard backpropagation.

This is a proof-of-concept implementation, not a benchmark suite.

## Non-goals

- No custom local learning rules in v1.
- No distributed training in v1.
- No large-scale environment suite in v1.
- No claims of state-of-the-art performance.

## Deliverables

The first complete repo should produce:

- a trainable visual gridworld environment,
- a GRU baseline agent,
- a Chevron recurrent agent,
- an optional Chevron + tension-gate agent,
- a PPO trainer shared by all models,
- reproducible training scripts,
- evaluation scripts,
- episode logging and a small set of visualizations.

## Recommended Repo Layout

```text
chevron_agent/
  README.md
  pyproject.toml
  chevron_agent/
    __init__.py
    config.py
    envs/
      __init__.py
      cue_grid.py
      render.py
    models/
      __init__.py
      encoders.py
      baseline_gru.py
      chevron_core.py
      chevron_agent.py
    rl/
      __init__.py
      rollout.py
      ppo.py
      advantages.py
      buffers.py
    analysis/
      __init__.py
      metrics.py
      visualize.py
    train.py
    evaluate.py
    export_episode.py
  configs/
    baseline.yaml
    chevron.yaml
    chevron_gate.yaml
```

Keep the codebase small. Avoid introducing abstractions that are only useful at scale.

## Dependencies

Recommended minimum:

- `torch`
- `numpy`
- `gymnasium`
- `matplotlib`
- `pyyaml`

Optional:

- `imageio` for GIF export
- `pandas` for analysis tables
- `tensorboard` or `wandb` if needed, but not required for v1

## Environment Spec

### Class

Implement one environment class:

- `CueGridEnv(gym.Env)`

### Observation

Use image observation from the start.

Recommended shape:

- `(3, 9, 9)` float32 in `[0, 1]`

Each object type maps to a fixed RGB color.

Keep rendering deterministic and simple. Do not start with sprites.

### Actions

Discrete action space of size 6:

0. up
1. down
2. left
3. right
4. interact
5. wait

### Objects

At minimum:

- agent
- cue
- target_a
- target_b
- trap
- lure
- empty

### Episode Logic

Each episode should:

1. sample a cue-target mapping,
2. place cue, targets, trap, lure, and agent,
3. run for up to `max_steps`,
4. terminate on correct target, wrong target, trap, or timeout.

### Reversal Logic

Support blockwise reversal by flipping the cue-target mapping every `reversal_period` episodes.

This should be controlled outside the model. The environment state should expose enough info to log whether the current episode is pre- or post-reversal.

### WAIT-Useful Cue Reveal

If WAIT is included, implement a concrete information mechanism:

- the cue starts partially masked for the first `reveal_wait_steps`,
- each `wait` action reduces the mask,
- after one or two waits the cue becomes fully visible.

This prevents WAIT from being purely decorative.

### Reward Function

Use one fixed reward scheme in code:

- correct target: `+1.0`
- wrong target: `-1.0`
- trap: `-1.0`
- lure pickup: `+0.2`
- delayed lure penalty after `lure_delay_steps`: `-0.6`
- wait cost: `-0.01`
- step cost: `-0.005`
- timeout: `0.0` additional terminal reward

Do not vary this inside the first implementation.

### Info Dict

The environment `info` dict should include:

- `cue_type`
- `correct_target`
- `reversal_block`
- `was_ambiguous`
- `lure_triggered`
- `delayed_lure_penalty_applied`
- `episode_step`

These fields make analysis much easier later.

## Rendering Rules

Keep rendering inside the environment or a small helper module.

Recommended color map:

- background: black
- agent: white
- cue blue: blue
- cue red: red
- target_a: blue
- target_b: red
- trap: yellow
- lure: green
- masked cue: gray-blue or gray-red

Do not optimize visual style. Optimize legibility.

## Model Spec

## Shared Encoder

Use the same encoder for baseline and Chevron models.

Recommended encoder:

- `Conv2d(3, 16, kernel_size=3, padding=1)`
- `ReLU`
- `Conv2d(16, 32, kernel_size=3, padding=1)`
- `ReLU`
- flatten
- `Linear(32 * 9 * 9, hidden_dim)`
- `ReLU`

Recommended `hidden_dim`:

- `128`

This keeps the task nontrivial while still small.

## Baseline Model

### Class

- `BaselineGRUAgent(nn.Module)`

### Structure

- shared visual encoder
- `GRUCell(hidden_dim, hidden_dim)`
- policy head: `Linear(hidden_dim, action_dim)`
- value head: `Linear(hidden_dim, 1)`

### Forward Signature

```python
def forward(self, obs, hidden, done_mask):
    ...
    return {
        "logits": logits,
        "value": value,
        "hidden": next_hidden,
    }
```

`done_mask` should reset recurrent state on episode boundaries inside batched rollouts.

## Chevron Core

### Core Idea

Represent the recurrent hidden state as two equal-width channels:

- proposal channel `A`
- norm channel `N`

If total hidden size is `128`, use:

- `A`: 64 dims
- `N`: 64 dims

### Class

- `ChevronCore(nn.Module)`

### Parameters

Recommended learnable blocks:

- input projection into `A`
- input projection into `N`
- recurrent `AA`
- recurrent `AN`
- recurrent `NA`
- recurrent `NN`
- optional gate projection if using dynamic relevance gates

For v1, keep it simple:

```python
AA = nn.Linear(d, d, bias=False)
AN = nn.Linear(d, d, bias=False)
NA = nn.Linear(d, d, bias=False)
NN = nn.Linear(d, d, bias=False)
in_A = nn.Linear(input_dim, d)
in_N = nn.Linear(input_dim, d)
```

where `d = hidden_dim // 2`.

### Update Rule

Recommended v1 recurrent update:

```python
m_A = AA(A) + AN(N) + in_A(z)
m_N = NA(A) + NN(N) + in_N(z)
```

Without tension gate:

```python
A_next = (1 - leak) * A + m_A
N_next = (1 - leak) * N + m_N
```

With tension gate:

```python
pi = torch.sigmoid(A - N)
u = torch.sqrt(eps + pi * (1.0 - pi))
A_next = (1 - leak) * A + u * m_A
N_next = (1 - leak) * N + u * m_N
```

Then apply a bounded nonlinearity:

```python
A_next = torch.tanh(A_next)
N_next = torch.tanh(N_next)
```

Using elementwise gating is good enough for v1. Do not overcomplicate this with graph message passing yet.

### Reset Handling

On episode termination, multiply `A` and `N` by a `not_done` mask before applying the next update.

### Optional Relevance Gate

Skip `g_ij(t)` in v1 unless the basic Chevron core is already working. It adds complexity without being necessary for the first demo.

## Chevron Agent

### Class

- `ChevronAgent(nn.Module)`

### Structure

- shared encoder
- chevron core
- policy readout
- value readout

### Readout Options

Recommended default:

- policy from `A`
- value from `N`

Implementation:

```python
policy_head = nn.Linear(d, action_dim)
value_head = nn.Linear(d, 1)
```

Alternative ablation:

- read both from concatenated `[A, N]`

### Forward Output

Return enough data for PPO and analysis:

```python
{
    "logits": logits,
    "value": value,
    "state": next_state,
    "A": A_next,
    "N": N_next,
    "tension": u if gated else None,
}
```

## PPO Spec

Use one PPO implementation shared across all models.

### Rollout Storage

Store:

- observations
- actions
- rewards
- dones
- logprobs
- values
- recurrent states
- optional Chevron diagnostics: `A`, `N`, `tension`

### GAE

Use standard Generalized Advantage Estimation.

Recommended defaults:

- `gamma = 0.99`
- `gae_lambda = 0.95`
- `clip_coef = 0.2`
- `entropy_coef = 0.01`
- `value_coef = 0.5`
- `max_grad_norm = 0.5`

### Batch Structure

Use vectorized environments if convenient, but do not make this a requirement for v1.

Simple starting point:

- `num_envs = 8`
- `rollout_steps = 128`

### Recurrent PPO Handling

Training batches must preserve sequence order for recurrent updates.

Recommended approach:

- segment rollouts by environment,
- slice them into fixed unroll windows,
- pass initial recurrent state for each window,
- mask episode boundaries.

Do not flatten everything as if the model were feedforward.

## Config Spec

Use simple YAML config files.

Important fields:

```yaml
seed: 0
device: cpu
total_updates: 1000
num_envs: 8
rollout_steps: 128
learning_rate: 3e-4
hidden_dim: 128
leak: 0.1
epsilon: 1e-3
use_tension_gate: false
max_steps: 40
reversal_period: 50
reveal_wait_steps: 2
```

Avoid a large config surface in v1.

## Training Script

`train.py` should:

1. load config,
2. set seed,
3. build envs,
4. build model,
5. build PPO trainer,
6. run updates,
7. periodically evaluate,
8. save checkpoints and metrics.

### Save Artifacts

Save:

- model checkpoint
- config snapshot
- training metrics JSON or CSV
- optional small episode visualization outputs

## Evaluation Script

`evaluate.py` should:

- load a checkpoint,
- run fixed-number evaluation episodes,
- report average return,
- report success rate,
- report reversal recovery stats,
- report lure/trap rates,
- optionally export one or two episodes.

Keep evaluation deterministic where possible.

## Analysis Outputs

The first analysis suite should generate:

- training curves
- return vs update
- success rate vs update
- reversal recovery time
- lure/trap error rate
- WAIT usage rate

For Chevron models, also generate:

- mean `A` magnitude by episode
- mean `N` magnitude by episode
- mean `A - N` before action
- mean tension near reversals and ambiguous cues

## Episode Export

Implement one utility script:

- `export_episode.py`

It should dump a single evaluation episode to:

- frame images or a GIF
- action sequence
- rewards
- cue metadata
- optional Chevron state traces

This is important for making the architecture legible.

## Logging Priorities

Log these every update:

- `train/return_mean`
- `train/episode_length_mean`
- `train/value_loss`
- `train/policy_loss`
- `train/entropy`
- `eval/return_mean`
- `eval/success_rate`

For Chevron:

- `chevron/A_mean`
- `chevron/N_mean`
- `chevron/tension_mean`
- `chevron/A_minus_N_mean`

Do not add many custom metrics before the training loop is stable.

## Testing Plan

Minimum tests:

1. Environment reset and step behave deterministically under a fixed seed.
2. Observation tensor has correct shape and dtype.
3. Reward logic matches spec.
4. GRU agent forward pass returns correct shapes.
5. Chevron agent forward pass returns correct shapes.
6. Chevron gated and ungated variants run without NaNs.
7. PPO update step runs one batch without crashing.

If time is limited, prioritize shape and rollout tests over deeper unit tests.

## Implementation Order

Build in this order:

1. environment
2. renderer
3. baseline GRU agent
4. PPO loop
5. baseline training sanity check
6. Chevron core
7. Chevron agent
8. Chevron training sanity check
9. tension-gated Chevron
10. metrics and episode export

Do not start with the tension-gated version. Get the ungated Chevron core learning first.

## Sanity Checks

Before full training, verify:

- the baseline learns Task 1,
- the Chevron model overfits a tiny fixed environment slice,
- no recurrent state explosion occurs,
- value estimates stay finite,
- gated tension values stay in a sensible range.

Expected tension range with the chosen formula should be roughly:

- minimum near `sqrt(eps)`
- maximum near `sqrt(eps + 0.25)`

If `eps = 1e-3`, max tension is just above `0.5`.

## Comparison Rules

To keep the comparison honest:

- same encoder width,
- same total hidden budget where practical,
- same PPO hyperparameters unless there is a clear failure reason,
- same reward scheme,
- same rollout budget,
- same evaluation protocol.

Do not tune the baseline heavily while leaving Chevron untuned, or vice versa.

## Minimal Success Criteria

The implementation succeeds if:

1. the baseline learns the stable task,
2. the Chevron agent also learns the stable task,
3. the Chevron agent trains stably across multiple seeds,
4. the Chevron architecture produces interpretable `A` and `N` traces,
5. the repo is simple enough that another reader could reproduce the result.

## Stretch Goals

Only after the core demo works:

- add partial observability beyond masked cue reveal,
- add readout ablations,
- add `wAN` and `wNA` ablations,
- add the no-WAIT ablation,
- add a relevance-gating mechanism,
- add predictive-coding or local-update variants.

## Practical Notes

- Keep all tensor shapes explicit in code comments around recurrent state.
- Prefer `GRUCell` and a custom Chevron cell rather than a more abstract recurrent wrapper.
- Store `A` and `N` separately in the Chevron code. Do not hide them in one tensor everywhere if that hurts readability.
- Keep the first implementation CPU-friendly.
- Favor clarity over premature optimization.

## Suggested README Claim

This repository is a minimal PyTorch proof of concept showing that Chevron networks can implement a trainable distributed proposal/norm reinforcement-learning agent in a small visual world.
