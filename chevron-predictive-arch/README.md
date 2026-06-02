# Chevron Predictive Architecture Experiment

This is a small PyTorch implementation of the experiment in `exp.txt`.

It compares:

- `mlp`: feedforward next-bit baseline
- `transformer`: tiny causal Transformer baseline
- `cpa`: recurrent fast/slow Chevron Predictive Architecture model

The task is synthetic binary next-bit prediction. Hidden regimes generate the next bit from recent bits, switch after random durations, and can include noise plus short distractor bursts. The dataset records switch, distractor, noise, regime, sequence id, and target index metadata so recovery metrics are event-aware.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 train.py --model mlp --use-distractors --epochs 5
python3 train.py --model transformer --use-distractors --epochs 5
python3 train.py --model cpa --use-distractors --epochs 5 --stateful-eval
```

Harder lagged-regime run:

```bash
python3 train.py --model cpa --regime-set lagged --use-distractors --distractor-prob 0.006 --min-distractor 20 --max-distractor 40 --epochs 5 --stateful-eval
```

Useful CPA knobs:

```bash
--rho 0.05
--lambda-band 0.01
--lambda-slow 0.001
--target-dist 1.0
--n-lr-mult 0.25
--detach-a-to-n / --no-detach-a-to-n
--use-diff-to-n
--regime-set easy|lagged
```

The default evaluation is window-reset for all models: every context window is processed independently. For CPA, `--stateful-eval` also reports a continuous sequence pass where A/N state persists across the whole generated test sequence after an initial warmup window.

## Outputs

Each run writes to `runs/<model>_seed<seed>/`:

- `history.csv`
- `metrics.json`
- `model.pt`

Generate plots:

```bash
python3 plots.py runs/mlp_seed0 runs/transformer_seed0 runs/cpa_seed0
```

This creates per-run `history.png` files and, when multiple run dirs are supplied, `runs/noise_comparison.png`.

## Metrics

The main reported metrics are:

- overall next-bit accuracy
- accuracy on switch, distractor, and noise positions
- average post-switch recovery time
- distractor recovery time
- post-distractor accuracy
- CPA-only A/N diagnostics during training

CPA is not expected to win purely on raw accuracy. The intended signal is whether it gives a better adaptation-retention trade-off while maintaining non-collapsed, non-divergent A/N separation.
