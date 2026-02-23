# Baseline Experiments

PyTorch experiments for comparing:

- `baseline`: standard MLP
- `graph`: graph-style baseline (2-node message passing)
- `chevron`: 2-channel + 2x2 chevron operators (`full`, `diag_only`, `offdiag_frozen`)

The `graph` baseline is intended as a structural comparison point: it uses explicit message passing on a graph (the word pair as a 2-node graph), while keeping training setup and parameter budget comparable.

## Files

- `data.py`: WordNet antonym/non-antonym dataset builder and splits
- `models.py`: `BaselineMLP`, `GraphBaseline`, `ChevronMLP`
- `train.py`: training/eval CLI and swap-consistency metrics
- `requirements.txt`: dependencies
- `sweep_results.csv`: recent sweep outputs

## Setup

```bash
cd baseline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Example runs

Baseline MLP:

```bash
python train.py --model baseline
```

Graph baseline:

```bash
python train.py --model graph
```

Chevron full:

```bash
python train.py --model chevron --chevron-variant full
```

Chevron ablations:

```bash
python train.py --model chevron --chevron-variant diag_only
python train.py --model chevron --chevron-variant offdiag_frozen
```

## Fair-comparison defaults

Parameter-matched pair used in this repo:

- `baseline`: `--hidden-dim 256`
- `graph`: `--hidden-dim 256`
- `chevron`: `--hidden-groups 128`

All with `--emb-dim 64`.

## Latest 3-way comparison

Settings used:

- Date: 2026-02-23
- Command shape: `python train.py --epochs 8 --batch-size 128 --seed 7`
- Models:
  - `baseline`: `--model baseline --hidden-dim 256`
  - `graph`: `--model graph --hidden-dim 256`
  - `chevron`: `--model chevron --chevron-variant full --hidden-groups 128`

Results (test split):

| Model | Test loss | Test match_acc | Test polarity_acc | Test swap_match_consistency | Test swap_polarity_flip |
|---|---:|---:|---:|---:|---:|
| baseline | 0.8531 | 0.6632 | 0.5177 | 1.0000 | 0.4922 |
| graph | 0.9417 | 0.6257 | 0.5219 | 0.9156 | 0.7525 |
| chevron full | 0.8537 | 0.6632 | 0.4950 | 1.0000 | 0.4272 |
