# Phase-one five-seed results

Run date: 2026-07-04

Command:

```bash
python multi_seed.py --seeds 7 17 27 37 47 \
  --steps 1000 --batch-size 96 --eval-batches 10 --log-every 1000
```

Each final seed evaluation used 1,920 freshly generated, balanced examples.
Values below are mean ± sample standard deviation across the five seeds, followed
by the observed seed range.

## Chevron Attention

| Metric | Mean ± SD | Range |
|---|---:|---:|
| Answer accuracy | 0.9999 ± 0.0002 | 0.9995–1.0000 |
| Active accuracy | 1.0000 ± 0.0000 | 1.0000–1.0000 |
| Revoked accuracy | 1.0000 ± 0.0000 | 1.0000–1.0000 |
| Restored accuracy | 0.9997 ± 0.0007 | 0.9984–1.0000 |
| Retrieval accuracy | 1.0000 ± 0.0000 | 1.0000–1.0000 |
| Target attention mass | 0.9986 ± 0.0001 | 0.9985–0.9987 |
| All-item permission accuracy | 0.7327 ± 0.0653 | 0.6805–0.8059 |
| Target gate, active | 0.8975 ± 0.0684 | 0.8404–0.9727 |
| Target gate, revoked | 0.2261 ± 0.0846 | 0.1251–0.2953 |
| Target gate, restored | 0.9007 ± 0.0652 | 0.8507–0.9735 |

Exact perfect-answer runs: 4/5. The remaining run missed one restored example,
giving 9,599 correct answers over 9,600 combined evaluation examples.

## Standard Transformer

| Metric | Mean ± SD | Range |
|---|---:|---:|
| Answer accuracy | 1.0000 ± 0.0000 | 1.0000–1.0000 |
| Active accuracy | 1.0000 ± 0.0000 | 1.0000–1.0000 |
| Revoked accuracy | 1.0000 ± 0.0000 | 1.0000–1.0000 |
| Restored accuracy | 1.0000 ± 0.0000 | 1.0000–1.0000 |

Exact perfect-answer runs: 5/5, giving 9,600 correct answers over 9,600
combined evaluation examples.

## Interpretation

The result reliably supports the mechanistic claim: the Chevron model retrieves
the target in every condition while its target permission gate closes during
revocation and reopens after restoration. It does not support a task-performance
advantage because the standard Transformer also solves the task perfectly.

The lower all-item permission accuracy concerns distractor memories that do not
affect the current answer. The target-item gate—the gate directly relevant to the
query—shows clear separation in all five seeds. A future experiment should test
all memory permissions through multiple queries per sequence rather than relying
on auxiliary labels alone.
