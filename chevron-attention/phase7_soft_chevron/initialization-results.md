# Phase 7.2 results: gate initialization robustness

Answer-only Soft Chevron was trained for 700 steps from seven theta/sharpness
initializations. Each condition used the same five seeds and held-out task as
the supervision comparison.

## Results

| Initialization | theta / k initially | Accuracy | Target r | Non-target r | No-match null mass | theta / k finally |
|---|---:|---:|---:|---:|---:|---:|
| Calibrated | 0.10 / 30 | 1.0000 ± 0.0000 | 0.9333 ± 0.0042 | 0.0207 ± 0.0010 | 0.9793 ± 0.0013 | 0.1039 / 29.99 |
| Low theta | 0.02 / 30 | 1.0000 ± 0.0000 | 0.7209 ± 0.0024 | 0.0201 ± 0.0003 | 0.9801 ± 0.0004 | 0.0391 / 30.12 |
| High theta | 0.30 / 30 | 0.9999 ± 0.0002 | 0.9018 ± 0.0038 | 0.0623 ± 0.0013 | 0.9382 ± 0.0019 | 0.1865 / 30.12 |
| Soft gate | 0.10 / 5 | 0.9999 ± 0.0002 | 0.5989 ± 0.0063 | 0.1309 ± 0.0043 | 0.8692 ± 0.0035 | 0.0801 / 6.37 |
| Sharp gate | 0.10 / 80 | 1.0000 ± 0.0000 | 0.9980 ± 0.0002 | 0.0010 ± 0.0001 | 0.9990 ± 0.0001 | 0.0947 / 78.82 |
| Closed and sharp | 0.02 / 80 | 1.0000 ± 0.0000 | 0.8989 ± 0.0084 | 0.0002 ± 0.0000 | 0.9998 ± 0.0000 | 0.0359 / 78.86 |
| Open and sharp | 0.30 / 80 | 1.0000 ± 0.0000 | 0.9926 ± 0.0017 | 0.0114 ± 0.0012 | 0.9890 ± 0.0010 | 0.2189 / 78.70 |

## Interpretation

The learned system is robust to the tested theta and sharpness initializations.
Even strongly open or closed saturated gates recover a high-separation solution
from answer labels alone.

There is no unique learned calibration. Theta moves substantially when it
starts too low or high, but sharpness remains close to its initial regime. A
gate initialized at `k=5` ends near `6.37`, admits about 13% of non-target
templates, and sends only 87% of no-match mass to V_null. It nevertheless
achieves effectively perfect answer accuracy because the value and answer
projections compensate. A gate initialized at `k=80` remains very sharp and
implements a much cleaner separation.

The result therefore supports optimization robustness more strongly than
mechanism identifiability:

> Answer-only training finds a successful Soft Chevron system across the tested
> theta/k initializations, but task accuracy does not determine one canonical
> admission boundary.

The A and N matching projections still begin from a favorable centered identity
geometry, and the synthetic categories are cleanly separated. Random matching
projections, representation noise, and out-of-distribution template corruption
remain necessary before claiming that the network discovers normalized matching
without architectural initialization.

