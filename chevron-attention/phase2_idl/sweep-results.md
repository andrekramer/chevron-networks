# Phase-two IDL sensitivity sweeps

Date: 2026-07-06

```bash
python -m phase2_idl.sweep --seeds 7 17 27 37 47
```

Each sweep varies one factor from the default configuration. All reported values
are five-seed means. The experiment starts from the learned base mapping.

## Disturbance duration: absolute IDL

| Steps | IDL drift | Always-slow drift | Protection | IDL adaptation steps | Brief gate | Sustained gate |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.0001 | 0.0206 | 99.7% | 140.0 | 0.003 | 0.496 |
| 5 | 0.0003 | 0.0807 | 99.6% | 139.0 | 0.004 | 0.517 |
| 10 | 0.0018 | 0.2174 | 99.2% | 137.0 | 0.007 | 0.555 |
| 20 | 0.0286 | 0.5161 | 94.5% | 130.2 | 0.039 | 0.678 |
| 40 | 0.4932 | 1.0046 | 50.9% | 123.0 | 0.355 | 0.887 |
| 80 | 1.3243 | 1.5563 | 14.9% | 119.8 | 0.674 | 0.995 |

This is the predicted duration response: short contradictions are rejected, while
longer mismatches progressively open retention.

## Shift magnitude: absolute IDL

| RMS shift | IDL drift | Protection | Adaptation steps | Final error | Sustained gate |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.0001 | 99.7% | 301.0 | 0.1336 | 0.007 |
| 0.50 | 0.0002 | 99.6% | 301.0 | 0.0619 | 0.020 |
| 1.00 | 0.0005 | 99.5% | 169.2 | 0.0234 | 0.155 |
| 1.50 | 0.0010 | 99.4% | 146.0 | 0.0133 | 0.395 |
| 2.00 | 0.0018 | 99.2% | 137.0 | 0.0101 | 0.555 |
| 3.00 | 0.0051 | 98.4% | 128.8 | 0.0098 | 0.726 |

The absolute threshold fails on sustained RMS-0.25 and RMS-0.50 changes.

## Absolute versus scale-aware magnitude response

| RMS shift | Absolute drift | Scaled drift | Absolute adapt | Scaled adapt | Absolute final error | Scaled final error |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.0001 | 0.0002 | 301.0 | 148.6 | 0.1336 | 0.0013 |
| 0.50 | 0.0002 | 0.0004 | 301.0 | 147.0 | 0.0619 | 0.0024 |
| 1.00 | 0.0005 | 0.0008 | 169.2 | 145.8 | 0.0234 | 0.0045 |
| 1.50 | 0.0010 | 0.0012 | 146.0 | 145.0 | 0.0133 | 0.0066 |
| 2.00 | 0.0018 | 0.0016 | 137.0 | 144.6 | 0.0101 | 0.0086 |
| 3.00 | 0.0051 | 0.0024 | 128.8 | 134.6 | 0.0098 | 0.0106 |

Scale-aware IDL consolidates every tested magnitude without retuning. Its brief
gate remains about 0.006 across all magnitudes, and its adaptation time remains
between 134.6 and 148.6 steps.

## Persistence decay: absolute IDL

| Beta | Brief drift | Protection | Adaptation steps | Final error |
|---:|---:|---:|---:|---:|
| 0.9800 | 0.0482 | 77.8% | 121.2 | 0.0492 |
| 0.9900 | 0.0062 | 97.1% | 127.0 | 0.0190 |
| 0.9950 | 0.0018 | 99.2% | 137.0 | 0.0101 |
| 0.9975 | 0.0010 | 99.5% | 158.8 | 0.0138 |
| 0.9990 | 0.0007 | 99.7% | 237.2 | 0.0691 |

The persistence timescale exposes the expected tradeoff. Extremely slow
persistence protects N but substantially delays consolidation.

## Retention threshold: absolute IDL

| Theta | Brief drift | Protection | Adaptation steps | Final error |
|---:|---:|---:|---:|---:|
| 0.05 | 0.0877 | 59.7% | 122.0 | 0.0056 |
| 0.10 | 0.0304 | 86.0% | 125.0 | 0.0062 |
| 0.20 | 0.0018 | 99.2% | 137.0 | 0.0101 |
| 0.40 | 0.0000 | 100.0% | 161.6 | 0.0441 |
| 0.80 | 0.0000 | 100.0% | 301.0 | 0.2620 |

Theta controls the stability-plasticity frontier directly. Theta 0.80 makes N
effectively rigid within the experiment window.

## Fixed-rate frontier

| Fixed eta_N | Brief drift | Adaptation steps | Final error |
|---:|---:|---:|---:|
| 0.002 | 0.0231 | 301.0 | 1.0938 |
| 0.004 | 0.0460 | 301.0 | 0.6031 |
| 0.008 | 0.0907 | 291.0 | 0.1849 |
| 0.012 | 0.1341 | 196.0 | 0.0566 |
| 0.020 | 0.2174 | 120.0 | 0.0053 |

No tested fixed rate matches either IDL variant's combination of brief protection
and sustained adaptation.

## Scale-aware parameter sensitivity

| Noise margin | Brief drift at shift 2.0 | Adapt at shift 0.25 | Adapt at shift 2.0 |
|---:|---:|---:|---:|
| 2 | 0.0111 | 142.2 | 130.2 |
| 3 | 0.0016 | 148.6 | 144.6 |
| 4 | 0.0016 | 149.0 | 145.0 |
| 6 | 0.0016 | 149.8 | 146.0 |

| Scale update rate | Brief drift at shift 2.0 | Adapt at shift 0.25 | Adapt at shift 2.0 |
|---:|---:|---:|---:|
| 0.002 | 0.0016 | 148.0 | 137.6 |
| 0.005 | 0.0016 | 148.2 | 142.8 |
| 0.010 | 0.0016 | 148.6 | 144.6 |
| 0.050 | 0.0016 | 148.4 | 144.6 |

The scale-aware result is stable for margins 3–6 and scale update rates
0.002–0.05. Margin 2 remains functional but admits more temporary drift.

## Conclusion

The absolute formulation demonstrates the duration-sensitive retention mechanism
but fails when persistent changes fall below its fixed magnitude threshold. The
scale-aware formulation repairs that failure across a twelve-fold shift range
without condition-specific tuning. This remains a controlled linear experiment;
the next question is whether the same normalization remains useful with nonlinear
models, nonstationary noise, and gradual rather than abrupt drift.
