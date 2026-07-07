# Phase 3 sweep results

Command:

```bash
.venv/bin/python -m phase3_integrated.sweep
```

As in the initial Phase 3 run, the current seed changes the key/value
assignment, but the control schedule and update dynamics are deterministic.
The sweep therefore reports mean values with no visible seed variance.

## Transient revoke duration

Method: `integrated_idl`

| transient steps | retained drift | recovery active | transient update gate | post-revoke retained | final active |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.0001 | 1.0000 | 0.0004 | 1.0000 | 1.0000 |
| 5 | 0.0003 | 1.0000 | 0.0007 | 1.0000 | 1.0000 |
| 10 | 0.0018 | 1.0000 | 0.0023 | 1.0000 | 1.0000 |
| 20 | 0.0353 | 1.0000 | 0.0224 | 1.0000 | 1.0000 |
| 40 | 0.6254 | 0.0000 | 0.2985 | 1.0000 | 1.0000 |
| 80 | 0.9862 | 0.0000 | 0.6442 | 1.0000 | 1.0000 |
| 120 | 0.9995 | 0.0000 | 0.7628 | 1.0000 | 1.0000 |

This is the expected persistence boundary. Very short revocations affect current
behavior but do not rewrite retained policy. By 40 steps, the system treats the
revoke as persistent enough to consolidate, so recovery without context fails.

## Sustained duration

Method: `integrated_idl`

| sustained steps | post-revoke retained | final active | revoke consolidation steps | restore consolidation steps |
|---:|---:|---:|---:|---:|
| 20 | 0.0000 | 1.0000 | 21.0000 | 1.0000 |
| 40 | 1.0000 | 1.0000 | 32.0000 | 15.0000 |
| 60 | 1.0000 | 1.0000 | 32.0000 | 14.0000 |
| 90 | 1.0000 | 1.0000 | 32.0000 | 11.0000 |
| 120 | 1.0000 | 1.0000 | 32.0000 | 10.0000 |
| 180 | 1.0000 | 1.0000 | 32.0000 | 12.0000 |

A 20-step sustained revoke is too short to consolidate under this setting. By
40 steps, retained revocation consolidates reliably and later restoration also
consolidates.

## Fixed-slow update-rate sweep

Method: `fixed_slow`

| eta_n_low | retained drift | recovery active | post-revoke retained | final active | revoke steps | restore steps |
|---:|---:|---:|---:|---:|---:|---:|
| 0.002 | 0.0198 | 1.0000 | 0.0000 | 1.0000 | 181.0000 | 1.0000 |
| 0.004 | 0.0393 | 1.0000 | 1.0000 | 1.0000 | 163.0000 | 16.0000 |
| 0.008 | 0.0772 | 1.0000 | 1.0000 | 1.0000 | 77.0000 | 56.0000 |
| 0.012 | 0.1137 | 1.0000 | 1.0000 | 1.0000 | 48.0000 | 49.0000 |
| 0.020 | 0.1829 | 1.0000 | 1.0000 | 1.0000 | 25.0000 | 34.0000 |
| 0.040 | 0.3352 | 1.0000 | 1.0000 | 1.0000 | 7.0000 | 17.0000 |
| 0.080 | 0.5656 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 9.0000 |

The fixed-slow baseline shows the tradeoff directly. Low rates protect
transients but may fail to consolidate. High rates consolidate quickly but begin
to overwrite retained policy during a brief revoke. Intermediate rates can pass
this exact schedule, but only by choosing a schedule-specific timescale.

## IDL threshold sweep

Method: `integrated_idl`

| threshold | retained drift | recovery active | post-revoke retained | final active | revoke steps | restore steps |
|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 0.1437 | 1.0000 | 1.0000 | 1.0000 | 13.0000 | 9.0000 |
| 0.25 | 0.0190 | 1.0000 | 1.0000 | 1.0000 | 23.0000 | 10.0000 |
| 0.35 | 0.0018 | 1.0000 | 1.0000 | 1.0000 | 32.0000 | 12.0000 |
| 0.45 | 0.0002 | 1.0000 | 1.0000 | 1.0000 | 43.0000 | 14.0000 |
| 0.55 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 56.0000 | 22.0000 |

The threshold behaves as intended: lower thresholds consolidate sooner but admit
more transient drift; higher thresholds protect transients more strongly but
delay consolidation.

## Interpretation

This sweep strengthens the Phase 3 result. The integrated IDL mechanism is not
merely solving the toy cycle by updating slowly. It has an explicit
persistence-sensitive boundary: brief overrides remain contextual, while long
overrides become retained policy.

The main limitation remains that this is still a controlled mechanism test, not
a learned Transformer benchmark. The next version should randomize query keys
and control timing, then test whether a learned contextual N module can infer
the same update signal rather than receiving the contextual gate directly.
