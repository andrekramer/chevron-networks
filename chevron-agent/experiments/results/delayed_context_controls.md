# Delayed-context task-validity controls

Results over 100 seeds. These are task controls, not model-comparison results.

| Condition | Delay | Return/decision | Final old accuracy | Final new accuracy | Overall accuracy |
|---|---:|---:|---:|---:|---:|
| memoryless | 0 | 0.001 +/- 0.027 | 0.504 +/- 0.042 | 0.508 +/- 0.062 | 0.501 +/- 0.013 |
| memoryless | 3 | 0.001 +/- 0.027 | 0.504 +/- 0.042 | 0.508 +/- 0.062 | 0.501 +/- 0.013 |
| oracle_context_memory | 0 | 0.990 +/- 0.003 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.995 +/- 0.002 |
| oracle_context_memory | 3 | 0.986 +/- 0.003 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.993 +/- 0.002 |

The memoryless control should remain near chance. The oracle-context memory is
allowed to use latent IDs only to verify that the delayed task is solvable when
eligibility is handled correctly; experimental agents will not receive those IDs.
