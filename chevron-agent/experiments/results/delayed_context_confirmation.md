# Delayed-context confirmation results

Seeds 100-119.

| Condition | Return/decision | Final old | Final new | q unresolved-resolved | Premature writes | Old overwrite rate | Promotion precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard_attention | 0.948 +/- 0.051 | 0.984 +/- 0.031 | 0.980 +/- 0.038 | 0.633 +/- 0.046 | 0.0000 | 0.000000 | nan |
| standard_attention_buffer | 0.912 +/- 0.072 | 0.965 +/- 0.052 | 0.937 +/- 0.078 | 0.681 +/- 0.055 | 0.0000 | 0.000000 | 1.000 |
| chevron_buffer | 0.944 +/- 0.048 | 0.991 +/- 0.008 | 0.979 +/- 0.051 | 0.753 +/- 0.052 | 0.0000 | 0.000000 | 0.998 |
| chevron_immediate | 0.909 +/- 0.067 | 0.965 +/- 0.051 | 0.922 +/- 0.121 | 0.542 +/- 0.075 | 0.0095 | 0.000000 | nan |
| chevron_scalar_residual | 0.942 +/- 0.048 | 0.991 +/- 0.008 | 0.966 +/- 0.078 | 0.752 +/- 0.053 | 0.0000 | 0.000000 | 0.992 |
| chevron_coupled_write | 0.890 +/- 0.083 | 0.960 +/- 0.053 | 0.939 +/- 0.082 | 0.741 +/- 0.057 | 0.0000 | 0.000000 | 1.000 |

## Paired diagnostics

```json
{
  "chevron_buffer_minus_standard_attention": {
    "old_retention_mean": 0.007603175511873839,
    "old_retention_std": 0.029926526503009723,
    "old_retention_wins": 8,
    "old_retention_approx_95ci_low": -0.005512703015997206,
    "old_retention_approx_95ci_high": 0.02071905403974488,
    "new_acquisition_mean": -0.0009760659209119449,
    "new_acquisition_std": 0.06596274171069756,
    "new_acquisition_wins": 8,
    "new_acquisition_approx_95ci_low": -0.029885512112598524,
    "new_acquisition_approx_95ci_high": 0.027933380270774633,
    "return_per_decision_mean": -0.00416666666666668,
    "return_per_decision_std": 0.022232819110810813,
    "return_per_decision_wins": 5,
    "return_per_decision_approx_95ci_low": -0.013910629259859643,
    "return_per_decision_approx_95ci_high": 0.005577295926526284
  },
  "chevron_buffer_minus_standard_attention_buffer": {
    "old_retention_mean": 0.026739777676717553,
    "old_retention_std": 0.05357356781065598,
    "old_retention_wins": 8,
    "old_retention_approx_95ci_low": 0.003260126350040568,
    "old_retention_approx_95ci_high": 0.05021942900339454,
    "new_acquisition_mean": 0.04206758360991345,
    "new_acquisition_std": 0.08584540480842028,
    "new_acquisition_wins": 11,
    "new_acquisition_approx_95ci_low": 0.0044441761112209086,
    "new_acquisition_approx_95ci_high": 0.079690991108606,
    "return_per_decision_mean": 0.03150000000000001,
    "return_per_decision_std": 0.045237838015361576,
    "return_per_decision_wins": 16,
    "return_per_decision_approx_95ci_low": 0.011673643332335368,
    "return_per_decision_approx_95ci_high": 0.05132635666766465
  },
  "chevron_buffer_minus_chevron_immediate": {
    "old_retention_mean": 0.026649105158956677,
    "old_retention_std": 0.051619089619849796,
    "old_retention_wins": 10,
    "old_retention_approx_95ci_low": 0.004026041666935643,
    "old_retention_approx_95ci_high": 0.04927216865097771,
    "new_acquisition_mean": 0.057704899745413415,
    "new_acquisition_std": 0.13757743706590406,
    "new_acquisition_wins": 10,
    "new_acquisition_approx_95ci_low": -0.002591070538700474,
    "new_acquisition_approx_95ci_high": 0.1180008700295273,
    "return_per_decision_mean": 0.03524999999999998,
    "return_per_decision_std": 0.05011179023411393,
    "return_per_decision_wins": 15,
    "return_per_decision_approx_95ci_low": 0.013287539590212966,
    "return_per_decision_approx_95ci_high": 0.057212460409787
  },
  "chevron_buffer_minus_chevron_scalar_residual": {
    "old_retention_mean": 0.0003759398496240629,
    "old_retention_std": 0.0016812541184209068,
    "old_retention_wins": 1,
    "old_retention_approx_95ci_low": -0.00036090225563910035,
    "old_retention_approx_95ci_high": 0.0011127819548872261,
    "new_acquisition_mean": 0.013432835820895522,
    "new_acquisition_std": 0.06007346805223317,
    "new_acquisition_wins": 1,
    "new_acquisition_approx_95ci_low": -0.012895522388059705,
    "new_acquisition_approx_95ci_high": 0.03976119402985075,
    "return_per_decision_mean": 0.002083333333333337,
    "return_per_decision_std": 0.00931694990624914,
    "return_per_decision_wins": 1,
    "return_per_decision_approx_95ci_low": -0.0020000000000000026,
    "return_per_decision_approx_95ci_high": 0.006166666666666677
  },
  "chevron_buffer_minus_chevron_coupled_write": {
    "old_retention_mean": 0.031713933009846734,
    "old_retention_std": 0.051549467307892276,
    "old_retention_wins": 11,
    "old_retention_approx_95ci_low": 0.009121382841394094,
    "old_retention_approx_95ci_high": 0.05430648317829938,
    "new_acquisition_mean": 0.04015392443866471,
    "new_acquisition_std": 0.08587391250388109,
    "new_acquisition_wins": 8,
    "new_acquisition_approx_95ci_low": 0.0025180228915654412,
    "new_acquisition_approx_95ci_high": 0.07778982598576398,
    "return_per_decision_mean": 0.054333333333333324,
    "return_per_decision_std": 0.06559311046406381,
    "return_per_decision_wins": 15,
    "return_per_decision_approx_95ci_low": 0.02558588517808663,
    "return_per_decision_approx_95ci_high": 0.08308078148858002
  }
}
```
