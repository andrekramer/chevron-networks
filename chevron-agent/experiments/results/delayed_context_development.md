# Delayed-context development results

Seeds 0-4.

| Condition | Return/decision | Final old | Final new | q unresolved-resolved | Premature writes | Old overwrite rate | Promotion precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard_attention | 0.944 +/- 0.044 | 0.972 +/- 0.050 | 0.966 +/- 0.052 | 0.627 +/- 0.062 | 0.0000 | 0.000000 | nan |
| standard_attention_buffer | 0.931 +/- 0.042 | 0.967 +/- 0.065 | 0.952 +/- 0.052 | 0.700 +/- 0.042 | 0.0000 | 0.000000 | 1.000 |
| chevron_buffer | 0.949 +/- 0.004 | 0.986 +/- 0.006 | 0.979 +/- 0.021 | 0.764 +/- 0.026 | 0.0000 | 0.000000 | 1.000 |
| chevron_immediate | 0.900 +/- 0.062 | 0.985 +/- 0.015 | 0.875 +/- 0.217 | 0.469 +/- 0.062 | 0.0093 | 0.000000 | nan |
| chevron_scalar_residual | 0.949 +/- 0.004 | 0.986 +/- 0.006 | 0.979 +/- 0.021 | 0.764 +/- 0.026 | 0.0000 | 0.000000 | 1.000 |
| chevron_coupled_write | 0.849 +/- 0.107 | 0.936 +/- 0.076 | 0.877 +/- 0.106 | 0.741 +/- 0.041 | 0.0000 | 0.000000 | 1.000 |

## Paired diagnostics

```json
{
  "chevron_buffer_minus_standard_attention": {
    "old_retention_mean": 0.014459942839591866,
    "old_retention_std": 0.05342672298911865,
    "old_retention_wins": 1,
    "old_retention_approx_95ci_low": -0.032370644652546396,
    "old_retention_approx_95ci_high": 0.061290530331730125,
    "new_acquisition_mean": 0.01243872236725616,
    "new_acquisition_std": 0.042545991617751425,
    "new_acquisition_wins": 2,
    "new_acquisition_approx_95ci_low": -0.024854483568295782,
    "new_acquisition_approx_95ci_high": 0.04973192830280811,
    "return_per_decision_mean": 0.004333333333333323,
    "return_per_decision_std": 0.04056202383730104,
    "return_per_decision_wins": 1,
    "return_per_decision_approx_95ci_low": -0.031220848167894097,
    "return_per_decision_approx_95ci_high": 0.03988751483456074
  },
  "chevron_buffer_minus_standard_attention_buffer": {
    "old_retention_mean": 0.019281371411020442,
    "old_retention_std": 0.06813555140139142,
    "old_retention_wins": 1,
    "old_retention_approx_95ci_low": -0.04044207263921293,
    "old_retention_approx_95ci_high": 0.0790048154612538,
    "new_acquisition_mean": 0.026167608286252354,
    "new_acquisition_std": 0.040600164489801784,
    "new_acquisition_wins": 3,
    "new_acquisition_approx_95ci_low": -0.009420004970920628,
    "new_acquisition_approx_95ci_high": 0.06175522154342533,
    "return_per_decision_mean": 0.017333333333333333,
    "return_per_decision_std": 0.039150705967808265,
    "return_per_decision_wins": 3,
    "return_per_decision_approx_95ci_low": -0.016983773511828053,
    "return_per_decision_approx_95ci_high": 0.05165044017849472
  },
  "chevron_buffer_minus_chevron_immediate": {
    "old_retention_mean": 0.0016819539844753484,
    "old_retention_std": 0.017753291460260365,
    "old_retention_wins": 1,
    "old_retention_approx_95ci_low": -0.013879492095092056,
    "old_retention_approx_95ci_high": 0.017243400064042753,
    "new_acquisition_mean": 0.10322697405056243,
    "new_acquisition_std": 0.20257145431832546,
    "new_acquisition_wins": 2,
    "new_acquisition_approx_95ci_low": -0.07433473447489104,
    "new_acquisition_approx_95ci_high": 0.2807886825760159,
    "return_per_decision_mean": 0.04833333333333332,
    "return_per_decision_std": 0.05865388118255925,
    "return_per_decision_wins": 4,
    "return_per_decision_approx_95ci_low": -0.003079060330278785,
    "return_per_decision_approx_95ci_high": 0.09974572699694542
  },
  "chevron_buffer_minus_chevron_scalar_residual": {
    "old_retention_mean": 0.0,
    "old_retention_std": 0.0,
    "old_retention_wins": 0,
    "old_retention_approx_95ci_low": 0.0,
    "old_retention_approx_95ci_high": 0.0,
    "new_acquisition_mean": 0.0,
    "new_acquisition_std": 0.0,
    "new_acquisition_wins": 0,
    "new_acquisition_approx_95ci_low": 0.0,
    "new_acquisition_approx_95ci_high": 0.0,
    "return_per_decision_mean": 0.0,
    "return_per_decision_std": 0.0,
    "return_per_decision_wins": 0,
    "return_per_decision_approx_95ci_low": 0.0,
    "return_per_decision_approx_95ci_high": 0.0
  },
  "chevron_buffer_minus_chevron_coupled_write": {
    "old_retention_mean": 0.050306484295845985,
    "old_retention_std": 0.07838422471642235,
    "old_retention_wins": 3,
    "old_retention_approx_95ci_low": -0.01840031799733352,
    "old_retention_approx_95ci_high": 0.11901328658902549,
    "new_acquisition_mean": 0.10171413461290406,
    "new_acquisition_std": 0.10787234545872637,
    "new_acquisition_wins": 3,
    "new_acquisition_approx_95ci_low": 0.007159854856387329,
    "new_acquisition_approx_95ci_high": 0.19626841436942077,
    "return_per_decision_mean": 0.1,
    "return_per_decision_std": 0.1037692418568988,
    "return_per_decision_wins": 4,
    "return_per_decision_approx_95ci_low": 0.009042249123868251,
    "return_per_decision_approx_95ci_high": 0.19095775087613176
  }
}
```
