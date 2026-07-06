# Initial phase-two implementation check

Date: 2026-07-06

The experiment begins from an already learned base mapping. A 600-step stable
phase estimates ordinary online variation before the temporary and sustained
changes. Values are mean ± sample standard deviation across seeds 7, 17, 27,
37, and 47.

```bash
python -m phase2_idl.experiment --seeds 7 17 27 37 47
```

| Metric | Absolute IDL | Scale-aware IDL | Always slow | Fixed low rate | Fast only |
|---|---:|---:|---:|---:|---:|
| Base error before contradiction | 0.0000 | 0.0000 | 0.0005 | 0.0004 | 0.0017 |
| N drift during brief contradiction | 0.0018 ± 0.0002 | 0.0016 ± 0.0001 | 0.2174 ± 0.0116 | 0.0907 ± 0.0049 | 1.6609 ± 0.0489 |
| Base error after recovery | 0.0064 ± 0.0007 | 0.0054 ± 0.0002 | 0.0075 ± 0.0004 | 0.0333 ± 0.0013 | 0.0018 ± 0.0003 |
| Sustained adaptation steps | 137.0 ± 0.0 | 144.6 ± 0.5 | 120.0 ± 0.0 | 291.0 ± 0.0 | 14.2 ± 0.4 |
| Final sustained error | 0.0101 ± 0.0002 | 0.0086 ± 0.0001 | 0.0053 ± 0.0002 | 0.1849 ± 0.0001 | 0.0018 ± 0.0002 |
| Mean gate during brief contradiction | 0.0066 | 0.0061 | 1.0 | 1.0 | 1.0 |
| Mean gate, first 50 sustained steps | 0.5549 | 0.4196 | 1.0 | 1.0 | 1.0 |

Both IDL variants strongly protect N during a ten-step contradiction and then
consolidate the sustained change. The scale-aware method pays roughly eight
additional adaptation steps at the default RMS-2 shift, while removing the
absolute method's failure on smaller persistent shifts.

