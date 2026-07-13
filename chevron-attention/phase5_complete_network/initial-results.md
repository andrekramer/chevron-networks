# Phase 5 initial results

Default run on CPU, three independently initialized networks (`7`, `17`, and
`27`), 700 training steps per network.

## Held-out neural task

Each network was evaluated on 2,560 fresh random associative-memory examples.

| Metric | Mean ± SD |
|---|---:|
| Answer accuracy | 1.0000 ± 0.0000 |
| Q/K retrieval accuracy | 1.0000 ± 0.0000 |
| N contextual-state accuracy | 1.0000 ± 0.0000 |
| Attention mass on queried fact | 0.9999 ± 0.0000 |

The bindings are regenerated for every batch, so perfect answer accuracy cannot
come from memorizing a fixed key/value lookup table. Retained permissions are
also randomized independently of context.

## Frozen-network online cycle

After training, all neural parameters were frozen. The same learned signals
then drove four retained-state rules through a short revoke, a long revoke, and
a long restore.

| Method | N after short revoke | N after long revoke | N after long restore |
|---|---:|---:|---:|
| Integrated IDL | 0.9982 ± 0.0000 | 0.0191 ± 0.0000 | 0.9675 ± 0.0000 |
| Always update | 0.4344 ± 0.0000 | 0.0013 ± 0.0000 | 0.9971 ± 0.0000 |
| Fixed slow | 0.9228 ± 0.0000 | 0.5259 ± 0.0000 | 0.7298 ± 0.0000 |
| Context only | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

| Method | Preserve after short | Consolidate revoke | Final active | Full revoke/restore cycle |
|---|---:|---:|---:|---:|
| Integrated IDL | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Always update | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| Fixed slow | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| Context only | 1.0000 | 0.0000 | 1.0000 | 0.0000 |

The integrated update gate averaged `0.0023` during the short revoke and
`0.6814` during the long revoke. Current behavior obeyed both contexts; the
difference was whether the contextual state survived after context disappeared.

The zero spread in retained-state endpoints is expected here. Once the learned
networks classify the simple control events perfectly, every seed supplies
effectively the same binary contextual signal to the deterministic IDL rule.
The neural measurements still come from independently trained parameters.

## Claim and limitation

This closes the specific Phase 4 scaffold: retrieval is no longer algorithmic.
It shows that jointly learned Q/K retrieval, N values, N contextual control, and
answer production can supply clean signals to online IDL retention.

It remains a deliberately small associative-memory network. The consolidation
equation is architecturally specified rather than discovered by gradient
descent, and the tested control grammar is simple. The next informative
experiment is a noisy multi-template task where an ART-style vigilance/reset
path has a legitimate alternative category to search for.
