# Phase 4: Replacing the Supplied Gate with a Learned One

Phase 3 still had one obvious scaffold.

The contextual gate was supplied by the experiment. The system was told, directly, whether the current context meant `revoked`, `restored`, or `no override`.

That was useful for testing the retention mechanism, but it left a gap. If Chevron/IDL needs a hand-supplied contextual gate, the result is much less interesting.

So Phase 4 made one small change.

Keep retrieval algorithmic. Keep the same stochastic short/long retention protocol. Keep the same IDL update rule. But replace the supplied contextual gate with a learned \(N\) module.

The learned module reads a short token sequence:

```text
REVOKE K3
QUERY K3
```

or:

```text
RESTORE K7
QUERY K2
```

and predicts one of three classes:

- `no context`
- `revoked`
- `restored`

That prediction becomes the contextual gate that drives the Phase 3 retention loop.

The result was positive, but simple.

The learned gate reached perfect class accuracy on the stochastic runtime stream. With that learned gate in place, the retention result matched Phase 3.

| Method | Gate accuracy | Short probe preservation | Long probe consolidation |
|---|---:|---:|---:|
| Integrated IDL | 1.0000 | 0.9785 ± 0.0295 | 0.9765 ± 0.0275 |
| Always update | 1.0000 | 0.6938 ± 0.0780 | 1.0000 ± 0.0000 |
| Fixed slow | 1.0000 | 0.7906 ± 0.0538 | 0.7229 ± 0.0614 |
| Context only | 1.0000 | 0.6612 ± 0.0580 | 0.3675 ± 0.0313 |

So the Phase 3 result does survive replacing the oracle contextual gate with a learned token-conditioned gate.

The interpretation should stay modest. This is not yet a full neural architecture. The gate task is easy, retrieval is still algorithmic, and the retained state is still a simple per-key permission variable.

But it closes one useful gap.

Phase 3 showed:

\[
\text{supplied context gate} + \text{IDL} \rightarrow \text{stable retained policy}.
\]

Phase 4 shows:

\[
\text{learned context gate} + \text{IDL} \rightarrow \text{same retention behaviour}.
\]

That is a promising early signal of non-collapsing retention under controlled continual change. The next step is the harder one: learn retrieval, contextual gating, and retained updating in the same neural model.
