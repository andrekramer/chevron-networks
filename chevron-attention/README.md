# Chevron Attention: phase one

This repository implements the first, mechanistic experiment from the Chevron
Attention proposal: can a model retrieve a key-value fact while independently
revoking or restoring permission to use it?

The implementation deliberately does **not** claim to test IDL consolidation.
Facts live in the input context, so there is no slow retained-memory update in
this phase.

## Architecture

- The `A` stream sees facts and the final query. Control instructions are masked.
- `Q_A`, `K_A`, and `V_A` retrieve the target fact.
- The contextual `N` stream sees facts, controls, and the query.
- A query-dependent scalar gate `g_ij` determines whether each memory value can
  participate.
- The answer head receives only the gated value mixture plus an explicit learned
  null value. It has no residual bypass around the gate.

The training objective is:

```text
L = L_answer + lambda_r L_retrieval + lambda_g L_permission
```

## Run

```bash
source .venv/bin/activate
python chevron_attention.py --model all --steps 1000
```

Run the fixed five-seed comparison with:

```bash
python multi_seed.py
```

For a quick smoke run:

```bash
python chevron_attention.py --model chevron --steps 10 --eval-batches 2
python -m unittest -v
```

Useful ablations are available directly through the loss weights:

```bash
# Answer supervision only
python chevron_attention.py --model chevron \
  --retrieval-weight 0 --permission-weight 0

# Retrieval auxiliary only
python chevron_attention.py --model chevron --permission-weight 0

# Permission auxiliary only
python chevron_attention.py --model chevron --retrieval-weight 0
```

The principal metrics are answer accuracy by control mode, retrieval accuracy,
attention mass on the correct fact, permission accuracy, and the target gate by
mode. A successful mechanistic result has high target attention in every mode,
a closed target gate only when revoked, and a reopened gate after restoration.
