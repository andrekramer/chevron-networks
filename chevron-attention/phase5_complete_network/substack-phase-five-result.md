# Chevron Attention Phase 5: When Context Becomes Memory

## A learned Q/K–A, V–N network for reversible stability and plasticity

Can a neural system learn what to retrieve, decide whether the retrieved
information should currently participate, and then decide whether that current
permission should become its future default?

That is the question behind Phase 5 of the Chevron Attention experiments.

The result is positive, within a deliberately controlled setting.

A small neural associative-memory system learned retrieval, contextual control,
value representation, and answer production jointly. When its learned signals
were connected to Inertial Difference Learning, temporary contextual changes
altered current behavior without overwriting retained policy. Sustained changes
were consolidated. Sustained restoration reversed them again.

The result survived independent training seeds, larger memory and control sets,
moderate continuous noise, and missing retention signals. A robustness test
also found a failure in the first directional update rule. Replacing hard reset
with two independent directional persistence traces repaired that failure and
completed every tested cycle even when 30% of the retention signals were
randomly flipped.

The strongest defensible claim is therefore:

> In a controlled learned associative-memory system, separating fast Q/K
> retrieval from retained V/content and applying persistence-gated,
> two-trace retention enables temporary contextual changes to affect behavior
> without overwriting retained policy, while sustained changes are reversibly
> consolidated.

This is not yet a claim about language models or general continual learning.
But it is the first experiment in this sequence where all the visible neural
computation is learned and the complete stability–plasticity cycle remains
measurable.

## The last remaining scaffold

The earlier experiments built the mechanism one distinction at a time.

Phase 1 showed:

\[
\text{retrieval} \neq \text{permission}.
\]

The model could retrieve a fact while a separate channel prevented that fact
from controlling its answer.

Phase 2 showed:

\[
\text{difference} \neq \text{retained change}.
\]

A brief contradiction could remain temporary, while a persistent
contradiction opened slow retention.

Phase 3 combined those ideas:

\[
\text{contextual permission} \neq \text{retained policy}.
\]

The system could obey an override immediately without immediately turning that
override into memory.

Phase 4 replaced the supplied contextual gate with a learned token-conditioned
module. But retrieval was still algorithmic. The experiment handed the system
the correct key/value association and tested only whether learned context could
drive retention.

Phase 5 removes that scaffold.

The neural network must now learn:

- which fact answers the query;
- which value the selected fact contains;
- whether context says `revoked`, `restored`, or `no context`;
- whether to return the value or abstain.

The online retention transition remains rule-specified rather than learned.

## Q and K from A, V from N

The Phase 5 network uses an asymmetric form of Chevron Attention.

The adaptive channel, \(A\), supplies the query and keys:

\[
Q_A=W_Q e_A(q),
\]

\[
K_A=W_K e_A(k_j).
\]

Their similarity selects the relevant fact:

\[
\alpha_j=\operatorname{softmax}_j
\left(\frac{Q_AK_{A,j}^{\top}}{\sqrt d}\right).
\]

The retained channel, \(N\), supplies the values:

\[
V_{N,j}=W_V e_N(v_j).
\]

So \(A\) answers:

> Which relation is relevant now?

And \(N\) supplies:

> What retained content is associated with that relation?

This is the \(Q_A,K_A,V_N\) form proposed in the original Chevron Attention
taxonomy. It is also ART-adjacent: present evidence selects a retained
template or value. Full Adaptive Resonance Theory adds vigilance, reset, and
category search. We did not add those yet because this task contains one
correct association. Resetting after a veto would merely select an irrelevant
fact rather than perform meaningful category search.

## Learned contextual permission

A second learned \(N\) path reads key-addressed control events.

For every fact, it predicts:

\[
p(\text{no context}),\quad
p(\text{revoked}),\quad
p(\text{restored}).
\]

Retained \(N\) stores the default permission for that fact:

\[
N_{\text{retained},j}\in[0,1].
\]

The current permission gate is:

\[
g_j=
p_j(\text{no context})N_{\text{retained},j}
+p_j(\text{restored}).
\]

This has the intended three-way interpretation:

- no contextual instruction: use retained policy;
- revoked: close the gate;
- restored: open the gate.

The final attention output is:

\[
Y=\sum_j \alpha_jg_jV_{N,j}
+\left(1-\sum_j\alpha_jg_j\right)V_{\varnothing},
\]

where \(V_{\varnothing}\) is a learned abstention value.

Because the gated attention mass is not renormalized, the network can retrieve
the correct fact while allowing none of its value to participate.

That preserves the defining Chevron Attention state:

\[
\alpha_{\text{target}}\approx1,
\qquad
g_{\text{target}}\approx0.
\]

The system has found the relevant information but declined to use it.

## The learning task

Every training example contains a fresh random associative memory.

For example:

```text
K7 -> BLUE
K2 -> HORSE
K9 -> GLASS
```

It also contains a query, a randomly assigned retained permission for each
fact, and zero or more contextual controls:

```text
REVOKE K2
QUERY K2
```

or:

```text
RESTORE K7
QUERY K7
```

The key/value bindings are regenerated for every batch. The network therefore
cannot solve the task by memorizing one global mapping from keys to values.
Retained permissions are randomized independently, so `no context` cannot be
treated as a synonym for either active or revoked.

Training combines three objectives:

\[
L=L_{\text{answer}}+L_{\text{retrieval}}+L_{\text{context}}.
\]

The auxiliary objectives make the intended internal mechanism directly
measurable. They also mean that the decomposition is architecturally and
supervisionally encouraged. Phase 5 does not claim that gradient descent
discovered Chevron Attention without scaffolding.

## From learned context to retained policy

After training, every neural parameter is frozen.

The model then passes through an online permission cycle:

1. ordinary active use;
2. a short revoke;
3. a no-context probe;
4. a sustained revoke;
5. another no-context probe;
6. a sustained restore;
7. a final no-context probe.

During a contextual override, behavior should change immediately.

After context disappears, the answer must come from retained policy alone.
That is the stability–plasticity test.

The initial IDL rule accumulates difference between current contextual
permission and retained permission:

\[
P_t=\beta P_{t-1}+(1-\beta)E_t.
\]

It opens a slow update gate:

\[
\rho_t=\sigma\left(s(P_t-\theta)\right).
\]

Retained permission then changes through:

\[
N_{t+1}=N_t+
\eta_N\rho_t(g_t-N_t).
\]

The intended behavior is:

\[
\text{short context}
\rightarrow
\text{current compliance without retained change},
\]

\[
\text{persistent context}
\rightarrow
\text{current compliance followed by retained change}.
\]

## The clean result

Three independently initialized networks were first trained for 700 steps and
evaluated on 2,560 fresh examples each.

| Metric | Mean ± standard deviation |
|---|---:|
| Answer accuracy | 1.0000 ± 0.0000 |
| Q/K retrieval accuracy | 1.0000 ± 0.0000 |
| N contextual-state accuracy | 1.0000 ± 0.0000 |
| Attention mass on queried fact | 0.9999 ± 0.0000 |

A later seed audit trained ten independent networks and tested each on five
independent online streams. All fifty default integrated-IDL runs completed
the short-preserve, sustained-revoke, and sustained-restore cycle correctly.

The retained permission values make the mechanism visible:

| Method | After short revoke | After long revoke | After long restore |
|---|---:|---:|---:|
| Integrated IDL | 0.9982 | 0.0191 | 0.9675 |
| Always update | 0.4344 | 0.0013 | 0.9971 |
| Fixed slow | 0.9228 | 0.5259 | 0.7298 |
| Context only | 1.0000 | 1.0000 | 1.0000 |

Integrated IDL barely moved retained policy during the short revoke:

\[
1.0000\rightarrow0.9982.
\]

During sustained revocation it consolidated:

\[
0.9982\rightarrow0.0191.
\]

During sustained restoration it reversed:

\[
0.0191\rightarrow0.9675.
\]

The internal update gate averaged \(0.0023\) during short revocation and
\(0.6814\) during sustained revocation.

The model was not stable because it ignored context. It obeyed the short revoke
immediately. It simply did not treat that brief context as evidence that the
future default should change.

## Why the alternatives fail

| Method | Preserve after short | Consolidate revoke | Final active | Full revoke/restore cycle |
|---|---:|---:|---:|---:|
| Integrated IDL | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Always update | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| Fixed slow | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| Context only | 1.0000 | 0.0000 | 1.0000 | 0.0000 |

The always-update model treated a ten-step revoke as retained policy. It was
plastic, but unstable.

The fixed-slow model preserved the brief revoke but remained above the revoked
decision boundary after seventy sustained steps. It was stable, but too slow
to adapt within the tested horizon.

The context-only model behaved correctly while context was present but never
changed retained policy.

One metric needs care. A model can finish active after the restore phase simply
because it never consolidated the earlier revoke. We therefore count a full
revoke/restore cycle only if revocation succeeded first. This prevents
unchanged active behavior from being misreported as successful restoration.

The result is not merely that one update rate happened to work. IDL changes its
effective rate as evidence persists.

## Where does context become memory?

The duration sweep makes the temporal boundary visible.

| Revoke duration | Retained N after revoke | Preserved as temporary? |
|---:|---:|---:|
| 2 | 0.9999 | Yes |
| 5 | 0.9997 | Yes |
| 10 | 0.9982 | Yes |
| 20 | 0.9647 | Yes |
| 30 | 0.7313 | Yes |
| 40 | 0.3746 | No |

In the sustained schedule, twenty steps were insufficient to consolidate, and
thirty steps crossed the retained decision boundary.

The exact number is not universal. It is determined by \(\beta\), \(\theta\),
\(\eta_N\), the probe schedule, and the evidence scale. The result demonstrates
a tunable transition, not a metaphysical boundary between temporary and
permanent.

That transition is the point of IDL:

\[
\text{the same difference}
+\text{different duration}
\rightarrow
\text{different retained outcome}.
\]

## Robustness beyond a few friendly seeds

Perfect accuracy on a small task can conceal brittleness. We therefore varied
the model, the online stream, the memory size, the number of controls, signal
quality, and IDL parameters.

The frozen networks were trained with six facts and at most four controls. They
retained perfect answer, retrieval, and contextual accuracy when evaluated with
up to twelve facts and eight controls.

This generalization remains inside the trained key vocabulary. It shows set-size
generalization, not recognition of unseen symbol identities.

The original retention rule also survived:

- Gaussian retention-signal noise through standard deviation \(0.20\);
- randomly missing retention signals through 25% dropout;
- \(\beta\) values from \(0.970\) to \(0.990\) across every tested threshold;
- retained update rates \(\eta_N\) from \(0.020\) to \(0.160\).

Very slow update rates failed within the seventy-step horizon. At
\(\eta_N=0.010\), retained permission was still \(0.6125\) after sustained
revocation. At \(\eta_N=0.020\), it reached \(0.3735\) and crossed the
decision boundary.

So the result occupies a broad parameter region, but not every parameter
setting. Inertia can become rigidity if the timescale is too slow.

## A failure in the first directional rule

The robustness sweep exposed a more interesting problem.

The initial implementation used hard directional reset. If the current signal
changed from revoke to restore, or restore to revoke, accumulated persistence
was cleared.

That seems reasonable during a genuine reversal. But it means one wrongly
signed observation can erase a long run of consistent evidence.

With independent random sign flips, full-cycle success fell to:

| Flipped retention signals | Hard-reset full-cycle success |
|---:|---:|
| 1% | 80% |
| 5% | 20% |
| 10% | 0% |
| 20% | 0% |

This is a useful negative result.

The network still retrieved correctly. Immediate contextual behavior remained
correct. The failure occurred specifically in the rule that decided whether
noisy contextual evidence should become retained policy.

## Two directional traces

The repair is simple.

Instead of storing one persistence value and clearing it on reversal, maintain
two traces:

\[
P_t^+=\beta P_{t-1}^+ +(1-\beta)E_t^+,
\]

\[
P_t^-=\beta P_{t-1}^- +(1-\beta)E_t^-.
\]

\(P^+\) accumulates evidence toward restoration. \(P^-\) accumulates evidence
toward revocation.

An opposing observation no longer deletes the other trace. It contributes to
its own trace while previous evidence decays normally.

We compared:

- **hard reset:** clear persistence on any directional change;
- **two-trace:** maintain independent revoke and restore evidence;
- **signed hysteresis:** maintain one signed average that must cross zero before
  direction reverses.

On clean data:

| Rule | Revoke crossing | Restore crossing | N after revoke | N after restore |
|---|---:|---:|---:|---:|
| Hard reset | 29 steps | 36 steps | 0.0191 | 0.9675 |
| Two-trace | 29 steps | 36 steps | 0.0191 | 0.9675 |
| Signed hysteresis | 29 steps | 54 steps | 0.0191 | 0.8696 |

Two-trace exactly matched the clean behavior of hard reset.

Signed hysteresis resisted noise, but it imposed an eighteen-step restoration
delay. Genuine restore evidence first had to cancel accumulated revoke evidence.
It therefore failed shorter restoration windows.

Two-trace did not pay that penalty.

## Robustness to wrongly signed observations

The comparison used three independently trained networks and five independent
online streams per network, giving fifteen runs per noise setting.

| Flipped signals | Hard reset | Two-trace | Signed hysteresis |
|---:|---:|---:|---:|
| 1% | 80% | 100% | 100% |
| 5% | 20% | 100% | 100% |
| 10% | 0% | 100% | 100% |
| 20% | 0% | 100% | 20% |
| 30% | 0% | 100% | 0% |

The cells report successful full revoke/restore cycles.

Two-trace IDL completed every tested cycle through 30% independently flipped
retention observations. Its response slowed smoothly rather than collapsing:

| Flipped signals | Revoke crossing | Restore crossing |
|---:|---:|---:|
| 0% | 29.0 | 36.0 |
| 5% | 31.0 | 39.0 |
| 10% | 35.0 | 41.6 |
| 20% | 40.8 | 41.4 |
| 30% | 50.6 | 44.8 |

This makes two directional traces the strongest retention rule in the current
experiment. They preserve the clean temporal behavior, tolerate inconsistent
evidence, and still reverse when the environment genuinely changes.

## What Phase 5 establishes

Phase 5 supports a stronger claim than the earlier mechanism-isolation tests:

\[
\text{learned retrieval}
+\text{learned contextual control}
+\text{explicit persistent retention}
\rightarrow
\text{reversible stability and plasticity}.
\]

More precisely:

> In a controlled learned associative-memory system, separating fast Q/K
> retrieval from retained V/content and applying persistence-gated,
> two-trace retention enables temporary contextual changes to affect behavior
> without overwriting retained policy, while sustained changes are reversibly
> consolidated. This behavior survived independent initializations and streams,
> larger tested memory/control sets, moderate signal degradation, and up to 30%
> independently flipped retention signals in the tested regime.

The important result is not perfect toy-task accuracy.

The important result is that the system exposes and controls three different
operations:

\[
\text{retrieve now},
\]

\[
\text{permit now},
\]

\[
\text{retain for later}.
\]

Ordinary attention often collapses the first two. Ordinary online updating can
collapse the second and third. Chevron Attention with IDL keeps them distinct.

## What Phase 5 does not establish

The interpretation should remain disciplined.

This is a small associative-memory network, not a language Transformer.

The control grammar is simple. The retained state is a scalar permission for
each known key. Retrieval and context receive explicit auxiliary supervision.
The IDL consolidation equation is designed, not discovered by gradient descent.
Neural parameters are frozen during the online retention cycle, so this is not
yet parameter-level continual learning.

The larger-set test stays inside the trained symbol vocabulary. The noise is
synthetic and mostly independent, not adversarial or temporally correlated.
Fifteen runs per corrupted setting provide a useful early comparison, not a
universal statistical guarantee.

The experiment therefore does not show that Chevron Attention:

- solves catastrophic forgetting;
- improves standard Transformer benchmarks;
- scales to natural language;
- discovers its own A/N decomposition;
- constitutes a complete ART mechanism;
- or provides a general alignment solution.

Those would require different experiments.

## The next experiment

Phase 5 makes the next step clearer.

The present task has one correct association, so ART-style reset has nowhere
meaningful to search. A stronger task should contain several plausible retained
templates and genuinely novel inputs.

That would allow a direct comparison between:

\[
\text{low mismatch}\rightarrow\text{resonate and proceed},
\]

\[
\text{intermediate mismatch}\rightarrow\text{veto and search},
\]

\[
\text{persistent high mismatch}\rightarrow\text{revise or create memory}.
\]

It should also test temporally correlated errors, gradual drift, learned
retained representations rather than scalar permissions, and parameter-level
adaptation.

For now, Phase 5 provides a complete small-scale demonstration of the central
Chevron/IDL idea:

> A system can change what it does now without immediately changing what it
> retains—and can later change what it retains when the difference persists.

With two directional traces, it can do so without allowing a few contradictory
observations to erase the history that made the change meaningful.

The code and full results are available here:

https://github.com/andrekramer/chevron-networks/tree/main/chevron-attention/phase5_complete_network
