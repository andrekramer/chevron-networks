# Chevron Attention Phase 3: When Context Should Not Become Memory

## Combining retrieval, permission, and retained change

The first Chevron Attention experiment showed that retrieval and assent can be separated.

A model can find the relevant fact while a separate permission channel decides whether that fact is allowed to influence the answer.

The second experiment tested retention.

A fast adaptive state, \(A\), tracked current evidence. A slower retained state, \(N\), changed only when the difference between \(A\) and \(N\) persisted. Brief contradiction was treated as disturbance. Persistent contradiction became retained change.

The third experiment combines those two ideas.

The question is:

> Can a system obey context immediately without rewriting itself immediately?

That is the missing integration step.

If a user says, "For now, do not use this fact," the system should comply now. But it should not necessarily alter its retained policy. If that instruction persists, then retained policy should eventually change.

Phase 3 tests exactly that.

The result was clean. The integrated Chevron/IDL mechanism followed contextual overrides immediately, preserved retained policy after short overrides, and consolidated retained policy after long overrides. The fixed alternatives failed one side of that tradeoff.

## The combined mechanism

The experiment has three moving parts.

First, \(A\) retrieves the queried fact:

\[
\text{retrieved value}=A(q).
\]

Second, contextual \(N\) determines whether the retrieved value is currently permitted:

\[
g_t \in [0,1].
\]

If \(g_t\) is open, the retrieved value is returned. If \(g_t\) is closed, the system returns `IDK`.

Third, retained \(N\) stores the default permission policy for each key:

\[
N_{\text{retained}}(k).
\]

The current gate is contextual when an override is present:

\[
g_t = N_{\text{context}}(k,t).
\]

When no contextual override is present, behavior falls back to retained policy:

\[
g_t = N_{\text{retained}}(k).
\]

The important question is not whether context can control current behavior. That part is easy. The important question is whether current contextual control should update retained policy.

IDL makes that decision from persistence.

For a controlled key, the system measures the difference between contextual permission and retained permission:

\[
D_t = |N_{\text{context}}(k,t)-N_{\text{retained}}(k)|.
\]

It accumulates persistent evidence:

\[
P_t=\beta P_{t-1}+(1-\beta)E_t.
\]

The retained-update gate is:

\[
\rho_t=\sigma\left(s(P_t-\theta)\right).
\]

Retained permission changes only through that gate:

\[
N_{\text{retained}}(k) \leftarrow
N_{\text{retained}}(k)+
\eta_N\rho_t\left(N_{\text{context}}(k,t)-N_{\text{retained}}(k)\right).
\]

The intended behaviour is:

\[
\text{short override}\rightarrow \text{current compliance without retained rewrite},
\]

\[
\text{long override}\rightarrow \text{current compliance followed by retained rewrite}.
\]

That is the full Chevron/IDL loop:

\[
A \text{ retrieves},\quad N \text{ gates},\quad A-N \text{ difference persists},\quad N \text{ updates}.
\]

## The experiment

Each run contained multiple keys. Each key had a retained permission state.

The model repeatedly encountered override episodes. Each episode selected:

- a random controlled key;
- a contextual override that flips the key's current retained permission;
- a random duration;
- a random stream of queries;
- a no-context probe period after the override.

Short override episodes lasted only a few steps. Long override episodes lasted many more steps.

During an override, current behavior should follow context immediately. If the controlled key is revoked, the answer should be `IDK`. If it is restored, the answer should be the retrieved value.

After the override ends, the context disappears. The system must answer from retained policy alone.

That probe is the test.

After a short override, retained policy should mostly be preserved. After a long override, retained policy should mostly have changed.

The comparisons were:

- **Integrated IDL:** immediate contextual behavior, persistence-gated retained updates;
- **Always update:** retained policy follows every contextual override;
- **Fixed slow:** retained policy follows every contextual override slowly;
- **Context only:** contextual behavior changes immediately, but retained policy never changes.

The stochastic experiment was run over five seeds. Seeds changed the key/value assignments, controlled keys, event durations, probe durations, and query streams.

## The result

The main result is the tradeoff between short-probe preservation and long-probe consolidation.

| Method | Overall answer accuracy | Short probe preservation | Long probe consolidation |
|---|---:|---:|---:|
| Integrated IDL | 0.9914 ± 0.0101 | 0.9785 ± 0.0295 | 0.9765 ± 0.0275 |
| Always update | 0.9426 ± 0.0418 | 0.6938 ± 0.0780 | 1.0000 ± 0.0000 |
| Fixed slow | 0.9062 ± 0.0258 | 0.7906 ± 0.0538 | 0.7229 ± 0.0614 |
| Context only | 0.8179 ± 0.0088 | 0.6612 ± 0.0580 | 0.3675 ± 0.0313 |

The integrated IDL model preserved retained policy after short overrides with accuracy \(0.9785\). It consolidated retained policy after long overrides with accuracy \(0.9765\).

That is the target pattern.

The current behavior was also correct during the override itself. Short-context and long-context accuracies were both \(1.0000\). So the model was not protecting retained state by ignoring context. It obeyed the context immediately.

The difference was in what happened after the context disappeared.

Short context did not usually become retained policy. Long context usually did.

## Why the baselines fail

The always-update baseline consolidated every contextual override.

That made it perfect on long-probe consolidation:

\[
1.0000.
\]

But it also rewrote retained policy during short overrides. Its short-probe preservation fell to:

\[
0.6938.
\]

This is the brittle system that treats every instruction as a permanent change.

The context-only baseline made the opposite mistake. It followed immediate context, but it never changed retained policy. Its long-probe consolidation was only:

\[
0.3675.
\]

This is the system that can comply in the moment but cannot learn that a policy has changed.

The fixed-slow baseline sat between those two extremes. It updated retained policy slowly after every override. That helped somewhat, but it did not solve the problem. It reached:

\[
0.7906
\]

on short-probe preservation and:

\[
0.7229
\]

on long-probe consolidation.

A fixed rate has one timescale. If it is too fast, temporary context becomes memory. If it is too slow, persistent context does not become memory quickly enough.

IDL changes the effective retention rate according to persistence.

## The update gate opens with duration

The internal update gate showed the intended pattern.

| Method | Short-context update gate | Long-context update gate |
|---|---:|---:|
| Integrated IDL | 0.0118 ± 0.0137 | 0.3951 ± 0.0105 |
| Always update | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Fixed slow | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Context only | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

During short contexts, the IDL update gate was almost closed. Current behavior still followed the contextual gate, but retained policy barely moved.

During long contexts, the update gate opened. Retained policy began to follow the persistent contextual instruction.

That is the important distinction.

Current assent and retained change are not the same operation.

The model can say, computationally:

> I will obey this now, but I will not yet rewrite what I retain.

Then, if the same instruction persists:

> This is no longer just context. This has become policy.

## Context without immediate memory

This phase gives the first integrated Chevron result.

Phase 1 showed:

\[
\text{retrieval} \neq \text{permission}.
\]

Phase 2 showed:

\[
\text{difference} \neq \text{retained change}.
\]

Phase 3 connects them:

\[
\text{contextual permission} \neq \text{retained policy}.
\]

A useful system needs all three distinctions.

It must be able to retrieve a fact without admitting it. It must be able to obey a contextual instruction without immediately internalizing it. And it must be able to internalize that instruction when persistence makes it look less like a temporary exception and more like a real change.

That is what the integrated Phase 3 experiment demonstrates.

The clean result is not that the model gets a toy task right. The clean result is the pattern of failures across the alternatives.

Always updating remembers too much too quickly. Context-only remembers too little. Fixed-slow updating has a single compromise timescale. IDL produces a conditional timescale: short overrides remain contextual, long overrides become retained.

That is the core Chevron/IDL claim in miniature.

This result is still a controlled mechanism test: retrieval is algorithmic, contextual gates are supplied by the experiment rather than learned from tokens, the retained state is a simple per-key permission variable, and the task is far smaller than a real Transformer setting. It demonstrates the integrated logic of retrieval, immediate contextual gating, and persistence-gated retention; it does not yet show that a learned Transformer will discover this decomposition without scaffolding, or that the method improves large-scale continual learning. The next step is to replace the supplied contextual gate with a learned Phase-1-style \(N\) module while keeping the stochastic retention protocol.
