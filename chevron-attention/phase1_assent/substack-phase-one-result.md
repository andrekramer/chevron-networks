# Chevron Attention: Retrieval Without Assent

## A clean result in revocable associative recall

Can a neural network retrieve a fact without allowing that fact to determine its answer?

That was the question behind the first Chevron Attention experiment.

The answer was yes.

The model learned to retrieve the correct fact whether it was active, revoked, or restored. A separate channel learned whether that retrieved fact was currently permitted to participate in the answer.

When the fact was active, the model used it. When it was revoked, the model continued to retrieve it but answered `IDK`. When it was restored, the original answer returned.

The internal pattern was clear:

\[
\text{retrieval remains high},\qquad
\text{permission closes},\qquad
\text{permission reopens}.
\]

This is retrieval without assent.

## Separating retrieval from permission

Ordinary attention finds relevant information and allows that information to influence the next state in one operation.

Chevron Attention separates those roles between two channels.

The adaptive channel, \(A\), performs semantic retrieval:

\[
Q=A W_Q,\qquad K=A W_K,\qquad V=A W_V.
\]

The normative channel, \(N\), produces a permission gate for each memory item. For query \(i\) and memory item \(j\):

\[
g_{ij}=\sigma\left(F_G(N_i,N_j)\right).
\]

The output is:

\[
Y_i=\sum_j \alpha_{ij}g_{ij}V_j+(1-m_i)V_{\varnothing},
\]

where:

\[
m_i=\sum_j \alpha_{ij}g_{ij}.
\]

Here, \(V_{\varnothing}\) is a learned abstention value. When permission closes, admitted attention mass falls and the null value can take over.

This makes it possible for the model to enter the defining Chevron Attention state:

\[
\alpha_{ij^*}\approx1,\qquad g_{ij^*}\approx0.
\]

The relevant fact has been found, but it has not been admitted into the answer.

## The experiment

Each example contained a fresh set of random key–value associations:

```text
K7 -> BLUE
K2 -> HORSE
K9 -> GLASS
```

Control instructions could later revoke or restore a fact:

```text
REVOKE K2
```

The sequence ended with a query:

```text
QUERY K2
```

There were three balanced conditions:

- **Active:** return `HORSE`.
- **Revoked:** return `IDK`.
- **Restored:** return `HORSE` after a revoke–restore sequence.

Fact order, key–value assignments, and query targets were randomized. The sequence also contained controls for unrelated keys. All three conditions had the same length.

The two streams were given distinct jobs.

The \(A\)-stream saw the facts and final query, but not the control instructions. It could learn which fact answered the query, but it could not know whether that fact had been revoked.

The \(N\)-stream saw the complete sequence, including `REVOKE` and `RESTORE`. At query time, it produced a scalar permission gate for every memory item.

The answer head received only the gated value mixture and the explicit null value. There was no residual route around the gate.

Training used three losses:

\[
L=L_{\text{answer}}+\lambda_rL_{\text{retrieval}}+\lambda_gL_{\text{permission}}.
\]

The retrieval loss trained \(A\) to find the queried fact in every condition. The permission loss trained \(N\) to close and reopen the relevant gate. The answer loss trained the complete system to produce the correct response.

## The result

Five independently initialized models were trained for 1,000 optimization steps. Each was evaluated on 1,920 fresh, balanced examples, giving 9,600 final evaluation examples.

| Measure | Mean ± standard deviation |
|---|---:|
| Overall answer accuracy | 99.99% ± 0.02% |
| Active answer accuracy | 100% ± 0% |
| Revoked answer accuracy | 100% ± 0% |
| Restored answer accuracy | 99.97% ± 0.07% |
| Retrieval accuracy | 100% ± 0% |
| Attention mass on the target fact | 0.9986 ± 0.0001 |
| Target gate while active | 0.8975 ± 0.0684 |
| Target gate while revoked | 0.2261 ± 0.0846 |
| Target gate after restoration | 0.9007 ± 0.0652 |

The model answered 9,599 of the 9,600 examples correctly. Four runs were perfect. The remaining run missed one restored example.

More importantly, the intended internal mechanism appeared in every run.

The \(A\)-stream placed effectively all its retrieval mass on the correct fact in all three conditions. Revocation did not make the model retrieve a different memory or lose the original association.

Instead, the \(N\)-stream changed whether the retrieved value could participate. The target gate averaged approximately \(0.90\) while active, fell to \(0.23\) while revoked, and returned to \(0.90\) after restoration:

\[
\begin{array}{lll}
\text{active:} & \alpha_{j^*}\approx1, & g_{j^*}\approx0.90,\\
\text{revoked:} & \alpha_{j^*}\approx1, & g_{j^*}\approx0.23,\\
\text{restored:} & \alpha_{j^*}\approx1, & g_{j^*}\approx0.90.
\end{array}
\]

The content remained available throughout. What changed was its permission to influence the answer.

## Recognition without assent

The experiment demonstrates a concrete computational distinction between recognizing relevant information and admitting it into ongoing computation.

The model can, in effect, represent:

> I found the relevant proposition, but it should not currently govern the result.

Revocation does not require deletion. Restoration does not require relearning. The association remains in the retrieval channel while the permission channel changes its current status.

This distinction is directly visible in three quantities:

\[
\alpha_{ij}: \text{what was retrieved},
\]

\[
g_{ij}: \text{what was permitted},
\]

\[
\alpha_{ij}g_{ij}: \text{what participated}.
\]

That is the first clean result for Chevron Attention: relevant content can remain retrievable while its participation is revoked, then return when permission is restored.

This result is deliberately limited in scope: the standard Transformer baseline also achieved 100% task accuracy, the separation was enforced architecturally and trained with explicit auxiliary losses, and the facts lived in the input context rather than slow retained memory. It therefore demonstrates a reliable retrieval–permission mechanism, not a performance advantage, emergent decomposition, catastrophic-forgetting result, or test of IDL consolidation.
