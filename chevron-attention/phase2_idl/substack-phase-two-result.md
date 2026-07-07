# Inertial Difference Learning: When Should Difference Become Change?

## Phase two of the Chevron Attention experiments

How should a learning system respond when its current experience contradicts what it has retained?

It should not rewrite itself after every disturbance. But it should not remain fixed when the world has genuinely changed.

This is the stability–plasticity problem in its smallest form.

The second Chevron experiment tested whether persistent difference can regulate retained change. A fast adaptive state, \(A\), tracked current evidence. A slower state, \(N\), represented what the system retained. Their difference determined when the slower state was allowed to move.

The main result was clean.

During a brief contradiction, Inertial Difference Learning kept \(N\) almost unchanged. When the same contradiction persisted, its retention gate opened and \(N\) adapted. A fixed slow learning rate could protect against the temporary disturbance or adapt to the sustained change, but no tested fixed rate did both.

The experiment also exposed a failure in the original rule. An absolute difference threshold ignored small but persistent changes. A scale-aware version repaired that failure across a twelve-fold range of change magnitudes without retuning for each condition.

## The IDL rule

The model contains two parameter states.

The adaptive state \(A\) learns quickly from the current data:

\[
A_{t+1}=A_t-\eta_A\nabla L_t(A_t).
\]

The retained state \(N\) changes more slowly. First, the model measures their difference:

\[
D_t=\operatorname{RMS}(A_t-N_t).
\]

It then accumulates persistent difference:

\[
P_t=\beta P_{t-1}+(1-\beta)D_t.
\]

The retention gate is:

\[
\rho_t=\sigma\left(k(P_t-\theta)\right).
\]

Finally, the retained state follows the adaptive state only through that gate:

\[
N_{t+1}=N_t+\eta_N\rho_t(A_t-N_t).
\]

This produces the intended qualitative behaviour:

\[
\text{brief difference}\rightarrow\rho\approx0,
\]

\[
\text{persistent difference}\rightarrow\rho\approx1.
\]

The mechanism does not ask whether difference exists. Difference is expected. It asks whether that difference has lasted long enough to deserve retention.

## The experiment

The task used a small online linear predictor. This made the state dynamics directly measurable without representation learning obscuring the result.

The predictor began from an already learned base mapping and passed through four phases:

1. 600 stable steps on the base mapping;
2. a 10-step contradictory disturbance;
3. 200 steps back on the base mapping;
4. a 300-step sustained change to the contradictory mapping.

The stable phase established the model's ordinary online variation. The short contradiction tested retention. The later sustained change tested plasticity.

Every method received identical data and used the same fast learner \(A\). Only the retained-state rule changed.

The comparisons were:

- **Absolute IDL:** persistence-gated retention using raw \(A-N\) difference;
- **Always slow:** \(N\) follows \(A\) at every step using IDL's maximum slow rate;
- **Fixed low rate:** \(N\) always follows \(A\), but at a smaller constant rate;
- **Fast only:** no distinct retained state; the model uses \(A\) directly.

The default experiment was repeated with five independently generated data streams.

## Protection followed by adaptation

The first question was whether IDL could resist the brief contradiction without becoming unable to learn later.

| Metric | Absolute IDL | Always slow | Fixed low rate | Fast only |
|---|---:|---:|---:|---:|
| Retained-state drift during brief contradiction | 0.0018 | 0.2174 | 0.0907 | 1.6609 |
| Steps to adapt after sustained change | 137.0 | 120.0 | 291.0 | 14.2 |
| Final sustained-change error | 0.0101 | 0.0053 | 0.1849 | 0.0018 |

IDL reduced temporary movement of \(N\) by more than one hundredfold compared with the full-rate always-slow model. It then adapted to the sustained change only 17 steps later.

The lower fixed-rate model reduced temporary drift, but paid for that protection throughout the experiment. It required 291 steps to adapt and still ended with substantially higher error.

This is the central phase-two result.

IDL did not merely choose a slower compromise. It changed its effective retention rate according to the persistence of the discrepancy.

## Duration changes the decision

The duration sweep varied the contradictory disturbance from 2 to 80 steps.

| Contradiction length | IDL drift | Always-slow drift | Drift reduction | Mean IDL gate |
|---:|---:|---:|---:|---:|
| 2 | 0.0001 | 0.0206 | 99.7% | 0.003 |
| 5 | 0.0003 | 0.0807 | 99.6% | 0.004 |
| 10 | 0.0018 | 0.2174 | 99.2% | 0.007 |
| 20 | 0.0286 | 0.5161 | 94.5% | 0.039 |
| 40 | 0.4932 | 1.0046 | 50.9% | 0.355 |
| 80 | 1.3243 | 1.5563 | 14.9% | 0.674 |

This is the predicted IDL response.

Two-, five-, and ten-step disturbances were treated as temporary. At forty and eighty steps, the same contradiction increasingly counted as persistent evidence and the gate opened.

The model did not need an explicit token saying `TEMPORARY` or `PERMANENT`. The temporal structure of the difference changed the retention decision.

## Why a fixed rate is not enough

One obvious alternative is simply to make \(N\) update very slowly.

The fixed-rate sweep showed the resulting frontier:

| Fixed update rate | Brief drift | Adaptation steps | Final error |
|---:|---:|---:|---:|
| 0.002 | 0.0231 | Did not adapt | 1.0938 |
| 0.004 | 0.0460 | Did not adapt | 0.6031 |
| 0.008 | 0.0907 | 291.0 | 0.1849 |
| 0.012 | 0.1341 | 196.0 | 0.0566 |
| 0.020 | 0.2174 | 120.0 | 0.0053 |

Smaller rates protected retained structure by making all learning slow. Larger rates adapted successfully by allowing temporary disturbances to move \(N\).

No tested constant rate matched IDL's combination of 0.0018 temporary drift, 137-step adaptation, and 0.0101 final error.

## The absolute-threshold failure

The duration result was strong, but the magnitude sweep exposed a problem.

The original IDL rule used an absolute threshold \(\theta\). Large persistent differences crossed it. Small persistent differences did not.

| RMS change | Absolute IDL adaptation | Final error |
|---:|---:|---:|
| 0.25 | Did not adapt | 0.1336 |
| 0.50 | Did not adapt | 0.0619 |
| 1.00 | 169.2 steps | 0.0234 |
| 2.00 | 137.0 steps | 0.0101 |
| 3.00 | 128.8 steps | 0.0098 |

A small change can be real even when its magnitude is small. Persistence should distinguish it from ordinary noise, but an absolute threshold confuses small persistent change with insignificant difference.

That required a change to the mechanism.

## Scale-aware persistence

The revised model estimates the ordinary discrepancy scale \(S_t\) during stable operation.

Difference inside a configurable noise margin produces no persistence evidence. Difference beyond that margin becomes a bounded signal:

\[
E_t=\operatorname{clip}\left(\frac{D_t}{mS_t}-1,0,1\right).
\]

Persistence then accumulates \(E_t\), rather than raw difference:

\[
P_t=\beta P_{t-1}+(1-\beta)E_t.
\]

The scale estimate updates while discrepancy remains ordinary. When an anomaly appears, the estimate freezes. This prevents a persistent change from redefining itself as normal before the retention gate opens.

The same parameters were then tested across RMS changes from 0.25 to 3.0.

| RMS change | Absolute adaptation | Scale-aware adaptation | Absolute final error | Scale-aware final error |
|---:|---:|---:|---:|---:|
| 0.25 | Did not adapt | 148.6 steps | 0.1336 | 0.0013 |
| 0.50 | Did not adapt | 147.0 steps | 0.0619 | 0.0024 |
| 1.00 | 169.2 steps | 145.8 steps | 0.0234 | 0.0045 |
| 1.50 | 146.0 steps | 145.0 steps | 0.0133 | 0.0066 |
| 2.00 | 137.0 steps | 144.6 steps | 0.0101 | 0.0086 |
| 3.00 | 128.8 steps | 134.6 steps | 0.0098 | 0.0106 |

Scale-aware IDL consolidated every tested change magnitude without condition-specific tuning. Its adaptation time remained between approximately 135 and 149 steps across the entire twelve-fold range.

Temporary protection also remained intact. At the default RMS-2 change, brief retained-state drift was 0.0016, compared with 0.2174 for the always-updating slow model.

The result remained stable for noise margins from 3 to 6 and scale-estimator update rates from 0.002 to 0.05.

## Retained change under inertia

The phase-two experiment gives a concrete meaning to the phrase:

> Learning is the controlled retention of difference under inertia.

The fast state changes because current evidence demands it. The retained state does not immediately follow. Difference is allowed to exist while the system determines what kind of difference it is.

If the contradiction disappears, retained structure largely survives. If the contradiction persists, the retention gate opens. If the magnitude changes, scale-aware persistence asks whether the difference is unusual relative to ordinary variation rather than whether it exceeds one universal absolute value.

The useful object is therefore not difference alone:

\[
\text{retained change}=f(\text{difference},\text{duration},\text{scale}).
\]

That is a more precise version of the IDL hypothesis.

This result remains limited to a controlled online linear system initialized from a learned base mapping, with abrupt regime changes, stationary noise, hand-selected timescales, and direct access to parameter-space difference. It does not yet establish an advantage in nonlinear representation learning, Transformers, gradual drift, nonstationary noise, or real continual-learning benchmarks; it establishes that persistence-gated retention produces the predicted stability–plasticity behaviour in the minimal setting, and that scale normalization is necessary for small persistent changes.
