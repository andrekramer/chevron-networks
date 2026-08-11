# Delayed-context memory experiment

Status: frozen before comparative runs.

## Purpose

This is a small, mechanism-first reinforcement-learning bridge between the
current grid-world proof of trainability and a persistent NPC. It asks whether
an agent can retain established context-action memories while learning new
ones from delayed consequences.

It does not test language, planning, or a complete self. It isolates the part
of the Chevron proposal that should make those later systems safer: retrieval
does not by itself grant read or write permission.

## Task

One lifetime contains twelve latent contexts and two consequential actions.
Eight contexts occur during an establishment phase. Four additional contexts
are introduced later while the old contexts continue to occur.

Each context has a fixed correct action within a lifetime. Correct actions are
randomly assigned, so a memoryless classifier cannot infer the mapping from a
context's appearance.

The consequence of action at time t arrives at t + 3:

- correct action: +1
- incorrect action: -1

The agent must retain enough eligibility information to associate the delayed
outcome with the earlier observation and action. This eligibility queue is
required by every condition and is not counted as a Chevron provisional
buffer.

### Observations

Every context produces two noisy views:

- address evidence, used by Chevron retrieval;
- diagnostic evidence, used by Chevron assent.

The views share context identity but are independently corrupted. All
comparators receive both views. A conventional comparator may concatenate
them, so Chevron is not given extra information.

Contexts are arranged in four related families. Each family contains two old
contexts and one later context. Family similarity makes address retrieval
occasionally ambiguous, while diagnostic evidence can still accept or reject
the proposed memory. This is the intended test of genuinely different
retrieval and assent computations.

### Phases

Default development lifetime:

1. Establishment: 600 decisions sampled from eight old contexts.
2. Introduction: 600 decisions sampled from all twelve contexts, with old and
   new contexts equally represented per context.
3. Flush: three steps deliver the remaining outcomes but do not count as new
   decisions.

The permanent memory has twelve slots. Capacity is therefore sufficient for
all true contexts. Failures cannot be excused as an unavoidable capacity
shortage.

## Chevron computation

For occupied permanent slots j:

    alpha_j = softmax(similarity(Q_A(x_address), K_A(A_mem,j)))

    r_j = sigmoid(k * (theta_read - normalized_mismatch(x_diagnostic, N_mem,j)))

    w_read,j = alpha_j * r_j
    u_j = alpha_j * (1 - r_j)
    q = sum_j u_j

    z = sum_j w_read,j * V_N,j + q * V_null

`q` is unresolved evidence, not automatically novelty. Novel allocation needs
both high total residual and no strongly assenting slot.

Permanent writes use a stricter threshold than reads. Address and value
updates are both gated by write permission. A rejected observation may enter
a bounded provisional buffer; it can be promoted only after its delayed
outcome becomes eligible. Updates to permanent values remain convex.

Empty memory is a defined case: q = 1, z = V_null, and the observation may
enter the provisional buffer.

## Conditions

The first comparison will implement these four conditions:

1. Standard attention: retrieval and read mass are the same; eligible writes
   use a conventional similarity/allocation rule.
2. Standard attention + buffer: same model and capacity with bounded
   provisional storage.
3. Chevron + buffer: separate address retrieval, diagnostic assent, residual
   mass, conservative write permission, and provisional promotion.
4. Chevron immediate write: Chevron read path but no provisional protection.

After those pass basic task-validity checks, two diagnostic ablations are
added:

5. Chevron with only scalar residual q: same equations and allocation trigger,
   but the provisional bank does not retain the per-slot rejected-mass
   signature u_j when associating candidates.
6. Chevron with coupled read and write thresholds.

A generic learned comparator may be added after the fixed mechanics are
stable. It must receive the same inputs and comparable memory capacity.

## Metrics

Primary behavioral metrics:

- old retention: accuracy on old contexts over the final 200 decisions;
- new acquisition: accuracy on new contexts over the final 200 decisions;
- mean return per decision.

Mechanism metrics:

- premature write rate: permanent writes before the relevant outcome is
  eligible;
- established overwrite rate: destructive updates or evictions of established
  old slots;
- residual calibration: mean q on unresolved cases minus mean q on resolved
  cases;
- promotion precision: promoted provisional entries that later support a
  correct decision divided by all promoted entries;
- read/write separation: frequency of admitted reads versus permanent writes
  on incompatible evidence;
- conservation error: absolute error in
  `sum(w_read) + sum(u) = 1` over occupied retrievals.

The diagnostic retention-plasticity score is:

    S = R_old + P_new - 0.5 * O_est - 0.5 * W_prem

The separate terms remain primary; S must not conceal a trade-off.

## Task-validity controls

Before model comparison:

- an oracle-context memory should learn both old and new mappings;
- a memoryless policy should remain near 50% accuracy;
- delay zero and delay three variants should agree once eligibility is handled;
- outcome/context pairing must pass deterministic unit tests;
- no condition may access latent context IDs or correct actions.

## Predictions

Evidence for the Chevron mechanism would be the joint pattern:

1. old retention remains high when new contexts arrive;
2. new acquisition rises after an initial high-q period;
3. q falls for a new context after useful promotion;
4. buffering reduces premature writes and established overwrites relative to
   immediate write;
5. write permission is materially stricter than read admission;
6. changing diagnostic evidence while holding address evidence fixed changes
   assent but not retrieval;
7. changing address evidence while holding diagnostic evidence fixed changes
   retrieval but not the per-slot mismatch computation.

A Chevron performance win without these causal signatures is not sufficient.
If the strong conventional comparator matches Chevron on all behavioral
metrics, the honest conclusion is that this task does not require the Chevron
factorization.

## Seeds and reporting rule

Development uses seeds 0-4 for implementation and parameter sanity checks.
Thresholds are then frozen. Confirmation uses seeds 100-119 and reports mean,
standard deviation, paired differences, and all individual seed results.

No confirmation seed will be used to select thresholds, buffer size,
temperature, or update rates.

## Scope boundary

This first version has two consequential actions and no explicit WAIT action.
High q and low admitted mass provide a measurable internal form of caution.
A later version may add an information-gathering action, but only if taking it
has a real cost and yields information that can rationally reduce uncertainty.
