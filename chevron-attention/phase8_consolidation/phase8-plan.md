# Phase 8 plan: provisional memory and consolidation

## Purpose

Phase 8 should close the current synthetic research programme rather than open
another indefinite sequence of architecture variants.

It has two objectives:

1. test the one extension that follows directly from Phase 7: route genuinely
   unassented evidence into provisional memory before it can alter retained
   structure;
2. consolidate Phases 1–7 into a final claim ledger stating what the evidence
   supports, what it does not support, and what belongs to future research.

The phase ends after a locked confirmation experiment. Kimi Linear, Attention
Residuals, DeepSeek sparse/latent attention, language-model scaling, and a
coherently evolving agent self remain future research directions rather than
being added to this phase.

## Starting point

Phase 7 leaves the learned read and write rule:

```text
alpha_j = softmax(Q_A K_A^T)_j
r_j = sigmoid(k * (theta - M(A,N_j)))
w_j = alpha_j * r_j
q = 1 - sum_j(w_j)
```

`w_j` is admitted mass and controls the write to retained slot `j`. `q` is the
conserved remaining mass.

The closure experiments established that `alpha * r` produces substantially
cleaner writes than alpha alone, a fixed raw gate, or a slot-misaligned gate.
They did not establish a general final-accuracy advantage over a
parameter-matched joint-attention controller. Chevron's strongest performance
result occurred when novel and retained categories were unusually close.

## Important correction: remaining mass is not novelty by itself

Phase 7 often produced high remaining mass even on correct matched reads. A
retrieved A group contained several plausible members; once `r` removed the
decoys, their attention mass became null rather than being redistributed.

Therefore:

```text
q = conserved unused mass
```

does not imply:

```text
q = probability that the observation is novel
```

Phase 8 must keep these meanings separate.

Define the best assent among the slots that A actually retrieved:

```text
r_best = max r_j over the top-A candidate set
nu = 1 - r_best
u = q * nu
```

`nu` is a novelty or non-assent score. `u` is the portion of remaining mass
eligible to enter provisional memory. A smooth maximum can replace `max` if an
end-to-end differentiable version is later required.

The top-A set, smoothness, and threshold are development parameters. They must
be frozen before confirmation.

## Proposed provisional-memory rule

Use a small bank of fast candidate slots so interleaved novel observations do
not overwrite one another. Every compared method receives the same candidate
capacity and state budget.

For candidate `m`, compute coherence with the rejected observation:

```text
c_m = sigmoid(k_c * (theta_c - M(x, C_m)))
```

Assign the observation to the most coherent candidate. If no candidate is
coherent enough, initialize or replace the lowest-persistence candidate.

For the selected candidate:

```text
e_m = u * c_m
C_m <- (1 - eta_C * e_m) * C_m + eta_C * e_m * x
P_m <- beta * P_m + (1 - beta) * e_m
```

Unselected candidates decay:

```text
P_l <- beta * P_l
```

Consolidate a candidate into retained memory only when:

```text
P_m >= tau_P
and observations_m >= minimum_support
and the candidate remains distinct from every retained template
```

This is a soft, learned-attention version of the persistence principle tested
in earlier IDL and category experiments. It is ART-inspired but does not add
hard ordered reset/search.

The first implementation should use no additional learned network. The
candidate dynamics should be transparent and shared by all relevant methods.

## Three-timescale interpretation

The provisional mechanism has a close functional relationship to Grossberg's
medium-term memory, but the correspondence should remain precise:

- current A activation is STM-like present activity;
- candidate content `C_m` is a fast provisional representation;
- persistence `P_m` is an MTM-inspired consolidation or eligibility state;
- retained templates `N_j` are LTM-like learned structure.

`P_m` accumulates coherent evidence. Grossberg's canonical MTM equations often
describe habituative transmitter depletion: a gate weakens during continued
activity and later recovers, helping reset and attentional search. These are
not the same dynamics. A separate habituative variable may be tested as an
ablation if the core experiment leaves a reset/search failure, but it is not
part of the initial Phase 8 mechanism.

## What Phase 8 will test

The benchmark should distinguish four cases that immediate allocation can
confuse:

### 1. Isolated contradiction

A single observation conflicts with retained memory, then disappears.

Desired behaviour: place it provisionally, do not change retained memory, and
allow the candidate to decay.

### 2. Short coherent disturbance

The same conflicting pattern lasts for a small block and then the original
category returns.

Desired behaviour: track the temporary pattern without consolidating it or
splitting the retained category.

### 3. Sustained novel category

A genuinely new category persists and recurs.

Desired behaviour: incur a finite acquisition delay, then allocate it and
retain it.

### 4. Near-category novelty

The persistent novel category differs from a retained category by only one
feature component.

Desired behaviour: protect the old template while eventually consolidating the
new one. This is the decisive condition suggested by Phase 7.

The stream should include clean final probes and an unseen noisy probe. Writes
must remain enabled during the learning phases and disabled during probes.

## Methods

The minimum comparison is:

1. `chevron_immediate`: learned `alpha * r` with Phase 7 immediate allocation;
2. `chevron_quarantine`: the same trained network plus provisional memory;
3. `joint_immediate`: the parameter-matched learned joint write controller;
4. `joint_quarantine`: the same joint controller plus the identical
   provisional-memory bank;
5. `alpha_quarantine`: Chevron novelty and candidate logic, but alpha-only
   retained writes;
6. `oracle`: label-informed allocation and correct-slot writes.

The comparison between immediate and quarantine isolates the value of temporal
persistence. The comparison between Chevron and joint quarantine asks whether
the benefit belongs to provisional memory generally or to the A/N assent rule.
The alpha ablation asks whether `r` still protects retained slots after both
systems receive the same temporal buffer.

All methods must receive:

- the same retained and provisional memory capacity;
- the same stream and observation order;
- the same write-rate budget;
- the same threshold-selection budget;
- the same supervision available at each step;
- comparable parameter counts.

## Metrics

### Stability

- retained-category accuracy after isolated contradictions;
- retained-category accuracy after short coherent disturbances;
- false consolidations from transient evidence;
- false category splits;
- category evictions and memory churn;
- wrong-category retained-write mass;
- retained-template drift and template MSE.

### Plasticity

- sustained-novel online accuracy;
- probability that a sustained category is eventually consolidated;
- observations required before consolidation;
- final new-category accuracy;
- failure to learn genuine novelty.

### Provisional-state behaviour

- candidate persistence during a transient block;
- decay time after the disturbance ends;
- candidate coherence and purity;
- candidate replacements caused by interleaved novelty;
- mass routed to provisional memory.

### Overall

- final clean and shifted accuracy;
- number of distinct correct categories retained;
- paired per-seed difference from the learned joint controller;
- distance from the oracle stability–plasticity frontier.

The primary visual should be a stability–plasticity plot: false consolidation
on one axis and sustained-novel acquisition delay on the other. A method is
better only if it improves one without an unacceptable loss in the other.

## Experimental stages

### Phase 8.0: freeze the inherited system

- Reproduce the Phase 7 closure result from the current code.
- Freeze the learned Soft Chevron and parameter-matched joint architectures.
- Preserve answer-only training for Chevron and the favorable direct
  no-match supervision for the joint write controller.
- Save all per-seed outputs in a machine-readable file rather than relying only
  on printed summaries.

No Phase 7 architecture or baseline should be changed after this point unless
a correctness bug is found.

### Phase 8.1: implement provisional memory

- Add the candidate bank, coherence update, persistence decay, and consolidation
  rule.
- Keep residual mass `q` and novelty `nu` separately observable.
- Log every retained write, provisional write, consolidation, rejection,
  replacement, split, and eviction.
- Add conservation, decay, allocation, and state-budget tests.
- Add deterministic unit streams for isolated, transient, and sustained cases.

### Phase 8.2: development sweep

Use development seeds that are not part of the final confirmation set.

Sweep only parameters that define the provisional mechanism:

- top-A candidate size or cumulative A mass;
- candidate coherence threshold;
- candidate update rate;
- persistence decay `beta`;
- consolidation threshold `tau_P`;
- minimum supporting observations;
- provisional-bank size.

Do not select parameters by final accuracy alone. Choose a point on the
stability–plasticity frontier using a predeclared objective that penalizes both
false consolidation and failure or excessive delay in learning sustained
novelty.

The joint and Chevron methods receive the same search budget. Prefer shared
candidate parameters unless the data clearly show that a common scale is
invalid; any method-specific calibration must be reported.

### Phase 8.3: locked confirmation

Freeze the mechanism and run new seeds not used in Phases 6, 7, or development.

Recommended confirmation budget:

- 20 paired seeds for the default and near-category conditions;
- 10 paired seeds for each robustness condition;
- bootstrap 95 percent confidence intervals for paired differences;
- win, loss, and tie counts alongside means and standard deviations.

Robustness conditions:

- transient duration sweep;
- observation noise 0.06, 0.08, and 0.10;
- category distance of one, two, and three flipped components;
- retained capacity below, equal to, and above required category count where
  the fixed model shape permits it;
- blocked and interleaved novelty;
- doubled stream duration.

No thresholds may be retuned on confirmation results.

### Phase 8.4: consolidation report

Produce:

- a concise methods document;
- raw per-seed JSON or CSV results;
- stability–plasticity frontier and memory-state timeline figures;
- ablation and robustness tables;
- a final claim ledger covering Phases 1–8;
- a plain-text Substack follow-up only if Phase 8 adds a result beyond the
  Phase 7 report.

## Predeclared hypotheses

### H1: quarantine protects stability

Compared with immediate allocation, provisional memory will reduce false
consolidations, splits, evictions, and retained-template drift after isolated
and short coherent disturbances.

### H2: quarantine preserves plasticity

Sustained coherent novelty will still be consolidated after a measurable delay.
If false consolidation falls only because the system stops learning new
categories, the extension has failed.

### H3: assent remains causal

Within the same provisional-memory system, `alpha * r` will send less write mass
to incorrect retained slots than alpha-only writes and will preserve more old
categories.

### H4: the specific near-category advantage replicates

Chevron quarantine will outperform the parameter-matched joint quarantine
system when new and retained categories differ by only one component.

### H5: no general superiority is assumed

The default expectation is that the learned joint controller may tie Chevron
on broad final accuracy. A general Chevron advantage will be claimed only if a
paired confirmation interval excludes zero across more than one condition.

## Decision rules

### Outcome A: quarantine helps both systems equally

Conclusion: provisional persistence is the main stability–plasticity mechanism.
Chevron supplies a cleaner interpretation and write decomposition but no unique
performance advantage.

### Outcome B: Chevron retains a replicated near-category advantage only

Conclusion: Chevron is a specific inductive bias for protecting close retained
categories while novelty consolidates. This is the current leading hypothesis.

### Outcome C: Chevron wins broadly after confirmation

Conclusion: learned A/N assent provides a more general continual-memory
advantage than Phase 7 established. The claim remains limited to controlled
synthetic memory tasks until tested on less synthetic data.

### Outcome D: quarantine prevents sustained learning

Conclusion: the proposed residual-to-provisional rule is too conservative and
should be rejected or left as an unvalidated future direction. Do not tune on
the confirmation set.

### Outcome E: the joint controller wins

Conclusion: explicit A/N structure is not the best implementation of the tested
write-control problem. Preserve the mechanistic observations but do not claim a
Chevron performance benefit.

## Final claim ledger to produce

The final report should classify every major proposition as established,
supported in a limited regime, not supported, or untested.

### Already supported

- retrieval can remain active while assent is withdrawn;
- persistent difference can regulate slower retained change in controlled
  online tasks;
- learned soft A/N matching can arise from answer supervision;
- independently initialized A/N projections can learn a useful comparison
  space;
- explicit null routing is architecturally identifiable even when standard
  attention produces the same answers downstream;
- `alpha * r` sharply reduces cross-category write mass;
- learned slot-specific admission beats fixed and misaligned gates;
- Chevron has a promising near-category interference result.

### Not supported as general claims

- Chevron is universally more accurate than standard attention;
- full ART reset/search is required;
- fixed vigilance is robust;
- cleaner internal writes always produce higher short-horizon final accuracy;
- the current experiments demonstrate biological equivalence, consciousness,
  agency, or safety.

### Phase 8 should decide

- whether rejected mass is usefully held as provisional state;
- whether temporal quarantine improves the stability–plasticity frontier;
- whether that improvement is specific to Chevron or shared by a learned joint
  controller;
- whether the near-category result replicates under locked parameters and new
  seeds.

### Future research after Phase 8

- legitimate revision of an existing retained belief rather than only category
  allocation;
- end-to-end training through long write trajectories;
- label-free value and category formation;
- realistic episodic or agent-memory data;
- prompt-injection, provenance, rollback, and safety-constraint tests;
- Kimi-style delta memory and Attention Residual comparisons;
- DeepSeek-style latent or sparse retrieval comparisons;
- efficiency and scaling in larger deep networks;
- a formally defined persistent computational self for agency and safety.

## Stopping rule

After the locked confirmation and claim ledger, stop tuning the current
synthetic category benchmark.

If the result is positive, the next research programme should move to less
synthetic persistent-agent memory. If it is negative or tied, report that
result and retain only the mechanisms directly supported by the evidence.

The purpose of Phase 8 is to finish with a conclusion, not to guarantee a win.
