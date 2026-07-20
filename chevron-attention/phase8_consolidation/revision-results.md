# Phase 8 focused revision result

## Outcome

The revised Phase 8 mechanism passed every predeclared development-readiness
check and is now frozen for confirmation on new seeds.

Two changes were made in response to the first robustness matrix:

1. observations already matching retained memory are rejected before they can
   occupy a provisional candidate slot; mature non-distinct candidates are
   also rejected and cleared;
2. the joint controller's raw novelty score is expressed as a clipped fraction
   of its already-calibrated Phase 7 threshold in odds space. This adds no
   learned parameters and preserves score ordering.

The consolidation threshold was set to 0.20, with the hard minimum support
unchanged at five observations. These choices were made on development seeds
and must not be changed during confirmation.

## Focused five-seed result

| Condition | Method | Old accuracy | New accuracy | Categories retained |
|---|---|---:|---:|---:|
| Blocked | Chevron immediate | 1.000 | 0.667 | 2.00 / 3 |
| Blocked | Chevron quarantine | **1.000** | **0.933** | **2.80 / 3** |
| Blocked | Joint immediate | 1.000 | 0.867 | 2.60 / 3 |
| Blocked | Joint quarantine | 1.000 | 0.800 | 2.40 / 3 |
| Interleaved | Chevron immediate | 1.000 | 0.667 | 2.00 / 3 |
| Interleaved | Chevron quarantine | **1.000** | **0.933** | **2.80 / 3** |
| Interleaved | Joint immediate | 1.000 | 0.867 | 2.60 / 3 |
| Interleaved | Joint quarantine | 1.000 | 0.600 | 1.80 / 3 |
| Near, one flip | Chevron immediate | 1.000 | 0.467 | 1.40 / 3 |
| Near, one flip | Chevron quarantine | **1.000** | **0.867** | **2.60 / 3** |
| Near, one flip | Joint immediate | 1.000 | 0.000 | 0.00 / 3 |
| Near, one flip | Joint quarantine | 1.000 | 0.067 | 0.40 / 3 |

Joint quarantine reached the exact predeclared 60 percent interleaved
acquisition boundary. This is sufficient to proceed to confirmation, but it is
not a strong joint result. Its purpose here is to show that the shared
provisional bank does not reduce the controller to complete non-learning.

## Temporal boundary

Both quarantine methods produced zero false consolidation for evidence lasting
one, three, or four observations. Both began consolidating at observation
five. Old-category accuracy remained 1.000 throughout.

Thus the revision improved plasticity without moving the declared temporal
boundary or trading away measured stability.

## Candidate-state correction

Mean Chevron candidate replacements in the interleaved stream fell from 38.6
to 2.2. The system rejected 94.2 familiar observations at the retained-memory
boundary, but those observations no longer created or displaced candidate
state. This is filtering traffic rather than provisional-memory churn.

Chevron retained 3, 3, 2, 3, and 3 interleaved categories across the five
development seeds. The remaining missed category occurred on one seed and did
not damage old memory.

## Locked interpretation

The development evidence now supports taking the following hypotheses into
confirmation:

- temporal quarantine distinguishes brief contradiction from persistent
  novelty at an explicit support boundary;
- slot-specific Chevron assent remains necessary to protect retained writes;
- Chevron quarantine is particularly useful when novel categories closely
  resemble retained categories;
- provisional memory is a general temporal mechanism, but the quality of its
  plasticity still depends on the upstream novelty signal.

These are hypotheses, not confirmed claims. The exact configuration is stored
in `locked-config.json`. Confirmation must use new seeds and report all failures
without retuning.

