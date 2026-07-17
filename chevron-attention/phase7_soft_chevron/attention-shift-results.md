# Phase 7.4 results: the standard-attention litmus test

Phase 7.3 showed that Soft Chevron could learn independently initialized A/N
matching projections and remain accurate under unseen representation noise.
The missing comparison was whether its separate retrieval and admission stages
made it more robust than ordinary joint A/N attention.

This experiment trains both systems from answer labels only for 700 steps and
evaluates the same five seeds on fresh held-out memories. Soft Chevron starts
with independent A/N matching projections. The joint-attention baseline keeps
its favorable shared N-query/N-key initialization and has 4,254 parameters,
compared with Chevron's 3,759.

## Answer accuracy

| Condition | Soft Chevron | Joint attention |
|---|---:|---:|
| Clean train and test | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Train and test with 0.05 noise | 0.9998 ± 0.0002 | 0.9998 ± 0.0002 |
| Clean train, unseen 0.08 test noise | 0.9887 ± 0.0029 | 0.9888 ± 0.0025 |

The standard-attention litmus test is therefore negative on final accuracy.
Neither method has a meaningful performance advantage in the tested
conditions.

## Unseen-noise breakdown

| Measurement | Soft Chevron | Joint attention |
|---|---:|---:|
| Matched accuracy | 0.9879 ± 0.0033 | 0.9872 ± 0.0032 |
| No-match accuracy | 0.9908 ± 0.0052 | 0.9934 ± 0.0043 |
| Target-group mass selects correct group | 0.9955 ± 0.0018 | 0.9562 ± 0.0061 |
| Matched null mass | 0.7812 ± 0.0046 | 0.0037 ± 0.0002 |
| No-match null mass | 0.8994 ± 0.0021 | 0.0067 ± 0.0002 |

The equal answer accuracy hides very different internal solutions. Chevron's
A-only retrieval remains stable while noise affects its separate A/N admission
stage. It sends about 90% of unmatched mass to the explicit null value. Joint
attention combines A and N in one softmax, so its group selection degrades
under N-channel noise. It nevertheless produces the correct answer through
the learned value and answer projections while assigning less than 1% mass to
its null slot.

This repeats the earlier mechanism-bypass finding under distribution shift:
ordinary attention can solve the labels without representing veto and null
routing in the intended place.

## Conclusion

The result argues for a narrower Chevron claim:

> On this task, Soft Chevron matches a somewhat larger joint-attention baseline
> under clean, trained-noise, and unseen-noise conditions while preserving an
> explicit retrieval, assent, and null-routing decomposition.

It does not show that Chevron is more accurate or generally more robust than
standard attention. Its demonstrated advantage is architectural: the
stability/plasticity decision remains inspectable and localized instead of
being recoverable only from the final output.

If performance superiority remains a goal, the next experiment must make that
decomposition operationally necessary. Online category writes, irreversible
memory contamination, or continual distribution shifts would test whether an
explicit veto protects retained structure in a way that a one-pass classifier
cannot simply compensate for downstream.
