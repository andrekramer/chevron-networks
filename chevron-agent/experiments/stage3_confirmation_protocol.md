# Stage 3 independent checkpoint evaluation protocol

## Purpose

Test whether the initial three-seed Stage 3 advantage for plain Chevron
survives a stricter success definition and evaluation episodes that were not
used to select checkpoints.

## Existing runs

- Training seeds: 0, 1, 2
- Models: recurrent GRU baseline and plain ungated Chevron
- Training budget: 120 PPO updates
- Existing checkpoint interval: 10 updates
- Existing development evaluation: 50 episodes per checkpoint

The existing `best_eval.pt` checkpoints were selected using the original
evaluation metric. That metric incorrectly counted a positive shaped reward on
the last step of a timeout as success. This confirmation does not repeat
checkpoint selection. It reevaluates the saved checkpoints with success
defined as a positively rewarded terminal interaction.

## Independent evaluation

- Primary checkpoint: final update 120
- Primary metric: strict success rate over 200 episodes
- Evaluation episode seeds: 1,000,000 through 1,000,199 for every model
- Secondary metric: mean held-out success across updates 10 through 120
- Diagnostic only: the previously selected `best_eval.pt` checkpoint
- Return is reported separately from success.

The same evaluation episodes are used for all models and training seeds. No
threshold, checkpoint, or configuration will be selected using these episodes.

## Decision

If final-checkpoint Chevron success exceeds the GRU baseline on the paired
three-seed mean without relying on timeout rewards, train two additional
paired seeds under the unchanged Stage 3 configuration. Otherwise, retain the
result only as proof that Chevron recurrence is trainable and move directly to
the delayed-context memory task.
