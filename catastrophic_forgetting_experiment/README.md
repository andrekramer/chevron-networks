Experiment Note: Structural Elimination of Catastrophic Forgetting via 2×2 Chevron Routing
Date: February 24, 2026
Author: Andre
Topic: Continual Learning, Category Veto, and Inertial VSR
1. The Hypothesis
Standard Artificial Neural Networks (MLPs) suffer from catastrophic forgetting. When the environment changes and a network must learn a new rule exception (a "Category Veto"), backpropagation overwrites the globally tangled scalar weights, destroying the core foundational knowledge.
In the Chevron Network notes, we proposed a structural solution:
"A chevron doesn’t just scale a signal, it can separate and transform two coupled streams... keep the diagonals as a retained backbone, let off-diagonals learn category veto and routing... retention is built into the wiring."
This experiment tests whether a 
2
×
2
2×2
 Chevron layer, constrained by phase-based gradient routing (Inertial Variation-Selection-Retention), can perfectly retain a base skill while learning a contextual override.
2. Experimental Design
We designed a 3-Phase synthetic curriculum. The model is forced to rely purely on the 
ℓ
=
x
+
−
x
−
ℓ=x 
+
 −x 
−
 
 oppositional collapse (no unconstrained MLP readout heads are permitted).
Phase 1 (The Base Concept): The network learns a simple "Opposites Match" rule. Context tags are strictly 0.0.
Phase 2 (The Category Veto): The environment shifts. Context tags become active (-1.0 or 1.0). If the context is negative, the network must VETO the match, regardless of the inputs.
Phase 3 (Catastrophic Forgetting Test): The network is evaluated again on Phase 1 data (Context = 0.0) to see if learning the Phase 2 veto destroyed its Phase 1 baseline.
3. Architectural Implementation (Inertial VSR)
To test the thesis, we applied strict gradient hooks to the ChevronLinear's 
2
×
2
2×2
 matrices during training:
Phase 1 (Core Formation): We froze the off-diagonals and the Antithesis channel. The network was forced to build the Base Concept entirely within the Thesis channel (w00).
Phase 2 (Contextual Veto): We froze the Core (w00). The network learned the new Category Veto entirely by routing the new Context signals into the Antithesis channel via the cross-couplings (w11, w01). We strictly prevented the Value channel from leaking into the Antithesis channel (w10 = 0.0).
4. Results
Across multiple initializations and reruns, the Baseline MLP's retention of the Base Concept varied significantly (ranging from severe degradation to ~15% memory loss), depending on how the optimizer happened to scramble its scalar weights.
The Chevron Network achieved 100.0% retention every single time.
Model	Phase 1 Baseline Accuracy	Retention after Phase 2	Memory Loss
Baseline MLP	100.0%	~50.0% to 84.4% (varies by seed)	15% - 50%
Chevron Net	100.0%	100.0% (Constant)	0.0%
5. Conclusion & Mechanics
The Chevron Network did not just empirically outperform the MLP; it structurally guaranteed the outcome.
Because the 
2
×
2
2×2
 geometry allowed us to physically separate Representation (the diagonals) from Contextual Relation (the off-diagonals), the Base Concept was walled off. When Phase 3 fed data where Context = 0.0, the Phase 2 cross-couplings mathematically vanished (
w
×
0
=
0
w×0=0
), cleanly revealing the perfectly untouched Phase 1 core logic beneath.
Takeaway: This serves as a hard empirical proof-of-concept for Inertial VSR. A slow core + fast contextual modulation is a viable path to continual learning without catastrophic drift.
Next Steps:
Port this 2-Phase VSR gradient constraint to the primary train.py repository to test it on high-dimensional WordNet data.
Profile the 
2
×
2
2×2
 operator packed into 4x FP8 using Triton to measure inference bandwidth advantages.
