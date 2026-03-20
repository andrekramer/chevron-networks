Empirical Addendum: The Long-Context Veto Task

To test whether the distinction between these three regimes actually matters in practice, I set up a small, targeted sequence-modeling experiment.

If the theory holds, an unconstrained network (Free Chevron) should struggle with long-term memory; a rotation-based network (Complex Chevron / Mamba-like) should track state beautifully but struggle to deliberately erase it; and a functionally paired network (Structured Chevron) should be able to do both.

The Experiment
I built a “Long-Context Veto Task.” The model is fed a sequence of 40 tokens one by one.





Tokens 1 and 2 represent evidence (+1 or -1).



Token 0 is background noise (ignore).



Token 3 is a Veto / Reset token. It means: “Forget everything you heard before this moment. Start over.”

The goal is to read the 40-token sequence and predict the sign of the cumulative evidence only after the final Veto token. This task specifically stresses two competing demands: stable memory (carrying evidence across dozens of steps without it fading) and active control (instantly erasing that memory when the context demands it).

I trained three small, parameter-matched Recurrent Neural Networks (RNNs), each using a different Chevron regime as its recurrent cell:





Free Chevron RNN: The 2×2 operators are completely unconstrained real matrices.



Complex Chevron RNN (Mamba-like): The 2×2 operators are mathematically constrained to pure rotation, evolving the state by shifting its phase angle.



Structured Chevron RNN (Self / World): One channel explicitly tracks the memory state (”Self”), while the other channel operates a physical sigmoid relevance gate over the first (”World”).

The Results
After training on 8,000 sequences and evaluating on 2,000 unseen sequences, the difference in regimes became starkly visible:





Free Chevron: 77.8% Accuracy



Complex Chevron: 99.1% Accuracy



Structured Chevron: 100.0% Accuracy

What this tells us
The 77.8% baseline is the familiar failure of ordinary recurrent networks. Over 40 time steps, unconstrained matrix multiplication suffers from vanishing and exploding gradients. The Free Chevron physically lost track of the early evidence. It is, as suspected, just unstructured recurrent baseline on this task.

The massive jump to 99.1% for the Complex Chevron strongly supports the intuition that structured rotational dynamics help state tracking (a core insight of models like Mamba-3). By constraining the update to a rotation, the network preserves the magnitude of the state vector indefinitely. It remembered the evidence flawlessly across time. But why didn’t it score 100%? Because a pure rotation cannot easily shrink a vector to zero. When the Veto token demanded a hard reset, the Complex Chevron had to awkwardly spin its phase angle to approximate an empty state, occasionally leaving “ghost” memories that ruined the final count.

The Structured Chevron achieved a perfect 100.0% because it possessed what pure geometry lacks: functional semantics. By assigning the “World” channel the role of an active control gate, it cleanly separated tracking from intervention. It used the “Self” channel to stably accumulate evidence, but when the Veto token appeared, the “World” channel simply snapped the gate to 0.0, mathematically flatlining the memory and instantly resetting the state.

The Takeaway
Complex numbers and rotational dynamics give a network stable memory. But treating the two channels as functional opposites—a structured tension between content and control—gives the network agency over that memory.

This is exactly why the Structured regime is the most promising frontier. When we stop treating the two channels merely as geometric partners and start treating them as interacting roles—Self and World, Policy and Value, Proposal and Veto—the Chevron architecture moves from a mathematical curiosity to a genuinely controllable cognitive substrate.

(with Gemini 3.1 Pro Preview)

This is not just a comparison of mathematical forms; it is also a comparison of inductive biases. The structured regime has a mechanism especially suited to the task: the World channel drives a sigmoid gate that can squash the Self memory. That is exactly why it wins here. That is not a weakness — it is the point of the experiment.





Figure 5. A small Long-Context Veto experiment comparing the three Chevron regimes. 

On a 40-step sequence task requiring both persistent evidence accumulation and hard reset after veto tokens, the three regimes separate cleanly: Free Chevron struggles, Complex Chevron preserves state well, and Structured Chevron adds explicit control over memory.


