# Elo–Persona Coupling Audit

The opponent Elo model previously fed its point estimate directly into the weak-opponent Punish trigger. This was unsafe during the first moves of a game: the 1500 prior has a wide confidence band, but a single +250 cp advantage could satisfy the point-estimate threshold and cause premature Punish behavior.

The trigger now uses an upper confidence bound. The weak-opponent branch requires `target_elo + confidence + 500` to remain below the effective engine ceiling, in addition to the evaluation lead. Fresh or uncertain opponents therefore remain in Match, while sustained evidence of weak play can still enter Punish. Fresh blunders continue to use the separate immediate Punish emergency path.

Match target Elo now includes one quarter of the confidence band. With little evidence, the target is biased modestly upward toward safer, stronger play; as evidence accumulates and confidence narrows, the target converges toward the established `estimate + 60` policy. Saturating arithmetic prevents integer overflow from malformed or extreme model state.

Regression tests cover both premature-Punish prevention and the expected reduction in uncertainty contribution after sustained observations.
