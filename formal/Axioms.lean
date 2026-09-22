import QuadratureField

/-! Axiom audit: every main theorem must depend only on `propext`, `Classical.choice`
and `Quot.sound` (no `sorryAx`). Run with `lake env lean Axioms.lean`. -/

open QuadratureField

#print axioms mmdSq_expand
#print axioms worstCase_error
#print axioms crit_exact
#print axioms mmd_triangle
#print axioms twoArm_chain
#print axioms mmdSq_reweight_le
#print axioms deployed_weights
#print axioms ridgeMin_unique
#print axioms exact_selection
#print axioms neverWorse_exact
#print axioms floor_identity
#print axioms floor_identity_unitDiag
#print axioms selfAffinity_le
#print axioms engine_identity
#print axioms paired_diff
#print axioms witness_bound_of_bounded
#print axioms selection_excess_of_conc
#print axioms certified_sign_of_conc
#print axioms slack_zero_of_coincide
#print axioms certified_selection_of_events
#print axioms certified_selection_min
#print axioms hoeffding_pair_event
#print axioms pair_gain_paper
#print axioms strict_gain_sharper
#print axioms strict_gain
#print axioms spectralRatio_le
#print axioms spectralRatio_pos
#print axioms NoTargetFree.expected_gain_le
#print axioms NoTargetFree.no_target_free_constant
