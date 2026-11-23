# SDD Progress Ledger — RQ1-RQ3 Extension

Branch: feature/rq1-rq3-extension
Merge base (branch start): 01b214dd7707a5678dd766e69dbd95e5f0ee920a
Plan: docs/superpowers/plans/2026-11-17-rq1-rq2-rq3-plan.md

## Task Status

- Task 1: Observation Builders (uavbench/obs.py + env.py) — COMPLETE (commits 01b214d..904a7a3, review clean)
- Task 2: CNN Feature Extractor (uavbench/policies/cnn_policy.py) — PENDING
- Task 3: GNN Feature Extractor (uavbench/policies/gnn_policy.py) — PENDING
- Task 4: train_rl.py updates — PENDING
- Task 5: Stronger Baselines (warm_start_pso + mpc) — PENDING
- Task 6: Failure Model (single_failure EnvConfig + mechanic) — PENDING
- Task 7: scripts/stats.py — Paired diff CI — PENDING
- Task 8: sweep_rq2.py + plot_rq2.py — PENDING
- Task 9: eval_generalization.py + plot_rq3.py — PENDING
- Task 10: plot_rq1.py + docs + requirements.txt — PENDING

## Minor Findings Log
(accumulated for final whole-branch review)

Task 1 minors:
- obs.py build_graph: edge_index dtype is float32; PyG will need .long() cast (downstream Task 3 concern)
- obs.py build_graph: Python loops over users/UAVs in hot path — violates AGENTS.md "NumPy vectorized" convention; not a correctness issue at n_uavs=8/n_users=150 scale
- env.py EnvConfig: user_speed docstring still says "m per step" — should say "m/s" after step_duration_s addition
Task 2: complete (commits 904a7a3..6a84429, review clean — minor: padding=1 on convs not in spec but harmless, unused np import in test_policies.py)
Task 3: in_progress
Task 3: complete (commits 6a84429..241235e, review clean after fix — C1: flexible features_dim projection, M2: importorskip guard added)
Task 4: in_progress
Task 4: complete (commits 241235e..90086e0, review approved — I1: graph smoke incomplete/reporting gap; I2: timing_path mkdir hardening applied directly)
Task 5: in_progress
Task 5: complete (commits 90086e0..0f041d4, review approved — Important: MPC overshoots budget by calibration overhead (spec says "not strict"); Minor: run_baselines defaults fixed inline)
Task 6: in_progress
Task 6: complete (commits 0f041d4..e679b0e, 2nd review approved — fix added failure_rate Bernoulli wiring that was stored but unused; 8 tests, 68 passing)
Task 7: in_progress
Task 7: complete (commits e679b0e..b159591, fix: replaced scipy.stats.mannwhitneyu with pure numpy — constraint violation)
Task 8: in_progress
Task 8: complete (commits b159591..8648049, review approved — minor: unused json/time imports in sweep_rq2.py; checkpoint path omits obs_mode; test_failure_affects_coverage fix is valid)
Task 9: in_progress
Task 9: complete (commits 8648049..1339566, 2nd review approved — I1: stats imported+used in plot_rq3; I2: _load_model fallback warning added; M3: removed unused import)
Task 10: in_progress
Task 10: complete (commits 1339566..975cb6f, 2nd review approved — C1: added paired_diff_ci import+usage for significance stars)
All 10 tasks complete. Ready for final whole-branch review.
Final whole-branch review: complete (merge-base 01b214d, HEAD ef3a1c5)
C1 (warm_start_pso wiring), I2 (DECISIONS.md), I3 (eval split default), I4 (comment), M1 (unused imports) — all fixed.
Remaining open item: I1 (step_duration_s physical semantics) — requires supervisor sign-off, documented in STATUS.md.
Branch ready for supervisor review before sweep.
