# SDD Progress Ledger

Branch: feature/uav-benchmark-impl
Merge base (branch start): 6f0f8f7fc09ac91f080bed507ff49c21abb2737b
Plan: docs/superpowers/plans/2026-11-16-uav-benchmark-full-plan.md

## Task Status

- Task 1: Channel Verification Script — COMPLETE (commits 6f0f8f7..3d4c100, review clean)
- Task 2: Lyu Spiral Baseline — COMPLETE (commits 3d4c100..1626ec0, review clean)
- Task 3: Experiment Protocol Freeze — HUMAN CHECKPOINT (skip; no code)
- Task 4: Observation Design (obs_mode) — COMPLETE (commits 1626ec0..df96c1e, review clean)
- Task 5: SINR / Co-Channel Interference — COMPLETE (commits df96c1e..2d914a7, review clean)
- Task 6: Hyperparameter Tuning Infrastructure — COMPLETE (commits 2d914a7..c5ea95f, review clean after device fix)
- Task 7: Full 10-Seed Sweep & Paper Figures — COMPLETE (commits c5ea95f..90a784c, review clean after 3 fixes)
- Task 8: Regime III Disaster Shock — COMPLETE (commits 90a784c..3ebadbf, review clean)
- Final whole-branch review — COMPLETE (fixes 3ebadbf..86ddd25, re-review: READY TO MERGE)

## Minor Findings Log
(accumulated for final whole-branch review)

Task 4:
- env.py: KMeans imported inside `_obs()` on every step — move to top of file for cleanliness (no correctness risk)
- env.py: cluster_summary KMeans uses fixed random_state=0, not seeded via self.np_random — intentional per spec, worth noting in paper
- run_baselines.py: obs_mode not saved to result JSON for baselines (only train_rl.py saves it) — not required by spec

Task 5:
- DECISIONS.md: D-011→D-013 gap (D-012 is Task 3 human checkpoint, written by supervisor when freezing protocol — expected)
- channel.py sinr_matrix: uses sum-max for interference (correct; negligible float cancellation at this scale)
- sinr_coverage() callable even when use_sinr=False — intentional, docstring could clarify

Task 7:
- run_all.sh: plot_figures.py only called for final_cov metric; avg_cov figure not generated (likely intentional — single paper figure)
- plot_figures.py: hi_errs[present.index(algo)] is O(n) in annotation loop; could use enumerate (cosmetic)
- run_baselines.py: --obs-mode not saved in result JSON (pre-existing gap, not introduced here)

Task 8:
- env.py: stale module docstring still says "Regime III NOT implemented yet" — 1-line fix
- env.py: _too_close_frac() uses all UAVs (including failed ones) in collision penalty — may be intentional (failed UAV still physically present), but undocumented
- env.py: test_shock_user_migration_moves_users checks post-5-step state, not isolated shock effect — generous lower bound masks it; could be tighter
- env.py: shock_active/n_active_uavs always present in non-shock regime info dicts (benign additive change)
- env.py: zero-active-UAVs edge case (shock_n_uavs_fail >= n_uavs) would crash with ValueError from min() on empty array — safe with defaults (1 fail, 4 UAVs) but worth a guard
