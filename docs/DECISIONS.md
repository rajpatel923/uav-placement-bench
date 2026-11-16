# Decision log

One entry per design decision. Newest at the bottom. Never delete entries; mark them
`Superseded by D-xxx` instead. Agents: add an entry whenever you change behavior,
defaults, metrics, or dependencies, and ask the human first for anything marked
"research decision" in AGENTS.md.

Template:

```
## D-0XX: <title>
Date: YYYY-MM-DD | Status: proposed / accepted / superseded | Decided by: <name>
Context: why a decision was needed
Decision: what we chose
Alternatives: what else was considered
Consequences: what this affects; does it invalidate existing results?
```

---

## D-001: Objective is coverage, not sum-rate
Date: 2026-11-13 | Status: accepted (v0, revisit) | Decided by: student (starter kit)
Context: Needed a simple, interpretable objective for the first benchmark version.
Decision: Coverage = fraction of users with best-UAV mean path loss ≤ `pl_max` (100 dB).
Alternatives: Sum-rate with QoS (as in Meta-A3C), SINR-based coverage, fairness.
Consequences: Results are not directly comparable to Meta-A3C's sum-rate numbers. Must
be stated in the paper. Moving to sum-rate/SINR later invalidates all results.

## D-002: Default config matches the Meta-A3C setup
Date: 2026-11-13 | Status: accepted | Decided by: student (starter kit)
Context: An earlier default (2000 m area, 50–300 m altitude) was too easy: PSO
reached ~0.96 coverage, which leaves no room to tell methods apart.
Decision: 3000 × 3000 m, altitude 30–150 m, U = 4, K = 100, 2 GHz, urban, matching
Cyprien/Sookhak GLOBECOM 2025. v_max = 60 m/step (30 m/s × 2 s), vz_max = 10 m/step.
Consequences: Smoke-test baselines now reach ~0.61–0.64 static coverage, which leaves headroom.

## D-003: Baselines obey the same flight limits as RL
Date: 2026-11-13 | Status: accepted | Decided by: student (starter kit)
Decision: Baselines compute targets, match UAVs to targets (Hungarian), and fly toward
them under `v_max`/`vz_max` via `TargetTrackingPolicy`. In the mobile regime they
re-plan every 5 steps by default.
Alternatives: Teleporting baselines (unfair to RL; rejected).
Consequences: The re-plan interval is a baseline hyperparameter and counts toward the tuning budget.

## D-004: Evaluation splits and metric window
Date: 2026-11-13 (val split added 2026-11-16) | Status: accepted
Decision: Test layouts are seeds 10000+i (n = 20). Validation layouts are seeds 20000+i.
Primary metric `final_cov` uses the mean over the last 10 steps.
Consequences: Changing any of these invalidates all results.

## D-005: DQN uses a joint discrete action space
Date: 2026-11-13 | Status: accepted
Context: SB3 DQN supports only `spaces.Discrete`.
Decision: 7 moves per UAV (±x, ±y, ±z, hover), joint space 7^U = 2401 for U = 4.
Consequences: Does not scale beyond ~5 UAVs. DQN is disadvantaged by coarse moves;
state this when interpreting results.

## D-006: SB3 default hyperparameters for the starter kit
Date: 2026-11-13 | Status: accepted (temporary)
Decision: No tuning yet. Tuning will follow EXPERIMENT_PROTOCOL.md section 5 with an equal budget.
Consequences: Smoke-test RL numbers are not meaningful.

## D-007: Dependency pins
Date: 2026-11-13 | Status: accepted
Context: `rliable 1.2.0` imports `arch`, and `arch 7.x` crashes on import with `pandas 3`.
Decision: Pin `arch>=5.3.1,<8` and `pandas<3`.

## D-008: Observation is a flat, unordered list of user coordinates (OPEN)
Date: 2026-11-13 | Status: **open, research decision**
Context: Observation = normalized UAV xyz (3U) + user xy (2K) in arbitrary order. MLPs
handle unordered sets poorly, which likely explains weak untuned RL (PPO ~0.21 vs
k-means ~0.61 in a single smoke run).
Options: sort users; cluster summaries (centroids, counts); coverage bitmap + CNN (Qiu
et al. 2020; Bayerlein et al.); permutation-invariant encoder (DeepSets/attention).
Next step: decide with the supervisor; compare 2–3 options on the validation split under
an equal budget; log the outcome here.

## D-009: Regime III deferred
Date: 2026-11-13 | Status: accepted
Decision: Disaster-shock regime and the Meta-A3C comparison are stretch goals. They depend on
lab code access and semester time.

## D-010: Standalone environment instead of extending mobile-env
Date: 2026-11-13 | Status: accepted (revisit for release)
Context: Earlier planning suggested extending mobile-env (NOMS 2022).
Decision: Built a small standalone Gymnasium env for speed and full control of physics.
Alternatives: Upstream an Al-Hourani A2G channel to mobile-env later as an open-source contribution.
Consequences: We must justify not reusing mobile-env/Brunori et al. in the paper.

## D-011: Lyu spiral baseline added
Date: 2026-11-16 | Status: accepted | Decided by: student
Context: README roadmap item. Multi-UAV spiral placement to complement k-means and PSO.
Decision: Fermat spiral (golden-angle) with Al-Hourani altitude rule, clipped to [h_min, h_max].
Alternatives: Exact Lyu et al. formulation (needs paper to confirm multi-UAV extension).
Consequences: Must verify multi-UAV formulation against the paper before citing as "Lyu et al."

## D-013: SINR / co-channel interference extension
Date: 2026-11-16 | Status: proposed (needs supervisor approval before use in results)
Context: README roadmap item. No-interference model overestimates coverage; UAVs do interfere.
Decision: Added sinr_matrix() to channel; use_sinr flag in EnvConfig defaults to False.
Alternatives: SINR-based reward (changes the objective; would require supervisor approval and full re-run).
Consequences: Does NOT affect any existing results (flag is False by default). Enabling it
changes the metric and requires a full re-run with supervisor approval.

## D-014: Regime III (disaster shock) implementation
Date: 2026-11-16 | Status: proposed (needs supervisor approval before reported results)
Context: README/RESEARCH_CONTEXT.md stretch goal. Tests H3 hypothesis.
Decision: Added mobility="shock" to env. Two shock types: uav_failure and user_migration.
Defaults to uav_failure at step 30. Failed UAVs are excluded from coverage computation.
Consequences: Does not affect Regime I/II results (different mobility parameter). Must
get supervisor approval and add to EXPERIMENT_PROTOCOL.md before running reported sweeps.
