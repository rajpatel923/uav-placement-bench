# Research context

Last reviewed: 2026-11-16. Update when the framing, hypotheses, or target venue changes.

## Working title

*When Does Deep RL Beat Classical Optimization for UAV Base Station Placement?
A Statistically Rigorous Benchmark under the Al-Hourani Air-to-Ground Model*

## Research question

Across increasingly dynamic regimes, where does deep RL start to outperform cheap
classical placement methods (k-means with the Al-Hourani altitude rule, PSO)? We
compare under identical physics, flight constraints, and evaluation layouts, using
multiple seeds and confidence intervals. The regimes run from static users, to moving
users, and later to a disaster shock.

Framing: **map the complexity boundary**. This is not an attack on RL. A result
where RL never wins, always wins, or wins only in some regimes is equally publishable
if the evaluation is rigorous.

## Why this gap exists (the motivation, stated carefully)

1. Many DRL papers on UAV placement compare mainly against other RL algorithms or
   weak heuristics. Example: the supervising lab's Meta-A3C paper (below) compares
   only against A3C and DDPG. Not every paper does this: Qiu, Lyu & Fu (ICC 2020)
   compared against DQN *and* k-means. So say "few use strong classical baselines",
   not "none do".
2. Results are often single-run. One 2025 multi-agent UAV relay paper (arXiv:2512.08341)
   states all results come from a single training run with a fixed seed. Henderson et
   al. (AAAI 2018) and Agarwal et al. (NeurIPS 2021) show why that is unreliable.
3. Open environments already exist (Brunori et al. 2021, a Gym UAV env; mobile-env,
   NOMS 2022). **Do not claim "first open environment."** The contribution is the
   rigorous comparison and the complexity-boundary analysis.

## Hypotheses (neutral; freeze final wording in EXPERIMENT_PROTOCOL.md before the sweep)

- **H1 (static, Regime I):** Strong classical baselines match or exceed standard DRL
  in final coverage, at far lower compute.
- **H2 (mobile, Regime II):** The gap between DRL and periodically re-planned classical
  baselines narrows or reverses as user mobility increases.
- **H3 (shock, Regime III, stretch):** Under a mid-episode shock (UAV failure / user
  migration), methods differ in recovery speed. No prior expectation about which wins.

Each hypothesis can be confirmed or rejected. Report both outcomes honestly.

## Regimes

| Regime | Users | Status |
|---|---|---|
| I: static | Clustered, fixed for the episode | Implemented (`mobility="static"`) |
| II: mobile | Random waypoint, `user_speed` m/step | Implemented (`mobility="mobile"`) |
| III: shock | Mid-episode UAV failure or migration to shelters | **Not implemented** (stretch goal) |

## Supervising lab context (verified facts only)

- **Dr. Mehdi Sookhak**, Dept. of Computer Science, TAMU-CC. NSF CAREER award
  "SIM-Integrated Air-to-Ground Autonomous Vehicle Networks", 5 years, $632,414. It
  covers UAV-assisted networks, AI-driven positioning, and a disaster-relief
  communication system.
- **Cyprien, Sookhak, Zarini, Sekharan, Atiquzzaman, "Joint UAV-UGV Positioning and
  Trajectory Planning via Meta A3C for Reliable Emergency Communications"**, GLOBECOM 2025,
  arXiv:2512.22187. Verified details:
  - U = 4 UAVs (altitude 30–150 m, max speed 30 m/s), M = 4 UGVs (max 20 m/s, road
    limit 15 m/s), K = 100 users in a 3000 × 3000 m area, 2 GHz, min rate 0.5 Mbps.
  - Objective: sum-rate with QoS, with UGVs as ground-to-air backhaul.
  - Baselines: **A3C and DDPG only.** No non-RL baselines. No multiple seeds, CIs, or
    error bars reported. Reports +13.1% throughput vs A3C.
  - No code release found. Ask the lab for code before planning a Meta-A3C comparison.
- **Sobouti, Adarbah, Alaghehband, Chitsaz, Mohajerzadeh, Sookhak, Hosseini Seno,
  Vahedian, Afghah, "Efficient Fuzzy-Based 3-D Flying Base Station Positioning and
  Trajectory for Emergency Management in 5G and Beyond Cellular Networks,"** IEEE
  Systems Journal 18(2), 2024. Sookhak is 6th of 9 authors. **Do not call it "Rahimi
  et al." and do not call it "the lab's controller."** It is a fuzzy candidate-point
  selection method, not an adaptive real-time controller.
- Rahimi, ..., Sookhak: 3D UAV BS positioning and backhaul management via stochastic
  optimization, GLOBECOM 2022 (cited in the Meta-A3C paper; details not yet verified).

## How to talk about the lab's work

- Say: "We extend the evaluation of Meta-A3C with classical baselines and multi-seed
  statistics." Do not say: "Meta-A3C's evaluation is flawed."
- Our default `EnvConfig` deliberately matches the Meta-A3C setup (area, altitudes,
  U, K, frequency) for continuity. **Differences we must state:** our objective is
  coverage, not sum-rate; no UGVs/backhaul; no interference (yet).

## Claims we must NOT make (known errors from an earlier AI-generated report)

- "Static placement is solved / k-means is near-optimal / RL is mathematically
  redundant." **False.** Multi-UAV coverage maximization is NP-hard (Qiu et al. ICC
  2020); k-means is a heuristic (it minimizes squared distance, not coverage). The
  Al-Hourani optimal elevation angle applies to single-UAV coverage radius under a
  path-loss budget, not to sum-rate with interference.
- "Safe-DQN trajectory optimization" is NOT from Sookhak's group (it is Zhang, Lei,
  Liu, Feng, Nallanathan; just a reference in the Meta-A3C paper).
- Compute-time estimates from that report (e.g. "15.5 hours total", "<0.2 ms/step")
  were invented. Use measured numbers only (`docs/STATUS.md`).
- "First open benchmark/environment." False (see above).

## Target venues (verify every deadline on the official site)

1. **IEEE VTC2027-Fall** (Osaka, Sept 27–30, 2027). A third-party listing says the
   deadline is ~Feb 1, 2027 (unverified; VTC deadlines are often extended).
2. **AAAI-28 Student Abstract** (fall 2027). The lab published there in 2026.
   Non-archival, so a full paper can follow later.
3. **CCSC South Central** (undergraduate poster track): fallback.
4. Journal extension later: IEEE Open Journal of the Communications Society.

Final venue choice is the supervisor's call. Record it in `docs/DECISIONS.md`.

## RQ1–RQ3 Extension (added 2026-11-17)

### Research Questions

**RQ1 — Representation Sensitivity**
How do coordinate-based, spatial-grid, and graph-based observation representations
affect DRL performance, convergence, and generalization?

**H4:** Graph-based representations improve RL over coordinate-based in dynamic regimes.
*(Not assumed: GNN may not win in the static regime.)*

**RQ2 — Dynamic Complexity Threshold**
At what levels of user mobility and UAV failure frequency does DRL outperform online
re-optimization?

**H5:** RL begins to match PSO somewhere between 5 m/s and 15 m/s user mobility combined
with UAV failures. The crossover is an empirical finding, not a guaranteed outcome.

**RQ3 — Generalization and Deployment Cost**
- **RQ3-A (distribution shift):** Do policies trained on 100 users / 3000m area generalise
  zero-shot to 50/150 users and 2000/4000m areas?
- **RQ3-B (structural shift, GNN only):** Does a UAV-node-wise GNN trained on 4 UAVs execute
  successfully on 6 and 8 UAVs without retraining?

**H6-A:** MLP/CNN degrade significantly under distribution shift; GNN is more robust.
**H6-B:** Only the UAV-node-wise GNN supports structural shift; MLP/CNN cannot change their
action space at inference time.

### New Baselines
- **warm_start_pso:** PSO seeded with the prior episode's best solution.
- **mpc:** Time-budgeted receding-horizon re-plan (PSO horizon, same wall-clock per step as RL).

### Statistical Protocol
Primary significance test: **paired difference CI** on `final_cov`.
`point, lo, hi = paired_diff_ci(rl_scores, pso_scores)`
Significance: CI excludes zero. P(improvement) via Mann–Whitney U.
Single-metric CI overlap is NOT used as a significance criterion.

### Three Acceptable Outcomes
| Outcome | Interpretation |
|---------|---------------|
| A: GNN substantially improves | Relational obs effective for coordinated UAV deployment |
| B: DRL matches PSO only under high mobility+failure | Use RL when re-planning latency is the bottleneck |
| C: Strong online optimiser matches DRL | Establishes where learning fails; motivates future equivariant architectures |

No predetermined outcome is assumed or preferred.
