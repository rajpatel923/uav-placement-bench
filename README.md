# uav-placement-bench (starter kit)

When does deep RL beat classical placement for UAV aerial base stations?
A statistically rigorous benchmark under the Al-Hourani air-to-ground channel.

> **Status: starter scaffold, not results.** Every number in this README came from
> 1-2 seeds with untuned hyperparameters, to prove the pipeline runs. Do not cite them.

## For AI agents and collaborators

Agent instructions live in `AGENTS.md` (read by most coding agents; `CLAUDE.md` imports it).
Project context lives in `docs/`: `STATUS.md` (start here), `RESEARCH_CONTEXT.md`,
`EXPERIMENT_PROTOCOL.md`, `CHANNEL_MODEL.md`, `DECISIONS.md`, `REFERENCES.md`.

## What is here

| Piece | File | Status |
|---|---|---|
| Al-Hourani A2G channel + optimal elevation solver (Eq. 13) | `uavbench/channel.py` | Tested: reproduces published optimal angles 20.34 / 42.44 / 54.62 / 75.52 deg |
| Gymnasium env, Regime I (static) and II (mobile users) | `uavbench/env.py` | Passes Gymnasium and SB3 env checkers |
| Discrete wrapper for DQN (7 moves per UAV, joint space 7^U) | `uavbench/env.py` | Working |
| Baselines: k-means + Al-Hourani altitude, PSO, random, hover | `uavbench/baselines.py` | Working; obey the same flight limits as RL |
| Shared eval protocol (held-out layouts, deterministic eval) | `uavbench/evaluation.py` | Working |
| RL training (PPO, SAC, A2C, DQN via Stable-Baselines3) | `scripts/train_rl.py` | Working, untuned |
| rliable aggregation (IQM, mean, stratified bootstrap 95% CIs) | `scripts/aggregate.py` | Working |
| Regime III (disaster shock), SINR/interference, Lyu spiral, Meta-A3C | - | **Not implemented (your job)** |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build is enough
pip install -r requirements.txt
pytest -q                                              # expect 21 passed
```

Python 3.10+ recommended. A GPU is NOT needed: networks are tiny MLPs, CPU is faster here.

## Quickstart — launcher (recommended)

`run.sh` at the repo root handles everything. It auto-detects CUDA and splits
seeds across multiple GPUs when available.

```bash
bash run.sh smoke          # sanity check before anything (~2 min)
bash run.sh tune           # tune hyperparams on val split first
bash run.sh sweep          # full 10-seed reported run (hours)
bash run.sh figures        # regenerate figures after a sweep
bash run.sh help           # show all options
```

Machine-specific launchers (single-GPU machines):

```bash
bash run_5090.sh smoke     # Ryzen 9 9950X + RTX 5090 — same modes
bash run_5090.sh sweep
```

**Always run `tune` before `sweep`.** The order is: `smoke` → `tune` → `sweep`.

### Manual commands (advanced)

```bash
# classical baselines on the shared evaluation layouts
python scripts/run_baselines.py --regime static --seeds 0 1 2 3 4 5 6 7 8 9

# one RL run (repeat per seed; see scripts/run_all.sh for the full sweep)
python scripts/train_rl.py --algo ppo --regime static --seed 0 --timesteps 400000

# aggregate everything in results/static with rliable
python scripts/aggregate.py --regime static --metric final_cov
```

Metrics: `final_cov` = mean coverage over the last 10 steps (placement quality);
`avg_cov` = mean coverage over the whole episode (also rewards fast deployment).

## Default configuration (`EnvConfig`)

Chosen to match the setup in Cyprien, Sookhak et al., GLOBECOM 2025 (arXiv:2512.22187)
where possible: U = 4 UAVs, K = 100 users, 3000 x 3000 m area, altitude 30-150 m,
2 GHz, urban Al-Hourani parameters (eta_LoS = 1, eta_NLoS = 20).

**Differences from that paper you must state:** objective here is coverage
(PL <= 100 dB), not sum-rate with a QoS constraint; no UGVs or backhaul; no interference.

## Measured speed (2-core cloud CPU, single seed)

| Algorithm | Measured | Implied cost of 200k steps |
|---|---|---|
| PPO (4 envs) | 400k steps in 3.5 min | ~2 min |
| A2C (4 envs) | 20k steps in 0.1 min | ~1 min |
| DQN | 10k steps in 0.2 min | ~4 min |
| SAC | 5k steps in 1.0 min | **~40 min** (budget for this) |
| Baselines | k-means 0.1 s, PSO ~1 s (static) / ~10 s (mobile) per seed | - |

10 seeds x 2 regimes x 200k steps: SAC alone is ~13 h on this machine; the rest is
under 2 h. Your laptop or a lab workstation will differ: measure, don't assume.

## Smoke-test numbers (NOT results)

Static regime, 10 eval layouts: k-means 0.61, PSO 0.64, hover 0.13, random 0.09.
PPO after 400k untuned steps, 1 seed: 0.21. Mobile regime: k-means 0.42, PSO 0.49.
This does *not* show RL loses. It shows untuned SB3 defaults with this observation
design are weak, which is exactly the kind of thing you must investigate fairly.

## Design decisions you must own (your advisor WILL ask)

1. **Observation design.** Users are listed in arbitrary order (100 x 2 coordinates).
   MLPs handle unordered sets badly. Options: sort users, use cluster summaries,
   a coverage bitmap + CNN (Qiu et al. ICC 2020; Bayerlein et al.), or a
   permutation-invariant encoder. This choice can decide whether RL "wins".
2. **Reward.** Coverage fraction minus a UAV-proximity penalty. Sum-rate, fairness,
   and energy are alternatives; pick one and justify it.
3. **Equal tuning budget.** If PPO gets 50 tuning trials, SAC and the PSO swarm
   settings get comparable effort. Write the budget down before you start.
4. **Train/test separation.** Training resets use the run seed; evaluation uses
   layouts 10000+i. Never tune on the evaluation layouts.
5. **Seeds.** At least 10 per configuration. Report IQM with 95% CIs, never a single run.
6. **Pre-register hypotheses.** Write them in your notes *before* the full sweep and
   report the outcome whichever way it goes.

## Roadmap (suggested)

- [ ] Verify every constant in `channel.py` against the Al-Hourani PDFs
- [ ] Reproduce Al-Hourani's radius-vs-altitude figure with `coverage_radius()`
- [ ] Decide observation design; run a small, *equal-budget* tuning sweep
- [ ] Add SINR with co-channel interference (Regime II realism)
- [ ] Add Lyu et al. 2017 spiral placement baseline
- [ ] Regime III (mid-episode UAV failure / user migration) - only if time allows
- [ ] Meta-A3C comparison - only if the lab shares its code

## Key references

- Al-Hourani, Kandeepan, Lardner, "Optimal LAP Altitude for Maximum Coverage," IEEE WCL 2014.
- Al-Hourani, Kandeepan, Jamalipour, "Modeling air-to-ground path loss for LAPs in urban environments," GLOBECOM 2014.
- Cyprien, Sookhak, Zarini, Sekharan, Atiquzzaman, "Joint UAV-UGV Positioning and Trajectory Planning via Meta A3C," arXiv:2512.22187.
- Sobouti, ..., Sookhak, ..., Afghah, "Efficient Fuzzy-Based 3-D Flying Base Station Positioning...," IEEE Systems Journal 18(2), 2024.
- Qiu, Lyu, Fu, "Placement Optimization of Aerial Base Stations with Deep RL," ICC 2020.
- Agarwal et al., "Deep RL at the Edge of the Statistical Precipice," NeurIPS 2021 (rliable).
- Henderson et al., "Deep Reinforcement Learning that Matters," AAAI 2018.
- Schneider et al., "mobile-env," IEEE/IFIP NOMS 2022.
