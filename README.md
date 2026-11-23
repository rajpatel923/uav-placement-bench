# uav-placement-bench

When does deep RL beat classical placement for UAV aerial base stations?
A statistically rigorous benchmark under the Al-Hourani air-to-ground channel.

> **Status: RQ1–RQ3 extension complete — 73 tests pass. No full sweep results yet.**
> Every number in this README came from 1–2 seeds with untuned hyperparameters.
> Do not cite them. Full results require a 10-seed sweep — see below.

## For AI agents and collaborators

Agent instructions live in `AGENTS.md` (read by most coding agents; `CLAUDE.md` imports it).
Project context lives in `docs/`: `STATUS.md` (start here), `RESEARCH_CONTEXT.md`,
`EXPERIMENT_PROTOCOL.md`, `CHANNEL_MODEL.md`, `DECISIONS.md`, `REFERENCES.md`.

## Research questions

| # | Question | Method |
|---|---|---|
| **RQ0** | Does deep RL beat classical placement, and by how much? | IQM + 95% bootstrap CI across 10 seeds |
| **RQ1** | Does observation representation matter? (flat vs sorted vs CNN vs GNN) | Train all 5 obs_modes, compare IQM |
| **RQ2** | Where is the complexity threshold where RL starts winning? | 6-config sweep (mobility × failure) |
| **RQ3** | Does a trained agent generalise to unseen environments? | Distribution shift + fleet-size transfer |

## What is here

| Component | File | Status |
|---|---|---|
| Al-Hourani A2G channel + optimal elevation solver (Eq. 13) | `uavbench/channel.py` | Tested: reproduces published optimal angles |
| Gymnasium env — Regime I (static), II (mobile), III (shock) | `uavbench/env.py` | Passes SB3 env checker |
| 5 observation modes: flat, sorted, cluster_summary, spatial_grid, graph | `uavbench/obs.py` | Working |
| CNN feature extractor for spatial_grid obs | `uavbench/policies/cnn_policy.py` | Working |
| GNN feature extractor (GATv2) for graph obs, variable fleet size | `uavbench/policies/gnn_policy.py` | Working |
| Baselines: k-means, PSO, warm_start_PSO, MPC, random, hover | `uavbench/baselines.py` | Working |
| RL training — PPO, SAC, A2C, DQN; auto-selects policy for obs_mode | `scripts/train_rl.py` | Working |
| RQ2 complexity sweep (6 configs) + heatmap | `scripts/sweep_rq2.py`, `scripts/plot_rq2.py` | Working |
| RQ3 generalization eval + 3-panel figure | `scripts/eval_generalization.py`, `scripts/plot_rq3.py` | Working |
| RQ1 IQM bar chart with significance stars | `scripts/plot_rq1.py` | Working |
| Paired-difference CI + probability of improvement (pure numpy) | `scripts/stats.py` | Working |
| rliable aggregation (IQM, mean, stratified bootstrap 95% CIs) | `scripts/aggregate.py` | Working |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build is fine
pip install -r requirements.txt
pip install torch_geometric                            # for graph obs_mode (GNN)
pytest -q                                              # expect 73 passed
```

Python 3.10+ required. A GPU is NOT needed for development; it speeds up large sweeps.

## Quickstart — launcher

`run.sh` handles everything. Auto-detects CUDA, splits seeds across GPUs when available.

```bash
bash run.sh smoke          # sanity check — all 5 obs_modes, 2 seeds (~5 min)
bash run.sh verify         # full pipeline verification end-to-end
bash run.sh tune           # tune hyperparams on val split (always before sweep)
bash run.sh sweep          # full 10-seed reported sweep (hours)
bash run.sh rq1            # RQ1: train all 5 obs_modes + plot
bash run.sh rq2            # RQ2: 6-config complexity sweep + heatmap
bash run.sh rq3            # RQ3: generalization eval + figure
bash run.sh figures        # regenerate all figures from existing results
bash run.sh help           # show all options and recommended order
```

**Recommended order:** `smoke` → `verify` → `tune` → `rq1` → `rq2` → `rq3`

### Explicit options

```bash
bash run.sh sweep 10 200000 flat static      # 10 seeds, 200k steps, flat obs, static regime
bash run.sh sweep 10 200000 graph static     # same with GNN policy
bash run.sh rq1 10 200000 static             # RQ1 with explicit seeds/steps/regime
```

### Manual commands

```bash
# Classical baselines
python scripts/run_baselines.py --regime static --seeds 0 1 2 3 4 5 6 7 8 9

# One RL run with a specific obs_mode
python scripts/train_rl.py --algo ppo --regime static --seed 0 --timesteps 200000 --obs-mode graph

# RQ2 sweep (smoke: 2 seeds, 20k steps)
python scripts/sweep_rq2.py --seeds 0 1 --n-eval 5 --timesteps 20000 --configs C1 C4 --out /tmp/rq2
python scripts/plot_rq2.py --results /tmp/rq2 --out /tmp/figures

# Significance testing
python -c "
from scripts.stats import paired_diff_ci, prob_of_improvement
import numpy as np
pt, lo, hi = paired_diff_ci(rl_scores, pso_scores)
print(f'RL - PSO: {pt:.3f} [{lo:.3f}, {hi:.3f}]  PoI={prob_of_improvement(rl_scores, pso_scores):.2f}')
"
```

## Output directories

| Directory | Contents | Rule |
|---|---|---|
| `results/` | Full reported sweep (RQ0 baseline) | **Never delete or overwrite** |
| `results_val/` | Val-split tuning runs | Never used for reported numbers |
| `results_rq1/` | RQ1 obs_mode comparison | Written by `bash run.sh rq1` |
| `results_rq2/` | RQ2 complexity sweep | Written by `bash run.sh rq2` |
| `results_rq3/` | RQ3 generalization | Written by `bash run.sh rq3` |
| `figures/` | All generated PDFs | Auto-created by plot scripts |

## Observation modes (`--obs-mode`)

| Mode | Description | Policy |
|---|---|---|
| `flat` | Raw UAV + user coordinates (default) | MlpPolicy |
| `sorted` | Users sorted by distance from centroid | MlpPolicy |
| `cluster_summary` | K-means cluster centres + counts | MlpPolicy |
| `spatial_grid` | 32×32 binary occupancy image + UAV vector | MultiInputPolicy + CnnExtractor |
| `graph` | UAVs and users as graph nodes (GATv2) | MultiInputPolicy + GNNExtractor |

## Default configuration (`EnvConfig`)

Chosen to match Cyprien, Sookhak et al., GLOBECOM 2025 (arXiv:2512.22187):
U = 4 UAVs, K = 100 users, 2000 × 2000 m area, altitude 50–120 m, 2 GHz, urban parameters.

**Differences from that paper:** objective is coverage (PL ≤ 100 dB), not sum-rate;
no UGVs or backhaul; no interference.

## Measured speed (2-core cloud CPU, single seed, 200k steps)

| Algorithm | Time | Note |
|---|---|---|
| PPO (flat, 4 envs) | ~2 min | Fastest RL |
| A2C (flat, 4 envs) | ~1 min | |
| DQN (flat) | ~4 min | |
| SAC (flat) | ~40 min | Budget carefully |
| PPO (spatial_grid CNN) | ~3–5 min | CNN overhead minimal |
| PPO (graph GNN) | ~15–25 min | GNN per-sample loop |
| PSO | ~1 s/seed (static), ~10 s (mobile) | |
| warm_start_PSO | Similar to PSO | Warm prior helps on mobile |
| MPC | ~budget_ms × max_steps per seed | Calibrated per machine |

10 seeds × 5 obs_modes × 200k steps: SAC-graph is the slowest combination (~4 h/seed on CPU).
Measure on your actual machine before planning a sweep.

## Smoke-test numbers (NOT results)

Static regime, 10 eval layouts: k-means 0.61, PSO 0.64, hover 0.13, random 0.09.
PPO after 400k untuned steps, 1 seed: 0.21.
This does *not* show RL loses — untuned SB3 defaults with flat obs are weak.
Run the full pipeline to get meaningful comparisons.

## Open question before full sweep

**Confirm with Dr. Sookhak:** `step_duration_s = 1.0 s` or `2.0 s`?
The UAV `v_max` comment suggests a 2-second step, but the default is 1.0 s.
This affects user mobility distance per step and the RQ2 crossover threshold.
See `docs/DECISIONS.md` D-017 and `docs/STATUS.md`.

## Design decisions you must own

1. **Observation design.** Now implemented as 5 options (RQ1). Best mode is an empirical question — that's the point of the experiment.
2. **Reward.** Coverage fraction minus a UAV-proximity penalty. Sum-rate and fairness are alternatives.
3. **Equal tuning budget.** Every method gets the same number of val-split tuning trials. Log each trial in `docs/EXPERIMENT_PROTOCOL.md`.
4. **Train/test separation.** Training resets use the run seed; evaluation uses `EVAL_SEED_OFFSET=10000`. Never tune on the test layouts.
5. **Seeds.** At least 10 per configuration. Report IQM with 95% CIs via `scripts/aggregate.py`.
6. **Pre-register hypotheses.** Written in `docs/RESEARCH_CONTEXT.md` before the full sweep.

## Key references

- Al-Hourani, Kandeepan, Lardner, "Optimal LAP Altitude for Maximum Coverage," IEEE WCL 2014.
- Al-Hourani, Kandeepan, Jamalipour, "Modeling air-to-ground path loss for LAPs in urban environments," GLOBECOM 2014.
- Cyprien, Sookhak, Zarini, Sekharan, Atiquzzaman, "Joint UAV-UGV Positioning and Trajectory Planning via Meta A3C," arXiv:2512.22187.
- Qiu, Lyu, Fu, "Placement Optimization of Aerial Base Stations with Deep RL," ICC 2020.
- Agarwal et al., "Deep RL at the Edge of the Statistical Precipice," NeurIPS 2021 (rliable).
- Henderson et al., "Deep Reinforcement Learning that Matters," AAAI 2018.
