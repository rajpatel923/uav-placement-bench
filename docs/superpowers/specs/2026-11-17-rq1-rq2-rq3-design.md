# RQ1–RQ3 Extension Design (Revised)

**Revision:** Incorporates six fixes from supervisor review. See §8 for change log.

**Goal:** Extend the UAV placement benchmark to answer three research questions
for VTC2027/AAAI-28 submission. Central contribution:

> A controlled investigation of how spatial and relational observation representations
> affect the performance, resilience, and generalization of deep RL relative to
> computationally constrained online optimization for multi-UAV base station deployment.

**Approach:** Approach B revised — Full RQ1 + RQ2, corrected RQ3 split into two
sub-experiments (distribution shift and structural shift).

**Compute target:** Threadripper PRO 9965WX + 2× RTX PRO 6000 Blackwell.
Estimated sweep time: ~15–18 hours (conservative, correctness-first).

---

## Global Constraints

- Python 3.10+, NumPy vectorized, no global `np.random.*`
- New `EnvConfig` fields must default to values that preserve existing Regime I/II/III behaviour
- `pytest -q` must pass after every task
- Never tune on the test split (seeds 10000+i); tuning uses val split (seeds 20000+i)
- No physics constants changed without citing source + equation number
- All figures use IQM + 95% bootstrap CI; primary significance test is **paired difference CI**,
  not single-metric CI overlap (see §6)
- PyTorch Geometric (`torch_geometric>=2.5`) is the only new dependency permitted
- Smoke test (`--timesteps 20000 --n-eval 5 --seeds 0 1`) required for every new script
- Mobility speed is in **m/s** (not m/step); document the step duration in `EnvConfig`
- Do not encode expected DRL victory into hypotheses; all three outcomes (§7) are acceptable

---

## Research Questions and Hypotheses

### RQ1 — Representation Sensitivity
> How do coordinate-based, spatial-grid, and graph-based observation representations
> affect DRL performance, convergence, scalability, and generalization?

**H4:** Graph-based representations improve RL performance over coordinate-based
representations in dynamic regimes, because relational structure helps coordinate UAVs.
*(Not assumed: GNN may not win in static regime.)*

**Key ablation:** compare graph-level pooling GNN vs UAV-node-wise GNN to separate
the effect of graph structure from the effect of the improved actor architecture.

### RQ2 — Dynamic Complexity Threshold
> At what levels of user mobility and UAV failure frequency does DRL outperform
> online re-optimization, measured by final coverage, episode-average coverage,
> and outage duration?

**H5:** RL begins to match PSO somewhere between 5 m/s and 15 m/s user mobility
when combined with UAV failures. The crossover point is an empirical finding, not
a guaranteed outcome.

### RQ3 — Generalization and Deployment Cost
Two independent sub-experiments:

**RQ3-A (Distribution shift):** Do policies trained on 100 users / 3000m area
generalise zero-shot to 50/150 users and 2000/4000m areas?

**RQ3-B (Structural shift):** Does the UAV-node-wise GNN policy trained on 4 UAVs
execute successfully on 6 and 8 UAVs without retraining? Does coverage degrade
gracefully?

**H6-A:** MLP and CNN policies degrade significantly under distribution shift
(input statistics change); GNN is more robust because it encodes relative positions.

**H6-B:** Only the UAV-node-wise GNN supports structural shift (fleet size change);
MLP and CNN architectures cannot change their action space at inference time.

---

## New Files

| Path | Role |
|---|---|
| `uavbench/obs.py` | All 5 obs builders; fixed-capacity graph tensor construction |
| `uavbench/policies/__init__.py` | Package marker |
| `uavbench/policies/cnn_policy.py` | SB3 `BaseFeaturesExtractor` for `spatial_grid` |
| `uavbench/policies/gnn_policy.py` | GNN extractor + UAV-node-wise actor for `graph` |
| `scripts/sweep_rq2.py` | RQ2 6-config complexity sweep |
| `scripts/eval_generalization.py` | RQ3-A/B zero-shot evaluation |
| `scripts/plot_rq1.py` | Figure 1: representation comparison bar chart |
| `scripts/plot_rq2.py` | Figure 2: complexity threshold heatmap |
| `scripts/plot_rq3.py` | Figure 3: generalization lines + cost table |
| `tests/test_policies.py` | CNN + GNN extractor tests incl. variable fleet |
| `tests/test_rq2.py` | failure_rate + recovery mechanic tests |
| `tests/test_generalization.py` | Zero-shot eval script tests |

### Modified Files

| Path | Change |
|---|---|
| `uavbench/env.py` | Add `spatial_grid`, `graph` obs_modes; add `failure_rate`, `step_duration_s`, `max_fleet_size`; delegate to `obs.py` |
| `scripts/train_rl.py` | Auto-select policy; save checkpoint `.zip`; log wall-clock + inference latency |
| `uavbench/baselines.py` | Add `warm_start_pso_planner`, `mpc_planner` |
| `docs/RESEARCH_CONTEXT.md` | Add RQ1–RQ3, H4–H6, updated baselines list |
| `requirements.txt` | Add `torch_geometric>=2.5` |

---

## RQ1: Observation Representations

### Five obs_modes

| obs_mode | Observation | Policy integration | Status |
|---|---|---|---|
| `flat` | `(2K + 3U,)` vector | `MlpPolicy` | Existing |
| `sorted` | `(2K + 3U,)` sorted by centroid dist | `MlpPolicy` | Existing |
| `cluster_summary` | `(5C + 3U,)` | `MlpPolicy` | Existing |
| `spatial_grid` | Dict `{"grid":(1,32,32), "uavs":(3U,)}` | `MlpPolicy` + `CnnExtractor` | New |
| `graph` | Dict fixed-capacity tensors (see below) | Custom `GNNPolicy` | New |

### Spatial Grid (`cnn_policy.py`)

```python
class CnnExtractor(BaseFeaturesExtractor):
    # Conv(1→16, 3×3) → ReLU → Conv(16→32, 3×3) → ReLU
    # → AdaptiveAvgPool(4,4) → flatten(512)
    # → concat UAV vector (3*n_uavs,) → FC(512+3U, 128)
```

Grid: 32×32 binary occupancy over [0, area_size]². UAV positions normalised to [0,1].
SB3 Dict obs space: `{"grid": Box(0,1,(1,32,32)), "uavs": Box(-1,1,(3*n_uavs,))}`.

### Graph (`gnn_policy.py`) — Fix for Problems 1, 2, 3

**Core architectural fix:** Use a **UAV-node-wise actor head**, not global mean pooling.
The actor reads each UAV's individual node embedding and outputs per-UAV actions.
This enables variable fleet sizes via padding and masking.

**Fixed-capacity graph tensors (Fix for Problem 2 — PyG/SB3 integration):**

SB3 requires fixed-shape numpy arrays. We pad all graphs to `max_fleet_size=8` UAVs
and `max_users=150` users = max 158 nodes total, `max_edges=4000`.

```python
# Observation space keys (all fixed shape, stored as float32):
{
  "node_feats":    Box(-1, 1, (max_nodes, node_feat_dim)),  # (158, 4)
  "edge_index":    Box(0, max_nodes, (2, max_edges)),        # (2, 4000)
  "node_type":     Box(0, 2, (max_nodes,)),                  # 0=pad, 1=UAV, 2=user
  "uav_active":    Box(0, 1, (max_fleet_size,)),             # 1=active, 0=failed/pad
  "n_nodes":       Box(0, max_nodes, (1,)),                  # actual node count
  "n_edges":       Box(0, max_edges, (1,)),                  # actual edge count
}
```

**Node features (4 dims):**
- UAV nodes: `[x/area, y/area, z/h_max, 1.0]` (is_uav flag)
- User nodes: `[x/area, y/area, 0.0, 0.0]`
- Padding nodes: all zeros

**Edges:**
- UAV↔UAV: fully connected (all pairs)
- UAV↔User: within `graph_edge_radius` metres (default 1500m)
- User↔User: none

**GATv2 architecture:**

```python
class GNNExtractor(BaseFeaturesExtractor):
    # Layer 1: GATv2Conv(in=4, out=64, heads=4, concat=True) → 256 dims
    # Layer 2: GATv2Conv(in=256, out=64, heads=1, concat=False) → 64 dims
    # UAV node embeddings extracted by uav_active mask → shape (max_fleet, 64)
    # Padded UAV slots zeroed out
    # Output: flattened UAV embeddings (max_fleet * 64,) = (512,)
```

**UAV-node-wise actor head (continuous, SAC/PPO):**

```python
class GNNActorCritic(ActorCriticPolicy):
    # Actor: FC(64, 3) applied per-UAV node → (max_fleet, 3) actions
    # Mask inactive/padded UAVs to zero before env step
    # Critic: mean pool active UAV embeddings → FC(64, 1) value
```

For DQN (discrete): actor outputs Q-values over 7 moves per active UAV node.
Joint action = tuple of per-UAV moves; handled by existing `DiscreteActionWrapper`.

**New `EnvConfig` fields:**

```python
graph_edge_radius: float = 1500.0   # metres; UAV↔user edge threshold
max_fleet_size:    int   = 8        # padding target; must be >= n_uavs
step_duration_s:   float = 1.0      # seconds per env step (for m/s conversion)
```

---

## RQ2: Dynamic Complexity Threshold

### Fix for Problem 4 — Revised failure model

The original 0.02/step rate caused ~63% per-UAV failure probability per episode —
this primarily tests network collapse, not adaptation.

**Revised model:** At most **one UAV failure per episode**, triggered at a uniformly
random step in [episode_len//3, 2*episode_len//3]. The failed UAV stays offline for
the rest of the episode. This is a stress test, not a cascade.

Controlled via:

```python
single_failure: bool  = False  # one random-step failure per episode
failure_rate:   float = 0.0    # retained for future continuous-failure experiments
                               # (not used in RQ2 sweep)
```

Both default to False/0.0 — no change to existing behaviour.

### Fix for Problem 5 — Stronger baselines

Add two new baselines to `baselines.py`:

**`warm_start_pso_planner`:** Initialises PSO particle swarm from the previous
episode's best solution (tracks last positions across episodes). Equivalent
computational budget to cold-start PSO.

**`mpc_planner`:** Receding-horizon MPC. At each step, runs PSO for `mpc_horizon`
steps ahead, executes the first action, discards the rest. Time-budgeted: same
wall-clock per step as RL inference. Uses `TargetTrackingPolicy` for movement.

Both obey the same `v_max`/`vz_max` flight constraints via `TargetTrackingPolicy`.

### 6-config sweep

| Config | user_speed (m/s) | single_failure | Label |
|---|---|---|---|
| C1 | 0.0 | False | static |
| C2 | 5.0 | False | mobile-low |
| C3 | 15.0 | False | mobile-high |
| C4 | 0.0 | True | static+failure |
| C5 | 5.0 | True | mobile-low+failure |
| C6 | 15.0 | True | mobile-high+failure |

Methods per config: PSO, warm-start PSO, MPC, k-means + best-obs RL (from RQ1).
10 seeds, 20 test layouts. Results to `results_rq2/{config}/`.

If evidence of a crossover between C2 and C3: add C2.5 (user_speed=10.0) and
C2.75 (12.5 m/s) for finer resolution.

### Fix for Problem 6 — Correct statistical test

**Primary test:** paired performance difference on `final_cov`.

For each (seed, layout) pair, compute `d = RL_score - PSO_score`. Bootstrap the
distribution of mean(d) with 5000 reps. Report:

- Point estimate: mean(d)
- 95% CI on d: `[d_lo, d_hi]`
- Significance: CI excludes zero → "statistically significant difference"
- Probability of improvement: `P(RL > PSO)` via rliable's `probability_of_improvement()`

**Do not** use single-metric CI overlap as the significance criterion.
Add `scripts/stats.py` with `paired_diff_ci()` and `prob_of_improvement()` helpers.

---

## RQ3: Generalization and Deployment Cost

### RQ3-A: Distribution shift

Train on default config (100 users, 3000m area). Evaluate saved policy on:

| Config | n_users | area_size | Policies |
|---|---|---|---|
| Trained | 100 | 3000m | all |
| Small | 50 | 3000m | all |
| Large | 150 | 3000m | all |
| Small area | 100 | 2000m | all |
| Large area | 100 | 4000m | all |

"All policies" = flat MLP, sorted MLP, spatial_grid CNN, graph GNN.
Obs dimensions do not change with n_users for flat/sorted (both use 2K features —
if K changes, MLP input changes too). **Important:** flat/sorted MLPs cannot handle
n_users ≠ training n_users either. This is a finding, not a bug — report it.

Correct approach for fair RQ3-A: train one MLP per user-count variant to establish
the retrained baseline; zero-shot eval shows the gap.

### RQ3-B: Structural shift (fleet size)

**Only the UAV-node-wise GNN is tested here.** MLP and CNN are explicitly excluded
because their action space dimensions are fixed at training time — this is a
documented limitation, not an oversight.

| Fleet size | n_uavs | Test |
|---|---|---|
| Trained | 4 | Reference |
| Larger | 6 | Zero-shot |
| Larger | 8 | Zero-shot |

The GNN pads UAV nodes to `max_fleet_size=8`, masks padded nodes. At inference on
6 UAVs: nodes 5–6 are active, nodes 7–8 are masked. At inference on 8 UAVs: all
nodes active.

**Expected outcome (hypothesis H6-B):** coverage degrades gracefully (< 15% drop)
for fleet sizes within the padding range (≤ 8), because the GNN has seen padding
masks during training.

### Deployment cost table

Measured values only. Logged during each run to `results/{algo}/timing.json`.

| Metric | Measured how |
|---|---|
| Training wall-time | `time.time()` around SB3 `learn()` |
| Inference latency | mean of 100 `policy.predict()` calls, real env obs |
| PSO re-plan latency | mean of 20 `pso_planner()` calls on test layouts |
| MPC step latency | mean of 50 `mpc_planner()` step calls |

---

## Figures

### Figure 1 — RQ1 (`plot_rq1.py`)
Bar chart: `final_cov` IQM ± 95% CI for all 5 obs_modes, best RL algo.
Secondary panel: training wall-clock per obs_mode (cost of each representation).

### Figure 2 — RQ2 (`plot_rq2.py`)
3×2 heatmap of paired difference `mean(RL − PSO)` per config.
Annotate: point estimate, 95% CI on difference, P(RL > PSO). Colour: green (RL
wins, CI excludes zero) / grey (inconclusive) / red (PSO wins, CI excludes zero).

### Figure 3 — RQ3 (`plot_rq3.py`)
Two panels:
- (a) RQ3-A: line chart of `final_cov` vs n_users / area_size per policy type
- (b) RQ3-B: bar chart of GNN coverage at fleet sizes 4/6/8 with 95% CI
- (c) Cost table: training time, inference latency, re-plan cost per method

---

## Implementation Order (correctness-first)

1. **Physics + env audit** — verify reward accounting, flight constraint consistency,
   existing baseline correctness before adding new components
2. **`uavbench/obs.py`** — extract and add `spatial_grid` + fixed-capacity `graph` builders
3. **`uavbench/policies/cnn_policy.py`** + tests
4. **`uavbench/policies/gnn_policy.py`** (UAV-node-wise actor) + tests incl. variable fleet
5. **`scripts/train_rl.py`** checkpoint saving + latency logging + policy auto-selection
6. **`uavbench/baselines.py`** warm-start PSO + MPC + tests
7. **`scripts/stats.py`** paired diff CI + prob of improvement
8. **`EnvConfig`** `failure_rate` / `single_failure` + `step_duration_s` + tests
9. **`scripts/sweep_rq2.py`** + `plot_rq2.py`
10. **`scripts/eval_generalization.py`** + `plot_rq3.py`
11. **`scripts/plot_rq1.py`**
12. **`docs/RESEARCH_CONTEXT.md`** + `requirements.txt`

---

## Testing Requirements

### `tests/test_policies.py`
- `test_cnn_extractor_output_shape`: forward pass → `(batch, 128)`
- `test_gnn_extractor_4uav`: graph with 4 UAVs + 100 users → `(1, 512)` UAV embeddings
- `test_gnn_extractor_6uav_variable`: same extractor, 6 UAVs → `(1, 512)` with correct mask
- `test_gnn_actor_output_shape`: actor outputs `(n_active_uavs, 3)` for 4 and 6 UAV cases
- `test_gnn_inactive_uavs_masked`: failed UAV nodes produce zero gradient contribution

### `tests/test_rq2.py`
- `test_single_failure_fires_once`: 100 episodes with `single_failure=True` → exactly 1 failure per episode
- `test_single_failure_timing`: failure step in `[ep_len//3, 2*ep_len//3]`
- `test_failure_rate_zero_no_failure`: 1000 steps, `failure_rate=0.0` → n_active unchanged
- `test_warm_start_pso_uses_prior_solution`: particle initialisation includes prior best

### `tests/test_generalization.py`
- `test_eval_generalization_runs`: completes on a saved flat-obs checkpoint
- `test_gnn_fleet_6_no_crash`: GNN policy trained on 4 UAVs runs on 6-UAV env
- `test_gnn_fleet_exceeds_max_raises`: 9 UAVs with max_fleet_size=8 → clear error

---

## Three Publishable Outcomes (do not favour any)

| Outcome | Interpretation |
|---|---|
| A: GNN substantially improves coverage and generalises | Relational obs are especially effective for coordinated UAV deployment |
| B: DRL matches PSO only under high mobility+failure | Practical decision boundary: use RL when re-planning latency is the bottleneck |
| C: Strong online optimiser matches DRL in most conditions | Establishes where learning fails and why; motivates future permutation-equivariant architectures |

---

## §8 Change Log (from initial design)

| Problem | Fix applied |
|---|---|
| P1: GNN cannot auto-generalise fleet sizes | UAV-node-wise actor head + padding/masking to `max_fleet_size` |
| P2: PyG/SB3 integration not automatic | Fixed-capacity graph tensors in Dict obs space; documented padding scheme |
| P3: Global pooling loses UAV-specific info | Individual UAV node embeddings preserved; active mask applied to actor |
| P4: Failure rate too aggressive | Replaced 0.02/step with single-failure-per-episode model; `failure_rate` retained but unused in RQ2 |
| P5: Baselines not strong enough | Added warm-start PSO and time-budgeted MPC to `baselines.py` |
| P6: CI overlap is wrong statistical test | Replaced with paired difference CI + P(improvement) via rliable |
| Additional: mobility in m/step | Added `step_duration_s` to `EnvConfig`; all speeds documented as m/s |
| Additional: RQ3 split | RQ3-A (distribution shift, all policies) and RQ3-B (structural shift, GNN only) |
| Additional: outage duration metric | Added to RQ2 metric list |
| Additional: inference latency measured | `timing.json` logged per run; cost table in Figure 3 |
