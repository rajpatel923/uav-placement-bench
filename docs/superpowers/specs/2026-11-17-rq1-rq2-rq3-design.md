# RQ1–RQ3 Extension Design

**Goal:** Extend the UAV placement benchmark with three research questions to increase
paper novelty for VTC2027/AAAI-28 submission.

**Approach selected:** Approach B — Full RQ1 + RQ2, lightweight RQ3.

**Compute target:** Threadripper PRO 9965WX + 2× RTX PRO 6000 (192 GB VRAM total).
Estimated total additional sweep time: ~12–15 hours.

---

## Global Constraints

- Python 3.10+, NumPy vectorized, no global `np.random.*`
- New `EnvConfig` fields must default to values that preserve existing Regime I/II/III behaviour
- `pytest -q` must pass after every task
- Never tune on the test split (seeds 10000+i); tuning uses val split (seeds 20000+i)
- No physics constants changed without citing source + equation number
- All new figures use the same IQM + 95% bootstrap CI reporting as existing figures
- PyTorch Geometric (`torch_geometric`) is the only new dependency permitted
- Smoke test: `--timesteps 20000 --n-eval 5 --seeds 0 1 --out /tmp/smoke` for every new script

---

## Research Questions

### RQ1 — Representation Sensitivity

> How do coordinate-based, spatial-grid, and graph-based observation representations
> affect DRL performance, convergence, scalability, and generalization in multi-UAV
> placement?

**Hypothesis H4:** Graph-based representations (GNN) will close the RL-vs-PSO gap
more than coordinate or grid representations, because they encode spatial relationships
invariantly across fleet sizes.

### RQ2 — Dynamic Complexity Threshold

> Under what levels of user mobility and UAV failure frequency does DRL outperform
> online re-optimization in service continuity and recovery?

**Hypothesis H5:** RL begins to match or exceed PSO between moderate (5 m/step) and
high (15 m/step) mobility when combined with failure rates ≥ 0.02/step, because
classical re-planning latency becomes the bottleneck.

### RQ3 — Generalization and Deployment Cost

> Do policies trained on the default config generalize zero-shot to unseen fleet sizes,
> user distributions, and area sizes, and how do training/inference costs compare?

**Hypothesis H6:** The GNN policy generalizes to unseen fleet sizes (4→6→8 UAVs);
MLP and CNN policies do not, because their input dimensions are fixed at training time.

---

## New Files

| Path | Role |
|---|---|
| `uavbench/obs.py` | All 5 observation builders extracted from `env.py` |
| `uavbench/policies/__init__.py` | Package marker |
| `uavbench/policies/cnn_policy.py` | SB3 `BaseFeaturesExtractor` for `spatial_grid` obs |
| `uavbench/policies/gnn_policy.py` | SB3 `BaseFeaturesExtractor` for `graph` obs (PyG) |
| `scripts/sweep_rq2.py` | RQ2: 6-config complexity sweep |
| `scripts/eval_generalization.py` | RQ3: zero-shot transfer evaluation |
| `scripts/plot_rq1.py` | Figure 1: representation comparison bar chart |
| `scripts/plot_rq2.py` | Figure 2: complexity threshold heatmap |
| `scripts/plot_rq3.py` | Figure 3: generalization lines + cost table |
| `tests/test_policies.py` | Tests for CNN and GNN extractors |
| `tests/test_rq2.py` | Tests for failure_rate env behaviour |
| `tests/test_generalization.py` | Tests for zero-shot eval script |

### Modified Files

| Path | Change |
|---|---|
| `uavbench/env.py` | Add `obs_mode: "spatial_grid"` and `"graph"`; add `failure_rate: float = 0.0`; delegate obs building to `obs.py` |
| `scripts/train_rl.py` | Auto-select policy class from obs_mode; save checkpoint `.zip` after training |
| `docs/RESEARCH_CONTEXT.md` | Add RQ1–RQ3, H4–H6 |
| `requirements.txt` | Add `torch_geometric>=2.5` |

---

## RQ1: Observation Representations

### Five obs_modes

| obs_mode | Observation shape | Policy class | Note |
|---|---|---|---|
| `flat` | `(2K + 3U,)` | `MlpPolicy` | Existing |
| `sorted` | `(2K + 3U,)` | `MlpPolicy` | Existing |
| `cluster_summary` | `(5C + 3U,)` | `MlpPolicy` | Existing |
| `spatial_grid` | Dict `{"grid": (1,32,32), "uavs": (3U,)}` | `MlpPolicy` + `CnnExtractor` | New |
| `graph` | Dict `{"x": (N,3), "edge_index": (2,E), "batch": (N,)}` | `MlpPolicy` + `GNNExtractor` | New |

Where U = `n_uavs`, K = `n_users`, C = `n_clusters` (default 10), N = U+K nodes.

### Spatial Grid (`uavbench/policies/cnn_policy.py`)

```python
class CnnExtractor(BaseFeaturesExtractor):
    """2-layer CNN over 32×32 user occupancy grid + UAV position vector."""
    def __init__(self, observation_space, features_dim=128):
        # Conv1: 1→16 channels, 3×3, ReLU
        # Conv2: 16→32 channels, 3×3, ReLU
        # AdaptiveAvgPool2d(4,4) → flatten → 32*4*4 = 512
        # Concat UAV vector → FC(512 + 3U, features_dim)
```

Grid construction: bin each user (x,y) into a 32×32 grid over [0, area_size]². Value = 1 if
any user in cell, 0 otherwise. Normalised to [0,1]. UAV positions normalised by area_size and h_max.

### Graph (`uavbench/policies/gnn_policy.py`)

```python
class GNNExtractor(BaseFeaturesExtractor):
    """2-layer GATv2 over UAV+user graph → global mean pool → embedding."""
```

**Graph construction (in `uavbench/obs.py`):**
- Nodes: U UAV nodes + K user nodes. Total N = U + K = 104 (default).
- Node features (3 dims each):
  - UAV: `[x/area, y/area, z/h_max]`
  - User: `[x/area, y/area, 0.0]` (z=0, on ground)
- Edges (undirected, stored as directed both ways):
  - UAV↔UAV: fully connected (all pairs)
  - UAV↔User: connect if 2D distance ≤ `graph_edge_radius` (default: 1500m = area/2)
  - User↔User: none (too many edges; 100×100 = 10k)
- Edge features: none (GATv2 uses node features only for attention)

**GATv2 architecture:**
- Layer 1: GATv2Conv(in=3, out=64, heads=4, concat=True) → 256 dims per node
- Layer 2: GATv2Conv(in=256, out=64, heads=1, concat=False) → 64 dims per node
- Global mean pool over all nodes → 64-dim graph embedding
- FC(64, features_dim=128) → actor/critic heads

**SB3 observation space:** `gym.spaces.Dict` with keys `x` (node features), `edge_index`,
`batch` (all-zeros for single-graph). SB3's `MultiInputPolicy` handles Dict spaces natively.

**New `EnvConfig` field:**
```python
graph_edge_radius: float = 1500.0  # metres; UAV↔user edge threshold
```

---

## RQ2: Dynamic Complexity Threshold

### New `EnvConfig` field

```python
failure_rate: float = 0.0
# Per-step probability that each active UAV independently fails.
# 0.0 = no failure (default; preserves all existing behaviour).
# Applied in step() after action execution, before coverage computation.
# Interacts with existing shock logic: failure_rate fires every step;
# shock_step fires once. They are independent flags.
```

### Failure mechanic (in `env.py step()`)

```python
if cfg.failure_rate > 0.0:
    for i in range(self.n_uavs):
        if self._failed_mask[i] and self.np_random.random() < cfg.failure_rate:
            self._failed_mask[i] = False  # UAV fails
            self.n_active_uavs -= 1
```

(Uses existing `_failed_mask` infrastructure from Regime III shock.)

### 6-config sweep (`scripts/sweep_rq2.py`)

| Config | `user_speed` | `failure_rate` | Label |
|---|---|---|---|
| C1 | 0.0 | 0.0 | static-no-failure |
| C2 | 5.0 | 0.0 | mobile-low |
| C3 | 15.0 | 0.0 | mobile-high |
| C4 | 0.0 | 0.02 | static-failure |
| C5 | 5.0 | 0.02 | mobile-low-failure |
| C6 | 15.0 | 0.02 | mobile-high-failure |

Methods per config: PSO, k-means (baselines) + best-obs RL algo from RQ1 (determined after RQ1 sweep). 10 seeds each. Results written to `results_rq2/{config}/`.

**Output:** `plot_rq2.py` produces a 3×2 heatmap of (RL IQM − PSO IQM) per cell.
Green = RL wins (positive gap), red = PSO wins (negative gap).

---

## RQ3: Generalization and Deployment Cost

### Zero-shot transfer protocol (`scripts/eval_generalization.py`)

Trained policy checkpoint (`.zip` from SB3) is loaded and evaluated on 9 transfer configs
(20 test layouts each, seeds 10000+i). No retraining.

**Transfer configs:**

| Dimension | Values | Policies tested |
|---|---|---|
| Fleet size (n_uavs) | 4 (trained), 6, 8 | GNN only (MLP/CNN input dim changes) |
| User count (n_users) | 100 (trained), 50, 150 | GNN + MLP + CNN |
| Area size (area_size) | 3000 (trained), 2000, 4000 | GNN + MLP + CNN |

**Checkpoint saving:** `train_rl.py` saves `results/{regime}/{algo}/seed{s}/policy.zip`
after training. `eval_generalization.py` takes `--checkpoint path/to/policy.zip` and
`--transfer-config` JSON.

**Key finding framing:** Report absolute `final_cov` and relative drop vs trained-config
performance. A drop > 10% with non-overlapping CIs = "does not generalise."

### Deployment cost table

Measured values only (no estimates). Columns: algo, training wall-time (hours),
inference latency per step (ms), re-plan latency (ms), hardware.

Measured during the sweep and logged to `results/{algo}/timing.json`.

---

## Figures

### Figure 1 — RQ1 (`plot_rq1.py`)
Bar chart: `final_cov` IQM ± 95% CI for each obs_mode × SAC (best RL algo assumed).
Grouped bars, same colour scheme as existing `plot_figures.py`. X-axis: obs_mode.

### Figure 2 — RQ2 (`plot_rq2.py`)
3×2 heatmap of (best-RL IQM − PSO IQM). Rows: mobility level. Columns: failure mode.
Annotate each cell with the numeric gap and a ✓/✗ for statistical significance.

### Figure 3 — RQ3 (`plot_rq3.py`)
Two panels:
- (a) Line chart: `final_cov` vs transfer config, one line per policy type.
  X-axis: transfer dimension value. Dashed line = trained-config performance.
- (b) Cost table rendered as a matplotlib table: training time, inference latency,
  re-plan cost per method.

---

## Updated Hypotheses

Add to `docs/RESEARCH_CONTEXT.md`:

| ID | Hypothesis |
|---|---|
| H4 | Graph-based observation closes the RL-vs-PSO gap more than coordinate or grid representations |
| H5 | RL begins to match PSO between C2 and C3 when failure_rate ≥ 0.02 |
| H6 | GNN policy generalises zero-shot to unseen fleet sizes; MLP/CNN do not |

---

## Testing Requirements

### `tests/test_policies.py`
- `test_cnn_extractor_output_shape`: forward pass returns `(batch, features_dim)`
- `test_gnn_extractor_output_shape`: forward pass on a 4-UAV + 100-user graph returns `(1, 128)`
- `test_gnn_extractor_variable_fleet`: same extractor handles 6-UAV graph without error

### `tests/test_rq2.py`
- `test_failure_rate_zero_no_failure`: 1000 steps with failure_rate=0.0 → all UAVs active
- `test_failure_rate_one_all_fail`: failure_rate=1.0 → all UAVs fail by step 2
- `test_failure_rate_independent`: failure events are independent across UAVs (chi-square)

### `tests/test_generalization.py`
- `test_eval_generalization_runs`: eval_generalization.py completes on a saved checkpoint
- `test_gnn_handles_larger_fleet`: GNN policy loaded on 4-UAV env runs on 6-UAV env

---

## Implementation Order

1. `uavbench/obs.py` + env.py refactor (extract existing obs builders, add spatial_grid + graph)
2. `uavbench/policies/cnn_policy.py` + tests
3. `uavbench/policies/gnn_policy.py` + tests
4. `scripts/train_rl.py` checkpoint saving + policy auto-selection
5. `scripts/sweep_rq2.py` + `EnvConfig.failure_rate` + tests
6. `scripts/eval_generalization.py` + tests
7. `scripts/plot_rq1.py`, `plot_rq2.py`, `plot_rq3.py`
8. `docs/RESEARCH_CONTEXT.md` + `requirements.txt` update
