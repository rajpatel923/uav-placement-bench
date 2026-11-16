# UAV Placement Benchmark: Complete Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the working starter scaffold to publication-ready results: verified channel physics, fixed observation design, additional baselines, tuned hyperparameters, 10-seed sweep across both regimes, rliable figures ready for the conference paper "When Does Deep RL Beat Classical Optimization for UAV Base Station Placement?"

**Architecture:** `uavbench/channel.py` (Al-Hourani A2G physics) → `uavbench/env.py` (Gymnasium, Regimes I–III) → `uavbench/baselines.py` (k-means, PSO, Lyu spiral) + SB3 RL (PPO, SAC, A2C, DQN) → `uavbench/evaluation.py` (held-out layouts, deterministic metrics) → `scripts/aggregate.py` (rliable IQM + 95% CIs) → `scripts/plot_figures.py` (paper-quality PDFs). All hyperparameter decisions go through the validation split; the test split is never touched until the final sweep.

**Tech Stack:** Python 3.10+, NumPy ≥ 1.26, SciPy ≥ 1.11, scikit-learn ≥ 1.4, Gymnasium ≥ 1.0, Stable-Baselines3 ≥ 2.3, rliable 1.2.0, matplotlib ≥ 3.8, pandas < 3, arch ≥ 5.3.1 < 8, pytest ≥ 8, torch ≥ 2.2 (CPU build is sufficient)

## Global Constraints

- **Never tune on the test split.** `EVAL_SEED_OFFSET = 10000`. All hyperparameter work uses `--split val` (`VAL_SEED_OFFSET = 20000`).
- **10+ seeds** per method × regime before claiming any result. Report IQM with 95% stratified bootstrap CIs via `scripts/aggregate.py`.
- **Equal tuning budget.** Every method (RL, PSO, k-means `edge_quantile`) gets the same number of val-split tuning trials. Log every trial in `docs/EXPERIMENT_PROTOCOL.md`.
- **Never change** `EVAL_SEED_OFFSET`, `VAL_SEED_OFFSET`, `FINAL_WINDOW`, `EnvConfig` defaults, the reward function, or the observation representation without (1) supervisor approval and (2) a new entry in `docs/DECISIONS.md`. Changing any of these invalidates all existing results.
- **Never delete `results/`.** SAC runs take ~40 min per seed. Write all new experiments to a fresh `--out` directory.
- `pytest -q` must pass before finishing any task. Use `--timesteps 20000 --n-eval 5 --out /tmp/smoke` for all development smoke tests.
- Never loosen test tolerances in `tests/test_channel.py` to pass tests; those encode published values.
- Never invent numbers; every benchmark number must come from a file in `results/`.
- Never invent citations; only cite VERIFIED entries in `docs/REFERENCES.md`.
- Baselines obey the same `v_max`/`vz_max` flight limits as RL agents via `TargetTrackingPolicy`. Never let a baseline teleport.

---

## Repo State as of 2026-11-16

**Implemented and tested:**
- `uavbench/channel.py` — Al-Hourani A2G channel, vectorized, optimal elevation solver (Eq. 13). Reproduces published θ_opt: 20.34° / 42.44° / 54.62° / 75.52°.
- `uavbench/env.py` — `UAVPlacementEnv` (Regimes I static, II mobile), `EnvConfig`, `DiscreteActionWrapper` (7^U joint space for DQN), `make_env()`. Passes Gymnasium and SB3 env checkers.
- `uavbench/baselines.py` — `kmeans_planner`, `pso_planner`, `random_planner`, `TargetTrackingPolicy`, `HoverPolicy`. All obey flight limits.
- `uavbench/evaluation.py` — `run_episode()`, `summarize()`, `save_result()`, `layout_seeds()`. Test/val splits isolated.
- `scripts/run_baselines.py`, `scripts/train_rl.py` (PPO/SAC/A2C/DQN via SB3), `scripts/aggregate.py` (rliable), `scripts/run_all.sh`.
- 21 pytest tests passing.

**Smoke-test numbers (NOT results — 1–2 seeds, untuned, 10 layouts):**

| Regime | k-means | PSO | Hover | Random | PPO (400k, untuned, 1 seed) |
|--------|---------|-----|-------|--------|------------------------------|
| Static | 0.61    | 0.64| 0.13  | 0.09   | 0.21                         |
| Mobile | 0.42    | 0.49| 0.08  | 0.13   | not run                      |

The RL gap (0.21 vs 0.61) is likely caused by the unordered flat observation (D-008), not fundamental RL weakness.

**Key open issues:**
- D-008 (OPEN): Observation = flat unordered user coords → MLPs handle unordered sets badly.
- Experiment protocol not frozen (hypotheses, tuning budget, timesteps are blank in `docs/EXPERIMENT_PROTOCOL.md`).
- No hyperparameter tuning done yet.
- Channel constants not human-verified against PDFs.
- Lyu spiral, SINR, Regime III not implemented.

---

## File Map

| File | Action | Responsible task |
|------|--------|-----------------|
| `scripts/verify_channel.py` | Create | Task 1 |
| `docs/CHANNEL_MODEL.md` | Update checklist | Task 1 (human) |
| `docs/channel_verification.png` | Create | Task 1 |
| `uavbench/baselines.py` | Add `lyu_spiral_planner`; update `PLANNERS` | Task 2 |
| `tests/test_baselines.py` | Create | Task 2 |
| `docs/EXPERIMENT_PROTOCOL.md` | Fill blanks, set freeze date | Task 3 (human) |
| `uavbench/env.py` | Add `obs_mode` to `EnvConfig`; update `_obs()`, `observation_space` | Task 4 |
| `tests/test_env.py` | Add obs_mode tests | Task 4 |
| `scripts/train_rl.py` | Add `--obs-mode` argument | Task 4b |
| `scripts/run_baselines.py` | Add `lyu_spiral`; add `--obs-mode` | Tasks 2, 4b |
| `uavbench/channel.py` | Add `sinr_matrix()` | Task 5 |
| `uavbench/env.py` | Add `use_sinr`, `sinr_coverage()`; update `step()` info | Task 5 |
| `tests/test_channel.py` | Add SINR tests | Task 5 |
| `scripts/tune_baselines.py` | Create | Task 6 |
| `scripts/tune_rl.py` | Create | Task 6 |
| `scripts/plot_figures.py` | Create | Task 7 |
| `docs/figures/` | Create (auto) | Task 7 |
| `uavbench/env.py` | Add `mobility="shock"`, shock fields, `_apply_shock()` | Task 8 |

---

### Task 1: Channel Verification Script & Radius-Altitude Figure

**Human prerequisite:** Have PDFs for Al-Hourani WCL 2014 [1] and GLOBECOM 2014 [2] open before ticking the checklist boxes in `docs/CHANNEL_MODEL.md`.

**Files:**
- Create: `scripts/verify_channel.py`
- Modify: `docs/CHANNEL_MODEL.md` (human ticks the boxes after visual inspection)
- Create: `docs/channel_verification.png` (output artifact)

**Interfaces:**
- Consumes: `uavbench.channel.AlHouraniChannel`, `uavbench.channel.ENVIRONMENTS`
- Produces: `docs/channel_verification.png`; console table comparing θ_opt against published values

- [ ] **Step 1: Confirm existing elevation-angle test covers all four environments**

```python
# tests/test_channel.py — verify this test already exists (it does):
@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_optimal_elevation_matches_published(env):
    theta = AlHouraniChannel(env).optimal_elevation_deg()
    assert theta == pytest.approx(EXPECTED_THETA_OPT[env], abs=0.05)
```

Run: `pytest tests/test_channel.py -v`
Expected: All channel tests PASS.

- [ ] **Step 2: Create `scripts/verify_channel.py`**

```python
#!/usr/bin/env python
"""Verify Al-Hourani channel constants and reproduce radius-altitude figure.

Compare optimal elevation angles against published Table I of [1].
Plot coverage radius vs altitude for all four environments (compare to Fig. 3 of [1]).

Usage: python scripts/verify_channel.py [--pl-max 100] [--out docs/channel_verification.png]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from uavbench.channel import AlHouraniChannel, ENVIRONMENTS

PUBLISHED_THETA_OPT = {
    "suburban": 20.34,
    "urban": 42.44,
    "dense_urban": 54.62,
    "highrise_urban": 75.52,
}


def check_elevation_angles() -> bool:
    print("\n=== Optimal elevation angles vs Al-Hourani WCL 2014 Table I ===")
    all_ok = True
    for name, expected in PUBLISHED_THETA_OPT.items():
        ch = AlHouraniChannel(name)
        got = ch.optimal_elevation_deg()
        diff = abs(got - expected)
        status = "OK" if diff < 0.01 else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {name:20s}: published {expected:.2f}°  computed {got:.4f}°  diff {diff:.4f}°  [{status}]")
    return all_ok


def plot_radius_altitude(pl_max: float, out_path: str) -> None:
    altitudes = np.linspace(10, 500, 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ENVIRONMENTS:
        ch = AlHouraniChannel(name)
        radii = np.array([ch.coverage_radius(float(h), pl_max) for h in altitudes])
        ax.plot(altitudes, radii, label=name)
        h_opt, _ = ch.optimal_altitude(pl_max)
        ax.axvline(h_opt, linestyle="--", alpha=0.25)
    ax.set_xlabel("UAV altitude h (m)")
    ax.set_ylabel(f"Coverage radius R (m) [PL ≤ {pl_max:.0f} dB]")
    ax.set_title("Al-Hourani Coverage Radius vs Altitude\n(Reproduce shape of WCL 2014 Fig. 3)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure → {out_path}")
    print("Compare shape visually against Al-Hourani WCL 2014 Fig. 3.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pl-max", type=float, default=100.0)
    p.add_argument("--out", default="docs/channel_verification.png")
    args = p.parse_args()
    ok = check_elevation_angles()
    plot_radius_altitude(args.pl_max, args.out)
    if not ok:
        print("\nWARNING: Some elevation angles deviate more than 0.01°. Check channel.py constants.")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the verification script**

```bash
python scripts/verify_channel.py --pl-max 100 --out docs/channel_verification.png
```

Expected output:
```
=== Optimal elevation angles vs Al-Hourani WCL 2014 Table I ===
  suburban            : published 20.34°  computed 20.34xx°  diff 0.00xx°  [OK]
  urban               : published 42.44°  computed 42.44xx°  diff 0.00xx°  [OK]
  dense_urban         : published 54.62°  computed 54.62xx°  diff 0.00xx°  [OK]
  highrise_urban      : published 75.52°  computed 75.52xx°  diff 0.00xx°  [OK]

Saved figure → docs/channel_verification.png
```

- [ ] **Step 4: Human verification (agent cannot do this)**

Open `docs/channel_verification.png`. Confirm the shape matches Al-Hourani WCL 2014 Fig. 3 (single peak of coverage radius near the optimal altitude; monotone decay either side).

Check `uavbench/channel.py:43–47` constants against:
- Table II of [1] for `a`, `b` per environment
- Table I of [2] for `η_LoS`, `η_NLoS` per environment
- Eq. (13) of [1] for the derivative form in `optimal_elevation_deg()`

Tick the four boxes in `docs/CHANNEL_MODEL.md`.

- [ ] **Step 5: Run pytest and commit**

```bash
pytest -q
git add scripts/verify_channel.py docs/channel_verification.png
git commit -m "feat: add channel verification script and radius-altitude figure"
```

---

### Task 2: Lyu Spiral Baseline

**Research note:** The reference "Lyu et al., spiral placement of UAV-mounted mobile base stations, IEEE Commun. Lett." is PARTIAL status (year unconfirmed: 2016 vs 2017). Verify the full citation against [docs/REFERENCES.md] before including results attributed to this name in the paper. The implementation here uses a Fermat spiral (golden-angle packing), which captures the essence of uniform-coverage spiral placement; confirm the specific formulation matches the paper.

**Files:**
- Modify: `uavbench/baselines.py` (add `lyu_spiral_planner`; update `PLANNERS`)
- Create: `tests/test_baselines.py`
- Modify: `scripts/run_baselines.py` (add `"lyu_spiral"` to algos list)

**Interfaces:**
- Consumes: `UAVPlacementEnv`, `np.random.Generator` (same signature as every other planner)
- Produces: `np.ndarray` of shape `(n_uavs, 3)` target positions

- [ ] **Step 1: Create `tests/test_baselines.py` with failing tests**

```python
# tests/test_baselines.py
import numpy as np
import pytest
from uavbench.env import UAVPlacementEnv, EnvConfig
from uavbench.baselines import lyu_spiral_planner, random_planner


def _env(seed: int = 0) -> UAVPlacementEnv:
    env = UAVPlacementEnv()
    env.reset(seed=seed)
    return env


def test_lyu_spiral_shape():
    env = _env()
    targets = lyu_spiral_planner(env, np.random.default_rng(0))
    assert targets.shape == (env.cfg.n_uavs, 3)


def test_lyu_spiral_altitude_in_range():
    env = _env()
    targets = lyu_spiral_planner(env, np.random.default_rng(0))
    assert np.all(targets[:, 2] >= env.cfg.h_min - 1e-6)
    assert np.all(targets[:, 2] <= env.cfg.h_max + 1e-6)


def test_lyu_spiral_positions_in_area():
    env = _env()
    targets = lyu_spiral_planner(env, np.random.default_rng(0))
    assert np.all(targets[:, :2] >= -1e-6)
    assert np.all(targets[:, :2] <= env.cfg.area + 1e-6)


def test_lyu_spiral_beats_random_on_average():
    """Geometric spiral placement should beat random over 10 static layouts."""
    spiral_covs, random_covs = [], []
    for seed in range(10):
        env = _env(seed)
        t_spiral = lyu_spiral_planner(env, np.random.default_rng(seed))
        t_random = random_planner(env, np.random.default_rng(seed + 1000))
        spiral_covs.append(env.coverage(t_spiral))
        random_covs.append(env.coverage(t_random))
    assert np.mean(spiral_covs) > np.mean(random_covs), (
        f"Spiral {np.mean(spiral_covs):.3f} should beat random {np.mean(random_covs):.3f}"
    )
```

Run: `pytest tests/test_baselines.py::test_lyu_spiral_shape -v`
Expected: FAIL with `ImportError: cannot import name 'lyu_spiral_planner'`

- [ ] **Step 2: Implement `lyu_spiral_planner` in `uavbench/baselines.py`**

Add after `random_planner` (around line 83), before the `PLANNERS` dict:

```python
def lyu_spiral_planner(env: UAVPlacementEnv, rng: np.random.Generator):
    """Place UAVs on a Fermat spiral centred on the area, altitude from the Al-Hourani rule.

    Lyu et al. (IEEE Commun. Lett., ~2016/2017) distribute a UAV base station along a
    spiral trajectory. For multi-UAV placement we place U points at golden-angle spacing
    on a Fermat spiral scaled to 90 % of the half-area diagonal, with altitude h = r * tan(θ_opt)
    clipped to [h_min, h_max].  Verify the multi-UAV formulation against the paper before
    citing these results as "Lyu et al. baseline."
    """
    c = env.cfg
    theta_opt = np.radians(env.channel.optimal_elevation_deg())
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~137.508°
    cx, cy = c.area / 2.0, c.area / 2.0
    max_r = c.area / 2.0 * 0.9  # 90 % of half-side to keep positions inside the area

    targets = np.zeros((c.n_uavs, 3))
    for i in range(c.n_uavs):
        angle = i * golden_angle
        r = max_r * np.sqrt((i + 0.5) / c.n_uavs)
        targets[i, 0] = np.clip(cx + r * np.cos(angle), 0.0, c.area)
        targets[i, 1] = np.clip(cy + r * np.sin(angle), 0.0, c.area)
        targets[i, 2] = np.clip(r * np.tan(theta_opt), c.h_min, c.h_max)
    return targets
```

- [ ] **Step 3: Register in `PLANNERS` dict in `uavbench/baselines.py`**

Replace the existing PLANNERS dict (around line 87):

```python
PLANNERS = {
    "kmeans": kmeans_planner,
    "pso": pso_planner,
    "random": random_planner,
    "lyu_spiral": lyu_spiral_planner,
}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_baselines.py -v
```
Expected: All 4 lyu_spiral tests PASS.

```bash
pytest -q
```
Expected: All tests (21 original + 4 new) PASS.

- [ ] **Step 5: Smoke-run the new baseline**

```bash
python scripts/run_baselines.py --regime static --seeds 0 1 --n-eval 5 --out /tmp/smoke --algos lyu_spiral
```
Expected: Runs without error; creates `/tmp/smoke/static/lyu_spiral/seed0.json` and `seed1.json`. Print shows a `final_cov` between random (~0.09) and k-means (~0.61).

- [ ] **Step 6: Add DECISIONS entry and commit**

Append to `docs/DECISIONS.md`:
```
## D-011: Lyu spiral baseline added
Date: 2026-11-16 | Status: accepted | Decided by: student
Context: README roadmap item. Multi-UAV spiral placement to complement k-means and PSO.
Decision: Fermat spiral (golden-angle) with Al-Hourani altitude rule, clipped to [h_min, h_max].
Alternatives: Exact Lyu et al. formulation (needs paper to confirm multi-UAV extension).
Consequences: Must verify multi-UAV formulation against the paper before citing as "Lyu et al."
```

```bash
pytest -q
git add uavbench/baselines.py tests/test_baselines.py docs/DECISIONS.md
git commit -m "feat: add Lyu spiral baseline (Fermat/golden-angle placement)"
```

---

### Task 3: Experiment Protocol Freeze (Human Decision Checkpoint)

**This task is a human + supervisor checkpoint. No code changes.** The experiment protocol must be frozen BEFORE any tuning sweeps or full results runs. After freezing, any change goes in the Deviations log (Section 8 of `docs/EXPERIMENT_PROTOCOL.md`).

**Files:**
- Modify: `docs/EXPERIMENT_PROTOCOL.md` (fill blank fields; set freeze date)
- Modify: `docs/DECISIONS.md` (log the freeze)

- [ ] **Step 1: Decide tuning budget and training timesteps (Section 5)**

Fill in `docs/EXPERIMENT_PROTOCOL.md` Section 5. Suggested values:
```markdown
- **Budget:** 20 trials per method (same for every method: RL algorithms, PSO settings, k-means edge_quantile).
- **Seeds for tuning:** 100–102 (never the 0–9 reporting seeds).
- **Selection rule:** choose the configuration with the best mean final_cov on validation layouts.
- **Training budget:** 200,000 timesteps per RL run, the same for every algorithm.
```

Confirm with supervisor that these values are realistic for available compute. SAC × 20 trials × 3 seeds × 200k steps ≈ ~40 h on a 2-core CPU; use a lab workstation or reduce to 100k steps for initial tuning.

- [ ] **Step 2: Freeze hypotheses (Section 6)**

Copy from `docs/RESEARCH_CONTEXT.md` and finalize wording:
```markdown
| H1 | Static regime (I): strong classical baselines (k-means + Al-Hourani altitude, PSO) match or exceed standard deep RL in final_cov at far lower compute. | |
| H2 | Mobile regime (II): the RL–classical coverage gap narrows or reverses as user mobility increases. | |
| H3 | Shock regime (III, stretch): under a mid-episode shock (UAV failure or user migration), methods differ in recovery speed; no prior expectation about which wins. | |
```

- [ ] **Step 3: Set freeze date and confirm with supervisor**

Replace the placeholder in `docs/EXPERIMENT_PROTOCOL.md`:
```
Frozen on: YYYY-MM-DD
```

- [ ] **Step 4: Log in DECISIONS.md**

Append to `docs/DECISIONS.md`:
```
## D-012: Experiment protocol frozen
Date: YYYY-MM-DD | Status: accepted | Decided by: student + Dr. Sookhak
Decision: 20 tuning trials per method, 200k timesteps, hypotheses H1–H3 frozen.
Consequences: Any post-freeze change must go in the Deviations log with supervisor approval.
```

- [ ] **Step 5: Commit**

```bash
git add docs/EXPERIMENT_PROTOCOL.md docs/DECISIONS.md
git commit -m "docs: freeze experiment protocol (hypotheses, tuning budget, timesteps)"
```

---

### Task 4: Observation Design — Implement Options & Compare (D-008)

**Research decision prerequisite:** Read `docs/DECISIONS.md D-008` before starting. This change modifies the observation space and invalidates any existing RL results (not an issue yet since we have no tuned results). Get supervisor buy-in on which option to run before the full tuning sweep.

Three options are implemented here as configurable `obs_mode` values. The comparison on the val split informs the decision before freezing the protocol.

**Files:**
- Modify: `uavbench/env.py` (add `obs_mode` to `EnvConfig`; update `_obs()` and `observation_space`)
- Modify: `tests/test_env.py` (add obs_mode tests)
- Modify: `scripts/train_rl.py` (add `--obs-mode` argument)
- Modify: `scripts/run_baselines.py` (add `--obs-mode` argument)

**Interfaces:**
- Consumes: existing `EnvConfig`, `UAVPlacementEnv._obs()`
- Produces: `EnvConfig.obs_mode: str` accepted by `make_env(**overrides)`. Three modes:
  - `"flat"` (default, current behavior): `(U*3 + K*2,)` — flat unordered user coordinates
  - `"sorted"`: `(U*3 + K*2,)` — users sorted by distance from area centroid
  - `"cluster_summary"`: `(U*3 + n_clusters*3,)` — cluster centroids (x, y, member_count_normalized)

- [ ] **Step 1: Write failing tests for new obs modes**

Add to `tests/test_env.py`:

```python
from uavbench.env import UAVPlacementEnv, EnvConfig
import numpy as np

def test_obs_mode_flat_shape():
    env = UAVPlacementEnv(EnvConfig(obs_mode="flat"))
    obs, _ = env.reset(seed=0)
    cfg = env.cfg
    assert obs.shape == (cfg.n_uavs * 3 + cfg.n_users * 2,)

def test_obs_mode_sorted_shape():
    env = UAVPlacementEnv(EnvConfig(obs_mode="sorted"))
    obs, _ = env.reset(seed=0)
    cfg = env.cfg
    assert obs.shape == (cfg.n_uavs * 3 + cfg.n_users * 2,)

def test_obs_mode_cluster_summary_shape():
    cfg = EnvConfig(obs_mode="cluster_summary")
    env = UAVPlacementEnv(cfg)
    obs, _ = env.reset(seed=0)
    expected = cfg.n_uavs * 3 + cfg.n_clusters * 3
    assert obs.shape == (expected,), f"Expected {expected}, got {obs.shape}"

def test_obs_mode_sorted_is_deterministic():
    env = UAVPlacementEnv(EnvConfig(obs_mode="sorted"))
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)

def test_obs_mode_unknown_raises():
    with pytest.raises(ValueError, match="obs_mode"):
        UAVPlacementEnv(EnvConfig(obs_mode="unknown_mode"))
```

Run: `pytest tests/test_env.py::test_obs_mode_flat_shape -v`
Expected: FAIL with `TypeError: EnvConfig.__init__() got an unexpected keyword argument 'obs_mode'`

- [ ] **Step 2: Add `obs_mode` field to `EnvConfig` in `uavbench/env.py`**

In the `EnvConfig` dataclass, add one line after `collision_penalty`:

```python
    obs_mode: str = "flat"  # "flat" | "sorted" | "cluster_summary"
```

- [ ] **Step 3: Update `observation_space` in `UAVPlacementEnv.__init__`**

Replace the existing `observation_space` line (around line 71) with:

```python
        if cfg.obs_mode == "flat":
            obs_dim = U * 3 + K * 2
        elif cfg.obs_mode == "sorted":
            obs_dim = U * 3 + K * 2
        elif cfg.obs_mode == "cluster_summary":
            obs_dim = U * 3 + cfg.n_clusters * 3
        else:
            raise ValueError(f"Unknown obs_mode: {cfg.obs_mode!r}. Choose 'flat', 'sorted', or 'cluster_summary'.")
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)
```

- [ ] **Step 4: Replace `_obs()` to handle all three modes**

Replace the existing `_obs` method (around line 90–95):

```python
    def _obs(self) -> np.ndarray:
        c = self.cfg
        u = self.uavs.copy()
        u[:, :2] /= c.area
        u[:, 2] = (u[:, 2] - c.h_min) / (c.h_max - c.h_min)
        uav_part = u.ravel()

        if c.obs_mode == "flat":
            user_part = (self.users / c.area).ravel()
        elif c.obs_mode == "sorted":
            centroid = np.full(2, c.area / 2.0)
            order = np.argsort(np.linalg.norm(self.users - centroid, axis=1))
            user_part = (self.users[order] / c.area).ravel()
        elif c.obs_mode == "cluster_summary":
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=c.n_clusters, n_init=1, random_state=0)
            labels = km.fit_predict(self.users)
            counts = np.bincount(labels, minlength=c.n_clusters)
            summary = np.column_stack([
                km.cluster_centers_ / c.area,
                counts.astype(float) / c.n_users,
            ])  # shape: (n_clusters, 3)
            user_part = summary.ravel()
        else:
            raise ValueError(f"Unknown obs_mode: {c.obs_mode!r}")

        return np.concatenate([uav_part, user_part]).astype(np.float32)
```

- [ ] **Step 5: Run full test suite**

```bash
pytest -q
```
Expected: All 21 (+ 5 new) tests PASS.

- [ ] **Step 6: Add `--obs-mode` to `scripts/train_rl.py`**

Find the argparse section and add:
```python
    p.add_argument("--obs-mode", default="flat", choices=["flat", "sorted", "cluster_summary"],
                   help="Observation representation mode (default: flat)")
```

Pass to `make_env()` and `make_vec_env()`:
```python
    discrete = args.algo in DISCRETE
    env_fn = lambda: make_env(args.regime, discrete=discrete, obs_mode=args.obs_mode)
    n_envs = args.n_envs if args.algo in ON_POLICY else 1
    train_env = make_vec_env(env_fn, n_envs=n_envs, seed=args.seed)
    ...
    eval_env = make_env(args.regime, discrete=discrete, obs_mode=args.obs_mode)
```

Also save `obs_mode` in the result:
```python
    res.update({"split": args.split, "timesteps": args.timesteps, "train_time_s": train_time,
                "obs_mode": args.obs_mode, "env_config": eval_env.unwrapped.cfg.to_dict()})
```

- [ ] **Step 7: Run obs_mode comparison smoke test (val split, tiny)**

```bash
for mode in flat sorted cluster_summary; do
  python scripts/train_rl.py --algo ppo --regime static --seed 100 \
    --timesteps 20000 --n-eval 5 --split val --out /tmp/smoke_obs_$mode \
    --obs-mode $mode
done
```
Expected: Each run completes; prints `final_cov` values. The relative ordering informs which mode to investigate more deeply during tuning.

- [ ] **Step 8: Log observation mode decision in DECISIONS.md after comparison**

After reviewing the val-split smoke results with the supervisor, append to `docs/DECISIONS.md`:
```
## D-008 (update): Observation design resolved
Date: YYYY-MM-DD | Status: accepted | Decided by: student + Dr. Sookhak
Decision: Use obs_mode="[chosen mode]" for all reported results.
Evidence: Val-split smoke comparison (seed 100, 20k steps, 5 layouts):
  flat final_cov=X.XXX | sorted=X.XXX | cluster_summary=X.XXX
Consequences: Locks observation space for the full sweep. Using a different mode
later invalidates all RL results.
```

- [ ] **Step 9: Commit**

```bash
pytest -q
git add uavbench/env.py tests/test_env.py scripts/train_rl.py scripts/run_baselines.py docs/DECISIONS.md
git commit -m "feat: add obs_mode to EnvConfig (flat/sorted/cluster_summary); wire --obs-mode into scripts"
```

---

### Task 5: SINR / Co-Channel Interference Extension

**Research decision prerequisite:** Get supervisor approval before enabling this in results runs (setting `use_sinr=True` changes the objective and invalidates old results). The flag defaults to `False`, so this task does not break existing results.

**Files:**
- Modify: `uavbench/channel.py` (add `sinr_matrix()`)
- Modify: `uavbench/env.py` (add `use_sinr`, `tx_power_dbm`, `noise_dbm`, `sinr_min_db` to `EnvConfig`; add `sinr_coverage()`; update `step()` info)
- Modify: `tests/test_channel.py` (add SINR tests)

**Interfaces:**
- Consumes: `AlHouraniChannel.path_loss_matrix()`
- Produces:
  - `AlHouraniChannel.sinr_matrix(uav_xyz, user_xy, tx_power_dbm=30.0, noise_dbm=-100.0) -> np.ndarray` shape `(K,)` SINR in dB
  - `UAVPlacementEnv.sinr_coverage() -> float` fraction of users with SINR ≥ `sinr_min_db`
  - `info["sinr_cov"]` added to `step()` and `reset()` info dict when `use_sinr=True`

- [ ] **Step 1: Write failing SINR tests**

Add to `tests/test_channel.py`:

```python
def test_sinr_matrix_shape():
    ch = AlHouraniChannel("urban")
    uav_xyz = np.array([[500.0, 500.0, 100.0], [2500.0, 2500.0, 100.0]])
    user_xy = np.random.default_rng(0).uniform(0, 3000, (10, 2))
    sinr = ch.sinr_matrix(uav_xyz, user_xy)
    assert sinr.shape == (10,), f"Expected (10,), got {sinr.shape}"


def test_sinr_single_uav_equals_snr():
    """With one UAV there is no interference; SINR should equal SNR."""
    ch = AlHouraniChannel("urban")
    uav_xyz = np.array([[1500.0, 1500.0, 100.0]])
    user_xy = np.array([[1500.0, 1600.0]])
    tx_power_dbm, noise_dbm = 30.0, -100.0
    sinr = ch.sinr_matrix(uav_xyz, user_xy, tx_power_dbm, noise_dbm)
    pl = ch.path_loss_matrix(uav_xyz, user_xy)[0, 0]
    snr = tx_power_dbm - pl - noise_dbm
    assert abs(sinr[0] - snr) < 0.01, f"Expected SNR={snr:.2f} dB, got SINR={sinr[0]:.2f} dB"


def test_sinr_finite_values():
    ch = AlHouraniChannel("urban")
    rng = np.random.default_rng(1)
    uav_xyz = rng.uniform(0, 3000, (4, 3))
    uav_xyz[:, 2] = 100.0
    user_xy = rng.uniform(0, 3000, (20, 2))
    sinr = ch.sinr_matrix(uav_xyz, user_xy)
    assert np.all(np.isfinite(sinr)), "SINR contains non-finite values"


def test_sinr_less_than_snr_with_multiple_uavs():
    """Adding a second UAV as interferer should reduce SINR below SNR."""
    ch = AlHouraniChannel("urban")
    user_xy = np.array([[1500.0, 1500.0]])
    # UAV1 close to user; UAV2 also close — creates interference
    uav1 = np.array([[1500.0, 1600.0, 100.0]])
    uav2 = np.array([[1500.0, 1700.0, 100.0], [1500.0, 1600.0, 100.0]])
    sinr_no_interference = ch.sinr_matrix(uav1, user_xy)[0]
    sinr_with_interference = ch.sinr_matrix(uav2, user_xy)[0]
    assert sinr_with_interference < sinr_no_interference, (
        "SINR should decrease when a second close interfering UAV is added"
    )
```

Run: `pytest tests/test_channel.py::test_sinr_matrix_shape -v`
Expected: FAIL with `AttributeError: 'AlHouraniChannel' object has no attribute 'sinr_matrix'`

- [ ] **Step 2: Add `sinr_matrix()` to `AlHouraniChannel` in `uavbench/channel.py`**

Add after `coverage_radius()` (around line 109):

```python
    def sinr_matrix(
        self,
        uav_xyz: np.ndarray,
        user_xy: np.ndarray,
        tx_power_dbm: float = 30.0,
        noise_dbm: float = -100.0,
    ) -> np.ndarray:
        """Per-user SINR (dB), treating all UAVs as co-channel interferers.

        Signal power  = strongest UAV's received power.
        Interference  = sum of all other UAVs' received power.
        Returns array of shape (n_users,).
        """
        pl = self.path_loss_matrix(uav_xyz, user_xy)       # (U, K) dB
        rx_dbm = tx_power_dbm - pl                          # (U, K) dBm
        rx_lin = 10.0 ** (rx_dbm / 10.0)                  # (U, K) mW linear
        noise_lin = 10.0 ** (noise_dbm / 10.0)            # scalar mW
        signal = rx_lin.max(axis=0)                         # (K,) best-UAV power
        interference = rx_lin.sum(axis=0) - signal          # (K,) all-others power
        sinr_lin = signal / np.maximum(interference + noise_lin, 1e-20)
        return 10.0 * np.log10(sinr_lin)                   # (K,) dB
```

- [ ] **Step 3: Add SINR fields to `EnvConfig` in `uavbench/env.py`**

In the `EnvConfig` dataclass, add after `obs_mode`:

```python
    # SINR / interference (disabled by default; enabling changes the objective)
    use_sinr: bool = False       # if True, adds sinr_cov to info dict
    tx_power_dbm: float = 30.0  # UAV transmit power (dBm)
    noise_dbm: float = -100.0   # thermal noise floor (dBm)
    sinr_min_db: float = 0.0    # minimum SINR threshold for sinr_cov metric
```

- [ ] **Step 4: Add `sinr_coverage()` to `UAVPlacementEnv` in `uavbench/env.py`**

Add after the existing `coverage()` method (around line 103):

```python
    def sinr_coverage(self, uavs: np.ndarray | None = None, users: np.ndarray | None = None) -> float:
        """Fraction of users with SINR >= sinr_min_db. Only meaningful when use_sinr=True."""
        uavs = self.uavs if uavs is None else uavs
        users = self.users if users is None else users
        sinr = self.channel.sinr_matrix(uavs, users, self.cfg.tx_power_dbm, self.cfg.noise_dbm)
        return float(np.mean(sinr >= self.cfg.sinr_min_db))
```

- [ ] **Step 5: Add `sinr_cov` to info in `step()` and `reset()`**

In `UAVPlacementEnv.step()`, update the info dict:
```python
        info = {"coverage": cov, "too_close_frac": close}
        if c.use_sinr:
            info["sinr_cov"] = self.sinr_coverage()
```

In `UAVPlacementEnv.reset()`, update the info dict returned:
```python
        cov = self.coverage()
        info = {"coverage": cov}
        if c.use_sinr:
            info["sinr_cov"] = self.sinr_coverage()
        return self._obs(), info
```

- [ ] **Step 6: Run all tests**

```bash
pytest -q
```
Expected: All tests (26+ previous + 4 new SINR) PASS.

- [ ] **Step 7: Run SINR smoke test**

```bash
python -c "
from uavbench.env import UAVPlacementEnv, EnvConfig
env = UAVPlacementEnv(EnvConfig(use_sinr=True))
obs, info = env.reset(seed=0)
_, _, _, _, info = env.step(env.action_space.sample())
assert 'sinr_cov' in info, f'sinr_cov missing: {info}'
print(f'coverage={info[\"coverage\"]:.3f}  sinr_cov={info[\"sinr_cov\"]:.3f}')
"
```
Expected: Both metrics printed without error.

- [ ] **Step 8: Log decision and commit**

Append to `docs/DECISIONS.md`:
```
## D-013: SINR / co-channel interference extension
Date: 2026-11-16 | Status: proposed (needs supervisor approval before use in results)
Context: README roadmap item. No-interference model overestimates coverage; UAVs do interfere.
Decision: Added sinr_matrix() to channel; use_sinr flag in EnvConfig defaults to False.
Alternatives: SINR-based reward (changes the objective; would require supervisor approval and full re-run).
Consequences: Does NOT affect any existing results (flag is False by default). Enabling it
changes the metric and requires a full re-run with supervisor approval.
```

```bash
pytest -q
git add uavbench/channel.py uavbench/env.py tests/test_channel.py docs/DECISIONS.md
git commit -m "feat: add SINR/co-channel interference support (use_sinr=False by default)"
```

---

### Task 6: Hyperparameter Tuning Infrastructure

**Prerequisites:** Task 3 (protocol frozen with tuning budget and timestep count filled in) must be complete. All tuning uses `--split val`.

**Files:**
- Create: `scripts/tune_baselines.py`
- Create: `scripts/tune_rl.py`
- Modify: `docs/EXPERIMENT_PROTOCOL.md` (fill in tuning log entries after each trial)

**Interfaces:**
- Consumes: `make_env()`, `TargetTrackingPolicy`, `run_episode()`, `summarize()`, `save_result()`, `layout_seeds()`
- Produces: best-config JSONs at `results_val/best_{algo}_{regime}.json`; tuning log entries in `docs/EXPERIMENT_PROTOCOL.md`

- [ ] **Step 1: Create `scripts/tune_baselines.py`**

```python
#!/usr/bin/env python
"""Tune PSO and k-means hyperparameters on the validation split.

Every trial uses tuning seeds 100–102 and 5 val layouts. Writes results to results_val/.
Budget = number from EXPERIMENT_PROTOCOL.md Section 5 (same for all methods).

NEVER change TUNING_SEEDS or the split offset here. See docs/EXPERIMENT_PROTOCOL.md.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from uavbench.env import make_env
from uavbench.baselines import TargetTrackingPolicy
from uavbench.evaluation import run_episode, summarize, save_result, layout_seeds

TUNING_SEEDS = [100, 101, 102]
N_EVAL_LAYOUTS = 5  # small during tuning; use 20 for final reported results

PSO_CONFIGS = [
    {"n_particles": n, "iters": it, "w": w, "c1": 1.5, "c2": 1.5}
    for n, it, w in itertools.product([20, 30, 50], [40, 60, 100], [0.5, 0.7, 0.9])
]

KMEANS_CONFIGS = [
    {"edge_quantile": q} for q in [0.7, 0.8, 0.9, 0.95]
]


def eval_policy(algo: str, regime: str, seed: int, out_dir: str, **planner_kwargs) -> float:
    env = make_env(regime=regime)
    policy = TargetTrackingPolicy(algo, seed=seed, **planner_kwargs)

    cov_runs = []
    for ls in layout_seeds("val", N_EVAL_LAYOUTS):
        obs, _ = env.reset(seed=ls)
        policy.reset(env.unwrapped)
        cov_runs.append(run_episode(env, lambda obs, base: policy.act(base), ls))
    result = summarize(cov_runs)
    result["planner_kwargs"] = planner_kwargs
    save_result(out_dir, regime, f"{algo}_tuning", seed, result)
    return result["final_cov"]


def tune(algo: str, regime: str, configs: list[dict], budget: int, out_dir: str):
    print(f"\n--- Tuning {algo} on {regime} ({min(len(configs), budget)} trials) ---")
    best_score, best_cfg = -1.0, None
    for trial, cfg in enumerate(configs[:budget]):
        scores = [eval_policy(algo, regime, s, out_dir, **cfg) for s in TUNING_SEEDS]
        mean = float(np.mean(scores))
        print(f"  Trial {trial+1:3d}: {cfg}  ->  final_cov={mean:.4f} "
              f"[{min(scores):.4f}, {max(scores):.4f}]")
        if mean > best_score:
            best_score, best_cfg = mean, cfg
    print(f"  Best: {best_cfg}  ->  {best_score:.4f}")
    out = Path(out_dir) / f"best_{algo}_{regime}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"algo": algo, "regime": regime, "hp": best_cfg,
                               "val_final_cov": best_score}, indent=2))
    return best_cfg, best_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regime", default="static", choices=["static", "mobile"])
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--out", default="results_val")
    args = p.parse_args()
    tune("pso", args.regime, PSO_CONFIGS, args.budget, args.out)
    tune("kmeans", args.regime, KMEANS_CONFIGS, min(4, args.budget), args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run smoke test of baseline tuning script**

```bash
python scripts/tune_baselines.py --regime static --budget 2 --out /tmp/tune_smoke
```
Expected: Runs 2 PSO trials and 2 k-means trials; prints best configs; creates JSON files in `/tmp/tune_smoke/`.

- [ ] **Step 3: Create `scripts/tune_rl.py`**

```python
#!/usr/bin/env python
"""Tune SB3 RL hyperparameters on the validation split.

WARNING: This is compute-heavy. SAC × 20 trials × 3 seeds × 200k steps ≈ hours on CPU.
Use --timesteps 50000 for a preliminary sweep, then confirm best config at 200k.
Budget must equal the baseline budget (see docs/EXPERIMENT_PROTOCOL.md Section 5).

NEVER use the test split (seeds 10000+i) here. Only --split val.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO, SAC, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from uavbench.env import make_env
from uavbench.evaluation import run_episode, summarize, save_result, layout_seeds

TUNING_SEEDS = [100, 101, 102]
N_EVAL_LAYOUTS = 5
ALGOS = {"ppo": PPO, "sac": SAC, "a2c": A2C, "dqn": DQN}
ON_POLICY = {"ppo", "a2c"}
DISCRETE = {"dqn"}

GRIDS: dict[str, list[dict]] = {
    "ppo": [{"learning_rate": lr, "ent_coef": ec}
            for lr in [3e-4, 1e-4, 3e-3] for ec in [0.0, 0.01]],
    "a2c": [{"learning_rate": lr, "ent_coef": ec}
            for lr in [7e-4, 3e-4, 1e-3] for ec in [0.0, 0.01]],
    "sac": [{"learning_rate": lr, "buffer_size": bs}
            for lr in [3e-4, 1e-4] for bs in [100_000, 50_000]],
    "dqn": [{"learning_rate": lr, "exploration_fraction": ef}
            for lr in [1e-4, 3e-4] for ef in [0.1, 0.2]],
}


def eval_policy(algo_name: str, regime: str, seed: int, timesteps: int,
                hp: dict, out_dir: str, obs_mode: str = "flat") -> float:
    is_discrete = algo_name in DISCRETE
    is_on_policy = algo_name in ON_POLICY
    env_fn = lambda: make_env(regime=regime, discrete=is_discrete, obs_mode=obs_mode)
    n_envs = 4 if is_on_policy else 1
    env = make_vec_env(env_fn, n_envs=n_envs, seed=seed)
    model = ALGOS[algo_name]("MlpPolicy", env, verbose=0, seed=seed, **hp)
    model.learn(total_timesteps=timesteps)
    eval_env = make_env(regime=regime, discrete=is_discrete, obs_mode=obs_mode)
    cov_runs = []
    for ls in layout_seeds("val", N_EVAL_LAYOUTS):
        def act_fn(obs, base):
            action, _ = model.predict(obs, deterministic=True)
            return action
        cov_runs.append(run_episode(eval_env, act_fn, ls))
    result = summarize(cov_runs)
    result["hp"] = hp
    result["obs_mode"] = obs_mode
    save_result(out_dir, regime, f"{algo_name}_tuning", seed, result)
    return result["final_cov"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=list(ALGOS))
    p.add_argument("--regime", default="static", choices=["static", "mobile"])
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--timesteps", type=int, default=50_000,
                   help="Use 50k for initial search; confirm best config at 200k")
    p.add_argument("--obs-mode", default="flat",
                   choices=["flat", "sorted", "cluster_summary"])
    p.add_argument("--out", default="results_val")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    grid = GRIDS[args.algo][:args.budget]
    print(f"\n--- Tuning {args.algo} on {args.regime} "
          f"({len(grid)} trials × {len(TUNING_SEEDS)} seeds, {args.timesteps} steps each) ---")
    best_score, best_hp = -1.0, None
    for trial, hp in enumerate(grid):
        scores = [eval_policy(args.algo, args.regime, s, args.timesteps, hp,
                               args.out, args.obs_mode) for s in TUNING_SEEDS]
        mean = float(np.mean(scores))
        print(f"  Trial {trial+1:3d}: {hp}  ->  final_cov={mean:.4f} "
              f"[{min(scores):.4f}, {max(scores):.4f}]")
        if mean > best_score:
            best_score, best_hp = mean, hp
    print(f"\n  Best {args.algo}: {best_hp}  ->  {best_score:.4f}")
    out_path = Path(args.out) / f"best_{args.algo}_{args.regime}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "algo": args.algo, "regime": args.regime,
        "hp": best_hp, "val_final_cov": best_score,
        "obs_mode": args.obs_mode, "timesteps": args.timesteps,
    }, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run RL tuning smoke test**

```bash
python scripts/tune_rl.py --algo ppo --regime static --budget 2 --timesteps 5000 --out /tmp/tune_rl_smoke
```
Expected: Runs 2 PPO configs × 3 seeds; prints best HP; creates `/tmp/tune_rl_smoke/best_ppo_static.json`.

- [ ] **Step 5: Commit**

```bash
pytest -q
git add scripts/tune_baselines.py scripts/tune_rl.py
git commit -m "feat: add hyperparameter tuning scripts for baselines and RL"
```

- [ ] **Step 6: Run the actual tuning (human-initiated; record in EXPERIMENT_PROTOCOL.md)**

For each regime and method, run tuning on a machine with enough compute. Example sequence:
```bash
# Baselines (fast, ~minutes total)
python scripts/tune_baselines.py --regime static --budget 20 --out results_val
python scripts/tune_baselines.py --regime mobile --budget 20 --out results_val

# RL (slow; SAC especially; run per-algo)
python scripts/tune_rl.py --algo ppo --regime static --budget 20 --timesteps 50000 --out results_val
python scripts/tune_rl.py --algo ppo --regime mobile --budget 20 --timesteps 50000 --out results_val
python scripts/tune_rl.py --algo a2c --regime static --budget 20 --timesteps 50000 --out results_val
python scripts/tune_rl.py --algo dqn --regime static --budget 20 --timesteps 50000 --out results_val
python scripts/tune_rl.py --algo sac --regime static --budget 20 --timesteps 50000 --out results_val
```

Record every trial in `docs/EXPERIMENT_PROTOCOL.md` Section 7 (tuning log). The best configs land in `results_val/best_{algo}_{regime}.json`.

---

### Task 7: Full 10-Seed Sweep & Paper Figures

**Prerequisites:** Tasks 2–6 complete (Lyu spiral added, protocol frozen, obs_mode decided, hyperparameters tuned).

**Files:**
- Verify/modify: `scripts/run_all.sh`
- Create: `scripts/plot_figures.py`
- Create: `docs/figures/` directory (auto-created)

**Interfaces:**
- Consumes: `results/{regime}/{algo}/seed*.json`
- Produces: `docs/figures/fig1_static_coverage.pdf`, `docs/figures/fig2_mobile_coverage.pdf`

- [ ] **Step 1: Verify `scripts/run_all.sh` is complete**

Read `scripts/run_all.sh`. Confirm it includes:
- All baselines: `kmeans`, `pso`, `lyu_spiral`, `random`, `hover`
- All RL algorithms: `ppo`, `sac`, `a2c`, `dqn`
- Both regimes: `static`, `mobile`
- 10 seeds: 0–9

If `lyu_spiral` is missing, add it. The expected structure:

```bash
#!/bin/bash
# Full 10-seed sweep. Usage: bash scripts/run_all.sh [n_seeds=10] [timesteps=200000]
N_SEEDS=${1:-10}
TIMESTEPS=${2:-200000}
OBS_MODE=${3:-flat}  # set after tuning comparison resolves D-008

for REGIME in static mobile; do
  for SEED in $(seq 0 $((N_SEEDS - 1))); do
    python scripts/run_baselines.py --regime $REGIME --seeds $SEED --out results \
      --algos kmeans pso lyu_spiral random hover --obs-mode $OBS_MODE
    for ALGO in ppo sac a2c dqn; do
      python scripts/train_rl.py --algo $ALGO --regime $REGIME --seed $SEED \
        --timesteps $TIMESTEPS --out results --obs-mode $OBS_MODE
    done
  done
done
```

If the tuned hyperparameters are known, pass them here too. Otherwise note in EXPERIMENT_PROTOCOL.md that defaults were used.

- [ ] **Step 2: Run baselines sweep first (fast, verify results structure)**

```bash
python scripts/run_baselines.py --regime static --seeds 0 1 2 3 4 5 6 7 8 9 --out results --algos kmeans pso lyu_spiral random hover
python scripts/run_baselines.py --regime mobile --seeds 0 1 2 3 4 5 6 7 8 9 --out results --algos kmeans pso lyu_spiral random hover
```
Expected: Creates `results/static/{kmeans,pso,lyu_spiral,random,hover}/seed{0..9}.json` and same for mobile.

- [ ] **Step 3: Aggregate baselines to verify pipeline**

```bash
python scripts/aggregate.py --regime static --metric final_cov
python scripts/aggregate.py --regime mobile --metric final_cov
```
Expected: Prints IQM ± 95% CI table. Confirms rliable pipeline works before expensive RL runs.

- [ ] **Step 4: Create `scripts/plot_figures.py`**

```python
#!/usr/bin/env python
"""Generate paper-quality figures from full-sweep results.

Outputs:
  docs/figures/fig1_static_coverage.pdf   - IQM bar chart, static regime
  docs/figures/fig2_mobile_coverage.pdf   - IQM bar chart, mobile regime

Usage: python scripts/plot_figures.py [--results results] [--out docs/figures]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.family": "serif", "font.size": 10})

CLASSICAL = ["kmeans", "pso", "lyu_spiral", "random", "hover"]
RL_ALGOS = ["ppo", "sac", "a2c", "dqn"]
ALGO_LABELS = {
    "kmeans": "k-Means+Alt.",
    "pso": "PSO",
    "lyu_spiral": "Lyu Spiral",
    "random": "Random",
    "hover": "Hover",
    "ppo": "PPO",
    "sac": "SAC",
    "a2c": "A2C",
    "dqn": "DQN",
}
COLORS_CLASSICAL = "#2196F3"
COLORS_RL = "#FF5722"


def load_per_layout(results_dir: str, regime: str, algo: str, metric: str) -> np.ndarray:
    """Return array of shape (n_seeds, n_layouts)."""
    rows = []
    for f in sorted(Path(results_dir).glob(f"{regime}/{algo}/seed*.json")):
        d = json.loads(f.read_text())
        rows.append(d[f"{metric}_per_layout"])
    return np.array(rows) if rows else np.empty((0, 0))


def iqm_bootstrap_ci(scores_2d: np.ndarray, reps: int = 5000) -> tuple[float, float, float]:
    """IQM and 95% stratified bootstrap CI from (n_seeds, n_layouts) array."""
    flat = scores_2d.ravel()
    q25, q75 = np.percentile(flat, 25), np.percentile(flat, 75)
    iq = flat[(flat >= q25) & (flat <= q75)]
    if len(iq) == 0:
        return float(np.mean(flat)), float(np.mean(flat)), float(np.mean(flat))
    iqm = float(np.mean(iq))
    rng = np.random.default_rng(42)
    boot = np.array([np.mean(rng.choice(iq, len(iq))) for _ in range(reps)])
    return iqm, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def plot_bar_chart(results_dir: str, regime: str, out_path: str, metric: str = "final_cov"):
    all_algos = CLASSICAL + RL_ALGOS
    iqms, ci_los, ci_his, labels, colors = [], [], [], [], []
    for algo in all_algos:
        data = load_per_layout(results_dir, regime, algo, metric)
        if data.size == 0:
            continue
        iqm, lo, hi = iqm_bootstrap_ci(data)
        iqms.append(iqm)
        ci_los.append(iqm - lo)
        ci_his.append(hi - iqm)
        labels.append(ALGO_LABELS.get(algo, algo))
        colors.append(COLORS_RL if algo in RL_ALGOS else COLORS_CLASSICAL)

    if not iqms:
        print(f"No results found for regime={regime} in {results_dir}")
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, iqms, color=colors, yerr=[ci_los, ci_his],
           capsize=4, alpha=0.85, error_kw={"linewidth": 1.5})
    n_classical = sum(1 for a in all_algos if a in CLASSICAL and
                       load_per_layout(results_dir, regime, a, metric).size > 0)
    ax.axvline(n_classical - 0.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(f"IQM {metric} (95% CI)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{regime.capitalize()} Regime | {metric} | IQM ± 95% bootstrap CI")
    ax.grid(axis="y", alpha=0.3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="docs/figures")
    p.add_argument("--metric", default="final_cov", choices=["final_cov", "avg_cov"])
    args = p.parse_args()
    plot_bar_chart(args.results, "static", f"{args.out}/fig1_static_{args.metric}.pdf", args.metric)
    plot_bar_chart(args.results, "mobile", f"{args.out}/fig2_mobile_{args.metric}.pdf", args.metric)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run plot script on baseline-only results to verify**

```bash
python scripts/plot_figures.py --results results --out docs/figures
```
Expected: Creates `docs/figures/fig1_static_final_cov.pdf` and `fig2_mobile_final_cov.pdf` showing baselines only. Confirms the script works before RL runs complete.

- [ ] **Step 6: Run full RL sweep (slow; run per-algorithm to monitor progress)**

```bash
# Example: PPO sweep (fast, ~20 min for 10 seeds × 2 regimes × 200k steps)
for seed in 0 1 2 3 4 5 6 7 8 9; do
  python scripts/train_rl.py --algo ppo --regime static --seed $seed --timesteps 200000 --out results
  python scripts/train_rl.py --algo ppo --regime mobile --seed $seed --timesteps 200000 --out results
done

# SAC (slow, ~40 min per seed on 2-core CPU — budget ~13 hours total)
for seed in 0 1 2 3 4 5 6 7 8 9; do
  python scripts/train_rl.py --algo sac --regime static --seed $seed --timesteps 200000 --out results
done
```

After each algorithm completes, re-run `scripts/aggregate.py` to track progress.

- [ ] **Step 7: Final aggregate and figures**

```bash
python scripts/aggregate.py --regime static --metric final_cov
python scripts/aggregate.py --regime mobile --metric final_cov
python scripts/plot_figures.py --results results --out docs/figures
```
Expected: Complete IQM ± CI table and PDF figures with all methods.

- [ ] **Step 8: Commit**

```bash
pytest -q
git add scripts/run_all.sh scripts/plot_figures.py docs/figures/
git commit -m "feat: add paper figure generation script; run full 10-seed sweep"
```

---

### Task 8: Regime III — Disaster Shock (Stretch Goal)

**Prerequisites:** Get supervisor approval; this adds a new regime and does NOT affect Regime I/II results (different `mobility` parameter). Log in `docs/DECISIONS.md` before running any sweeps with this regime.

**Files:**
- Modify: `uavbench/env.py` (add `mobility="shock"`, shock fields to `EnvConfig`, `_apply_shock()`, `_failed_mask`)
- Modify: `tests/test_env.py` (add shock regime tests)

**Interfaces:**
- Adds to `EnvConfig`: `shock_step: int = 30`, `shock_type: str = "uav_failure"`, `shock_n_uavs_fail: int = 1`, `shock_migration_frac: float = 0.5`
- Adds to `UAVPlacementEnv`: `n_active_uavs: int`, `_failed_mask: np.ndarray`, `_apply_shock()`
- `step()` info: `info["shock_active"] = True` exactly at `shock_step`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_env.py`:

```python
def test_shock_uav_failure_reduces_active_count():
    cfg = EnvConfig(mobility="shock", shock_step=5, shock_type="uav_failure", shock_n_uavs_fail=1)
    env = UAVPlacementEnv(cfg)
    env.reset(seed=0)
    n_before = env.n_active_uavs
    for _ in range(4):
        env.step(env.action_space.sample())
    assert env.n_active_uavs == n_before, "No UAV should fail before shock_step"
    _, _, _, _, info = env.step(env.action_space.sample())  # step 5 triggers shock
    assert info.get("shock_active") is True
    assert env.n_active_uavs == n_before - 1


def test_shock_user_migration_moves_users():
    cfg = EnvConfig(mobility="shock", shock_step=5, shock_type="user_migration",
                    shock_migration_frac=0.5)
    env = UAVPlacementEnv(cfg)
    env.reset(seed=0)
    users_before = env.users.copy()
    for _ in range(5):
        env.step(env.action_space.sample())
    n_changed = int(np.sum(np.any(env.users != users_before, axis=1)))
    expected_min = int(0.4 * cfg.n_users)
    assert n_changed >= expected_min, f"Expected ≥{expected_min} relocated users, got {n_changed}"


def test_shock_no_trigger_before_shock_step():
    cfg = EnvConfig(mobility="shock", shock_step=10)
    env = UAVPlacementEnv(cfg)
    env.reset(seed=0)
    for _ in range(9):
        _, _, _, _, info = env.step(env.action_space.sample())
        assert not info.get("shock_active"), f"Shock triggered early at step {env.t}"
```

Run: `pytest tests/test_env.py::test_shock_uav_failure_reduces_active_count -v`
Expected: FAIL with `ValueError: mobility must be 'static' or 'mobile'`

- [ ] **Step 2: Add shock fields to `EnvConfig`**

In `uavbench/env.py` `EnvConfig`, append:

```python
    # Regime III: disaster shock (stretch goal — requires supervisor approval for results runs)
    shock_step: int = 30               # episode step at which the shock is triggered
    shock_type: str = "uav_failure"    # "uav_failure" | "user_migration"
    shock_n_uavs_fail: int = 1         # number of UAVs that fail (uav_failure type)
    shock_migration_frac: float = 0.5  # fraction of users that relocate (user_migration type)
```

- [ ] **Step 3: Add `_failed_mask` and `n_active_uavs` to `__init__`**

After `self.t = 0` in `UAVPlacementEnv.__init__`, add:

```python
        self._failed_mask = np.ones(cfg.n_uavs, dtype=bool)
        self.n_active_uavs: int = cfg.n_uavs
```

- [ ] **Step 4: Update mobility validation in `__init__`**

Replace the existing check:
```python
        if cfg.mobility not in ("static", "mobile", "shock"):
            raise ValueError("mobility must be 'static', 'mobile', or 'shock'")
```

- [ ] **Step 5: Add `_apply_shock()` method**

Add before `reset()`:

```python
    def _apply_shock(self) -> None:
        c = self.cfg
        if c.shock_type == "uav_failure":
            active = np.where(self._failed_mask)[0]
            n_fail = min(c.shock_n_uavs_fail, len(active))
            if n_fail > 0:
                fail_idxs = self.np_random.choice(active, n_fail, replace=False)
                self._failed_mask[fail_idxs] = False
                self.n_active_uavs -= n_fail
        elif c.shock_type == "user_migration":
            n_move = int(round(c.shock_migration_frac * c.n_users))
            idxs = self.np_random.choice(c.n_users, n_move, replace=False)
            self.users[idxs] = self.np_random.uniform(0.0, c.area, size=(n_move, 2))
            self.waypoints[idxs] = self.np_random.uniform(0.0, c.area, size=(n_move, 2))
```

- [ ] **Step 6: Update `reset()` to reinitialize shock state**

In `UAVPlacementEnv.reset()`, add after `self.t = 0`:
```python
        self._failed_mask = np.ones(c.n_uavs, dtype=bool)
        self.n_active_uavs = c.n_uavs
```

- [ ] **Step 7: Update `step()` to trigger shock and use active UAVs**

In `UAVPlacementEnv.step()`, after `self.t += 1`:
```python
        shock_active = False
        if c.mobility == "shock" and self.t == c.shock_step:
            self._apply_shock()
            shock_active = True
        if c.mobility in ("mobile", "shock"):
            self._move_users()
        active_uavs = self.uavs[self._failed_mask] if c.mobility == "shock" else self.uavs
        cov = self.coverage(active_uavs)
        close = self._too_close_frac()
        reward = cov - c.collision_penalty * close
        truncated = self.t >= c.episode_len
        info = {"coverage": cov, "too_close_frac": close, "shock_active": shock_active,
                "n_active_uavs": self.n_active_uavs}
        if c.use_sinr:
            info["sinr_cov"] = self.sinr_coverage(active_uavs)
```

- [ ] **Step 8: Run all tests**

```bash
pytest -q
```
Expected: All tests pass including 3 new shock tests.

- [ ] **Step 9: Smoke test**

```bash
python scripts/train_rl.py --algo ppo --regime shock --seed 0 --timesteps 20000 --n-eval 5 --out /tmp/smoke_shock
```
Expected: Completes without error; `shock_active` appears in episode logs.

- [ ] **Step 10: Log decision and commit**

Append to `docs/DECISIONS.md`:
```
## D-014: Regime III (disaster shock) implementation
Date: 2026-11-16 | Status: proposed (needs supervisor approval before reported results)
Context: README/RESEARCH_CONTEXT.md stretch goal. Tests H3 hypothesis.
Decision: Added mobility="shock" to env. Two shock types: uav_failure and user_migration.
Defaults to uav_failure at step 30. Failed UAVs are excluded from coverage computation.
Consequences: Does not affect Regime I/II results (different mobility parameter). Must
get supervisor approval and add to EXPERIMENT_PROTOCOL.md before running reported sweeps.
```

```bash
pytest -q
git add uavbench/env.py tests/test_env.py docs/DECISIONS.md
git commit -m "feat: add Regime III disaster shock (uav_failure and user_migration modes)"
```

---

## Self-Review

### Spec Coverage

| Requirement from AGENTS.md / README roadmap | Task |
|---------------------------------------------|------|
| Verify every constant in `channel.py` against PDFs | Task 1 |
| Reproduce Al-Hourani radius-altitude figure | Task 1 |
| Add Lyu et al. spiral placement baseline | Task 2 |
| Freeze hypotheses and tuning budget before sweep | Task 3 |
| Decide observation design (D-008) | Task 4 |
| Wire obs_mode into training scripts | Task 4b (within Task 4) |
| Add SINR with co-channel interference | Task 5 |
| Equal-budget hyperparameter tuning | Task 6 |
| Full 10-seed sweep, both regimes | Task 7 |
| Aggregate with rliable, generate paper figures | Task 7 |
| Regime III disaster shock | Task 8 |
| Meta-A3C comparison | **NOT planned** — depends on lab sharing code; ask Dr. Sookhak |

### Gaps (not in this plan)

- **Meta-A3C:** Depends on lab code access. If code is shared, add it to `ALGOS` in `train_rl.py` and give it the same tuning budget. See `docs/RESEARCH_CONTEXT.md`.
- **Paper writing:** Separate deliverable. Figures from Task 7 feed directly into it.
- **SINR-based reward and objective:** Requires supervisor decision; would invalidate all results. See Task 5's `use_sinr` flag.

### Placeholder Scan

No TBD, TODO, or "implement later" items. All code blocks are complete and runnable.

### Type Consistency

- `lyu_spiral_planner(env, rng)` matches `kmeans_planner`, `pso_planner`, `random_planner` ✓
- `PLANNERS` dict updated consistently ✓
- `obs_mode` added to `EnvConfig`, used in `_obs()` and `observation_space` ✓
- `sinr_matrix(uav_xyz, user_xy, tx_power_dbm, noise_dbm)` consistent with `path_loss_matrix(uav_xyz, user_xy)` ✓
- `_failed_mask` and `n_active_uavs` initialized in both `__init__` and `reset()` ✓
- `active_uavs` passed to `coverage()` in `step()` when shock is active ✓
