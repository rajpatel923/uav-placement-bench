# AGENTS.md: uav-placement-bench

Instructions for AI coding agents working in this repository. Read this whole file
before making changes. Humans: see README.md.

## What this project is

An undergraduate Directed Individual Study (DIS) research project at Texas A&M
University–Corpus Christi, supervised by Dr. Mehdi Sookhak. It answers one question:

> **Where is the complexity boundary at which deep RL starts to beat classical
> optimization for multi-UAV aerial base station placement?** Evaluation uses the
> Al-Hourani air-to-ground channel and a statistically rigorous protocol
> (multiple seeds, IQM, bootstrap CIs).

The deliverable is a **conference paper**, so scientific validity matters more than
code cleverness. A fast change that compromises the comparison does damage.

## Read before you work

| Before you... | Read |
|---|---|
| Start any session | `docs/STATUS.md` (current state, next steps, open questions) |
| Touch research framing, claims, or paper text | `docs/RESEARCH_CONTEXT.md` |
| Run, tune, or change experiments or metrics | `docs/EXPERIMENT_PROTOCOL.md` |
| Touch `uavbench/channel.py` | `docs/CHANNEL_MODEL.md` |
| Make any design change | `docs/DECISIONS.md` (and add an entry) |
| Cite anything | `docs/REFERENCES.md` |

## Commands

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q                                                    # must pass before you finish

# smoke test (fast): use these flags while developing
python scripts/run_baselines.py --regime static --seeds 0 1 --n-eval 5 --out /tmp/smoke
python scripts/train_rl.py --algo ppo --regime static --seed 0 --timesteps 20000 --n-eval 5 --out /tmp/smoke

# tuning: ALWAYS on the validation split (writes to results_val/)
python scripts/train_rl.py --algo ppo --regime static --seed 100 --split val

# reported results: test split (writes to results/), then aggregate
bash scripts/run_all.sh 10 200000
python scripts/aggregate.py --regime static --metric final_cov
```

## Layout

```
uavbench/channel.py      Al-Hourani A2G model, optimal elevation solver (Eq. 13)
uavbench/env.py          Gymnasium env (Regime I static, II mobile), DQN discrete wrapper, make_env()
uavbench/baselines.py    k-means + Al-Hourani altitude, PSO, random, hover; TargetTrackingPolicy
uavbench/evaluation.py   Split seeds (test/val), episode runner, metrics, result saving
scripts/                 run_baselines.py, train_rl.py, aggregate.py (rliable), run_all.sh
tests/                   pytest suite (channel physics, env API, determinism, sanity)
docs/                    Project context for humans and agents (see table above)
results/  results_val/   Generated outputs (gitignored). Expensive to regenerate.
```

## Hard rules (never break these without explicit human approval)

1. **Never tune on the test split.** Hyperparameter, reward and observation tuning
   use `--split val` only. `results/` is for final reported runs.
2. **Equal tuning budget.** Every method (RL algorithms *and* PSO settings) gets the
   same number of tuning trials. Log every trial in `docs/EXPERIMENT_PROTOCOL.md`.
3. **Same constraints for everyone.** Baselines fly under the same `v_max`/`vz_max`
   limits as RL agents through `TargetTrackingPolicy`. Never let a baseline teleport.
4. **Never invent numbers.** Every number in docs, README, or paper text must come
   from a file in `results/` produced by the scripts. No estimates presented as results.
   Smoke-test numbers must be labeled as such.
5. **Never invent citations.** Cite only entries marked verified in `docs/REFERENCES.md`.
   If you add a reference, include DOI/URL and mark it `UNVERIFIED` until a human checks it.
6. **Do not change physics constants** in `channel.py` without a source (paper, table,
   equation number). Never loosen test tolerances in `tests/test_channel.py` to make
   tests pass; the tests encode published values.
7. **Do not change these silently:** `EVAL_SEED_OFFSET`, `VAL_SEED_OFFSET`,
   `FINAL_WINDOW`, `EnvConfig` defaults, the reward, or the observation. These are
   research decisions. Propose the change, get approval, and log it in `docs/DECISIONS.md`.
   Changing any of them invalidates all existing results.
8. **Never delete or overwrite `results/`.** SAC runs take hours. Write experiments to
   a new `--out` directory.
9. **One seed is not a result.** Never write a conclusion from fewer than 10 seeds per
   configuration. Report IQM with 95% CIs via `scripts/aggregate.py`.
10. **No predetermined conclusions.** Do not design or describe experiments to favor
    any method, including the supervising lab's Meta-A3C. Report outcomes either way.

## Coding conventions

- Python 3.10+, NumPy vectorized (no Python loops over users/UAVs in hot paths).
- Randomness only through `self.np_random` (env) or a passed `np.random.Generator`.
  No global `np.random.*` calls. Everything must be reproducible from a seed.
- Type hints and docstrings. Physics code cites the paper and equation number.
- New modules need tests. Bug fixes need a regression test.
- Keep dependencies minimal; pin anything that breaks (see `requirements.txt` note
  on `pandas<3` / `arch<8`).

## How to extend

- **New baseline:** write `my_planner(env, rng) -> (n_uavs, 3) targets` in
  `baselines.py`, register in `PLANNERS`, add a test that it beats `random`, and add it
  to `run_baselines.py --algos`.
- **New RL algorithm:** add to `ALGOS` in `scripts/train_rl.py`; add it to `DISCRETE`
  or `ON_POLICY` if needed. It must use `make_env()` and the shared evaluation.
- **New regime or env feature** (e.g. interference, disaster shock): add a config field
  with a default that preserves current behavior, tests for the new behavior, and a
  `docs/DECISIONS.md` entry.

## Known gotchas

- SB3 `DQN` supports only `Discrete`; we use a joint space of size `7**n_uavs`
  (2401 for 4 UAVs). This does not scale past ~5 UAVs.
- SAC is ~20x slower than PPO per step on CPU. Budget for it.
- A single-seed entry in `aggregate.py` shows a zero-width CI. That is expected (there's
  nothing to bootstrap), not precision.
- `rliable` breaks on import with `pandas>=3` (via `arch`). Keep the pins.
- The flat, unordered user list in the observation is a known weakness (see
  `docs/DECISIONS.md` D-008). Do not interpret weak RL results without addressing it.

## Definition of done (for any change)

1. `pytest -q` passes.
2. A smoke run of any script you touched completes.
3. `docs/DECISIONS.md` updated if behavior or design changed.
4. `docs/STATUS.md` updated with what you did and what's next.
5. No new numbers or citations that violate rules 4–5.

## Ask the human before

- Changing reward, observation, metrics, splits, or `EnvConfig` defaults.
- Anything that makes claims about Dr. Sookhak's lab papers.
- Long sweeps (> 1 hour of compute).
- Adding a dependency.
