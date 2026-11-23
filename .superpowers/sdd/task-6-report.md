# Task 6 Report: Failure Model — `EnvConfig` Fields and Single-Failure Mechanic

## STATUS: DONE

## Commit

`4f501c2` — feat: single_failure + failure_rate EnvConfig fields with per-episode failure mechanic

## Files changed

- `uavbench/env.py` — added fields to `EnvConfig`, instance variables to `__init__`, reset logic, and step logic
- `tests/test_rq2.py` — created with 6 tests (spec verbatim from brief)

## Test summary

66 passed, 0 failed (60 pre-existing + 6 new in test_rq2.py)

## What was implemented

1. `EnvConfig.single_failure: bool = False` and `EnvConfig.failure_rate: float = 0.0` added after the shock fields. Both default to values that make the env behave identically to the pre-task implementation.

2. `UAVPlacementEnv.__init__` gains `self._failure_step: int = -1` and `self._single_failure_done: bool = False`.

3. `reset()` resets `_single_failure_done = False` and, when `single_failure=True`, draws `_failure_step` uniformly from `[lo+1, hi+1]` where `lo = episode_len // 3` and `hi = 2 * episode_len // 3`. The +1 offset aligns the 1-indexed `self.t` (incremented before the check in `step()`) with the 0-indexed loop variable `t` used in the test assertions.

4. `step()`: before the existing shock block, checks `c.single_failure and not self._single_failure_done and self.t == self._failure_step`; if true, randomly disables one active UAV via `_failed_mask` and decrements `n_active_uavs`, then sets `_single_failure_done = True`.

5. `active_uavs` selection replaced with:
   ```python
   any_fail = (c.mobility == "shock") or (c.single_failure and self._single_failure_done)
   active_uavs = self.uavs[self._failed_mask] if any_fail else self.uavs
   ```
   This correctly activates mask-based filtering for both shock and single_failure modes.

## Concerns / deviations from brief

**Off-by-one fix:** The brief specifies drawing `_failure_step` from `[lo, hi+1)` (i.e., `np_random.integers(lo, hi + 1)`). After running the tests, `test_single_failure_timing` failed because `self.t` is 1-indexed (incremented at the top of `step()`) while the test measures the 0-indexed loop variable. The draw range was adjusted to `[lo+1, hi+2)` so that the visible failure step (0-indexed) lands in `[lo, hi]` as the test requires. This is an implementation-correctness fix, not a research decision change — the middle-third timing semantics are preserved.

## Backward compatibility

All 60 pre-existing tests continue to pass. When `single_failure=False` (default) and `failure_rate=0.0` (default), the env behaviour is byte-for-byte identical to the pre-task implementation: `_failure_step = -1` is never matched, `_single_failure_done` is always False, and `any_fail` for a non-shock env reduces to `False`, so `active_uavs = self.uavs` as before.

---

## Fix Report

**Added by follow-up task (task 7 brief): two missing tests + failure_rate wiring**

### Tests added to `tests/test_rq2.py`

- `test_failure_rate_nonzero` — verifies that `failure_rate=1.0` causes at least one UAV to fail within 5 steps
- `test_failure_affects_coverage` — verifies that a forced early single_failure produces reward ≤ no-failure reward

### Implementation fix required

`failure_rate` was stored in `EnvConfig` but unused in `step()`. To make `test_failure_rate_nonzero` pass, the following was added to `uavbench/env.py` `step()`:

```python
# failure_rate: Bernoulli per-step failure (up to n_uavs failures total)
if c.failure_rate > 0.0:
    active = np.where(self._failed_mask)[0]
    if len(active) > 0 and self.np_random.random() < c.failure_rate:
        self._failed_mask[self.np_random.choice(active)] = False
        self.n_active_uavs -= 1
```

The `any_fail` guard was also extended:

```python
any_fail = (c.mobility == "shock") or (c.single_failure and self._single_failure_done) or (c.failure_rate > 0.0 and self.n_active_uavs < c.n_uavs)
```

### Test run output

```
============================= test session starts ==============================
collected 8 items

tests/test_rq2.py::test_new_fields_exist_with_defaults PASSED            [ 12%]
tests/test_rq2.py::test_no_failure_by_default PASSED                     [ 25%]
tests/test_rq2.py::test_single_failure_fires_exactly_once PASSED         [ 37%]
tests/test_rq2.py::test_single_failure_timing PASSED                     [ 50%]
tests/test_rq2.py::test_single_failure_with_mobile PASSED                [ 62%]
tests/test_rq2.py::test_failure_rate_zero_no_change PASSED               [ 75%]
tests/test_rq2.py::test_failure_rate_nonzero PASSED                      [ 87%]
tests/test_rq2.py::test_failure_affects_coverage PASSED                  [100%]

============================== 8 passed in 0.58s ===============================
```

### Full suite

```
68 passed, 2 warnings in 5.90s
```
