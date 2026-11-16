# Channel model: Al-Hourani air-to-ground (A2G)

Implementation: `uavbench/channel.py`. Tests: `tests/test_channel.py`.

## Sources

- [1] A. Al-Hourani, S. Kandeepan, S. Lardner, "Optimal LAP Altitude for Maximum
  Coverage," *IEEE Wireless Communications Letters*, 3(6):569–572, 2014.
  DOI: 10.1109/LWC.2014.2342736
- [2] A. Al-Hourani, S. Kandeepan, A. Jamalipour, "Modeling air-to-ground path loss
  for low altitude platforms in urban environments," *IEEE GLOBECOM*, 2014.

## Equations (angles in degrees, as in [1])

- Elevation angle: θ = arctan(h / r), with h the altitude and r the horizontal distance.
- Probability of line of sight: `P_LoS(θ) = 1 / (1 + a·exp(−b(θ − a)))`
- Free-space path loss: `FSPL(d) = 20log10(d) + 20log10(f) + 20log10(4π/c)`, d = √(h² + r²)
- Mean path loss: `PL = FSPL + P_LoS·η_LoS + (1 − P_LoS)·η_NLoS`
- Coverage: a user is covered if min over UAVs of `PL ≤ pl_max` (default 100 dB).
- Optimal elevation, Eq. (13) of [1], with A = η_LoS − η_NLoS:
  `π/(9 ln 10)·tan(θ) + a·b·A·exp(−b(θ−a)) / (a·exp(−b(θ−a)) + 1)² = 0`.
  The root θ_opt does **not** depend on `pl_max` (tested).

## Parameters (2 GHz)

| Environment | a | b | η_LoS (dB) | η_NLoS (dB) | θ_opt (deg) | Verification |
|---|---|---|---|---|---|---|
| Suburban | 4.88 | 0.43 | 0.1 | 21 | 20.34 | Commonly cited; **check vs [1],[2]** |
| Urban (default) | 9.61 | 0.16 | 1.0 | 20 | 42.44 | η values match Meta-A3C paper; a,b **check vs [1]** |
| Dense urban | 12.08 | 0.11 | 1.6 | 23 | 54.62 | Commonly cited; **check vs [1],[2]** |
| High-rise urban | 27.23 | 0.08 | 2.3 | 34 | 75.52 | Commonly cited; **check vs [1],[2]** |

The θ_opt column is computed by our solver and matches the commonly reported values to
0.01°. This is good evidence that the equations are implemented correctly. It does not
prove the a, b constants are right: confirm them against the paper.

## Altitude rule used by the k-means baseline

For fixed horizontal radius R, minimizing PL over θ gives the same condition as Eq. (13).
So the elevation angle θ_opt also **minimizes path loss at a given cluster edge**, and
the k-means baseline sets h = R_c·tan(θ_opt), where R_c is the 90th-percentile
member distance, clipped to [h_min, h_max]. With the default 30–150 m limit, this
usually clips to h_max.

## Limitations (state these in the paper)

- Mean path loss only. No small-scale fading and no log-normal shadowing variance.
- No interference. Coverage is a path-loss threshold, not SINR. (SINR is on the roadmap.)
- Parameters are statistical averages for generic environments, not site-specific
  (contrast: Qiu et al. use site-specific LoS maps).

## Verification checklist (human must tick)

- [ ] a, b values for all four environments confirmed against [1] (Table/Fig.)
- [ ] η_LoS, η_NLoS confirmed against [2]
- [ ] Eq. (13) form confirmed against [1]
- [ ] Reproduced the radius-vs-altitude figure of [1] with `coverage_radius()`
