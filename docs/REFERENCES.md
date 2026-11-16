# References

Status key:
- **VERIFIED**: title, authors, and venue checked against a primary or official source.
- **PARTIAL**: exists (title/arXiv checked); venue, year, or details not fully checked.
- **UNVERIFIED**: do not cite until checked.
- **WRONG**: a known error from earlier drafts. Never use it.

Agents: only cite VERIFIED entries in paper text. New entries start as UNVERIFIED and
must include a DOI or URL.

## Channel model

| Ref | Status | Link |
|---|---|---|
| Al-Hourani, Kandeepan, Lardner, "Optimal LAP Altitude for Maximum Coverage," IEEE WCL 3(6), 2014 | VERIFIED (constants still to check, see CHANNEL_MODEL.md) | doi:10.1109/LWC.2014.2342736 |
| Al-Hourani, Kandeepan, Jamalipour, "Modeling air-to-ground path loss for low altitude platforms in urban environments," IEEE GLOBECOM 2014 | VERIFIED | |

## Supervising lab

| Ref | Status | Link |
|---|---|---|
| Cyprien, Sookhak, Zarini, Sekharan, Atiquzzaman, "Joint UAV-UGV Positioning and Trajectory Planning via Meta A3C for Reliable Emergency Communications," GLOBECOM 2025 | VERIFIED (arXiv text read) | arxiv.org/abs/2512.22187 |
| Sobouti, Adarbah, Alaghehband, Chitsaz, Mohajerzadeh, Sookhak, Hosseini Seno, Vahedian, Afghah, "Efficient Fuzzy-Based 3-D Flying Base Station Positioning and Trajectory for Emergency Management in 5G and Beyond Cellular Networks," IEEE Systems Journal 18(2), 2024 | VERIFIED | par.nsf.gov/servlets/purl/10520378 |
| Rahimi, ..., Sookhak, 3D UAV BS positioning and backhaul via stochastic optimization, GLOBECOM 2022 | PARTIAL (seen in Meta-A3C references only) | |
| Sookhak NSF CAREER award ($632,414, 5 yrs, "SIM-Integrated Air-to-Ground Autonomous Vehicle Networks") | VERIFIED | tamucc.edu/research/annual-report/2026/ |

## DRL for UAV placement / coverage

| Ref | Status | Link |
|---|---|---|
| Qiu, Lyu, Fu, "Placement Optimization of Aerial Base Stations with Deep Reinforcement Learning," IEEE ICC 2020. States the problem is NP-hard; compares against DQN and k-means | VERIFIED | arxiv.org/abs/1911.08111 |
| Bayerlein et al., UAV path planning for wireless data harvesting (DDQN), GLOBECOM 2020; multi-agent extension in IEEE OJ-COMS 2021 | PARTIAL | github.com/hbayerlein/uav_data_harvesting |
| Liu et al., DRL-EC3 (DDPG, energy-efficient fair coverage), IEEE JSAC 2018 | PARTIAL (secondary sources only) | |
| Liu, Liu, Chen, k-means + Q-learning for ABS movement, GLOBECOM 2018 | PARTIAL | |
| Lyu et al., spiral placement of UAV-mounted mobile base stations, IEEE Commun. Lett. | PARTIAL (check year: 2016 vs 2017) | |
| Demirtas et al., autonomous UAV BS placement, deep learning vs RL | PARTIAL (arXiv only; journal venue unverified) | arxiv.org/abs/2107.13869 |
| Wang & Farooq, DRL placement for integrated access backhaul | PARTIAL (venue unverified; NOT confirmed as IEEE IoT J.) | arxiv.org/abs/2312.14247 |
| Multi-agent DRL for collaborative UAV relay networks under jamming (single fixed seed) | PARTIAL (author names unverified) | arxiv.org/abs/2512.08341 |

## Environments

| Ref | Status | Link |
|---|---|---|
| Brunori, Colonnese, Cuomo, Iocchi, "A Reinforcement Learning Environment for Multi-Service UAV-enabled Wireless Systems," 2021 | VERIFIED | arxiv.org/abs/2105.05094 |
| Schneider et al., "mobile-env: An Open Platform for Reinforcement Learning in Wireless Mobile Networks," IEEE/IFIP NOMS 2022 | VERIFIED | doi via ieeexplore.ieee.org/document/9789886 |

## Evaluation methodology

| Ref | Status | Link |
|---|---|---|
| Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice," NeurIPS 2021 (rliable) | VERIFIED | arxiv.org/abs/2108.13264 |
| Henderson et al., "Deep Reinforcement Learning that Matters," AAAI 2018. TRPO 5-vs-5 seed split: t = −9.0916, p = 0.0016 | VERIFIED (that statistic only) | arxiv.org/abs/1709.06560 |
| Reproducibility study of RELIANCE for DRL UAV collision avoidance (PEMWN 2025 → Annals of Telecommunications 2026) | PARTIAL | doi.org/10.1007/s12243-026-01224-2 |
| DRL for 6G AI-RAN survey (reproducibility reporting requirements) | PARTIAL | arxiv.org/abs/2608.14877 |

## Surveys

| Ref | Status | Link |
|---|---|---|
| Survey on DRL-based UAV communications and networking, 2025 | PARTIAL | arxiv.org/abs/2502.12875 |
| Survey on RL in spatial resource allocation, 2024 | PARTIAL | arxiv.org/abs/2403.03643 |

## Known WRONG (from an earlier AI-generated report: never use)

- "Rahimi et al. (2024), fuzzy-logic controller": wrong first author. It's Sobouti et al.
- "Safe-DQN trajectory optimization" attributed to Sookhak's group: wrong. Different authors.
- Qiu et al. "compared only against a geometric design": wrong. They also compared against DQN and k-means.
- Henderson et al. "33.52× KL" and specific PPO/DDPG return numbers: UNVERIFIED. Do not quote.
- "15.5 hours total compute", "<0.2 ms per step", "PSO ~12 minutes": invented.
