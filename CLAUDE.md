@AGENTS.md

## Claude Code specifics

- Use plan mode before changing anything in `uavbench/env.py`, `uavbench/evaluation.py`,
  or the reward/observation. Present the plan and wait for approval.
- Use the smoke-test flags from AGENTS.md (`--timesteps 20000 --n-eval 5 --out /tmp/smoke`)
  while developing. Never start a full sweep without asking.
- At the end of every session, update `docs/STATUS.md` (session log + next steps).
- When unsure whether something is a research decision or an implementation detail,
  treat it as a research decision and ask.
