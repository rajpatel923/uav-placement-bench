#!/usr/bin/env bash
# run_5090.sh — launcher for Ryzen 9 9950X + RTX 5090 (single GPU, 16C/32T)
#
# Usage (from repo root):
#   bash run_5090.sh smoke          # quick sanity check (~2 min)
#   bash run_5090.sh tune           # val-split hyperparameter tuning
#   bash run_5090.sh sweep          # full 10-seed reported sweep
#   bash run_5090.sh sweep 5        # sweep with N seeds instead of 10
#   bash run_5090.sh sweep 10 400000  # sweep with custom timesteps
#   bash run_5090.sh figures        # regenerate figures from existing results/
#
# Hard rules (from AGENTS.md):
#   - Tune only on val split  (tune mode)
#   - Reported results go to results/ via sweep mode only
#   - Never run sweep without reading docs/EXPERIMENT_PROTOCOL.md first

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
MODE="${1:-help}"
N_SEEDS="${2:-10}"
TIMESTEPS="${3:-200000}"
OBS_MODE="${4:-flat}"
REGIME="${5:-static}"
DEVICE="cuda"

# Verify CUDA
if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[run_5090.sh] WARNING: CUDA not found — falling back to CPU"
    DEVICE="cpu"
fi

GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "N/A")
N_CPUS=$(python3 -c "import os; print(os.cpu_count())")

echo "[run_5090.sh] GPU: ${GPU_NAME}  |  CPUs: ${N_CPUS}  |  mode: ${MODE}  |  device: ${DEVICE}"
echo "─────────────────────────────────────────────────────────────────"

# ── Helpers ───────────────────────────────────────────────────────────────────

run_smoke() {
    echo ""
    echo "▶ Smoke test (2 seeds, 20k steps, 5 layouts) — not reported results"
    echo ""
    python3 scripts/run_baselines.py \
        --regime static --seeds 0 1 --n-eval 5 --out /tmp/smoke

    for algo in ppo sac; do
        python3 scripts/train_rl.py \
            --algo "$algo" --regime static --seed 0 \
            --timesteps 20000 --n-eval 5 --device "$DEVICE" \
            --out /tmp/smoke
    done

    echo ""
    echo "✓ Smoke test complete. Results at /tmp/smoke (not reported results)."
}

run_tune() {
    echo ""
    echo "▶ Hyperparameter tuning — VAL split only (results_val/)"
    echo ""
    python3 scripts/tune_baselines.py --regime static
    python3 scripts/tune_rl.py --regime static --device "$DEVICE"
    echo ""
    echo "✓ Tuning complete. Best configs written to results_val/best_*.json"
    echo "  Review before running sweep."
}

run_sweep() {
    echo ""
    echo "▶ Full sweep: ${N_SEEDS} seeds × ${TIMESTEPS} steps × regime=${REGIME}"
    echo "  Writing to results/ — this is the REPORTED dataset."
    echo ""

    SEEDS=$(python3 -c "print(' '.join(str(i) for i in range($N_SEEDS)))")

    # Baselines: fast, run sequentially
    echo "── Baselines ──"
    python3 scripts/run_baselines.py \
        --regime "$REGIME" \
        --seeds $SEEDS \
        --obs-mode "$OBS_MODE"

    # RL: single GPU — run 2 seeds in parallel to keep GPU saturated
    # while staying within the RTX 5090's 32 GB VRAM budget.
    echo ""
    echo "── RL training (2 seeds in parallel on RTX 5090) ──"

    for algo in ppo a2c dqn sac; do   # SAC last — it's the slowest
        echo "  ${algo}: ${N_SEEDS} seeds"
        PIDS=()
        COUNT=0
        for seed in $SEEDS; do
            python3 scripts/train_rl.py \
                --algo "$algo" --regime "$REGIME" --seed "$seed" \
                --timesteps "$TIMESTEPS" --device "$DEVICE" \
                --obs-mode "$OBS_MODE" &
            PIDS+=($!)
            COUNT=$(( COUNT + 1 ))
            # Allow at most 2 concurrent seeds to avoid OOM on 32 GB VRAM
            if [[ "$COUNT" -ge 2 ]]; then
                wait "${PIDS[@]}"
                PIDS=()
                COUNT=0
            fi
        done
        # Wait for any remaining seeds
        [[ "${#PIDS[@]}" -gt 0 ]] && wait "${PIDS[@]}"
    done

    echo ""
    echo "── Aggregate + figures ──"
    python3 scripts/aggregate.py --regime "$REGIME" --metric final_cov
    python3 scripts/plot_figures.py --results results --metric final_cov
    echo ""
    echo "✓ Sweep complete."
    echo "  Results:   results/"
    echo "  Figures:   results/figures/"
    echo "  Aggregate: results/aggregate_*.json"
}

run_figures() {
    echo ""
    echo "▶ Regenerating figures from existing results/"
    python3 scripts/aggregate.py --regime "$REGIME" --metric final_cov
    python3 scripts/plot_figures.py --results results --metric final_cov
    echo "✓ Figures written to results/figures/"
}

print_help() {
    echo ""
    echo "Usage: bash run_5090.sh <mode> [n_seeds] [timesteps] [obs_mode] [regime]"
    echo ""
    echo "Modes:"
    echo "  smoke          Quick sanity check — 2 seeds, 20k steps (~2 min)"
    echo "  tune           Val-split hyperparam tuning (always before sweep)"
    echo "  sweep          Full reported sweep (default: 10 seeds, 200k steps)"
    echo "  figures        Regenerate figures from existing results/"
    echo ""
    echo "Examples:"
    echo "  bash run_5090.sh smoke"
    echo "  bash run_5090.sh tune"
    echo "  bash run_5090.sh sweep"
    echo "  bash run_5090.sh sweep 10 200000 flat static"
    echo "  bash run_5090.sh sweep 10 200000 flat shock    # Regime III"
    echo "  bash run_5090.sh figures"
    echo ""
    echo "Machine: ${GPU_NAME} + ${N_CPUS}-thread CPU"
    echo "  Runs 2 seeds in parallel to stay within 32 GB VRAM."
    echo "  For the dual-GPU Threadripper workstation, use run.sh instead."
    echo ""
    echo "Hard rules (AGENTS.md):"
    echo "  - Run 'tune' before 'sweep'"
    echo "  - Never report results from smoke mode"
    echo "  - results/ is sacred — never delete it"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────

case "$MODE" in
    smoke)   run_smoke   ;;
    tune)    run_tune    ;;
    sweep)   run_sweep   ;;
    figures) run_figures ;;
    help|*)  print_help  ;;
esac
