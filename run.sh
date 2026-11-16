#!/usr/bin/env bash
# run.sh — one-stop launcher for uav-placement-bench
#
# Usage (from repo root):
#   bash run.sh smoke          # quick sanity check (~2 min)
#   bash run.sh tune           # val-split hyperparameter tuning
#   bash run.sh sweep          # full 10-seed reported sweep (hours)
#   bash run.sh sweep 5        # sweep with N seeds instead of 10
#   bash run.sh sweep 10 400000  # sweep with custom timesteps
#   bash run.sh figures        # regenerate figures from existing results/
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

# Detect CUDA and pick device
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
    DEVICE="cuda"
    echo "[run.sh] CUDA detected: ${N_GPUS} GPU(s)"
else
    N_GPUS=0
    DEVICE="cpu"
    echo "[run.sh] No CUDA — using CPU"
fi

N_CPUS=$(python3 -c "import os; print(os.cpu_count())")
echo "[run.sh] CPUs: ${N_CPUS}  |  mode: ${MODE}  |  device: ${DEVICE}"
echo "─────────────────────────────────────────────────────────────────"

# ── Helpers ───────────────────────────────────────────────────────────────────

run_smoke() {
    echo ""
    echo "▶ Smoke test (2 seeds, 20 k steps, 5 layouts) — not reported results"
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
    echo "  Baselines: PSO particle count / iterations"
    echo "  RL:        learning rate, batch size"
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

    # Build seed list
    SEEDS=$(python3 -c "print(' '.join(str(i) for i in range($N_SEEDS)))")

    # Baselines (CPU — all at once, they're fast)
    echo "── Baselines ──"
    python3 scripts/run_baselines.py \
        --regime "$REGIME" \
        --seeds $SEEDS \
        --obs-mode "$OBS_MODE"

    # RL — split seeds across GPUs if multiple are available
    echo ""
    echo "── RL training ──"

    if [[ "$N_GPUS" -ge 2 ]]; then
        echo "  Splitting seeds across ${N_GPUS} GPUs in parallel"
        HALF=$(( N_SEEDS / 2 ))
        SEEDS_GPU0=$(python3 -c "print(' '.join(str(i) for i in range($HALF)))")
        SEEDS_GPU1=$(python3 -c "print(' '.join(str(i) for i in range($HALF, $N_SEEDS)))")

        for algo in ppo sac a2c dqn; do
            echo "  ${algo} — GPU 0: seeds 0-$((HALF-1)), GPU 1: seeds ${HALF}-$((N_SEEDS-1))"
            (
                for seed in $SEEDS_GPU0; do
                    CUDA_VISIBLE_DEVICES=0 python3 scripts/train_rl.py \
                        --algo "$algo" --regime "$REGIME" --seed "$seed" \
                        --timesteps "$TIMESTEPS" --device cuda \
                        --obs-mode "$OBS_MODE"
                done
            ) &
            (
                for seed in $SEEDS_GPU1; do
                    CUDA_VISIBLE_DEVICES=1 python3 scripts/train_rl.py \
                        --algo "$algo" --regime "$REGIME" --seed "$seed" \
                        --timesteps "$TIMESTEPS" --device cuda \
                        --obs-mode "$OBS_MODE"
                done
            ) &
            wait
        done
    else
        for algo in ppo sac a2c dqn; do
            echo "  ${algo}"
            for seed in $SEEDS; do
                python3 scripts/train_rl.py \
                    --algo "$algo" --regime "$REGIME" --seed "$seed" \
                    --timesteps "$TIMESTEPS" --device "$DEVICE" \
                    --obs-mode "$OBS_MODE"
            done
        done
    fi

    echo ""
    echo "── Aggregate + figures ──"
    python3 scripts/aggregate.py --regime "$REGIME" --metric final_cov
    python3 scripts/plot_figures.py --results results --metric final_cov
    echo ""
    echo "✓ Sweep complete."
    echo "  Results:  results/"
    echo "  Figures:  results/figures/"
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
    echo "Usage: bash run.sh <mode> [n_seeds] [timesteps] [obs_mode] [regime]"
    echo ""
    echo "Modes:"
    echo "  smoke          Quick sanity check — 2 seeds, 20k steps (~2 min)"
    echo "  tune           Val-split hyperparam tuning (always before sweep)"
    echo "  sweep          Full reported sweep (default: 10 seeds, 200k steps)"
    echo "  figures        Regenerate figures from existing results/"
    echo ""
    echo "Examples:"
    echo "  bash run.sh smoke"
    echo "  bash run.sh tune"
    echo "  bash run.sh sweep"
    echo "  bash run.sh sweep 10 200000 flat static"
    echo "  bash run.sh sweep 10 200000 flat shock    # Regime III"
    echo "  bash run.sh figures"
    echo ""
    echo "Machine detected:"
    echo "  CPUs: ${N_CPUS}  |  GPUs: ${N_GPUS}  |  device: ${DEVICE}"
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
