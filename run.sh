#!/usr/bin/env bash
# run.sh — one-stop launcher for uav-placement-bench
#
# Usage (from repo root):
#   bash run.sh smoke          # quick sanity check — all obs_modes (~5 min)
#   bash run.sh verify         # full verification checklist end-to-end
#   bash run.sh tune           # val-split hyperparameter tuning
#   bash run.sh sweep          # full 10-seed reported sweep (hours)
#   bash run.sh sweep 5        # sweep with N seeds instead of 10
#   bash run.sh sweep 10 200000 flat static   # explicit obs_mode + regime
#   bash run.sh rq1            # RQ1: train all 5 obs_modes, then plot
#   bash run.sh rq2            # RQ2: 6-config complexity sweep + heatmap
#   bash run.sh rq3            # RQ3: generalization eval + figure
#   bash run.sh figures        # regenerate figures from existing results/
#
# Hard rules (from AGENTS.md):
#   - Tune only on val split  (tune mode)
#   - Reported results go to results/ via sweep mode only
#   - rq1/rq2/rq3 write to results_rq1/ results_rq2/ results_rq3/ — never to results/
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
    echo "▶ Smoke test — 2 seeds, 20k steps, 5 layouts, all obs_modes"
    echo "  NOT reported results."
    echo ""

    # Classical baselines
    echo "── Baselines ──"
    python3 scripts/run_baselines.py \
        --regime static --seeds 0 1 --n-eval 5 --out /tmp/smoke

    # RL: all 5 obs_modes
    echo ""
    echo "── RL: all 5 obs_modes ──"
    for obs in flat sorted cluster_summary spatial_grid graph; do
        echo "  obs_mode=$obs"
        python3 scripts/train_rl.py \
            --algo ppo --regime static --seed 0 \
            --timesteps 20000 --n-eval 5 --device "$DEVICE" \
            --obs-mode "$obs" --out /tmp/smoke
    done

    # RQ2 sweep mini-run
    echo ""
    echo "── RQ2 smoke (configs C1 C4, 2 seeds) ──"
    python3 scripts/sweep_rq2.py \
        --seeds 0 1 --n-eval 5 --timesteps 20000 \
        --configs C1 C4 --out /tmp/rq2_smoke
    python3 scripts/plot_rq2.py --results /tmp/rq2_smoke --out /tmp/figures_smoke

    # RQ1 plot from smoke data
    echo ""
    echo "── RQ1 plot (smoke data) ──"
    python3 scripts/plot_rq1.py --results /tmp/smoke --regime static --out /tmp/figures_smoke

    echo ""
    echo "✓ Smoke test complete."
    echo "  Results : /tmp/smoke  (NOT reported results)"
    echo "  Figures : /tmp/figures_smoke"
}

run_verify() {
    echo ""
    echo "▶ Verification checklist — confirms full pipeline is functional"
    echo ""

    echo "── 1. Test suite ──"
    pytest -q
    echo ""

    echo "── 2. All 5 obs_modes (20k steps each) ──"
    for obs in flat sorted cluster_summary spatial_grid graph; do
        echo "  obs_mode=$obs"
        python3 scripts/train_rl.py \
            --algo ppo --regime static --seed 0 \
            --timesteps 20000 --n-eval 5 --device "$DEVICE" \
            --obs-mode "$obs" --out /tmp/verify
    done

    echo ""
    echo "── 3. New baselines (warm_start_pso, mpc) ──"
    python3 scripts/run_baselines.py \
        --regime static --algos warm_start_pso mpc \
        --seeds 0 1 --n-eval 5 --out /tmp/verify

    echo ""
    echo "── 4. RQ2 sweep (C1 + C4 only) ──"
    python3 scripts/sweep_rq2.py \
        --seeds 0 1 --n-eval 5 --timesteps 20000 \
        --configs C1 C4 --out /tmp/verify_rq2
    python3 scripts/plot_rq2.py --results /tmp/verify_rq2 --out /tmp/verify_figures

    echo ""
    echo "── 5. RQ3 generalization ──"
    python3 scripts/eval_generalization.py \
        --model-dir /tmp/verify/static \
        --seeds 0 --n-eval 5 --split val --out /tmp/verify_rq3
    python3 scripts/plot_rq3.py --results /tmp/verify_rq3 --out /tmp/verify_figures

    echo ""
    echo "── 6. RQ1 figure ──"
    python3 scripts/plot_rq1.py \
        --results /tmp/verify --regime static --out /tmp/verify_figures

    echo ""
    echo "✓ Verification complete — all steps passed."
    echo "  Figures written to /tmp/verify_figures/"
    echo ""
    echo "⚠  Before running a full sweep, confirm with supervisor:"
    echo "   docs/DECISIONS.md D-017 — step_duration_s=1.0 s vs 2.0 s (affects RQ2 crossover)"
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
    echo "▶ Full sweep: ${N_SEEDS} seeds × ${TIMESTEPS} steps × obs_mode=${OBS_MODE} × regime=${REGIME}"
    echo "  Writing to results/ — this is the REPORTED dataset."
    echo ""

    SEEDS=$(python3 -c "print(' '.join(str(i) for i in range($N_SEEDS)))")

    echo "── Baselines ──"
    python3 scripts/run_baselines.py \
        --regime "$REGIME" \
        --seeds $SEEDS \
        --obs-mode "$OBS_MODE"

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
    echo "  Results:   results/"
    echo "  Figures:   results/figures/"
    echo "  Aggregate: results/aggregate_*.json"
}

run_rq1() {
    echo ""
    echo "▶ RQ1: Representation comparison — train all 5 obs_modes, then plot"
    echo "  Seeds: ${N_SEEDS}  |  Timesteps: ${TIMESTEPS}  |  Regime: ${REGIME}"
    echo "  Writing to results_rq1/"
    echo ""

    SEEDS=$(python3 -c "print(' '.join(str(i) for i in range($N_SEEDS)))")

    for obs in flat sorted cluster_summary spatial_grid graph; do
        echo "── obs_mode=$obs ──"
        for seed in $SEEDS; do
            python3 scripts/train_rl.py \
                --algo ppo --regime "$REGIME" --seed "$seed" \
                --timesteps "$TIMESTEPS" --device "$DEVICE" \
                --obs-mode "$obs" --out results_rq1
        done
    done

    echo ""
    echo "── Generating RQ1 figure ──"
    mkdir -p figures
    python3 scripts/plot_rq1.py \
        --results results_rq1 --regime "$REGIME" --out figures
    echo ""
    echo "✓ RQ1 complete."
    echo "  Results : results_rq1/"
    echo "  Figure  : figures/rq1_${REGIME}.pdf"
}

run_rq2() {
    echo ""
    echo "▶ RQ2: Complexity threshold sweep — 6 configs × all methods"
    echo "  Seeds: ${N_SEEDS}  |  Timesteps: ${TIMESTEPS}"
    echo "  Writing to results_rq2/"
    echo ""

    SEEDS=$(python3 -c "print(' '.join(str(i) for i in range($N_SEEDS)))")

    python3 scripts/sweep_rq2.py \
        --seeds $SEEDS \
        --timesteps "$TIMESTEPS" \
        --device "$DEVICE" \
        --out results_rq2

    echo ""
    echo "── Generating RQ2 heatmap ──"
    mkdir -p figures
    python3 scripts/plot_rq2.py --results results_rq2 --out figures
    echo ""
    echo "✓ RQ2 complete."
    echo "  Results : results_rq2/"
    echo "  Figure  : figures/rq2_heatmap.pdf"
}

run_rq3() {
    echo ""
    echo "▶ RQ3: Generalization — distribution shift + fleet-size transfer"
    echo "  Seeds: ${N_SEEDS}  |  Model dir: results/static"
    echo "  Writing to results_rq3/"
    echo ""

    SEEDS=$(python3 -c "print(' '.join(str(i) for i in range($N_SEEDS)))")

    # RQ3-A: distribution shift (requires trained models in results/static)
    python3 scripts/eval_generalization.py \
        --model-dir results/static \
        --seeds $SEEDS \
        --split test \
        --out results_rq3

    # RQ3-B: GNN fleet-size generalization
    python3 scripts/eval_generalization.py \
        --model-dir results/static \
        --seeds $SEEDS \
        --split test \
        --rq3b \
        --out results_rq3

    echo ""
    echo "── Generating RQ3 figure ──"
    mkdir -p figures
    python3 scripts/plot_rq3.py --results results_rq3 --out figures
    echo ""
    echo "✓ RQ3 complete."
    echo "  Results : results_rq3/"
    echo "  Figure  : figures/rq3_generalization.pdf"
    echo ""
    echo "  Note: RQ3 requires trained models from a completed sweep."
    echo "  Run 'bash run.sh sweep' or 'bash run.sh rq1' first."
}

run_figures() {
    echo ""
    echo "▶ Regenerating all figures from existing results"
    python3 scripts/aggregate.py --regime "$REGIME" --metric final_cov
    python3 scripts/plot_figures.py --results results --metric final_cov
    [[ -d results_rq1 ]] && python3 scripts/plot_rq1.py --results results_rq1 --regime "$REGIME" --out figures
    [[ -d results_rq2 ]] && python3 scripts/plot_rq2.py --results results_rq2 --out figures
    [[ -d results_rq3 ]] && python3 scripts/plot_rq3.py --results results_rq3 --out figures
    echo "✓ Figures written to results/figures/ and figures/"
}

print_help() {
    echo ""
    echo "Usage: bash run.sh <mode> [n_seeds] [timesteps] [obs_mode] [regime]"
    echo ""
    echo "Modes:"
    echo "  smoke          Quick sanity check — all 5 obs_modes, 2 seeds, 20k steps (~5 min)"
    echo "  verify         Full verification checklist — confirms everything works end-to-end"
    echo "  tune           Val-split hyperparam tuning (always before sweep)"
    echo "  sweep          Full reported sweep (default: 10 seeds, 200k steps)"
    echo "  rq1            RQ1: train all 5 obs_modes + plot comparison figure"
    echo "  rq2            RQ2: 6-config complexity sweep + heatmap"
    echo "  rq3            RQ3: generalization eval + figure (needs prior sweep)"
    echo "  figures        Regenerate all figures from existing results"
    echo ""
    echo "Examples:"
    echo "  bash run.sh smoke"
    echo "  bash run.sh verify"
    echo "  bash run.sh tune"
    echo "  bash run.sh sweep"
    echo "  bash run.sh sweep 10 200000 flat static    # explicit options"
    echo "  bash run.sh sweep 10 200000 graph static   # GNN obs_mode"
    echo "  bash run.sh rq1"
    echo "  bash run.sh rq1 10 200000 static           # explicit seeds/steps/regime"
    echo "  bash run.sh rq2"
    echo "  bash run.sh rq3                            # requires prior sweep"
    echo "  bash run.sh figures"
    echo ""
    echo "Recommended order:"
    echo "  1. smoke   — confirm nothing is broken"
    echo "  2. verify  — confirm full pipeline end-to-end"
    echo "  3. tune    — tune hyperparams on val split"
    echo "  4. rq1     — train all obs_modes (pick best for rq2/rq3)"
    echo "  5. rq2     — complexity threshold sweep"
    echo "  6. rq3     — generalization (needs rq1 models)"
    echo "  7. figures — regenerate all plots"
    echo ""
    echo "Machine detected:"
    echo "  CPUs: ${N_CPUS}  |  GPUs: ${N_GPUS}  |  device: ${DEVICE}"
    echo ""
    echo "Hard rules (AGENTS.md):"
    echo "  - Run 'tune' before 'sweep'"
    echo "  - Never report results from smoke mode"
    echo "  - results/ is sacred — never delete it"
    echo "  - rq1/rq2/rq3 write to results_rq1/ results_rq2/ results_rq3/"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────

case "$MODE" in
    smoke)   run_smoke   ;;
    verify)  run_verify  ;;
    tune)    run_tune    ;;
    sweep)   run_sweep   ;;
    rq1)     run_rq1     ;;
    rq2)     run_rq2     ;;
    rq3)     run_rq3     ;;
    figures) run_figures ;;
    help|*)  print_help  ;;
esac
