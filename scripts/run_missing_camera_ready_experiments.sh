#!/usr/bin/env bash

# Run the remaining camera-ready experiments called out by
# reference/rebuttal_main_tex_gap_checklist.md.
#
# Usage:
#   bash scripts/run_missing_camera_ready_experiments.sh
#
# Optional overrides, for example:
#   RUN_SENSITIVITY=0 DEVICE=cuda \
#   SENS_DATASETS="scene15 bdgp cub nus_wide" \
#   bash scripts/run_missing_camera_ready_experiments.sh

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_CMD="${PY_CMD:-uv run python}"
DATA_ROOT="${DATA_ROOT:-data}"
DEVICE="${DEVICE:-}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"

RUN_ROBUSTNESS="${RUN_ROBUSTNESS:-1}"
RUN_ABLATION_IMVC="${RUN_ABLATION_IMVC:-1}"
RUN_ABLATION_MIXED="${RUN_ABLATION_MIXED:-1}"
RUN_SENSITIVITY="${RUN_SENSITIVITY:-1}"
RUN_RUNTIME="${RUN_RUNTIME:-1}"
RUN_CONVERGENCE="${RUN_CONVERGENCE:-1}"
RUN_TSNE="${RUN_TSNE:-1}"

BATCH_SIZE="${BATCH_SIZE:-256}"
ROBUST_EPOCHS="${ROBUST_EPOCHS:-100}"
ROBUST_RUNS="${ROBUST_RUNS:-5}"
ABLATION_EPOCHS="${ABLATION_EPOCHS:-100}"
ABLATION_RUNS="${ABLATION_RUNS:-3}"
SENS_EPOCHS="${SENS_EPOCHS:-200}"
SENS_RUNS="${SENS_RUNS:-5}"
RUNTIME_ITERATIONS="${RUNTIME_ITERATIONS:-10}"
RUNTIME_OPTION_EPOCHS="${RUNTIME_OPTION_EPOCHS:-20}"
RUNTIME_OPTION_PRETRAIN_EPOCHS="${RUNTIME_OPTION_PRETRAIN_EPOCHS:-6}"
CONVERGENCE_EPOCHS="${CONVERGENCE_EPOCHS:-100}"
TSNE_EPOCHS="${TSNE_EPOCHS:-100}"

ROBUSTNESS_DATASETS="${ROBUSTNESS_DATASETS:-bdgp coil20 cub handwritten nus_wide scene15}"
ABLATION_DATASETS="${ABLATION_DATASETS:-scene15 bdgp coil20 cub handwritten nus_wide}"
SENS_DATASETS="${SENS_DATASETS:-scene15 bdgp coil20 cub handwritten nus_wide}"
RUNTIME_DATASETS="${RUNTIME_DATASETS:-nus-wide}"
CONVERGENCE_DATASETS="${CONVERGENCE_DATASETS:-bdgp nus_wide}"
TSNE_DATASETS="${TSNE_DATASETS:-coil20}"

ROBUSTNESS_OUT="${ROBUSTNESS_OUT:-results/robustness/camera_ready_full}"
ABLATION_IMVC_OUT="${ABLATION_IMVC_OUT:-results/ablation/camera_ready_imvc}"
ABLATION_MIXED_OUT="${ABLATION_MIXED_OUT:-results/ablation/camera_ready_mixed}"
SENS_OUT_ROOT="${SENS_OUT_ROOT:-sensitivity_results/camera_ready_full}"
RUNTIME_OUT="${RUNTIME_OUT:-benchmark_results/camera_ready_nus_wide}"
CONVERGENCE_OUT_ROOT="${CONVERGENCE_OUT_ROOT:-multi_seed_results/camera_ready}"
TSNE_OUT="${TSNE_OUT:-figures/camera_ready_tsne}"

read -r -a PY <<< "$PY_CMD"
DEVICE_ARGS=()
if [[ -n "$DEVICE" ]]; then
  DEVICE_ARGS=(--device "$DEVICE")
fi

run_step() {
  local step_name="$1"
  shift

  echo
  echo "================================================================================"
  echo "$step_name"
  printf '%q ' "$@"
  echo
  echo "================================================================================"

  if "$@"; then
    return 0
  else
    local status=$?
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      echo "ERROR: $step_name failed with exit code $status."
      exit "$status"
    fi
    echo "ERROR: $step_name failed with exit code $status; continuing because STOP_ON_ERROR=0."
  fi
}

echo "================================================================================"
echo "Camera-ready missing experiment runner"
echo "================================================================================"
echo "PY_CMD: $PY_CMD"
echo "DATA_ROOT: $DATA_ROOT"
echo "DEVICE: $DEVICE"
echo "STOP_ON_ERROR: $STOP_ON_ERROR"
echo
echo "Phases:"
echo "  RUN_ROBUSTNESS=$RUN_ROBUSTNESS"
echo "  RUN_ABLATION_IMVC=$RUN_ABLATION_IMVC"
echo "  RUN_ABLATION_MIXED=$RUN_ABLATION_MIXED"
echo "  RUN_SENSITIVITY=$RUN_SENSITIVITY"
echo "  RUN_RUNTIME=$RUN_RUNTIME"
echo "  RUN_CONVERGENCE=$RUN_CONVERGENCE"
echo "  RUN_TSNE=$RUN_TSNE"
echo "================================================================================"

if [[ "$RUN_ROBUSTNESS" == "1" ]]; then
  read -r -a datasets <<< "$ROBUSTNESS_DATASETS"
  for dataset in "${datasets[@]}"; do
    run_step "Full six-dataset robustness baselines: $dataset" \
      "${PY[@]}" scripts/run_robustness_test.py \
      --test_type both \
      --dataset "$dataset" \
      --data_root "$DATA_ROOT" \
      --epochs "$ROBUST_EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --num_runs "$ROBUST_RUNS" \
      --save_dir "$ROBUSTNESS_OUT" \
      --missing_rates 0.0 0.1 0.3 0.5 0.7 \
      --unaligned_rates 0.0 0.2 0.4 0.6 \
      "${DEVICE_ARGS[@]}"
  done
fi

if [[ "$RUN_ABLATION_IMVC" == "1" ]]; then
  read -r -a datasets <<< "$ABLATION_DATASETS"
  for dataset in "${datasets[@]}"; do
    run_step "Aligned incomplete component ablation: $dataset" \
      "${PY[@]}" scripts/run_ablation.py \
      --dataset "$dataset" \
      --data_root "$DATA_ROOT" \
      --analysis component \
      --modes full no_gw no_flow no_contrastive no_cluster no_recon \
      --epochs "$ABLATION_EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --num_runs "$ABLATION_RUNS" \
      --missing_rate 0.7 \
      --unaligned_rate 0.0 \
      --save_dir "$ABLATION_IMVC_OUT" \
      "${DEVICE_ARGS[@]}"
  done
fi

if [[ "$RUN_ABLATION_MIXED" == "1" ]]; then
  read -r -a datasets <<< "$ABLATION_DATASETS"
  for dataset in "${datasets[@]}"; do
    run_step "Mixed missing+unaligned stress ablation: $dataset" \
      "${PY[@]}" scripts/run_ablation.py \
      --dataset "$dataset" \
      --data_root "$DATA_ROOT" \
      --analysis component \
      --modes full no_gw no_flow no_cluster no_recon \
      --epochs "$ABLATION_EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --num_runs "$ABLATION_RUNS" \
      --missing_rate 0.7 \
      --unaligned_rate 0.5 \
      --save_dir "$ABLATION_MIXED_OUT" \
      "${DEVICE_ARGS[@]}"
  done
fi

if [[ "$RUN_SENSITIVITY" == "1" ]]; then
  read -r -a datasets <<< "$SENS_DATASETS"
  for dataset in "${datasets[@]}"; do
    run_step "Complete 9-parameter sensitivity: $dataset" \
      "${PY[@]}" scripts/run_sensitivity_analysis.py \
      --dataset "$dataset" \
      --data_root "$DATA_ROOT" \
      --mode full \
      --epochs "$SENS_EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --num_runs "$SENS_RUNS" \
      --output_dir "$SENS_OUT_ROOT/$dataset" \
      "${DEVICE_ARGS[@]}"
  done
fi

if [[ "$RUN_RUNTIME" == "1" ]]; then
  read -r -a runtime_datasets <<< "$RUNTIME_DATASETS"
  run_step "NUS-WIDE non-diffusion runtime benchmark" \
    "${PY[@]}" scripts/benchmark_comprehensive.py \
    --datasets "${runtime_datasets[@]}" \
    --benchmark_mode baselines \
    --methods MRG-UMC CANDY SURE \
    --data_dir "$DATA_ROOT" \
    --batch_size "$BATCH_SIZE" \
    --iterations "$RUNTIME_ITERATIONS" \
    --output_dir "$RUNTIME_OUT" \
    --option_epochs "$RUNTIME_OPTION_EPOCHS" \
    --option_pretrain_epochs "$RUNTIME_OPTION_PRETRAIN_EPOCHS" \
    --option_batch_size 128
fi

if [[ "$RUN_CONVERGENCE" == "1" ]]; then
  read -r -a datasets <<< "$CONVERGENCE_DATASETS"
  for dataset in "${datasets[@]}"; do
    run_step "Multi-seed convergence: $dataset" \
      "${PY[@]}" scripts/run_multi_seed_convergence.py \
      --dataset "$dataset" \
      --epochs "$CONVERGENCE_EPOCHS" \
      --n_seeds 5 \
      --start_seed 42 \
      --output_dir "$CONVERGENCE_OUT_ROOT/$dataset"
  done
fi

if [[ "$RUN_TSNE" == "1" ]]; then
  read -r -a datasets <<< "$TSNE_DATASETS"
  for dataset in "${datasets[@]}"; do
    run_step "t-SNE visualization: $dataset" \
      "${PY[@]}" scripts/run_tsne_visualization.py \
      --dataset "$dataset" \
      --epochs "$TSNE_EPOCHS" \
      --checkpoints "0,$TSNE_EPOCHS" \
      --seed 42 \
      --output_dir "$TSNE_OUT" \
      --batch_size "$BATCH_SIZE"
  done
fi

echo
echo "================================================================================"
echo "Requested experiment phases finished."
echo "================================================================================"
