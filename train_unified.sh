#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-$(command -v python)}"
accelerate_bin="${ACCELERATE_BIN:-$(command -v accelerate)}"
num_gpus="${FISH_NUM_PROCESSES:-3}"
mode="${1:-all}"

stage1_config=./configs/fish_unified_stage1.yaml
stage2_config=./configs/fish_unified_stage2.yaml
smoke_config=./configs/fish_unified_smoke.yaml
bootstrap=./checkpoints/fishmotip_unified_bootstrap.pth
stage1_dir=./outputs/fish_unified_stage1
stage2_dir=./outputs/fish_unified_stage2_joint
stage1_final="${stage1_dir}/checkpoint_7.pth"
stage2_final="${stage2_dir}/checkpoint_29.pth"

usage() {
  cat <<'EOF'
Usage: bash train_unified.sh [all|stage1|stage2|smoke]

  all     Run/resume Stage 1, then run/resume Stage 2 (default).
  stage1  Run/resume the 8-epoch association integration stage.
  stage2  Run/resume the 30-epoch joint fine-tuning stage.
  smoke   Run a two-step, three-GPU end-to-end training check.

Environment variables:
  FISH_NUM_PROCESSES=3   Number of training GPUs/processes.
  FISH_SKIP_PREFLIGHT=1  Skip the two Python preflight tests.
  FISH_FORCE_RESTART=1   Ignore existing epoch checkpoints and restart a stage.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

latest_epoch_checkpoint() {
  local checkpoint_dir="$1"
  [[ -d "${checkpoint_dir}" ]] || return 0
  find "${checkpoint_dir}" -maxdepth 1 -type f -name 'checkpoint_*.pth' -print 2>/dev/null \
    | sort -V \
    | tail -n 1
}

launch_distributed() {
  local launch_args=(
    launch
    --num_processes="${num_gpus}"
    --num_machines=1
    --mixed_precision=no
    --dynamo_backend=no
  )
  if (( num_gpus > 1 )); then
    launch_args+=(--multi_gpu)
  fi
  "${accelerate_bin}" "${launch_args[@]}" "$@"
}

ensure_environment() {
  [[ -x "${python_bin}" ]] || die "Python not found: ${python_bin}"
  [[ -x "${accelerate_bin}" ]] || die "Accelerate not found: ${accelerate_bin}"
  [[ "${num_gpus}" =~ ^[1-9][0-9]*$ ]] || die "FISH_NUM_PROCESSES must be a positive integer."
  [[ -f ./train.py ]] || die "train.py not found under ${project_root}."

  if [[ ! -f "${bootstrap}" || "${FISH_REBUILD_BOOTSTRAP:-0}" == "1" ]]; then
    printf '[prepare] Building unified bootstrap checkpoint...\n'
    PYTHONPATH="${project_root}" "${python_bin}" tools/merge_unified_checkpoints.py
  else
    printf '[prepare] Reusing bootstrap checkpoint: %s\n' "${bootstrap}"
  fi
}

run_preflight() {
  if [[ "${FISH_SKIP_PREFLIGHT:-0}" == "1" ]]; then
    printf '[prepare] Preflight tests skipped by FISH_SKIP_PREFLIGHT=1.\n'
    return
  fi
  PYTHONPATH="${project_root}" "${python_bin}" tests/test_unified_model.py
  PYTHONPATH="${project_root}" "${python_bin}" tests/test_real_fish_clip_unified.py \
    "${stage1_config}"
}

run_stage() {
  local stage_name="$1"
  local config_path="$2"
  local output_dir="$3"
  local final_checkpoint="$4"
  local initial_checkpoint="$5"
  local latest_checkpoint=""
  local resume_args=()

  mkdir -p "${output_dir}"

  if [[ "${FISH_FORCE_RESTART:-0}" != "1" && -f "${final_checkpoint}" ]]; then
    printf '[%s] Already complete; found %s\n' "${stage_name}" "${final_checkpoint}"
    return
  fi

  if [[ "${FISH_FORCE_RESTART:-0}" != "1" ]]; then
    latest_checkpoint="$(latest_epoch_checkpoint "${output_dir}")"
  fi

  if [[ -n "${latest_checkpoint}" ]]; then
    printf '[%s] Resuming optimizer, scheduler and epoch state from %s\n' \
      "${stage_name}" "${latest_checkpoint}"
    resume_args=(
      --resume-model "${latest_checkpoint}"
      --resume-optimizer True
      --resume-scheduler True
      --resume-states True
    )
  else
    [[ -f "${initial_checkpoint}" ]] || die \
      "${stage_name} initial checkpoint is missing: ${initial_checkpoint}"
    printf '[%s] Starting from %s\n' "${stage_name}" "${initial_checkpoint}"
    resume_args=(
      --resume-model "${initial_checkpoint}"
      --resume-optimizer False
      --resume-scheduler False
      --resume-states False
    )
  fi

  launch_distributed train.py \
    --config-path "${config_path}" \
    "${resume_args[@]}"

  [[ -f "${final_checkpoint}" ]] || die \
    "${stage_name} exited without producing its final checkpoint: ${final_checkpoint}"
  printf '[%s] Complete: %s\n' "${stage_name}" "${final_checkpoint}"
}

run_smoke() {
  printf '[smoke] Starting the two-step distributed training check on %s GPU(s).\n' "${num_gpus}"
  launch_distributed train.py \
    --config-path "${smoke_config}" \
    --resume-model "${bootstrap}" \
    --resume-optimizer False \
    --resume-scheduler False \
    --resume-states False
  printf '[smoke] Distributed training check completed successfully.\n'
}

case "${mode}" in
  all|stage1|stage2|smoke) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    die "Unknown mode: ${mode}"
    ;;
esac

cd "${project_root}"
ensure_environment

if [[ "${mode}" == "all" || "${mode}" == "stage1" || "${mode}" == "smoke" ]]; then
  run_preflight
fi

case "${mode}" in
  all)
    run_stage "stage1" "${stage1_config}" "${stage1_dir}" "${stage1_final}" "${bootstrap}"
    run_stage "stage2" "${stage2_config}" "${stage2_dir}" "${stage2_final}" "${stage1_final}"
    ;;
  stage1)
    run_stage "stage1" "${stage1_config}" "${stage1_dir}" "${stage1_final}" "${bootstrap}"
    ;;
  stage2)
    run_stage "stage2" "${stage2_config}" "${stage2_dir}" "${stage2_final}" "${stage1_final}"
    ;;
  smoke)
    run_smoke
    ;;
esac
