#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
accelerate_bin="${ACCELERATE_BIN:-$(command -v accelerate)}"
checkpoint_arg="${1:-latest}"
num_gpus="${FISH_EVAL_PROCESSES:-2}"
outputs_dir="${FISH_EVAL_OUTPUTS_DIR:-./outputs/unified_eval}"
id_thresholds="${FISH_ID_THRESHOLDS:-0.02 0.05 0.10 0.15}"

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

cd "${project_root}"
[[ -x "${accelerate_bin}" ]] || die "Accelerate not found: ${accelerate_bin}"
[[ "${num_gpus}" =~ ^[1-9][0-9]*$ ]] || die "FISH_EVAL_PROCESSES must be a positive integer."

case "${checkpoint_arg}" in
  latest)
    checkpoint="$(latest_epoch_checkpoint ./outputs/fish_unified_stage2_joint)"
    [[ -n "${checkpoint}" ]] || die \
      "No Stage-2 checkpoint exists. Train first with: FISH_NUM_PROCESSES=3 bash train_unified.sh all"
    ;;
  stage1-latest)
    checkpoint="$(latest_epoch_checkpoint ./outputs/fish_unified_stage1)"
    [[ -n "${checkpoint}" ]] || die \
      "No Stage-1 checkpoint exists. Train first with: FISH_NUM_PROCESSES=3 bash train_unified.sh stage1"
    ;;
  bootstrap)
    checkpoint=./checkpoints/fishmotip_unified_bootstrap.pth
    printf 'WARNING: Evaluating bootstrap weights, not a trained unified checkpoint.\n' >&2
    ;;
  *)
    checkpoint="${checkpoint_arg}"
    ;;
esac

if [[ "${checkpoint}" != /* ]]; then
  checkpoint="${project_root}/${checkpoint#./}"
fi
[[ -f "${checkpoint}" ]] || die \
  "Checkpoint not found: ${checkpoint}. Run training first, or use 'bash evaluate_unified.sh latest'."

printf '[evaluate] checkpoint=%s\n' "${checkpoint}"
printf '[evaluate] processes=%s, ID thresholds=%s\n' "${num_gpus}" "${id_thresholds}"

for id_threshold in ${id_thresholds}; do
  printf '[evaluate] ID threshold=%s\n' "${id_threshold}"
  launch_distributed submit_and_evaluate.py \
    --config-path ./configs/fish_unified_stage2.yaml \
    --inference-model "${checkpoint}" \
    --outputs-dir "${outputs_dir}" \
    --inference-group "unified_id_${id_threshold}" \
    --id-thresh "${id_threshold}"
done
