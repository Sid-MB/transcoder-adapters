#!/usr/bin/env bash
# Submit a staged circuit-tracer pipeline:
#   1. CPU Slurm job prepares circuit-tracer transcoders/features and writes a manifest.
#   2. GPU Slurm job depends on the CPU job and runs attribution from that manifest.
#
# Example:
#   ./sh/attribution/slurm_circuit_tracer_pipeline.sh --transcoder_model_path siddharthmb/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860 --base_model google/gemma-2-2b --feature_data_path /nlp/scr/siddharth/sparse-adaptation/feature_data/2026.TA.gemma2_2b_tc8192_decb_l1w0.001_tarbb_lb2.0_ln1_dr20000_lr8e-04_bs4_sl14793860_20260519_171751_15493160 --prompts analysis/attribution/prompts/interesting_small --run_name interesting_small --prompt_format chat --max_feature_nodes 256 --batch_size 4 --max_n_logits 5
# 
# Duplicate work is skipped at each stage, so the transcoder model will only be converted once and the feature data will only be converted if the packed circuit-tracer cache is missing. Attribution graph filenames include a short hash of each prompt file's contents, so if a prompt .txt file changes without changing names, the stale same-stem graph is removed and that prompt is rerun. The logs state which prompts are skipped, new, or rerun because their hash changed.
#
# Serving the tracing visualization
# After the job finishes, run (no GPU needed):
#   uv run --extra viz circuit-tracer start-server --graph_file_dir <graph_output_dir> --features_dir <feature_output_dir>/circuit_tracer_features
# 
# Note: collect_feature_activations writes <feature_dir>/circuit_tracer_features by default. If collection used --no-export_circuit_tracer_features, manually convert features to the correct format with analysis.attribution.export_circuit_tracer_feature_data and use that folder for --features_dir.

set -euo pipefail

CPU_PARTITION="${CT_CPU_PARTITION:-john}"
GPU_PARTITION="${CT_GPU_PARTITION:-jag-standard}"
CPU_CPUS="${CT_CPU_CPUS:-8}"
GPU_CPUS="${CT_GPU_CPUS:-8}"
CPU_MEM="${CT_CPU_MEM:-64G}"
GPU_MEM="${CT_GPU_MEM:-128G}"
CPU_TIME="${CT_CPU_TIME:-12:00:00}"
GPU_TIME="${CT_GPU_TIME:-02:00:00}"
ATTRIBUTION_GPUS="${ATTRIBUTION_GPUS:-1}"
MANIFEST_PATH=""
RUN_NAME="circuit_tracer"

PREP_ARGS=()
ATTR_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --transcoder_model_path|--base_model|--prompts|--feature_data_path|--transcoder_output_dir|--feature_output_dir|--graph_output_dir|--n_layers|--n_features|--feature_input_hook|--feature_output_hook|--activation)
      PREP_ARGS+=("$1" "$2")
      shift 2
      ;;
    --run_name)
      RUN_NAME="$2"
      PREP_ARGS+=("$1" "$2")
      shift 2
      ;;
    --prompt_format|--max_n_logits|--batch_size|--max_feature_nodes|--node_threshold|--edge_threshold|--device|--device_map|--port)
      ATTR_ARGS+=("$1" "$2")
      shift 2
      ;;
    --auto_shard_gpus|--serve)
      ATTR_ARGS+=("$1")
      shift
      ;;
    --manifest_path)
      MANIFEST_PATH="$2"
      shift 2
      ;;
    --cpu_partition)
      CPU_PARTITION="$2"
      shift 2
      ;;
    --gpu_partition)
      GPU_PARTITION="$2"
      shift 2
      ;;
    --cpu_time)
      CPU_TIME="$2"
      shift 2
      ;;
    --gpu_time)
      GPU_TIME="$2"
      shift 2
      ;;
    --cpu_mem)
      CPU_MEM="$2"
      shift 2
      ;;
    --gpu_mem)
      GPU_MEM="$2"
      shift 2
      ;;
    --cpu_cpus)
      CPU_CPUS="$2"
      shift 2
      ;;
    --gpu_cpus)
      GPU_CPUS="$2"
      shift 2
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ${#PREP_ARGS[@]} -eq 0 ]]; then
  echo "ERROR: Missing required prepare arguments. Pass --transcoder_model_path, --base_model, and --prompts." >&2
  exit 1
fi

if [[ -z "$MANIFEST_PATH" ]]; then
  mkdir -p logs/attribution
  timestamp="$(date +%Y%m%d_%H%M%S)"
  MANIFEST_PATH="logs/attribution/${RUN_NAME}_${timestamp}_manifest.json"
fi

if [[ -f ~/.shell/secrets/hf_token_write ]]; then
  export HF_TOKEN
  HF_TOKEN="$(cat ~/.shell/secrets/hf_token_write)"
fi

quote_command() {
  printf "%q " "$@"
}

prepare_cmd=(
  uv run --extra viz python -m analysis.attribution.prepare_circuit_tracer_assets
  "${PREP_ARGS[@]}"
  --manifest_path "$MANIFEST_PATH"
)
attribution_cmd=(
  uv run --extra viz python -m analysis.attribution.run_attribution_from_manifest
  --manifest_path "$MANIFEST_PATH"
  "${ATTR_ARGS[@]}"
)

prepare_job="$(
  ./sh/sbatch \
    --parsable \
    --gres=gpu:0 \
    --cpus-per-task="$CPU_CPUS" \
    --mem="$CPU_MEM" \
    --time="$CPU_TIME" \
    --partition="$CPU_PARTITION" \
    --job-name=ct_prepare \
    --output=logs/attribution/%x_%j.out \
    --error=logs/attribution/%x_%j.err \
    --wrap="$(quote_command "${prepare_cmd[@]}")"
)"

attribution_job="$(
  ./sh/sbatch \
    --parsable \
    --dependency="afterok:${prepare_job}" \
    --gres="gpu:${ATTRIBUTION_GPUS}" \
    --constraint=48G \
    --cpus-per-task="$GPU_CPUS" \
    --mem="$GPU_MEM" \
    --time="$GPU_TIME" \
    --partition="$GPU_PARTITION" \
    --job-name=ct_attribution \
    --output=logs/attribution/%x_%j.out \
    --error=logs/attribution/%x_%j.err \
    --wrap="$(quote_command "${attribution_cmd[@]}")"
)"

echo "Submitted CPU prepare job: ${prepare_job}"
echo "Submitted GPU attribution job: ${attribution_job}"
echo "Manifest: ${MANIFEST_PATH}"
