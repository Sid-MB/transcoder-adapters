#!/usr/bin/env bash

# Example:
# ./sh/slurm_batch_token_metrics_multiple.sh --models "model1 model2" --transcoder --reference_model google/gemma-2-2b-it --data_source lmsys_chat

# Parse arguments to extract models
MODELS=()
OTHER_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODELS+=("$2")
      shift 2
      ;;
    --models)
      # Split by space or comma
      IFS=' ,' read -r -a array <<< "$2"
      MODELS+=("${array[@]}")
      shift 2
      ;;
    *)
      OTHER_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "Usage: $0 --models \"model1 model2 ...\" [other args]"
  echo "   or: $0 --model model1 --model model2 [other args]"
  exit 1
fi

for model in "${MODELS[@]}"; do
  # Use the last part of the model name for the job name
  JOB_NAME_SUFFIX=$(basename "$model")
  
  echo "Submitting job for model: $model (job name: token_metrics_${JOB_NAME_SUFFIX})"

  ./sh/slurm_batch_token_metrics.sh --job-name "token_metrics_${JOB_NAME_SUFFIX}" --model "$model" "${OTHER_ARGS[@]}"
done
