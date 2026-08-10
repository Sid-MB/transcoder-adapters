#!/usr/bin/env bash

# [08-10-26] (session 1fee7826-a308-4d81-b53c-9cf6bf8512ea)
# Fast (~10 min) prompt-template sensitivity probe: generation only, no attribution.
# Answers "does the base model's gibberish come from the chat template, and does the adapter
# keep refusing without it?" before committing GPU to a full base-vs-adapter retrace.
#
# Logs: logs/template_probe/<timestamp>_<job_id>.{out,err}

SLURM_LOG_DIR="logs/template_probe"
source run_on_gpu/common.sh

run uv run --extra viz python -m analysis.evals.template_sensitivity_probe "$@"
