---
name: slurm-run-workflow
description: Repo-specific workflow for submitting, monitoring, and debugging Slurm jobs for transcoder-adapters. Use when the user says things like "run this with slurm", "launch a run", "look at the output for the last run", asks Codex to run scripts from the repo's sh/ directory, submit jobs with sbatch, wait for Slurm completion, inspect squeue/sacct state, or read job logs under logs/ on the Stanford cluster; when running from a local machine, all such commands must be executed through ssh to sc.stanford.edu in /nlp/u/siddharth/transcoder-adapters.
---

# Slurm Run Workflow

## Overview

Use this workflow for Slurm work in this repository. The canonical cluster checkout is `/nlp/u/siddharth/transcoder-adapters`; local machines must reach it with `ssh sc.stanford.edu`.

Prefer the bundled helper scripts because they encode the local-vs-cluster rule, cluster checkout path, job ID parsing, and recursive log lookup. From the repository root, run them under `.codex/skills/slurm-run-workflow/scripts/`:

- `.codex/skills/slurm-run-workflow/scripts/cluster_exec.sh <command> [args...]`: run a command in the canonical cluster checkout, using ssh from local.
- `.codex/skills/slurm-run-workflow/scripts/submit_job.sh [--wait] <sh/slurm_batch_*.sh|slurm_batch_*.sh> [args...]`: submit a repo Slurm wrapper and print `job_id=<id>`.
- `.codex/skills/slurm-run-workflow/scripts/find_job_logs.sh <job-id>`: find matching `.out` and `.err` files recursively under `logs/`.
- `.codex/skills/slurm-run-workflow/scripts/summarize_job.sh [--tail N] <job-id>`: show `squeue`, `sacct`, matching logs, failure markers, and log tails.

## Always Detect Location First

Run this before any Slurm command:

```bash
command -v sbatch
```

If it prints a path, run commands on the current machine, but first make sure the command executes from the cluster checkout:

```bash
cd /nlp/u/siddharth/transcoder-adapters && <command>
```

If it does not print a path, assume the current machine is local. Run every Slurm, `sh/`, `logs/`, `squeue`, and `sacct` command through ssh:

```bash
ssh sc.stanford.edu 'cd /nlp/u/siddharth/transcoder-adapters && <command>'
```

Do not inspect local `logs/` files or run local `sh/` scripts when `sbatch` is unavailable locally.

## Submit Jobs

Use the repo's existing scripts in `sh/` to submit jobs. Choose the script that matches the user's requested job type, then run it from `/nlp/u/siddharth/transcoder-adapters`.

Examples:

```bash
.codex/skills/slurm-run-workflow/scripts/submit_job.sh sh/slurm_batch_train.sh <args>
```

Use `SBATCH_WAIT=1` when the user asks to wait for completion, when debugging a short run, or when final logs are needed before responding:

```bash
.codex/skills/slurm-run-workflow/scripts/submit_job.sh --wait sh/slurm_batch_train.sh <args>
```

Capture the Slurm job ID from the submission output. It is usually the numeric ID in output like `Submitted batch job 123456`.

## Monitor Jobs

Use the captured job ID for status checks:

```bash
.codex/skills/slurm-run-workflow/scripts/cluster_exec.sh squeue -j <job-id>
.codex/skills/slurm-run-workflow/scripts/cluster_exec.sh sacct -j <job-id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

Apply the same location rule: run these directly on the cluster or through `ssh sc.stanford.edu` from local.

## Read Logs

Logs live under `logs/`, inside a job-type-specific subfolder. File names start with the Slurm job ID and follow:

```text
<SLURM_JOB_ID>_<DATE>_<TIME>.out
<SLURM_JOB_ID>_<DATE>_<TIME>.err
```

Find the exact files by matching the job ID prefix recursively:

```bash
.codex/skills/slurm-run-workflow/scripts/find_job_logs.sh <job-id>
```

Then read the relevant tail:

```bash
.codex/skills/slurm-run-workflow/scripts/cluster_exec.sh tail -n 200 <log-file>
```

For failures, inspect `.err` first, then `.out`. Search both logs for common failure markers:

```bash
.codex/skills/slurm-run-workflow/scripts/cluster_exec.sh grep -nE 'Traceback|Error|Exception|FAILED|CANCELLED|OutOfMemory|OOM|CUDA|No space left' <log-file>
```

For a single status-and-log pass, prefer:

```bash
.codex/skills/slurm-run-workflow/scripts/summarize_job.sh <job-id>
```

## Reporting Back

When summarizing a run, include the script name, job ID, current/final Slurm state, and the exact log files inspected. If the job failed, quote only the short error excerpt needed to diagnose it and name the next concrete fix or rerun command.
