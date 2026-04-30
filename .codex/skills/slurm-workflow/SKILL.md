---
name: slurm-workflow
description: Repo-specific workflow for submitting, monitoring, and debugging Slurm jobs for transcoder-adapters. Use when the user says things like "run this with slurm", "launch a run", "look at the output for the last run", asks Codex to run scripts from the repo's sh/ directory, sync local commits to the cluster before a run, submit jobs with sbatch, wait for Slurm completion, inspect squeue/sacct state, or read job logs under logs/ on the Stanford cluster; when running from a local machine, all such commands must be executed through ssh to sc.stanford.edu in /nlp/u/siddharth/transcoder-adapters.
---

# Slurm Run Workflow

## Overview

Use this workflow for Slurm work in this repository. The canonical cluster checkout is `/nlp/u/siddharth/transcoder-adapters`; local machines must reach it with `ssh sc.stanford.edu`.

Prefer the bundled helper scripts because they encode the local-vs-cluster rule, cluster checkout path, job ID parsing, and recursive log lookup. From the repository root, run them under `.codex/skills/slurm-workflow/scripts/`:

- `.codex/skills/slurm-workflow/scripts/cluster_exec.sh <command> [args...]`: run a command in the canonical cluster checkout, using ssh from local.
- `.codex/skills/slurm-workflow/scripts/submit_job.sh [--wait] <sh/slurm_batch_*.sh|slurm_batch_*.sh> [args...]`: submit a repo Slurm wrapper and print `job_id=<id>`.
- `.codex/skills/slurm-workflow/scripts/find_job_logs.sh <job-id>`: find matching `.out` and `.err` files recursively under `logs/`.
- `.codex/skills/slurm-workflow/scripts/summarize_job.sh [--tail N] <job-id>`: show `squeue`, `sacct`, matching logs, failure markers, and log tails.

## Checking if we're on the local machine or the cluster

To figure out if you're on the cluster or a local machine, check if `sbatch` is available with `command -v sbatch`: if it finds it, you're on the cluster.

If you're not on the cluster, you'll need to ssh into the cluster to read logs and start Slurm runs. Use:

```bash
ssh sc.stanford.edu 'cd /nlp/u/siddharth/transcoder-adapters && <command>'
```

If you can't find logs, make sure you're looking on the cluster in the right place.


## Submit Jobs

Use the repo's existing scripts in `sh/` to submit jobs. Choose the script that matches the user's requested job type, then run it from `/nlp/u/siddharth/transcoder-adapters`.

Examples:

```bash
.codex/skills/slurm-workflow/scripts/submit_job.sh sh/slurm_batch_train.sh <args>
```

Use `SBATCH_WAIT=1` when the user asks to wait for completion, when debugging a short run, or when final logs are needed before responding:

```bash
.codex/skills/slurm-workflow/scripts/submit_job.sh --wait sh/slurm_batch_train.sh <args>
```

Capture the Slurm job ID from the submission output. It is usually the numeric ID in output like `Submitted batch job 123456`.

## Monitor Jobs

Use the captured job ID for status checks:

```bash
.codex/skills/slurm-workflow/scripts/cluster_exec.sh squeue -j <job-id>
.codex/skills/slurm-workflow/scripts/cluster_exec.sh sacct -j <job-id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

Apply the same location rule: run these directly on the cluster or through `ssh sc.stanford.edu` from local.

## Read Logs

Logs live under `logs/`, inside a job-type-specific subfolder. File names end with the Slurm job ID suffix and follow:

```text
<DATE>_<TIME>_<SLURM_JOB_ID>.out
<DATE>_<TIME>_<SLURM_JOB_ID>.err
```

Find the exact files by matching the job ID suffix recursively:

```bash
.codex/skills/slurm-workflow/scripts/find_job_logs.sh <job-id>
```

Then read the relevant tail:

```bash
.codex/skills/slurm-workflow/scripts/cluster_exec.sh tail -n 200 <log-file>
```

For failures, inspect `.err` first, then `.out`. Search both logs for common failure markers:

```bash
.codex/skills/slurm-workflow/scripts/cluster_exec.sh grep -nE 'Traceback|Error|Exception|FAILED|CANCELLED|OutOfMemory|OOM|CUDA|No space left' <log-file>
```

For a single status-and-log pass, prefer:

```bash
.codex/skills/slurm-workflow/scripts/summarize_job.sh <job-id>
```
Also note that the second line of a Slurm log has the command that was used for the run.

<!-- ## Reporting Back

When summarizing a run, include the script name, job ID, current/final Slurm state, and the exact log files inspected. If the job failed, quote only the short error excerpt needed to diagnose it and name the next concrete fix or rerun command. -->

## Iterative development
If asked for iterative development or similar, you can follow the workflow of launching a job with wait, waiting until it finishes, inspecting the logs, and if it failed, launching a new job with a fix, looping forever until success.

## Sync Code Before Submitting jobs if not working on the cluster

If you made code or config changes locally and you're not working on the cluster, do not start a Slurm run until the cluster checkout has those changes. The cluster runs `/nlp/u/siddharth/transcoder-adapters`, so a local unpushed or unfetched edit means Slurm will run old code.

Before submitting:

1. Commit the local changes.
2. Push the commit if the cluster needs to fetch it from a remote.
3. Fetch and fast-forward the same branch on the cluster.
4. Verify the cluster checkout is at the intended commit before launching Slurm.

Useful command pattern from the local repo:

```bash
branch=$(git branch --show-current)
sha=$(git rev-parse HEAD)
git push origin "$branch"
.codex/skills/slurm-workflow/scripts/cluster_exec.sh git fetch origin "$branch"
.codex/skills/slurm-workflow/scripts/cluster_exec.sh git checkout "$branch"
.codex/skills/slurm-workflow/scripts/cluster_exec.sh git merge --ff-only "$sha"
.codex/skills/slurm-workflow/scripts/cluster_exec.sh git rev-parse HEAD
```

The final cluster `git rev-parse HEAD` must match the local `sha`. If the cluster checkout has uncommitted changes or cannot fast-forward, stop and ask before submitting; do not run stale or ambiguous code.
