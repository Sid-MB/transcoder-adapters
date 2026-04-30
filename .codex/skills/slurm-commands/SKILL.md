---
name: slurm-commands
description: Information for launching Slurm jobs and reviewing their logs.
---

# Slurm — Run Jobs and Inspect Outputs

## In General

Use this workflow to work with Slurm and inspect job outputs.

If you run into errors about like slurm commands being unavailable or slurm connection failures, run slurm and `./sh/...` commands outside of the sandbox

## Checking if we're on the local machine or the cluster

To figure out if you're on the cluster or a local machine, check if `sbatch` is available with `command -v sbatch`: if it finds it, you're on the cluster.

If you're not on the cluster, you'll need to ssh into the cluster to read logs and start Slurm runs. Use:

```sh
ssh sc.stanford.edu 'cd /nlp/u/siddharth/transcoder-adapters/ && <command>'
```

For example (also waits until the job finishes, optional):
```sh
ssh sc.stanford.edu 'cd /nlp/u/siddharth/transcoder-adapters/ && SBATCH_WAIT=1 ./sh/slurm_batch_train --config training/configs/gemma2_2b.yaml'
```

## Submit Jobs

Use the repo's existing scripts in `sh/` to submit most jobs.

Remember the Slurm job ID from the submission output, (`15292344` in "Submitted batch job 123456"), so you can find associated logs after.

Tips:
- If you initiate your Slurm jobs with the env variable `SBATCH_WAIT=1`, the submission command will block until the job finishes, which is useful for iterative development.
- Use `scancel <job-id>` to stop a job early.

## Read Logs

Logs live under `logs/`, inside a job-type-specific subfolder. File names typically follow the pattern:

```
<DATE>_<TIME>_<SLURM_JOB_ID>.out
<DATE>_<TIME>_<SLURM_JOB_ID>.err
```
(there is one .out and one .err file for each job with the exact same names beside the extension). The best way to find the correct logs is to match the job ID. And, if someone provides one of the files, you can find the other by looking at the same path with the other extension.


Tip: Logs can be very long, especially the `.err` files since they include every step of `tqdm` progress bars! Use `head` and `tail` to your advantage:
- You can usually just `tail` logs to figure out if the job errored
- The second line of `*.out` logs tells you the exact command that the job ran
- You can also `tail -f` logs of running jobs

## Iterative development
If asked for iterative development or similar, you can follow the workflow of:
1. Launch a job, use `SBATCH_WAIT=1` or monitor the logs
2. Wait until it finishes or until something goes wrong
3. Inspect the logs and outputs, if any
4. If it failed, launch a new job with a fix, and repeat until success


## Monitor Jobs

Good commands to monitor jobs:

```bash
squeue -j <job-id>
sacct -j <job-id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
squeue --me
```

## Using Wandb

Training runs will log information to wandb in the entity "siddharth-stanford", project name "sparse-adaptation". You can use the wandb MCP server to get this information. To correlate with your slurm runs, wandb run names end with "_sl<SLURM_JOB_ID>".

## Sync Code Before Submitting jobs if not working on the cluster

If you made code or config changes locally and you're *not* working on the cluster, don't start a Slurm run until the cluster checkout has those changes. The cluster runs `/nlp/u/siddharth/transcoder-adapters`, so a local unpushed or unfetched edit means Slurm will run old code.
