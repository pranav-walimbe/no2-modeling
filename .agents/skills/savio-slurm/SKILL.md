---
name: savio-slurm
description: Write or review Slurm batch scripts for UC Berkeley's Savio cluster, including account/partition/QoS selection, CPU/GPU and memory-aware hardware requests, email notifications, logs, environment setup, and submission checks. Use whenever creating, modifying, explaining, or troubleshooting a Savio sbatch script in this repository.
---

# Savio Slurm

Create runnable, workload-specific Savio batch scripts. Treat Berkeley Research IT documentation and live scheduler associations as authoritative; do not infer hardware access from an old script alone.

## Workflow

1. Inspect the command being scheduled, its parallelism, expected memory, expected duration, and whether it needs a GPU.
2. Inspect nearby job scripts for repository paths, environment activation, account, and logging conventions.
3. Run `sacctmgr -p show associations user=$USER format=account,user,partition,qos` when available. Only select an account, partition, and QoS combination listed for the user.
4. Read [references/savio-resources.md](references/savio-resources.md) and check the linked official pages when hardware details may have changed.
5. Start from [assets/job-template.sh](assets/job-template.sh). Replace every `__PLACEHOLDER__`; never leave a placeholder in a script presented as runnable.
6. Validate with `bash -n <script>`. If Savio commands are available, also use `sbatch --test-only <script>` before recommending submission.

## Resource selection

- Prefer `savio4_htc` for serial or modestly threaded CPU work because it schedules by core. Request only the cores the program can actually use.
- Use one node and one task for a normal Python process. Increase `--cpus-per-task` only when the code or its libraries use threads or subprocesses, and propagate `$SLURM_CPUS_PER_TASK` to the application when needed.
- Do not add `--mem` by default. Savio generally assigns memory with the node or in proportion to requested cores. On `savio4_htc`, ordinary nodes provide 4 GB per requested core; `--constraint=savio4_m512` provides 8 GB per core. Requesting extra cores can therefore be a memory request even for serial code.
- Use per-node partitions only when the job can use the node. A small serial job wastes allocation and service units there.
- For a GPU job, specify the GPU partition, model, and count with `--gres=gpu:<MODEL>:<COUNT>`. Match `--cpus-per-task` to the current CPU:GPU ratio and add the required FCA QoS. Verify both in the official hardware table before writing the script.
- Use `savio_debug` for short validation runs and normal/condo QoS for real runs only when the user's association permits it. Do not copy `savio_normal` or a model-specific GPU QoS without checking.
- Set the shortest realistic wall time with contingency. Savio requires a time limit; shorter accurate requests can schedule sooner.

## Required script properties

Every generated script must include:

- a Bash shebang and `set -euo pipefail`;
- `--job-name`, `--account`, `--partition`, `--nodes`, `--ntasks`, `--cpus-per-task`, and `--time`;
- `--mail-type=BEGIN,END,FAIL` and a real `--mail-user` address supplied by the user or already established in repository-local scripts;
- stdout at `/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.log` and stderr at the matching `.err` path, with the ignored `logs/` directory created before `sbatch` runs;
- an explicit repository working directory, the Savio Python 3.11 module, `source .venv/bin/activate`, and `srun` for the primary command.

Never fabricate an email address. If it is unknown, leave a conspicuous placeholder only in a draft and state that the script is not submission-ready.

## Repository defaults

Existing project scripts use the `fc_nitrates` account, Python 3.11, and the uv-managed repository-local `.venv`. Slurm jobs should activate the existing environment directly. Run `make setup` on a login node only when `.venv` is missing or the dependency files change; never run `uv sync` as part of a batch job. Route generated visualizations to `/global/home/users/pranavwalimbe/vis`, never into `src/eda/`, which contains scripts only. Route Slurm stdout and stderr to the repository's ignored `logs/` directory with `.log` and `.err` extensions. The older examples elsewhere in the user's workspace are useful for command structure, but current Savio documentation controls hardware and QoS choices.
