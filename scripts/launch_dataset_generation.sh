#!/usr/bin/env bash

# Launch sharded dataset generation from a login node. The CLI submits a worker
# array and a dependent finalizer, so this script does no processing itself.

set -euo pipefail

repo_dir="/global/home/users/pranavwalimbe/no2-modeling"
shard_size=16000
batch_size=500
afterok_job_id=""

usage() {
    cat <<'EOF'
Usage: launch_dataset_generation.sh [--shard-size N] [--batch-size N] [--afterok JOB_ID] [-- CLI_ARGS ...]

Submits the dataset-generation worker array and its dependent finalizer.

Options:
  --shard-size N   Source records per array task (default 16000).
  --batch-size N   Records staged together by each worker (default 500).
  --afterok JOB_ID Hold the shard array until JOB_ID completes successfully.

Common CLI arguments passed through after --:
  --split NAME       Limit the run to one split.

Every sharded launch replaces prior generated outputs. The persistent TEMPO and
weather caches are reused. Pass --refresh-tempo or --refresh-weather only when
their raster contracts changed.
EOF
}

while (($# > 0)); do
    case "$1" in
        --shard-size)
            if (($# < 2)); then
                echo "--shard-size requires a value" >&2
                exit 2
            fi
            shard_size="$2"
            shift 2
            ;;
        --batch-size)
            batch_size="$2"
            shift 2
            ;;
        --afterok)
            if (($# < 2)); then
                echo "--afterok requires a job ID" >&2
                exit 2
            fi
            afterok_job_id="$2"
            shift 2
            ;;
        --help | -h)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this launcher on a login node, not inside job ${SLURM_JOB_ID}" >&2
    exit 2
fi

cd "${repo_dir}"
export PYTHONPATH="${repo_dir}/src:${repo_dir}/src/delta-model"

echo "Launching dataset generation with shard size ${shard_size} and batch size ${batch_size}"
launcher_args=(--shard-size "${shard_size}" --batch-size "${batch_size}")
if [[ -n "${afterok_job_id}" ]]; then
    launcher_args+=(--afterok-job-id "${afterok_job_id}")
fi
exec .venv/bin/python -u -m preprocessing.generate_dataset "${launcher_args[@]}" "$@"
