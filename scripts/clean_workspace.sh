#!/bin/bash

set -euo pipefail

repository_dir="/global/home/users/pranavwalimbe/no2-modeling"
user_home_dir="/global/home/users/pranavwalimbe"
pip_cache_dir="${user_home_dir}/.cache/pip"
uv_cache_dir="${user_home_dir}/.cache/uv"
codex_cache_dirs=(
    "${user_home_dir}/.codex_seed/cache"
    "${user_home_dir}/.codex.bak/cache"
)
codex_cache_files=(
    "${user_home_dir}/.codex_seed/models_cache.json"
    "${user_home_dir}/.codex.bak/models_cache.json"
)

if [[ ! -d "${repository_dir}/.git" ]]; then
    echo "Expected repository is missing: ${repository_dir}" >&2
    exit 1
fi

for cache_dir in "${pip_cache_dir}" "${uv_cache_dir}"; do
    if [[ -d "${cache_dir}" ]]; then
        echo "Removing package cache: ${cache_dir}"
        rm -rf -- "${cache_dir}"
    fi
done

for cache_dir in "${codex_cache_dirs[@]}"; do
    if [[ -d "${cache_dir}" ]]; then
        echo "Clearing Codex cache: ${cache_dir}"
        find "${cache_dir}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
    fi
done

for cache_file in "${codex_cache_files[@]}"; do
    if [[ -f "${cache_file}" ]]; then
        echo "Removing Codex cache file: ${cache_file}"
        rm -f -- "${cache_file}"
    fi
done

find "${repository_dir}/src" "${repository_dir}/tests" "${repository_dir}/scripts" \
    -type d -name __pycache__ -prune -exec rm -rf -- {} +
rm -rf -- \
    "${repository_dir}/build" \
    "${repository_dir}/dist" \
    "${repository_dir}/.pytest_cache" \
    "${repository_dir}/.ruff_cache"
find "${repository_dir}/src" -mindepth 1 -maxdepth 1 -type d -name '*.egg-info' -exec rm -rf -- {} +

echo "Workspace cleanup complete"
