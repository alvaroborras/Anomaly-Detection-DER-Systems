#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$script_dir"
image="der-kaggle-cpu"
platform="linux/amd64"
competition_dir="/kaggle/input/competitions/cyber-physical-anomaly-detection-for-der-systems"
workspace_dir="$repo_root"
data_dir="$repo_root/data"
working_dir="$repo_root/kaggle-working"
docker_bin=""

usage() {
  cat <<EOF
Usage:
  ./run_docker.sh [--train-row-limit N] [--profile-memory]

This script runs the fixed reproducible Docker workflow for main.py.

Allowed override:
  --train-row-limit N  Train on a deterministic sample of N training rows.
                       Use 0 for the full dataset.
  --profile-memory     Log stage-level timing and RSS memory summary.

Pinned runtime:
  - image: ${image}
  - platform: ${platform}
  - output: /kaggle/working/submission.csv

Build the image first if needed:
  docker build --platform ${platform} -t ${image} .
EOF
}

train_row_limit=""
profile_memory=0
main_args=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --train-row-limit)
      if [[ "$#" -lt 2 ]]; then
        printf 'Missing value for --train-row-limit\n' >&2
        exit 2
      fi
      if [[ -n "$train_row_limit" ]]; then
        printf '%s\n' '--train-row-limit may only be provided once' >&2
        exit 2
      fi
      train_row_limit="$2"
      shift 2
      ;;
    --train-row-limit=*)
      if [[ -n "$train_row_limit" ]]; then
        printf '%s\n' '--train-row-limit may only be provided once' >&2
        exit 2
      fi
      train_row_limit="${1#*=}"
      shift
      ;;
    --profile-memory)
      if [[ "$profile_memory" -eq 1 ]]; then
        printf '%s\n' '--profile-memory may only be provided once' >&2
        exit 2
      fi
      profile_memory=1
      shift
      ;;
    *)
      printf 'Unsupported argument: %s\n' "$1" >&2
      printf 'run_docker.sh only accepts --train-row-limit N and --profile-memory\n' >&2
      exit 2
      ;;
  esac
done

if [[ -n "$train_row_limit" ]]; then
  if [[ ! "$train_row_limit" =~ ^[0-9]+$ ]]; then
    printf 'Invalid --train-row-limit value: %s\n' "$train_row_limit" >&2
    printf 'Expected a non-negative integer.\n' >&2
    exit 2
  fi
  main_args+=(--train-row-limit "$train_row_limit")
fi

if [[ "$profile_memory" -eq 1 ]]; then
  main_args+=(--profile-memory)
fi

if [[ ! -d "$data_dir" ]]; then
  printf 'Missing data directory: %s\n' "$data_dir" >&2
  exit 1
fi

mkdir -p "$working_dir"

if command -v docker >/dev/null 2>&1; then
  docker_bin="$(command -v docker)"
elif [[ -x "/Applications/Docker.app/Contents/Resources/bin/docker" ]]; then
  docker_bin="/Applications/Docker.app/Contents/Resources/bin/docker"
else
  printf 'docker was not found. Open Docker Desktop so the pinned Docker workflow can run.\n' >&2
  exit 1
fi

if [[ -z "$("$docker_bin" image ls -q "$image")" ]]; then
  printf 'Docker image %s was not found. Build it with:\n' "$image" >&2
  printf '  docker build --platform %s -t %s .\n' "$platform" "$image" >&2
  exit 1
fi

docker_cmd=(
  "$docker_bin"
  run
  --rm
  --platform "$platform"
  -v "$workspace_dir:/workspace"
  -v "$data_dir:$competition_dir:ro"
  -v "$working_dir:/kaggle/working"
  -w /workspace
  "$image"
  uv run python main.py
)

if [[ ${#main_args[@]} -gt 0 ]]; then
  docker_cmd+=("${main_args[@]}")
fi

exec "${docker_cmd[@]}"
