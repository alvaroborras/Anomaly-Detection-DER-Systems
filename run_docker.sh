#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$script_dir"
image="der-kaggle-cpu"
platform="linux/amd64"
competition_dir="/kaggle/input/competitions/cyber-physical-anomaly-detection-for-der-systems"
workspace_dir="$repo_root"
data_dir="$repo_root/data"
working_dir="$repo_root/"
docker_bin=""

usage() {
  cat <<EOF
Usage:
  ./run_docker.sh

This script runs the fixed reproducible Docker workflow for main.py.

Pinned runtime:
  - image: ${image}
  - platform: ${platform}
  - output: /kaggle/working/submission.csv

Build the image first if needed:
  docker build --platform ${platform} -t ${image} .
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unsupported argument: %s\n' "$1" >&2
      printf 'run_docker.sh does not accept any arguments\n' >&2
      exit 2
      ;;
  esac
done

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

exec "${docker_cmd[@]}"
