#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$script_dir"
image="${DER_DOCKER_IMAGE:-der-kaggle-cpu}"
platform="${DER_DOCKER_PLATFORM:-linux/amd64}"
competition_dir="/kaggle/input/competitions/cyber-physical-anomaly-detection-for-der-systems"
workspace_dir="$repo_root"
data_dir="$repo_root/data"
working_dir="$repo_root/kaggle-working"
docker_bin="${DER_DOCKER_BIN:-}"

usage() {
  cat <<EOF
Usage:
  ./run_docker.sh
  ./run_docker.sh shell
  ./run_docker.sh pipeline [main.py args...]
  ./run_docker.sh <command> [args...]

Defaults:
  - Opens an interactive shell when no command is provided
  - Uses image: ${image}
  - Uses platform: ${platform}

Examples:
  ./run_docker.sh
  ./run_docker.sh shell
  ./run_docker.sh pipeline
  ./run_docker.sh pipeline --chunksize 10000
  ./run_docker.sh uv run python main.py

Environment overrides:
  DER_DOCKER_IMAGE     Override the Docker image name
  DER_DOCKER_PLATFORM  Override the Docker platform
  DER_DOCKER_BIN       Override the docker executable path
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$data_dir" ]]; then
  printf 'Missing data directory: %s\n' "$data_dir" >&2
  exit 1
fi

mkdir -p "$working_dir"

if [[ -z "$docker_bin" ]]; then
  if command -v docker >/dev/null 2>&1; then
    docker_bin="$(command -v docker)"
  elif [[ -x "/Applications/Docker.app/Contents/Resources/bin/docker" ]]; then
    docker_bin="/Applications/Docker.app/Contents/Resources/bin/docker"
  fi
fi

if [[ -z "$docker_bin" ]]; then
  printf 'docker was not found. Open Docker Desktop or set DER_DOCKER_BIN first.\n' >&2
  exit 1
fi

if [[ -z "$("$docker_bin" image ls -q "$image")" ]]; then
  printf 'Docker image %s was not found. Build it with:\n' "$image" >&2
  printf '  docker build --platform %s -t %s .\n' "$platform" "$image" >&2
  exit 1
fi

docker_args=(
  run
  --rm
  --platform "$platform"
  -v "$workspace_dir:/workspace"
  -v "$data_dir:$competition_dir:ro"
  -v "$working_dir:/kaggle/working"
  -w /workspace
)

if [[ -t 0 && -t 1 ]]; then
  docker_args+=( -it )
fi

if [[ "$#" -eq 0 ]]; then
  set -- shell
fi

case "$1" in
  shell)
    shift
    if [[ "$#" -eq 0 ]]; then
      set -- /bin/bash
    fi
    ;;
  pipeline)
    shift
    set -- uv run python main.py "$@"
    ;;
esac

exec "$docker_bin" "${docker_args[@]}" "$image" "$@"
