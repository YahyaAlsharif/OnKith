#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROOT_DIR}/run.sh" benchmark \
  --labels "${ROOT_DIR}/data/benchmark_labels.txt" "$@"
