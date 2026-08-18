#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Checking shell syntax..."
for file in "${ROOT_DIR}"/*.sh "${ROOT_DIR}"/scripts/*.sh; do
  [[ -f "${file}" ]] || continue
  bash -n "${file}"
done

echo "Checking Python syntax..."
python3 -m py_compile \
  "${ROOT_DIR}/src/moonshine_pi_live.py" \
  "${ROOT_DIR}/tests/test_core.py"

echo "Running dependency-free self-tests..."
python3 "${ROOT_DIR}/src/moonshine_pi_live.py" --self-test
python3 "${ROOT_DIR}/tests/test_core.py"

echo "ALL STATIC CHECKS PASSED"
