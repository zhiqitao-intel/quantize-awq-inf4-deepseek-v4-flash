#!/usr/bin/env bash
# CI Smoke Driver
#
# Invoked by GitHub Actions (.github/workflows/smoke.yml) on every push.
# Runs the surrogate smoke test plus the lighter-weight sanity checks.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> preflight checks (warning-mode)"
python -m scripts.preflight_check || {
    echo "preflight reported warnings or errors; continuing because CI is warning-mode by default"
}

echo "==> pytest (unit suite)"
python -m pytest -q tests/test_pipeline.py

echo "==> surrogate smoke (RUN_SMOKE=1)"
RUN_SMOKE=1 python -m pytest -q tests/test_pipeline.py::test_smoke_surrogate_full

echo "==> calibration determinism (RUN_SMOKE=1)"
RUN_SMOKE=1 python -m pytest -q tests/test_pipeline.py::test_calibration_determinism

echo "==> cleanup tmp pollution"
rm -rf /tmp/dsv4_probe /tmp/_smoke_out /tmp/det-test-A /tmp/det-test-B

echo "==> SMOKE TESTS PASSED"