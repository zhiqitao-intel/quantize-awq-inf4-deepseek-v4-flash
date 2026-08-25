#!/usr/bin/env bash
# Run the DeepSeek-V4-Flash-0731 AWQ INT4 recipe in a disposable Intel XPU container.
#
# This is a REQUANTIZATION run: the source ships as FP4 experts + FP8 dense
# layers. The first step materializes a dense-BF16 mirror under $WORK, then
# the standard AWQ sweep runs against that mirror. See README.md for why.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
MODEL=${MODEL:?set MODEL to a complete local Hugging Face snapshot of deepseek-ai/DeepSeek-V4-Flash-0731}
MODEL=$(cd "$MODEL" && pwd -P)
WORK=${WORK:-$PWD/.work}
WORK=$(mkdir -p "$WORK" && cd "$WORK" && pwd -P)
OUTPUT=${OUTPUT:-$WORK/out/DeepSeek-V4-Flash-0731-AWQ-INT4}
CHECKPOINTS=${CHECKPOINTS:-$WORK/checkpoints/deepseek-v4-flash-awq-int4}
IMAGE=${IMAGE:-rahulunair/sglang-xpu@sha256:feb7b9130eff2fa26dfa09ab1a8b9a8423db013ad7d18f804f8b55a57cb2c175}
GPUS=${GPUS:-8}
SAMPLES=${SAMPLES:-512}
SEQ_LEN=${SEQ_LEN:-1024}
GROUP_SIZE=${GROUP_SIZE:-32}
N_GRID=${N_GRID:-20}
CODE_FRACTION=${CODE_FRACTION:-0.7}

case "$GPUS" in (*[!0-9]*|'') echo "GPUS must be a positive integer"; exit 2;; esac
[ "$GPUS" -gt 0 ] || { echo "GPUS must be greater than zero"; exit 2; }
[ -f "$MODEL/config.json" ] || { echo "config.json not found in MODEL"; exit 2; }

mkdir -p "$OUTPUT" "$CHECKPOINTS" "$WORK/logs"
OUTPUT=$(cd "$OUTPUT" && pwd -P)
CHECKPOINTS=$(cd "$CHECKPOINTS" && pwd -P)

echo "== preflight: inspecting source snapshot =="
(cd "$ROOT" && python3 -m model_compress.preflight --model "$MODEL" --allow-requantize)

CPU_COUNT=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
OMP_THREADS=$((CPU_COUNT / GPUS))
[ "$OMP_THREADS" -gt 0 ] || OMP_THREADS=1

echo "== launching containerized AWQ run =="
docker run --rm \
  --device=/dev/dri -v /dev/dri:/dev/dri --group-add video \
  --group-add "$(getent group render | cut -d: -f3)" \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --shm-size=64g \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  -e OMP_NUM_THREADS="$OMP_THREADS" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v "$ROOT:/workspace:ro" \
  -v "$MODEL:/model:ro" \
  -v "$WORK:/work" \
  -v "$OUTPUT:/output" \
  -v "$CHECKPOINTS:/checkpoints" \
  --entrypoint bash "$IMAGE" -lc "
    set -euo pipefail
    trap 'chown -R \"\$HOST_UID:\$HOST_GID\" /output /checkpoints /work 2>/dev/null || true' EXIT
    PY=/opt/venv/bin/python
    BEFORE_VERSION=\$(\$PY -c 'import torch; print(torch.__version__)')
    BEFORE_PATH=\$(\$PY -c 'import os, torch; print(os.path.realpath(torch.__file__))')
    UV_PYTHON=\$PY uv pip install -q --no-deps \
      -r /workspace/recipes/deepseek_v4_flash/requirements.lock
    cd /workspace
    \$PY -m model_compress.verify_environment --require-xpu \
      --expect-version \"\$BEFORE_VERSION\" --expect-path \"\$BEFORE_PATH\"
    cd /workspace/recipes/deepseek_v4_flash
    \$PY -m torch.distributed.run --standalone --nnodes=1 \
      --nproc-per-node=$GPUS awq_int4.py \
      --model /model \
      --output /output \
      --checkpoint-dir /checkpoints \
      --work-dir /work/mirror-scratch \
      --samples $SAMPLES --seq-len $SEQ_LEN \
      --group-size $GROUP_SIZE --n-grid $N_GRID \
      --code-fraction $CODE_FRACTION
  "

printf 'artifact: %s\n' "$OUTPUT"
