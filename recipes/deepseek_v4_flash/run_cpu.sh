#!/usr/bin/env bash
# Run the DeepSeek AWQ INT4 recipe on CPU only (no XPU required).
#
# This is a reduced-scope variant of run.sh for hosts without Intel XPU.
# It runs single-process, uses llm-compressor's default CPU dispatch, and
# is intended for:
#   - pipeline validation on small proxy models (minutes)
#   - proof-of-concept runs on real checkpoints (hours per layer)
#   - not for full 304B production runs (would take days)
#
# Env vars:
#   MODEL      path to HF snapshot directory
#   OUTPUT     where to write the packed artifact
#   WORK       scratch dir for calibration cache and intermediate state
#   SAMPLES    number of calibration samples (default 32)
#   SEQ_LEN    max sequence length for calibration (default 256)
#   GROUP_SIZE quantization group size (default 32)
#   N_GRID     AWQ grid search points (default 5)
#   MAX_LAYERS if set, only quantize the first N decoder layers

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
MODEL=${MODEL:?set MODEL to a local Hugging Face snapshot}
MODEL=$(cd "$MODEL" && pwd -P)
WORK=${WORK:-$PWD/.cpu-work}
OUTPUT=${OUTPUT:-$WORK/out/cpu-awq-int4}
SAMPLES=${SAMPLES:-32}
SEQ_LEN=${SEQ_LEN:-256}
GROUP_SIZE=${GROUP_SIZE:-32}
N_GRID=${N_GRID:-5}

[ -f "$MODEL/config.json" ] || { echo "config.json not found in MODEL"; exit 2; }
mkdir -p "$OUTPUT" "$WORK/logs"
OUTPUT=$(cd "$OUTPUT" && pwd -P)

# Preflight on the host (no container needed for inspection).
python3 -m model_compress.preflight --model "$MODEL" --allow-requantize || true

# Use repo-local venv so we don't pollute system Python.
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3

export OMP_NUM_THREADS=$(getconf _NPROCESSORS_ONLN)
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TOKENIZERS_PARALLELISM=false

cd "$ROOT/recipes/deepseek_v4_flash"
exec "$PY" awq_int4_cpu.py \
  --model "$MODEL" \
  --output "$OUTPUT" \
  --checkpoint-dir "$WORK/checkpoints" \
  --work-dir "$WORK/scratch" \
  --samples "$SAMPLES" \
  --seq-len "$SEQ_LEN" \
  --group-size "$GROUP_SIZE" \
  --n-grid "$N_GRID" \
  ${MAX_LAYERS:+--max-layers "$MAX_LAYERS"}
