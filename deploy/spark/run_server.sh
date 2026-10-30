#!/usr/bin/env bash
# XTier on ONE DGX Spark (GB10: 20 ARMv8.6 cores, 121 GB unified memory).
# Measured with DeepSeek-V4-Flash-0731 (MXFP4, 43 layers, 256 experts, top-6, 156 GB).
#
# Defaults below are the settings that measured best for a single interactive slot.
# Every value can be overridden from the environment; see deploy/spark/README.md.
set -eo pipefail

M=${M:-$HOME/models/DeepSeek-V4-Flash-0731}
COLD=${XTIER_COLD_PATHS:-$HOME/models/xtier-cold}
PY=${PY:-$HOME/venvs/sglang/bin/python}
mkdir -p "$COLD"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# GB10 is sm_121: flashinfer/tvm must JIT for it, not for the default arch list
export FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-12.1}
export TVM_FFI_GPU_BACKEND=cuda TVM_FFI_CUDA_ARCH_LIST=12.1

# Three JIT traps on a stock DGX Spark OS image (all three cost an evening if you hit them blind):
#   1. triton compiles a small C extension -> needs Python.h, which ships only in ~/include/python3.12
#      unless python3-dev is installed system-wide;
#   2. triton shells out to ninja, which usually lives in the venv, not in PATH;
#   3. tilelang compiles CUDA itself, and a pip nvidia-cu13 wheel ships nvcc 13.3 next to cccl headers
#      from a different version ("CUDA compiler and CUDA toolkit headers are incompatible"), so point
#      it at the self-consistent system toolkit instead.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CPATH=${CPATH:+$CPATH:}$HOME/include/python3.12
export PATH=$(dirname "$PY"):$HOME/.local/bin:$PATH
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$HOME/logs/triton-cache}
export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-$HOME/logs/flashinfer-cache}
mkdir -p "$TRITON_CACHE_DIR" "$FLASHINFER_CACHE_DIR"

# --- XTier: the ported NEON kernels. 20 cores total, so leave ~4 for the server:
# 19 workers starves the HTTP/scheduler threads and measures *slower* than 16.
export XTIER=${XTIER:-1}
export XTIER_THREADS=${XTIER_THREADS:-16}
export XTIER_MAIN_CPUS=${XTIER_MAIN_CPUS:-auto}
export XTIER_SPIN_COUNT=${XTIER_SPIN_COUNT:-1048576}
export XTIER_DYNAMIC=${XTIER_DYNAMIC:-2}
export XTIER_DOWN_BF16=${XTIER_DOWN_BF16:-1}
export XTIER_DOWN_GRAIN=${XTIER_DOWN_GRAIN:-24}
export XTIER_THP=${XTIER_THP:-0}
export XTIER_CACHE_SLOTS=${XTIER_CACHE_SLOTS:-40}
export XTIER_CACHE_EVERY=${XTIER_CACHE_EVERY:-2}
export XTIER_CACHE_SWAPS=${XTIER_CACHE_SWAPS:-32}
export XTIER_CACHE_LOG=${XTIER_CACHE_LOG:-200}
export XTIER_COLD=${XTIER_COLD:-1} XTIER_COLD_PATHS=$COLD
export XTIER_COLD_FRACTION=${XTIER_COLD_FRACTION:-0.66}
export XTIER_COLD_THREADS=${XTIER_COLD_THREADS:-8}
export XTIER_COLD_HALF_LIFE=${XTIER_COLD_HALF_LIFE:-50}
export XTIER_PREFILL_FP4=${XTIER_PREFILL_FP4:-1} XTIER_PREFILL_UNPACK=1
export XTIER_CPU_PREFILL_BATCH=${XTIER_CPU_PREFILL_BATCH:-1024} XTIER_GPU_PREFILL_MIN=1024
# Load experts straight into the CPU engine instead of materialising whole layers first: 2.2 GB per
# layer instead of 3.8 GB, i.e. 97 GB instead of 163 GB for this checkpoint. Mandatory on one Spark.
export XTIER_FAST_LOAD=${XTIER_FAST_LOAD:-1}
# Draft (MTP) layers go to the CPU engine too; without this a 3-layer draft cannot fit next to the
# target model. Only needed with --speculative-algorithm DSPARK.
export XTIER_DRAFT=${XTIER_DRAFT:-1}

# WARM=0 restores --skip-server-warmup. Do not: without the warmup pass tilelang never prepares the
# shapes it is later asked to capture, and any --max-running-requests above 2 dies in
# "RuntimeError: Prepare this stream/shape before graph capture".
EXTRA=(); [ "${WARM:-1}" = "1" ] || EXTRA+=(--skip-server-warmup)

exec "$PY" -m sglang.launch_server \
  --model-path "$M" --served-model-name "${SERVED:-DeepSeek-V4-Flash}" \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-30000}" --tp-size 1 --trust-remote-code "${EXTRA[@]}" \
  --max-running-requests "${MAXREQ:-1}" \
  --chunked-prefill-size "${BATCH:-2048}" --max-prefill-tokens "${BATCH:-2048}" \
  --context-length "${CONTEXT:-262144}" --max-total-tokens "${CTXPOOL:-700000}" \
  --mem-fraction-static "${MEM:-0.92}" --disable-shared-experts-fusion \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode full \
  --watchdog-timeout 2400 ${APIKEY:+--api-key "$APIKEY"} "$@"
