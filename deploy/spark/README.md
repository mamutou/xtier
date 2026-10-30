# Running XTier on a single DGX Spark

Two scripts and nothing else:

```bash
# build once (CUDA toolkit must be a self-consistent install)
XTIER_BUILD_JOBS=20 bash tools/build_engine.sh /usr/local/cuda
python -m pip install --no-deps -e .

# start (cgroup-capped, in the background)
PY=$HOME/venvs/sglang/bin/python MEMCAP=118G MEMHIGH=108G bash deploy/spark/start_cgroup.sh

# start with DSPARK speculative decoding (~+87% e2e, needs the model's MTP head)
PY=$HOME/venvs/sglang/bin/python MAXREQ=1 bash deploy/spark/start_cgroup.sh --speculative-algorithm DSPARK
```

Full measurements behind these defaults: [../../docs/reports/single-spark-gb10.md](../../docs/reports/single-spark-gb10.md).
Reproducing from a bare box, including the Python environment: [REPRODUCE.md](REPRODUCE.md).

## Why the cgroup wrapper

The CPU expert engine and the GPU share one 121 GB pool and the checkpoint is ~156 GB. Without a cap,
exceeding memory makes the kernel spend all its time in reclaim: `ping` still answers, `sshd` accepts
TCP but never completes a handshake, and the box needs a power cycle. `start_cgroup.sh` puts the
server in a transient user scope with `MemoryMax`, so the scope dies instead of the machine.

Two wedges happened that way during this work. It is not a theoretical concern.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `M` | `$HOME/models/DeepSeek-V4-Flash-0731` | checkpoint |
| `PY` | `$HOME/venvs/sglang/bin/python` | interpreter with sglang + torch cu130 |
| `XTIER_COLD_PATHS` | `$HOME/models/xtier-cold` | NVMe cold-expert pool |
| `MAXREQ` | 1 | concurrent requests; raise only with warmup on (§ below) |
| `CONTEXT` / `CTXPOOL` | 262144 / 700000 | max context / KV pool tokens |
| `MEM` | 0.92 | `--mem-fraction-static`; 0.95 buys a bigger pool and leaves 3-4 GB free |
| `BATCH` | 2048 | chunked-prefill size |
| `XTIER_THREADS` | 16 | CPU workers. 19 on 20 cores is *slower* (starves the server) |
| `XTIER_CACHE_SLOTS` | 40 | experts held in GPU memory |
| `XTIER_COLD_FRACTION` | 0.66 | share of the pool kept cold on NVMe |
| `XTIER_FAST_LOAD` | 1 | stream experts into the engine (2.2 GB/layer instead of 3.8). **Required here** |
| `XTIER_DRAFT` | 1 | put the MTP draft layers on the CPU engine too. Required for DSPARK |
| `WARM` | 1 | keep server warmup. `0` restores `--skip-server-warmup` and caps you at 2 requests |
| `APIKEY` | unset | passed as `--api-key`; without it the endpoint is open to the whole network |
| `STOP_PATTERN` | unset | pkill pattern for competing IO (e.g. a checkpoint downloader) before loading |

## Verifying it took over

The log must show all layers on the CPU engine, and no fallback warnings:

```bash
grep -c "on the CPU engine" "$LOG"        # expect 43 for this checkpoint (+3 with DSPARK)
grep -i "xtier" "$LOG" | grep -i "fallback\|not taken over"
curl -s localhost:30000/get_server_info | grep -o '"max_total_num_tokens": [0-9]*'
```

## Traps that cost the most time

1. `XTIER_FAST_LOAD=1` or the load phase needs 163 GB and never finishes.
2. `--mem-fraction-static` 0.80 leaves no room for the KV cache here; use 0.92-0.95.
3. `systemd-run --unit=` must be unique, otherwise "Unit already loaded".
4. Without warmup (`WARM=0`), `--max-running-requests > 2` fails in
   `Prepare this stream/shape before graph capture`.
5. `CUDA_HOME` must point at a self-consistent toolkit, or tilelang JIT dies with
   `CUDA compiler and CUDA toolkit headers are incompatible`.
