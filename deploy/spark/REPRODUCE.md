# Reproducing the single-Spark numbers from scratch

This is the checklist a third party needs. It is deliberately blunt about the one thing you cannot
download from this repo: a 156 GB checkpoint.

## 0. What you need

| | Requirement | Notes |
|---|---|---|
| Machine | 1 × DGX Spark (GB10), 121 GB unified memory | a 128 GB box will not hold this checkpoint; 96 GB boxes need a bigger cold fraction and will be slower |
| CPU | 20 × Armv8.6-A cores | `bf16`+`i8mm` for the fast kernel; without them the baseline NEON build runs at ~2.5× less throughput |
| Disk | ~1 TB free NVMe | 156 GB checkpoint + 275 GB cold-expert pool + room for the KV/offload scratch |
| OS | DGX OS / Ubuntu 24.04 aarch64 | `python3.12`, `build-essential`, `cmake` |
| CUDA | 13.0 toolkit (`/usr/local/cuda`) | must be *self-consistent*: a pip `nvidia-cu13` wheel's nvcc will fail with `CUDA compiler and CUDA toolkit headers are incompatible` |
| Model | `DeepSeek-V4-Flash-0731` (43 layers, 256 experts, top-6, MXFP4, 3 MTP layers, 156 GB) | see §2 |

## 1. Kernel-level acceptance needs no checkpoint

If all you want is to know whether the ARM port is correct on your box, stop here - these run in
seconds and prove the kernels:

```bash
XTIER_BUILD_JOBS=20 bash tools/build_engine.sh /usr/local/cuda
python tests/neon_check.py --threads 20            # expect 36/36 ok
XTIER_ISA=neon python tests/neon_check.py --threads 20
bash tools/isa_check.sh build                      # expect ISA-CHECK PASS
```

`tests/neon_check.py` builds a synthetic MoE layer and compares against a float64 numpy reference, so
it works on any aarch64 machine, not only a Spark.

## 2. The model

Every throughput number in [the report](../../docs/reports/single-spark-gb10.md) is for one specific
checkpoint, `DeepSeek-V4-Flash-0731`. It is not in this repository and not something we can redistribute.
Point `M=` at whatever you have:

```bash
export M=/path/to/your/checkpoint      # config.json + *.safetensors, MXFP4 experts
```

What carries over to a *different* checkpoint: the kernels, the build, the traps, the shape of the
results (prefill arithmetic-bound, decode ~half of the compute ceiling, memory/context trade-off).
What does not carry over: the exact tok/s, the layer count (43 here), the expert count (256), the cold
pool size, and the optimal slot/fraction pair - re-run the sweep in the report's §5.1 for a new model.

## 3. Python environment

`deploy/spark/requirements-sglang-0.5.21.txt` is a `pip freeze` of the venv that produced the report's
numbers (204 packages). On a slow link `pip` dependency resolution can spin for an hour, so install
with `--no-deps` from that file:

```bash
python3.12 -m venv ~/venvs/sglang && ~/venvs/sglang/bin/pip install -U pip
~/venvs/sglang/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
~/venvs/sglang/bin/pip install "flashinfer_python[cu13]==0.6.18" \
    --extra-index-url https://flashinfer.ai/whl/cu130
~/venvs/sglang/bin/pip install --no-deps -r deploy/spark/requirements-sglang-0.5.21.txt
~/venvs/sglang/bin/pip install --no-deps -e .
```

Sanity check before going further - this import must succeed:

```bash
~/venvs/sglang/bin/python -c "import sglang.srt.entrypoints.http_server; print('ok')"
```

If it fails with `ImportError: "".join(failed)`, you are missing **torchvision**; transformers builds
that message from an empty list, so the real cause is hidden.

Version notes:

- **sglang 0.5.21 is required.** 0.5.17 routes the DSPARK draft decode into the prefill-shaped sparse-MLA
  kernel (`num_tokens > 64` assertion) and DSPARK cannot run at all; 0.5.22 needs `torch==2.14.1`, which
  has no cu130 aarch64 wheel.
- `transformers==5.12.1` / `tokenizers==0.22.2` is what the working venv resolved to; do not "upgrade" them.

## 4. Build the engine

```bash
cd <repo>
XTIER_BUILD_JOBS=20 bash tools/build_engine.sh /usr/local/cuda
ls build/    # libxtier_engine.so, ..._neonbf16.so, ..._neon.so, libxtier_mailbox.so, libxtier_tables.so
```

Expect `[xtier] CPU kernels: neonbf16` and `1 worker nodes, 20 threads` in the output. On a non-BF16 ARM
core you get `neon` and roughly 2.5× lower kernel throughput - everything else is identical.

## 5. First load

```bash
STOP_PATTERN='fetch.py' PY=~/venvs/sglang/bin/python MEMCAP=118G MEMHIGH=108G \
  bash deploy/spark/start_cgroup.sh --speculative-algorithm DSPARK
```

The first start writes the cold-expert pool to NVMe and is therefore slower than every later one
(minutes, not seconds, per layer group). Expected shape of a healthy log:

```
xtier: layer 28: 77 of 256 experts in RAM, 256 records (3.4 GB) on 1 drives written in 1.5 s
xtier: layer 28: 256 experts (H 4096, I 2048, mxfp4) on the CPU engine in 1.83 s
Load weight end. elapsed=45.15 s
```

Verify:

```bash
grep -c "on the CPU engine" "$LOG"                       # 43, or 46 with DSPARK (3 draft layers)
curl -s localhost:30000/get_server_info | grep -o '"max_total_num_tokens": [0-9]*'
```

## 6. What to compare against

| measurement | report value | how tolerant |
|---|---|---|
| kernel throughput, m=512 | 889 GF/s (`tests/real_layer_check.py`) | ±10 % for NVMe/thermal; 343 GF/s means you fell back to the baseline build |
| single-stream decode, warm | 20.4 e2e / 30.0 steady tok/s | ±20 %; below ~12 tok/s means speculation or the hot-expert slots are not working |
| 8 concurrent aggregate | 28.0 tok/s | ±20 % |
| prefill, uncached, 9K prompt | 63 tok/s | ±25 %; **far** higher means you hit the prefix cache, see below |
| KV pool, DSPARK, 1M ctx | 973,056 tokens | check with `/get_server_info`, it depends on `mem-fraction` |

Three things that will otherwise waste a day:

1. **The prefix cache fakes a 40× speed-up.** Re-sending the same prompt reported 2,502 tok/s on a 9k
   prompt. Salt the first tokens (`bench/spark_lengths.py` does it by default).
2. **A cold server is ~7× slower** (38 vs 281 tok/s on a 1.1k prompt) because the hot-expert slots start
   empty. Warm up before timing.
3. **Concurrency above 2 needs server warmup.** Without it, CUDA-graph capture fails in
   `Prepare this stream/shape before graph capture`.

## 7. If it breaks

| symptom | cause | fix |
|---|---|---|
| load phase never ends, memory climbing past 120 GB | `XTIER_FAST_LOAD` off | `XTIER_FAST_LOAD=1`: 2.2 GB/layer instead of 3.8 |
| `loaded weights leave no GPU memory for the KV cache` | `mem-fraction` too low | 0.92-0.95 on this box |
| `Unit xtier-spark.scope was already loaded` | fixed systemd unit name | use the timestamped name in `start_cgroup.sh` |
| machine stops answering ssh during a long prefill | no cgroup cap | always launch through `start_cgroup.sh` |
| `RuntimeError: xtier: 3 draft MoE layers on XTier without XTIER_DRAFT=1` | draft placement | `XTIER_DRAFT=1` |
