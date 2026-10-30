# One DGX Spark (GB10): deployment and benchmark report

**Scope.** Everything here was measured on a single NVIDIA DGX Spark (GB10) running the ARM port of
XTier with SGLang 0.5.21 and `DeepSeek-V4-Flash-0731`. It is the reference document for "how fast is
this on one Spark, and what can you actually serve with it". The port itself (kernels, dispatch,
build) is described in [../arm-port.md](../arm-port.md).

Short version: **the port works, decode is ~52 % of a two-node TP2 deployment on half the hardware,
and prefill is the wall** - 63-70 tok/s regardless of prompt length, which is what limits long-context
use far more than memory does.

## Abstract

- The port works. One DGX Spark (GB10, 121 GB unified memory) serves a 156 GB `DeepSeek-V4-Flash-0731` - 43 layers plus
  3 MTP layers - with every routed expert computed by the ported ARM NEON kernels.
- **Decode: 20-31 tok/s** single-stream with DSPARK speculative decoding, and **28 tok/s aggregate at 8 concurrent**
  requests.
- That is **52 % of a two-node TP2 deployment (38.95 tok/s at c1) on half the hardware** - slightly better than linear,
  because TP2 pays an interconnect round-trip inside every layer while one node keeps all experts on its own CPU engine.
- **Prefill is the wall: 63-70 tok/s, flat in prompt length**, because it is arithmetic-bound on the CPU expert engine
  (~0.9 TFLOP/s at m=2048) rather than length-bound. A 900K-token request therefore costs **~3.7 hours**.
- Memory and context trade off hard: the KV pool holds **973K tokens with speculation and 4.0M without** (three
  concurrent 1M requests), so the 18.8 GB draft is what decides how much context the box can hold.
- **Recommended production configuration: one slot, 262K context, `mem-fraction 0.92`**, which leaves ~7 GB free. At 0.95
  only 3-4 GB remain, and a long prefill then wedges the whole machine (observed twice; needs a power cycle).

## 中文摘要

- 移植成功：单台 Spark（GB10，121 GB 统一内存）跑起 156 GB 的 `DeepSeek-V4-Flash-0731`，43 层 + 3 层投机的 MoE，全部经 XTier 的 ARM NEON 内核计算。
- 单流解码 **20-31 tok/s**（短上下文，DSPARK 投机解码），是双 Spark TP2 部署（38.95 tok/s @ c1）的 **52%**，而硬件只有一半。
- **预填充是瓶颈**：绕过前缀缓存实测 **63-70 tok/s**，和不随长度变化——因为它受 CPU 引擎算力限制。900K 上下文单个请求要 **约 3.7 小时**，没有生产价值。
- 并发有效：8 并发聚合 **28 tok/s**（单并发 ~15），再往上被 CPU 核数卡死；19 线程反而比 16 线程慢。
- 显存与上下文是硬取舍：开投机解码 KV 池 **97.3 万 token**（装不下一个 1M 请求），关掉投机 **400 万 token**（可并行 3 个 1M）。
- 推荐生产配置：单并发 + 262K 上下文 + `mem-fraction 0.92`，留 7 GB 余量；`0.95` 只剩 3 GB，长 prefill 会把整机拖死（实测两次，需断电）。

## 1. Test bed

| | |
|---|---|
| Machine | 1 × NVIDIA DGX Spark, GB10 |
| CPU | 20 × Armv8.6-A cores (`bf16`, `i8mm`, `sve2`, no SMT), NEON |
| Memory | 121 GB unified LPDDR5X, shared by CPU and GPU |
| GPU | GB10, tensor core arch sm_121 |
| Storage | NVMe, 3.7 TB, ~1.35 TB used during these runs |
| OS | DGX OS (Ubuntu-based), CUDA 13.0 at `/usr/local/cuda` |
| Model | `DeepSeek-V4-Flash-0731`: 43 layers, 256 experts, top-6, MXFP4, `swiglu_limit` 10, 3 MTP layers, 156 GB |
| Cold-expert pool | 275 GB on NVMe |
| SGLang | 0.5.21 (torch 2.13.0+cu130, sglang-kernel 0.4.7, flashinfer 0.6.18) |

## 2. What the port changed

Summarised from [../arm-port.md](../arm-port.md); all of it is in this tree.

| Area | Change |
|---|---|
| Kernels | NEON FP4 unpack (14 instructions per 32 weights), `dot8` with `vbfdotq_f32` on Armv8.6 and a widening `vfmaq_f32` fallback, NEON `gate_rows`/`down_rows`/`down_rows_bf16`/`read_rows` |
| Build | one `.so` per ISA (`libxtier_engine_neonbf16.so`, `libxtier_engine_neon.so`, plus the x86 ones); nothing from one ISA can leak into another |
| Dispatch | AArch64 picks the library from `AT_HWCAP2` BF16, `XTIER_ISA=neon|neonbf16` forces one |
| Topology | no NUMA sysfs on ARM: single node from `/sys/devices/system/cpu/online`, `mbind` failure is non-fatal |
| Mailbox | `_mm_pause()` → `yield` |
| Python | works on released SGLang 0.5.x, not only `main` (`MoeFlags.in_speculative_scope`, `MoeRunnerConfig` fields, `swiglu_limit` fallback) |

Kernel throughput on real checkpoint weights (one layer, 256 experts, `tests/real_layer_check.py`):

| | m=1 | m=64 | m=512 |
|---|---|---|---|
| NEON + BFDOT (`neonbf16`) | 849 µs / 356 GF/s | 27.6 ms / 700 GF/s | 174 ms / **889 GF/s** |
| baseline NEON (any Armv8-A) | 3,416 µs / 88 GF/s | - | 451 ms / 343 GF/s |

~0.9 TFLOP/s is the ceiling that explains every number in this report: a token needs ~26 GFLOP of
expert work (43 layers × 6 of 256 experts), so ~34 tok/s of decode is the arithmetic limit and
everything else is overhead.

## 3. Correctness

Both checks run without a GPU and without loading a whole checkpoint.

- `tests/neon_check.py` - synthetic MoE layer, 4 formats × 9 token counts, numpy float64 reference:
  **36/36 pass** on both `neonbf16` and forced-baseline `XTIER_ISA=neon`, RMS 1.4e-7 … 2e-6.
- `tests/real_layer_check.py` - layer 0 of the real checkpoint, per-expert dequantisation, m=1…64:
  RMS **9.3e-8 … 4.6e-6** against the reference (identical results on the baseline kernel).
- `tools/isa_check.sh` - the baseline library contains **0** BFDOT instructions, the BF16 library
  contains 4,416; no AVX512/BFMMA leaks into either.

End-to-end sanity on the live server: factual Q&A and arithmetic answered correctly.

## 4. Deploying on one Spark

```bash
# 1. build the engine (CUDA toolkit path may differ)
cd <repo> && XTIER_BUILD_JOBS=20 bash tools/build_engine.sh /usr/local/cuda
python -m pip install --no-deps -e .

# 2. run it (see deploy/spark/ for both scripts)
STOP_PATTERN='fetch.py' PY=$HOME/venvs/sglang/bin/python \
  MEMCAP=118G MEMHIGH=108G bash deploy/spark/start_cgroup.sh
```

`deploy/spark/run_server.sh` carries the full environment. The four things that are not obvious:

1. **`XTIER_FAST_LOAD=1` is mandatory.** Without it each layer is materialised whole before the
   experts are handed to the CPU engine: 3.8 GB per layer, 163 GB for 43 layers, i.e. more than the
   machine has. With it experts stream straight into the engine: **2.2 GB per layer, ~97 GB total**.
2. **`--mem-fraction-static` has to be high on this box (0.92-0.95).** The CPU engine's unified memory
   already counts as used memory for CUDA, so the usual 0.80 leaves "no GPU memory for the KV cache".
3. **The scope needs a unique unit name** (`--unit=xtier-spark-$(date +%s)`); a fixed name fails with
   `Unit xtier-spark.scope was already loaded`.
4. **Do not pass `--skip-server-warmup`** unless you also keep `--max-running-requests 2`
   (see §5.3).

The cgroup wrapper is not optional on a 121 GB box running a 156 GB model: two wedges (whole machine
unresponsive, `ping` fine, `sshd` unable to fork, needs a power cycle) both happened without it.

## 5. Results

### 5.1 Placement sweep (no speculation)

18 configurations, 3 rounds × 256 tokens each, `XTIER_FAST_LOAD=1`, 16 workers, 43 layers. GPU expert
slots and the RAM-resident fraction are the two knobs; `mem-fraction` has to move with them.

| slots | cold fraction | mem-frac | e2e tok/s | steady tok/s |
|---|---|---|---|---|
| 8 | 0.75 | 0.93 | 7.53 | 8.90 |
| 32 | 0.70 | 0.93 | 8.82 | 10.34 |
| 48 | 0.70 | 0.93 | 9.51 | 11.58 |
| 48 | 0.62 | 0.93 | 10.64 | 12.03 |
| 56 | 0.62 | 0.93 | 10.48 | 12.32 |
| 48 | 0.58 | 0.90 | *failed: no GPU memory left for KV cache* | - |
| 48 | 0.58 | 0.93 | 10.52 | 12.14 |
| **56** | **0.58** | **0.95** | **11.49** | **12.32** |
| 64 | 0.62 | 0.97 | 10.01 | 12.28 |
| 48 | 0.55 | 0.96 | 10.47 | 12.00 |

Two conclusions: GPU slots saturate around 56, and the RAM-resident fraction is the bigger lever but
only pays off if `--mem-fraction-static` is raised with it. Everything plateaus at ~12.3 tok/s, which
is the arithmetic ceiling above (~34 tok/s) minus the CPU↔GPU handoff.

### 5.2 Speculative decoding (DSPARK)

DSPARK (the model's own MTP head as draft, `--speculative-algorithm DSPARK`) is the single biggest win,
`XTIER_DRAFT=1` is required so the 3 draft MoE layers also live on the CPU engine (they do not fit on
the GPU next to the target model: target 87.6 GB, draft 6.1 GB, 18.8 GB after warm-up).

512-token requests, 2 × 512 warm-up discarded, 4 measured:

| configuration | e2e tok/s | steady tok/s | accept len / rate |
|---|---|---|---|
| no speculation (SGLang 0.5.17) | 10.91 | 12.73 | - |
| DSPARK block=7 | 11.08 | 16.31 | 3.62 / 0.38 |
| DSPARK block=3 | 16.10 | 21.97 | 2.80 / 0.60 |
| DSPARK block=5 (cold pool) | 17.28 | 21.85 | 2.70 / 0.34 |
| **DSPARK block=5 (warm)** | **20.41** | **29.95** | **4.50 / 0.70** |
| EAGLE topk=1 | - | - | not usable on 0.5.21 |

+87 % e2e, +135 % steady over no speculation. Block 7 drafts too much (acceptance collapses to 0.38),
block 3 is safe but a notch slower; 5 is the sweet spot.

**SGLang version is a hard gate.** On 0.5.17 DSPARK aborts with

```
tvm.error.InternalError: Check failed: num_tokens > 64 (5 vs. 64) : Decode (num_tokens <= 64) must go
through sparse_mla_sm120_decode_dsv3_2 or sparse_mla_sm120_decode_dsv4; got num_tokens=5
```

- the draft decode was routed into the prefill-shaped sparse-MLA kernel. All three
  `--dsa-decode-backend` values reproduce it, with and without CUDA graphs. 0.5.21 fixes it (the log
  shows `AutoTuner: Tuning sparse_mla_sm120_decode_dsv4` at startup, which never happened on 0.5.17).

Version map for ARM: 0.5.18-0.5.20 have no `deepseek_v41_vit.py`; **0.5.21** is the only release that
has both the fixed sparse-MLA kernels and the vision tower with `torch==2.13.0` (the last version with
cu130 aarch64 wheels); 0.5.22 wants `torch==2.14.1`, which has no aarch64 cu130 wheel.

### 5.3 Batching (and a bug worth knowing about)

`--max-running-requests` above 2 died in

```
RuntimeError: Prepare this stream/shape before graph capture
```

for every value tried (8, 16), with and without
`--speculative-dspark-align-verify-tokens-to-graph-tier`. The cause was our own launcher: it passed
**`--skip-server-warmup`**. Without the warm-up pass tilelang never prepares the shapes it is later
asked to capture, so any capture set beyond the trivial sizes fails. Drop the flag and concurrency
works - no graph-tier flag needed.

8 slots, 1M context, 384-token requests, 2 rounds each:

| XTier workers | c1 | c4 | c8 |
|---|---|---|---|
| 16 | 14.9 | 25.7 | **28.0** |
| 19 | 16.8 | - | 21.4 |

Batching buys ~+88 % aggregate over c1 and then plateaus: with 8 sequences in flight the GPU has more
to do but the CPU engine becomes the wall. 19 workers on 20 cores is *worse* than 16 because the
HTTP/scheduler threads get starved - the same reason 18 workers lost to 16 in §5.1.

### 5.4 Prefill is the wall

Single slot, `--context-length 921600`, pool 921,600 tokens, DSPARK on, 16 workers. Every prompt starts
with a unique sentence so nothing can be served from the radix cache:

| prompt | TTFT | prefill | decode |
|---|---|---|---|
| 9,082 tok | 143.9 s | **63 tok/s** | 16.0 tok/s |
| 36,246 tok | 518.6 s | **70 tok/s** | 17.1 tok/s |

Prefill is flat in length because it is arithmetic-bound on the CPU expert engine, not length-bound.
Decode barely changes with context either (16-17 tok/s), because attention cost is small next to the
expert work.

Two measurement traps that cost real time here:

- **The radix cache fakes a 40× speed-up.** Re-sending the same text reported **2,502 tok/s** on a 9k
  prompt. That is a cache hit. Salt the first tokens or you will publish a fantasy number.
- **A cold server is ~7× slower**: 38 tok/s cold versus 281 tok/s warm on a 1.1k prompt, because the
  hot-expert slots start empty and every expert read comes off the drive. Warm up before timing.

Extrapolated at 70 tok/s:

| context | prefill | verdict |
|---|---|---|
| 8K | ~2 min | batch jobs only |
| 32K | ~8.5 min | batch jobs only |
| 128K | ~30 min | not practical |
| 900K | **~3.7 h** | single request costs half a day |

### 5.5 KV pool and what it costs

fp8 KV, target 87.6 GB + draft 18.8 GB:

| configuration | `max_total_num_tokens` | concurrent 1M requests |
|---|---|---|
| DSPARK on, `mem-fraction 0.95` | **973,056** | 0 (one 1M does not fit; ~1 at 900K) |
| DSPARK off, `mem-fraction 0.95` | **4,004,608** | 3 |
| DSPARK on, `mem-fraction 0.92`, ctx 262K | 529,664 | - (7 GB RAM left free) |

That is the real price of speculation on one box: the 18.8 GB draft plus its KV turns a 4M-token pool
into a 973K one. At `mem-fraction 0.95` with a 900K pool the machine reports only **3-4 GB available**,
which is exactly the condition under which a 512K prefill wedged it.

### 5.6 Where the GPU actually is

During 500-token single-stream generation: util 35-86 % (mean ~64 %), SM clock 2463-2476 MHz (already
pinned at max), **power 30-36 W**, CPU ~60 % user / 36 % idle. Both sides idle because they alternate:
one decode step is GPU (attention, dense) → CPU (this layer's experts) → GPU (combine), and the GPU
waits through the CPU half. The way to fill it is more sequences in flight, i.e. §5.3 - and that then
plateaus on CPU, not on GPU.

## 6. One Spark versus two

Decode tok/s at concurrency 1 / 4 / 8, same model family:

| rig | c1 | c4 | c8 | note |
|---|---|---|---|---|
| **single GB10, SGLang 0.5.21 + XTier + DSPARK** | **20.5-30.0** | - | 28.0 agg | this report |
| dual GB10, vLLM TP2, DSPARK k=3 | 38.95 | 71.98 | 89.92 | two nodes, RoCE |
| dual GB10, vLLM TP2, DSPARK k=5 | 38.42 | 64.97 | 81.90 | |

Single-stream is **~52 % of the dual-Spark number on half the hardware** - slightly better than linear,
because TP2 pays an interconnect round-trip inside every layer while one node keeps all experts on its
own CPU engine. Batching is where the second node pays off, and that is the gap this box cannot close
(the CPU engine, not the GPU, is the wall at c8).

## 7. Production verdict

| workload | verdict |
|---|---|
| Interactive chat, short prompts | **usable but not pleasant**: TTFT ~20 s for an uncached 1K prompt, then 20-30 tok/s |
| Batch/offline, prompts ≤ 8K | **usable**: ~2 min prefill, 16-30 tok/s decode |
| 32K prompts | batch only, ~8.5 min each |
| 128K | not practical (~30 min) |
| 900K | no: ~3.7 h prefill |

Recommended, stable configuration (what `deploy/spark/` defaults to):

```bash
MAXREQ=1 CONTEXT=262144 CTXPOOL=700000 MEM=0.92 BATCH=2048 \
XTIER_CACHE_SLOTS=40 XTIER_COLD_FRACTION=0.66 XTIER_THREADS=16 \
XTIER_FAST_LOAD=1 XTIER_DRAFT=1 \
bash deploy/spark/start_cgroup.sh --speculative-algorithm DSPARK
```

Throughput-oriented variant (needs warmup, 8 slots): add `MAXREQ=8 CONTEXT=1048576 CTXPOOL=1048576
MEM=0.95`, which gives 28 tok/s aggregate at c8 but leaves only 3-4 GB free and a 973K-token pool.

If you need reproducible long-context serving, this is not the machine for it: go back to GPU-native
prefill (two nodes with TP2), where prefill is not computed on ARM cores.

## 8. Reproducing

From a bare box, including the Python environment and the model caveat: `deploy/spark/REPRODUCE.md`.
Quick version:

```bash
# correctness, no GPU, no full checkpoint
python tests/neon_check.py --threads 20
XTIER_ISA=neon python tests/neon_check.py --threads 20
python tests/real_layer_check.py --model <ckpt> --layer 0
bash tools/isa_check.sh build

# throughput of a running server
python bench/spark_decode.py                       # e2e + server-reported steady
python bench/spark_lengths.py --warm               # TTFT / prefill / decode per length
CONC=8 python bench/spark_concurrency.py           # aggregate at 8 concurrent
```

## 9. What is not done

- **Vision** is blocked on the checkpoint, not on this port. SGLang 0.5.21 has the tower
  (`models/deepseek_v41_vit.py`, and `deepseek_v4.py` builds `ViT` when
  `config.model_type == "deepseek_v41" and config.vision_n_layers > 0`). The experimental vision
  checkpoint ships `vision_n_layers=32` but declares `model_type: deepseek_v4`, so it is served as
  text-only; switching the field does activate the branch, which then asks for `image_token_id` and
  the checkpoint's tokenizer has **no image special tokens at all** (`added_tokens_decoder` empty,
  `special_tokens_map` empty, no image handling in the chat template). Guessing ids would map image
  patches to arbitrary embedding rows and answer confidently wrong, so nothing is remapped.
- **EAGLE** does not run on 0.5.21 for this model: `speculative_eagle_topk is None` unless set, and
  with it set the XTier hook needs `qsa.sparse_attn.pack_qsa_prefill_kv`,
  `models.deepseek_v4_replay_tail` and `tilelang.backend.target`, none of which exist in that release.
- **x86 kernels were not rebuilt** in this work; the x86 sources are unchanged and only
  diff/preprocessor-checked.
- The `m=4` slot being slower than `m=2` in the synthetic layer benchmark (5.3 ms vs 1.7 ms) is still
  unexplained.
