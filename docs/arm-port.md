# ARM（AArch64 / NEON）移植

CPU 专家引擎原来只有 x86 的 AVX512 BF16 和 AVX2/FMA 两套内核。这份文档记录把它搬到 AArch64 时改了什么、
为什么这么改、以及在 DGX Spark（GB10，20 核）上实测到什么。

## 结论先说

| | 状态 |
|---|---|
| 内核 | `libxtier_engine_neonbf16.so`（Armv8.6-A BF16，`vbfdotq_f32`）与 `libxtier_engine_neon.so`（任何 Armv8-A，BF16 加宽 + FMA），`libxtier_engine.so` 运行时二选一 |
| 数值 | 合成层对拍（float64 反量化参考，两种 FP4 格式 × 9 种 token 数 × 2 个 scale 量程 = 36 例）RMS 相对误差 1.4e-7…2e-6（真实量程），全部通过；两个 ARM 构建都通过 |
| 真实检查点 | `~/models/DeepSeek-V4-Flash-0731` 第 0 层（H=4096、I=2048、256 专家、top-6、MXFP4）对 float64 参考：m=1…64 rms 9e-8…4.6e-6；两个 ARM 构建在真实权重上互相一致到 1e-7（decode） |
| 速度（GB10，20 核，单层 H=5120 / I=2304 / top-6，64 专家，生产开关） | decode m=1 1.29 ms/层（330 GFLOP/s），prefill m=512 242 ms（899 GFLOP/s），m=2048 951 ms（914 GFLOP/s） |
| 相对基线 NEON（无 BFDOT） | BFDOT 构建 decode 快 3.0×、prefill 快 2.6× |
| 信箱（`libxtier_mailbox.so`） | 编译链接通过、`ctypes` 可加载（CUDA 依赖解析正常）；端到端握手需要真实检查点，本机没有，未跑 |

## 改了什么

x86 专有面只有四个文件，平台层（futex、NUMA、线程池、信箱协议、冷专家池、行表）本来就与指令集无关。

| 文件 | 改动 |
|---|---|
| `csrc/engine/engine.cpp` | 新增 `XTIER_NEON` 分支：解包、`dot8()`、`gate_rows` / `down_rows` / `down_rows_bf16` / `read_rows` 四组内核；`_mm_pause()` → `XTIER_PAUSE()`（AArch64 上是 `yield`）；无 NUMA sysfs 时不再臆造 x86 的"4 节点 × 16 核"拓扑；`mbind` 在单节点机器上失败不再致命 |
| `csrc/engine/dispatch.cpp` | AArch64 用 `getauxval(AT_HWCAP2)` 的 BF16 位选内核库，`XTIER_ISA=neon\|neonbf16` 可强制；x86 行为不变 |
| `csrc/mailbox/mailbox.cpp` | 两处自旋提示改为 `XTIER_PAUSE()` |
| `csrc/CMakeLists.txt` | `CMAKE_SYSTEM_PROCESSOR` 匹配 `aarch64` 时构建 `neonbf16` + `neon`，否则仍是 `avx512` + `avx2` |

`XTIER_DOWN_R4`（AVX512 的行数微内核开关）只对 AVX512 构建有意义，NEON 内核按 token 数自己选行数。

## 内核设计（为什么这样写 NEON）

AArch64 的向量寄存器是 32 个 128 位、没有 SMT、也没有 `vpermt2b`/`dpbf16` 这类 512 位指令，所以照抄 x86 的
分块没有意义，关键是让**解包**便宜。

- **解包**：16 字节压缩 FP4 = 32 个权重。低/高半字节用 `vandq_u8` + `vshrq_n_u8` 分开，`vzip1q_u8/vzip2q_u8`
  还原成权重序（权重 2i 在低半字节），每个 16 权重的块用 `vqtbl2q_u8` 在"16 个低字节 + 16 个高字节"的 32 字节
  表里一次查出两个字节半边，再 `vzip1q_u8` 拼回 BF16 通道。**32 个权重 14 条指令**，不用 gather、不用 FP32 表。
- **点积**：`dot8()` —— Armv8.6 上就是一条 `vbfdotq_f32`（8 个 BF16 乘加进 FP32 累加器）；基线构建退化成
  `vshll_n_u16` 把 BF16 精确加宽到 FP32 再 `vfmaq_f32`。同一份源码，靠 `__ARM_FEATURE_BF16_VECTOR_ARITHMETIC`
  切成两个库，运行时按 HWCAP2 选。
- **Down 的中间值**：保持 AVX512 内核的 FP32 中间值——解包出的 BF16 权重加宽到 FP32（精确，等于 `asfloat()`），
  与 FP32 中间值做 FMA。所以 `XTIER_DOWN_BF16=0` 的数值路径和 x86 一样是"BF16 权重 × FP32 激活"。
- **寄存器预算**：32 个向量寄存器，所以 Gate 在 M≤4 时两行一起做（`2*R*M` 累加器 ≤16），M>4 时一行；
  Down 在 M≤2 时 4 行、M≤4 时 2 行、更大 1 行；输入 BF16 直接在点积处从 L1 重载，不占常驻寄存器。
- **归约**：4 个 FP32 通道各累加 `H/4` 项，最后 `vaddvq_f32`。和 x86 一样，只有求和顺序不同。

## 构建与验证

```bash
tools/build_engine.sh /usr/local/cuda        # aarch64 上产出 neonbf16 / neon / dispatch / mailbox / tables
tools/isa_check.sh                           # 基线库不得含 BFDOT/BFMMLA，BFDOT 库必须有
python tests/neon_check.py --threads 20      # 合成层：数值对拍 + 各 token 数耗时（不需要检查点）
XTIER_ISA=neon python tests/neon_check.py --threads 20   # 强制基线 NEON 构建
XTIER_DYNAMIC=2 XTIER_DOWN_BF16=1 python tests/neon_check.py --threads 20   # 生产开关
python tests/real_layer_check.py --model ~/models/DeepSeek-V4-Flash-0731 --layer 0 --experts 64   # 真实检查点一层
python tests/real_layer_check.py --model ... --save /tmp/a   # 另一个进程用 XTIER_ISA 跑同样的用例后：
python tests/real_layer_check.py --model ... --compare /tmp/a /tmp/b      # 两个构建逐元素对比
```

`tests/neon_check.py` 是给新 ISA 用的验收：两种 FP4 格式、9 种 token 数（1…2048，覆盖 decode / verify /
prefill 三条路径）、每组都用 numpy 在 float64 里反量化同一份权重算参考。门限看 **RMS 相对误差**（1e-5），
逐元素最大误差只打印：单个输出元素的抵消会把 max_rel 抬到 1e-4 量级，那不是错。

## 实测（DGX Spark GB10，20 核 Cortex-X925 + A725，单 NUMA 节点）

单层 DeepSeek V4.1 Flash 1017 形状（H=5120、I=2304、top-6、64 个专家 = 1.13 GB 专家权重），`--threads 20`，
每格是单次调用的中位数耗时 / 折算吞吐：

| token 数 | 默认（DYNAMIC=0，FP32 Down） | 生产（DYNAMIC=2，DOWN_BF16=1） | 基线 NEON（生产开关） |
|---:|---:|---:|---:|
| 1 | 2417 µs / 176 GF/s | **1286 µs / 330 GF/s** | 3923 µs / 108 GF/s |
| 2 | 3139 µs / 271 GF/s | 1737 µs / 489 GF/s | 2655 µs / 320 GF/s |
| 4 | 5487 µs / 310 GF/s | 5299 µs / 321 GF/s | 5978 µs / 284 GF/s |
| 8 | 9954 µs / 341 GF/s | 7058 µs / 481 GF/s | 14912 µs / 228 GF/s |
| 64 | 49251 µs / 552 GF/s | 36965 µs / 735 GF/s | 85140 µs / 319 GF/s |
| 512 | 344880 µs / 631 GF/s | 241974 µs / 899 GF/s | 625332 µs / 348 GF/s |
| 2048 | 1372132 µs / 634 GF/s | 951537 µs / 914 GF/s | 2494668 µs / 349 GF/s |

- decode（m=1）每层 1.29 ms，按 40 层线性折算 ≈ 51 ms/步 ≈ 20 tok/s——这是**整层都跑在 CPU 上**的上限；
  混合推理只有 PCIe/CX7 另一侧的那部分专家走 CPU，所以实际 decode 速度取决于热专家比例，不是这个数。
- `XTIER_DYNAMIC=2` + `XTIER_DOWN_BF16=1` 在 GB10 上是必须开的：decode 快 1.9×，prefill 快 1.4×。
- m=4 比 m=2 慢（5.3 ms 对 1.7 ms）在这台机器上可复现，x86 上没观察过；分组/领取逻辑没改，属于待查项。
- 与大核小核混排有关：GB10 是 10×X925 + 10×A725，静态均分行数会让 A725 拖住 barrier，这是 `XTIER_DYNAMIC`
  存在的理由；只用大核可以用 `XTIER_NODE_CORES` + `--threads 10` 对照。

## 真实权重（DeepSeek-V4-Flash-0731，第 0 层，MXFP4）

`tests/real_layer_check.py` 直接读检查点里的 `w1/w3/w2.weight` + `.scale`（int8 打包 + 每 32 个权重一个 UE8M0
字节，正好是引擎的 MXFP4 布局），所以不经 GPU、不经 SGLang 就能在真实权重上验证。第 0 层是 H=4096、I=2048、
256 个路由专家、top-6、`swiglu_limit=10`、无全局 scale（g=1）——顺带覆盖了 `run<4096,2048>` 这组特化模板。

| token 数 | BFDOT 构建 | 基线 NEON | 对 float64 参考的 rms |
|---:|---:|---:|---:|
| 1 | 849 µs / 356 GF/s | 3416 µs / 88 GF/s | 9.3e-8 |
| 2 | 1431 µs / 422 GF/s | 5970 µs / 101 GF/s | 9.6e-8 |
| 4 | 2447 µs / 494 GF/s | 7542 µs / 160 GF/s | 4.9e-7 |
| 8 | 5542 µs / 436 GF/s | 8748 µs / 276 GF/s | 9.4e-8 |
| 64 | 27598 µs / 700 GF/s | 62799 µs / 308 GF/s | 4.6e-6 |
| 512 | 174019 µs / 889 GF/s | 451136 µs / 343 GF/s | — |

耗时是 `XTIER_DYNAMIC=2 XTIER_DOWN_BF16=1` 且 `--ref-tokens 0` 的那一次：带参考的跑法会让参考实现把权重挤出
缓存，m=2/8 的耗时翻倍（8000 / 9217 µs），报价时用不带参考的那次。

256 个专家（整层 4.3 GB 常驻）时 m=1 336 GF/s、m=512 814 GF/s，比 64 专家略低——每步触到的权重更散，和 x86 上
的现象一致。

两个 ARM 构建在真实权重上互相一致：m=1…4 逐元素最大差 1e-6、rms 1e-7；m=64/512 的差升到 1e-5…1.8e-5，是 bfdot
成对求和与加宽 FMA 顺序不同、在真实权重的抵消下被放大，不是错（两者各自对 float64 参考的误差都在 1e-6 内）。

## 坑

1. **MXFP4 的 UE8M0 指数 254 = 2^127，乘上 FP4 码就超出 BF16**：内核表里那一格是 inf，真实检查点离得很远
   （DeepSeek 用 2^-9…2^0）。合成测试别用全量程 scale 跑 MXFP4，否则只会得到 NaN。
2. **AArch64 没有 SMT**：`node_cores()` 的"物理核 + 兄弟线程"模型退化成"只有物理核"，`XTIER_NODE_CORES`
   配得比 `threads/节点数` 小会直接报错，这是预期行为。
3. **单 NUMA 节点**：`/sys/devices/system/node/online` 只有 `0`，`mbind` 失败无意义，所以只在一个节点时才忽略失败；
   多节点机器仍然报错（权重分片绑错节点的代价比启动失败大）。
4. **`_mm_pause()` → `yield`**：worker 自旋和信箱 dispatcher 两处。ARM 的 `yield` 是提示指令，语义等价，
   但少了 x86 的"让出超线程"作用（本来也没有 SMT）。
5. **数值不是逐位相同**：x86 AVX512 用 `vdpbf16ps` 成对累加、AVX2 用加宽 FMA、NEON 用 `bfdot` 或加宽 FMA，
   三者的求和顺序都不同。`XTIER_DOWN_BF16=1` 时参考实现要和内核一样先把中间值舍入到 BF16，否则会有
   ~1e-3 的"假失败"（那是 BF16 的 2^-8，不是内核错）。

## Single Spark (GB10) run

One DGX Spark (GB10, 121 GB unified, 20 cores) cannot hold DeepSeek-V4-Flash (156 GB, 43 layers x
256 experts, MXFP4) in memory, so a cold tier on NVMe is mandatory. Measured recipe:

```bash
export CUDA_HOME=/usr/local/cuda                 # the pip nvidia-cu13 wheel ships nvcc 13.3 with
                                                 # mismatched cccl headers -> tilelang dies with
                                                 # "CUDA compiler and CUDA toolkit headers are incompatible"
export CPATH=$HOME/include/python3.12            # triton JITs driver.c; no system python3-dev here
export PATH=$HOME/venvs/sglang-dspark/bin:$PATH  # triton shells out to ninja
export TRITON_CACHE_DIR=$HOME/logs/triton-cache  # ~/.triton/cache is root-owned on the DGX image
export XTIER=1 XTIER_THREADS=16 XTIER_FAST_LOAD=1 XTIER_CACHE_SLOTS=8 XTIER_COLD_FRACTION=0.75
export XTIER_COLD_PATHS=$HOME/models/ds41-cold XTIER_DYNAMIC=2 XTIER_DOWN_BF16=1
python -m sglang.launch_server --model-path <ckpt> --tp-size 1 --mem-fraction-static 0.93 \
  --context-length 16384 --disable-shared-experts-fusion --cuda-graph-backend-prefill disabled
```

- `XTIER_THREADS` must leave CPUs free: the mailbox dispatcher needs one, and SGLang its own main
  thread. 20 cores -> 16 workers (20 fails with "no CPU left for the mailbox dispatcher").
- `XTIER_FAST_LOAD=1` is what makes it fit: experts stream straight into the engine, 2.2 GB/layer
  instead of 3.8 GB/layer (43 layers: 97 GB instead of 163 GB).
- `--mem-fraction-static` must be raised because the CPU engine's unified memory counts as used by
  CUDA: at 0.55 SGLang refuses with "loaded weights leave no GPU memory for the KV cache".
- Run the server inside a cgroup so a runaway load cannot wedge the box (a wedge needs a power
  cycle): `systemd-run --user --scope -p MemoryHigh=104G -p MemoryMax=113G --swap-max 0`.
  MemoryHigh throttles, MemoryMax kills inside the scope; the rest of the machine stays up.

### Tuning a single GB10

DeepSeek-V4-Flash-0731 (43 layers x 256 experts, MXFP4), XTIER_FAST_LOAD=1, 16 workers, 3 rounds of
3 x 256-token requests each, `steady` = the server's own `gen throughput` at the end of the run:

| slots | cold | mem-frac | e2e tok/s | steady tok/s | note |
|---|---|---|---|---|---|
| 8  | 0.75 | 0.93 | 7.53 | 8.90 | starting point |
| 32 | 0.70 | 0.93 | 8.82 | 10.34 | |
| 48 | 0.70 | 0.93 | 9.51 | 11.58 | |
| 48 | 0.62 | 0.93 | 10.64 | 12.03 | |
| 56 | 0.62 | 0.93 | 10.48 | 12.32 | |
| 48 | 0.58 | 0.90 | FAILED | - | "loaded weights leave no GPU memory for the KV cache" |
| 48 | 0.58 | 0.93 | 10.52 | 12.14 | the same config passes once mem-frac rises |
| **56** | **0.58** | **0.95** | **11.49** | **12.32** | 512-token bench, shipped as the default |
| 64 | 0.62 | 0.97 | 10.01 | 12.28 | more slots stop helping |
| 48 | 0.55 | 0.96 | 10.47 | 12.00 | less cold stops helping |

What the sweep says:

- GPU expert slots are monotone up to ~56; past that the extra copies cost more RAM than they save.
- More RAM-resident experts (lower `XTIER_COLD_FRACTION`) is the bigger lever, but it raises the
  weight footprint, and SGLang then refuses the KV cache: raise `--mem-fraction-static` with it
  (0.90 -> 0.95). That trade is worth ~15 %.
- 16 workers beats 18 (the mailbox dispatcher and the main thread need the cores); `XTIER_CACHE_EVERY=1`
  does nothing.
- The plateau is ~12.3-12.5 tok/s at 70 GB resident: the budget is no longer memory but the CPU
  expert engine (a token needs ~26 GFLOP of routed-expert work, i.e. 27-33 tok/s at the measured
  700-900 GF/s), so the next lever is speculative decoding (the checkpoint ships 3 MTP layers).

### Speculative decoding (DSPARK)

The checkpoint bundles a DSpark head (`num_nextn_predict_layers=3`, `dspark_block_size=5`), and
sglang 0.5.17 accepts `--speculative-algorithm DSPARK` for `DeepseekV4ForCausalLM`, so the plumbing is
there. Two things have to be right before the first token:

1. **Memory.** A single GB10 cannot hold the target and the draft at the default placement: the target
   loaded at 107.75 GB left 6.09 GB, and the draft died. Drop the target to `slots=40 cold=0.66`
   (87.56 GB), which leaves 26 GB, and the draft loads in 5.81 s at 6.09 GB.
2. **`XTIER_DRAFT=1`.** XTier counts the draft's MoE layers and refuses them otherwise:
   `RuntimeError: xtier: 3 draft MoE layers on XTier without XTIER_DRAFT=1`. With the flag the draft
   runs its 3 layers on the CPU engine (GPU-only draft slots by default).

After that the run stops in SGLang's sparse-MLA dispatch, not in XTier:

```
tvm.error.InternalError: Check failed: num_tokens > 64 (5 vs. 64) :
Decode (num_tokens <= 64) must go through sparse_mla_sm120_decode_dsv3_2 or
sparse_mla_sm120_decode_dsv4; got num_tokens=5
```

Reproduced with `--dsa-decode-backend` = flashinfer_sparse_mla, flashmla_sparse and tilelang, and
with `--disable-cuda-graph` (then the token count changes from 10 to 5), so it is the draft forward
calling the prefill-shaped kernel for a <= 64-token decode, not graph capture. The kernels it names
exist; the selection does not pick them on sm_121. Until that dispatch is patched (or a newer SGLang
is used), the single-Spark best is the non-speculative configuration above.

### Speculative decoding, measured

Both algorithms the release allows for `DeepseekV4ForCausalLM` were tried on one GB10 with the same
placement (`slots=40 cold=0.66`, target 87.56 GB + draft 6.09 GB, `XTIER_DRAFT=1`):

| algorithm | flags | result |
|---|---|---|
| DSPARK (gamma 5, 6 draft tokens from `dspark_block_size`) | `--speculative-algorithm DSPARK` | target and draft load, then `tvm.error.InternalError: ... Decode (num_tokens <= 64) must go through sparse_mla_sm120_decode_dsv3_2 or sparse_mla_sm120_decode_dsv4; got num_tokens=10` |
| EAGLE (topk 1) | `--speculative-algorithm EAGLE --speculative-eagle-topk 1` | dies earlier, in XTier's draft pack hooks: `http_server has no attribute _freeze_gc_after_server_warmup`, `openai has no attribute encoding_dsv41` - both are 0.5.21+ APIs |

DSPARK was reproduced with every `--dsa-decode-backend` value (flashinfer_sparse_mla, flashmla_sparse,
tilelang) and with `--disable-cuda-graph`, so the draft's <= 64-token decode is handed to the
prefill-shaped sparse-MLA kernel by 0.5.17's dispatch on sm_121. That is the whole gap: XTier takes the
draft's 3 MoE layers, the memory fits, only the attention kernel selection is wrong.

The fix is a newer SGLang. Version map, from the wheel metadata:

| sglang | `deepseek_v41_vit.py` | torch | sglang-kernel |
|---|---|---|---|
| 0.5.17 (works here) | no | 2.11.0 | 0.4.5 |
| 0.5.18 / 0.5.19 / 0.5.20 | no | 2.13.0 | 0.4.6-0.4.7 |
| 0.5.21 | **yes** | 2.13.0 | 0.4.7 |
| 0.5.22 | yes | 2.14.1 (no cu130 aarch64 wheel) | >=0.4.9 |

So 0.5.21 is the target: torch 2.13.0+cu130 (wheel exists on download.pytorch.org), flashinfer
0.6.18, sglang-kernel 0.4.7. An isolated venv at `~/venvs/sglang-521` is built to that point; the
resolver cannot finish on its own (`ResolutionImpossible`, PyPI's aarch64 torch is the CPU build), so
the remaining steps are:

```bash
NEW=~/venvs/sglang-521
$NEW/bin/pip install --no-deps pyzmq attrs transformers==5.19.0 tokenizers==0.23.2
$NEW/bin/python -c "import sglang.srt.entrypoints.http_server"   # repeat: install whatever it names
```

Then launch with `PY=$NEW/bin/python` (the launcher honours `PY=`) and re-run the DSPARK and vision
tests there. Note the working 0.5.17 stack must stay on torch 2.11 + sglang-kernel 0.4.5: 0.4.10 is
built against torch 2.14 and fails with `undefined symbol: torch::Library::_def`.

### DSPARK works on sglang 0.5.21

0.5.21 is the release that fixes the sparse-MLA dispatch (it auto-tunes `sparse_mla_sm120_decode_dsv4`)
and it also ships `deepseek_v41_vit.py`. Getting it running takes a hand-built venv: the resolver
cannot solve it on aarch64 (plain PyPI `torch` is the CPU build), so the order matters -

```bash
python -m venv ~/venvs/sglang-521
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install "flashinfer_python[cu13]==0.6.18" --extra-index-url https://flashinfer.ai/whl/cu130
# torchvision from the same cu130 index, or AutoImageProcessor refuses to register and the
# server dies with an ImportError whose message is the transformers backends complaint
pip install --no-deps torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
pip install --no-deps sglang==0.5.21 sglang-kernel==0.4.7
# then copy the working venv's other 172 packages --no-deps and fill whatever import still names
```

Measured on one GB10, same XTier placement for every row (target `slots=40 cold=0.66` = 87.6 GB,
draft 6.1 GB, `XTIER_DRAFT=1`, 512-token requests, 2x512 warmup discarded):

| config | e2e tok/s | steady tok/s | accept len / rate |
|---|---|---|---|
| no speculative decoding (0.5.17) | 10.91 | 12.73 | - |
| DSPARK block 7 | 11.08 | 16.31 | 3.62 / 0.38 |
| DSPARK block 3 | 16.10 | 21.97 | 2.80 / 0.60 |
| DSPARK block 5 (cold caches) | 17.28 | 21.85 | 2.70 / 0.34 |
| **DSPARK block 5, warm (shipped)** | **20.41** | **29.95** | **4.50 / 0.70** |

So speculative decoding is worth **+87% end to end and +135% steady** over the best non-speculative
configuration, and the default block size of 5 is the right one - block 7 drafts tokens the model
rejects, block 3 is close but consistently behind. The acceptance rate is the whole story: at 0.70
each verify step yields 4.5 tokens, which is what turns 12.7 tok/s into 30.

EAGLE is not a usable alternative for this checkpoint: with `--speculative-eagle-topk 1` and no
`--speculative-num-steps` the hook asserts `speculative_eagle_topk is None`, and with explicit
`--speculative-num-steps 3 --speculative-eagle-topk 1` it dies in XTier's EAGLE-path hooks
(`qsa.sparse_attn.pack_qsa_prefill_kv`, `models.deepseek_v4_replay_tail`, `tilelang.backend.target`).
The DSpark head bundled in this checkpoint is only driven by DSPARK.

Launch line for the shipped configuration:

```bash
PY=$HOME/venvs/sglang-521/bin/python MEM=0.95 MEMCAP=118G MEMHIGH=110G \
  XTIER_CACHE_SLOTS=40 XTIER_COLD_FRACTION=0.66 XTIER_THREADS=16 \
  XTIER_FAST_LOAD=1 XTIER_DRAFT=1 \
  bash ~/start_single_spark.sh --speculative-algorithm DSPARK
```

### Vision, the GPU and a dual-Spark comparison

**Vision is blocked on the checkpoint, not on the port.** sglang 0.5.21 does have the tower
(`models/deepseek_v41_vit.py`, and `deepseek_v4.py:4876` builds `ViT` + `Aligner` plus
`image_start/end/newline`). The gate is:

```python
if config.model_type == "deepseek_v41" and config.vision_n_layers > 0:
```

`DeepSeek-V4-Flash-Vision-Exp` has `vision_n_layers=32` but declares `model_type: deepseek_v4` and
`vision_config: null`, so it is served as text-only ("Model only supports text input; received
unsupported content type 'image_url'"). Setting `model_type: deepseek_v41` (the original is kept as
`config.json.orig`) does activate the branch - and then it asks for `image_token_id`, which the
checkpoint cannot answer: its tokenizer has **no image special tokens at all** (`added_tokens_decoder`
has none, `special_tokens_map` has none, the chat template never mentions images). Guessing ids would
map image patches to arbitrary embedding rows and produce confidently wrong answers, so the honest
state is "waiting for a checkpoint that ships its tokenizer side".

**The GPU is not a knob that can be turned up.** During 500-token single-stream generation:
util 35-86% (mean ~64%), SM 2463-2476 MHz (already pinned at max), **power 30-36 W**, CPU 60% user /
36% idle. Both sides idle *because they alternate*: one decode step is GPU (attention, dense) ->
CPU (this layer's 6 of 256 experts, on the XTier engine) -> GPU (combine), and the GPU waits during
the CPU half. The only way to fill it is more sequences in flight, and that is exactly what this stack
cannot do yet: `--max-running-requests 8` or `16` dies in `RuntimeError: Prepare this stream/shape
before graph capture` (the DSPARK verify shape has no captured CUDA-graph tier; neither
`--speculative-dspark-align-verify-tokens-to-graph-tier` nor `--disable-cuda-graph` avoided it, and
with `--cuda-graph-backend-decode disabled` plus 16 slots the server starts but concurrent requests
stall). Treat ~64% / 31 tok/s as what an offloaded MoE looks like on a single GB10.

**Single Spark vs dual Spark**, same model family, decode tok/s, concurrency 1/4/8:

| rig | c1 | c4 | c8 | notes |
|---|---|---|---|---|
| single GB10, sglang 0.5.21 + XTier + DSPARK | **20.5-30.0** | - | - | this work, 512-token requests |
| dual GB10 (vLLM TP2, RoCE), DSpark k=3 | 38.95 | 71.98 | 89.92 | same harness family |
| dual GB10 (vLLM TP2), DSpark k=5 | 38.42 | 64.97 | 81.90 | |
| dual GB10, V4.1-Flash EXL3 2.9bpw | 74.9 (prose) | 130 | - | different checkpoint/quant |

Single-stream is **~52% of the dual-Spark number on half the hardware** - i.e. slightly better than
linear, because TP2 pays an interconnect round-trip inside every layer while one node keeps all
experts on its own CPU engine. Batching is where the second GB10 actually pays: the dual rig reaches
~90 tok/s at c8, and that is the gap to close once the CUDA-graph tiers for multi-request DSPARK are
sorted out.

### Batching, and what the KV pool actually is

The `Prepare this stream/shape before graph capture` failure was **the bring-up launcher's own fault**:
it passed `--skip-server-warmup`. Without the warmup pass, tilelang never prepares the shapes it is
later asked to capture, and with `--max-running-requests` above 2 the capture set includes shapes that
were never touched. Drop the flag (`WARM=1` in the launcher) and concurrency works - no graph-tier
flag needed.

Measured with 8 slots, 1M context, same placement (384-token requests, 2 rounds each):

| threads | c1 | c4 | c8 |
|---|---|---|---|
| 16 | 14.9 | 25.7 | **28.0** |
| 19 | 16.8 | - | 21.4 |

More threads is *worse*: 19 workers on 20 cores starve the HTTP/scheduler threads, which is the same
effect that made 18 threads lose to 16 in the single-stream sweep. Batching buys ~+88% aggregate over
c1, and aggregate then plateaus at the CPU engine's throughput - the GPU has more to do, the ARM cores
are now the wall.

**KV pool** (fp8 KV, `--mem-fraction-static 0.95`, target 87.6 GB + draft 18.8 GB):

| config | `max_total_num_tokens` | concurrent 1M requests |
|---|---|---|
| DSPARK on (shipped) | **973,056** | 0 - one 1M request does not fit, ~1 at 900K |
| DSPARK off | **4,004,608** | 3 (4 at ~950K each) |

That is the real cost of speculative decoding here: the 18.8 GB draft plus its KV turns a 4M-token
pool into a 973K one. Pick per workload:

```bash
# speed (shipped): ~20-28 tok/s, ~950K context, 8 slots
PY=$HOME/venvs/sglang-521/bin/python WARM=1 MAXREQ=8 CONTEXT=1048576 CTXPOOL=1048576 \
  MEM=0.95 MEMCAP=118G MEMHIGH=110G XTIER_CACHE_SLOTS=40 XTIER_COLD_FRACTION=0.66 \
  XTIER_THREADS=16 XTIER_FAST_LOAD=1 XTIER_DRAFT=1 \
  bash ~/start_single_spark.sh --speculative-algorithm DSPARK

# context: ~12.7 tok/s, 4M pool, three 1M requests at once
PY=$HOME/venvs/sglang-521/bin/python WARM=1 MAXREQ=8 CONTEXT=1048576 CTXPOOL=4194304 \
  MEM=0.95 MEMCAP=118G MEMHIGH=110G XTIER_CACHE_SLOTS=40 XTIER_COLD_FRACTION=0.66 \
  XTIER_THREADS=16 XTIER_FAST_LOAD=1 bash ~/start_single_spark.sh
```

### Prefill is the wall

Single-slot production layout (`MAXREQ=1 CONTEXT=921600 CTXPOOL=921600 MEM=0.95 BATCH=2048`,
DSPARK on, 16 workers, pool 921,600 tokens), measured with a **unique first sentence per request**
so sglang's radix cache cannot serve the prompt:

| prompt | TTFT | prefill | decode |
|---|---|---|---|
| 9,082 tok | 143.9 s | **63 tok/s** | 16.0 tok/s |
| 36,246 tok | 518.6 s | **70 tok/s** | 17.1 tok/s |

Prefill is flat at 63-70 tok/s because it is arithmetic-bound on the CPU expert engine, not
length-bound: every token needs the 43 layers x 6-of-256 experts to be computed on the ARM cores
(~0.9 TFLOP/s measured at m=2048). Longer prompts buy nothing.

Two traps when measuring this box:

- **The radix cache makes prefill look 40x faster.** Repeating the same prompt text gave 2,502 tok/s
  on a 9k prompt - that is a cache hit, not compute. Always salt the first tokens.
- **The first request after startup is 7x slower** (38 vs 281 tok/s cold-vs-warm) because the hot
  expert slots are empty and every expert read comes off the drive.

Consequences for a 900K context, extrapolating at 70 tok/s:

| context | prefill | verdict |
|---|---|---|
| 8K | ~2 min | batch jobs only |
| 32K | ~8.5 min | batch jobs only |
| 128K | ~30 min | not practical |
| 900K | **~3.7 h** | single request costs half a day |

So the 900K figure is a *capacity* number, not a usable operating point on one GB10: the pool holds
921,600 tokens and `--max-running-requests 1` makes them all available to one request, but nothing
can prefill that much. Keep `MEM=0.95` only if you actually intend to try; at that setting `free`
reports **3-4 GB available**, which is what wedged the box on a 512K prefill. For a stable server use
`MEM=0.92 CTXPOOL=700000` and treat ~32K as the practical ceiling.
