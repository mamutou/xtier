# ARM（AArch64 / NEON）移植

CPU 专家引擎原来只有 x86 的 AVX512 BF16 和 AVX2/FMA 两套内核。这份文档记录把它搬到 AArch64 时改了什么、
为什么这么改、以及在 DGX Spark（GB10，20 核）上实测到什么。

## 结论先说

| | 状态 |
|---|---|
| 内核 | `libxtier_engine_neonbf16.so`（Armv8.6-A BF16，`vbfdotq_f32`）与 `libxtier_engine_neon.so`（任何 Armv8-A，BF16 加宽 + FMA），`libxtier_engine.so` 运行时二选一 |
| 数值 | 合成层对拍（float64 反量化参考，两种 FP4 格式 × 9 种 token 数 × 2 个 scale 量程 = 36 例）RMS 相对误差 1.4e-7…2e-6（真实量程），全部通过；两个 ARM 构建都通过 |
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
