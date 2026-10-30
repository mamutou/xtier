# 版本更新

按日期倒序，README 只列最近 5 条。

- 2026-10-30：单机 DGX Spark（GB10）部署与压测报告（[文档](reports/single-spark-gb10.md)）：ARM 专家引擎 + SGLang 0.5.21
  + DSpark 在单台 121 GB 机器上跑 156 GB 的 DeepSeek-V4-Flash-0731。热身后单流 20.4 e2e / 30.0 steady tok/s，8 并发聚合
  28.0 tok/s；单流是双 Spark TP2（38.95 tok/s @ c1）的 52%，而硬件只有一半。**预填充是墙**：绕过前缀缓存实测 63–70 tok/s
  且与长度无关（900K 请求要约 3.7 小时），decode 在 9K/36K 上下文为 16.0/17.1 tok/s。KV 池开投机 97.3 万 token、关投机
  400 万（3 个 1M 并发）。报告记录了四个坑：`XTIER_FAST_LOAD` 必须开（每层 3.8 GB→2.2 GB）、`mem-fraction` 必须 0.92+
  、`systemd-run` 单元名必须唯一、**去掉 `--skip-server-warmup` 并发才能超过 2**（否则 `Prepare this stream/shape before
  graph capture`）。新增 `deploy/spark/`（cgroup 限内存启动器，防整机 wedge）与 `bench/spark_decode.py` /
  `bench/spark_lengths.py` / `bench/spark_concurrency.py` 三个基准；vision 因检查点缺 image 特殊 token 未接。

- 2026-10-29：CPU 专家引擎移植到 AArch64（NEON）：`neonbf16` 构建用 Armv8.6 BF16 点积、`neon` 构建是任意 ARMv8-A 的
  加宽 FMA 版本，`libxtier_engine.so` 按 HWCAP2 自动选（`XTIER_ISA=neon|neonbf16` 可强制）；信箱的自旋提示改 `yield`，
  单 NUMA / 无 SMT 的机器不再套用 x86 的 4 节点拓扑。新验收脚本 `tests/neon_check.py`（合成层对 float64 参考，不需要检查点）
  在两种 ARM 构建上都通过；`tools/isa_check.sh` 同时覆盖 AVX512 与 BFDOT。DGX Spark GB10 单层实测 decode 330 GFLOP/s
  （m=1）、prefill 914 GFLOP/s（m=2048），基线 NEON 构建的 3.0× / 2.6×。新 `tests/real_layer_check.py` 直接读检查点跑一层
  （DeepSeek-V4-Flash-0731 第 0 层 MXFP4：356 / 889 GFLOP/s，对 float64 参考 rms 9e-8…4.6e-6）。详见[文档](arm-port.md)。

- 2026-10-28：decode 时可把一部分未命中的专家交给 GPU 代算（可选，`XTIER_DECODE_MISS_GPU`，默认关；启动台三张卡可选），
  给 CPU 比 PCIe 慢的机器用，结果与命中的专家相同。详见[文档](reports/2026-10-28-decode-miss-gpu.md)。
- 2026-10-27（下午）：快速加载（可选，`XTIER_FAST_LOAD=1`，默认关）：专家直接从模型文件或冷存储读进 CPU 引擎，仍是全部装好
  才报在线；DeepSeek 启动 472 → 186 s，卸载 30% 409 → 124 s。详见[文档](reports/2026-10-27-fast-load.md)。
- 2026-10-27：启动台新卡片「DeepSeek V4.1 Flash 1017 · 最小化」：32 GB 显存 + 128 GB 内存（模拟 RTX 5090），70% 的专家放硬盘，
  prefill 约 1,160、decode 约 30 tok/s。详见[文档](reports/2026-10-27-ds41-minimal.md)。
- 2026-10-26（晚）：冷专家卸载到硬盘（可选，`XTIER_COLD=1`），四个模型都能用；DeepSeek 卸载 30% 时 decode 每步 +44%。
  详见[文档](reports/2026-10-26-cold-experts.md)。
- 2026-10-26：弹性热专家槽（DeepSeek 每步 −4.3%、GLM −1.5%）；GLM 常备槽 60 → 58；启动台报在线前先预热；MiniMax-M3.1-Flash
  暂停开发。详见[文档](reports/2026-10-26-elastic-cache.md)。
- 2026-10-25：测速表改为官方 main 对加速包；Qwen 启动台默认用自动规划布局；热专家池可选用提示的路由预热（`XTIER_CACHE_PROMPT=1`，
  默认关）；修了 MiniMax 包钩子和 GC 冻结两个启动报错。详见[文档](reports/2026-10-25-prompt-warmup.md)。
- 2026-10-24（晚）：自动规划器第一版：本机测出各层 GPU / CPU 耗时，在显存预算内逐层定热专家槽数（`XTIER_CACHE_PLAN=auto`）；
  Qwen 启动时自动布局冷 prefill +2.4–2.9%，其余模型保留手工布局。详见[文档](reports/2026-10-24-planner.md)。
- 2026-10-24（下午）：MiniMax 固定 512K、KV 存 NVFP4（42.5 → 24 GiB），DSpark 按置信度截断验证（无损，agent decode +11%）。详见[文档](reports/2026-10-24-minimax-512k.md)。
- 2026-10-24（上午，二）：SM120 上 prefill 大小的 MXFP8 GEMM 改用 128×128 tile（输出逐位相同），MiniMax 64K 冷 prefill
  1,606 → 1,963 tok/s。详见[文档](reports/2026-10-24-ds-woa-minimax.md)。
- 2026-10-24（上午）：新模型 MiniMax-M3.1-Flash（SGLang M3.1 分支），加速包 agent 每步 70.6 → 59.5 ms；DeepSeek 索引器 decode
  走分页路径，每步 33.0 → 32.5 ms。详见[文档](reports/2026-10-24-ds-woa-minimax.md)。
- 2026-10-24（清晨）：DeepSeek 的 wo_a 保持 FP8（每步 33.9 → 33.0 ms，显存 −1.25 GiB），提示按 deepseek-recipe 分词；修了 TP 1
  下 NCCL 临时申请显存导致的 OOM。详见[文档](reports/2026-10-24-ds-woa-minimax.md)。
- 2026-10-24（夜）：测速统一用 agent 速度；启动参数补上生产的接口参数（缓存命中上报、工具调用解析）。详见[文档](reports/2026-10-24-qwen-kv-ds-agent.md)。
- 2026-10-24：控制台 `xtier/console/`：一个网页选模型、调参数、启动 / 停止，加模型只需加配置目录。详见[文档](console.md)。
- 2026-10-24：Qwen 默认 NVFP4 KV，省下的显存多放一层常驻（冷 prefill +0.5–1.4%）；DeepSeek 按生产 agent_bench 实测冷 prefill 快
  10–25%。详见[文档](reports/2026-10-24-qwen-kv-ds-agent.md)。
- 2026-10-23（夜）：通用投机解码包：GLM 用热词表拒绝采样和置信度截断，decode 93.0 → 95.6 tok/s（生产 84.8）。详见[文档](reports/2026-10-23-speculative-fp4kv.md)。
- 2026-10-23（晚）：512K prefill 追平生产：Qwen 125 → 102 s（生产 106 s），GLM 224 → 192 s（生产 201 s）。详见[文档](reports/2026-10-23-prefill.md)。
- 2026-10-23（晚）：DeepSeek 移植 Engram 提前查表和草稿词表头 FP8，decode 每步 −8%。详见[文档](reports/2026-10-23-ds-engram-prefetch.md)。
- 2026-10-23：README 测速表；加载不再换出，DeepSeek 启动 811 → 409 s。详见[文档](reports/2026-10-23-step3-ds-tables.md)。
- 2026-10-23：第 3 步：DeepSeek 跑通官方 main，通用 NVMe 表，GLM 的 FP8 权重，双卡准备。详见[文档](reports/2026-10-23-step3-ds-tables.md)。
- 2026-10-23：第 2 步：后端、热专家缓存、GPU prefill 在官方 main 上跑通 Qwen 与 GLM。详见[文档](reports/2026-10-23-step2-qwen.md)。
- 2026-10-23：第 1 步：CPU 引擎迁入 `csrc/`，三个模型的真实层与 lkqmoe 0.5.6 逐位一致（`bench/results/engine-ab-20261023.json`）。
