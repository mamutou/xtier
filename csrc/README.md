# csrc

The CPU expert engine and the CPU-GPU mailbox, ported from lkqmoe 0.5.6 (commit `93edcc1`) by
[tools/port_engine.py](../tools/port_engine.py): identifiers, strings and comments renamed only.

| File | What |
|---|---|
| `engine/engine.cpp` | MoE experts on the CPU, generic over hidden size H and expert intermediate size I: NVFP4 (E4M3 / 16) and MXFP4 (UE8M0 / 32) weights, BF16 input, FP32 SiLU, intermediate, down projection and accumulation. AVX512 BF16 (+VBMI) and AVX2 / FMA kernels on x86-64, NEON kernels on AArch64 (BF16 dot product with `-march=armv8.6-a+bf16`, else widening FMAs); NUMA topology from sysfs, each expert's rows split over the nodes, one worker pool per node. One translation unit on purpose: the kernels must compile to the same code as before. |
| `engine/dispatch.cpp` | `libxtier_engine.so`: exports the API and loads the kernel build next to it on first use — AVX512 or AVX2 on x86-64, NEON+BFDOT or baseline NEON on AArch64 (`XTIER_ISA=avx512\|avx2\|neon\|neonbf16` forces one). |
| `mailbox/mailbox.cpp` | `libxtier_mailbox.so`: mapped host mailboxes polled by one dispatcher thread, stream memory operations instead of host callbacks (CUDA-graph capturable), and the expert-pool command ring with its policy thread. |
| `include/xtier_engine.h`, `include/xtier_mailbox.h` | C APIs. |
| `tables/rows.cpp` | `libxtier_tables.so`: rows of large embedding tables read from the checkpoint files on NVMe (DeepSeek V4.1 Engram, Qwen3.8 PLE n-grams): any row width with optional scale bytes, one table over many tensors and files, O_DIRECT reads on a reader pool, bounded row cache, preloaded scales, TP row range; grown from the 1017 pipeline's `row_store.cpp`; its lookup calls no CUDA API ([design](../docs/design-tables-nvme.md)). |

Build: `tools/build_engine.sh [CUDA_ROOT]` (CMake; the AVX512 build uses the one-row Down microkernel, `XTIER_AVX512_DOWN_R4=1`,
like the lkqmoe release; on AArch64 it builds `libxtier_engine_neonbf16.so` and `libxtier_engine_neon.so`, see
[docs/arm-port.md](../docs/arm-port.md)). Check: `tests/engine_ab.py` (one real layer, output digests and timings);
`tests/neon_check.py` (synthetic layer against a float64 reference, any ISA, no checkpoint); `tools/isa_check.sh` (no AVX512
instruction in the libraries an AVX2-only host loads, no BFDOT instruction in the libraries an Armv8-A host loads).

Status (2026-10-23): on one real layer of Qwen3.8 Flash Next (NVFP4), GLM-5.3 Flash (NVFP4) and DeepSeek V4.1 Flash 1017 (MXFP4),
9 shapes each (1 to 2048 tokens, decode and verify routing), the outputs equal lkqmoe 0.5.6 bit for bit with both the AVX512 and
the AVX2 kernels. The AVX512 kernel library disassembles to the same instructions as the production build except for the removed
`lkqmoe_create` compatibility entry point.

Status (2026-10-29, AArch64): `tests/neon_check.py` passes on both ARM builds (RMS relative error 1.4e-7 … 2e-6 against a float64
reference, both FP4 formats, 1 to 2048 tokens); `tools/isa_check.sh` shows no BF16 instruction in the baseline NEON library and
none of the dispatch/mailbox/tables libraries. On one DGX Spark GB10 (20 cores) a DeepSeek V4.1 Flash 1017 layer runs at
330 GFLOP/s decode (m=1) and 914 GFLOP/s prefill (m=2048) with the BFDOT build — 3.0x / 2.6x the baseline NEON build.
