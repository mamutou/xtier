"""Synthetic MoE layer check of the CPU engine: numerics against a dequantised float64 reference, then
per-call time at DeepSeek V4.1 Flash's shape. No model checkpoint, no GPU: it exercises whichever
kernel build the host's ISA selects (avx512 / avx2 / neonbf16 / neon; ``XTIER_ISA`` forces one), so it
is the acceptance test for a new ISA port and a regression test for the existing ones.

Reference: both weight formats are unpacked here in float64 (E2M1 codes times the E4M3 block scale,
or the UE8M0 power of two for MXFP4), the products of BF16 inputs are exact in float64, so the only
difference to the engine is the order of the FP32 accumulation.

  python tests/neon_check.py --threads 20                       # numerics + speed, current ISA
  XTIER_ISA=neon python tests/neon_check.py --threads 20        # force the generic NEON build
  python tests/neon_check.py --threads 20 --skip-perf            # numerics only
"""
import argparse
import ctypes as C
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

A = argparse.ArgumentParser()
A.add_argument('--lib-dir', default=str(Path(__file__).resolve().parents[1] / 'xtier' / 'engine' / 'lib'))
A.add_argument('--threads', type=int, default=20)
A.add_argument('--rounds', type=int, default=3)
A.add_argument('--skip-perf', action='store_true')
A.add_argument('--peak-gflops', type=float, default=0.0, help='optional: report %% of this machine peak')
A = A.parse_args()

LIB_DIR = Path(A.lib_dir).resolve()
os.environ['XTIER_ENGINE_LIB'] = str(LIB_DIR)
# XTIER_DOWN_BF16: the Gate -> Down intermediate is rounded to BF16, as the reference below does.
DOWN_BF16 = os.environ.get('XTIER_DOWN_BF16') == '1'

# E2M1 code values: low nibble first, and 8..15 are the negatives of 0..7 (sign bit = code >> 3).
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)


def scale_table(fmt):
    """Block-scale values by byte: E4M3 for NVFP4 (0..126; 127 is NaN), UE8M0 (a power of two) for MXFP4."""
    s = np.arange(256)
    if fmt == 'mxfp4':
        return np.ldexp(np.ones(256), s - 127)
    e, m = s >> 3, s & 7
    return np.where(e > 0, (1 + m / 8) * np.ldexp(1.0, e - 7), m * np.ldexp(1.0, -9))


def dequant(packed, scales, fmt):
    """(rows, K/2) uint8 nibbles + (rows, K/G) scale bytes -> (rows, K) float64 weights."""
    rows, half = packed.shape
    k = half * 2
    codes = np.empty((rows, k), dtype=np.int64)
    codes[:, 0::2] = packed & 0x0F          # weight 2i = low nibble
    codes[:, 1::2] = packed >> 4            # weight 2i+1 = high nibble
    g = k // scales.shape[1]
    return FP4[codes] * np.repeat(scale_table(fmt)[scales], g, axis=1)


def random_layer(hidden, inter, experts, fmt, seed=20261016, limit=10.0, band=None):
    """Engine inputs (packed weights, block scales, global scales) with random FP4 codes and block scales.

    ``band`` is the (low, high) byte range of the block scales: a realistic one keeps the layer inside
    the range a checkpoint uses, a wide one exercises the whole scale table. MXFP4's UE8M0 byte 254 is
    a power of 2^127, which overflows BF16 once it is multiplied by an FP4 code, so no real checkpoint
    gets near it and the kernel tables are built for that.
    """
    rng = np.random.default_rng(seed)
    g = 32 if fmt == 'mxfp4' else 16
    top = 254 if fmt == 'mxfp4' else 126     # the engine rejects wider block scales
    lo, hi = band or ((118, 133) if fmt == 'mxfp4' else (16, 48))
    assert 0 <= lo <= hi <= top
    w13 = rng.integers(0, 256, (experts, 2 * inter, hidden // 2), dtype=np.uint8)
    w2 = rng.integers(0, 256, (experts, hidden, inter // 2), dtype=np.uint8)
    s13 = rng.integers(lo, hi + 1, (experts, 2 * inter, hidden // g), dtype=np.uint8)
    s2 = rng.integers(lo, hi + 1, (experts, hidden, inter // g), dtype=np.uint8)
    g13 = rng.uniform(0.5, 2.0, (experts, 2)).astype(np.float32)
    g2 = rng.uniform(0.5, 2.0, (experts,)).astype(np.float32)
    return dict(w13=w13, w2=w2, s13=s13, s2=s2, g13=g13, g2=g2,
                hidden=hidden, inter=inter, experts=experts, fmt=fmt, limit=limit)


def reference(layer, x, ids, rw):
    """float64 MoE forward of the same layer: route weights applied to the Down output, as the engine does."""
    fmt, hidden, inter, limit = layer['fmt'], layer['hidden'], layer['inter'], layer['limit']
    i = inter
    wg = np.stack([dequant(layer['w13'][e, :i], layer['s13'][e, :i], fmt) for e in range(layer['experts'])])
    wu = np.stack([dequant(layer['w13'][e, i:], layer['s13'][e, i:], fmt) for e in range(layer['experts'])])
    wd = np.stack([dequant(layer['w2'][e], layer['s2'][e], fmt) for e in range(layer['experts'])])
    m, k = ids.shape
    y = np.zeros((m, hidden), dtype=np.float64)
    for t in range(m):
        xt = from_bf16(x[t]).astype(np.float64)
        for q in range(k):
            e = int(ids[t, q])
            if e < 0:
                continue
            gate = wg[e] @ xt * float(layer['g13'][e, 0])
            up = wu[e] @ xt * float(layer['g13'][e, 1])
            if limit > 0:
                up = np.clip(up, -limit, limit)
                gate = np.minimum(gate, limit)
            with np.errstate(over='ignore'):
                mid = gate / (1 + np.exp(-gate)) * up
            if DOWN_BF16:       # the engine keeps the intermediate as BF16 words in this mode
                mid = from_bf16(bf16_bits(mid.astype(np.float32))).astype(np.float64)
            y[t] += wd[e] @ mid * float(layer['g2'][e]) * float(rw[t, q])
    return y


def engine():
    lib = C.CDLL(str(LIB_DIR / 'libxtier_engine.so'))
    lib.xtier_isa.restype = C.c_char_p
    lib.xtier_last_error.restype = C.c_char_p
    lib.xtier_create3.restype = C.c_void_p
    lib.xtier_create3.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int, C.c_float, C.c_int] + [C.c_void_p] * 6
    lib.xtier_forward.restype = C.c_int
    lib.xtier_forward.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
    lib.xtier_destroy.argtypes = [C.c_void_p]
    lib.xtier_nodes.restype = C.c_int
    lib.xtier_nodes.argtypes = [C.c_void_p]
    lib.xtier_worker_cpus.restype = C.c_int
    lib.xtier_worker_cpus.argtypes = [C.c_int, C.c_void_p, C.c_int]
    return lib


def plan_for(lib, layer, threads):
    ptrs = [C.c_void_p(layer[n].ctypes.data) for n in ('w13', 'w2', 's13', 's2', 'g13', 'g2')]
    fmt = 1 if layer['fmt'] == 'mxfp4' else 0
    p = lib.xtier_create3(layer['hidden'], layer['inter'], layer['experts'], threads,
                          C.c_float(layer['limit']), fmt, *ptrs)
    if not p:
        raise SystemExit('create failed: ' + lib.xtier_last_error().decode())
    return p


def forward(lib, plan, x, ids, rw):
    y = np.empty((x.shape[0], x.shape[1]), dtype=np.float32)
    rc = lib.xtier_forward(plan, C.c_int(x.shape[0]), C.c_int(ids.shape[1]),
                           C.c_void_p(x.ctypes.data), C.c_void_p(ids.ctypes.data),
                           C.c_void_p(rw.ctypes.data), C.c_void_p(y.ctypes.data))
    if rc < 0:
        raise SystemExit('forward failed: ' + lib.xtier_last_error().decode())
    return y


def make_batch(rng, m, k, hidden, experts, fmt='nvfp4'):
    """BF16 inputs (as uint16 words, the engine's layout), top-k routes (a repeated expert for the
    m<=4 groups) and softmax route weights."""
    x = bf16_bits(rng.standard_normal((m, hidden)).astype(np.float32))
    ids = np.stack([rng.permutation(experts)[:k] for _ in range(m)]).astype(np.int32)
    if m > 1:                                   # shared experts: one group with several tokens
        ids[1:, :k // 2] = ids[0, :k // 2]
    logits = rng.standard_normal((m, k)).astype(np.float32)
    rw = np.exp(logits - logits.max(-1, keepdims=True))
    rw = (rw / rw.sum(-1, keepdims=True)).astype(np.float32)
    return x, ids, rw


def bf16_bits(x):
    """float32 -> the BF16 words the engine reads (round to nearest even), 2 bytes per value."""
    u = x.view(np.uint32)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def from_bf16(bits):
    """BF16 words -> float32 (exact)."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def main():
    lib = engine()
    isa = lib.xtier_isa().decode()
    print(json.dumps(dict(isa=isa, threads=A.threads, lib_dir=str(LIB_DIR),
                          switches={k: v for k, v in os.environ.items() if k.startswith('XTIER_')})))
    bad = 0
    # --- numerics: every token-count path (decode m<=8, prefill above), both weight formats, and one
    #     realistic plus one wide block-scale band per format. A wide band (17 decades of weight scale in
    #     one row) is dominated by the cancellation of the FP32 sum, hence its looser gate.
    for fmt, band, seed, gate in (('nvfp4', (16, 48), 7, 1e-5), ('nvfp4', (0, 126), 8, 1e-4),
                                  ('mxfp4', (118, 133), 11, 1e-5), ('mxfp4', (60, 140), 12, 1e-4)):
        if DOWN_BF16:
            # The reference rounds its float64 intermediate to BF16; the engine rounds a float32 one, so
            # about one value in 2^15 lands on the other side of a BF16 boundary (2^-8 for that value).
            gate = max(gate, 1e-4)
        hidden, inter, experts, k = 512, 256, 16, 4
        layer = random_layer(hidden, inter, experts, fmt, seed=seed, band=band)
        plan = plan_for(lib, layer, A.threads)
        print(f'[{fmt} scales {band[0]}..{band[1]}] nodes={lib.xtier_nodes(plan)}', flush=True)
        rng = np.random.default_rng(3)
        for m in (1, 2, 3, 4, 5, 6, 8, 17, 64):
            x, ids, rw = make_batch(rng, m, k, hidden, experts)
            y = forward(lib, plan, x, ids, rw)
            ref = reference(layer, x, ids, rw)
            diff = y - ref
            err = np.abs(diff).max()
            rel = err / (np.abs(ref).max() + 1e-30)
            # RMS relative error: robust to the cancellation of a single output element, and a wrong
            # code table, scale table or nibble order would show up at 1e-1..1e-3 here, not at 1e-7.
            rms = float(np.linalg.norm(diff) / np.linalg.norm(ref))
            ok = rms < gate
            bad += not ok
            print(f'[{fmt}] m={m:4d} rms={rms:.2e} max_rel={rel:.2e} max_abs={err:.3e} {"ok" if ok else "FAIL"}',
                  flush=True)
        lib.xtier_destroy(plan)
    if bad:
        raise SystemExit(f'{bad} numeric case(s) failed')

    if A.skip_perf:
        return
    # --- speed: one DeepSeek V4.1 Flash 1017 layer (H=5120, I=2304, top-6), production switches
    hidden, inter, experts, k = 5120, 2304, 64, 6
    layer = random_layer(hidden, inter, experts, 'mxfp4', seed=99)
    t0 = time.perf_counter()
    plan = plan_for(lib, layer, A.threads)
    print(f'[perf] plan built in {time.perf_counter() - t0:.2f}s '
          f'({experts} experts x {3 * inter * hidden / 1e6:.1f}M MAC)', flush=True)
    macs = k * 3 * inter * hidden          # gate + up + down per token per route
    out = {}
    for m in (1, 2, 4, 8, 64, 512, 2048):
        x, ids, rw = make_batch(np.random.default_rng(m), m, k, hidden, experts)
        forward(lib, plan, x, ids, rw)                     # warm up this token count
        times = []
        for _ in range(A.rounds):
            t0 = time.perf_counter()
            forward(lib, plan, x, ids, rw)
            times.append(time.perf_counter() - t0)
        us = statistics.median(times) * 1e6
        gflops = 2 * macs * m / (us * 1e-6) / 1e9
        out[m] = dict(us=round(us, 1), gflops=round(gflops, 1),
                      tok_per_s=round(m / (us * 1e-6), 1) if m <= 8 else None)
        print(f'[perf] m={m:4d} {us:9.1f} us  {gflops:7.1f} GFLOP/s'
              + (f'  {m / (us * 1e-6):7.1f} tok/s/layer  (~{40 * us / 1e3:.0f} ms per token over 40 layers)'
                 if m <= 8 else ''), flush=True)
    print(json.dumps(dict(isa=isa, threads=A.threads, per_call=out)))


if __name__ == '__main__':
    main()
