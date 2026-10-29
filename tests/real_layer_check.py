"""One real checkpoint layer through the CPU engine: numerics against a float64 reference and per-token-count
timings, with no GPU and no SGLang. The checkpoint must store packed FP4 experts the way the engine takes them
(MXFP4: uint8 nibbles + one UE8M0 byte per 32 weights, or NVFP4: uint8 nibbles + one E4M3 byte per 16 weights).

  python tests/real_layer_check.py --model ~/models/DeepSeek-V4-Flash-0731 --layer 0 --experts 64 --threads 20
  XTIER_ISA=neon python tests/real_layer_check.py ... --save /tmp/neon      # then compare two builds:
  python tests/real_layer_check.py ... --compare /tmp/neonbf16 /tmp/neon

The reference dequantises every expert the case routes to (once per expert) in float64, so `--ref-tokens` keeps
its cost bounded: decode-sized token counts check the numbers, larger ones only measure.
"""
import argparse
import ctypes as C
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

P = argparse.ArgumentParser()
P.add_argument('--model', required=True, help='checkpoint directory (config.json + *.safetensors + index)')
P.add_argument('--layer', type=int, default=0)
P.add_argument('--experts', type=int, default=64, help='experts to load (uniformly from the layer)')
P.add_argument('--topk', type=int, default=6)
P.add_argument('--threads', type=int, default=20)
P.add_argument('--format', choices=('mxfp4', 'nvfp4'), default='mxfp4')
P.add_argument('--key', default='layers.{layer}.ffn.experts.{e}.', help='expert key prefix in the checkpoint')
P.add_argument('--names', default='w1,w3,w2', help='gate, up, down tensor names')
P.add_argument('--seed', type=int, default=20261016)
P.add_argument('--rounds', type=int, default=3)
P.add_argument('--ref-tokens', type=int, default=4, help='largest token count checked against the reference')
P.add_argument('--save', help='write the outputs of every case to this directory (for --compare)')
P.add_argument('--compare', nargs=2, metavar=('DIR_A', 'DIR_B'), help='compare two saved runs and exit')
P.add_argument('--lib-dir', default=str(Path(__file__).resolve().parents[1] / 'xtier' / 'engine' / 'lib'))
A = P.parse_args()

# XTIER_DOWN_BF16: the Gate -> Down intermediate is rounded to BF16, as the reference below does.
DOWN_BF16 = os.environ.get('XTIER_DOWN_BF16') == '1'

FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)


def scale_values(fmt):
    s = np.arange(256)
    if fmt == 'mxfp4':
        return np.ldexp(np.ones(256), s - 127)
    e, m = s >> 3, s & 7
    return np.where(e > 0, (1 + m / 8) * np.ldexp(1.0, e - 7), m * np.ldexp(1.0, -9))


def dequant(packed, scales, fmt):
    """(rows, K/2) uint8 nibbles + (rows, K/G) scale bytes -> (rows, K) float64."""
    rows, half = packed.shape
    k = half * 2
    codes = np.empty((rows, k), dtype=np.int64)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    return FP4[codes] * np.repeat(scale_values(fmt)[scales], k // scales.shape[1], axis=1)


def bf16_bits(x):
    u = x.view(np.uint32)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def from_bf16(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def load_layer(model, layer, experts, names, key, fmt):
    """Engine inputs from a checkpoint: packed weights, block scales, global scales (1 for MXFP4)."""
    import torch
    from safetensors import safe_open
    root = Path(model).expanduser()
    cfg = json.loads((root / 'config.json').read_text())
    hidden, inter = cfg['hidden_size'], cfg['moe_intermediate_size']
    total = cfg.get('n_routed_experts') or cfg.get('num_experts')
    step = max(1, total // experts)
    chosen = [e * step for e in range(experts)]
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    g = 32 if fmt == 'mxfp4' else 16
    w13 = np.empty((experts, 2 * inter, hidden // 2), dtype=np.uint8)
    w2 = np.empty((experts, hidden, inter // 2), dtype=np.uint8)
    s13 = np.empty((experts, 2 * inter, hidden // g), dtype=np.uint8)
    s2 = np.empty((experts, hidden, inter // g), dtype=np.uint8)
    gate, up, down = names
    by_file = {}
    for slot, e in enumerate(chosen):
        p = key.format(layer=layer, e=e)
        for name, role in ((gate, 'gate'), (up, 'up'), (down, 'down')):
            by_file.setdefault(index[p + name + '.weight'], []).append((slot, p, name, role))
    for file, items in by_file.items():
        with safe_open(str(root / file), framework='pt', device='cpu') as f:
            for slot, p, name, role in items:
                packed = f.get_tensor(p + name + '.weight').view(torch.uint8).numpy()
                scales = f.get_tensor(p + name + '.scale').view(torch.uint8).numpy()
                if role == 'down':
                    w2[slot], s2[slot] = packed, scales
                else:
                    rows = slice(0, inter) if role == 'gate' else slice(inter, 2 * inter)
                    w13[slot][rows], s13[slot][rows] = packed, scales
    limit = float(cfg.get('swiglu_limit', 0.0) or 0.0)
    return dict(w13=w13, w2=w2, s13=s13, s2=s2, g13=np.ones((experts, 2), np.float32),
                g2=np.ones(experts, np.float32), hidden=hidden, inter=inter, experts=experts,
                fmt=fmt, limit=limit)


def make_batch(rng, m, k, hidden, experts, topk_scale=1.5):
    x = bf16_bits(rng.standard_normal((m, hidden)).astype(np.float32))
    ids = np.stack([rng.permutation(experts)[:k] for _ in range(m)]).astype(np.int32)
    if m > 1:
        ids[1:, :k // 2] = ids[0, :k // 2]
    logits = rng.standard_normal((m, k)).astype(np.float32)
    rw = np.exp(logits - logits.max(-1, keepdims=True))
    rw = (rw / rw.sum(-1, keepdims=True) * topk_scale).astype(np.float32)
    return x, ids, rw


def reference(layer, x, ids, rw):
    """float64 forward, one distinct expert at a time (dequantise, use, drop): memory stays at one expert."""
    hidden, inter, limit, down_bf16 = layer['hidden'], layer['inter'], layer['limit'], DOWN_BF16
    m, k = ids.shape
    y = np.zeros((m, hidden), dtype=np.float64)
    for e in np.unique(ids):
        if e < 0:
            continue
        wg = dequant(layer['w13'][e, :inter], layer['s13'][e, :inter], layer['fmt'])
        wu = dequant(layer['w13'][e, inter:], layer['s13'][e, inter:], layer['fmt'])
        wd = dequant(layer['w2'][e], layer['s2'][e], layer['fmt'])
        for t in range(m):
            for q in range(k):
                if int(ids[t, q]) != e:
                    continue
                xt = from_bf16(x[t]).astype(np.float64)
                gate = wg @ xt * float(layer['g13'][e, 0])
                up = wu @ xt * float(layer['g13'][e, 1])
                if limit > 0:
                    up = np.clip(up, -limit, limit)
                    gate = np.minimum(gate, limit)
                with np.errstate(over='ignore'):
                    mid = gate / (1 + np.exp(-gate)) * up
                if down_bf16:       # the engine keeps the intermediate as BF16 words in this mode
                    mid = from_bf16(bf16_bits(mid.astype(np.float32))).astype(np.float64)
                y[t] += wd @ mid * float(layer['g2'][e]) * float(rw[t, q])
        del wg, wu, wd
    return y


def bind(lib_dir):
    lib = C.CDLL(str(Path(lib_dir).resolve() / 'libxtier_engine.so'))
    lib.xtier_isa.restype = C.c_char_p
    lib.xtier_last_error.restype = C.c_char_p
    lib.xtier_create3.restype = C.c_void_p
    lib.xtier_create3.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int, C.c_float, C.c_int] + [C.c_void_p] * 6
    lib.xtier_forward.restype = C.c_int
    lib.xtier_forward.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
    lib.xtier_destroy.argtypes = [C.c_void_p]
    return lib


def create(lib, layer, threads):
    ptrs = [C.c_void_p(layer[n].ctypes.data) for n in ('w13', 'w2', 's13', 's2', 'g13', 'g2')]
    p = lib.xtier_create3(layer['hidden'], layer['inter'], layer['experts'], threads, C.c_float(layer['limit']),
                          1 if layer['fmt'] == 'mxfp4' else 0, *ptrs)
    if not p:
        raise SystemExit('create failed: ' + lib.xtier_last_error().decode())
    return p


def forward(lib, plan, x, ids, rw):
    y = np.empty((x.shape[0], x.shape[1]), dtype=np.float32)
    if lib.xtier_forward(plan, C.c_int(x.shape[0]), C.c_int(ids.shape[1]), C.c_void_p(x.ctypes.data),
                         C.c_void_p(ids.ctypes.data), C.c_void_p(rw.ctypes.data), C.c_void_p(y.ctypes.data)) < 0:
        raise SystemExit('forward failed: ' + lib.xtier_last_error().decode())
    return y


def compare(a, b):
    bad = 0
    for f in sorted(Path(a).glob('*.npy')):
        other = Path(b) / f.name
        if not other.exists():
            print(f'{f.name}: missing in {b}')
            bad += 1
            continue
        ya, yb = np.load(f), np.load(other)
        if ya.shape != yb.shape:
            print(f'{f.name}: shape {ya.shape} vs {yb.shape}')
            bad += 1
            continue
        d = np.abs(ya - yb).max()
        rms = float(np.linalg.norm(ya - yb) / (np.linalg.norm(yb) + 1e-30))
        ok = rms < 1e-5
        bad += not ok
        print(f'{f.name}: max_abs_diff={d:.3e} rms={rms:.2e} {"ok" if ok else "FAIL"}')
    raise SystemExit(f'{bad} case(s) differ' if bad else 0)


def main():
    if A.compare:
        compare(*A.compare)
    lib = bind(A.lib_dir)
    print(f'[real] kernels={lib.xtier_isa().decode()} model={A.model} layer={A.layer}', flush=True)
    t0 = time.perf_counter()
    layer = load_layer(A.model, A.layer, A.experts, A.names.split(','), A.key, A.format)
    print(f'[real] loaded {A.experts} experts ({A.names}) H={layer["hidden"]} I={layer["inter"]} '
          f'limit={layer["limit"]} in {time.perf_counter() - t0:.1f}s', flush=True)
    plan = create(lib, layer, A.threads)
    hidden, k = layer['hidden'], A.topk
    macs = k * 3 * layer['inter'] * hidden
    out = A.save and Path(A.save) or None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for m in (1, 2, 4, 8, 64, 512):
        x, ids, rw = make_batch(np.random.default_rng(A.seed + m), m, k, hidden, A.experts)
        y = forward(lib, plan, x, ids, rw)
        if out:
            np.save(out / f'm{m}.npy', y)
        if m <= A.ref_tokens:
            ref = reference(layer, x, ids, rw)
            diff = y - ref
            rms = float(np.linalg.norm(diff) / np.linalg.norm(ref))
            rel = float(np.abs(diff).max() / (np.abs(ref).max() + 1e-30))
            # With XTIER_DOWN_BF16 the reference's float64 intermediate lands on the other side of a BF16
            # boundary for about one value in 2^15 (2^-8 for that value), so the gate is looser there.
            ok = rms < (1e-4 if DOWN_BF16 else 1e-5)
            bad += not ok
            print(f'[real] m={m:4d} reference: rms={rms:.2e} max_rel={rel:.2e} {"ok" if ok else "FAIL"}', flush=True)
        times = []
        for _ in range(A.rounds):
            t0 = time.perf_counter()
            forward(lib, plan, x, ids, rw)
            times.append(time.perf_counter() - t0)
        us = statistics.median(times) * 1e6
        print(f'[real] m={m:4d} {us:9.1f} us  {2 * macs * m / (us * 1e-6) / 1e9:7.1f} GFLOP/s', flush=True)
    lib.xtier_destroy(plan)
    if bad:
        raise SystemExit(f'{bad} numeric case(s) failed')


if __name__ == '__main__':
    main()
