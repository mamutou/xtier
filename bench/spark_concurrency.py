#!/usr/bin/env python3
"""Aggregate throughput at N concurrent requests, to answer "does batching help here?".

Each round fires CONC requests in parallel and reports aggregate completion tok/s plus the average
per-request latency. Compare CONC=1 against CONC=4 / 8: on one GB10 the aggregate climbs and then
plateaus at what the CPU expert engine can compute, while per-request latency grows.

Requires the server to have been started with --max-running-requests high enough, and *with* warmup
(see deploy/spark/README.md - without it, concurrency above 2 fails during CUDA-graph capture).

    CONC=8 ROUNDS=2 N_TOK=384 python3 bench/spark_concurrency.py
"""
import json
import os
import threading
import time
import urllib.request

URL = os.environ.get('URL', 'http://127.0.0.1:30000')
MODEL = os.environ.get('BENCH_MODEL', 'DeepSeek-V4-Flash')
CONC = int(os.environ.get('CONC', '4'))
ROUNDS = int(os.environ.get('ROUNDS', '2'))
N_TOK = int(os.environ.get('N_TOK', '384'))
PROMPT = os.environ.get('BENCH_PROMPT', '用中文写一段关于海洋与气候的科普文字。')


def one(out, idx):
    body = json.dumps({'model': MODEL, 'messages': [{'role': 'user', 'content': PROMPT}],
                       'max_tokens': N_TOK, 'temperature': 0}).encode()
    req = urllib.request.Request(f'{URL}/v1/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=3600))
        out[idx] = (d['usage']['completion_tokens'], time.time() - t0)
    except Exception as e:
        out[idx] = (0, time.time() - t0)


def round_once():
    out = [None] * CONC
    threads = [threading.Thread(target=one, args=(out, i)) for i in range(CONC)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    toks = sum(o[0] for o in out if o)
    lats = [o[1] for o in out if o]
    return toks, wall, (sum(lats) / len(lats) if lats else 0)


def main():
    round_once()                                  # warm
    agg = []
    for i in range(ROUNDS):
        toks, wall, lat = round_once()
        rate = toks / wall if wall else 0.0
        agg.append(rate)
        print(f'  c{CONC} round{i + 1}: {toks} tok in {wall:.1f}s -> {rate:.2f} agg tok/s, '
              f'avg lat {lat:.1f}s', flush=True)
    print(f'CONC={CONC} aggregate={sum(agg) / len(agg):.2f} tok/s  detail={[round(a, 2) for a in agg]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
