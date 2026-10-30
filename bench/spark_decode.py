#!/usr/bin/env python3
"""Decode throughput of one running server: end-to-end tok/s plus the server's own steady-state rate.

The e2e number includes prefill, so it moves with prompt length; the steady number is what the
server itself logs per request ("gen throughput (token/s): ...") and is the honest decode rate.
Point LOG at the server's log file to get the steady number, or leave it unset for e2e only.

    N_TOK=512 ROUNDS=3 python3 bench/spark_decode.py
"""
import json
import os
import re
import subprocess
import sys
import time

URL = os.environ.get('URL', 'http://127.0.0.1:30000')
MODEL = os.environ.get('BENCH_MODEL', 'DeepSeek-V4-Flash')
LOG = os.environ.get('LOG', '')
PROMPT = os.environ.get('BENCH_PROMPT', '请用中文解释热力学第二定律与信息熵的关系，约三百字。')
N_TOK = int(os.environ.get('N_TOK', '256'))
ROUNDS = int(os.environ.get('ROUNDS', '3'))


def one(max_tokens):
    body = json.dumps({'model': MODEL, 'messages': [{'role': 'user', 'content': PROMPT}],
                       'max_tokens': max_tokens, 'temperature': 0})
    t0 = time.time()
    r = subprocess.run(['curl', '-s', '-m', '1800', f'{URL}/v1/chat/completions',
                        '-H', 'Content-Type: application/json', '-d', body],
                       capture_output=True, text=True)
    dt = time.time() - t0
    try:
        d = json.loads(r.stdout)
        return d['usage']['completion_tokens'], dt
    except Exception:
        return None, dt


def steady_rates(n=3):
    if not LOG or not os.path.exists(LOG):
        return []
    out = subprocess.run(['grep', '-a', 'gen throughput', LOG], capture_output=True, text=True).stdout
    vals = [float(m) for m in re.findall(r'gen throughput \(token/s\): ([\d.]+)', out)]
    return vals[-n:]


def main():
    one(16)                      # warm: CUDA graphs, cold experts, page cache
    rates = []
    for _ in range(ROUNDS):
        ct, dt = one(N_TOK)
        if not ct:
            print('BENCH FAIL (request error or empty completion)')
            return 1
        rates.append(ct / dt)
    steady = steady_rates()
    e2e = sorted(rates)[len(rates) // 2]
    print(f'decode: e2e={e2e:.2f} tok/s  server_steady={max(steady) if steady else 0:.2f} tok/s  '
          f'rounds={[round(r, 2) for r in rates]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
