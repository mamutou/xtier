#!/usr/bin/env python3
"""Prefill (TTFT) and decode rate at several prompt lengths, measured honestly.

Two traps this script exists to avoid:

1. **The radix/prefix cache.** Re-sending the same text makes prefill look tens of times faster
   than it is - a cache hit, not compute. Every request here starts with a random sentence
   (SALT=1, the default) so the prompt cannot be served from cache.
2. **A cold server.** The first request after startup is several times slower because the hot-expert
   slots are empty and every expert read comes off the drive. Warm up first (--warm) and only then
   believe the numbers.

    LENS=1000,8000,32000 python3 bench/spark_lengths.py --warm
"""
import json
import os
import sys
import time
import urllib.request
import uuid

URL = os.environ.get('URL', 'http://127.0.0.1:30000')
MODEL = os.environ.get('BENCH_MODEL', 'DeepSeek-V4-Flash')
CHARS_PER_TOK = float(os.environ.get('CPT', '1.6'))   # Chinese filler: ~1.6 chars per token
MAX_NEW = int(os.environ.get('MAX_NEW', '64'))
LENS = [int(x) for x in os.environ.get('LENS', '1000,8000,32000').split(',')]
SALT = os.environ.get('SALT', '1') == '1'
CHUNK = '秋天的大海在黄昏时呈现出层次分明的蓝色，浪花拍打礁石的声音由远及近，鸥鸟掠过水面。'


def prompt_for(tokens):
    n = int(tokens * CHARS_PER_TOK)
    body = (CHUNK * (n // len(CHUNK) + 1))[:n]
    salt = f'（文档编号 {uuid.uuid4().hex[:12]}，请只依据本文回答。）\n\n' if SALT else ''
    return salt + body + '\n\n请用一句话概括上文的主题。'


def stream(prompt, max_tokens):
    body = json.dumps({'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
                       'max_tokens': max_tokens, 'temperature': 0, 'stream': True,
                       'stream_options': {'include_usage': True}}).encode()
    req = urllib.request.Request(f'{URL}/v1/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    ttft, usage = None, None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            raw = raw.strip()
            if not raw.startswith(b'data: '):
                continue
            payload = raw[6:]
            if payload == b'[DONE]':
                break
            j = json.loads(payload)
            if j.get('usage'):
                usage = j['usage']
            ch = (j.get('choices') or [{}])[0].get('delta', {}).get('content') or ''
            if ch and ttft is None:
                ttft = time.time() - t0
    return ttft, time.time() - t0, usage


def warm():
    """Fill the hot-expert slots before timing anything."""
    for _ in range(3):
        stream(prompt_for(400), 256)


def main():
    if '--warm' in sys.argv:
        print('# warming up', flush=True)
        warm()
    print(f'# length sweep (salt={SALT})', flush=True)
    for target in LENS:
        try:
            ttft, total, usage = stream(prompt_for(target), MAX_NEW)
        except Exception as e:
            print(f'prompt~{target} FAILED {type(e).__name__}: {str(e)[:160]}', flush=True)
            continue
        ptok = (usage or {}).get('prompt_tokens', target)
        ctok = (usage or {}).get('completion_tokens', 0)
        dec = (ctok - 1) / max(1e-9, total - ttft) if ttft and ctok > 1 else 0.0
        pre = ptok / ttft if ttft else 0.0
        print(f'prompt={ptok:>7d} tok | TTFT {ttft:7.2f}s (prefill {pre:6.0f} tok/s) | '
              f'decode {dec:6.2f} tok/s ({ctok} tok, total {total:.1f}s)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
