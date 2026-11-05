"""Response tests for an OpenAI-compatible vLLM server. Stdlib only, so it runs in any Python.

    python vllm_test/vllm_client_test.py --base-url http://<node>:<port>/v1 --model <served name>

API key is read from $VLLM_API_KEY. Exits 0 only if every test passes.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def request(url, key, body=None, timeout=120):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body is not None else None
    return urllib.request.urlopen(urllib.request.Request(url, data, headers), timeout=timeout)


def chat_body(model, prompt, **kw):
    return {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 32, **kw}


def test_models(a):
    with request(f"{a.base_url}/models", a.key) as r:
        ids = [m["id"] for m in json.load(r)["data"]]
    assert a.model in ids, f"{a.model!r} not in served models {ids}"
    return f"served models: {ids}"


def test_auth(a):
    try:
        request(f"{a.base_url}/models", "wrong-key").close()
    except urllib.error.HTTPError as e:
        assert e.code == 401, f"expected 401, got {e.code}"
        return "wrong key rejected with 401"
    raise AssertionError("request with a wrong API key succeeded")


def test_chat(a):
    t = time.perf_counter()
    with request(f"{a.base_url}/chat/completions", a.key,
                 chat_body(a.model, "What is 2 + 3? Reply with only the number.")) as r:
        resp = json.load(r)
    text = resp["choices"][0]["message"]["content"] or ""
    assert text.strip(), "empty reply"
    return f"{time.perf_counter() - t:.2f}s reply={text.strip()!r} usage={resp.get('usage')}"


def test_stream(a):
    t = time.perf_counter()
    first, parts = None, []
    with request(f"{a.base_url}/chat/completions", a.key,
                 chat_body(a.model, "Count from 1 to 5.", stream=True)) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
            if delta:
                first = first or time.perf_counter() - t
                parts.append(delta)
    assert parts, "stream returned no content"
    return f"ttft={first:.2f}s total={time.perf_counter() - t:.2f}s chunks={len(parts)} text={''.join(parts).strip()!r}"


def test_concurrent(a):
    def one(i):
        with request(f"{a.base_url}/chat/completions", a.key,
                     chat_body(a.model, f"Write one short sentence about the number {i}.", max_tokens=64)) as r:
            return json.load(r)["usage"]["completion_tokens"]

    t = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as pool:
        tokens = sum(pool.map(one, range(a.concurrency)))
    dt = time.perf_counter() - t
    return f"{a.concurrency} requests in {dt:.2f}s, {tokens} tokens, {tokens / dt:.1f} tok/s"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True, help="e.g. http://node:8000/v1")
    p.add_argument("--model", required=True, help="served model name")
    p.add_argument("--concurrency", type=int, default=8)
    a = p.parse_args()
    a.base_url = a.base_url.rstrip("/")
    a.key = os.environ.get("VLLM_API_KEY", "")

    tests = [test_models, test_chat, test_stream, test_concurrent]
    if a.key:  # vLLM only enforces auth when started with --api-key
        tests.insert(1, test_auth)

    failed = 0
    for test in tests:
        name = test.__name__.removeprefix("test_")
        try:
            print(f"[PASS] {name}: {test(a)}", flush=True)
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
