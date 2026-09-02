#!/usr/bin/env python3
"""Dump a K3 greedy-decode fixture from an OpenAI-compatible server (vLLM,
pegainfer) in the shape of pegainfer's `k3_4l_greedy.json`, for
crates/kern-run/examples/k3_golden.rs.

    python3 tools/k3_oracle_dump.py --url http://host:8100 --steps 40 \
        --seed-fixture <pegainfer>/pegainfer-k3/tests/fixtures/k3_4l_greedy.json \
        --out /data/susun/kern-k3/oracle-vllm-93l.json

Teacher-forced: step i sends `feed[:i+1]` as token ids with max_tokens=1,
temperature 0 and top-5 logprobs, records the argmax and top-5 at that
position, and appends the argmax to `feed` once the seed prompt is used up
(so the continuation is the oracle's own greedy path). Logprobs stand in for
logits: the softmax shift cancels in the top-1/top-2 margin the runner uses
for its noise-floor excusal, and the runner treats `top5_logits` as such.
"""
import argparse
import json
import sys
import urllib.request

LOGPROBS = True


def complete(url, ids, model):
    body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0, "return_token_ids": True}
    if LOGPROBS:
        body.update({"logprobs": 5, "return_tokens_as_token_ids": True})
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def token_id(s):
    assert s.startswith("token_id:"), s
    return int(s[len("token_id:"):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8100")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed-fixture", help="take the prompt ids from this fixture")
    ap.add_argument("--prompt-text", help="or tokenize this text through the server's /tokenize")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--layers", type=int, default=93)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-last", type=int, default=0,
                    help="only query the reference for the last N prompt positions (and every continuation "
                         "step); earlier prompt positions are recorded with argmax -1, which the runner "
                         "feeds but does not check — long-prompt fixtures without a request per token")
    ap.add_argument("--no-logprobs", action="store_true",
                    help="server returns no logprobs (pegainfer K3): record the argmax alone, every step must match exactly")
    a = ap.parse_args()
    global LOGPROBS
    LOGPROBS = not a.no_logprobs
    model = a.model
    if model is None:
        with urllib.request.urlopen(a.url + "/v1/models") as r:
            model = json.load(r)["data"][0]["id"]
    if a.seed_fixture:
        prompt = list(json.load(open(a.seed_fixture))["prompt"])
    else:
        req = urllib.request.Request(a.url + "/tokenize", data=json.dumps({"model": model, "prompt": a.prompt_text}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            prompt = json.load(r)["tokens"]
        print(f"prompt: {len(prompt)} tokens", file=sys.stderr)
    feed = list(prompt)
    steps = []
    for i in range(a.steps):
        if i >= len(feed):
            feed.append(steps[-1]["argmax"])
        if a.check_last and i < len(prompt) - a.check_last:
            steps.append({"feed": feed[i], "argmax": -1, "top5_ids": [-1, -1], "top5_logits": [0.0, -1e9]})
            continue
        resp = complete(a.url, feed[:i + 1], model)
        choice = resp["choices"][0]
        if LOGPROBS:
            lp = choice["logprobs"]
            argmax = token_id(lp["tokens"][0])
            top = sorted(((token_id(t), v) for t, v in lp["top_logprobs"][0].items()), key=lambda kv: -kv[1])
            if top[0][0] != argmax:
                print(f"step {i}: sampled {argmax} is not the top logprob {top[0]}", file=sys.stderr)
        else:
            argmax = choice["token_ids"][0]
            top = [(argmax, 0.0), (-1, -1e9)]
        steps.append({"feed": feed[i], "argmax": argmax, "top5_ids": [t for t, _ in top],
                      "top5_logits": [v for _, v in top]})
        print(i, feed[i], "->", argmax, [round(v, 3) for _, v in top[:2]], file=sys.stderr, flush=True)
    json.dump({"num_layers": a.layers, "prompt": prompt, "steps": steps,
               "note": f"teacher-forced greedy from {a.url} ({model}); top5_logits are logprobs, "
                       "so compare margins with the runner's --margin-abs"},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
