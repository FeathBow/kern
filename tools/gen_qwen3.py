#!/usr/bin/env python3
"""Generate examples/qwen3-4b.json.

The manifest is a generated artifact, like Cargo.lock: a provider writes a
generator, not a manifest. Everything the runtime must never know about the
model (head counts, layer count, per-layer state layout) lives here as
plain arithmetic and is baked into opaque numbers before it reaches the
manifest.

Kernel signatures below are placeholders pending the real kernel plan; the
cubin hash is a placeholder for the same reason.
"""

import hashlib
import json
import pathlib

HIDDEN = 2560
LAYERS = 36
HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
FFN = 9728
VOCAB = 151936
EPS = 1e-6

Q_DIM = HEADS * HEAD_DIM        # 4096
KV_DIM = KV_HEADS * HEAD_DIM    # 1024
BF16 = 2

# Opaque state: per token, each layer stores K and V rows (bf16).
# The runtime sees only the total; kernels receive a per-layer byte offset.
KV_LAYER_STRIDE = 2 * KV_DIM * BF16
KV_BYTES_PER_TOKEN = LAYERS * KV_LAYER_STRIDE

MAX_TOKENS = 8192
MAX_POS = 32768


def sym(s):
    return {"sym": s}


def cdiv(e, c):
    return {"ceil_div": [e, c]}


def mul(e, c):
    return {"mul": [e, c]}


def buf(n):
    return {"buf": n}


def i32(v):
    return {"i32": v}


def i64(v):
    return {"i64": v}


def f32(v):
    return {"f32": v}


def state(n):
    return {"state": n}


buffers = {
    "token_ids": {"dtype": "i32", "shape": ["tokens"], "class": "input"},
    "logits": {"dtype": "bf16", "shape": ["tokens", VOCAB], "class": "output"},
}

for name, shape in {
    "hidden": ["tokens", HIDDEN],
    "x_norm": ["tokens", HIDDEN],
    "q": ["tokens", Q_DIM],
    "k": ["tokens", KV_DIM],
    "v": ["tokens", KV_DIM],
    "attn_out": ["tokens", Q_DIM],
    "attn_proj": ["tokens", HIDDEN],
    "gate": ["tokens", FFN],
    "up": ["tokens", FFN],
    "ffn_act": ["tokens", FFN],
    "ffn_down": ["tokens", HIDDEN],
}.items():
    buffers[name] = {"dtype": "bf16", "shape": shape, "class": "workspace"}


def weight(name, shape):
    buffers[name] = {"dtype": "bf16", "shape": shape, "class": "weight"}


weight("model.embed_tokens.weight", [VOCAB, HIDDEN])
weight("model.norm.weight", [HIDDEN])
weight("lm_head.weight", [VOCAB, HIDDEN])
for i in range(LAYERS):
    p = f"model.layers.{i}."
    weight(p + "input_layernorm.weight", [HIDDEN])
    weight(p + "post_attention_layernorm.weight", [HIDDEN])
    weight(p + "self_attn.q_proj.weight", [Q_DIM, HIDDEN])
    weight(p + "self_attn.k_proj.weight", [KV_DIM, HIDDEN])
    weight(p + "self_attn.v_proj.weight", [KV_DIM, HIDDEN])
    weight(p + "self_attn.q_norm.weight", [HEAD_DIM])
    weight(p + "self_attn.k_norm.weight", [HEAD_DIM])
    weight(p + "self_attn.o_proj.weight", [HIDDEN, Q_DIM])
    weight(p + "mlp.gate_proj.weight", [FFN, HIDDEN])
    weight(p + "mlp.up_proj.weight", [FFN, HIDDEN])
    weight(p + "mlp.down_proj.weight", [HIDDEN, FFN])

ATTN_PARAMS = [
    "in buffer<bf16>",   # q
    "in buffer<bf16>",   # k (new tokens)
    "in buffer<bf16>",   # v (new tokens)
    "inout ptr",         # opaque kv state
    "out buffer<bf16>",  # attn out
    "i32",               # tokens
    "i32",               # pos (past length)
    "i64",               # this layer's byte offset within per-token state
]

kernels = {
    "embedding": {
        "symbol": "embedding_bf16",
        "params": ["in buffer<i32>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
        "block": [256, 1, 1],
    },
    "rmsnorm": {
        "symbol": "rmsnorm_bf16",
        "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "f32"],
        "block": [256, 1, 1],
    },
    "matmul": {
        # c[m, n] = a[m, k] @ w[n, k]^T  (weights kept in HF [out, in] layout)
        "symbol": "matmul_bf16",
        "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
        "block": [128, 1, 1],
    },
    "qk_norm_rope": {
        "symbol": "qk_norm_rope_bf16",
        "params": ["inout buffer<bf16>", "inout buffer<bf16>", "in buffer<bf16>",
                   "in buffer<bf16>", "i32", "i32", "i32", "i32", "i32"],
        "block": [128, 1, 1],
    },
    "attn_prefill": {
        "symbol": "attn_prefill_bf16",
        "params": ATTN_PARAMS,
        "block": [128, 1, 1],
    },
    "attn_decode": {
        "symbol": "attn_decode_bf16",
        "params": ATTN_PARAMS,
        "block": [128, 1, 1],
    },
    "silu_mul": {
        "symbol": "silu_mul_bf16",
        "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
        "block": [256, 1, 1],
    },
    "add_residual": {
        "symbol": "add_residual_bf16",
        "params": ["inout buffer<bf16>", "in buffer<bf16>", "i32", "i32"],
        "block": [256, 1, 1],
    },
}


def d(label, kernel, grid, args):
    return {"label": label, "kernel": kernel, "grid": grid, "args": args}


def matmul(label, a, w, c, n, k):
    return d(label, "matmul",
             [cdiv(sym("tokens"), 64), n // 128, 1],
             [buf(a), buf(w), buf(c), sym("tokens"), i32(n), i32(k)])


def rmsnorm(label, x, w, out):
    return d(label, "rmsnorm",
             [sym("tokens"), 1, 1],
             [buf(x), buf(w), buf(out), sym("tokens"), i32(HIDDEN), f32(EPS)])


def add_residual(label, x):
    return d(label, "add_residual",
             [cdiv(mul(sym("tokens"), HIDDEN), 1024), 1, 1],
             [buf("hidden"), buf(x), sym("tokens"), i32(HIDDEN)])


def layer(i, attn_kernel):
    p = f"model.layers.{i}."
    l = f"l{i}."
    return [
        rmsnorm(l + "attn_norm", "hidden", p + "input_layernorm.weight", "x_norm"),
        matmul(l + "q_proj", "x_norm", p + "self_attn.q_proj.weight", "q", Q_DIM, HIDDEN),
        matmul(l + "k_proj", "x_norm", p + "self_attn.k_proj.weight", "k", KV_DIM, HIDDEN),
        matmul(l + "v_proj", "x_norm", p + "self_attn.v_proj.weight", "v", KV_DIM, HIDDEN),
        d(l + "qk_norm_rope", "qk_norm_rope",
          [sym("tokens"), HEADS, 1],
          [buf("q"), buf("k"), buf(p + "self_attn.q_norm.weight"),
           buf(p + "self_attn.k_norm.weight"), sym("tokens"), sym("pos"),
           i32(HEADS), i32(KV_HEADS), i32(HEAD_DIM)]),
        d(l + "attn", attn_kernel,
          [sym("tokens"), HEADS, 1],
          [buf("q"), buf("k"), buf("v"), state("kv"), buf("attn_out"),
           sym("tokens"), sym("pos"), i64(i * KV_LAYER_STRIDE)]),
        matmul(l + "o_proj", "attn_out", p + "self_attn.o_proj.weight", "attn_proj", HIDDEN, Q_DIM),
        add_residual(l + "attn_res", "attn_proj"),
        rmsnorm(l + "mlp_norm", "hidden", p + "post_attention_layernorm.weight", "x_norm"),
        matmul(l + "gate_proj", "x_norm", p + "mlp.gate_proj.weight", "gate", FFN, HIDDEN),
        matmul(l + "up_proj", "x_norm", p + "mlp.up_proj.weight", "up", FFN, HIDDEN),
        d(l + "silu_mul", "silu_mul",
          [cdiv(mul(sym("tokens"), FFN), 1024), 1, 1],
          [buf("gate"), buf("up"), buf("ffn_act"), sym("tokens"), i32(FFN)]),
        matmul(l + "down_proj", "ffn_act", p + "mlp.down_proj.weight", "ffn_down", HIDDEN, FFN),
        add_residual(l + "mlp_res", "ffn_down"),
    ]


programs = {}
for prog, attn_kernel in [("prefill", "attn_prefill"), ("decode", "attn_decode")]:
    ds = [d("embed", "embedding",
            [cdiv(sym("tokens"), 256), 1, 1],
            [buf("token_ids"), buf("model.embed_tokens.weight"), buf("hidden"),
             sym("tokens"), i32(HIDDEN)])]
    for i in range(LAYERS):
        ds += layer(i, attn_kernel)
    ds.append(rmsnorm("final_norm", "hidden", "model.norm.weight", "x_norm"))
    ds.append(matmul("lm_head", "x_norm", "lm_head.weight", "logits", VOCAB, HIDDEN))
    programs[prog] = {"dispatches": ds}

manifest = {
    "meta": {
        "version": 1,
        "model": "qwen3-4b",
        "cubin": {
            "file": "kernels.cubin",
            "sha256": hashlib.sha256(b"placeholder: kernels not built yet").hexdigest(),
        },
    },
    "symbols": {
        "tokens": {"max": MAX_TOKENS},
        "pos": {"min": 0, "max": MAX_POS},
    },
    "states": {
        "kv": {"bytes_per_token": KV_BYTES_PER_TOKEN},
    },
    "buffers": buffers,
    "kernels": kernels,
    "programs": programs,
}

out = pathlib.Path(__file__).resolve().parent.parent / "examples" / "qwen3-4b.json"
out.write_text(json.dumps(manifest, indent=1) + "\n")
n_dispatch = sum(len(p["dispatches"]) for p in programs.values())
print(f"wrote {out} ({out.stat().st_size // 1024} KiB, "
      f"{len(buffers)} buffers, {n_dispatch} dispatches, "
      f"kv {KV_BYTES_PER_TOKEN} B/token)")
