#!/usr/bin/env python3
"""Generate examples/qwen3.8-27b.json from the vLLM capture of Qwen3.8-27B.

Qwen3.8-27B is a Qwen3.5-family hybrid: 64 layers, 48 gated-delta-net
(GDN, "linear attention") layers and 16 full-attention layers (every 4th),
gated attention output, Gemma-style norms, mrope.  bs=1 manifest, two
programs sized by the `tokens` symbol:

- `prefill` (tokens ∈ [1, CHUNK_MAX]): the vLLM chunked-prefill forward —
  KV writes for the attention layers, conv/SSM state carry for the GDN
  layers — *plus* the final norm / lm_head / argmax on the last row.  Unlike
  the qwen3-4b manifests, prefill emits `next_token`: the GDN prefill path
  (FLA chunk kernels) and the decode path (recurrent kernel) are different
  arithmetic, and vLLM runs the last prompt token through the chunk path.
  Putting it through `decode` instead would change bits.  The driver
  (kern-run) sees the output buffer in the program and prefills all prompt
  tokens.
- `decode` (tokens = 1): one token, recurrent GDN kernels, split-KV
  attention, logits + argmax.

Data source: the TRITON_ATTN + `gdn_prefill_backend=triton` capture (the
only flat-ABI kernel set: every kernel is a plain CUDA/Triton launch).  The
generator takes launch geometry, scalar literals and instance identity from
the capture, and asserts the wiring it hand-writes against the captured
pointers (buffer identities between consecutive kernels) and the grid
formulas against four prefill forwards of different lengths.

What the runtime cannot express and had to be handwritten (all bit-exact
against vLLM's own ops, tools/test_kernels_qwen35.py):

- `gemma_rms_norm.cu`: GemmaRMSNorm is a chain of ATen ops in vLLM (pow /
  mean / rsqrt / mul); the kernel reproduces ATen's reduction order.
- `sigmoid_mul.cu`: `attn_out * sigmoid(gate)` (two ATen ops).
- `copy_rows.cu`: the strided z-gate copy (`layer_norm_fwd_kernel` bakes
  `%16` assumptions into its strides, so the [T,16384] view cannot be fed
  directly) and the last-row select for the prefill tail (`tokens-1` is not
  in the manifest expression set).

Triton instances: same-symbol kernels compiled for different
specializations (an int arg == 1, or % 16) have identical param layouts, so
the runtime's layout matching cannot pick one.  Every Triton step is pinned
to a dump module by sha256; the module is chosen by matching the dump
against the Triton cache and reading the `.ttir` signature — the least
specialized instance (most runtime args, fewest divisibility attributes)
among those with the launch's register count.  A generic instance computes
the same values as a specialized one; it just makes fewer assumptions.

State layout (baked into the kernels by vLLM's constexprs):

- GDN: one 3211264-B line per layer = [conv state 10240×3 bf16 | SSM state
  48×128×128 f32 | pad]; line 0 is the null line (kernels skip index <= 0),
  layer i uses line i+1; indices come from a constant table via byte
  offsets.  `bytes_fixed` state (per sequence, not per token) — the one
  schema extension this bring-up needed.
- KV: BLOCK_SIZE = 784 (vLLM aligned it to the mamba page), k/v interleaved
  per head, layers interleaved per page: [page][16][784][4][K|V][256],
  bytes_per_token = 65536.

Usage: python3 tools/gen_qwen35.py [dump_dir] [out.json]
"""

import hashlib
import json
import os
import pathlib
import re
import struct
import subprocess
import sys

# --- model geometry (config.json; asserted against the capture below)
HIDDEN = 5120
LAYERS = 64
FFN = 17408
VOCAB = 248320
HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
Q_DIM = HEADS * HEAD_DIM                 # 6144
KV_DIM = KV_HEADS * HEAD_DIM             # 1024
QKV_DIM = HEADS * 2 * HEAD_DIM + 2 * KV_DIM   # 14336: per head [q | gate], then k, v
GATE_OFF = HEAD_DIM                      # gate half inside a [q | gate] head pair
ROT_HALF = 32                            # rotary_dim 64
MAX_POS = 8192                           # exported rope table rows
# GDN
GDN_V_HEADS = 48
GDN_K_HEADS = 16
GDN_D = 128
GDN_Q = GDN_K_HEADS * GDN_D              # 2048
GDN_V = GDN_V_HEADS * GDN_D              # 6144
CONV_DIM = 2 * GDN_Q + GDN_V             # 10240 (q, k, v go through the conv)
QKVZ_DIM = CONV_DIM + GDN_V              # 16384 (+ z)
Z_OFF = CONV_DIM                         # z columns start after q|k|v
BA_DIM = 2 * GDN_V_HEADS                 # 96: [b | a]
FLA_CHUNK = 64
CONV_BLOCK_M = 8
POST_CONV_BLOCK = 16
LN_ROWS_PER_BLOCK = 4                    # prefill layer_norm instance
BF16 = 2
ATTN_LAYERS = [i for i in range(LAYERS) if (i + 1) % 4 == 0]
GDN_LAYERS = [i for i in range(LAYERS) if (i + 1) % 4 != 0]
CHUNK_MAX = 2048
NT_MAX = CHUNK_MAX // FLA_CHUNK          # 32 chunks
# attention
BLOCK_SIZE = 784                         # vLLM block_size (constexpr in the kernels)
BLOCK_Q = 5                              # unified 2D: query rows per block
NUM_SEGMENTS = 16                        # unified 3D: grid.z
BLOCK_TABLE_LEN = 8                      # vLLM block_table_stride
BLOCK_ELEMS_PER_LAYER = BLOCK_SIZE * KV_HEADS * 2 * HEAD_DIM   # 1605632
LAYER_KV_BYTES = BLOCK_ELEMS_PER_LAYER * BF16                  # 3211264
BLOCK_STRIDE = len(ATTN_LAYERS) * BLOCK_ELEMS_PER_LAYER        # elems
KV_BYTES_PER_TOKEN = len(ATTN_LAYERS) * KV_HEADS * 2 * HEAD_DIM * BF16  # 65536
V_BYTE_OFF = HEAD_DIM * BF16             # v = k + 256 elems inside a head
# GDN state lines
CONV_STATE_BYTES = CONV_DIM * 3 * BF16   # 61440
SSM_STATE_BYTES = GDN_V_HEADS * GDN_D * GDN_D * 4   # 3145728
GDN_LINE_BYTES = 3211264                 # conv + ssm + 4096 pad (vLLM page)
GDN_LINES = len(GDN_LAYERS) + 1          # line 0 = null
GDN_STATE_BYTES = GDN_LINES * GDN_LINE_BYTES

TRITON = {
    "conv_fwd": "_causal_conv1d_fwd_kernel",
    "conv_update": "_causal_conv1d_update_kernel",
    "post_conv": "_fused_post_conv_kernel",
    "cumsum": "chunk_local_cumsum_scalar_kernel",
    "kkt": "chunk_scaled_dot_kkt_fwd_kernel",
    "solve_tril": "merge_16x16_to_64x64_inverse_kernel",
    "recompute": "recompute_w_u_fwd_kernel",
    "chunk_h": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_o": "chunk_fwd_kernel_o",
    "recurrent": "fused_recurrent_gated_delta_rule_packed_decode_kernel",
    "layer_norm": "layer_norm_fwd_kernel",
    "mrope": "_triton_mrope_forward",
    "cache": "reshape_and_cache_kernel_flash",
    "unified": "kernel_unified_attention",
    "reduce": "reduce_segments",
}


# ----------------------------------------------------------------- capture
def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def forwards(recs):
    """Forward = launches up to and including the sampler's ArgMax reduce."""
    out, start = [], 0
    for i, r in enumerate(recs):
        if "ArgMaxOps" in r["symbol"]:
            out.append(recs[start:i + 1])
            start = i + 1
    return out


def tokens_of(fwd):
    for r in fwd:
        if r["symbol"] == TRITON["mrope"]:
            return r["grid"][0]
    return None


def group(fwd):
    by = {k: [] for k in TRITON}
    by["silu"] = []
    for r in fwd:
        s = r["symbol"]
        if not isinstance(r.get("params"), list):
            continue
        if "act_and_mul" in s:
            by["silu"].append(r)
            continue
        for tag, sym in TRITON.items():
            if s == sym:
                by[tag].append(r)
                break
    return by


def pv(rec, i):
    return int.from_bytes(bytes.fromhex(rec["params"][i]["data"]), "little")


def pf(rec, i):
    return struct.unpack("<f", bytes.fromhex(rec["params"][i]["data"]))[0]


def cdiv_i(a, b):
    return -(-a // b)


# ------------------------------------------------------ instance selection
def cuobjdump():
    return str(pathlib.Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "cuobjdump")


def module_functions(mod):
    """{function: regs} for a cubin."""
    out = subprocess.run([cuobjdump(), "-res-usage", str(mod)],
                         capture_output=True, text=True).stdout
    fns, cur = {}, None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Function "):
            cur = s.split()[1].rstrip(":")
        elif cur and "REG:" in s:
            fns[cur] = int(s.split("REG:")[1].split()[0])
            cur = None
    return fns


def triton_cache_by_sha():
    root = pathlib.Path(os.environ.get("TRITON_CACHE_DIR", pathlib.Path.home() / ".triton" / "cache"))
    idx = {}
    for cb in root.glob("*/*.cubin"):
        idx[hashlib.sha256(cb.read_bytes()).hexdigest()] = cb
    return idx


def ttir_signature(ttir_path, symbol):
    """(runtime int args, int args carrying specialization attributes)."""
    text = ttir_path.read_text()
    m = re.search(r"tt\.func public @%s\((.*?)\)\s*attributes" % re.escape(symbol), text, re.S)
    assert m, f"no tt.func signature in {ttir_path}"
    sig = re.sub(r'loc\("[^"]*"\([^)]*\)\)', "", m.group(1))
    sig = re.sub(r"loc\([^)]*\)", "", sig)
    ints, attrs = 0, 0
    for arg in re.split(r",\s*(?=%)", sig):
        arg = arg.strip()
        if not arg:
            continue
        if "!tt.ptr" in arg:
            continue
        ints += 1
        if "{" in arg:
            attrs += 1
    return ints, attrs


class Pinner:
    """Pick the dump module for a Triton symbol: register count from the
    launch narrows to the constexpr instance, the Triton cache's .ttir picks
    the least specialized runtime-arg instance among the survivors."""

    def __init__(self, dump_dir):
        self.dump = pathlib.Path(dump_dir)
        self.mods = {}
        for mod in sorted(self.dump.glob("module_*.cubin")):
            fns = module_functions(mod)
            if fns:
                self.mods[mod] = fns
        self.shas = {m: hashlib.sha256(m.read_bytes()).hexdigest() for m in self.mods}
        self.cache = triton_cache_by_sha()

    def pin(self, symbol, regs):
        cands = {}
        for mod, fns in self.mods.items():
            if fns.get(symbol) == regs:
                cands.setdefault(self.shas[mod], mod)  # dedupe repeated loads
        assert cands, f"{symbol} REG={regs}: no dump module"
        if len(cands) == 1:
            sha, mod = next(iter(cands.items()))
            return mod.name, sha
        scored = []
        for sha, mod in cands.items():
            cb = self.cache.get(sha)
            assert cb, f"{symbol}: dump module {mod.name} not in the Triton cache, cannot read its signature"
            ints, attrs = ttir_signature(cb.with_suffix(".ttir"), symbol)
            scored.append(((-ints, attrs), sha, mod))
        scored.sort()
        assert scored[0][0] != scored[1][0], f"{symbol}: instances {scored} not separable"
        return scored[0][2].name, scored[0][1]


# ------------------------------------------------------------ verification
def check_prefill(by, T):
    """Falsify the hand-written GDN/attention wiring against one prefill
    forward's pointers and scalars."""
    n_gdn, n_attn = len(GDN_LAYERS), len(ATTN_LAYERS)
    for tag in ("conv_fwd", "post_conv", "cumsum", "kkt", "solve_tril", "recompute",
                "chunk_h", "chunk_o", "layer_norm"):
        assert len(by[tag]) == n_gdn, (tag, len(by[tag]))
    for tag in ("mrope", "cache", "unified"):
        assert len(by[tag]) == n_attn, (tag, len(by[tag]))
    assert len(by["silu"]) == LAYERS and len(by["reduce"]) == 0
    assert len(by["unified"][0]["params"]) == 28, "not the 2D prefill instance"
    for i in range(n_gdn):
        conv, pc, cs, kkt = by["conv_fwd"][i], by["post_conv"][i], by["cumsum"][i], by["kkt"][i]
        st, rc, ch, co, ln = by["solve_tril"][i], by["recompute"][i], by["chunk_h"][i], by["chunk_o"][i], by["layer_norm"][i]
        assert conv["grid"] == [cdiv_i(T, CONV_BLOCK_M), CONV_DIM // 256, 1], conv["grid"]
        assert [pv(conv, 10), pv(conv, 11)] == [QKVZ_DIM, CONV_DIM], "conv strides"
        assert pv(pc, 0) == pv(conv, 8), "post_conv input is not the conv output"
        assert pc["grid"] == [cdiv_i(T, POST_CONV_BLOCK), GDN_V_HEADS + GDN_K_HEADS, 1], pc["grid"]
        assert [pv(pc, j) for j in range(10, 17)] == \
            [CONV_DIM, GDN_V_HEADS, GDN_V_HEADS, GDN_Q, GDN_Q, GDN_V, T], "post_conv strides/L"
        nt = cdiv_i(T, FLA_CHUNK)
        for r in (cs, kkt, st, rc):
            assert r["grid"] == [nt, GDN_V_HEADS, 1], (r["symbol"], r["grid"])
        assert ch["grid"] == [4, GDN_V_HEADS, 1] and co["grid"] == [2, nt, GDN_V_HEADS]
        assert pv(cs, 0) == pv(pc, 8) and pv(kkt, 1) == pv(pc, 9), "g/beta not from post_conv"
        assert pv(kkt, 0) == pv(pc, 6) == pv(rc, 0) == pv(ch, 0) == pv(co, 1), "k drift"
        assert pv(kkt, 2) == pv(cs, 1) == pv(rc, 6) == pv(ch, 4) == pv(co, 4), "cumsum g drift"
        assert pv(st, 0) == pv(kkt, 3) and pv(rc, 5) == pv(st, 1), "A / Ai chain"
        assert pv(rc, 4) == pv(kkt, 3), "vLLM aliases u onto A (same size); we give u its own buffer"
        assert pv(rc, 1) == pv(pc, 7), "recompute v is not post_conv v"
        assert pv(ch, 1) == pv(rc, 4) and pv(ch, 2) == pv(rc, 3), "chunk_h v/w"
        assert pv(co, 0) == pv(pc, 5) and pv(co, 2) == pv(ch, 3) and pv(co, 3) == pv(ch, 5), "chunk_o q/v_new/h"
        assert pv(ch, 6) != pv(ch, 7), "vLLM h0/ht are temporaries; kern aliases both onto the state line"
        for r, j in ((cs, 2), (kkt, 4), (st, 2), (rc, 7), (ch, 8), (co, 6), (conv, 5)):
            assert pv(r, j) == pv(cs, 2), "cu_seqlens drift"
        for r, j in ((cs, 3), (kkt, 5), (st, 3), (rc, 8), (co, 7)):
            assert pv(r, j) == pv(cs, 3), "chunk_indices drift"
        for r, j in ((cs, 4), (kkt, 6), (st, 4), (rc, 9), (ch, 10), (co, 9)):
            assert pv(r, j) == T, "T arg"
        assert pv(ln, 0) == pv(ln, 1), "layer_norm not in place"
        assert [pv(ln, j) for j in (5, 6, 7, 8)] == [GDN_D, GDN_D, GDN_D, T * GDN_V_HEADS]
        assert ln["grid"] == [cdiv_i(T * GDN_V_HEADS, LN_ROWS_PER_BLOCK), 1, 1]
    for i in range(n_attn):
        mr, ca, un = by["mrope"][i], by["cache"][i], by["unified"][i]
        assert mr["grid"] == [T, 1, 1] and ca["grid"] == [T, 1, 1]
        assert pv(mr, 4) == T, "mrope num_tokens"
        assert pv(ca, 0) == pv(mr, 1), "cache key is not the roped k"
        assert pv(un, 1) == pv(mr, 0), "attention query is not the roped q"
        assert pv(ca, 3) - pv(ca, 2) == V_BYTE_OFF and pv(un, 2) == pv(ca, 2) and pv(un, 3) == pv(ca, 3)
        assert [pv(ca, j) for j in range(7, 16)] == \
            [KV_DIM, QKV_DIM, BLOCK_ELEMS_PER_LAYER, 2 * HEAD_DIM, 0, 0, KV_HEADS * 2 * HEAD_DIM, 0, 0]
        assert un["grid"] == [T // BLOCK_Q + 1, KV_HEADS, 1], un["grid"]
        assert pv(un, 17) == pv(un, 5), "rswa_prefix_lens is not the seq_lens dup"
        assert [pv(un, j) for j in range(11, 17)] == [BLOCK_TABLE_LEN, Q_DIM, HEAD_DIM, Q_DIM, HEAD_DIM, 0]
        assert [pv(un, j) for j in range(18, 24)] == [BLOCK_ELEMS_PER_LAYER, KV_HEADS * 2 * HEAD_DIM, 2 * HEAD_DIM] * 2
        assert pv(un, 25) == 1 and pv(un, 7) == pv(ca, 5) and pv(un, 8) == pv(ca, 6)
    assert len(by["silu"][0]["params"]) == 6 and pv(by["silu"][0], 2) == FFN


def check_decode(by):
    n_gdn, n_attn = len(GDN_LAYERS), len(ATTN_LAYERS)
    for tag in ("conv_update", "recurrent", "layer_norm"):
        assert len(by[tag]) == n_gdn, (tag, len(by[tag]))
    for tag in ("mrope", "cache", "unified", "reduce"):
        assert len(by[tag]) == n_attn, (tag, len(by[tag]))
    assert len(by["unified"][0]["params"]) == 31 and len(by["reduce"][0]["params"]) == 12
    for i in range(n_gdn):
        cu, rec, ln = by["conv_update"][i], by["recurrent"][i], by["layer_norm"][i]
        assert pv(cu, 0) == pv(cu, 4) == pv(rec, 0), "conv update not in place on the qkvz row"
        assert [pv(cu, j) for j in (5, 7, 8)] == [1, 1, 1]
        assert pv(cu, 3) == pv(rec, 8), "conv/ssm index tables differ"
        assert pv(rec, 6) == pv(rec, 7), "decode h0 != ht"
        assert pv(rec, 1) - pv(rec, 2) == GDN_V_HEADS * BF16, "ba layout is not [b | a]"
        assert pv(ln, 0) == pv(ln, 1) == pv(rec, 5), "layer_norm not in place on the recurrent output"
        assert pv(ln, 3) == pv(rec, 0) + Z_OFF * BF16, "z is not the qkvz row tail"
        assert [pv(ln, j) for j in (5, 6, 7, 8)] == [GDN_D, GDN_D, GDN_D, GDN_V_HEADS]
        assert ln["grid"] == [GDN_V_HEADS, 1, 1]
    for i in range(n_attn):
        un, rd = by["unified"][i], by["reduce"][i]
        assert un["grid"] == [1, KV_HEADS, NUM_SEGMENTS]
        assert [pv(rd, j) for j in (1, 2, 3)] == [pv(un, j) for j in (26, 27, 28)]
        assert pv(rd, 0) == pv(un, 0) and pv(rd, 4) == pv(un, 5) and pv(rd, 9) == pv(un, 24)
        assert [pv(rd, j) for j in (6, 7, 8)] == [Q_DIM, HEAD_DIM, BLOCK_TABLE_LEN]
    assert len(by["mrope"][0]["params"]) == 6, "decode mrope should be the num_tokens==1 instance"


# ---------------------------------------------------------------- builders
def sym(s):
    return {"sym": s}


def mul(e, c):
    return {"mul": [e, c]}


def cdiv(e, c):
    return {"ceil_div": [e, c]}


def expr(e):
    return {"expr": e}


def buf(n, off=0):
    return {"buf": n, "offset": off} if off else {"buf": n}


def state(n, off=0):
    return {"state": n, "offset": off} if off else {"state": n}


def i32(v):
    return {"i32": v}


def i64(v):
    return {"i64": v}


def f32(v):
    return {"f32": v}


def d(label, kernel, args):
    return {"label": label, "kernel": kernel, "args": args}


def a(i):
    return {"arg": i}


def scr(name, off=0):
    return {"scratch": name, "offset": off} if off else {"scratch": name}


def step(symbol, params, block, grid, args, shared_mem=None, cubin=None, sha256=None):
    s = {"symbol": symbol, "params": params, "block": block, "grid": grid, "args": args}
    if shared_mem is not None:
        s["shared_mem"] = shared_mem
    if cubin is not None:
        s["cubin"] = cubin
    if sha256 is not None:
        s["sha256"] = sha256
    return s


def single(symbol, params, block, grid, shared_mem=None, cubin=None, sha256=None):
    return {"params": params,
            "impl": {"steps": [step(symbol, params, block, grid, [a(i) for i in range(len(params))],
                                    shared_mem, cubin, sha256)]}}


TOKEN_DOMAIN = {"index_into": "model.embed_tokens.weight"}
DOMAINS = {
    "token_ids": TOKEN_DOMAIN,
    "positions": {"index_into": "rope.cos"},
    "slot_mapping": {"index_into": "kv"},
    "block_table": {"index_into": "kv", "unit": BLOCK_SIZE},
    "seq_lens": {"min": 1},
    "cu_seqlens_q": {"min": 0, "max": {"sym": "tokens"}, "monotone": True},
    "next_token": TOKEN_DOMAIN,
    "kv_scales": {"min": 0.0},
}

I2 = ["i64", "i64"]   # Triton's trailing global/profile scratch pointers (always 0)


def build(pre, dec, pins, eps, attn_scale, gdn_scale, silu_sym):
    T = sym("tokens")
    buffers = {
        "token_ids": {"dtype": "i64", "shape": ["tokens"], "class": "input"},
        "positions": {"dtype": "i64", "shape": ["tokens"], "class": "input"},
        "slot_mapping": {"dtype": "i64", "shape": ["tokens"], "class": "input"},
        "block_table": {"dtype": "i32", "shape": [BLOCK_TABLE_LEN], "class": "input"},
        "seq_lens": {"dtype": "i32", "shape": [1], "class": "input"},
        "cu_seqlens_q": {"dtype": "i32", "shape": [2], "class": "input"},
        "logits": {"dtype": "bf16", "shape": [1, VOCAB], "class": "workspace"},
        "next_token": {"dtype": "i64", "shape": [1], "class": "output"},
    }
    ws = {
        "residual": ["tokens", HIDDEN], "x": ["tokens", HIDDEN], "y": ["tokens", HIDDEN],
        "final_x": [1, HIDDEN],
        # GDN
        "qkvz": ["tokens", QKVZ_DIM], "ba": ["tokens", BA_DIM], "conv_out": ["tokens", CONV_DIM],
        "gdn_q": ["tokens", GDN_Q], "gdn_k": ["tokens", GDN_Q], "gdn_v": ["tokens", GDN_V],
        "Ai": ["tokens", GDN_V_HEADS, FLA_CHUNK],
        "w": ["tokens", GDN_V], "u": ["tokens", GDN_V], "v_new": ["tokens", GDN_V],
        "h": [NT_MAX, GDN_V_HEADS, GDN_D, GDN_D],
        "core_attn_out": ["tokens", GDN_V], "z_c": ["tokens", GDN_V],
        # attention
        "qkv": ["tokens", QKV_DIM], "q_n": ["tokens", Q_DIM], "k_n": ["tokens", KV_DIM],
        "cos_g": ["tokens", ROT_HALF], "sin_g": ["tokens", ROT_HALF],
        "attn_out": ["tokens", Q_DIM], "gated": ["tokens", Q_DIM],
        # MLP
        "gate_up": ["tokens", 2 * FFN], "act": ["tokens", FFN],
    }
    for name, shape in ws.items():
        buffers[name] = {"dtype": "bf16", "shape": shape, "class": "workspace"}
    for name, shape in {"g": ["tokens", GDN_V_HEADS], "beta": ["tokens", GDN_V_HEADS],
                        "g_cum": ["tokens", GDN_V_HEADS], "A": ["tokens", GDN_V_HEADS, FLA_CHUNK]}.items():
        buffers[name] = {"dtype": "f32", "shape": shape, "class": "workspace"}

    def weight(name, shape, dtype="bf16"):
        buffers[name] = {"dtype": dtype, "shape": shape, "class": "weight"}

    weight("model.embed_tokens.weight", [VOCAB, HIDDEN])
    weight("lm_head.weight", [VOCAB, HIDDEN])
    weight("model.norm.weight_p1", [HIDDEN], "f32")
    weight("rope.cos", [MAX_POS, ROT_HALF])
    weight("rope.sin", [MAX_POS, ROT_HALF])
    weight("kv_scales", [2], "f32")
    weight("fla.chunk_indices", [NT_MAX, 2], "i32")
    weight("fla.chunk_offsets", [2], "i64")
    weight("conv.batch_ptr", [CHUNK_MAX // CONV_BLOCK_M], "i32")
    weight("conv.token_chunk_offset", [CHUNK_MAX // CONV_BLOCK_M], "i32")
    weight("gdn.line_index", [64], "i32")
    weight("gdn.has_initial", [16], "u8")
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        weight(p + "input_layernorm.weight_p1", [HIDDEN], "f32")
        weight(p + "post_attention_layernorm.weight_p1", [HIDDEN], "f32")
        weight(p + "mlp.gate_up_proj.weight", [2 * FFN, HIDDEN])
        weight(p + "mlp.down_proj.weight", [HIDDEN, FFN])
        if i in ATTN_LAYERS:
            weight(p + "self_attn.qkv_proj.weight", [QKV_DIM, HIDDEN])
            weight(p + "self_attn.q_norm.weight_p1", [HEAD_DIM], "f32")
            weight(p + "self_attn.k_norm.weight_p1", [HEAD_DIM], "f32")
            weight(p + "self_attn.o_proj.weight", [HIDDEN, Q_DIM])
        else:
            weight(p + "linear_attn.in_proj_qkvz.weight", [QKVZ_DIM, HIDDEN])
            weight(p + "linear_attn.in_proj_ba.weight", [BA_DIM, HIDDEN])
            weight(p + "linear_attn.conv1d.weight", [CONV_DIM, 4])
            weight(p + "linear_attn.A_log", [GDN_V_HEADS], "f32")
            weight(p + "linear_attn.dt_bias", [GDN_V_HEADS])
            weight(p + "linear_attn.norm.weight", [GDN_D])
            weight(p + "linear_attn.out_proj.weight", [HIDDEN, GDN_V])

    def blk(tag, src=pre):
        return src[tag][0]["block"]

    def smem(tag, src=pre):
        return src[tag][0]["dynamic_shared_mem_bytes"]

    def tri(tag, params, grid, src=pre, **kw):
        """Single-step Triton kernel pinned to its dump module."""
        cubin, sha = pins[(tag, src is pre)]
        return single(TRITON[tag], params, blk(tag, src), grid,
                      shared_mem=smem(tag, src) or None, cubin=cubin, sha256=sha, **kw)

    GEMMA_PARAMS = ["out buffer<bf16>", "in buffer<bf16>", "in buffer<f32>",
                    "i32", "i32", "i32", "i32", "i32", "i32", "i32", "f32"]
    GEMMA_FUSED_PARAMS = ["out buffer<bf16>", "in buffer<bf16>", "inout buffer<bf16>",
                          "in buffer<f32>", "i32", "i32", "i32", "i32", "f32"]
    GEMMA_SMEM = (2 * HIDDEN + 512) * 4
    GEMMA_HEAD_SMEM = (2 * HEAD_DIM + 512) * 4
    LN_PARAMS = ["inout buffer<bf16>", "out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                 "out buffer<f32>", "i32", "i32", "i32", "i32", "f32"] + I2
    LN_IFACE = LN_PARAMS[:4] + LN_PARAMS[5:]
    UNIFIED_PARAMS = [
        "out buffer<bf16>", "in buffer<bf16>", "inout ptr", "inout ptr",
        "in buffer<i32>", "in buffer<i32>", "f32",
        "in buffer<f32>", "in buffer<f32>", "f32", "f32",
        "i64", "i64", "i64", "i64", "i64", "i64",
        "in buffer<i32>", "i64", "i64", "i64", "i64", "i64", "i64",
        "in buffer<i32>", "i32", "out buffer<f32>", "out buffer<f32>", "out buffer<f32>"] + I2
    ATTN_IFACE = UNIFIED_PARAMS[:26] + UNIFIED_PARAMS[29:]
    REDUCE_PARAMS = ["out buffer<bf16>", "in buffer<f32>", "in buffer<f32>", "in buffer<f32>",
                     "in buffer<i32>", "f32", "i64", "i64", "i64", "in buffer<i32>"] + I2

    def layer_norm_kernel(src, grid, rows_shape):
        """Gated RMSNorm (FLA layer_norm_fwd) with its Rstd side output as
        impl scratch."""
        cubin, sha = pins[("layer_norm", src is pre)]
        return {
            "params": LN_IFACE,
            "impl": {
                "scratch": {"rstd": {"dtype": "f32", "shape": rows_shape}},
                "steps": [step(TRITON["layer_norm"], LN_PARAMS, blk("layer_norm", src), grid,
                               [a(0), a(1), a(2), a(3), scr("rstd")] + [a(i) for i in range(4, 11)],
                               cubin=cubin, sha256=sha)],
            },
        }

    kernels = {
        "embedding": single("kern_embedding_i64_bf16",
                            ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
                            [256, 1, 1], [T, 1, 1], cubin="embedding.cubin"),
        "gemm": single("extern:cublaslt_bf16_tn",
                       ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
                       [1, 1, 1], [1, 1, 1]),
        # Gemma norms: one block per row; rows/grid differ per use (ATen's
        # reduction width depends on the row count, so `rows` is an arg).
        "gemma_norm": single("kern_gemma_rms_norm_bf16", GEMMA_PARAMS, [512, 1, 1], [T, 1, 1],
                             shared_mem=GEMMA_SMEM, cubin="gemma_rms_norm.cubin"),
        "gemma_norm_qhead": single("kern_gemma_rms_norm_bf16", GEMMA_PARAMS, [512, 1, 1],
                                   [mul(T, HEADS), 1, 1], shared_mem=GEMMA_HEAD_SMEM,
                                   cubin="gemma_rms_norm.cubin"),
        "gemma_norm_khead": single("kern_gemma_rms_norm_bf16", GEMMA_PARAMS, [512, 1, 1],
                                   [mul(T, KV_HEADS), 1, 1], shared_mem=GEMMA_HEAD_SMEM,
                                   cubin="gemma_rms_norm.cubin"),
        "gemma_fused_norm": single("kern_gemma_fused_add_rms_norm_bf16", GEMMA_FUSED_PARAMS,
                                   [512, 1, 1], [T, 1, 1], shared_mem=GEMMA_SMEM,
                                   cubin="gemma_rms_norm.cubin"),
        "silu_mul": single(silu_sym, ["out buffer<bf16>", "in buffer<bf16>", "i32", "i32", "f32", "i32"],
                           blk("silu"), [T, 1, 1]),
        "copy_rows": single("kern_copy_rows_bf16",
                            ["out buffer<bf16>", "in buffer<bf16>", "i32", "i32", "i32"],
                            [256, 1, 1], [T, 1, 1], cubin="copy_rows.cubin"),
        "last_row": single("kern_last_row_bf16",
                           ["out buffer<bf16>", "in buffer<bf16>", "i32", "i32", "i32"],
                           [256, 1, 1], [1, 1, 1], cubin="copy_rows.cubin"),
        "sigmoid_mul": single("kern_sigmoid_mul_bf16",
                              ["out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                               "i32", "i32", "i32", "i32"],
                              [256, 1, 1], [T, 1, 1], cubin="sigmoid_mul.cubin"),
        "argmax_row": {
            "params": ["in buffer<bf16>", "out buffer<i64>", "i32"],
            "impl": {
                "scratch": {"pmax": {"dtype": "f32", "shape": [1, 64]},
                            "pidx": {"dtype": "i32", "shape": [1, 64]}},
                "steps": [
                    step("kern_argmax_partial_bf16",
                         ["in buffer<bf16>", "out buffer<f32>", "out buffer<i32>", "i32"],
                         [1024, 1, 1], [1, 64, 1], [a(0), scr("pmax"), scr("pidx"), a(2)],
                         cubin="argmax.cubin"),
                    step("kern_argmax_final_i64",
                         ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32"],
                         [64, 1, 1], [1, 1, 1], [scr("pmax"), scr("pidx"), a(1), i32(64)],
                         cubin="argmax.cubin"),
                ],
            },
        },
        # --- GDN prefill chain (vLLM triton backend, FLA chunk kernels)
        "conv_fwd": tri("conv_fwd",
                        ["in buffer<bf16>", "in buffer<bf16>", "inout ptr", "in buffer<i32>",
                         "in buffer<u8>", "in buffer<i32>", "in buffer<i32>", "in buffer<i32>",
                         "out buffer<bf16>", "i32", "i64", "i64"] + I2,
                        [cdiv(T, CONV_BLOCK_M), CONV_DIM // 256, 1]),
        "post_conv": tri("post_conv",
                         ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>",
                          "in buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>",
                          "out buffer<f32>", "out buffer<f32>",
                          "i32", "i32", "i32", "i32", "i32", "i32", "i32"] + I2,
                         [cdiv(T, POST_CONV_BLOCK), GDN_V_HEADS + GDN_K_HEADS, 1]),
        "cumsum": tri("cumsum", ["in buffer<f32>", "out buffer<f32>", "in buffer<i32>",
                                 "in buffer<i32>", "i32"] + I2,
                      [cdiv(T, FLA_CHUNK), GDN_V_HEADS, 1]),
        "kkt": tri("kkt", ["in buffer<bf16>", "in buffer<f32>", "in buffer<f32>", "out buffer<f32>",
                           "in buffer<i32>", "in buffer<i32>", "i32"] + I2,
                   [cdiv(T, FLA_CHUNK), GDN_V_HEADS, 1]),
        # writes only the lower/diagonal 16x16 tiles of each 64x64 block; the
        # upper tiles are the zeros the buffer was allocated with
        "solve_tril": tri("solve_tril", ["in buffer<f32>", "out buffer<bf16>", "in buffer<i32>",
                                         "in buffer<i32>", "i32"] + I2,
                          [cdiv(T, FLA_CHUNK), GDN_V_HEADS, 1]),
        "recompute": tri("recompute",
                         ["in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>", "out buffer<bf16>",
                          "out buffer<bf16>", "in buffer<bf16>", "in buffer<f32>", "in buffer<i32>",
                          "in buffer<i32>", "i32"] + I2,
                         [cdiv(T, FLA_CHUNK), GDN_V_HEADS, 1]),
        # h0/ht are the SSM state line itself: each program loads its h0
        # tile first and stores ht last, so in-place is race-free
        "chunk_h": tri("chunk_h",
                       ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>",
                        "in buffer<f32>", "out buffer<bf16>", "inout ptr", "inout ptr",
                        "in buffer<i32>", "in buffer<i64>", "i32"] + I2,
                       [GDN_D // 32, GDN_V_HEADS, 1]),
        "chunk_o": tri("chunk_o",
                       ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                        "in buffer<f32>", "out buffer<bf16>", "in buffer<i32>", "in buffer<i32>",
                        "f32", "i32"] + I2,
                       [GDN_D // 64, cdiv(T, FLA_CHUNK), GDN_V_HEADS]),
        "gated_norm": layer_norm_kernel(pre, [mul(T, GDN_V_HEADS // LN_ROWS_PER_BLOCK), 1, 1],
                                        ["tokens", GDN_V_HEADS]),
        # --- GDN decode
        "conv_update": tri("conv_update",
                           ["in buffer<bf16>", "in buffer<bf16>", "inout ptr", "in buffer<i32>",
                            "out buffer<bf16>", "i32", "i32", "i64", "i64"] + I2,
                           [1, CONV_DIM // 256, 1], src=dec),
        "recurrent": tri("recurrent",
                         ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>",
                          "in buffer<bf16>", "out buffer<bf16>", "inout ptr", "inout ptr",
                          "in buffer<i32>", "f32"] + I2,
                         [GDN_D // 32, GDN_V_HEADS, 1], src=dec),
        "gated_norm_decode": layer_norm_kernel(dec, [GDN_V_HEADS, 1, 1], [1, GDN_V_HEADS]),
        # --- attention
        # one generic instance for both programs (decode's launch in vLLM is
        # the num_tokens==1 specialization: same arithmetic, one arg fewer)
        "mrope": tri("mrope", ["inout buffer<bf16>", "inout buffer<bf16>", "in buffer<bf16>",
                               "in buffer<bf16>", "i32"] + I2, [T, 1, 1]),
        "reshape_and_cache": tri("cache",
                                 ["in buffer<bf16>", "in buffer<bf16>", "inout ptr", "inout ptr",
                                  "in buffer<i64>", "in buffer<f32>", "in buffer<f32>"] + ["i64"] * 9,
                                 [T, 1, 1]),
        "attn_prefill": tri("unified", ATTN_IFACE, [cdiv(T, BLOCK_Q), KV_HEADS, 1]),
        "attn": {
            "params": ATTN_IFACE,
            "impl": {
                "scratch": {
                    "segm_out": {"dtype": "f32", "shape": [1, HEADS, NUM_SEGMENTS, HEAD_DIM]},
                    "segm_max": {"dtype": "f32", "shape": [1, HEADS, NUM_SEGMENTS]},
                    "segm_expsum": {"dtype": "f32", "shape": [1, HEADS, NUM_SEGMENTS]},
                },
                "steps": [
                    step(TRITON["unified"], UNIFIED_PARAMS, blk("unified", dec),
                         [1, KV_HEADS, NUM_SEGMENTS],
                         [a(i) for i in range(26)]
                         + [scr("segm_out"), scr("segm_max"), scr("segm_expsum"), a(26), a(27)],
                         shared_mem=smem("unified", dec), cubin=pins[("unified", False)][0],
                         sha256=pins[("unified", False)][1]),
                    step(TRITON["reduce"], REDUCE_PARAMS, blk("reduce", dec), [1, HEADS, 1],
                         [a(0), scr("segm_out"), scr("segm_max"), scr("segm_expsum"), a(5),
                          f32(1.0), i64(Q_DIM), i64(HEAD_DIM), i64(BLOCK_TABLE_LEN), a(24),
                          i64(0), i64(0)],
                         shared_mem=smem("reduce", dec), cubin=pins[("reduce", False)][0],
                         sha256=pins[("reduce", False)][1]),
                ],
            },
        },
    }

    Z2 = [i64(0), i64(0)]

    def gemm(label, ab, w, c, m, n, k):
        return d(label, "gemm", [buf(ab), buf(w), buf(c), m, i32(n), i32(k)])

    def fused(label, x_in, w):
        return d(label, "gemma_fused_norm",
                 [buf("x"), buf(x_in), buf("residual"), buf(w), i32(HIDDEN), T,
                  i32(HIDDEN), i32(HIDDEN), f32(eps)])

    def gdn_layer(i, decode):
        p = f"model.layers.{i}.linear_attn."
        l = f"l{i}."
        line = GDN_LAYERS.index(i) + 1
        idx = buf("gdn.line_index", 4 * line)
        ds = [
            gemm(l + "in_proj_qkvz", "x", p + "in_proj_qkvz.weight", "qkvz", T, QKVZ_DIM, HIDDEN),
            gemm(l + "in_proj_ba", "x", p + "in_proj_ba.weight", "ba", T, BA_DIM, HIDDEN),
        ]
        a_ = buf("ba", GDN_V_HEADS * BF16)   # ba row = [b | a]
        b_ = buf("ba")
        if not decode:
            ds += [
                d(l + "conv", "conv_fwd",
                  [buf("qkvz"), buf(p + "conv1d.weight"), state("gdn"), idx, buf("gdn.has_initial"),
                   buf("cu_seqlens_q"), buf("conv.batch_ptr"), buf("conv.token_chunk_offset"),
                   buf("conv_out"), i32(GDN_LINES), i64(QKVZ_DIM), i64(CONV_DIM)] + Z2),
                # a/b are strided views into ba (vLLM copies them contiguous;
                # the kernel takes row strides, so no copy here)
                d(l + "post_conv", "post_conv",
                  [buf("conv_out"), a_, b_, buf(p + "A_log"), buf(p + "dt_bias"),
                   buf("gdn_q"), buf("gdn_k"), buf("gdn_v"), buf("g"), buf("beta"),
                   i32(CONV_DIM), i32(BA_DIM), i32(BA_DIM), i32(GDN_Q), i32(GDN_Q), i32(GDN_V), T] + Z2),
                d(l + "cumsum", "cumsum",
                  [buf("g"), buf("g_cum"), buf("cu_seqlens_q"), buf("fla.chunk_indices"), T] + Z2),
                d(l + "kkt", "kkt",
                  [buf("gdn_k"), buf("beta"), buf("g_cum"), buf("A"), buf("cu_seqlens_q"),
                   buf("fla.chunk_indices"), T] + Z2),
                d(l + "solve_tril", "solve_tril",
                  [buf("A"), buf("Ai"), buf("cu_seqlens_q"), buf("fla.chunk_indices"), T] + Z2),
                d(l + "recompute_wu", "recompute",
                  [buf("gdn_k"), buf("gdn_v"), buf("beta"), buf("w"), buf("u"), buf("Ai"),
                   buf("g_cum"), buf("cu_seqlens_q"), buf("fla.chunk_indices"), T] + Z2),
                d(l + "chunk_h", "chunk_h",
                  [buf("gdn_k"), buf("u"), buf("w"), buf("v_new"), buf("g_cum"), buf("h"),
                   state("gdn", line * GDN_LINE_BYTES + CONV_STATE_BYTES),
                   state("gdn", line * GDN_LINE_BYTES + CONV_STATE_BYTES),
                   buf("cu_seqlens_q"), buf("fla.chunk_offsets"), T] + Z2),
                d(l + "chunk_o", "chunk_o",
                  [buf("gdn_q"), buf("gdn_k"), buf("v_new"), buf("h"), buf("g_cum"),
                   buf("core_attn_out"), buf("cu_seqlens_q"), buf("fla.chunk_indices"),
                   f32(gdn_scale), T] + Z2),
                d(l + "z_copy", "copy_rows",
                  [buf("z_c"), buf("qkvz", Z_OFF * BF16), i32(GDN_V), i32(QKVZ_DIM), i32(GDN_V)]),
                d(l + "gated_norm", "gated_norm",
                  [buf("core_attn_out"), buf("core_attn_out"), buf(p + "norm.weight"), buf("z_c"),
                   i32(GDN_D), i32(GDN_D), i32(GDN_D), expr(mul(T, GDN_V_HEADS)), f32(eps)] + Z2),
            ]
        else:
            ds += [
                d(l + "conv_update", "conv_update",
                  [buf("qkvz"), buf(p + "conv1d.weight"), state("gdn"), idx, buf("qkvz"),
                   i32(1), i32(GDN_LINES), i64(1), i64(1)] + Z2),
                d(l + "recurrent", "recurrent",
                  [buf("qkvz"), a_, b_, buf(p + "A_log"), buf(p + "dt_bias"), buf("core_attn_out"),
                   state("gdn", CONV_STATE_BYTES), state("gdn", CONV_STATE_BYTES), idx,
                   f32(gdn_scale)] + Z2),
                d(l + "gated_norm", "gated_norm_decode",
                  [buf("core_attn_out"), buf("core_attn_out"), buf(p + "norm.weight"),
                   buf("qkvz", Z_OFF * BF16), i32(GDN_D), i32(GDN_D), i32(GDN_D), i32(GDN_V_HEADS),
                   f32(eps)] + Z2),
            ]
        ds.append(gemm(l + "out_proj", "core_attn_out", p + "out_proj.weight", "y", T, HIDDEN, GDN_V))
        return ds

    def attn_layer(i, decode):
        p = f"model.layers.{i}.self_attn."
        l = f"l{i}."
        koff = ATTN_LAYERS.index(i) * LAYER_KV_BYTES
        ks, vs = buf("kv_scales"), buf("kv_scales", 4)
        kv_k, kv_v = state("kv", koff), state("kv", koff + V_BYTE_OFF)
        return [
            gemm(l + "qkv_proj", "x", p + "qkv_proj.weight", "qkv", T, QKV_DIM, HIDDEN),
            d(l + "q_norm", "gemma_norm_qhead",
              [buf("q_n"), buf("qkv"), buf(p + "q_norm.weight_p1"), i32(HEAD_DIM),
               expr(mul(T, HEADS)), i32(HEADS), i32(QKV_DIM), i32(2 * HEAD_DIM), i32(Q_DIM),
               i32(HEAD_DIM), f32(eps)]),
            d(l + "k_norm", "gemma_norm_khead",
              [buf("k_n"), buf("qkv", HEADS * 2 * HEAD_DIM * BF16), buf(p + "k_norm.weight_p1"),
               i32(HEAD_DIM), expr(mul(T, KV_HEADS)), i32(KV_HEADS), i32(QKV_DIM), i32(HEAD_DIM),
               i32(KV_DIM), i32(HEAD_DIM), f32(eps)]),
            d(l + "rope", "mrope", [buf("q_n"), buf("k_n"), buf("cos_g"), buf("sin_g"), i32(0)] + Z2),
            d(l + "kv_write", "reshape_and_cache",
              [buf("k_n"), buf("qkv", (HEADS * 2 * HEAD_DIM + KV_DIM) * BF16), kv_k, kv_v,
               buf("slot_mapping"), ks, vs,
               i64(KV_DIM), i64(QKV_DIM), i64(BLOCK_STRIDE), i64(2 * HEAD_DIM), i64(0), i64(0),
               i64(KV_HEADS * 2 * HEAD_DIM), i64(0), i64(0)]),
            d(l + "attn", "attn" if decode else "attn_prefill",
              [buf("attn_out"), buf("q_n"), kv_k, kv_v, buf("block_table"), buf("seq_lens"),
               f32(attn_scale), ks, vs, f32(1.0), f32(0.0),
               i64(BLOCK_TABLE_LEN), i64(Q_DIM), i64(HEAD_DIM), i64(Q_DIM), i64(HEAD_DIM), i64(0),
               buf("seq_lens"),
               i64(BLOCK_STRIDE), i64(KV_HEADS * 2 * HEAD_DIM), i64(2 * HEAD_DIM),
               i64(BLOCK_STRIDE), i64(KV_HEADS * 2 * HEAD_DIM), i64(2 * HEAD_DIM),
               buf("cu_seqlens_q"), i32(1)] + Z2),
            d(l + "gate", "sigmoid_mul",
              [buf("gated"), buf("attn_out"), buf("qkv", GATE_OFF * BF16), i32(HEADS), i32(HEAD_DIM),
               i32(QKV_DIM), i32(2 * HEAD_DIM)]),
            gemm(l + "o_proj", "gated", p + "o_proj.weight", "y", T, HIDDEN, Q_DIM),
        ]

    def forward(decode):
        ds = [
            d("embed", "embedding",
              [buf("token_ids"), buf("model.embed_tokens.weight"), buf("residual"), T, i32(HIDDEN)]),
            # rope tables gathered by position; mrope gets num_tokens=0 so its
            # three (t/h/w) planes alias this one table — text-only positions
            d("rope_cos", "embedding", [buf("positions"), buf("rope.cos"), buf("cos_g"), T, i32(ROT_HALF)]),
            d("rope_sin", "embedding", [buf("positions"), buf("rope.sin"), buf("sin_g"), T, i32(ROT_HALF)]),
            d("l0.input_norm", "gemma_norm",
              [buf("x"), buf("residual"), buf("model.layers.0.input_layernorm.weight_p1"),
               i32(HIDDEN), T, i32(1), i32(HIDDEN), i32(0), i32(HIDDEN), i32(0), f32(eps)]),
        ]
        for i in range(LAYERS):
            p = f"model.layers.{i}."
            l = f"l{i}."
            ds += attn_layer(i, decode) if i in ATTN_LAYERS else gdn_layer(i, decode)
            ds += [
                fused(l + "post_attn_norm", "y", p + "post_attention_layernorm.weight_p1"),
                gemm(l + "gate_up", "x", p + "mlp.gate_up_proj.weight", "gate_up", T, 2 * FFN, HIDDEN),
                d(l + "silu_mul", "silu_mul",
                  [buf("act"), buf("gate_up"), i32(FFN), i32(0), f32(1.0), i32(0)]),
                gemm(l + "down_proj", "act", p + "mlp.down_proj.weight", "y", T, HIDDEN, FFN),
            ]
            last = i + 1 == LAYERS
            ds.append(fused(l + ("final_norm" if last else "next_input_norm"), "y",
                            "model.norm.weight_p1" if last else f"model.layers.{i + 1}.input_layernorm.weight_p1"))
        # The final norm runs over all rows like vLLM (ATen's reduction width
        # depends on the row count); lm_head only needs the last one.
        if decode:
            ds.append(gemm("lm_head", "x", "lm_head.weight", "logits", i32(1), VOCAB, HIDDEN))
        else:
            ds.append(d("last_row", "last_row", [buf("final_x"), buf("x"), i32(HIDDEN), i32(HIDDEN), T]))
            ds.append(gemm("lm_head", "final_x", "lm_head.weight", "logits", i32(1), VOCAB, HIDDEN))
        ds.append(d("sample", "argmax_row", [buf("logits"), buf("next_token"), i32(VOCAB)]))
        return ds

    programs = {
        "prefill": {"dispatches": forward(False)},
        "decode": {"dispatches": forward(True)},
    }
    for name, dom in DOMAINS.items():
        if name in buffers:
            buffers[name]["domain"] = dom
    return {
        "meta": {"version": 2, "model": "qwen3.8-27b"},
        # bs=1; prefill per chunk (tokens <= CHUNK_MAX) over *all* prompt
        # tokens (it emits next_token), decode at tokens=1
        "symbols": {"tokens": {"max": CHUNK_MAX}},
        "states": {"kv": {"bytes_per_token": KV_BYTES_PER_TOKEN},
                   "gdn": {"bytes_fixed": GDN_STATE_BYTES}},
        "buffers": buffers,
        "kernels": kernels,
        "programs": programs,
    }


def main():
    repo = pathlib.Path(__file__).resolve().parent.parent
    dump = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else repo / "dumped-kernels" / "pid1898802")
    out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else repo / "examples" / "qwen3.8-27b.json")

    recs = load(dump / "launches.jsonl")
    fwds = forwards(recs)
    ts = [tokens_of(f) for f in fwds]
    # real prefill forwards (multi-token, after the profiling passes) + the
    # decode forward that follows the longest one
    prefills = [(t, i) for i, t in enumerate(ts) if t and 1 < t <= CHUNK_MAX and i > ts.index(1)]
    assert len(prefills) >= 3, f"need several prefill forwards to fit grids, got {prefills}"
    t_ref, i_ref = max(prefills)
    pre = group(fwds[i_ref])
    dec = group(fwds[i_ref + 1])
    assert tokens_of(fwds[i_ref + 1]) == 1
    for t, i in prefills:
        check_prefill(group(fwds[i]), t)
    check_decode(dec)
    print(f"capture: prefill forwards T={sorted(t for t, _ in prefills)} verified, "
          f"decode after T={t_ref} verified", file=sys.stderr)

    eps = pf(pre["layer_norm"][0], 9)
    attn_scale = pf(pre["unified"][0], 6)
    gdn_scale = pf(pre["chunk_o"][0], 8)
    assert pf(dec["recurrent"][0], 9) == gdn_scale and pf(dec["unified"][0], 6) == attn_scale
    assert pf(dec["layer_norm"][0], 9) == eps
    assert abs(eps - 1e-6) < 1e-12 and attn_scale == 0.0625
    silu_sym = pre["silu"][0]["symbol"]

    pinner = Pinner(dump)
    pins = {}
    for tag in TRITON:
        for src, is_pre in ((pre, True), (dec, False)):
            if src[tag]:
                pins[(tag, is_pre)] = pinner.pin(TRITON[tag], src[tag][0]["attributes"]["num_regs"])
    for (tag, is_pre), (mod, sha) in sorted(pins.items()):
        print(f"  pin {TRITON[tag]:<52} {'prefill' if is_pre else 'decode '} -> {mod} {sha[:12]}",
              file=sys.stderr)

    m = build(pre, dec, pins, eps, attn_scale, gdn_scale, silu_sym)
    out.write_text(json.dumps(m, indent=1) + "\n")
    n_disp = {k: len(v["dispatches"]) for k, v in m["programs"].items()}
    print(f"wrote {out}: {len(m['buffers'])} buffers, {len(m['kernels'])} kernels, "
          f"dispatches {n_disp}", file=sys.stderr)


if __name__ == "__main__":
    main()
