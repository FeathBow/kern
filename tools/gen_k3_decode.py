#!/usr/bin/env python3
"""Generate kern's Kimi-K3 (pruned, 224 experts) decode superstep at EP<R>.

    python3 tools/gen_k3_decode.py --layers 4 --ranks 1 > examples/k3-4l-ep1.json
    python3 tools/gen_k3_decode.py --ranks 4 > examples/k3-ep4.json

One SPMD manifest per world: every rank runs the whole dense trunk on its own
sequence(s) and serves its expert shard to the world through MegaMoE
(tools/gen_k3_moe.py). Program `decode`: one token per sequence through all
layers — attention-residual mix, KDA (conv + delta rule, state in a
`bytes_per_seq` line) or absorbed paged MLA (latent cache in `kv`), latent
MoE (router → down-proj → MegaMoE → norm → up-proj, plus the shared
experts) or the dense MLP — then the output mix, final norm, lm_head and
argmax into `next_token`.

The launch sequence is pegainfer's certified `k3_step` line for line
(pegainfer-k3/src/executor/forward/{step,decode}.rs): every dense projection
is a cublasGemmEx f32 partial (`extern:cublas_bf16_tn_f32`) landed to bf16 by
the certified `k3_land` kernel, and everything else is the certified TileLang
kernel (tools/k3-tilelang) or pegainfer's hand-written MLA kernel, called
with the same operands. Batch is one row per rank (`tokens` = 1): the
TileLang kernels are bucket-instantiated and the state kernels resolve one
state line per launch (tools/k3_line_shim.py).
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gen_k3_moe
import handwritten
import kern_manifest

REPO = pathlib.Path(__file__).resolve().parent.parent
CUBINS = REPO / "target" / "cubins"
TL = REPO / "tools" / "k3-tilelang"

H = 7168
V = 163840
HEADS, HEAD_DIM = 96, 128
INNER = HEADS * HEAD_DIM           # 12288
Q_LORA, KV_LORA, ROPE = 1536, 512, 64
KV_A = KV_LORA + ROPE              # 576
Q_B = HEADS * 192                  # 18432
MLA_FUSED = Q_LORA + KV_A + INNER  # 14400
KDA_FUSED = 4 * INNER              # 49152
WSM = 256                          # b_proj 96 | f_a 128 | pad
EXPERTS, TOPK, LATENT, INTER = 224, 16, 3584, 3072
SHARED = 2 * INTER                 # 6144
DENSE_I = 33792
ATTN_RES_BLOCK = 12
LAYERS = 93
MLA_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92}

# KDA state line: recurrent f32 [96,128,128] then the three conv windows
# bf16 [3][12288] (q, k, v).
KDA_REC_BYTES = HEADS * HEAD_DIM * HEAD_DIM * 4
KDA_WIN_BYTES = 3 * INNER * 2
KDA_LINE_BYTES = KDA_REC_BYTES + 3 * KDA_WIN_BYTES
# MLA latent cache: 64-token pages of [mla_layers][64][576] bf16.
PAGE = 64
LATENT_ROW = KV_A


def is_mla(i):
    return i in MLA_LAYERS


# ── TileLang kernel index ────────────────────────────────────────────────

PROTO = re.compile(r'^extern "C" __global__ void (\w+)\((.*)\);$')
DEFN = re.compile(r'^extern "C" __global__ void __launch_bounds__\(\d+, \d+\) (\w+)\((.*)\) \{$')
LAUNCH = re.compile(r'(\w+)<<<dim3\((\d+), (\d+), (\d+)\), (\d+), (\d+), stream>>>')
OUT_NAMES = {"O", "Out", "Sc", "Idx", "Wts", "X", "Y", "StateN", "Sn"}


def ptype(ctype, name):
    if "bfloat16_t" in ctype:
        dt = "bf16"
    elif "float" in ctype:
        dt = "f32"
    elif "int" in ctype and "long" not in ctype:
        dt = "i32"
    elif "long long" in ctype:
        return "i64"
    else:
        raise ValueError(ctype)
    if "*" not in ctype:
        return dt
    if name == "Base":
        return "inout state"
    d = "out" if name in OUT_NAMES else "in"
    return f"{d} buffer<{dt}>"


class Family:
    """One TileLang family cubin: kernels by name with params and geometry."""

    def __init__(self, family, line=False):
        src = CUBINS / f"k3_tl_{family}_line.cu" if line else TL / f"k3_{family}_batched.cu"
        cubin = CUBINS / (f"k3_tl_{family}_line.cubin" if line else f"k3_tl_{family}.cubin")
        if not cubin.exists():
            sys.exit(f"{cubin} missing: run tools/build_k3_kernels.sh (inside kernel-lab)")
        self.mod = {"cubin": cubin.name, "sha256": hashlib.sha256(cubin.read_bytes()).hexdigest()}
        self.params = {}
        for l in src.read_text().splitlines():
            m = PROTO.match(l) or DEFN.match(l)
            if m:
                plist = [p.strip() for p in m.group(2).split(",")]
                self.params[m.group(1)] = [ptype(p, p.split()[-1]) for p in plist]
        self.geom = {}
        base = TL / f"k3_{family}_batched.cu"
        for m in LAUNCH.finditer(base.read_text()):
            self.geom[m.group(1)] = ([int(m.group(2)), int(m.group(3)), int(m.group(4))],
                                     [int(m.group(5)), 1, 1], int(m.group(6)))

    def op(self, name):
        base = name[len("kern_"):].replace("_line", "_kernel") if name.startswith("kern_") else name
        grid, block, smem = self.geom[base]
        launch = {**self.mod, "entry": name, "block": block, "grid": grid}
        if smem:
            launch["shared_mem"] = smem
        return {"params": self.params[name], "impl": {"launches": [launch]}}


# ── manifest ─────────────────────────────────────────────────────────────

def build(layers, ranks, max_ctx, seqs_max=1):
    assert 1 <= layers <= LAYERS
    n_kda = sum(1 for i in range(layers) if not is_mla(i))
    mla_index = {i: k for k, i in enumerate(i for i in range(layers) if is_mla(i))}
    n_mla = len(mla_index)
    assert n_mla > 0, "the decode program needs at least one MLA layer (layer 3)"
    max_pages = -(-max_ctx // PAGE)
    page_stride = n_mla * PAGE * LATENT_ROW  # elements
    tokens_max = seqs_max
    blocks_total = -(-layers // ATTN_RES_BLOCK)
    NB_MAX = 8

    fam = {f: Family(f) for f in ["rms_norm_rbs", "land", "land_rms_norm_rbs", "add2", "mul_sigmoid",
                                  "situ", "router_topk", "attnres_scores", "attnres_mix"]}
    fam["kda_core"] = Family("kda_core", line=True)
    fam["conv_silu"] = Family("conv_silu", line=True)
    mp = gen_k3_moe.mega_pieces(ranks, tokens_max)

    ops = {}

    def tl(family, name):
        if name not in ops:
            ops[name] = fam[family].op(name)
        return name

    hw_embed = handwritten.hw("embedding")
    hw_argmax = handwritten.hw("argmax")
    hw_copy = handwritten.hw("copy_rows")
    hw_mla = handwritten.hw("k3_mla_paged_attn")
    hw_kv = handwritten.hw("k3_kv_append")
    T = "tokens"
    ops["embedding"] = {
        "params": ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
        "impl": {"launches": [{**hw_embed, "entry": "kern_embedding_i64_bf16", "block": [256, 1, 1], "grid": [T, 1, 1]}]},
    }
    ops["copy_rows"] = {
        "params": ["out buffer<bf16>", "in buffer<bf16>", "i32", "i32", "i32"],
        "impl": {"launches": [{**hw_copy, "entry": "kern_copy_rows_bf16", "block": [256, 1, 1], "grid": [T, 1, 1]}]},
    }
    ops["gemm_f32"] = {
        "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32", "i32"],
        "impl": {"launches": [{"entry": "extern:cublas_bf16_tn_f32"}]},
    }
    ops["kv_append"] = {
        "params": ["in buffer<bf16>", "in buffer<bf16>", "in buffer<i64>", "inout state", "i64", "i64", "i32"],
        "impl": {"launches": [{**hw_kv, "entry": "kern_k3_kv_append", "block": [288, 1, 1], "grid": [T, 1, 1]}]},
    }
    ops["mla_paged_attn"] = {
        "params": ["in buffer<bf16>", "in buffer<bf16>", "inout state", "in buffer<i32>", "i32", "i64",
                   "in buffer<i32>", "in buffer<bf16>", "out buffer<bf16>"],
        "impl": {"launches": [{**hw_mla, "entry": "kern_k3_mla_paged_attn", "block": [128, 1, 1], "grid": [T, HEADS, 1]}]},
    }
    ops["argmax"] = {
        "params": ["in buffer<bf16>", "out buffer<i64>", "i32"],
        "impl": {
            "scratch": {"pmax": {"dtype": "f32", "shape": [T, 64]}, "pidx": {"dtype": "i32", "shape": [T, 64]}},
            "launches": [
                {**hw_argmax, "entry": "kern_argmax_partial_bf16",
                 "params": ["in buffer<bf16>", "out buffer<f32>", "out buffer<i32>", "i32"],
                 "block": [1024, 1, 1], "grid": [T, 64, 1],
                 "args": [{"param": 0}, {"scratch": "pmax"}, {"scratch": "pidx"}, {"param": 2}]},
                {**hw_argmax, "entry": "kern_argmax_final_i64",
                 "params": ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32"],
                 "block": [64, 1, 1], "grid": [T, 1, 1],
                 "args": [{"scratch": "pmax"}, {"scratch": "pidx"}, {"param": 1}, {"i32": 64}]},
            ],
        },
    }
    ops.update(mp["ops"])

    # ---- buffers
    buffers = {
        "token_ids": {"dtype": "i64", "shape": [T], "kind": "input", "domain": {"index_into": "embed"}},
        "slot_mapping": {"dtype": "i64", "shape": [T], "kind": "input", "domain": {"index_into": "kv"}},
        "block_table": {"dtype": "i32", "shape": ["seqs", max_pages], "kind": "input",
                        "domain": {"index_into": "kv", "stride": PAGE}},
        "seq_lens": {"dtype": "i32", "shape": ["seqs"], "kind": "input", "domain": {"min": 1}},
        "kda.line_index": {"dtype": "i32", "shape": [n_kda, "seqs"], "kind": "input",
                           "domain": {"index_into": "kda", "stride": KDA_LINE_BYTES}},
        "next_token": {"dtype": "i64", "shape": ["seqs"], "kind": "output", "domain": {"index_into": "embed"}},
        **mp["buffers"],
    }
    states = {
        "kv": {"bytes_per_token": n_mla * LATENT_ROW * 2},
        "kda": {"bytes_per_seq": n_kda * KDA_LINE_BYTES},
    }

    def weight(name, shape, dtype="bf16"):
        buffers[name] = {"dtype": dtype, "shape": list(shape), "kind": "weight"}

    def work(name, width, dtype="bf16"):
        buffers[name] = {"dtype": dtype, "shape": [T, width], "kind": "workspace"}

    weight("embed", [V, H])
    weight("gamma_final", [H])
    weight("sw_out", [H], "f32")
    weight("w_lm", [V, H])
    for n in ["hidden", "prefix", "mixed", "prefix2", "mixed2", "attn_out", "mlp_out", "normed",
              "routed", "shared"]:
        work(n, H)
    buffers["blocks"] = {"dtype": "bf16", "shape": [T, NB_MAX, H], "kind": "workspace"}
    work("scores", NB_MAX + 1, "f32")
    work("hidden_partial", H, "f32")
    work("kda_gate_partial", KDA_FUSED, "f32")
    work("kda_conv_partial", INNER, "f32")
    work("kda_wsm_partial", WSM, "f32")
    work("kda_forget_partial", INNER, "f32")
    work("beta", HEADS)
    work("forget_low", HEAD_DIM)
    for n in ["out_gate", "conv_x", "conv_q", "conv_k", "conv_v", "gated", "mla_gate", "attn"]:
        work(n, INNER)
    work("mla_fused_partial", MLA_FUSED, "f32")
    work("q_norm", Q_LORA)
    work("kv_a", KV_A)
    work("kv_latent", KV_LORA)
    work("kv_norm", KV_LORA)
    work("rope", ROPE)
    work("q_partial", Q_B, "f32")
    work("query", Q_B)
    work("router_partial", EXPERTS, "f32")
    work("topk_idx", TOPK, "i32")
    work("topk_weight", TOPK, "f32")
    work("latent_partial", LATENT, "f32")
    for n in ["latent", "routed_latent", "routed_latent_norm"]:
        work(n, LATENT)
    work("shared_partial", 2 * SHARED, "f32")
    for n in ["shared_gate", "shared_up", "shared_act"]:
        work(n, SHARED)
    work("dense_partial", 2 * DENSE_I, "f32")
    for n in ["dense_gate", "dense_up", "dense_act"]:
        work(n, DENSE_I)
    work("logit_partial", V, "f32")
    work("logits", V)

    # ---- program
    prog = []
    b = lambda name, off=0: {"buf": name, "offset": off} if off else {"buf": name}
    i32 = lambda v: {"i32": v}
    i64 = lambda v: {"i64": v}
    tok = {"var": T}

    def step(label, op, *args):
        prog.append({"label": label, "op": op, "args": list(args)})

    def copy(label, dst, src, width, dst_stride=None, src_stride=None):
        step(label, "copy_rows", dst, src, i32(dst_stride or width), i32(src_stride or width), i32(width))

    def gemm(label, a, w, c, n, k, ldc=None):
        """c[tokens, ldc] (cols 0..n from c's offset) = a[tokens, k] @ w[n, k]^T, f32."""
        step(label, "gemm_f32", a, w, c, tok, i32(n), i32(k), i32(ldc or n))

    def land(label, p, o, nt, n, off):
        step(label, tl("land", f"k3_land_b1_nt{nt}_n{n}_off{off}_sk1_kernel"), o, p)

    def rms(label, x, gamma, o, h):
        step(label, tl("rms_norm_rbs", f"k3_rms_norm_rbs_b1_h{h}_kernel"), gamma, o, x)

    def attn_res(label, nb, prefix, sw, out):
        step(label + ".scores", tl("attnres_scores", f"k3_attnres_scores_b1_nb{nb}_h{H}_kernel"),
             b("blocks"), prefix, b("scores"), sw)
        step(label + ".mix", tl("attnres_mix", f"k3_attnres_mix_b1_nb{nb}_h{H}_kernel"),
             b("blocks"), out, prefix, b("scores"))

    def add2(label, a, c, o):
        step(label, tl("add2", f"k3_add2_b1_n{H}_kernel"), a, c, o)

    def situ(label, gate, up, out, n):
        step(label, tl("situ", f"k3_situ_b1_n{n}_kernel"), gate, out, up)

    step("embed", "embedding", b("token_ids"), b("embed"), b("hidden"), tok, i32(H))

    blocks = 0
    kda_k = 0
    for i in range(layers):
        L = f"l{i}."
        w = lambda n, off=0, i=i: b(f"layers.{i}.{n}", off)
        snapshot = i % ATTN_RES_BLOCK == 0
        nb_in = blocks
        if snapshot:
            blocks += 1
        nb_mlp = blocks

        copy(L + "prefix", b("prefix"), b("hidden"), H)
        if nb_in > 0:
            attn_res(L + "res_in", nb_in, b("prefix"), w("sw_attn"), b("mixed"))
        else:
            copy(L + "mixed", b("mixed"), b("prefix"), H)
        if snapshot:
            copy(L + "snapshot", b("blocks", nb_in * H * 2), b("prefix"), H, dst_stride=NB_MAX * H)

        rms(L + "norm_in", b("mixed"), w("gamma_in"), b("normed"), H)
        if is_mla(i):
            k = mla_index[i]
            layer_off = k * PAGE * LATENT_ROW  # elements
            gemm(L + "wfu", b("normed"), w("wfu"), b("mla_fused_partial"), MLA_FUSED, H)
            step(L + "q_norm", tl("land_rms_norm_rbs", f"k3_land_rms_norm_rbs_b1_nt{MLA_FUSED}_n{Q_LORA}_off0_sk1_kernel"),
                 w("gamma_q_a"), b("q_norm"), b("mla_fused_partial"))
            land(L + "kv_a", b("mla_fused_partial"), b("kv_a"), MLA_FUSED, KV_A, Q_LORA)
            land(L + "mla_gate", b("mla_fused_partial"), b("mla_gate"), MLA_FUSED, INNER, Q_LORA + KV_A)
            copy(L + "kv_latent", b("kv_latent"), b("kv_a"), KV_LORA, src_stride=KV_A)
            copy(L + "rope", b("rope"), b("kv_a", KV_LORA * 2), ROPE, src_stride=KV_A)
            rms(L + "kv_norm", b("kv_latent"), w("gamma_kv_a"), b("kv_norm"), KV_LORA)
            step(L + "kv_append", "kv_append", b("kv_norm"), b("rope"), b("slot_mapping"), {"state": "kv"},
                 i64(layer_off), i64(page_stride), tok)
            gemm(L + "q_b", b("q_norm"), w("w_q_b"), b("q_partial"), Q_B, Q_LORA)
            land(L + "query", b("q_partial"), b("query"), Q_B, Q_B, 0)
            step(L + "attn", "mla_paged_attn", b("query"), w("w_kv_b"), {"state": "kv", "offset": layer_off * 2},
                 b("block_table"), i32(max_pages), i64(page_stride), b("seq_lens"), w("scale"), b("attn"))
            step(L + "gate", tl("mul_sigmoid", f"k3_mul_sigmoid_b1_n{INNER}_kernel"), b("attn"), b("mla_gate"), b("gated"))
        else:
            line = b("kda.line_index", kda_k * seqs_max * 4)
            kda_k += 1
            gemm(L + "gate_proj", b("normed"), w("wbig", 3 * INNER * H * 2), b("kda_gate_partial", 3 * INNER * 4),
                 INNER, H, ldc=KDA_FUSED)
            gemm(L + "wsm", b("normed"), w("wsm"), b("kda_wsm_partial"), WSM, H)
            land(L + "beta", b("kda_wsm_partial"), b("beta"), WSM, HEADS, 0)
            land(L + "forget_low", b("kda_wsm_partial"), b("forget_low"), WSM, HEAD_DIM, HEADS)
            land(L + "out_gate", b("kda_gate_partial"), b("out_gate"), KDA_FUSED, INNER, 3 * INNER)
            gemm(L + "f_b", b("forget_low"), w("w_f_b"), b("kda_forget_partial"), INNER, HEAD_DIM)
            for s, name in enumerate("qkv"):
                gemm(L + f"{name}_proj", b("normed"), w("wbig", s * INNER * H * 2), b("kda_conv_partial"), INNER, H)
                # window s of the line, in bf16 elements
                off = (KDA_REC_BYTES + s * KDA_WIN_BYTES) // 2
                step(L + f"conv_{name}", tl("conv_silu", f"kern_k3_conv_silu_b1_kp{INNER}_w4_sk1_line"),
                     {"state": "kda"}, line, i64(KDA_LINE_BYTES // 2), i64(off),
                     w(f"cw_{name}"), b("kda_conv_partial"), b("conv_x"), b(f"conv_{name}"))
            step(L + "kda_core", tl("kda_core", f"kern_k3_kda_core_b1_kh{HEADS}_kd{HEAD_DIM}_line"),
                 w("a_log"), b("beta"), w("dt_bias"), b("out_gate"), b("kda_forget_partial"), w("gamma_o"),
                 b("conv_k"), b("gated"), b("conv_q"), {"state": "kda"}, line, i64(KDA_LINE_BYTES // 4), i64(0),
                 b("conv_v"))
        gemm(L + "o_proj", b("gated"), w("w_o"), b("hidden_partial"), H, INNER)
        land(L + "attn_out", b("hidden_partial"), b("attn_out"), H, H, 0)

        if snapshot:
            copy(L + "prefix2", b("prefix2"), b("attn_out"), H)
        else:
            add2(L + "prefix2", b("prefix"), b("attn_out"), b("prefix2"))
        attn_res(L + "res_mlp", nb_mlp, b("prefix2"), w("sw_mlp"), b("mixed2"))
        rms(L + "norm_post", b("mixed2"), w("gamma_post"), b("normed"), H)

        if i == 0:
            gemm(L + "wgu", b("normed"), w("wgu"), b("dense_partial"), 2 * DENSE_I, H)
            land(L + "dense_gate", b("dense_partial"), b("dense_gate"), 2 * DENSE_I, DENSE_I, 0)
            land(L + "dense_up", b("dense_partial"), b("dense_up"), 2 * DENSE_I, DENSE_I, DENSE_I)
            situ(L + "situ", b("dense_gate"), b("dense_up"), b("dense_act"), DENSE_I)
            gemm(L + "w_dn", b("dense_act"), w("w_dn"), b("hidden_partial"), H, DENSE_I)
            land(L + "mlp_out", b("hidden_partial"), b("mlp_out"), H, H, 0)
        else:
            gemm(L + "router", b("normed"), w("w_router"), b("router_partial"), EXPERTS, H)
            step(L + "topk", tl("router_topk", f"k3_router_topk_b1_e{EXPERTS}_topk{TOPK}_kernel"),
                 w("bias"), b("topk_idx"), w("rs"), b("router_partial"), b("topk_weight"))
            gemm(L + "lat_down", b("normed"), w("w_lat_down"), b("latent_partial"), LATENT, H)
            land(L + "latent", b("latent_partial"), b("latent"), LATENT, LATENT, 0)
            prog.extend(gen_k3_moe.mega_pieces(ranks, tokens_max, wprefix=f"layers.{i}.")["steps"](
                b("latent"), b("topk_idx"), b("topk_weight"), b("routed_latent"), label=L))
            rms(L + "lat_norm", b("routed_latent"), w("gamma_lat"), b("routed_latent_norm"), LATENT)
            gemm(L + "lat_up", b("routed_latent_norm"), w("w_lat_up"), b("hidden_partial"), H, LATENT)
            land(L + "routed", b("hidden_partial"), b("routed"), H, H, 0)
            gemm(L + "wsh", b("normed"), w("wsh"), b("shared_partial"), 2 * SHARED, H)
            land(L + "shared_gate", b("shared_partial"), b("shared_gate"), 2 * SHARED, SHARED, 0)
            land(L + "shared_up", b("shared_partial"), b("shared_up"), 2 * SHARED, SHARED, SHARED)
            situ(L + "shared_situ", b("shared_gate"), b("shared_up"), b("shared_act"), SHARED)
            gemm(L + "sh_down", b("shared_act"), w("sh_down"), b("hidden_partial"), H, SHARED)
            land(L + "shared", b("hidden_partial"), b("shared"), H, H, 0)
            add2(L + "mlp_out", b("routed"), b("shared"), b("mlp_out"))
        add2(L + "hidden", b("prefix2"), b("mlp_out"), b("hidden"))

        # weights
        weight(f"layers.{i}.gamma_in", [H])
        weight(f"layers.{i}.gamma_post", [H])
        if nb_in > 0:
            weight(f"layers.{i}.sw_attn", [H], "f32")
        weight(f"layers.{i}.sw_mlp", [H], "f32")
        if is_mla(i):
            weight(f"layers.{i}.wfu", [MLA_FUSED, H])
            weight(f"layers.{i}.gamma_q_a", [Q_LORA])
            weight(f"layers.{i}.gamma_kv_a", [KV_LORA])
            weight(f"layers.{i}.w_q_b", [Q_B, Q_LORA])
            weight(f"layers.{i}.w_kv_b", [HEADS * 256, KV_LORA])
            weight(f"layers.{i}.scale", [1])
        else:
            weight(f"layers.{i}.wbig", [KDA_FUSED, H])
            weight(f"layers.{i}.wsm", [WSM, H])
            weight(f"layers.{i}.w_f_b", [INNER, HEAD_DIM])
            for s in "qkv":
                weight(f"layers.{i}.cw_{s}", [4, INNER], "f32")
            weight(f"layers.{i}.dt_bias", [INNER], "f32")
            weight(f"layers.{i}.a_log", [HEADS], "f32")
            weight(f"layers.{i}.gamma_o", [HEAD_DIM], "f32")
        weight(f"layers.{i}.w_o", [H, INNER])
        if i == 0:
            weight(f"layers.{i}.wgu", [2 * DENSE_I, H])
            weight(f"layers.{i}.w_dn", [H, DENSE_I])
        else:
            weight(f"layers.{i}.w_router", [EXPERTS, H])
            weight(f"layers.{i}.bias", [EXPERTS], "f32")
            weight(f"layers.{i}.rs", [1])
            weight(f"layers.{i}.w_lat_down", [LATENT, H])
            weight(f"layers.{i}.w_lat_up", [H, LATENT])
            weight(f"layers.{i}.gamma_lat", [LATENT])
            weight(f"layers.{i}.wsh", [2 * SHARED, H])
            weight(f"layers.{i}.sh_down", [H, SHARED])
            for n, d in mp["weights"].items():
                buffers[f"layers.{i}.{n}"] = dict(d)

    assert blocks == blocks_total
    attn_res("out.res", blocks_total, b("hidden"), b("sw_out"), b("mixed"))
    rms("out.norm", b("mixed"), b("gamma_final"), b("normed"), H)
    gemm("out.lm_head", b("normed"), b("w_lm"), b("logit_partial"), V, H)
    land("out.logits", b("logit_partial"), b("logits"), V, V, 0)
    step("out.argmax", "argmax", b("logits"), b("next_token"), i32(V))

    m = {
        "schema_version": kern_manifest.SCHEMA_VERSION,
        "model": f"kimi-k3-pruned-75pct/{layers}l/ep{ranks}",
        "vars": {T: {"max": tokens_max}, "seqs": {"max": seqs_max}},
        "topology": {"groups": {"ep": ranks}},
        "states": states,
        "buffers": buffers,
        "ops": ops,
        "programs": {"decode": prog},
    }
    return kern_manifest.normalize(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--max-ctx", type=int, default=16384)
    a = ap.parse_args()
    json.dump(build(a.layers, a.ranks, a.max_ctx), sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
