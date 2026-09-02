#!/usr/bin/env python3
"""Generate kern's Kimi-K3 (pruned, 224 experts) decode superstep at EP<R>.

    python3 tools/gen_k3_decode.py --layers 4 --ranks 1 > examples/k3-4l-ep1.json
    python3 tools/gen_k3_decode.py --ranks 4 > examples/k3-ep4.json

One SPMD manifest per world: every rank runs the whole dense trunk on its own
batch of sequences and serves its expert shard to the world through MegaMoE
(tools/gen_k3_moe.py). Program `decode`: one token per sequence (`tokens` ==
`seqs`, up to --seqs) through all layers — attention-residual mix, KDA (conv +
delta rule, state in a `bytes_per_seq` line) or absorbed paged MLA (latent
cache in `kv`), latent MoE (router → down-proj → MegaMoE → norm → up-proj,
plus the shared experts) or the dense MLP — then the output mix, final norm,
lm_head and argmax into `next_token`.

The kernels are kern's own (docs/k3-kernel-abi.md, tools/kernels-src/k3_*.cu):
B is a runtime argument, every launch takes one row per block.x, and the
landing / residual / norm / append work is fused into its neighbours, so a
layer is 8 cuBLAS GEMMs (`extern:cublas_bf16_tn_f32`, f32 partials) plus a
dozen kernels. The launch sequence still follows pegainfer's certified
`k3_step` operand for operand; only the kernel boundaries moved.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gen_k3_moe
import handwritten
import kern_manifest

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
NB_MAX = 8
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

T = "tokens"

# Launch geometry per entry, as the kernel headers document it
# (docs/k3-kernel-abi.md §1). grid.x is always the batch.
GEOM = {
    "kern_k3_attnres_rms": ([T, 1, 1], [1024, 1, 1], 0),
    "kern_k3_land_add_attnres_rms": ([T, 1, 1], [1024, 1, 1], 0),
    "kern_k3_land_add2": ([T, 1, 1], [1024, 1, 1], 0),
    "kern_k3_conv_silu": ([T, 3, INNER // 512], [128, 1, 1], 0),  # 4 columns per thread
    "kern_k3_kda_core": ([T, HEADS, 1], [128, 1, 1], 0),
    "kern_k3_mla_prep": ([T, 4, 1], [512, 1, 1], 0),  # 1 norm/append block + 3 gate segments
    "kern_k3_mla_paged_attn": ([T, 48, 1], [512, 1, 1], 0),  # 6 head groups x 8 KV splits (cluster of 8), 216 KB static smem
    "kern_k3_router_topk": ([T, 1, 1], [256, 1, 1], 0),
    "kern_k3_argmax_f32_partial": ([T, 64, 1], [1024, 1, 1], 0),
    "kern_k3_argmax_f32_final": ([T, 1, 1], [64, 1, 1], 0),
    "kern_k3_rms": ([T, 1, 1], [1024, 1, 1], 0),
}


def is_mla(i):
    return i in MLA_LAYERS


def launch(cubin, entry, grid=None, block=None, smem=None, **extra):
    g, b, s = GEOM.get(entry, (None, None, 0))
    l = {**handwritten.hw(cubin), "entry": entry, "block": block or b, "grid": grid or g, **extra}
    s = s if smem is None else smem
    if s:
        l["shared_mem"] = s
    return l


def build(layers, ranks, max_ctx, seqs_max):
    assert 1 <= layers <= LAYERS
    n_kda = sum(1 for i in range(layers) if not is_mla(i))
    mla_index = {i: k for k, i in enumerate(i for i in range(layers) if is_mla(i))}
    n_mla = len(mla_index)
    assert n_mla > 0, "the decode program needs at least one MLA layer (layer 3)"
    max_pages = -(-max_ctx // PAGE)
    page_stride = n_mla * PAGE * LATENT_ROW  # elements
    blocks_total = -(-layers // ATTN_RES_BLOCK)
    assert blocks_total <= NB_MAX
    mp = gen_k3_moe.mega_pieces(ranks, seqs_max)

    def per_row(n):
        return [T, -(-n // 1024), 1]

    ops = {
        "embedding": {
            "params": ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
            "impl": {"launches": [launch("embedding", "kern_embedding_i64_bf16", grid=[T, 1, 1], block=[256, 1, 1])]},
        },
        "gemm_f32": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32", "i32"],
            "impl": {"launches": [{"entry": "extern:cublas_bf16_tn_f32"}]},
        },
        # K1 residual stream
        "attnres_rms": {
            "params": ["in buffer<bf16>", "inout buffer<bf16>", "in buffer<f32>", "in buffer<bf16>", "out buffer<bf16>",
                       "i32", "i32", "i32"],
            "impl": {"launches": [launch("k3_residual", "kern_k3_attnres_rms")]},
        },
        # Layer 0: nb == 0 reads no snapshot, so `blocks` is a pure output there
        # (the verifier wants the first touch of a workspace to be a write).
        "attnres_rms_first": {
            "params": ["in buffer<bf16>", "out buffer<bf16>", "in buffer<f32>", "in buffer<bf16>", "out buffer<bf16>",
                       "i32", "i32", "i32"],
            "impl": {"launches": [launch("k3_residual", "kern_k3_attnres_rms")]},
        },
        "land_add_attnres_rms": {
            "params": ["in buffer<f32>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>", "in buffer<bf16>",
                       "out buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
            "impl": {"launches": [launch("k3_residual", "kern_k3_land_add_attnres_rms")]},
        },
        "land_add2": {
            "params": ["in buffer<f32>", "in buffer<f32>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
            "impl": {"launches": [launch("k3_residual", "kern_k3_land_add2")]},
        },
        # K2 / K3 KDA
        "conv_silu": {
            "params": ["in buffer<f32>", "in buffer<f32>", "inout state", "in buffer<i32>", "i64",
                       "out buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>", "i32"],
            "impl": {"launches": [launch("k3_conv_silu", "kern_k3_conv_silu")]},
        },
        "kda_core": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>", "in buffer<f32>",
                       "in buffer<bf16>", "in buffer<f32>", "in buffer<f32>", "in buffer<f32>",
                       "inout state", "in buffer<i32>", "i64", "out buffer<bf16>", "i32"],
            "impl": {"launches": [launch("k3_kda_core", "kern_k3_kda_core")]},
        },
        # K4 / K5 MLA
        "mla_prep": {
            "params": ["in buffer<f32>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<i64>", "inout state",
                       "i64", "i64", "out buffer<bf16>", "out buffer<bf16>", "i32"],
            "impl": {"launches": [launch("k3_mla_prep", "kern_k3_mla_prep")]},
        },
        "mla_paged_attn": {
            "params": ["in buffer<f32>", "in buffer<bf16>", "inout state", "in buffer<i32>", "i32", "i64",
                       "in buffer<i32>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
            "impl": {"launches": [launch("k3_mla_paged_attn", "kern_k3_mla_paged_attn")]},
        },
        # K6 / K7
        "router_topk": {
            "params": ["in buffer<f32>", "in buffer<f32>", "in buffer<bf16>", "out buffer<i32>", "out buffer<f32>", "i32"],
            "impl": {"launches": [launch("k3_router_argmax", "kern_k3_router_topk")]},
        },
        "argmax_f32": {
            "params": ["in buffer<f32>", "out buffer<i64>", "i32"],
            "impl": {
                "scratch": {"pmax": {"dtype": "f32", "shape": [T, 64]}, "pidx": {"dtype": "i32", "shape": [T, 64]}},
                "launches": [
                    launch("k3_router_argmax", "kern_k3_argmax_f32_partial",
                           params=["in buffer<f32>", "out buffer<f32>", "out buffer<i32>", "i32"],
                           args=[{"param": 0}, {"scratch": "pmax"}, {"scratch": "pidx"}, {"param": 2}]),
                    launch("k3_router_argmax", "kern_k3_argmax_f32_final",
                           params=["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32"],
                           args=[{"scratch": "pmax"}, {"scratch": "pidx"}, {"param": 1}, {"i32": 64}]),
                ],
            },
        },
        "rms": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
            "impl": {"launches": [launch("k3_land", "kern_k3_rms")]},
        },
    }
    # land / land_situ: grid.y depends on the width, one op per width.
    land_ops = {}

    def land_op(n):
        name = f"land_n{n}"
        if name not in ops:
            ops[name] = {
                "params": ["in buffer<f32>", "out buffer<bf16>", "i32", "i32", "i32", "i32"],
                "impl": {"launches": [launch("k3_land", "kern_k3_land", grid=per_row(n), block=[1024, 1, 1])]},
            }
        return name

    def situ_op(n):
        name = f"land_situ_n{n}"
        if name not in ops:
            ops[name] = {
                "params": ["in buffer<f32>", "out buffer<bf16>", "i32", "i32"],
                "impl": {"launches": [launch("k3_land", "kern_k3_land_situ", grid=per_row(n), block=[1024, 1, 1])]},
            }
        return name

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
    for n in ["hidden", "prefix2", "normed"]:
        work(n, H)
    buffers["blocks"] = {"dtype": "bf16", "shape": [T, NB_MAX, H], "kind": "workspace"}
    work("hidden_partial", H, "f32")
    work("kda_partial", KDA_FUSED, "f32")
    work("wsm_partial", WSM, "f32")
    for n in ["conv_q", "conv_k", "conv_v", "gated", "mla_gate"]:
        work(n, INNER)
    work("mla_fused_partial", MLA_FUSED, "f32")
    work("q_norm", Q_LORA)
    work("q_partial", Q_B, "f32")
    work("router_partial", EXPERTS, "f32")
    work("topk_idx", TOPK, "i32")
    work("topk_weight", TOPK, "f32")
    work("latent_partial", LATENT, "f32")
    for n in ["latent", "routed_latent", "routed_latent_norm"]:
        work(n, LATENT)
    work("routed_partial", H, "f32")
    work("shared_partial", 2 * SHARED, "f32")
    work("shared_act", SHARED)
    work("shared_partial2", H, "f32")
    work("dense_partial", 2 * DENSE_I, "f32")
    work("dense_act", DENSE_I)
    work("logit_partial", V, "f32")

    # ---- program
    prog = []
    b = lambda name, off=0: {"buf": name, "offset": off} if off else {"buf": name}
    i32 = lambda v: {"i32": v}
    i64 = lambda v: {"i64": v}
    B = {"var": T}

    def step(label, op, *args):
        prog.append({"label": label, "op": op, "args": list(args)})

    def gemm(label, a, w, c, n, k, ldc=None):
        """c[tokens, ldc] (cols 0..n from c's offset) = a[tokens, k] @ w[n, k]^T, f32."""
        step(label, "gemm_f32", a, w, c, B, i32(n), i32(k), i32(ldc or n))

    def land(label, p, o, n, off, ldc):
        step(label, land_op(n), p, o, i32(n), i32(off), i32(ldc), B)

    def land_situ(label, p, act, n):
        step(label, situ_op(n), p, act, i32(n), B)

    step("embed", "embedding", b("token_ids"), b("embed"), b("hidden"), B, i32(H))

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

        # residual mix in + snapshot + norm → normed
        step(L + "res_in", "attnres_rms" if nb_in > 0 else "attnres_rms_first", b("hidden"), b("blocks"),
             w("sw_attn") if nb_in > 0 else w("sw_mlp"),
             w("gamma_in"), b("normed"), i32(nb_in), i32(int(snapshot)), B)
        if is_mla(i):
            k = mla_index[i]
            layer_off = k * PAGE * LATENT_ROW  # elements
            gemm(L + "wfu", b("normed"), w("wfu"), b("mla_fused_partial"), MLA_FUSED, H)
            step(L + "mla_prep", "mla_prep", b("mla_fused_partial"), w("gamma_q_a"), w("gamma_kv_a"), b("slot_mapping"),
                 {"state": "kv"}, i64(layer_off), i64(page_stride), b("q_norm"), b("mla_gate"), B)
            gemm(L + "q_b", b("q_norm"), w("w_q_b"), b("q_partial"), Q_B, Q_LORA)
            step(L + "attn", "mla_paged_attn", b("q_partial"), w("w_kv_b"), {"state": "kv", "offset": layer_off * 2},
                 b("block_table"), i32(max_pages), i64(page_stride), b("seq_lens"), w("scale"), b("mla_gate"),
                 b("gated"), B)
        else:
            line = b("kda.line_index", kda_k * seqs_max * 4)
            kda_k += 1
            gemm(L + "qkvg", b("normed"), w("wbig"), b("kda_partial"), KDA_FUSED, H)
            gemm(L + "wsm", b("normed"), w("wsm"), b("wsm_partial"), WSM, H)
            step(L + "conv", "conv_silu", b("kda_partial"), w("cw"), {"state": "kda"}, line, i64(KDA_LINE_BYTES),
                 b("conv_q"), b("conv_k"), b("conv_v"), B)
            step(L + "kda_core", "kda_core", b("conv_q"), b("conv_k"), b("conv_v"), b("wsm_partial"), b("kda_partial"),
                 w("w_f_b"), w("dt_bias"), w("a_log"), w("gamma_o"), {"state": "kda"}, line, i64(KDA_LINE_BYTES),
                 b("gated"), B)
        gemm(L + "o_proj", b("gated"), w("w_o"), b("hidden_partial"), H, INNER)
        # attn_out landing + residual (or snapshot replace) + mix + norm → prefix2, normed
        step(L + "res_mlp", "land_add_attnres_rms", b("hidden_partial"), b("hidden"), b("blocks"), w("sw_mlp"),
             w("gamma_post"), b("prefix2"), b("normed"), i32(nb_mlp), i32(int(snapshot)), B)

        if i == 0:
            gemm(L + "wgu", b("normed"), w("wgu"), b("dense_partial"), 2 * DENSE_I, H)
            land_situ(L + "situ", b("dense_partial"), b("dense_act"), DENSE_I)
            gemm(L + "w_dn", b("dense_act"), w("w_dn"), b("routed_partial"), H, DENSE_I)
            step(L + "hidden", "land_add2", b("routed_partial"), b("routed_partial"), b("prefix2"), b("hidden"), i32(0), B)
        else:
            gemm(L + "router", b("normed"), w("w_router"), b("router_partial"), EXPERTS, H)
            step(L + "topk", "router_topk", b("router_partial"), w("bias"), w("rs"), b("topk_idx"), b("topk_weight"), B)
            gemm(L + "lat_down", b("normed"), w("w_lat_down"), b("latent_partial"), LATENT, H)
            land(L + "latent", b("latent_partial"), b("latent"), LATENT, 0, LATENT)
            prog.extend(gen_k3_moe.mega_pieces(ranks, seqs_max, wprefix=f"layers.{i}.")["steps"](
                b("latent"), b("topk_idx"), b("topk_weight"), b("routed_latent"), label=L))
            step(L + "lat_norm", "rms", b("routed_latent"), w("gamma_lat"), b("routed_latent_norm"), i32(LATENT), B)
            gemm(L + "lat_up", b("routed_latent_norm"), w("w_lat_up"), b("routed_partial"), H, LATENT)
            gemm(L + "wsh", b("normed"), w("wsh"), b("shared_partial"), 2 * SHARED, H)
            land_situ(L + "shared_situ", b("shared_partial"), b("shared_act"), SHARED)
            gemm(L + "sh_down", b("shared_act"), w("sh_down"), b("shared_partial2"), H, SHARED)
            step(L + "hidden", "land_add2", b("routed_partial"), b("shared_partial2"), b("prefix2"), b("hidden"), i32(1), B)

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
            weight(f"layers.{i}.cw", [3, 4, INNER], "f32")
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
    step("out.res", "attnres_rms", b("hidden"), b("blocks"), b("sw_out"), b("gamma_final"), b("normed"),
         i32(blocks_total), i32(0), B)
    gemm("out.lm_head", b("normed"), b("w_lm"), b("logit_partial"), V, H)
    step("out.argmax", "argmax_f32", b("logit_partial"), b("next_token"), i32(V))

    m = {
        "schema_version": kern_manifest.SCHEMA_VERSION,
        "model": f"kimi-k3-pruned-75pct/{layers}l/ep{ranks}",
        "vars": {T: {"max": seqs_max}, "seqs": {"max": seqs_max}},
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
    ap.add_argument("--seqs", type=int, default=64, help="sequences per rank (the `tokens`/`seqs` bound)")
    a = ap.parse_args()
    json.dump(build(a.layers, a.ranks, a.max_ctx, a.seqs), sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
