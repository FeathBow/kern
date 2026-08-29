#!/usr/bin/env python3
"""Generate examples/qwen3-4b.json from mined vLLM data.

bs=1 manifest，两个 program：`prefill`（chunk 级，tokens ∈ [1, CHUNK_MAX]，
只落 KV 不出 logits）+ `decode`（tokens=1，出 next_token）。chunked prefill
= caller 连调 prefill 若干次 + 最后一个 token 走 decode（decode 就是
"prefill_last"，免掉 symbol 依赖的 offset）。数据源是 TRITON_ATTN backend
的 capture（唯一 flat-ABI 的 attention backend——FA4/trtllm-gen 都是
packed struct / TMA descriptor，不可 rebind）：

- 四个 CUDA flat 核（rms/rms_head/rope/fused）+ Triton 版 reshape_and_cache
  + `kernel_unified_attention`(3D decode 实例) + `reduce_segments`，全部用
  真实挖到的 symbol、逐参数类型/方向、标量字面量（取自代表性 decode
  forward）。注意 Triton 同名核不同 constexpr 实例 ABI 不同（unified 的
  2D prefill 实例 28 参数、3D decode 实例 31 参数；reduce 的 num_seqs=1
  被 Triton 特化进 binary 不再传参）——这里 pin 的是 decode 实例。
- GEMM 是 runtime 特判（symbol 前缀 `extern:`，cublasLt）；embedding 是
  待写的 Triton 占位。
- KV state 布局从 vLLM 的逐层池改为层交织 `[page][layer][16][8][2][128]`
  （同一批 kernel，靠 stride 参数 ×LAYERS 和 state offset 字面量适配），
  bytes_per_token = 36*2*8*128*2 = 147456。
- 发射前对挖矿数据做结构断言：q/k/v 在 qkv 中的视图偏移、residual 全程
  同址、逐层权重互异、KV 池 k/v 相距 256B、cache/attention 共享同一
  KV 池与 scale 指针、unified 与 reduce 共享 segm 缓冲——连线是手写的，
  挖矿数据负责证伪它。

跑法：python3 tools/gen_qwen3_decode.py [dumped-kernels/pid<N>/launches.jsonl]
"""

import json
import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mine_capture as mc

HIDDEN = 2560
LAYERS = 36
HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
FFN = 9728
VOCAB = 151936
MAX_POS = 4096
Q_DIM = HEADS * HEAD_DIM      # 4096
KV_DIM = KV_HEADS * HEAD_DIM  # 1024
QKV_DIM = Q_DIM + 2 * KV_DIM  # 6144
BF16 = 2
NUM_SEGMENTS = 16             # unified 3D 实例的 NUM_SEGMENTS_PER_SEQ（grid.z）
CHUNK_MAX = 2048              # prefill 单 chunk 上限（tokens symbol 的 max）
BLOCK_Q = 4                   # unified 2D 实例每 block 的 query 行数（拟合证实）

# 层交织 KV 布局（vLLM 逐层池 -> 我们的单 state）：
# 一个 block(16 token) 在一层里占 16*8*2*128 = 32768 elems；层间连续。
BLOCK_ELEMS_PER_LAYER = 16 * KV_HEADS * 2 * HEAD_DIM
BLOCK_STRIDE = LAYERS * BLOCK_ELEMS_PER_LAYER          # 传给 kernel 的 elems
LAYER_KV_BYTES = BLOCK_ELEMS_PER_LAYER * BF16          # state offset 步长
KV_BYTES_PER_TOKEN = LAYERS * 2 * KV_DIM * BF16        # 147456
V_BYTE_OFF = 2 * HEAD_DIM                              # 挖矿实测 k/v 相距 256B
MAX_BLOCKS = MAX_POS // 16                             # block_table_stride=256 实测吻合

SYMS = {
    "rms": "rms_norm_kernelIN3c108BFloat16ELi8ELi2",
    "rms_head": "rms_norm_kernelIN3c108BFloat16ELi8ELi3",
    "rope": "rotary_embedding",
    "silu": "act_and_mul",
    "fused": "fused_add_rms",
    "cache": "reshape_and_cache_kernel_flash",
    "unified": "kernel_unified_attention",
    "reduce": "reduce_segments",
}


def pv(rec, i):
    return int.from_bytes(bytes.fromhex(rec["params"][i]["data"]), "little")


def pick_forwards(jsonl):
    """代表性 decode forward + 最后两个真实 prefill forward（多 token、
    位于 profiling dummy pass 之后，用两个不同长度拟合/证伪 2D grid）。"""
    recs = mc.load(jsonl)
    windows = mc.slice_windows(recs, mc.GAP_MS_DEFAULT)
    _, _, forwards = mc.slice_forwards(windows)
    _, tokens, forwards, _ = mc.pick_tokens_reference(forwards)
    di = max(i for i, t in enumerate(tokens) if t == 1)
    prefills = [(t, forwards[i][1]) for i, t in enumerate(tokens)
                if 1 < t <= CHUNK_MAX and i > tokens.index(1)][-2:]
    assert len(prefills) == 2, "需要两个真实 prefill forward 拟合 grid"
    return forwards[di][1], prefills


def extract(fwd):
    """代表性 decode forward -> 各 flat 核按序分组。"""
    by = {k: [] for k in SYMS}
    for r in fwd:
        for tag, pat in SYMS.items():
            if pat in r["symbol"] and isinstance(r.get("params"), list):
                by[tag].append(r)
                break
    assert len(by["rms"]) == 1, by["rms"]
    assert len(by["rms_head"]) == 2 * LAYERS
    assert len(by["rope"]) == LAYERS
    assert len(by["silu"]) == LAYERS
    assert len(by["fused"]) == 2 * LAYERS
    assert len(by["cache"]) == LAYERS
    assert len(by["unified"]) == LAYERS
    assert len(by["reduce"]) == LAYERS
    # Triton 同名不同实例：decode 3D 实例 31 参数 / reduce 12 参数
    assert len(by["unified"][0]["params"]) == 31, "不是 3D decode 实例"
    assert len(by["reduce"][0]["params"]) == 12, "num_seqs 未被特化，不是 bs=1 实例"
    return by


def check_topology(by):
    """挖矿地址证伪手写连线。"""
    residual = pv(by["rms"][0], 1)
    eps = pv(by["rms"][0], 9)
    weights = set()
    for i in range(LAYERS):
        q, k = by["rms_head"][2 * i], by["rms_head"][2 * i + 1]
        rope, cache = by["rope"][i], by["cache"][i]
        uni, red = by["unified"][i], by["reduce"][i]
        post, nxt = by["fused"][2 * i], by["fused"][2 * i + 1]
        qkv_base = pv(q, 1)
        assert pv(k, 1) - qkv_base == Q_DIM * BF16, "k 视图不在 qkv+8192"
        assert pv(rope, 1) == pv(q, 0) and pv(rope, 2) == pv(k, 0), \
            "rope 读的不是 normed q/k"
        assert pv(cache, 0) == pv(k, 0), "cache 的 key 不是 normed k"
        assert pv(cache, 1) == qkv_base + (Q_DIM + KV_DIM) * BF16, \
            "v 视图不在 qkv+10240"
        assert pv(cache, 3) - pv(cache, 2) == V_BYTE_OFF, "KV 池 k/v 间距非 256B"
        assert pv(uni, 1) == pv(q, 0), "attention 的 query 不是 normed q"
        assert pv(uni, 2) == pv(cache, 2) and pv(uni, 3) == pv(cache, 3), \
            "attention 与 cache 的 KV 池不一致"
        assert pv(uni, 7) == pv(cache, 5) and pv(uni, 8) == pv(cache, 6), \
            "attention 与 cache 的 k/v scale 不一致"
        assert pv(uni, 17) == pv(uni, 5), "unified [17] 不是 seq_lens 复用"
        assert [pv(red, j) for j in (1, 2, 3)] == [pv(uni, j) for j in (26, 27, 28)], \
            "reduce 读的不是 unified 写的 segm 缓冲"
        assert pv(red, 0) == pv(uni, 0), "reduce 输出与 unified 占位输出不同址"
        assert pv(red, 4) == pv(uni, 5) and pv(red, 9) == pv(uni, 24), \
            "reduce 的 seq_lens/cu_seqlens 与 unified 不一致"
        assert pv(post, 2) == residual and pv(nxt, 2) == residual, "residual 漂移"
        for r, wi in [(q, 7), (k, 7), (post, 3), (nxt, 3), (cache, 5), (cache, 6)]:
            weights.add(pv(r, wi))
        for r, ei in [(q, 9), (k, 9), (post, 4), (nxt, 4)]:
            assert pv(r, ei) == eps, "eps 不一致"
    assert len(weights) == 6 * LAYERS, "逐层权重指针有重合"
    scale = struct.unpack("<f", struct.pack("<I", pv(by["unified"][0], 6)))[0]
    return struct.unpack("<f", struct.pack("<I", eps))[0], scale


def check_prefill(prefills, by_dec):
    """真实 prefill forward 证伪 prefill program 的连线与 2D 实例几何。
    返回 (symbol, block, smem)。decode 时 tokens=1 掩盖的三个字面量在这里
    现形：head-norm 输入 stride=QKV_DIM（融合 qkv 行距）、head-norm [10]
    = tokens*heads（需要表达式标量）、cache value stride=QKV_DIM。"""
    uni_dec = by_dec["unified"][0]
    ref = None
    for t, fwd in prefills:
        by = {k: [] for k in SYMS}
        for r in fwd:
            for tag, pat in SYMS.items():
                if pat in r["symbol"] and isinstance(r.get("params"), list):
                    by[tag].append(r)
                    break
        assert len(by["unified"]) == LAYERS and len(by["reduce"]) == 0, \
            "prefill forward 不该出现 reduce_segments"
        u = by["unified"][0]
        assert len(u["params"]) == 28, "不是 2D prefill 实例"
        # 2D grid = [ceil_div(tokens, BLOCK_Q), kv_heads, 1]
        assert u["grid"] == [-(-t // BLOCK_Q), KV_HEADS, 1], (t, u["grid"])
        # 接口即 2D launch ABI：26 个前缀参数 + 两个尾部 i64 与 decode 3D
        # 实例的标量逐位一致（segm 三参恰好是被裁掉的实现细节）
        for j in list(range(9, 17)) + list(range(18, 24)) + [25]:
            assert pv(u, j) == pv(uni_dec, j), (j, pv(u, j), pv(uni_dec, j))
        assert pv(u, 6) == pv(uni_dec, 6), "softmax scale 不一致"
        assert pv(u, 17) == pv(u, 5), "unified [17] 不是 seq_lens 复用"
        q, k = by["rms_head"][0], by["rms_head"][1]
        assert pv(q, 3) == QKV_DIM and pv(k, 3) == QKV_DIM, \
            "head-norm 输入 stride 不是融合 qkv 行距"
        assert pv(q, 10) == t * HEADS and pv(k, 10) == t * KV_HEADS, \
            "head-norm [10] 不是 tokens*heads"
        assert pv(u, 1) == pv(by["rope"][0], 1), "attention query 不是 roped q"
        assert pv(by["cache"][0], 8) == QKV_DIM, \
            "cache value stride 不是融合 qkv 行距"
        ref = u
    return ref["symbol"], ref["block"], ref["dynamic_shared_mem_bytes"]


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


def u8(v):
    return {"u8": v}


def d(label, kernel, args):
    return {"label": label, "kernel": kernel, "args": args}


def a(i):
    return {"arg": i}


def scr(name, off=0):
    return {"scratch": name, "offset": off} if off else {"scratch": name}


def step(symbol, params, block, grid, args, shared_mem=None, cubin=None):
    s = {"symbol": symbol, "params": params, "block": block, "grid": grid,
         "args": args}
    if shared_mem is not None:
        s["shared_mem"] = shared_mem
    if cubin is not None:
        s["cubin"] = cubin
    return s


def single(symbol, params, block, grid, shared_mem=None, cubin=None):
    """单步实现，恒等布线：接口即该核的 launch ABI。"""
    return {"params": params,
            "impl": {"steps": [step(symbol, params, block, grid,
                                    [a(i) for i in range(len(params))],
                                    shared_mem, cubin)]}}


def build(by, eps, scale, pf):
    pf_sym, pf_block, pf_smem = pf
    buffers = {
        "token_ids": {"dtype": "i64", "shape": ["tokens"], "class": "input"},
        "positions": {"dtype": "i64", "shape": ["tokens"], "class": "input"},
        "slot_mapping": {"dtype": "i64", "shape": ["tokens"], "class": "input"},
        "block_table": {"dtype": "i32", "shape": [MAX_BLOCKS], "class": "input"},
        # bs=1：seq_lens 内容 = 已见 token 数，cu_seqlens_q = [0, 本次 q 数]，
        # caller 每次调用前填
        "seq_lens": {"dtype": "i32", "shape": [1], "class": "input"},
        "cu_seqlens_q": {"dtype": "i32", "shape": [2], "class": "input"},
        # logits/next_token 只属于 decode（tokens=1），定常形状——否则按
        # CHUNK_MAX 上界分配要多付 ~600MB
        "logits": {"dtype": "bf16", "shape": [1, VOCAB], "class": "workspace"},
        "next_token": {"dtype": "i64", "shape": [1], "class": "output"},
    }
    for name, shape in {
        "residual": ["tokens", HIDDEN],
        "x": ["tokens", HIDDEN],
        "y": ["tokens", HIDDEN],
        "qkv": ["tokens", QKV_DIM],
        "q_n": ["tokens", Q_DIM],
        "k_n": ["tokens", KV_DIM],
        "attn_out": ["tokens", Q_DIM],
        "gate_up": ["tokens", 2 * FFN],
        "ffn_act": ["tokens", FFN],
    }.items():
        buffers[name] = {"dtype": "bf16", "shape": shape, "class": "workspace"}

    def weight(name, shape, dtype="bf16"):
        buffers[name] = {"dtype": dtype, "shape": shape, "class": "weight"}

    weight("model.embed_tokens.weight", [VOCAB, HIDDEN])
    weight("model.norm.weight", [HIDDEN])
    weight("lm_head.weight", [VOCAB, HIDDEN])
    weight("rope.cos_sin_cache", [MAX_POS, HEAD_DIM])
    weight("kv_scales", [2 * LAYERS], "f32")
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        weight(p + "input_layernorm.weight", [HIDDEN])
        weight(p + "post_attention_layernorm.weight", [HIDDEN])
        weight(p + "self_attn.qkv_proj.weight", [QKV_DIM, HIDDEN])
        weight(p + "self_attn.q_norm.weight", [HEAD_DIM])
        weight(p + "self_attn.k_norm.weight", [HEAD_DIM])
        weight(p + "self_attn.o_proj.weight", [HIDDEN, Q_DIM])
        weight(p + "mlp.gate_up_proj.weight", [2 * FFN, HIDDEN])
        weight(p + "mlp.down_proj.weight", [HIDDEN, FFN])

    def blk(tag):
        return by[tag][0]["block"]

    def smem(tag):
        return by[tag][0]["dynamic_shared_mem_bytes"]

    T = sym("tokens")
    RMS_PARAMS = ["out buffer<bf16>", "in buffer<bf16>", "i64", "i64", "i64",
                  "i64", "i64", "in buffer<bf16>", "i64", "f32", "i32", "i32"]
    # unified 的完整 launch ABI（31 参数）；接口砍掉三个 segm 分部和缓冲
    # （26/27/28），它们是实现细节，降为 impl scratch
    UNIFIED_PARAMS = [
        "out buffer<bf16>",  # 3D 实例不写它，ABI 要求非空指针；reduce 写
        "in buffer<bf16>", "inout ptr", "inout ptr",
        "in buffer<i32>", "in buffer<i32>", "f32",
        "in buffer<f32>", "in buffer<f32>", "f32", "f32",
        "i64", "i64", "i64", "i64", "i64", "i64",
        "in buffer<i32>", "i64", "i64", "i64", "i64", "i64",
        "i64", "in buffer<i32>", "i32", "out buffer<f32>",
        "out buffer<f32>", "out buffer<f32>", "i64", "i64"]
    ATTN_IFACE = UNIFIED_PARAMS[:26] + UNIFIED_PARAMS[29:]
    REDUCE_PARAMS = ["out buffer<bf16>", "in buffer<f32>", "in buffer<f32>",
                     "in buffer<f32>", "in buffer<i32>", "f32", "i64", "i64",
                     "i64", "in buffer<i32>", "i64", "i64"]

    # 真实挖到的 kernel ABI。kernel = 接口 + impl（可整体替换的实现）：
    # 多数是单步恒等布线；attn / argmax 是"微程序 + 自报 scratch"的两步实现，
    # 分部和缓冲不再泄漏进调用方的 buffer 表。
    kernels = {
        "rms_norm": single(by["rms"][0]["symbol"], RMS_PARAMS, blk("rms"),
                           [T, 1, 1]),
        # 同一 head-norm 核的两个实例化：grid 随 head 数烘焙在实现里
        "rms_norm_qhead": single(by["rms_head"][0]["symbol"], RMS_PARAMS,
                                 blk("rms_head"), [mul(T, HEADS), 1, 1]),
        "rms_norm_khead": single(by["rms_head"][0]["symbol"], RMS_PARAMS,
                                 blk("rms_head"), [mul(T, KV_HEADS), 1, 1]),
        "rotary_embedding": single(
            by["rope"][0]["symbol"],
            ["in buffer<i64>", "inout buffer<bf16>", "inout buffer<bf16>",
             "in buffer<bf16>", "i32", "i64", "i64", "i64", "i32", "i32",
             "i32", "i64", "u8"],
            blk("rope"), [T, 1, 1]),
        "reshape_and_cache": single(
            by["cache"][0]["symbol"],
            ["in buffer<bf16>", "in buffer<bf16>", "inout ptr",
             "inout ptr", "in buffer<i64>", "in buffer<f32>",
             "in buffer<f32>", "i64", "i64", "i64", "i64", "i64", "i64",
             "i64", "i64", "i64"],
            blk("cache"), [T, 1, 1]),
        # decode attention：3D split-KV 微程序。decode 恒 tokens=1，grid 与
        # scratch 定常（scratch 若挂 tokens 会按 CHUNK_MAX 上界多付 ~500MB）
        "attn": {
            "params": ATTN_IFACE,
            "impl": {
                "scratch": {
                    "segm_out": {"dtype": "f32",
                                 "shape": [1, HEADS, NUM_SEGMENTS, HEAD_DIM]},
                    "segm_max": {"dtype": "f32",
                                 "shape": [1, HEADS, NUM_SEGMENTS]},
                    "segm_expsum": {"dtype": "f32",
                                    "shape": [1, HEADS, NUM_SEGMENTS]},
                },
                "steps": [
                    step(by["unified"][0]["symbol"], UNIFIED_PARAMS,
                         blk("unified"), [1, KV_HEADS, NUM_SEGMENTS],
                         [a(i) for i in range(26)]
                         + [scr("segm_out"), scr("segm_max"), scr("segm_expsum"),
                            a(26), a(27)],
                         shared_mem=smem("unified")),
                    step(by["reduce"][0]["symbol"], REDUCE_PARAMS,
                         blk("reduce"), [1, HEADS, 1],
                         [a(0), scr("segm_out"), scr("segm_max"),
                          scr("segm_expsum"), a(5), f32(1.0), i64(Q_DIM),
                          i64(HEAD_DIM), i64(MAX_BLOCKS), a(24), i64(0), i64(0)],
                         shared_mem=smem("reduce")),
                ],
            },
        },
        # prefill attention：同一接口的另一份实现——2D 实例单步无 scratch
        # （28 参 launch ABI 恰好就是接口本身，这是接口切分正确的实证）
        "attn_prefill": single(pf_sym, ATTN_IFACE, pf_block,
                               [cdiv(T, BLOCK_Q), KV_HEADS, 1],
                               shared_mem=pf_smem),
        "silu_mul": single(
            by["silu"][0]["symbol"],
            ["out buffer<bf16>", "in buffer<bf16>", "i32", "i32", "f32", "i32"],
            blk("silu"), [T, 1, 1]),
        "fused_add_rms_norm": single(
            by["fused"][0]["symbol"],
            ["inout buffer<bf16>", "i64", "inout buffer<bf16>",
             "in buffer<bf16>", "f32", "i32", "i32", "i64"],
            blk("fused"), [T, 1, 1]),
        "embedding": single(
            "kern_embedding_i64_bf16",
            ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>",
             "i32", "i32"],
            [256, 1, 1], [T, 1, 1], cubin="embedding.cubin"),
        # c[m,n] = a[m,k] @ w[n,k]^T；runtime 按 extern: 前缀特判走 cublasLt
        "gemm": single(
            "extern:cublaslt_bf16_tn",
            ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>",
             "i32", "i32", "i32"],
            [1, 1, 1], [1, 1, 1]),
        # greedy 采样下沉：手写两段式（tools/kernels-src/argmax.cu）。分部
        # 缓冲与两次 launch 全在 impl 内，接口只有 (logits, next_token, n)
        "argmax": {
            "params": ["in buffer<bf16>", "out buffer<i64>", "i32"],
            "impl": {
                "scratch": {
                    "pmax": {"dtype": "f32", "shape": [1, 64]},
                    "pidx": {"dtype": "i32", "shape": [1, 64]},
                },
                "steps": [
                    step("kern_argmax_partial_bf16",
                         ["in buffer<bf16>", "out buffer<f32>",
                          "out buffer<i32>", "i32"],
                         [1024, 1, 1], [1, 64, 1],
                         [a(0), scr("pmax"), scr("pidx"), a(2)],
                         cubin="argmax.cubin"),
                    step("kern_argmax_final_i64",
                         ["in buffer<f32>", "in buffer<i32>",
                          "out buffer<i64>", "i32"],
                         [64, 1, 1], [1, 1, 1],
                         [scr("pmax"), scr("pidx"), a(1), i32(64)],
                         cubin="argmax.cubin"),
                ],
            },
        },
    }

    def gemm(label, ab, w, c, m, n, k):
        return d(label, "gemm", [buf(ab), buf(w), buf(c), m, i32(n), i32(k)])

    def head_norm(label, kernel, out, off, w, heads):
        # 标量字面量与挖到的 launch 逐位一致。[3]=输入行距=QKV_DIM（q/k 视
        # 图都在融合 qkv 里，行距 6144——decode capture 里的 4096/1024 是
        # vLLM 自己的连续布局，tokens=1 时掩盖了差异，prefill capture 现形）；
        # [10]=tokens*heads（总 head 数上界），需要表达式标量
        return d(label, kernel,
                 [buf(out), buf("qkv", off), i64(HEAD_DIM), i64(QKV_DIM), i64(0),
                  i64(heads), i64(0), buf(w), i64(0), f32(eps),
                  expr(mul(T, heads)), i32(HEAD_DIM)])

    def fused(label, x, w):
        return d(label, "fused_add_rms_norm",
                 [buf(x), i64(HIDDEN), buf("residual"), buf(w), f32(eps), T,
                  i32(HIDDEN), i64(HIDDEN)])

    def forward(attn_kernel, with_head):
        """embed + 36 层的直线 dispatch 表。prefill 与 decode 共享全部连线，
        只差 attention 实现（同一接口的两份 impl）和收尾（final_norm +
        lm_head + argmax 只在 decode——最后一个 prompt token 走 decode 出
        首个 logits，prefill 只落 KV）。"""
        ds = [
            d("embed", "embedding",
              [buf("token_ids"), buf("model.embed_tokens.weight"),
               buf("residual"), T, i32(HIDDEN)]),
            d("l0.input_norm", "rms_norm",
              [buf("x"), buf("residual"), i64(HIDDEN), i64(0), i64(0), i64(0),
               i64(0), buf("model.layers.0.input_layernorm.weight"), i64(0),
               f32(eps), T, i32(HIDDEN)]),
        ]
        for i in range(LAYERS):
            p = f"model.layers.{i}."
            l = f"l{i}."
            koff = i * LAYER_KV_BYTES
            ks, vs = buf("kv_scales", i * 8), buf("kv_scales", i * 8 + 4)
            last = i + 1 == LAYERS
            ds += [
                gemm(l + "qkv_proj", "x", p + "self_attn.qkv_proj.weight",
                     "qkv", T, QKV_DIM, HIDDEN),
                head_norm(l + "q_norm", "rms_norm_qhead", "q_n", 0,
                          p + "self_attn.q_norm.weight", HEADS),
                head_norm(l + "k_norm", "rms_norm_khead", "k_n", Q_DIM * BF16,
                          p + "self_attn.k_norm.weight", KV_HEADS),
                d(l + "rope", "rotary_embedding",
                  [buf("positions"), buf("q_n"), buf("k_n"),
                   buf("rope.cos_sin_cache"),
                   i32(HEAD_DIM), i64(Q_DIM), i64(KV_DIM), i64(HEAD_DIM),
                   i32(HEADS), i32(KV_HEADS), i32(HEAD_DIM), i64(0), u8(0)]),
                d(l + "kv_write", "reshape_and_cache",
                  # [8]=value 行距=QKV_DIM（v 视图在融合 qkv 里；decode
                  # capture 的 1024 同样是 tokens=1 下的假常量）
                  [buf("k_n"), buf("qkv", (Q_DIM + KV_DIM) * BF16),
                   state("kv", koff), state("kv", koff + V_BYTE_OFF),
                   buf("slot_mapping"), ks, vs,
                   i64(KV_DIM), i64(QKV_DIM), i64(BLOCK_STRIDE),
                   i64(2 * HEAD_DIM), i64(0), i64(0),
                   i64(KV_HEADS * 2 * HEAD_DIM), i64(0), i64(0)]),
                d(l + "attn", attn_kernel,
                  [buf("attn_out"), buf("q_n"),
                   state("kv", koff), state("kv", koff + V_BYTE_OFF),
                   buf("block_table"), buf("seq_lens"), f32(scale), ks, vs,
                   f32(1.0), f32(0.0),
                   i64(MAX_BLOCKS), i64(Q_DIM), i64(HEAD_DIM), i64(Q_DIM),
                   i64(HEAD_DIM), i64(0), buf("seq_lens"),
                   i64(BLOCK_STRIDE), i64(KV_HEADS * 2 * HEAD_DIM),
                   i64(2 * HEAD_DIM),
                   i64(BLOCK_STRIDE), i64(KV_HEADS * 2 * HEAD_DIM),
                   i64(2 * HEAD_DIM),
                   buf("cu_seqlens_q"), i32(1),
                   i64(0), i64(0)]),
                gemm(l + "o_proj", "attn_out", p + "self_attn.o_proj.weight",
                     "y", T, HIDDEN, Q_DIM),
                fused(l + "post_attn_norm", "y",
                      p + "post_attention_layernorm.weight"),
                gemm(l + "gate_up", "y", p + "mlp.gate_up_proj.weight",
                     "gate_up", T, 2 * FFN, HIDDEN),
                d(l + "silu_mul", "silu_mul",
                  [buf("ffn_act"), buf("gate_up"), i32(FFN), i32(0), f32(1.0),
                   i32(0)]),
                gemm(l + "down_proj", "ffn_act", p + "mlp.down_proj.weight",
                     "x", T, HIDDEN, FFN),
            ]
            if last and not with_head:
                continue  # prefill：final_norm 喂 lm_head，没有 lm_head 就不需要
            ds.append(fused(l + ("final_norm" if last else "next_input_norm"),
                            "x",
                            f"model.layers.{i + 1}.input_layernorm.weight"
                            if not last else "model.norm.weight"))
        if with_head:
            # decode 恒 tokens=1：只算 x 第 0 行的 logits，m=1 字面量
            ds.append(gemm("lm_head", "x", "lm_head.weight", "logits",
                           i32(1), VOCAB, HIDDEN))
            ds.append(d("sample", "argmax",
                        [buf("logits"), buf("next_token"), i32(VOCAB)]))
        return ds

    return {
        "meta": {"version": 2, "model": "qwen3-4b"},
        # bs=1；prefill 按 chunk 调用（tokens ≤ CHUNK_MAX），decode 恒 tokens=1
        "symbols": {"tokens": {"max": CHUNK_MAX}},
        "states": {"kv": {"bytes_per_token": KV_BYTES_PER_TOKEN}},
        "buffers": buffers,
        "kernels": kernels,
        "programs": {
            "prefill": {"dispatches": forward("attn_prefill", with_head=False)},
            "decode": {"dispatches": forward("attn", with_head=True)},
        },
    }


def main():
    jsonl = sys.argv[1] if len(sys.argv) > 1 else str(
        pathlib.Path(__file__).resolve().parent.parent
        / "dumped-kernels" / "pid3977275" / "launches.jsonl")
    fwd, prefills = pick_forwards(jsonl)
    by = extract(fwd)
    eps, scale = check_topology(by)
    pf = check_prefill(prefills, by)
    manifest = build(by, eps, scale, pf)
    out = pathlib.Path(__file__).resolve().parent.parent / "examples" / "qwen3-4b.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    counts = {p: len(v["dispatches"]) for p, v in manifest["programs"].items()}
    print(f"topology checks passed (eps={eps!r}, attn scale={scale!r}, "
          f"prefill fwds={[t for t, _ in prefills]})")
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB, "
          f"{len(manifest['buffers'])} buffers, dispatches {counts})")


if __name__ == "__main__":
    main()
