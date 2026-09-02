#!/usr/bin/env python3
"""Export the pruned Kimi-K3 checkpoint into kern's K3 decode weight blobs.

    python3 tools/export_k3.py --out /data/susun/kern-k3/4l --layers 4 --ranks 1,4

Writes, resumably (existing files are kept):
  dense/bookends.safetensors   embed, gamma_final, sw_out, w_lm
  dense/l<i>.safetensors       layer i's dense slots (see below)
  experts/ep<R>-r<r>-l<i>.safetensors   layer i's experts [r*E/R, (r+1)*E/R)
                               in MegaMoE form (tools/export_k3_moe.py)
Tensor name = manifest buffer name (`layers.<i>.<slot>`), every tensor in
its natural shape; the dense blobs are shared by every rank, the expert blobs
are per rank. The runner mmaps whichever files a manifest needs.

Slot layout is pegainfer's certified weight plan (pegainfer-k3/src/model/
plan.rs), so the engine that gates this export and the kernels that consume
it agree on every stacking and transform:
  KDA   wbig [4*12288, 7168] = q|k|v|g_proj; wsm [256, 7168] = b_proj(96) |
        f_a_proj(128) | zero pad; w_f_b [12288, 128]; cw_q/k/v f32 [4, 12288]
        (conv1d weight [12288, 1, 4] transposed) and cw = the three stacked [3, 4, 12288]; dt_bias f32 [12288]; a_log
        f32 [96] (the checkpoint pads it to 128 lanes); gamma_o f32 [128];
        w_o [7168, 12288]
  MLA   wfu [14400, 7168] = q_a_proj(1536) | kv_a_proj_with_mqa(576) |
        g_proj(12288); gamma_q_a [1536]; gamma_kv_a [512]; w_q_b [18432, 1536];
        w_kv_b [24576, 512]; w_o [7168, 12288]; scale bf16 [1] = 192^-0.5
  MoE   w_router [224, 7168]; bias f32 [224]; rs bf16 [1] = 1; w_lat_down
        [3584, 7168]; w_lat_up [7168, 3584]; gamma_lat [3584]; wsh [12288, 7168]
        = shared gate|up; sh_down [7168, 6144]
  dense wgu [67584, 7168] = gate|up; w_dn [7168, 33792]
  every layer: gamma_in, gamma_post [7168] bf16; sw_attn, sw_mlp f32 [7168] =
        f32(res_norm.weight) * f32(res_proj.weight) (the folded scoring vector)
Needs torch + safetensors; CPU only.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from export_k3_moe import interleave_src_row, pack_sf  # noqa: E402

CKPT = "/mnt/shared/weights/kimi-k3-pruned-75pct"
LAYERS = 93
HIDDEN = 7168
HEADS, HEAD_DIM = 96, 128
INNER = HEADS * HEAD_DIM
EXPERTS, INTER, LATENT = 224, 3072, 3584
MLA_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92}


def is_mla(i):
    return i in MLA_LAYERS


class Ckpt:
    def __init__(self, root):
        self.root = root
        self.index = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.open = {}

    def get(self, name):
        name = "language_model." + name
        f = self.index[name]
        if f not in self.open:
            self.open[f] = safe_open(os.path.join(self.root, f), framework="pt")
        return self.open[f].get_tensor(name)


def bf16(t):
    assert t.dtype == torch.bfloat16, t.dtype
    return t.contiguous()


def f32(t):
    return t.float().contiguous()


def scoring(ck, prefix):
    norm = ck.get(prefix + "norm.weight").float()
    proj = ck.get(prefix + "proj.weight").float().reshape(-1)
    assert norm.shape == proj.shape == (HIDDEN,)
    return (norm * proj).contiguous()


def dense_layer(ck, i):
    p = f"model.layers.{i}."
    a = p + "self_attn."
    out = {
        "gamma_in": bf16(ck.get(p + "input_layernorm.weight")),
        "gamma_post": bf16(ck.get(p + "post_attention_layernorm.weight")),
        "sw_attn": scoring(ck, p + "self_attention_res_"),
        "sw_mlp": scoring(ck, p + "mlp_res_"),
    }
    if is_mla(i):
        out["wfu"] = bf16(torch.cat([ck.get(a + "q_a_proj.weight"), ck.get(a + "kv_a_proj_with_mqa.weight"),
                                     ck.get(a + "g_proj.weight")]))
        assert out["wfu"].shape == (1536 + 576 + INNER, HIDDEN)
        out["gamma_q_a"] = bf16(ck.get(a + "q_a_layernorm.weight"))
        out["gamma_kv_a"] = bf16(ck.get(a + "kv_a_layernorm.weight"))
        out["w_q_b"] = bf16(ck.get(a + "q_b_proj.weight"))
        out["w_kv_b"] = bf16(ck.get(a + "kv_b_proj.weight"))
        out["w_o"] = bf16(ck.get(a + "o_proj.weight"))
        out["scale"] = torch.tensor([192.0 ** -0.5], dtype=torch.bfloat16)
        assert out["w_q_b"].shape == (HEADS * 192, 1536) and out["w_kv_b"].shape == (HEADS * 256, 512)
    else:
        out["wbig"] = bf16(torch.cat([ck.get(a + n + "_proj.weight") for n in ("q", "k", "v", "g")]))
        assert out["wbig"].shape == (4 * INNER, HIDDEN)
        wsm = torch.zeros(256, HIDDEN, dtype=torch.bfloat16)
        wsm[:HEADS] = ck.get(a + "b_proj.weight")
        wsm[HEADS:HEADS + HEAD_DIM] = ck.get(a + "f_a_proj.weight")
        out["wsm"] = wsm
        out["w_f_b"] = bf16(ck.get(a + "f_b_proj.weight"))
        assert out["w_f_b"].shape == (INNER, HEAD_DIM)
        for s in "qkv":
            w = ck.get(a + f"{s}_conv1d.weight").float()
            assert w.shape == (INNER, 1, 4)
            out[f"cw_{s}"] = w.reshape(INNER, 4).t().contiguous()
        # kern's own conv_silu takes the three streams' taps as one [3][4][INNER] block
        out["cw"] = torch.stack([out[f"cw_{s}"] for s in "qkv"]).contiguous()
        out["dt_bias"] = f32(ck.get(a + "dt_bias"))
        out["a_log"] = f32(ck.get(a + "A_log")[:HEADS])
        out["gamma_o"] = f32(ck.get(a + "o_norm.weight"))
        out["w_o"] = bf16(ck.get(a + "o_proj.weight"))
        assert out["dt_bias"].shape == (INNER,) and out["gamma_o"].shape == (HEAD_DIM,)
    if i == 0:
        m = p + "mlp."
        out["wgu"] = bf16(torch.cat([ck.get(m + "gate_proj.weight"), ck.get(m + "up_proj.weight")]))
        out["w_dn"] = bf16(ck.get(m + "down_proj.weight"))
        assert out["wgu"].shape == (2 * 33792, HIDDEN) and out["w_dn"].shape == (HIDDEN, 33792)
    else:
        m = p + "block_sparse_moe."
        out["w_router"] = bf16(ck.get(m + "gate.weight"))
        out["bias"] = f32(ck.get(m + "gate.e_score_correction_bias"))
        out["rs"] = torch.tensor([1.0], dtype=torch.bfloat16)
        out["w_lat_down"] = bf16(ck.get(m + "routed_expert_down_proj.weight"))
        out["w_lat_up"] = bf16(ck.get(m + "routed_expert_up_proj.weight"))
        out["gamma_lat"] = bf16(ck.get(m + "routed_expert_norm.weight"))
        out["wsh"] = bf16(torch.cat([ck.get(m + "shared_experts.gate_proj.weight"),
                                     ck.get(m + "shared_experts.up_proj.weight")]))
        out["sh_down"] = bf16(ck.get(m + "shared_experts.down_proj.weight"))
        assert out["w_router"].shape == (EXPERTS, HIDDEN) and out["wsh"].shape == (4 * INTER, HIDDEN)
    return {f"layers.{i}.{k}": v for k, v in out.items()}


SRC13 = np.array([interleave_src_row(int(r), INTER) for r in range(2 * INTER)])


def expert_layer(ck, i, lo, hi):
    """Experts [lo, hi) of layer i in MegaMoE form (export_k3_moe's contract)."""
    n = hi - lo
    l1w = np.zeros((n, 2 * INTER, LATENT // 2), np.uint8)
    l1s = np.zeros((n, LATENT // 128, 2 * INTER), np.int32)
    l2w = np.zeros((n, LATENT, INTER // 2), np.uint8)
    l2s = np.zeros((n, INTER // 128, LATENT), np.int32)
    prefix = f"model.layers.{i}.block_sparse_moe.experts."
    for j, e in enumerate(range(lo, hi)):
        w1, w3, w2 = (ck.get(f"{prefix}{e}.{n_}.weight_packed").numpy() for n_ in ("w1", "w3", "w2"))
        s1, s3, s2 = (ck.get(f"{prefix}{e}.{n_}.weight_scale").numpy() for n_ in ("w1", "w3", "w2"))
        w13 = np.concatenate([w1, w3], axis=0)
        s13 = np.concatenate([s1, s3], axis=0)
        l1w[j] = w13[SRC13]
        l1s[j] = pack_sf(s13, interleave=True)
        l2w[j] = w2
        l2s[j] = pack_sf(s2, interleave=False)
    p = f"layers.{i}."
    return {p + "l1_weights": torch.from_numpy(l1w), p + "l1_weights_sf": torch.from_numpy(l1s),
            p + "l2_weights": torch.from_numpy(l2w), p + "l2_weights_sf": torch.from_numpy(l2s)}


def write(path, tensors):
    tmp = path.with_suffix(".tmp")
    save_file(tensors, str(tmp))
    os.replace(tmp, path)
    mb = sum(t.numel() * t.element_size() for t in tensors.values()) / 2**20
    print(f"  {path.name}: {len(tensors)} tensors, {mb:.0f} MiB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--ranks", default="1,4", help="comma-separated EP world sizes to shard experts for")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--no-experts", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    (out / "dense").mkdir(parents=True, exist_ok=True)
    (out / "experts").mkdir(parents=True, exist_ok=True)
    ck = Ckpt(a.ckpt)
    t0 = time.time()
    if not a.no_dense:
        f = out / "dense" / "bookends.safetensors"
        if not f.exists():
            write(f, {
                "embed": bf16(ck.get("model.embed_tokens.weight")),
                "gamma_final": bf16(ck.get("model.norm.weight")),
                "sw_out": scoring(ck, "model.output_attn_res_"),
                "w_lm": bf16(ck.get("lm_head.weight")),
            })
        for i in range(a.layers):
            f = out / "dense" / f"l{i}.safetensors"
            if not f.exists():
                write(f, dense_layer(ck, i))
    if not a.no_experts:
        for i in range(1, a.layers):
            for r_n in (int(x) for x in a.ranks.split(",")):
                per = EXPERTS // r_n
                for r in range(r_n):
                    f = out / "experts" / f"ep{r_n}-r{r}-l{i}.safetensors"
                    if not f.exists():
                        write(f, expert_layer(ck, i, r * per, (r + 1) * per))
            print(f"layer {i} experts done ({time.time() - t0:.0f}s)", flush=True)
    print(f"exported {a.layers} layers to {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
