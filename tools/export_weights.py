#!/usr/bin/env python3
"""HF Qwen3-4B checkpoint -> kern decode manifest 的权重文件。

输出一个 safetensors，tensor 名 = manifest buffer 名（weight class）：
- qkv / gate_up 合并（cat 行，q@0 / k@4096 / v@5120，gate@0 / up@9728，
  与挖矿实测的视图偏移一致）
- rope.cos_sin_cache 预计算（vLLM 布局 [max_pos, rot_dim]，前半 cos 后半 sin）
- kv_scales 全 1（bf16 KV 不量化，kernel ABI 仍要求指针）
- lm_head：Qwen3-4B tie_word_embeddings，复用 embed 矩阵

另外若 DSpark draft checkpoint 在（默认 weights/dspark_qwen3_4b_block7），
连带导出 weights/qwen3-4b-dspark.safetensors（target + draft 合一，配
examples/qwen3-4b-dspark.json）：
- draft 同样做 qkv / gate_up 合并；fused_kv = 逐层 cat qkv[q_size:]（即
  [k0;v0;k1;v1;...]，输出行内布局与 manifest 的列偏移一致）
- fc [2560,12800] 按列切成 5 块 draft.fc.{j}.weight [2560,2560]（tap 点
  各自累加，等价一次 concat GEMM）
- markov_w1 是 embedding 表、markov_w2 是无 bias 的 LMHead 权重，原样带走
- confidence head 丢弃（固定 k 验证不用自适应）

跑法：.venv/bin/python tools/export_weights.py [HF目录] [输出目录] [draft目录]
"""

import json
import pathlib
import sys

import torch
from safetensors.torch import safe_open, save_file

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/shared/weights/Qwen3-4B")
out_dir = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else
                       pathlib.Path(__file__).resolve().parent.parent / "weights")
draft_src = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else
                         pathlib.Path(__file__).resolve().parent.parent
                         / "weights" / "dspark_qwen3_4b_block7")
out_dir.mkdir(exist_ok=True)

cfg = json.loads((src / "config.json").read_text())
LAYERS = cfg["num_hidden_layers"]
HEAD_DIM = cfg["head_dim"]
MAX_POS = 4096  # 与 manifest MAX_POS 一致
assert cfg.get("tie_word_embeddings", False), "非 tied lm_head：需另外导出"

tensors = {}
for f in sorted(src.glob("*.safetensors")):
    with safe_open(f, framework="pt") as st:
        for name in st.keys():
            tensors[name] = st.get_tensor(name)
print(f"loaded {len(tensors)} tensors from {src}")

out = {}


def take(name):
    return tensors.pop(name)


out["model.embed_tokens.weight"] = take("model.embed_tokens.weight")
out["model.norm.weight"] = take("model.norm.weight")
out["lm_head.weight"] = out["model.embed_tokens.weight"]  # tied

for i in range(LAYERS):
    p = f"model.layers.{i}."
    out[p + "input_layernorm.weight"] = take(p + "input_layernorm.weight")
    out[p + "post_attention_layernorm.weight"] = take(p + "post_attention_layernorm.weight")
    out[p + "self_attn.q_norm.weight"] = take(p + "self_attn.q_norm.weight")
    out[p + "self_attn.k_norm.weight"] = take(p + "self_attn.k_norm.weight")
    out[p + "self_attn.o_proj.weight"] = take(p + "self_attn.o_proj.weight")
    out[p + "mlp.down_proj.weight"] = take(p + "mlp.down_proj.weight")
    out[p + "self_attn.qkv_proj.weight"] = torch.cat(
        [take(p + "self_attn.q_proj.weight"),
         take(p + "self_attn.k_proj.weight"),
         take(p + "self_attn.v_proj.weight")], dim=0)
    out[p + "mlp.gate_up_proj.weight"] = torch.cat(
        [take(p + "mlp.gate_proj.weight"), take(p + "mlp.up_proj.weight")], dim=0)

# rope cos_sin_cache：vLLM RotaryEmbedding 的布局与数值（bf16）
base = float(cfg["rope_theta"])
inv_freq = 1.0 / (base ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
t = torch.arange(MAX_POS, dtype=torch.float32)
freqs = torch.outer(t, inv_freq)
out["rope.cos_sin_cache"] = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)

out["kv_scales"] = torch.ones(2 * LAYERS, dtype=torch.float32)

assert not tensors, f"剩余未消费的权重: {sorted(tensors)[:5]}"
out = {k: v.contiguous() for k, v in out.items()}
dst = out_dir / "qwen3-4b-decode.safetensors"
# tied lm_head 共享 storage，safetensors 拒绝共享——clone 一份
out["lm_head.weight"] = out["lm_head.weight"].clone()
save_file(out, str(dst))

for f in ["tokenizer.json", "tokenizer_config.json"]:
    (out_dir / f).write_bytes((src / f).read_bytes())
print(f"wrote {dst} ({dst.stat().st_size >> 20} MiB) + tokenizer files")

if not draft_src.exists():
    sys.exit(0)

dcfg = json.loads((draft_src / "config.json").read_text())
DL = dcfg["num_hidden_layers"]
assert dcfg["hidden_size"] == 2560 and dcfg["head_dim"] == HEAD_DIM

dt = {}
for f in sorted(draft_src.glob("*.safetensors")):
    with safe_open(f, framework="pt") as st:
        for name in st.keys():
            dt[name] = st.get_tensor(name)
print(f"loaded {len(dt)} draft tensors from {draft_src}")

d = {}
d["draft.embed_tokens.weight"] = dt.pop("embed_tokens.weight")
d["draft.lm_head.weight"] = dt.pop("lm_head.weight")
d["draft.norm.weight"] = dt.pop("norm.weight")
d["draft.hidden_norm.weight"] = dt.pop("hidden_norm.weight")
d["draft.markov_w1"] = dt.pop("markov_head.markov_w1.weight")
d["draft.markov_w2.weight"] = dt.pop("markov_head.markov_w2.weight")
d["draft.kv_scales"] = torch.ones(2 * DL, dtype=torch.float32)

# fc [hidden, L*hidden] 按列切块：tap j 的累加 GEMM 用第 j 块
fc = dt.pop("fc.weight")
assert fc.shape == (2560, 2560 * DL), fc.shape
for j in range(DL):
    d[f"draft.fc.{j}.weight"] = fc[:, j * 2560:(j + 1) * 2560]

kv_blocks = []
for i in range(DL):
    p, q = f"layers.{i}.", f"draft.layers.{i}."
    d[q + "input_layernorm.weight"] = dt.pop(p + "input_layernorm.weight")
    d[q + "post_attention_layernorm.weight"] = dt.pop(p + "post_attention_layernorm.weight")
    d[q + "self_attn.q_norm.weight"] = dt.pop(p + "self_attn.q_norm.weight")
    d[q + "self_attn.k_norm.weight"] = dt.pop(p + "self_attn.k_norm.weight")
    d[q + "self_attn.o_proj.weight"] = dt.pop(p + "self_attn.o_proj.weight")
    d[q + "mlp.down_proj.weight"] = dt.pop(p + "mlp.down_proj.weight")
    k = dt.pop(p + "self_attn.k_proj.weight")
    v = dt.pop(p + "self_attn.v_proj.weight")
    d[q + "self_attn.qkv_proj.weight"] = torch.cat(
        [dt.pop(p + "self_attn.q_proj.weight"), k, v], dim=0)
    kv_blocks += [k, v]  # 融合 KV：行序 [k0;v0;k1;v1;...]
    d[q + "mlp.gate_up_proj.weight"] = torch.cat(
        [dt.pop(p + "mlp.gate_proj.weight"), dt.pop(p + "mlp.up_proj.weight")],
        dim=0)
d["draft.fused_kv.weight"] = torch.cat(kv_blocks, dim=0)

leftovers = sorted(dt)
assert all(n.startswith("confidence_head.") for n in leftovers), leftovers

combined = {k: v.contiguous() for k, v in {**out, **d}.items()}
dst2 = out_dir / "qwen3-4b-dspark.safetensors"
save_file(combined, str(dst2))
print(f"wrote {dst2} ({dst2.stat().st_size >> 20} MiB, "
      f"{len(d)} draft tensors; dropped {leftovers})")
