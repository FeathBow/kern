#!/usr/bin/env python3
"""HF Qwen3-4B checkpoint -> kern decode manifest 的权重文件。

输出一个 safetensors，tensor 名 = manifest buffer 名（weight class）：
- qkv / gate_up 合并（cat 行，q@0 / k@4096 / v@5120，gate@0 / up@9728，
  与挖矿实测的视图偏移一致）
- rope.cos_sin_cache 预计算（vLLM 布局 [max_pos, rot_dim]，前半 cos 后半 sin）
- kv_scales 全 1（bf16 KV 不量化，kernel ABI 仍要求指针）
- lm_head：Qwen3-4B tie_word_embeddings，复用 embed 矩阵

跑法：.venv/bin/python tools/export_weights.py [HF目录] [输出目录]
"""

import json
import pathlib
import sys

import torch
from safetensors.torch import safe_open, save_file

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/shared/weights/Qwen3-4B")
out_dir = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else
                       pathlib.Path(__file__).resolve().parent.parent / "weights")
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
