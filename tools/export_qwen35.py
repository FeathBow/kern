#!/usr/bin/env python3
"""Qwen3.5 (Qwen3.8-27B) HF checkpoint -> kern weight artifact.

One safetensors file, tensor name = manifest buffer name (weight class):

- `model.layers.{i}.*` (the `model.language_model.` prefix of the VL
  checkpoint is dropped; `model.visual.*` and `mtp.*` are not exported).
- GDN layers: `linear_attn.in_proj_qkvz.weight` = cat(in_proj_qkv, in_proj_z)
  rows (columns of the projection: q 2048 | k 2048 | v 6144 | z 6144 — the
  order vLLM's MergedColumnParallelLinear produces for Qwen3.5, which has no
  gqa-interleaved reordering); `in_proj_ba.weight` = cat(in_proj_b,
  in_proj_a) (b 48 | a 48); `conv1d.weight` [10240, 4] (the [dim, 1, width]
  Conv1d weight squeezed, the view vLLM hands the Triton conv kernels);
  `A_log` as f32 (vLLM keeps the parameter in f32; the Triton kernels take
  an f32 pointer); `dt_bias` and the gated-norm `norm.weight` stay bf16.
- Attention layers: `self_attn.qkv_proj.weight` = cat(q_proj, k_proj, v_proj)
  (q_proj rows are per head [q 256 | gate 256] — attn_output_gate).
- Gemma-style norms (`input_layernorm`, `post_attention_layernorm`,
  `q_norm`, `k_norm`, `model.norm`) are exported as `*.weight_p1`: f32
  `weight.float() + 1.0`, exactly the tensor vLLM's GemmaRMSNorm feeds to
  the ATen ops (so the f32 add is done once here, not per token).
- `rope.cos` / `rope.sin` [MAX_POS, 32] bf16: taken from vLLM's own
  MRotaryEmbedding instantiated on the GPU (same code path, same device
  pow/cos/sin implementations), then split out of its [pos, cos|sin] cache.
- Constant tables the mined Triton kernels want as pointers: FLA chunk
  indices / offsets (one sequence: (0, i) pairs), causal_conv1d's batch /
  token-chunk-offset tables (one sequence, up to 2048/8 programs), the GDN
  state line index table (line i+1 for GDN layer i, line 0 is the kernels'
  null block) and the has_initial_state flag (always true: kern's state is
  zero-initialised, and a zero initial state is what "no initial state"
  computes with).
- `kv_scales` f32 [2] = 1 (bf16 KV, the kernels still take the pointers).

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/export_qwen35.py \
        [HF snapshot dir] [weights/qwen3.8-27b]
"""
import os
import json
import pathlib
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HUB = os.path.expanduser("~/.cache/huggingface/hub")
src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                   f"{HUB}/models--Qwen--Qwen3.8-27B/snapshots/"
                   "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
out_dir = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else
                       os.environ.get("KERN_WEIGHTS", "weights") + "/qwen3.8-27b")
out_dir.mkdir(parents=True, exist_ok=True)

CHUNK_MAX = 2048        # prefill chunk bound (tokens symbol max)
FLA_CHUNK = 64
CONV_BLOCK_M = 8

cfg = json.loads((src / "config.json").read_text())["text_config"]
MAX_POS = cfg["max_position_embeddings"]   # rope table rows (= the manifest's MAX_POS); KV capacity is a runtime parameter
LAYERS = cfg["num_hidden_layers"]
assert cfg["layer_types"] and len(cfg["layer_types"]) == LAYERS
ATTN = [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]
GDN = [i for i, t in enumerate(cfg["layer_types"]) if t == "linear_attention"]
assert len(ATTN) + len(GDN) == LAYERS, cfg["layer_types"]
HEAD_DIM = cfg["head_dim"]
ROT = int(HEAD_DIM * cfg["rope_parameters"]["partial_rotary_factor"])
assert not cfg.get("tie_word_embeddings", False)
conv_dim = cfg["linear_key_head_dim"] * cfg["linear_num_key_heads"] * 2 \
    + cfg["linear_value_head_dim"] * cfg["linear_num_value_heads"]

t0 = time.time()
idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
want = [k for k in idx if k.startswith("model.language_model.") or k == "lm_head.weight"]
by_file = {}
for k in want:
    by_file.setdefault(idx[k], []).append(k)
tensors = {}
for f, keys in sorted(by_file.items()):
    with safe_open(str(src / f), framework="pt") as st:
        for k in keys:
            tensors[k] = st.get_tensor(k)
print(f"loaded {len(tensors)} text-model tensors from {len(by_file)} shards "
      f"in {time.time() - t0:.0f}s (skipped {len(idx) - len(want)} visual/mtp)")


def take(name):
    return tensors.pop("model.language_model." + name if not name.startswith("lm_head") else name)


def p1(w):
    return (w.float() + 1.0).contiguous()


out = {}
out["model.embed_tokens.weight"] = take("embed_tokens.weight")
out["lm_head.weight"] = take("lm_head.weight")
out["model.norm.weight_p1"] = p1(take("norm.weight"))

for i in range(LAYERS):
    p, q = f"layers.{i}.", f"model.layers.{i}."
    out[q + "input_layernorm.weight_p1"] = p1(take(p + "input_layernorm.weight"))
    out[q + "post_attention_layernorm.weight_p1"] = p1(take(p + "post_attention_layernorm.weight"))
    out[q + "mlp.gate_up_proj.weight"] = torch.cat(
        [take(p + "mlp.gate_proj.weight"), take(p + "mlp.up_proj.weight")], dim=0)
    out[q + "mlp.down_proj.weight"] = take(p + "mlp.down_proj.weight")
    if i in ATTN:
        a = p + "self_attn."
        out[q + "self_attn.qkv_proj.weight"] = torch.cat(
            [take(a + "q_proj.weight"), take(a + "k_proj.weight"), take(a + "v_proj.weight")], dim=0)
        out[q + "self_attn.q_norm.weight_p1"] = p1(take(a + "q_norm.weight"))
        out[q + "self_attn.k_norm.weight_p1"] = p1(take(a + "k_norm.weight"))
        out[q + "self_attn.o_proj.weight"] = take(a + "o_proj.weight")
    else:
        g = p + "linear_attn."
        out[q + "linear_attn.in_proj_qkvz.weight"] = torch.cat(
            [take(g + "in_proj_qkv.weight"), take(g + "in_proj_z.weight")], dim=0)
        out[q + "linear_attn.in_proj_ba.weight"] = torch.cat(
            [take(g + "in_proj_b.weight"), take(g + "in_proj_a.weight")], dim=0)
        w = take(g + "conv1d.weight")
        assert w.shape == (conv_dim, 1, cfg["linear_conv_kernel_dim"]), w.shape
        out[q + "linear_attn.conv1d.weight"] = w.reshape(conv_dim, -1)
        out[q + "linear_attn.A_log"] = take(g + "A_log").float()
        out[q + "linear_attn.dt_bias"] = take(g + "dt_bias")
        out[q + "linear_attn.norm.weight"] = take(g + "norm.weight")
        out[q + "linear_attn.out_proj.weight"] = take(g + "out_proj.weight")
assert not tensors, f"unconsumed tensors: {sorted(tensors)[:8]}"

# rope tables: vLLM's own class on the GPU, sliced.
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding  # noqa: E402
rp = cfg["rope_parameters"]
with torch.device("cuda"), set_current_vllm_config(VllmConfig()):
    rope = MRotaryEmbedding(
        head_size=HEAD_DIM, rotary_dim=ROT,
        max_position_embeddings=cfg["max_position_embeddings"],
        base=float(rp["rope_theta"]), is_neox_style=True, dtype=torch.bfloat16,
        mrope_section=rp["mrope_section"], mrope_interleaved=rp["mrope_interleaved"])
cache = rope.cos_sin_cache
assert cache.shape[1] == ROT and cache.dtype == torch.bfloat16, (cache.shape, cache.dtype)
out["rope.cos"] = cache[:MAX_POS, :ROT // 2].contiguous().cpu()
out["rope.sin"] = cache[:MAX_POS, ROT // 2:].contiguous().cpu()

out["kv_scales"] = torch.ones(2, dtype=torch.float32)
nt = CHUNK_MAX // FLA_CHUNK
out["fla.chunk_indices"] = torch.stack(
    [torch.zeros(nt, dtype=torch.int32), torch.arange(nt, dtype=torch.int32)], dim=1).contiguous()
out["fla.chunk_offsets"] = torch.zeros(2, dtype=torch.int64)
nprog = CHUNK_MAX // CONV_BLOCK_M
out["conv.batch_ptr"] = torch.zeros(nprog, dtype=torch.int32)
out["conv.token_chunk_offset"] = torch.arange(nprog, dtype=torch.int32)
out["gdn.line_index"] = torch.arange(64, dtype=torch.int32)
out["gdn.has_initial"] = torch.ones(16, dtype=torch.uint8)

out = {k: v.contiguous() for k, v in out.items()}
dst = out_dir / "qwen3.8-27b.safetensors"
t0 = time.time()
save_file(out, str(dst))
for f in ["tokenizer.json", "tokenizer_config.json"]:
    (out_dir / f).write_bytes((src / f).read_bytes())
print(f"wrote {dst} ({dst.stat().st_size >> 30} GiB, {len(out)} tensors) in "
      f"{time.time() - t0:.0f}s + tokenizer files; attn layers {ATTN}")
