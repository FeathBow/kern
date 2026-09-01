#!/usr/bin/env python3
"""DFlash2 draft checkpoint (incoai/Qwen3.8-27B-DFlash2) -> kern weight
artifact, plus the constant tables the speculative manifest needs.  Loaded
next to the target artifact (`kern run --weights target.safetensors
--weights draft.safetensors`); tensor name = manifest buffer name.

- `draft.fc.{j}.weight` [5120, 5120]: column block j of the 5-tap combiner
  (vLLM: one [5120, 25600] GEMM over the concatenated taps; kern: five
  accumulating GEMMs at the tap sites, no concat).
- `draft.layers.{l}.self_attn.qkv_proj.weight` = cat(q, k, v) [6144, 5120];
  `mlp.gate_up_proj.weight` = cat(gate, up) [34816, 5120]; norms as-is
  (plain RMSNorm, vLLM's CUDA kernel takes the bf16 weight).
- `draft.layers.{l}.{attention_conv,mlp_conv}.base_kernel` [4, 5120]
  (= (side, tap) rows) and `.kernel_projection.weight` [1280, 5120].
- `draft.fused_kv.weight` [10240, 5120] = per layer [k_proj; v_proj] rows
  (vLLM's `_build_fused_kv_buffers`, materialised here).
- `draft.selector.*`: hidden_projection [256, 5120], predecessor /
  successor codebooks [248320, 256].
- `draft.rope.cos_sin_cache` [MAX_POS, 128] bf16 from vLLM's own
  RotaryEmbedding (theta 1e7, neox) — vLLM's rotary_embedding_kernel reads
  this table.
- `draft.kv_scales` f32 [10] = 1.
- GDN speculative state tables: `gdn.spec_line_index` (GDN layer g -> its
  page 8g+1; page 0 is null), `gdn.spec_slots` [48, 8] (the 8 per-token
  SSM checkpoint pages of layer g), `gdn.one` (num_accepted_tokens = 1 for
  the non-speculative programs).

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/export_qwen38_draft.py \
        [draft snapshot dir] [weights/qwen3.8-27b]
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
                   f"{HUB}/models--incoai--Qwen3.8-27B-DFlash2/snapshots/"
                   "dedf8df68adfb1afeaf7b7480c0a0243108177b4")
out_dir = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("KERN_WEIGHTS", "weights") + "/qwen3.8-27b")

MAX_POS = 262144        # the target's max_position_embeddings (= the manifest's MAX_POS)
GDN_LAYERS = 48
SPEC_SLOTS = 8

cfg = json.loads((src / "config.json").read_text())
L = cfg["num_hidden_layers"]
H = cfg["hidden_size"]
HD = cfg["head_dim"]
assert cfg["dflash_config"]["target_layer_ids"] == [5, 19, 33, 47, 61]
assert cfg["dflash_config"]["block_size"] == SPEC_SLOTS

t0 = time.time()
out = {}
with safe_open(str(src / "model.safetensors"), "pt") as f:
    g = lambda k: f.get_tensor(k)  # noqa: E731
    fc = g("fc.weight")
    assert fc.shape == (H, 5 * H)
    for j in range(5):
        out[f"draft.fc.{j}.weight"] = fc[:, j * H:(j + 1) * H].contiguous()
    out["draft.hidden_norm.weight"] = g("hidden_norm.weight")
    out["draft.norm.weight"] = g("norm.weight")
    kv_rows = []
    for l in range(L):
        p = f"layers.{l}."
        q, k, v = g(p + "self_attn.q_proj.weight"), g(p + "self_attn.k_proj.weight"), g(p + "self_attn.v_proj.weight")
        out[f"draft.{p}self_attn.qkv_proj.weight"] = torch.cat([q, k, v], 0).contiguous()
        kv_rows += [k, v]
        for n in ("input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight",
                  "self_attn.k_norm.weight", "self_attn.o_proj.weight", "mlp.down_proj.weight",
                  "attention_conv.kernel_projection.weight", "mlp_conv.kernel_projection.weight"):
            out[f"draft.{p}{n}"] = g(p + n).contiguous()
        out[f"draft.{p}mlp.gate_up_proj.weight"] = torch.cat(
            [g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")], 0).contiguous()
        for c in ("attention_conv", "mlp_conv"):
            bk = g(f"{p}{c}.base_kernel")
            assert bk.shape == (2, 2, H)
            out[f"draft.{p}{c}.base_kernel"] = bk.reshape(4, H).contiguous()
    out["draft.fused_kv.weight"] = torch.cat(kv_rows, 0).contiguous()
    out["draft.selector.hidden_projection.weight"] = g("candidate_selector.hidden_projection.weight")
    out["draft.selector.predecessor"] = g("candidate_selector.predecessor_codebook")
    out["draft.selector.successor"] = g("candidate_selector.successor_codebook")

# rope table from vLLM's implementation (the draft's rotary_embedding_kernel reads it)
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.rotary_embedding import get_rope  # noqa: E402

with torch.device("cuda"), set_current_vllm_config(VllmConfig()):
    rope = get_rope(HD, max_position=MAX_POS, rope_parameters=cfg["rope_parameters"],
                    is_neox_style=True, dtype=torch.bfloat16)
    cache = rope.cos_sin_cache
assert cache.shape == (MAX_POS, HD), cache.shape
out["draft.rope.cos_sin_cache"] = cache.to(torch.bfloat16).cpu().contiguous()
out["draft.kv_scales"] = torch.ones(2 * L, dtype=torch.float32)

# GDN speculative state tables
li = torch.zeros(64, dtype=torch.int32)
for gi in range(GDN_LAYERS):
    li[gi + 1] = SPEC_SLOTS * gi + 1
out["gdn.spec_line_index"] = li
out["gdn.spec_slots"] = (torch.arange(GDN_LAYERS, dtype=torch.int32)[:, None] * SPEC_SLOTS
                         + torch.arange(1, SPEC_SLOTS + 1, dtype=torch.int32)[None, :]).contiguous()
out["gdn.one"] = torch.ones(1, dtype=torch.int32)

out_dir.mkdir(parents=True, exist_ok=True)
dst = out_dir / "qwen3.8-27b-dflash2-draft.safetensors"
save_file({k: v.contiguous() for k, v in out.items()}, str(dst))
nbytes = sum(v.numel() * v.element_size() for v in out.values())
print(f"wrote {dst}: {len(out)} tensors, {nbytes / 2**30:.2f} GiB in {time.time() - t0:.1f}s")
