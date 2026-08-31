#!/usr/bin/env python3
"""Tests for the three handwritten DFlash2 kernels against the ATen chains
they replace (qwen3_dflash2.py in vLLM's model zoo):

- kern_dflash_conv_bf16   vs DFlashGroupedConv.prepare / finish
- kern_topk16_bf16        vs torch.topk(16) (set equality + values)
- kern_dflash_select      vs the selector's bf16 bilinear scoring + greedy walk

Launched through the CUDA driver API the way kern-runtime launches them.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/test_kernels_dflash2.py <cubin_dir>
"""

import ctypes
import pathlib
import sys

import torch
from cuda.bindings import driver as cu

C_PTR, C_INT = ctypes.c_void_p, ctypes.c_int
BF = torch.bfloat16


def check(r):
    err = r[0]
    assert err == cu.CUresult.CUDA_SUCCESS, err
    return r[1] if len(r) == 2 else r[1:]


def launch(fn, grid, block, smem, args):
    grid = tuple(grid) + (1,) * (3 - len(grid))
    block = tuple(block) + (1,) * (3 - len(block))
    vals = tuple(a for a, _ in args)
    types = tuple(t for _, t in args)
    stream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
    check(cu.cuLaunchKernel(fn, *grid, *block, smem, stream, (vals, types), 0))
    torch.cuda.synchronize()


def ptr(t):
    return (t.data_ptr(), C_PTR)


def i32(v):
    return (v, C_INT)


def same(name, a, b):
    ok = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if not ok:
        diff = (a != b).sum().item() if a.shape == b.shape else -1
        print(f"  FAIL {name}: {diff} of {a.numel()} elements differ")
    return ok


# --- references (the ATen chains, in bf16 like vLLM runs them)
def ref_conv(x, delta, base, side, groups):
    T, D = x.shape
    gsz = D // groups
    coef = base.view(2, 2, D)[side][None] + delta.view(T, 2, 2, groups)[:, side].repeat_interleave(gsz, -1)
    prev = torch.cat([torch.zeros_like(x[:1]), x[:-1]], 0)
    mask = (torch.arange(T, device=x.device) % 8 != 0).to(BF)[:, None]
    return coef[:, 0] * x + coef[:, 1] * prev * mask


def ref_select(cand, unary, hidden_r, succ_cb, pred_cb, anchor):
    steps = cand.shape[0]
    out, prev = [], anchor
    for l in range(steps):
        pred = pred_cb[prev]                                # [rank]
        ph = (pred * hidden_r[l]).to(BF)                    # bf16 product
        succ = succ_cb[cand[l]]                             # [16, rank]
        edge = (succ.float() @ ph.float()).to(BF).float()   # bf16 matmul result
        s = unary[l] + edge
        best = int(torch.argmax(s).item())                  # first max
        prev = int(cand[l, best].item())
        out.append(prev)
    return torch.tensor(out, dtype=torch.int64, device=cand.device)


def main():
    cubins = pathlib.Path(sys.argv[1])
    torch.zeros(1, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(4321)
    mods = {k: check(cu.cuModuleLoad(str(cubins / f"{k}.cubin").encode()))
            for k in ("dflash_conv", "topk_row", "dflash_select")}
    fn = lambda m, s: check(cu.cuModuleGetFunction(mods[m], s.encode()))  # noqa: E731
    k_conv = fn("dflash_conv", "kern_dflash_conv_bf16")
    k_topk = fn("topk_row", "kern_topk16_bf16")
    k_sel = fn("dflash_select", "kern_dflash_select")
    fails = 0
    D, GROUPS, PROJ = 5120, 320, 1280

    # grouped conv, both sides, T = 8 and T = 16 (block boundary mask)
    for T in (8, 16, 5):
        for side in (0, 1):
            x = (torch.randn(T, D, device="cuda", generator=gen) * 2).to(BF)
            delta = (torch.randn(T, PROJ, device="cuda", generator=gen) * 0.1).to(BF)
            base = (torch.randn(4, D, device="cuda", generator=gen)).to(BF)
            want = ref_conv(x, delta, base, side, GROUPS)
            got = x.clone()
            launch(k_conv, [D // 256], [256], 0,
                   [ptr(got), ptr(delta), ptr(base), i32(T), i32(D), i32(GROUPS), i32(PROJ), i32(side)])
            fails += not same(f"dflash_conv T={T} side={side}", got, want)

    # top-16: the id set and values must match torch.topk (tie order aside)
    V = 248320
    for rows in (7, 1):
        logits = (torch.randn(rows, V, device="cuda", generator=gen) * 4).to(BF)
        logits[0, 12345] = logits[0].max() + 1  # a clear winner
        ids = torch.empty(rows, 16, dtype=torch.int64, device="cuda")
        vals = torch.empty(rows, 16, dtype=torch.float32, device="cuda")
        launch(k_topk, [rows], [1024], 0, [ptr(logits), ptr(ids), ptr(vals), i32(V), i32(V)])
        tv, ti = torch.topk(logits.float(), 16, dim=-1)
        ok = torch.equal(vals, tv) and all(set(ids[r].tolist()) == set(ti[r].tolist()) for r in range(rows))
        ok = ok and torch.equal(logits.gather(1, ids).float(), vals)
        ok = ok and all(ids[r].tolist() == sorted(ids[r].tolist(), key=lambda i, r=r: (-logits[r, i].float().item(), i))
                        for r in range(rows))
        if not ok:
            print(f"  FAIL topk16 rows={rows}")
            fails += 1

    # selector walk on random codebooks (rank 256, 7 steps, 16 candidates)
    RANK, STEPS, K = 256, 7, 16
    for trial in range(3):
        cand = torch.randint(0, V, (STEPS, K), device="cuda", generator=gen)
        unary = (torch.randn(STEPS, K, device="cuda", generator=gen) * 3)
        hidden_r = torch.randn(STEPS, RANK, device="cuda", generator=gen).to(BF)
        succ_cb = (torch.randn(V, RANK, device="cuda", generator=gen) * 0.2).to(BF)
        pred_cb = (torch.randn(V, RANK, device="cuda", generator=gen) * 0.2).to(BF)
        anchor = int(torch.randint(0, V, (1,), device="cuda", generator=gen).item())
        want = ref_select(cand, unary, hidden_r, succ_cb, pred_cb, anchor)
        # gathered rows, as the manifest feeds them (embedding kernel)
        succ_g = succ_cb[cand.reshape(-1)].contiguous()
        pred_g = pred_cb[cand.reshape(-1)].contiguous()
        pred_anchor = pred_cb[anchor].contiguous()
        out = torch.empty(STEPS, dtype=torch.int64, device="cuda")
        launch(k_sel, [1], [RANK], K * RANK * 4,
               [ptr(cand), ptr(unary), ptr(hidden_r), ptr(succ_g), ptr(pred_g), ptr(pred_anchor), ptr(out),
                i32(STEPS), i32(RANK), i32(RANK)])
        fails += not same(f"dflash_select trial={trial}", out, want)

    print("all passed" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
