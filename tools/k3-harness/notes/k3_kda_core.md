# K3 `kern_k3_kda_core` — notes

Source `tools/kernels-src/k3_kda_core.cu`, cubin `target/cubins/k3_kda_core.cubin`.
Built on tray03, measured on **tray07 GPU 2** (idle, nothing else on the card).

```
/usr/local/cuda-13.1/bin/nvcc -cubin -arch=sm_103a -O3 -Xptxas -v \
    -o target/cubins/k3_kda_core.cubin tools/kernels-src/k3_kda_core.cu
```

## 1. Results

`tools/k3-harness/harness --kernel kda_core --cubin <cubin> --B <B>` (median of 50,
each launch timed with its own event pair; a range is the spread over repeated runs):

| B  | out | rec(state) | median µs | implied GB/s (harness roofline 818 MB @B=64) |
|----|-----|-----------|-----------|-----------|
| 1  | PASS max\|err\|=1.907e-06 maxULP=36.00 relRMS=3.94e-08 | PASS max\|err\|=1.192e-07 maxULP=2.56 relRMS=7.27e-08 | 8.61–9.86 | 1850 |
| 2  | PASS max\|err\|=9.537e-07 maxULP=1.00 relRMS=1.45e-08 | PASS max\|err\|=1.192e-07 maxULP=9.59 relRMS=7.08e-08 | 10.43 | 2747 |
| 8  | PASS max\|err\|=9.766e-04 maxULP=2.00 relRMS=1.22e-05 | PASS max\|err\|=4.172e-06 maxULP=1564.91 relRMS=1.05e-07 | 18.53–18.59 | 5650 |
| 64 | PASS max\|err\|=3.906e-03 maxULP=136.50 relRMS=2.66e-05 | PASS max\|err\|=4.967e-05 maxULP=226.84 relRMS=5.75e-07 | 125.31–125.38 | 6524 |

(The large `maxULP` figures come from elements whose reference is ~0, where a bf16
ULP is meaningless; the rule that decides PASS is `3·ULP + 1e-3`, and both
`max|err|` and `relRMS` sit far inside it. See §5 on why the state is not bit-tight.)

Back-to-back timing (50 launches under one event pair — closer to what the decode
loop sees, since the harness's per-launch events add ~2–4 µs), from my scratch test
`tools/k3-scratch/k3/test_kda_core.cu`:

| B | µs | rec bytes | GB/s | % of 8 TB/s nominal | % of measured device ceiling |
|---|----|-----------|------|------|------|
| 1 | 5.44 | 12.6 MB | 2312 | 29% | — (latency bound, see §4) |
| 2 | 6.87 | 25.2 MB | 3662 | 46% | — |
| 8 | 15.23 | 100.7 MB | 6608 | 83% | 102% |
| 64 | 122.93 | 805.3 MB | 6551 | 82% | 101% |

**Device ceiling**: `tools/k3-scratch/k3/bw_ref.cu` streams the *same* 402 MB
footprint read-modify-write with a plain grid-stride `float4` loop
(`__ldcs`/`__stcs`, 6144 blocks × 128 threads) and gets **123.89 µs / 6500 GB/s
(81% of 8 TB/s)**. So at B=8 and B=64 this kernel is at the machine's practical
read+write streaming limit; the documented ≥60% target is met with margin.

## 2. ptxas -v

```
ptxas info    : Compiling entry function 'kern_k3_kda_core' for 'sm_103a'
ptxas info    : Function properties for kern_k3_kda_core
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 158 registers, used 1 barriers, 3872 bytes smem
```

`__launch_bounds__(128)`; 158 registers ⇒ 3 blocks/SM (Block Limit Registers = 3),
which is what ncu reports as the occupancy limiter. `cuobjdump -sass`: **0**
`.MULTICAST`, **0** `LDL`/`STL` (no local memory anywhere). The rec traffic shows up
as `16 × LDG.E.EF.128` + `16 × STG.E.EF.128` (EF = evict-first, from `__ldcs`/
`__stcs`) and the `w_f_b` tile as `16 × LDG.E.128.CONSTANT`.

## 3. Design and memory pattern

The block is (row b, head h); the only thing that costs is `rec`, the f32
`[128 dv][128 k]` state tile: 64 KB read + 64 KB written per block, 6.29 MB per row
per layer, 805 MB per layer at B=64.

**Thread mapping.** The obvious TileLang mapping (thread = dv, serial over k) makes
every thread walk its own 512-byte row: 32 threads of a warp then sit 512 B apart,
so a single 16-byte load instruction touches 32 distinct 128-B lines. It also forces
the whole 128-float row to stay live in registers between the `m` reduction and the
write-back (dlt depends on the full row sum), i.e. 128 registers of state per thread.

Instead: `kl = tid & 7` splits **k** across 8 lanes, `dvl = tid >> 3` gives 16 dv
rows per pass, and each lane owns 16 k values as 4 `float4` at
`k = 32j + 4·kl + i`. One warp instruction then covers 4 dv rows × 128 B of
contiguous, 16-B-aligned memory — full 128-B lines, 100% sector utilisation both
ways (ncu: "31.9 of the 32 bytes transmitted per sector are utilized", est. speedup
from fixing coalescing 0.18%). Only 16 k values per lane have to stay live, so
`ROWS_PER_ITER = 4` rows are kept in flight (16 `float4` = 64 registers) for memory
parallelism, and the per-row reduction is 3 `__shfl_xor` steps instead of 5 (a
warp-wide 32-lane split would need 5 steps × 2 values × 128 rows).

**One touch of rec.** `attn[dv] = Σ_k S'[dv,k]·qs[k]` normally needs the state again
*after* `dlt` is known. Expanding it,

```
attn[dv] = Σ_k (S[dv,k]·dec[k])·qs[k] + dlt[dv] · (Σ_k kn[k]·qs[k])
```

the second sum is a block-wide scalar (computed once in the prologue alongside the
l2norm), so the `m` and `attn` reductions ride on the same pass, `t = S·dec` is kept
in the registers already holding the row, and the write-back is `t + dlt·kn`. rec is
read exactly once and written exactly once — no second pass and no reliance on the
re-read hitting L2.

**Streaming hints** (`__ldcs`/`__stcs` on rec): nothing re-reads rec inside the
kernel, and letting it sit in L2 evicts the `w_f_b` tile. **+9%** at B=64
(157.7 → 144.4 µs at the time).

**Block swizzle**: `lin = blockIdx.x + gridDim.x·blockIdx.y`, `b = lin/96`,
`h = lin%96` — a bijection over the mandated `grid (B, 96)`. With the natural
mapping, blocks issued at the same time are 64 different rows at the same head
offset, i.e. 64 streams 6.5 MB apart; head-fastest makes concurrently-resident
blocks walk contiguous rec. **+8%** at B=64 (157.7 → 155.3 µs alone, and the two
together 140.4 µs).

**`w_f_b` (the f_b projection).** This was the single biggest miss in the first
working version: 32 KB per block, read row-per-thread, cost **17.6 µs of 140 µs at
B=64** — not because of the bytes (the head's tile is shared by all B rows and comes
out of L2; measured L2 hit rate 55.7%) but because a thread-per-row walk of a 256-B
row generates 32 wavefronts per warp instruction, 8× the L1 wavefronts of the entire
rec traffic. The tile is contiguous (`w_f_b + h·128·128`), so it is read with the
same 8-lane split (one instruction = 4 rows × 128 B) and reduced with the same
shuffle tree: 140.4 → 122.5 µs back-to-back (the adopted pairwise-tree form of
that loop costs ~2 µs more than the plain fma chain and is kept for accuracy, §5). `sh_flow` is padded (8-float groups on a 12-float
stride) so the four 16-B LDS in that loop are bank-conflict free — ncu's excessive
shared wavefronts drop from 393216 (13% of all wavefronts) to 24576 (1%); no
measurable time change, the kernel is DRAM-bound.

**Accuracy of ga** (see §5): every `flow[j]·w_f_b[j]` is exact in f32 (8+8 mantissa
bits), so the summation is the only error source. It is a full pairwise tree — fma
pairs, 3 in-lane levels, 3 shuffle levels, ~7·u instead of the ~128·u of a serial
walk. That matters because `ga` lands in bf16 before `dt_bias`.

**Landing points** are exactly the document's / TileLang `kda_core_batched`'s: bf16
per q·q and k·k term with an f32 sum; the l2 sum lands bf16 before `+1e-6`; rsqrt
lands bf16; `q·qr`, `k·kr` land bf16 (q then scaled by 128^-0.5 in f32); `ga` lands
bf16 before `dt_bias`; `attn` lands bf16 once; the rms product (f32 rsqrt × f32
gamma_o) lands bf16 once; the gate lands bf16 twice (partial, then sigmoid); the
final `out` is a bf16×bf16 product. The state stays f32 end to end.

## 4. ncu (`--set full -k regex:kern_k3_kda_core -c 1`, B=64)

Full report: `tools/k3-harness/reports/k3_kda_core.ncu.txt`.

| metric | value |
|---|---|
| Duration | 129.25 µs (instrumented; 125.31 µs uninstrumented median) |
| DRAM / Memory Throughput | **73.92 % = 5.86 TB/s** |
| Compute (SM) Throughput | 23.40 % |
| L2 Hit Rate | 55.69 % |
| L1/TEX Hit Rate | 18.98 % |
| Achieved Occupancy | 17.75 % (theoretical 18.75 %, 3 blocks/SM, register-limited) |
| Registers / thread | 158, 0 spill, 3872 B smem |
| Local memory spilling requests | 0 |
| Elapsed cycles / SM freq | 267 k @ 2.05 GHz |

ncu's 73.92% is measured against its own DRAM peak and over the instrumented
duration; the wall-clock number is 805.3 MB of rec traffic in 122.9 µs =
**6.55 TB/s = 82% of the 8 TB/s nominal peak**, which is 101% of the measured
read+write stream ceiling (§1). L2 hit rate ~56% is what a write-allocate on a line
that the same block just read looks like (reads miss, writes hit) plus the `w_f_b`
sharing — i.e. rec does move 400 MB in + 400 MB out of DRAM, as intended.

Occupancy is low and that is fine: the kernel is DRAM-bound with 23% SM throughput,
and 3 blocks/SM × 128 threads × 256 B of outstanding loads is ~15 MB in flight
across the device, more than enough to saturate. A deliberately smaller variant
(`ROWS_PER_ITER = 2`, 80 registers, 6 blocks/SM) was **slower** (145.8 vs 140.4 µs
in the same configuration) — fewer bytes in flight per thread beats the extra
occupancy. Fully unrolling the dv loop lets ptxas drop to 108 registers and it is
much worse (143.6 µs), for the same reason.

## 5. What I could not achieve / caveats

* **B=1 and B=2 are latency bound, not bandwidth bound** (5.44 µs / 6.87 µs
  back-to-back, ~29% / 46% of peak). At B=1 there are 96 blocks on 148 SMs — one
  block per SM, no second block to overlap with, and 128 threads × 256 B = 3.1 MB
  of loads in flight device-wide against a ~1 µs DRAM latency. The only levers are
  more threads per block (the ABI fixes block = 128 and the harness launches
  exactly `grid(B,96,1) block(128,1,1)`) or more rows in flight per thread
  (`ROWS_PER_ITER = 8` would need 128 state + 48 coefficient registers and spills).
  I left it: B=1 is 5.4 µs against a 12.6 MB floor, i.e. it is ~2.3 TB/s already,
  and the shape that matters for the 800 MB/layer problem is B ≫ 1.
* **No headroom left at B=8/64** short of moving less state — the kernel matches a
  bare `float4` read-modify-write stream over the same footprint to within 1%.
  The only real win from here is algorithmic (e.g. keeping rec in a lower precision
  than f32, which the ABI does not allow).
* **The state is not bit-tight against the reference, by construction.** `ga` lands
  in **bf16** before `dt_bias`, and the harness reference accumulates it in
  *double*; a last-bit f32 difference in that sum therefore flips the bf16 landing
  for a handful of the 786 k `ga` values (~2.5e-5 probability each), and a flipped
  `dec[k]` moves a whole dv row of the state by ~3e-4. Measured at B=64:
  `rec max|err| = 4.97e-05`, `relRMS = 5.7e-07` — an order of magnitude inside the
  `3·ULP + 1e-3` rule, but it is why the state does not come out at 1e-7. I chose
  the pairwise tree (§3) precisely to minimise this: a serial f32 walk in the
  reference's own j order, which I tried first (staging the tile through 34 KB of
  smem), is *less* accurate against a double reference — `rec max|err|` 1.02e-04
  vs 4.97e-05 at B=64 — and 3 µs slower. The tree is both faster and closer.
* **`attn` is computed by the algebraic identity**, not by re-reading the stored
  f32 `S'`. The reference does `Srow[k] = (float)(...); a += (double)Srow[k]*qs[k]`,
  i.e. it accumulates from the *rounded* stored value; the identity accumulates
  from the unrounded `S·dec` plus `dlt·(kn·qs)`. Mathematically identical, and it
  is what makes one-touch rec possible; the difference is ~1e-7 relative before a
  bf16 landing, so a few `attn` values land 1 ULP apart. `out relRMS` at B=64 is
  2.66e-05 against a 2e-3 limit.
* ncu still flags 1% excessive shared wavefronts and a 0.18% coalescing estimate;
  both are noise at 74% DRAM utilisation.

## 6. Document ambiguities (docs/k3-kernel-abi.md §K3)

1. `flow = bf16(wsm_partial[b, 96 .. 224])` reads as an inclusive-exclusive range
   of 128 values, i.e. columns 96..223. The harness reference agrees (`96 + j`,
   j < 128). Worth writing `96 .. 224)` or `96..223`.
2. The pseudo-code writes `attn[dv] = bf16(Σ_k S'[h,dv,k]·qs[k])` where `S'` is the
   value *already stored to rec*. That ordering (store f32, then read back for the
   dot product) is what the reference implements and it forbids the algebraic
   rearrangement that makes a single pass over rec possible. Since the contract is
   "tolerance, not bit-exact", the rearrangement is legal — but the document should
   say so explicitly, because the naive reading costs either a second pass over
   6.3 MB per row or 128 registers per thread.
3. Nothing in §K3 says whether `dec`/`ga` must be computed per-thread or may be
   redistributed; the bf16 landing of `ga` makes the result sensitive to the f32
   summation order (§5). If the intent is that any summation order is acceptable,
   a note in the "舍入链" paragraph would save the next author the experiment.
4. `gamma_o` is `const f32*` while every other `gamma` in the document is bf16
   (TileLang's comment says the checkpoint stores f32). Not a bug, just surprising.
5. `grid (B, HEADS, 1), block 128` is taken literally by the harness
   (`geo(o, HEADS, 1, 128, 1, 1, 0)`), so unlike K2/K5 there is no freedom to pick
   a different block size here. Fine for this kernel, but the "block 由你定"
   escape clause that K2/K4/K5 have does not exist for K3.

## 7. Scratch files

`tools/k3-scratch/k3/` holds my own pre-harness test (`test_kda_core.cu`, CPU
reference in f32 transcribed from the document + TileLang), the bandwidth ceiling
probe (`bw_ref.cu`), and the two GEMV variants I benchmarked against the harness
(`kda_exp.cu` = smem-staged serial order, `kda_var_c.cu` = the adopted pairwise
tree, `kda_var_u.cu` = fully unrolled dv loop). Nothing outside
`tools/kernels-src/k3_kda_core.cu`, `tools/k3-scratch/k3/` and
`tools/k3-harness/{notes,reports}/` was touched, and nothing was committed.
