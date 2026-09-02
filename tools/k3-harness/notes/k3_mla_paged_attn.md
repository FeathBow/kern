# K5 — `kern_k3_mla_paged_attn`

Absorbed-MLA decode attention over the paged latent KV cache, rewritten in
`tools/kernels-src/k3_mla_paged_attn.cu`.

Built on tray03 with `/usr/local/cuda-13.1/bin/nvcc -cubin -arch=sm_103a -O3 -Xptxas -v`,
measured on tray08 GPU 0 (GB300, sm_103, 152 SM, driver 610.57.04).

## 1. What the old kernel did, and why it was slow

The certified kernel assigns one block to one `(batch, head)` pair and walks the whole
context inside that block. All 96 heads share the *same* 576-wide latent rows (MLA is
MQA-like after absorption), so the context is streamed from HBM **96 times**. At
ctx=32768 that is 96 × 37.75 MB = 3.6 GB of reads per sequence for 37.75 MB of unique
data. It is also entirely FFMA — no tensor cores.

Measured on tray08 GPU 0: 10046 µs at ctx=32768 B=1.

## 2. Design of the new kernel

Launch geometry:

```
grid    (B, 48, 1)      blockIdx.y = g * 8 + s
block   (512, 1, 1)     16 warps
cluster (1, 8, 1)       the 8 splits of one head group are one cluster
smem    216320 B, STATIC __shared__, 0 dynamic
```

* `g ∈ [0,6)` selects a **group of 16 heads**; `s ∈ [0,8)` selects one eighth of the
  pages. So a page is read once for 16 heads (16× less HBM traffic than the old
  kernel) and the remaining parallelism comes from the KV split, not from head
  replication.
* The 8 splits of a head group form one **thread-block cluster**, so the split-KV merge
  happens in distributed shared memory. That matters because the ABI signature has no
  global scratch buffer — there is nowhere to put per-split partials in HBM.

Per-block flow:

1. **Land q.** `q_partial[b, h*192..+192]` (f32) → bf16 in smem: nope 128 | rope 64.
2. **Absorb.** `q_abs[0:512] = bf16(Σ_d q_nope[d] · W_UK_h[d, j])`, `q_abs[512:576] = q_rope`.
   Each block computes only the `16/8 = 2` heads it owns, then the cluster all-gathers
   the group's 16 rows through `ld.shared::cluster`. This is a 128×512 matvec per head;
   doing it once per cluster instead of once per block saves 8× the work.
3. **Page walk, pass 1.** Pages are staged with `cp.async.cg` (16 B/thread) into a
   double-buffered `[64][584]` bf16 tile — pitch 584 = 576 + 8 gives 4 banks of skew so
   `ldmatrix` is conflict-free. Scores are `mma.sync.m16n8k16.f32.bf16.bf16.f32` with
   the 16 `q_abs` rows as the A operand (M=16 exactly fills the tile) and the page rows
   as B. 16 warps split as 8 token groups × 2 k halves (288 wide each); the two k
   partials are folded in f32 and landed as `f32(bf16(dot) · scale)` with the multiply
   in bf16, per the document. Only the running `(m, l)` is kept.
4. **Cluster reduce (m, l).** Each block publishes its `(m, l)` and reads its 7 peers',
   producing the group-global `(m, l)`. This is what makes the documented landing
   `p = f32(bf16(exp(s − m)/l))` reproducible: `l` is the *final* `l`, so a single-pass
   online rescale cannot express it (see §5).
5. **Page walk, pass 2.** Scores are recomputed (cheaper than spilling 32768 f32 per
   head to smem/HBM), landed to bf16 probabilities, and `P·V` is another mma with the
   probability matrix as A and the latent half of the page as B, accumulating f32.
6. **Merge and expand.** Each block publishes its `[16][512]` f32 latent partial,
   the cluster sums all 8 (each block owns 2 of the 16 heads for this step), lands
   `bf16(lat)`, and does the `W_UV` expansion — 128 dots of length 512 per head, one
   warp per output chunk with a shuffle reduction.
7. **Gate.** `gated = bf16(bf16(o) · bf16(σ(f32(mla_gate))))`, written from a coalesced
   `tid`-strided loop over the block's `2 × 128` outputs.

Loop-invariant `q_abs` A-fragments are **pinned in registers** (`unsigned qf[18][4]`)
across the whole page walk — that removed the largest single source of smem traffic
(343 µs → measured below).

## 3. Head-group choice

`K5_GU` (heads per group) was made a compile-time knob in `tools/k3-scratch/k5/k5_grp.cu`
and swept at fixed total work. Median µs on tray08 GPU 0:

| heads/group | blocks (B=1) | ctx=32768 B=1 | ctx=32768 B=8 |
|---|---|---|---|
| 8  | 96 | 251.7 | 1764 |
| 16 | 48 | 258.1 | 1035 |
| 32 | 24 | 291   | 1012 |

GRP=8 wins marginally at B=1 (more blocks, 96 vs 48, on a 152-SM part) but loses badly
at B=8 because it doubles the HBM traffic — at B=8 the cache slice no longer fits L2 and
traffic dominates. GRP=16 is also the value that exactly fills the `m16n8k16` M
dimension with no masking. **GRP=16 shipped.**

## 4. Split-KV across blocks

This is implemented, not just designed. Splitting only by head group gives 6 blocks at
B=1, and a single SM streams at ~130 GB/s (measured in `tools/k3-scratch/k5/bw2.cu`,
TMA-1D and 1024-thread plain both top out there), so one block reading 37.75 MB needs
~515 µs no matter how good the math is. The split is therefore mandatory, not optional.

Shape of the split: block `s` of group `g` processes pages `s, s+8, s+16, …` and keeps
a partial `(m, l, acc[16][512])`. The merge is two cluster barriers:

* after pass 1, an 8-way max/sum reduction of `(m, l)` — `m_g = max_s m_s`,
  `l_g = Σ_s l_s · exp(m_s − m_g)`;
* after pass 2, an 8-way sum of the latent accumulators. Because pass 2 already
  normalises against the *group-global* `(m, l)`, the second merge is a plain sum with
  no rescale — the rescale is folded into the probability landing.

Both use `mapa.shared::cluster` + `ld.shared::cluster` with
`barrier.cluster.arrive.release.aligned` / `barrier.cluster.wait.acquire.aligned`.
The partials never touch HBM, which is the reason to use a cluster rather than a global
scratch + second kernel — the ABI gives no scratch pointer.

Expected and realised gain: 8 splits × 6 groups = 48 blocks at B=1 instead of 6, i.e.
the memory-bound floor drops from ~515 µs to ~64 µs; measured 258 µs (the gap is the
two-pass walk plus the mma work, both of which scale with the split too).

**SPL=16** (96 blocks, cluster size 16) was built and measured: **155 µs** at ctx=32768
B=1, a further 1.7×. It is *not* shipped because cluster sizes above 8 require
`cudaFuncSetAttribute(f, cudaFuncAttributeNonPortableClusterSizeAllowed, 1)`, and
neither `tools/k3-harness/harness.cu` nor the kern runtime
(`crates/kern-runtime/src/lib.rs`, the `cuLaunchKernelEx` path) sets it — the launch
silently produces zeros. If the runtime ever sets that attribute, flipping `K5_SPL` to
16 is a one-line change.

## 5. Numerics

The document's chain is reproduced exactly except for reduction order:

* `q_abs = bf16(Σ q_nope · W_UK)` — f32 accumulate, bf16 land.
* `score = f32(bf16(dot) · scale)`, multiply in bf16.
* online softmax in f32 over 64-token pages, `p = f32(bf16(exp(s − m)/l))` against the
  **final** `m` and `l` — this is what forces the two-pass walk. A single-pass online
  variant (rescaling `acc` as `m` moves, dividing by `l` at the end) was implemented and
  measured at 1.6–2.2e-3 relative RMS against the document chain — i.e. it would sit on
  top of the 2e-3 limit. Two passes cost ~35% and buy 3–4× the margin.
* latent accumulated f32, landed bf16; `W_UV` expansion f32, landed bf16; then the gate.

Tensor-core reductions reorder the dot products relative to the reference, which the
3 ULP + 1e-3 / 2e-3-RMS tolerance explicitly permits. Observed relative RMS is ~5e-4 at
ctx=32768 — the same order as the *old certified kernel* against a double-precision CPU
reference (`tools/k3-scratch/k5/ref.h`, which had to be moved to double accumulation:
naive f32 summation of `l` over 32768 terms is itself 1.2e-3 off and made both kernels
look bad).

## 6. ptxas -v

```
ptxas info    : Compiling entry function 'kern_k3_mla_paged_attn' for 'sm_103a'
ptxas info    : Function properties for kern_k3_mla_paged_attn
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 128 registers, used 1 barriers, 216320 bytes smem
```

0 spill / 0 local, as required. SASS check (`cuobjdump -sass`): **0 `.MULTICAST`**;
52 `HMMA.16816.F32.BF16`, 40 `LDSM.16.M88.4`, 8 `LDSM.16.MT88.4`.

## 7. Results

Harness, tray08 GPU 0, median of `--reps 30` (new) / `--reps 20` (old baseline), µs:

| ctx | B | old | new | speedup |
|---|---|---|---|---|
| 1     | 1 | 59.07    | 24.35  | 2.43× |
| 1     | 8 | 227.20   | 79.62  | 2.85× |
| 64    | 1 | 78.05    | 24.58  | 3.18× |
| 64    | 8 | 242.02   | 80.10  | 3.02× |
| 65    | 1 | 80.19    | 24.67  | 3.25× |
| 65    | 8 | 243.49   | 80.16  | 3.04× |
| 2048  | 1 | 679.97   | 36.19  | 18.8× |
| 2048  | 8 | 985.44   | 102.46 | 9.6×  |
| 32768 | 1 | 10046.62 | 258.11 | **38.9×** |
| 32768 | 8 | 15792.48 | 683.58 | 23.1× |

Targets: ≥3× at ctx=32768 B=1 → **38.9×**. Not slower at ctx ≤ 64 → **2.4–3.2× faster**.
All required shapes PASS (also PASS at B=2 for every ctx, and across seeds
1234 / 7 / 99 / 424242, with per-row varying `seq_lens` and shuffled page tables).

Optimisation history at ctx=32768 B=1: v1 grouped+split 387 → register-pinned q_abs
fragments 343 → 512 threads 284 → block-table prefetch, one fewer barrier 264 →
latency/unroll 261 → DSMEM unroll + accx padding 258 → coalesced epilogue stores 258
(but ctx=1 27.3 → 24.4 and ctx=2048 B=8 115 → 102).

Small-ctx budget, from the ablation in `tools/k3-scratch/k5/k5_abl.cu` (ctx=1, 24.9 µs
total): empty skeleton 11.6, `W_UV` expansion 5.0, page walk 4.8, `q_abs` absorb 3.2.
Raw launch overhead is 1.57 µs. The floor is the fixed per-call work (absorb + expand),
not the context.

## 8. ncu

`--set full`, ctx=32768, saved to `tools/k3-harness/reports/k3_mla_paged_attn.ncu.txt`.

| | B=1 | B=8 |
|---|---|---|
| Duration | 264.77 µs | 684 µs |
| DRAM throughput | 3.02 % | 28.79 % |
| Memory throughput | 11.49 % | — |
| L1/TEX throughput | 37.48 % | 37.04 % |
| Compute (SM) throughput | 9.83 % | 19.13 % |
| Executed IPC active | 1.28 | 1.27 |
| L2 hit rate | 85.17 % | 51.05 % |
| Achieved occupancy | 24.99 % | 24.99 % |
| Registers / thread | 128 | 128 |
| Static shared / block | 216.32 KB | 216.32 KB |
| Grid / cluster | 48 / 8 | 384 / 8 |
| Waves per SM | 0.32 | 2.53 |

Reading: at B=1 the kernel is *not* DRAM-bound any more (3 %, 85 % L2 hit — the 37.75 MB
slice is read once and reused by 16 heads). It is latency-bound: 0.32 waves/SM means
120 of 152 SMs are idle, and occupancy is capped at 25 % (1 block/SM) by the 216 KB
static smem. That is the deliberate trade — the smem buys the 16-head reuse and the
DSMEM merge. B=8 fills the machine (2.53 waves) and moves toward DRAM (28.8 %).

## 9. Not achieved / open

* **SPL=16 (155 µs, another 1.7×)** is blocked on
  `cudaFuncAttributeNonPortableClusterSizeAllowed`, which the harness and the kern
  runtime do not set. See §4.
* **Shared-store bank conflicts.** ncu reports the shared stores at ~2.8-way conflict
  (44 % of shared-store wavefronts, est. 16.5 % speedup). The conflicting pattern is the
  mma fragment write-out, whose (head, token) lane mapping cannot be made
  conflict-free with any linear pitch; transposing the tile trades it for a 4-way load
  conflict on the next mma. Left as is.
* **Occupancy 25 %.** One block per SM by construction. A smaller page tile (32 tokens)
  would allow 2 blocks/SM but halves the cp.async batch and measured worse.
* **B=64** (outside K5's documented acceptance set of B ∈ {1, 8}): at ctx=64 and 65 the
  harness fails the **old certified kernel** too — an elementwise-tolerance artifact on
  near-zero outputs. At ctx=32768 B=64 the new kernel misses on 1 element of 786432
  (max abs err 2.44e-3, max 18841 ULP, relative RMS 6.14e-4 — well inside the 2e-3 RMS
  bound); the old kernel passes there (max abs err 4.88e-4).

## 10. ABI ambiguities found

1. **>48 KB shared memory and the launch path.** The document does not say who is
   responsible for the `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)`
   opt-in. The harness (`harness.cu`, plain `cuLaunchKernel`) does not do it, so a
   kernel with a large *dynamic* smem request fails with `CUDA_ERROR_INVALID_VALUE`.
   Worked around by declaring the 216320 B as **static** `__shared__`, which sm_103a
   accepts with no opt-in (verified up to 200000 B + clusters in
   `tools/k3-scratch/k5/statshm2.cu`). Kernels in this repo that need a big carveout
   should therefore use static smem, or the launch path should learn the opt-in.
2. **Cluster launches.** Likewise nothing in the ABI covers `__cluster_dims__`. It works
   through both the harness and `cuLaunchKernelEx` up to size 8; above 8 it needs an
   attribute nobody sets (§4). Worth stating explicitly in the document, since clusters
   are the only merge mechanism available to a split-KV kernel with no scratch pointer.
3. **`scale` is `bf16[1]` on the device** and the document says the score multiply is in
   bf16; it does not say whether `scale` may be hoisted to f32 once per block. Kept in
   bf16 for the multiply, as written.
