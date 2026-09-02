# K1 — `tools/kernels-src/k3_residual.cu` (residual stream)

Three entries, one file, one cubin:
`kern_k3_attnres_rms` (K1a), `kern_k3_land_add_attnres_rms` (K1b), `kern_k3_land_add2` (K1c).

Built on tray03, measured on **tray07 GPU 0** (idle, verified with `nvidia-smi`):

```
/usr/local/cuda-13.1/bin/nvcc -cubin -arch=sm_103a -O3 -Xptxas -v \
    -o tools/k3-scratch/k1/k3_residual.cubin tools/kernels-src/k3_residual.cu
```

## Launch geometry (what the manifest should copy)

| entry | grid | block | dynamic smem |
|---|---|---|---|
| `kern_k3_attnres_rms` | (B, 1, 1) | 1024 | 0 |
| `kern_k3_land_add_attnres_rms` | (B, 1, 1) | 1024 | 0 |
| `kern_k3_land_add2` | (B, 1, 1) | 1024 | 0 |

Static shared memory is 2504 B for K1a/K1b and 0 for K1c. These are the harness defaults,
so no `--grid/--block/--smem` override is needed.

## Design

`H = 7168` bf16 is exactly **896 sixteen-byte vectors**, and 896 = 28 whole warps. Thread
`t` owns vector `t`; threads 896..1023 (warps 28..31) hold no data, so the `act` predicate
is warp-uniform and compiles to a branch rather than per-instruction predication. Every
load and store is a fully coalesced 16 B access. Every reduction is a fixed-order
`__shfl_xor` butterfly over a fixed 32-slot warp layout — no atomics, no schedule
dependence, bit-reproducible run to run.

Per row, for `nb > 0`:

1. **pass 1** — one load per candidate (the `nb` snapshot rows, plus the prefix which is
   already in registers), accumulating **both** `sum(x²)` and `sum(x·sw)` from that single
   load. The score is factored as `score_c = rsqrt(mean(x²)+eps) · sum(x·sw)` instead of
   scaling every element by the `rms_nw` scalar first: the same value up to f32 rounding,
   and it removes an entire extra pass over the row. `sw` lives in 8 registers and is
   reused by every candidate — that is why the row is split one vector per thread rather
   than one candidate per warp group. The candidate loop has a compile-time trip count
   (`NB_MAX`) and is two loads deep.
2. **combine** — one barrier, then warp *k* reduces the 32 warp partials of value *k*, so
   up to 18 warps do their butterfly in parallel. A second barrier publishes the 18 sums.
   (The obvious version — one warp walking 18 × 32 smem slots — has no latency to hide
   behind and was measurably worse.)
3. **softmax** — recomputed identically by every warp from those 18 sums, so lane `c` ends
   up holding `p_c` and pass 2 picks it up with a `__shfl_sync` broadcast; no smem round
   trip, no third barrier.
4. **pass 2** — re-read the candidates (they are in L1/L2 from pass 1) and mix in f32, one
   bf16 landing.
5. **rms** — one more block reduction over the landed `mixed` row, then `· gamma` as
   bf16 × bf16 → bf16.

`nb == 0` skips all of it (`mixed = prefix`, because `bf16(1.0f · f32(prefix)) == prefix`),
which is why `sw` is never dereferenced in that case, as the ABI requires.

K1a writes the snapshot straight from the registers holding the prefix, guarded by
`nb < NB_MAX` (see ambiguity 1). K1b computes `prefix2` first, keeps it in registers as the
last candidate, and writes it out; `blocks` stays read-only. K1c is a single elementwise
pass with `two` selecting whether `p2` is read at all.

### Memory pattern

Per row per launch, K1a/K1b at `nb = 8`: 9 × 14336 B read in pass 1 (cold — DRAM), the same
9 rows read again in pass 2 (warm — L1/L2; ncu reports L1/TEX hit ≈ 43 %), 14336 B of
`normed` written, plus 14336 B of snapshot (K1a) or `prefix2` (K1b). K1c reads 2 × 28672 B
of f32 partials plus 14336 B of `prefix2` and writes 14336 B. All accesses are 16 B per
thread, contiguous across the warp, one wavefront per warp instruction.

### Things tried and rejected (measured, not guessed)

* **Caching all `nb+1` candidate rows in dynamic shared memory** (up to 129024 B, read once
  instead of twice) — implemented behind a `%dynamic_smem_size` probe so `smem = 0` stayed
  correct. Worth **0.0 µs** at every shape: the kernel is instruction-issue bound, not L1
  bound, and `LDS.128` costs the same issue slot as the `LDG.128` it replaces. Removed, so
  the ABI is `smem = 0` and the kernel never needs
  `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`.
* **Deferring all 2·(nb+1) warp butterflies to the end of pass 1** (per-thread partials in
  registers so the loads pipeline freely): pushed registers 36 → 52 and was 15 % *slower*.
* **Four-deep candidate prefetch**: no better than two-deep. Two-deep kept (≈ 5 %).
* **One-candidate-per-128-thread-group reductions** (8 groups, 7 vectors per thread) would
  cut the shuffle instruction count ~5×, but shuffles are only ~15 % of the issued
  instructions and the scheme forces `sw` to be re-read per group; the modelled net win was
  ~10 % for a large rewrite. Not done — see "not achieved".

## `-Xptxas -v`

```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'kern_k3_land_add2' for 'sm_103a'
ptxas info    : Function properties for kern_k3_land_add2
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 30 registers, used 0 barriers
ptxas info    : Compiling entry function 'kern_k3_land_add_attnres_rms' for 'sm_103a'
ptxas info    : Function properties for kern_k3_land_add_attnres_rms
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 1 barriers, 2504 bytes smem
ptxas info    : Compiling entry function 'kern_k3_attnres_rms' for 'sm_103a'
ptxas info    : Function properties for kern_k3_attnres_rms
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 1 barriers, 2504 bytes smem
```

0 bytes spill / 0 bytes local on all three, and `cuobjdump -sass` has **0 `.MULTICAST`** and
**0 `LDL`/`STL`**. Register counts (40/40/30) are under the 64 implied by
`__launch_bounds__(1024, 1)`.

## Harness

`tools/k3-harness/harness`, tray07 GPU 0, `--reps 50`. **40/40 runs PASS, exit 0**:
B ∈ {1,2,8,64} × (nb,snapshot) ∈ {(0,0),(1,1),(4,1),(8,0)} for K1a and K1b, and
B ∈ {1,2,8,64} × two ∈ {0,1} for K1c. Median µs as the harness prints it (its timing puts a
pair of CUDA events around every individual launch, which adds ~4 µs of fixed overhead to
each number — see "Timing" below), next to the naive reference kernel from `run_all.log`:

| kernel | B | nb | snap | mine (µs) | naive (µs) | speedup | mine GB/s |
|---|---|---|---|---|---|---|---|
| attnres_rms | 1 | 0 | 0 | 6.43 | 9.73 | 1.5× | 11.1 |
| attnres_rms | 1 | 1 | 1 | 8.45 | 16.58 | 2.0× | 11.9 |
| attnres_rms | 1 | 4 | 1 | 10.02 | 25.76 | 2.6× | 14.3 |
| attnres_rms | 1 | 8 | 0 | 10.53 | 36.38 | 3.5× | 17.7 |
| attnres_rms | 64 | 0 | 0 | 6.72 | 9.25 | 1.4× | 279.5 |
| attnres_rms | 64 | 1 | 1 | 8.77 | 16.93 | 1.9× | 423.5 |
| attnres_rms | 64 | 4 | 1 | 10.62 | 25.34 | 2.4× | 608.6 |
| attnres_rms | 64 | 8 | 0 | 12.10 | 36.03 | 3.0× | 762.1 |
| land_add_attnres_rms | 1 | 8 | 0 | 10.88 | 35.07 | 3.2× | 21.1 |
| land_add_attnres_rms | 64 | 8 | 0 | 12.74 | 34.98 | 2.7× | 939.9 |
| land_add2 | 1 | – | two=1 | 6.59 | 7.74 | 1.2× | 13.0 |
| land_add2 | 64 | – | two=1 | 6.21 | 8.67 | 1.4× | 886.8 |

(B = 2 and B = 8 track B = 1 to within 3 %, as expected for a one-block-per-row kernel.)

Accuracy is not merely inside the tolerance: against a double-precision CPU reference
rounded to bf16, my own scratch tester
(`tools/k3-scratch/k1/test_k1.cc`, driver API, independent of the harness) reports
**relRMS = 0.00e+00 — bit-identical output at every shape**, including `nb = 8`, and for
`sw` scaled so the softmax is a real mixture rather than one-hot. The score factorisation
therefore costs nothing measurable.

## Timing without the launch floor

The harness number includes its per-launch event overhead. Back-to-back launches on one
stream (`tools/k3-scratch/k1/bench_k1.cc`, 2000 reps) give the pipelined cost, and an empty
kernel with the same signatures measures the floor at **1.8 µs** per launch on this machine
— which is the dominant term for this whole family:

| entry | B=1 nb=0 | B=1 nb=4 | B=1 nb=8 | B=64 nb=0 | B=64 nb=4 | B=64 nb=8 |
|---|---|---|---|---|---|---|
| `attnres_rms` | 2.06 | 5.42 | 6.78 | 2.52 | 5.92 | 7.30 |
| `land_add_attnres_rms` | 2.29 | 5.48 | 7.02 | 2.75 | 5.99 | 7.71 |
| `land_add2` | 2.02 | — | — | 2.48 | — | — |
| *empty kernel (floor)* | 1.76 | | | 1.84 | | |

So the GPU work itself is ≈ 0.3 µs (K1a, nb=0), ≈ 5.5 µs (K1a, B=64, nb=8) and ≈ 0.7 µs
(K1c, B=64). Time is flat from B=1 to B=148 (one wave of blocks) and doubles at B=256 —
this family is per-block latency/issue bound, never DRAM bound, at every batch size the
model uses.

## ncu

`tools/k3-harness/reports/k3_residual.ncu.txt` — `ncu --set full --launch-skip 20 -c 1`
through the harness at B = 64 (nb = 8, snapshot = 0 for K1a/K1b; two = 1 for K1c), tray07
GPU 0, SM clock 2.05 GHz.

| metric | `attnres_rms` | `land_add_attnres_rms` | `land_add2` |
|---|---|---|---|
| Duration | 11.17 µs | 11.87 µs | 5.66 µs |
| DRAM throughput | 745 GB/s = **9.4 %** of 7.9 TB/s | 856 GB/s = **10.9 %** | 811 GB/s = **10.4 %** |
| Achieved occupancy | 50.7 % (32.5 warps/SM) | 48.7 % (31.2) | 37.5 % (24.0) |
| L2 hit rate | 8.4 % | 9.0 % | 0.5 % |
| L1/TEX hit rate | 43.5 % | 43 % | 0 % |
| Registers / thread | 40 | 40 | 30 |
| Grid / block | 64 / 1024 | 64 / 1024 | 64 / 1024 |

The L1 hit rate of 43 % is exactly the second pass over the candidates hitting what the
first pass pulled in; L2 is nearly cold because ncu flushes it per replay.

### Why this is far from the §2 target of 60 % of peak bandwidth

The document's ≥ 60 % target is **not reachable by any kernel with this ABI**, and the
limit is the grid, not the code. `grid.x = B` is fixed by §0, so B = 64 gives 64 blocks on
152 SMs — 0.42 waves — before a single instruction executes. I calibrated the ceiling by
profiling a *pure streaming read* (1024 threads, 16 B per thread, nothing but loads and an
add) with the same geometry on the same card under the same `ncu --set full`:

| grid | pure-stream DRAM throughput |
|---|---|
| 64 blocks × 1024 threads | 1.31 TB/s = **16.7 %** of peak |
| 148 blocks × 1024 threads | 2.49 TB/s = **31.6 %** of peak |

So 16.7 % is the hardware ceiling for `grid = 64, block = 1024` on this machine, and
`attnres_rms` at 9.4 % is **56 % of that ceiling** (K1c at 10.4 % is 62 % of it). Reaching
60 % of 8 TB/s would need several waves of blocks, i.e. splitting a row across `grid.y`,
which K1a and K1b cannot do without a cross-block reduction for the scores — the whole row
has to be reduced before any of it can be mixed. §1's geometry line for K1a ("grid (B,1,1)")
rules that out, so I did not do it. If the residual family ever becomes the step's
bottleneck, the fix is not inside these kernels: it is fusing them into their neighbours
(the document's own goal 2) so the 1.8 µs launch floor and the 0.42-wave grid are paid
once instead of 186 times per step.

## Not achieved / left on the table

* **Instruction-issue bound, ~57 % of the streaming ceiling.** ncu measures IPC 2.41 with
  the top stall being an L1TEX scoreboard wait; the inner loop is ~3 ops/element in pass 1
  (unpack + 2 FFMA) and ~2 in pass 2, which is the floor for scalar SIMT code.
* **Tensor cores for the score pass.** `sum(x²)` and `sum(x·sw)` are a matrix–vector
  product; `mma.m16n8k16.f32.bf16.bf16` would cut the arithmetic ~7× (bf16×bf16→f32 is
  exact, so `sum(x²)` needs no extra care, and `sw` would have to be split into
  `bf16 hi + bf16 lo` — a bf16 `sw` alone shifts the scores by ~0.4 %, which moves the
  softmax far outside tolerance). Modelled ~30–40 % end-to-end (7.3 → ~5 µs at B=64 nb=8),
  against a large layout rewrite and a live, bit-exact kernel. Not attempted.
* **128-thread group reductions** (see "tried and rejected"): ~10 % modelled, not done.
* **128 of 1024 threads idle.** H/8 = 896 = 28 warps, so warps 28..31 never hold data.
  A block of **896** threads would be a perfect fit and would raise the useful fraction by
  14 %; the document pins block = 1024 for K1a/K1c, so I kept 1024. If the manifest is
  willing to take 896, say so and I will switch — the kernel is one constant away from it.

## Ambiguities in `docs/k3-kernel-abi.md` I hit

1. **K1a with `nb == NB_MAX` and `snapshot != 0` is out of bounds.** `blocks` is
   `[B, 8, H]`, so `blocks[b, 8]` does not exist, yet §1's tail is `attnres_rms(8)`. The
   harness's README raises the same point (its item 3) and forces `snapshot = 0` there. My
   kernel guards with `if (snapshot && nb < NB_MAX)` — it writes nothing rather than
   corrupting memory. Please either state that the tail call is snapshot-free or give
   `blocks` 9 slots.
2. **Score factorisation.** §0 defines `score_c = Σ_i rms_nw(cand_c)[i]·sw[i]`, i.e. scale
   every element and then dot. I compute the algebraically identical
   `rsqrt(mean+eps) · Σ_i x_i·sw_i`, which needs one pass over the row instead of two. The
   f32 rounding differs in principle; in practice the output is bit-identical to a
   double-precision reference rounded to bf16 at every tested shape. Flagging it because
   §0 says "照着做误差最小" about the landing chain.
3. **`snapshot` means two unrelated things** (write the prefix into `blocks` in K1a; select
   `prefix2 = p` in K1b). Implemented as written — the harness README flags it too (item 4).
   Worth two different parameter names.
4. **No geometry line for K1b.** §1 gives grid/block for K1a and K1c only. I assumed
   (B,1,1) / 1024, which is what the harness defaults to.
5. **K1c signature change** (`int two` before `int B`, `p2` always valid) arrived mid-task
   and is implemented in that form; `docs/`, `ref.h` and the harness agree.
6. **`block = 1024` vs `H/8 = 896`.** See "not achieved" — the documented block size cannot
   be fully occupied by 16 B vector loads of a 7168-wide row.

## Files

* `tools/kernels-src/k3_residual.cu` — the deliverable (three entries).
* `tools/k3-harness/reports/k3_residual.ncu.txt` — `ncu --set full` for all three entries.
* `tools/k3-scratch/k1/` — my own driver-API tester (`test_k1.cc`, bit-exactness against a
  double-precision CPU reference), timing harness (`bench_k1.cc`), the streaming ceiling
  micro-benchmark (`micro.cu`/`micro.cc`) and the ablation variants used above. Scratch, not
  part of the delivery.
