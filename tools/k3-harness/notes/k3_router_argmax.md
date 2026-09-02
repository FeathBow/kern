# K6 — `k3_router_argmax.cu` (`kern_k3_router_topk`, `kern_k3_argmax_f32_partial`, `kern_k3_argmax_f32_final`)

Source: `tools/kernels-src/k3_router_argmax.cu`
Built: `/usr/local/cuda-13.1/bin/nvcc -cubin -arch=sm_103a -O3` on tray03.
Run/profiled: tray08 GPU 1 (`CUDA_VISIBLE_DEVICES=1`), idle card.

## 1. Status

| check | result |
|---|---|
| harness `--kernel router_topk` B ∈ {1,2,8,64} | **PASS** (idx exact, wts max\|err\| ≤ 3.0e-8, relRMS ≤ 7.7e-8) |
| harness `--kernel argmax_f32` B ∈ {1,2,8,64} | **PASS** (`out` exact, `pmax`/`pidx` invariant) |
| deliberate tie cases (experts 40/41; duplicated row max) | **PASS** — smallest `e` / smallest index wins |
| `-Xptxas -v` spill / local | **0 bytes** on all three entries |
| `.MULTICAST` in SASS | **0 occurrences** |
| registers vs `__launch_bounds__` | consistent (see §3) |

## 2. Design

### `kern_k3_router_topk` — grid (B,1,1), block 256, 1920 B static smem, 0 dynamic

The semantics are strictly sequential — 16 max picks, each removing its winner —
so the round latency *is* the kernel. Everything else is a 896 B load.

* **Phase 1** (threads 0..223, one expert each): coalesced `S[b,e]` and `bias[e]`
  loads, `sig = 1/(1+expf(-S))`, and `ord(sig+bias)` written to smem alongside
  `sig`. `expf`, not `__expf`: `idx` has to match a CPU reference *exactly*, and a
  1e-7 wobble in `sig` flips a near-tie. `__syncthreads`.
* **Phase 2** (warp 0 only): lane `t` owns experts `t*7 .. t*7+6` in registers
  (224 = 32·7 exactly). Per round:
  1. 6 register maxes → this lane's best `ord`;
  2. `__reduce_max_sync` (one `REDUX.MAX` instruction) → the global best `ord`;
  3. an unrolled scan for this lane's *smallest* expert holding it (`0xFFFFFFFF`
     if none);
  4. `__reduce_min_sync` → the winning expert id, i.e. **ties resolve to the
     smallest `e` structurally**, because the layout `e = lane*7 + j` makes
     "smallest lane then smallest slot" equal "smallest `e`";
  5. the winning lane blanks that slot (`ord = 0`, strictly below `ord()` of any
     real float) and stages `(e, sig)` into `s_e[r] / s_w[r]`.
  Neither register array is ever dynamically indexed — that would spill to local.
* **Epilogue**: lanes 0..15 read their staged winner, a 4-step butterfly gives
  every lane `den = Σ sig`, and each writes one `idx`/`wts` element coalesced.

**Why REDUX and not a packed 64-bit key.** The obvious formulation packs
`(ord(biased) << 32) | (223-e)` into a `u64` and does a 5-step `__shfl_xor` max.
That is 10 `SHFL` instructions per round on a strictly serial dependence chain —
80 dependent shuffles over 16 rounds. Measured side by side on the same inputs
(bit-identical `idx`, max |Δwts| = 4.5e-8):

| variant | B=1 | B=64 |
|---|---|---|
| u64 key, 5× `__shfl_xor` | 3.74 µs | 3.95 µs |
| 32-bit `ord` + `__ballot` + broadcast shuffle | 3.74 µs | 3.94 µs |
| **32-bit `ord` + REDUX.MAX + REDUX.MIN (shipped)** | **2.90 µs** | **3.23 µs** |

A "floor" kernel — phase 1 plus a single round, no loop — measures 1.47 µs, so
the 16 picks now cost ~1.4 µs and the load + sigmoid + barrier ~0.9 µs on top of
a 0.53 µs launch node.

The removal sentinel differs from pegainfer's (`biased[bi] = -1e30`) but is
equivalent: `sig ∈ [0,1]` and `bias` is finite, so `biased > -1e30` always, and
`ord = 0` is below `ord()` of every real float (the most negative finite float
maps to `0x00800000`).

### `kern_k3_argmax_f32_partial` — grid (B,64), block 1024, 256 B static smem

Block `(b, p)` scans row `b`'s `p`-th contiguous chunk of `ceil(n/64)` = 2560
floats. `n % 4 == 0` and `chunk % 4 == 0` make `lo` and `hi` multiples of 4, so
the `float4` path covers the chunk exactly — no ragged tail. Each thread takes
**two** `float4`s (32 B), builds the packed `(ord(v) << 32) | (INT_MAX - i)` key
per element, and the block reduces with one `u64` max: 5 `__shfl_xor` steps, one
`u64` per warp through smem, warp 0 finishes. The tie rule ("largest value, then
smallest index") is therefore an integer `max`, no branches. A non-multiple-of-4
`n`/`chunk` falls back to a scalar grid-stride loop (verified at n = 100003).

**`__launch_bounds__(1024, 2)` is load-bearing.** 2048 threads/SM is the hardware
cap, so a 1024-thread block can be resident at most twice — and only if it fits
in 32 registers. The first version used 36 and ran **40.8 µs** at B=64; at 24
registers two blocks fit and it runs **20.5 µs**.

Steps, measured (graph replay, B=64):

| version | B=64 |
|---|---|
| 1 float4/thread, 36 regs (1 block/SM) | 40.8 µs |
| 1 float4/thread, ≤32 regs (2 blocks/SM) | 23.5 µs |
| 2 float4/thread | 19.9 µs |
| + generic scalar fallback & empty-chunk handling (shipped) | 20.5 µs |
| *load-only ceiling* (same grid, no reduction, no key build) | *15.5 µs* |

The shipped kernel is at **75 % of the load-only ceiling for this grid**.

Two rejected variants: `atomicMax` on a shared `u64` instead of the two-level
shuffle (25.4 µs), and a separate f32-max reduction followed by a min-index
reduction (23.3 µs — cheaper per element, but two serial block reductions).

### `kern_k3_argmax_f32_final` — grid (B,1,1), block 64

Same key, folds `parts` partials (skipping the `INT_MAX` sentinel a would-be
empty chunk writes), two warps through smem, thread 0 stores the i64 index.
Because the fold re-applies the global rule, the partial kernel's split does not
have to be the documented one — see ambiguity 2 below.

## 3. `ptxas -v`

```
ptxas info    : Compiling entry function 'kern_k3_argmax_f32_final' for 'sm_103a'
ptxas info    : Function properties for kern_k3_argmax_f32_final
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 16 registers, used 1 barriers, 256 bytes smem
ptxas info    : Compiling entry function 'kern_k3_argmax_f32_partial' for 'sm_103a'
ptxas info    : Function properties for kern_k3_argmax_f32_partial
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 1 barriers, 256 bytes smem
ptxas info    : Compiling entry function 'kern_k3_router_topk' for 'sm_103a'
ptxas info    : Function properties for kern_k3_router_topk
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 36 registers, used 1 barriers, 1920 bytes smem
```

`__launch_bounds__`: router (256, 1) → 36 regs, 6 blocks/SM by registers, 75 %
theoretical occupancy; argmax partial (1024, 2) → 24 regs, 2 blocks/SM, 100 %
theoretical; argmax final (64, 1) → 16 regs, 100 % theoretical.

## 4. Timing

Two clocks are reported because they measure different things.

* **graph replay** — 200 kernel nodes captured into one CUDA graph, replayed 5×,
  wall time / 1000. This is the number that matters for a captured decode step.
  A null kernel measures **0.525 µs** in the same harness, so that is the
  per-node floor included in every row below.
* **harness median** — `./harness --reps 50`, one `cudaEventRecord` pair per
  launch. It carries a ≈5.5 µs host-launch floor on this box (the naive `land`
  kernel, which does almost nothing, medians at 5.66 µs), so treat it as a
  like-for-like comparison only.

| entry | shape | B=1 | B=2 | B=8 | B=64 | naive (harness, B=1 / B=64) |
|---|---|---|---|---|---|---|
| `router_topk` graph | E=224, K=16 | **2.90** | 2.91 | 2.93 | **3.23** | — |
| `router_topk` harness | | 7.68 | 7.87 | 8.03 | 8.86 | 20.70 / 20.80 |
| `argmax_f32_partial` graph | n=163840 | **1.99** | 2.00 | 3.80 | **20.53** | — |
| `argmax_f32_final` graph | parts=64 | 1.29 | 1.31 | 1.32 | 1.52 | — |
| `argmax_f32` (both) harness | n=163840 | 8.67 | 8.64 | 11.01 | 27.90 | 9.63 / 48.74 |

Achieved bandwidth (roofline bytes ÷ graph time minus the 0.53 µs node floor):

| entry | B=64 traffic | device time | GB/s | % of 8 TB/s |
|---|---|---|---|---|
| `router_topk` | 66 KB | 2.70 µs | 24 | 0.3 % |
| `argmax_f32_partial` | 41.9 MB | 20.0 µs | **2100** | **26 %** |
| `argmax_f32_final` | 33 KB | 0.99 µs | 33 | 0.4 % |

## 5. ncu

`tools/k3-harness/reports/k3_router_argmax.ncu.txt` — `--set full
--clock-control none`, B=64, warm launch (`--launch-skip 4`), driver
`tools/k3-scratch/k6/prof_k6 64 5`.

| metric | `router_topk` | `argmax_f32_partial` | `argmax_f32_final` |
|---|---|---|---|
| Duration (ncu) | 7.23 µs | 29.70 µs | 5.60 µs |
| DRAM Throughput | 0.12 % | 17.84 % | 0.09 % |
| Memory Throughput | 9.35 GB/s | 1.41 TB/s | 6.81 GB/s |
| Compute (SM) Throughput | 1.22 % | 45.44 % | 0.23 % |
| L2 Hit Rate | 35.2 % | 0.58 % | 28.7 % |
| Achieved / Theoretical Occupancy | 5.96 % / 75 % | **76.4 % / 100 %** | 2.93 % / 100 % |
| Elapsed Cycles @ 2.06 GHz | 16187 | 108930 | 12225 |

**ncu's Duration is not the kernel's real duration on this machine.** It adds a
fixed ≈4–5 µs of instrumentation per profiled launch: `argmax_f32_final` does
~33 KB of work and cannot take 5.6 µs, and it measures 1.52 µs under graph
replay. Scale the DRAM percentages accordingly — `argmax_f32_partial` moves the
same 41.9 MB in 20.0 µs, i.e. 2.1 TB/s / 26 % of peak, not 17.8 %.

## 6. What was not achieved, and why

**§2 asks memory-dominated kernels for ≥ 60 % of the 8 TB/s peak.
`argmax_f32_partial` reaches 26 %. The document's own grid makes 60 %
unreachable**, and I measured the ceiling rather than asserting it:

* grid (B, 64) × block 1024 gives every block exactly `n/64` = 2560 floats
  = 10 KB, regardless of B.
* 1024-thread blocks cap at 2 per SM (2048 threads/SM), so at most
  148 × 2 × 10 KB ≈ 3.0 MB of the row can be in flight across the whole GPU.
* At ~700 ns of HBM latency that is ~4.2 TB/s of *issue* capability, and the
  measured **load-only** kernel — identical grid, loads and discards, no
  reduction at all — runs 15.5 µs = **2.79 TB/s (35 % of peak)**. That is the
  wall. The shipped kernel is at 75 % of it.
* Reaching 60 % needs the ABI to let one block own more of the row: grid
  (B, 8) with block 1024 gives 80 KB/block and ~24 MB in flight, or block 256
  with 8 blocks/SM. Either is a one-line manifest change and I am happy to
  deliver it if the grid formula can move. **I did not change the grid, because
  §0 says the manifest copies the document's formula.**

`router_topk` moves 66 KB and is 100 % latency: 16 strictly serial picks over a
896 B row, in a single block per row. Bandwidth is not a meaningful target for
it; the number that matters is the 2.90 µs, and it went there from 3.74 µs.
Remaining headroom is roughly 0.9 µs of DRAM round trip for `S`/`bias`, which no
kernel-side change removes, plus the 0.53 µs launch node.

I looked at and rejected two further router ideas, both worse on paper:
a full bitonic sort of the 224 keys across the warp (≈120 cross-lane shuffles vs
the current 32 REDUX ops), and rank-by-counting (224 × 224 comparisons ≈ 1568
per lane). One idea I did *not* pursue and that might be worth ~0.3 µs: sorting
each lane's 7 keys once so the per-round local max is free — it removes 3 of the
~15 ALU ops per round, but the round is dominated by the two REDUX latencies.

## 7. ABI ambiguities / notes for the document owner

1. **`EXPERTS` is compile-time here, and the vendored reference disagrees with
   the constant table.** §0 says `EXPERTS = 224` and `kern_k3_router_topk` has no
   expert-count parameter, so 224 is baked in (and the 32×7 lane layout depends
   on 224 = 32·7). But the certified pegainfer kernel this replaces,
   pegainfer's TileLang `k3_router_topk_batched.cu` (was vendored in `tools/k3-tilelang/`, since removed), is generated as
   `k3_router_topk_b*_e896_topk16` — 896 experts. Full-K3 (896 experts) needs
   either a recompile with `EXPERTS 896` (896 = 32·28, so the layout still
   works, at 28 registers of `ord` + 28 of `sig` per lane — that will want a
   different split) or an `int experts` parameter in the signature. Please say
   which. The harness fixes 224, so this is not exercised.
2. **The partial split is unspecified** (the harness raises this as its open
   question 5). I use contiguous chunks of `ceil(n/parts)`, block `p` taking
   chunk `p`, which is the natural reading and makes the partial indices ordered
   across blocks. Nothing depends on it: stage 2 re-applies the global rule, so
   any partition that covers the row exactly is correct. If a fixed split is
   intended the document should name it, because `pmax`/`pidx` are otherwise not
   a function of the inputs.
3. **`-1e30` as the pick sentinel is an implementation detail leaking into the
   spec.** §K6's "after each pick set that biased entry to −1e30" only reproduces
   "remove it" because `sig ∈ [0,1]` and `bias` is finite. Stating the intent
   ("remove the picked expert from further consideration") would let
   implementations pick their own sentinel without an argument. I use 0 on the
   `ord` domain and have written the equivalence argument into the kernel header.
4. **Denominator summation order.** The reference accumulates `den` in pick
   order; I accumulate it with a 16-lane butterfly. f32 addition is not
   associative, so `wts` differs in the last bits (measured max |Δ| = 4.5e-8,
   ~1e-7 relative — three orders of magnitude inside the tolerance). Flagging it
   only because it is a deliberate deviation from the written order.
5. **`expf` accuracy is load-bearing for an exact-match output.** `idx` must
   match the CPU reference bit-for-bit, but it is derived from a transcendental.
   Two experts whose `biased` values differ by less than a `expf` ULP can be
   ordered differently by GPU and CPU libm, and neither is "wrong". With 224
   random f32 logits the collision probability is small but not zero. If this
   ever bites, the fix is on the document's side: define the tie/near-tie regime,
   or accept a set-equality check on `idx` rather than an ordered exact match.
   (Passing today at every B with the harness's seed, including its deliberate
   exact tie on experts 40/41.)

## 8. Reproducing

```bash
# build
/usr/local/cuda-13.1/bin/nvcc -cubin -arch=sm_103a -O3 -Xptxas -v \
  -o target/cubins/k3_router_argmax.cubin tools/kernels-src/k3_router_argmax.cu
/usr/local/cuda-13.1/bin/cuobjdump -sass target/cubins/k3_router_argmax.cubin | grep -c MULTICAST   # 0

# harness (tray08 GPU 1)
cd tools/k3-harness
for k in router_topk argmax_f32; do for B in 1 2 8 64; do
  CUDA_VISIBLE_DEVICES=1 ./harness --kernel $k \
    --cubin ../../target/cubins/k3_router_argmax.cubin --B $B || exit 1
done; done

# graph-replay timings and the extra shapes (n = 100003 ragged, tie cases)
cd tools/k3-scratch/k6
/usr/local/cuda-13.1/bin/nvcc -arch=sm_103a -O3 -o test_k6 test_k6.cu
/usr/local/cuda-13.1/bin/nvcc -arch=sm_103a -O3 -o bench_k6 bench_k6.cu
CUDA_VISIBLE_DEVICES=1 ./test_k6 && CUDA_VISIBLE_DEVICES=1 ./bench_k6
```

`tools/k3-scratch/k6/` also holds the variant benchmarks the tables above come
from: `var.cu` (router loop cost, argmax occupancy, situ math), `var2.cu`
(argmax load-only ceiling, `tanh.approx` accuracy sweep), `var3.cu` (argmax
per-thread depth), `var4.cu` (router REDUX vs shuffle, with a bit-for-bit
cross-check against the shuffle version), `var5.cu` (cost of the argmax generic
fallbacks), `var6.cu` (land_situ work mappings).
