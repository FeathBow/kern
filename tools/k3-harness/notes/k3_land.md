# K6/K7 — `k3_land.cu` (`kern_k3_rms`, `kern_k3_land`, `kern_k3_land_situ`)

Source: `tools/kernels-src/k3_land.cu`
Built: `/usr/local/cuda-13.1/bin/nvcc -cubin -arch=sm_103a -O3` on tray03.
Run/profiled: tray08 GPU 1 (`CUDA_VISIBLE_DEVICES=1`), idle card.

## 1. Status

| check | result |
|---|---|
| harness `--kernel rms` B ∈ {1,2,8,64} (h=3584) | **PASS**, max\|err\| = 0 (bit-exact vs `ref.h`) |
| harness `--kernel land` B ∈ {1,2,8,64} (n=3584, off=128, ldc=4096) | **PASS**, max\|err\| = 0 |
| harness `--kernel land_situ` B ∈ {1,2,8,64} (n=3072) | **PASS**, maxULP = 1.00, relRMS ≤ 2.6e-4 |
| extra shapes from the task brief (scratch tests, see §7) | **PASS** — land n=3584/off=0/ldc=3584 and n=576/off=1536/ldc=14400; land_situ n=6144 and n=33792; rms h=7168 and h=100 (scalar path) |
| `-Xptxas -v` spill / local | **0 bytes** on all three entries |
| `.MULTICAST` in SASS | **0 occurrences** |
| registers vs `__launch_bounds__` | consistent (see §3) |

## 2. Design

All three are pure streams over one row per `blockIdx.x`.

### The vectorisation-vs-grid trade

The document fixes grid `(B, ceil(n/1024))` with block 1024 for `land` and
`land_situ` — i.e. **exactly one element per thread**. Taken literally that
forbids wide accesses. I hand each thread one `float4` (16 B load, 8 B store)
and keep a grid-stride loop, so the same block count still covers the row with
4× fewer and 4× wider memory instructions; bytes in flight per block are
unchanged, instruction count is not. The cost is that three quarters of the
threads (and, when `n/4 < gridDim.y·1024`, most of `grid.y`) find nothing to do.
Any `n`/`off`/`ldc` that is not a multiple of 4 takes a scalar grid-stride
fallback, so the entries stay correct outside the K3 shape set.

Measured alternatives for `land_situ` (graph replay, µs):

| mapping | n=3072 B=1 | n=3072 B=64 | n=6144 B=64 | n=33792 B=1 | n=33792 B=64 |
|---|---|---|---|---|---|
| **float4/thread, grid-stride (shipped)** | 1.82 | **1.91** | **2.64** | 2.02 | **7.36** |
| float4/thread, contiguous segment per block (all `grid.y` busy) | **1.59** | 2.17 | 3.08 | **1.79** | 9.55 |
| scalar, 1 element/thread (the naive mapping) | 1.44 | 2.26 | 2.95 | 1.62 | 10.06 |

Spreading the row over every `grid.y` block wins ~0.2–0.4 µs at B=1 (more SMs
touched, shorter critical path) and loses 2.2–2.7 µs at B=64 (each block then
has only ~8 KB in flight instead of ~32 KB, so a quarter of the memory-level
parallelism). I took the B=64-optimal mapping. **This is a real, measured
trade-off, not an oversight** — if B=1 tail latency turns out to matter more,
swap in the segmented mapping; both are ~10 lines.

### `kern_k3_rms` — grid (B,1,1), block 1024, 132 B static smem, 0 dynamic

`h % 8 == 0` (h = 3584 and h = 7168 both qualify) takes a `uint4` path — 8 bf16
per access. Pass 1 loads up to `RMS_REGS(4) × blockDim.x` vectors of **x and
gamma** into registers and accumulates `Σ f32(x)²`; the block reduction is a
5-step shuffle, one f32 per warp through smem, warp 0 finishing. Pass 2 lands
`bf16(x·rsqrt(mean+1e-5))` and multiplies by the already-resident gamma with
`__hmul2` — bf16 × bf16 → bf16, the document's two landings, in that order.
Rows longer than 32768 stream the overflow twice, out of L1. No register array
is dynamically indexed. Non-multiple-of-8 `h` takes a scalar two-pass path
(tested at h = 100).

**Prefetching gamma in pass 1 is the whole optimisation.** Loading gamma after
the reduction serialises a second DRAM round trip behind it; loading it up front
hides that trip inside the reduction: **1.93 → 1.71 µs at B=1, 2.19 → 1.98 µs at
B=64.** The cost is 64 registers instead of 56 — irrelevant here, because the
grid is only `B` blocks and never fills the machine anyway.

### `kern_k3_land` — grid (B, ceil(n/1024)), block 1024, 0 smem

`o[b,i] = bf16(p[b·ldc + off + i])`. One `float4` load, two
`__floats2bfloat162_rn`, one 8 B store. The vector path needs `off % 4 == 0` and
`ldc % 4 == 0` for the source to stay 16 B aligned, and `n % 4 == 0` for the
destination to stay 8 B aligned; all four shapes in play (the harness's
n=3584/off=128/ldc=4096 and the brief's two) satisfy that.

### `kern_k3_land_situ` — grid (B, ceil(n/1024)), block 1024, 0 smem

Gate at `p[b·2n + i]`, up at `p[b·2n + n + i]`; both land to bf16 **before** the
activation, then `act = bf16(4·tanh(g/4)·σ(g)·25·tanh(u/25))`.

This kernel is **transcendental-bound, not bandwidth-bound** — three special
functions per element, and ncu shows Compute (SM) Throughput 34.6 % against DRAM
18.8 %. `tanhf` and `expf` are the accurate libm implementations, tens of
instructions each. Replacing them with the hardware `tanh.approx.f32` (one SFU
instruction, via inline PTX) and `__expf`/`__frcp_rn` took n=33792 B=64 from
**11.2 µs to 6.8 µs**, and with the occupancy fix below the shipped kernel is at
**7.37 µs including the launch node**.

I measured `tanh.approx.f32` rather than trusting the PTX ISA's stated 2⁻¹¹
bound: over 2²⁰ points on sm_103a,

| range | max absolute error | max relative error (|tanh| > 1e-3) |
|---|---|---|
| \|x\| ≤ 1 | 7.10e-6 | 1.06e-5 |
| \|x\| ≤ 4 | 7.83e-6 | 1.05e-5 |
| \|x\| ≤ 8 | 7.77e-6 | 1.05e-5 |

i.e. ~2⁻¹⁷ — about 9 bits below one bf16 ULP even after the ×25 scale, which is
why the harness reports maxULP = 1.00 (single round-to-nearest flips near bf16
midpoints) and relRMS 1.5e-4 against a 2e-3 budget. If bit-tightness against the
CPU reference is ever wanted, `tanh_approx` → `tanhf` and `__expf` → `expf` in
`situ_f` is a two-line revert costing ~1.7×.

A third formulation — rewriting both tanh's in terms of a single `__expf` each,
`tanh(x) = (1-e^{-2x})/(1+e^{-2x})` — was tried and is slower (7.52 µs vs 6.81
at n=33792 B=64) because the two extra reciprocals cost more than the tanh SFU
op saves.

**`__launch_bounds__(1024, 2)` matters here too**: at 36 registers only one
1024-thread block fits per SM (2048 threads/SM is the cap) and n=33792 B=64 runs
**12.5 µs**; at 29 registers two fit and it runs **7.37 µs**.

## 3. `ptxas -v`

```
ptxas info    : Compiling entry function 'kern_k3_land_situ' for 'sm_103a'
ptxas info    : Function properties for kern_k3_land_situ
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 29 registers, used 0 barriers
ptxas info    : Compiling entry function 'kern_k3_land' for 'sm_103a'
ptxas info    : Function properties for kern_k3_land
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 32 registers, used 0 barriers
ptxas info    : Compiling entry function 'kern_k3_rms' for 'sm_103a'
ptxas info    : Function properties for kern_k3_rms
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 64 registers, used 1 barriers, 132 bytes smem
```

`__launch_bounds__`: `land_situ` (1024, 2) → 29 regs, 2 blocks/SM, 100 %
theoretical occupancy; `land` (1024, 1) → 32 regs, 2 blocks/SM, 100 %
theoretical; `rms` (1024, 1) → 64 regs, 1 block/SM, 50 % theoretical — deliberate
(grid is `B` ≤ 64 blocks, so occupancy is irrelevant and the extra registers buy
the gamma prefetch).

## 4. Timing

* **graph replay** — 200 kernel nodes in one CUDA graph, replayed 5×. A null
  kernel measures **0.525 µs** in the same harness; that per-node floor is
  included in every number below.
* **harness median** — `./harness --reps 50`, one event pair per launch, which
  carries a ≈5.5 µs host-launch floor on this box. Useful only for like-for-like
  comparison against `naive/`.

Graph replay, µs:

| entry | shape | B=1 | B=2 | B=8 | B=64 |
|---|---|---|---|---|---|
| `rms` | h=3584 | **1.71** | 1.75 | 1.75 | **1.98** |
| `land` | n=3584 off=0 ldc=3584 | **1.39** | 1.42 | 1.41 | **1.50** |
| `land` | n=576 off=1536 ldc=14400 | **1.22** | 1.25 | 1.25 | **1.48** |
| `land_situ` | n=6144 | **1.99** | 2.02 | 2.02 | **2.64** |
| `land_situ` | n=33792 | **2.02** | 2.04 | 2.16 | **7.37** |

Harness medians vs the naive references (same shapes, same card):

| entry | mine B=1 | naive B=1 | mine B=64 | naive B=64 |
|---|---|---|---|---|
| `rms` (h=3584) | 6.78 | 6.59 | 6.75 | 7.81 |
| `land` (n=3584, off=128, ldc=4096) | 6.62 | 5.66 | 7.01 | 6.34 |
| `land_situ` (n=3072) | 6.85 | 5.82 | 6.75 | 6.27 |

At the harness's shapes all three kernels are far below the ≈5.5 µs launch
floor, so those columns are measuring the timer, not the kernel — the graph
numbers and the ncu cycle counts are the real ones. (The naive kernels' small
edge at B=1 is the same mapping effect tabulated in §2; at the shapes that
actually cost time — n=33792 — the shipped mapping is 27 % faster than the naive
one.)

Achieved bandwidth (roofline bytes ÷ graph time minus the 0.525 µs node floor):

| entry | B=64 traffic | device time | GB/s | % of 8 TB/s |
|---|---|---|---|---|
| `rms` h=3584 | 924 KB | 1.45 µs | 640 | 8 % |
| `land` n=3584 | 1.38 MB | 0.97 µs | 1420 | 18 % |
| `land` n=576 off=1536 | 221 KB | 0.95 µs | 233 | 3 % |
| `land_situ` n=6144 | 3.93 MB | 2.11 µs | 1860 | 23 % |
| `land_situ` n=33792 | 21.6 MB | 6.84 µs | **3160** | **40 %** |

## 5. ncu

`tools/k3-harness/reports/k3_land.ncu.txt` — `--set full --clock-control none`,
B=64, warm launch (`--launch-skip 4`), driver `tools/k3-scratch/k6/prof_k6 64 5`
(`land_situ` at n=33792, `land` at n=3584/off=0/ldc=3584, `rms` at h=3584).

| metric | `land_situ` n=33792 | `land` n=3584 | `rms` h=3584 |
|---|---|---|---|
| Duration (ncu) | 11.68 µs | 5.18 µs | 5.92 µs |
| DRAM Throughput | **18.78 %** | 2.27 % | 1.05 % |
| Memory Throughput | 1.48 TB/s | 178 GB/s | 82.6 GB/s |
| Compute (SM) Throughput | **34.61 %** | 7.97 % | 4.29 % |
| L2 Hit Rate | 0.38 % | 2.29 % | 14.75 % |
| Achieved / Theoretical Occupancy | 71.4 % / 100 % | 68.6 % / 100 % | 48.0 % / 50 % |
| Elapsed Cycles @ 2.06 GHz | 37286 | 11815 | 12109 |

**ncu's Duration includes ≈4–5 µs of per-launch instrumentation on this box** —
`land` moves 1.4 MB and cannot take 5.18 µs; it measures 1.50 µs under graph
replay. The DRAM percentages scale with that: `land_situ` moves the same bytes
in 6.84 µs, i.e. 3.16 TB/s / 40 % of peak, not 18.8 %. The ratio that is *not*
distorted is Compute vs Memory throughput, and it says plainly that `land_situ`
is SFU-bound (34.6 % compute vs 18.8 % memory), which is what the `tanh.approx`
work was chasing.

## 6. What was not achieved, and why

**§2 asks memory-dominated kernels for ≥ 60 % of peak. None of these three
reach it; only `land_situ` at its largest shape gets close (40 %).**

* `rms` (h=3584) and `land` (n=3584 / n=576) move **0.2–1.4 MB at B=64**. The
  document pins `rms` to grid `(B,1,1)` — 64 blocks on 148 SMs, so more than half
  the machine is idle by construction, and the kernel is two dependent DRAM
  round trips (load the row, reduce, write it) no matter what. 0.97–1.45 µs of
  device time against a ~0.6 µs round-trip latency and a 0.53 µs launch node is
  essentially the floor. Bandwidth is not a meaningful target at these sizes;
  the meaningful target is latency, and that is what the gamma prefetch bought.
* `land_situ` at n=33792, the only shape here with enough bytes to talk about
  bandwidth, is **not memory-bound in the first place** — three transcendentals
  per element put it at 34.6 % compute vs 18.8 % memory in ncu. Getting from
  1.7 TB/s to 3.2 TB/s took removing library `tanhf`/`expf` and fixing
  occupancy; going further means removing SFU work, not memory work, and I do
  not see a cheaper exact formulation than one `tanh.approx` per tanh plus one
  `__expf` for the sigmoid. (`tanh(x) = 2σ(2x) − 1` would let `σ(g)` and
  `4·tanh(g/4)` share an exponential, but they need `e^{-g}` and `e^{-g/2}`
  respectively; sharing via `t = e^{-g/2}`, `e^{-g} = t²` is exactly the
  reformulation measured in §2 and it is slower.)
* The other structural limit is the one in §2: the document's
  `(B, ceil(n/1024))` × 1024 grid is sized for one element per thread. With
  `float4`s only a quarter of the launched threads are used, and at n=33792 only
  9 of the 33 blocks per row have work. Grid `(B, ceil(n/4096))` would express
  the shipped mapping exactly and drop 3/4 of the block launches. That is a
  manifest-formula change, so I did not take it unilaterally — see ambiguity 3.

## 7. ABI ambiguities / notes for the document owner

1. **The delivered `land`/`land_situ` shapes disagree between my brief and the
   harness.** My task named `land` at (n=3584, off=0, ldc=3584) and (n=576,
   off=1536, ldc=14400), and `land_situ` at n=6144 and n=33792; the harness pins
   `land` at (n=3584, off=128, ldc=4096) and `land_situ` at n = INTER = 3072.
   All of them pass — the harness cases through `./harness`, the brief's cases
   through `tools/k3-scratch/k6/test_k6.cu` — but the document should name the
   real call sites, because they drive the alignment assumptions (`off % 4`,
   `ldc % 4`) and the mapping trade-off in §2.
2. **`land_situ` has no `ldc`/`off`, contradicting §0.** §0 says every f32
   partial is `[B, ldc]` with `n` valid columns from `off`, but `land_situ`'s
   signature hard-codes `ldc = 2n`, `off = 0` — only `kern_k3_land` takes the
   two. Same point the harness raises as its open question 6. If a GEMM ever
   hands `land_situ` a padded partial, the signature has to grow.
3. **The `(B, ceil(n/1024))` block-1024 grid formula forces one element per
   thread.** It is the only shape in the K6/K7 set where the document's launch
   geometry costs performance rather than describing it. `(B, ceil(n/4096))`
   with block 1024 (one `float4` per thread) is what the shipped kernel actually
   wants; the kernel is grid-stride and correct under either, so this is purely
   a matter of not launching 4× the blocks. Happy to change it if the manifest
   can move.
4. **`rms`'s landing chain vs K3's output norm.** §0's `rms()` rounds before the
   scale and multiplies two bf16s, which is what `kern_k3_rms` implements
   (bit-exact against `ref.h` at every B). Worth noting alongside the harness's
   open question 2 that K3's `o[d]` norm is a *different* primitive with the same
   name in the prose — one landing after a f32 `gamma_o` multiply. They should
   not both be called "rms" in the document.
5. **`land_situ`'s pre-activation landing is inferred, not stated.** §K7 writes
   `gate = f32(bf16(p[...]))`, so both operands land to bf16 before `situ`. That
   is what I implemented and what `ref.h` does, but §0's landing-point rule
   ("f32 accumulate, land where each kernel says") reads as if the landing were
   only on the output. The explicit `f32(bf16(...))` in K7 is the thing to keep;
   it is easy to lose in a paraphrase.

## 8. Reproducing

```bash
# build
/usr/local/cuda-13.1/bin/nvcc -cubin -arch=sm_103a -O3 -Xptxas -v \
  -o target/cubins/k3_land.cubin tools/kernels-src/k3_land.cu
/usr/local/cuda-13.1/bin/cuobjdump -sass target/cubins/k3_land.cubin | grep -c MULTICAST   # 0

# harness (tray08 GPU 1)
cd tools/k3-harness
for k in rms land land_situ; do for B in 1 2 8 64; do
  CUDA_VISIBLE_DEVICES=1 ./harness --kernel $k \
    --cubin ../../target/cubins/k3_land.cubin --B $B || exit 1
done; done

# the brief's shapes + graph-replay timings
cd tools/k3-scratch/k6
/usr/local/cuda-13.1/bin/nvcc -arch=sm_103a -O3 -o test_k6 test_k6.cu
/usr/local/cuda-13.1/bin/nvcc -arch=sm_103a -O3 -o bench_k6 bench_k6.cu
CUDA_VISIBLE_DEVICES=1 ./test_k6 && CUDA_VISIBLE_DEVICES=1 ./bench_k6
```

`tools/k3-scratch/k6/var2.cu` holds the `tanh.approx` accuracy sweep, `var.cu`
the situ math variants, and `var6.cu` the `land_situ` work-mapping comparison in
§2.
