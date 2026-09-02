# K4 `kern_k3_mla_prep` — notes

Source: `tools/kernels-src/k3_mla_prep.cu`.
Build (tray03), run (tray07 GPU 3):

```
/usr/local/cuda-13.1/bin/nvcc -cubin -arch=sm_103a -O3 -Xptxas -v \
    -o target/cubins/k3_mla_prep.cubin tools/kernels-src/k3_mla_prep.cu
ssh pod4-gb300-3-tray07-f3 "CUDA_VISIBLE_DEVICES=3 .../harness --kernel mla_prep \
    --cubin .../k3_mla_prep.cubin --B 64 --grid 64,4,1 --block 512,1,1"
```

`-O3` is not load bearing: the cubin from the bare `nvcc -cubin -arch=sm_103a`
that `tools/build_kernels.sh` uses is byte-identical.

## grid / block / smem (for the manifest)

```
grid  (B, 4, 1)     block (512, 1, 1)     dynamic smem 0
static smem 12800 B/block
```

* `blockIdx.y == 0` — the norm/append head: columns 0..2112 of the row, i.e.
  `q_norm`, and the latent row `kv_norm | rope` appended at `slot_mapping[b]`.
* `blockIdx.y == s+1` — gate segment `s`: 4096 of the 12288 gate columns, a
  pure f32 → bf16 landing copy.

`grid.x` is exactly `B`, per §0 of the ABI.

## Design

At B = 1 this kernel moves 86 400 B. That is 11 ns of GB300 HBM. It is not a
bandwidth problem, it is *one memory round trip plus a launch*, so the design
targets (a) enough independent 16-byte loads in flight to cover that one round
trip, (b) as few instructions as possible around them, and (c) more than one
SM even at B = 1.

**Row split.** Splitting a row over 4 blocks (1 head + 3 gate segments) rather
than 1 gives 4 SMs at B = 1 and 256 blocks at B = 64. Measured sweep of
(gate segments × columns-per-thread-per-pass), launch-to-launch µs at B=1/B=64:

| segments | 4 cols/thread | 8 cols/thread |
|---|---|---|
| 2 | 2.157 / 2.376 | 2.158 / 2.484 |
| **3** | **2.156 / 2.322** | 2.158 / 2.463 |
| 4 | 2.148 / 2.389 | 2.154 / 2.469 |
| 6 | 2.151 / 2.420 | 2.145 / 2.415 |
| 8 | 2.149 / 2.386 | 2.150 / 2.441 |
| 12 | 2.143 / 2.462 | 2.144 / 2.502 |
| 16 | 2.149 / 2.720 | 2.149 / 2.733 |
| 24 | 2.146 / 3.103 | 2.147 / 3.141 |

3 gate segments × 4 columns per thread wins; below ~4 segments there is not
enough parallelism at B = 1, above ~8 the tail of tiny blocks costs more at
B = 64 than it buys. (Numbers from an earlier, single-path revision; the shape
choice was re-confirmed against the final kernel — 2 / 3 / 4 / 7 blocks per
row give 2.26 / 2.03 / **2.02** / 2.02 µs at B=1 and 2.29 / 2.32 / **2.25** /
2.53 µs at B=64.)

**Head block thread map** (512 threads, 4 columns each, `float4` loads,
8-byte bf16 stores):

```
t <  384        -> q_norm  column 4*t                 (warps 0..11)
384 <= t < 512  -> kv_norm column 1536 + 4*(t-384)    (warps 12..15)
t <  16         -> also rope column 2048 + 4*t
```

Q_LORA/4 = 384 is a multiple of 32, so the two block reductions split exactly
on a warp boundary: one `__shfl_down` tree, 16 warp partials into shared
memory, **one** `__syncthreads`, then each thread re-sums the 12 (q) or 4 (kv)
partials it needs (`LDS.128`). No second barrier, no separate reduction pass.
`slot_mapping[b]`, the gamma word and the 64 rope columns are all issued
*before* the barrier so their round trips overlap the reduction rather than
queue behind it — moving the gamma load after the barrier costs ~0.5 µs.

**Two paths.** The documented shape (`blockDim.x == 512 && gridDim.y == 4`)
takes a fully specialised path with compile-time trip counts. Any other shape
falls through to a geometry-agnostic path (any `gridDim.y >= 1`, any
`blockDim.x` that is a multiple of 32 in [32, 1024]; with `gridDim.y == 1` the
single block does the head and then the whole gate). Both paths write
identical bits. The fallback exists because the harness's built-in default for
this kernel is grid (B,1,1) block 1024 — the document's suggestion, not this
kernel's formula — and a kernel that silently writes garbage when launched
with the other legal shape is a trap. It costs ~25% (see the table below), so
it is a fallback and not the shape the manifest should use.

The generic path is slower for a boring reason: with a runtime `blockDim.x`
and runtime segment bounds, ptxas cannot unroll the strided loops, so each
iteration is load → wait → store with one request in flight, and the predicates
around every access add instructions to a kernel whose cost *is* instructions.
Recovering it with an explicit N-deep software pipeline (N strided loads
issued before the stores) got it to within 20% but pushed the register count
from 32 to 64, which halves the fast path's occupancy; the plain loop plus a
specialised fast path is both faster and smaller.

## Landing points

The chain follows pegainfer's `land_rms_norm_rbs` + `rms_norm_rbs`
(`pegainfer-k3/kernels/tilelang_defs.py`):

* every f32 partial column lands to bf16 **first**, `x = f32(bf16(P[i]))`,
  including the copies that feed the sums of squares;
* round-before-scale rms: `y = bf16(x * rsqrt(mean(x^2) + 1e-5))`, then
  `y * gamma` as a bf16 × bf16 → bf16 product (`__hmul2` — the exact product of
  two bf16 fits in f32, so this is a single rounding);
* `rope` and `mla_gate` are one bf16 landing, no arithmetic.

Sums of squares accumulate in f32 (per-thread serial, then a warp shuffle
tree); the CPU reference sums serially in double. Measured difference against
`tools/k3-harness/ref.h`: **0 ULP** on every shape below.

## Acceptance

### 1. Harness

`./harness --kernel mla_prep --cubin target/cubins/k3_mla_prep.cubin --B <B>
--grid <B>,4,1 --block 512,1,1 --nmla 4 --layer 2`

| B | q_norm | mla_gate | slab(state) | max ULP | rel RMS |
|---|---|---|---|---|---|
| 1 | PASS | PASS | PASS | 0.00 | 0.000e+00 |
| 2 | PASS | PASS | PASS | 0.00 | 0.000e+00 |
| 8 | PASS | PASS | PASS | 0.00 | 0.000e+00 |
| 64 | PASS | PASS | PASS | 0.00 | 0.000e+00 |

Bit-identical to the reference, including the 576-wide rows written into the
paged slab (the harness compares the whole `slab(state)` buffer, so the
untouched rows are checked too).

Also PASS, via the fallback path, at the harness's built-in default geometry
(grid (B,1,1) block 1024) for B ∈ {1,2,8,64}, and across a 5 × 5 sweep of
`gridDim.y ∈ {1,2,3,5,33}` × `blockDim.x ∈ {32,128,256,512,1024}` at B = 8.
A private test (`tools/k3-scratch/k4/test_mla_prep.cu`, its own CPU reference)
also passes with the partial scaled by 1e-4, 1, 40 and 4000.

### 2. SASS

```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'kern_k3_mla_prep' for 'sm_103a'
ptxas info    : Function properties for kern_k3_mla_prep
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 32 registers, used 1 barriers, 12800 bytes smem
```

`cuobjdump -res-usage`: `REG:32 STACK:0 SHARED:13824 LOCAL:0`.
`cuobjdump -sass | grep -c MULTICAST` → **0**; `grep -cE 'LDL|STL'` → **0**.
`__launch_bounds__(1024)` (the largest legal block for the fallback path) caps
registers at 64; the kernel uses 32, so the fast path gets the full 4 blocks
per SM that its 512-thread blocks allow.

All global accesses are wide: `LDG.E.128` for the f32 partial (float4),
`LDG.E.64` for gamma, `STG.E.64` for every bf16 output. Getting there needed
one non-obvious thing — bf16 pairs are carried as raw 32-bit words
(`pack2` / `mul2` bit-cast around `__nv_bfloat162`), because building the
stored value out of `__nv_bfloat162` struct members leaves the halves in
non-consecutive registers and ptxas emits four 32-bit `STG.E` per 16 bytes
instead of one `STG.E.128`.

### 3. ncu

`ncu --set full -k regex:kern_k3_mla_prep -c 1` at B = 64, grid (64,4,1),
block 512 → `tools/k3-harness/reports/k3_mla_prep.ncu.txt`.

| metric | value |
|---|---|
| Duration | 4.93 µs *(see caveat)* |
| Elapsed cycles / SM freq | 10 057 cyc @ 2.04 GHz |
| DRAM throughput | 753 GB/s = **9.61 %** of 8 TB/s |
| Compute (SM) throughput | 5.27 % |
| Achieved occupancy | 37.05 % (theoretical 100 %) |
| L2 hit rate | 2.43 % |
| L1/TEX hit rate | 0.53 % |
| Registers / thread | 32 |
| Static smem / block | 12.80 KB |
| Waves per SM | 0.42 |

**The two low numbers are both artefacts of the problem size, not of the
kernel.**

*Occupancy 37 %.* At B = 64 the whole launch is 256 blocks over 148 SMs —
0.42 waves. There is less than one block per SM slot to fill; theoretical
occupancy is 100 % and there is nothing to raise. Raising it would mean fewer,
larger blocks, which is exactly the direction the sweep above shows is slower.

*DRAM 9.6 % of peak.* Whole-kernel traffic at B = 64 is 5.53 MB, which is
0.7 µs of GB300 HBM. Nothing can be bandwidth bound at that size — and ncu's
`Duration` for a kernel this small is dominated by profiling overhead: ncu
reports 4.64 µs at **B = 1**, where the kernel reads 82 KB, so roughly 4 µs of
that 4.93 µs is instrumentation. The access *pattern* is provably optimal:
`dram__bytes_read.sum` = 3.71 MB against an ideal 3.69 MB (1.005×), and
`dram__bytes_write.sum` = 0 (the 1.8 MB of output is absorbed by L2 and never
reaches DRAM inside the kernel).

The honest bandwidth measurement is a back-to-back launch loop
(`tools/k3-scratch/k4/test_mla_prep.cu`, 2000 launches) minus the measured
empty-kernel launch floor on the same grid
(`tools/k3-scratch/k4/nullbench.cu`, 1.45–1.51 µs on this device):

| B | launch-to-launch | − launch floor | ideal bytes | achieved | % of 8 TB/s |
|---|---|---|---|---|---|
| 1 | 1.99 µs | ~0.50 µs | 0.086 MB | 173 GB/s | 2 % |
| 2 | 1.98 µs | ~0.49 µs | 0.173 MB | 353 GB/s | 4 % |
| 8 | 1.99 µs | ~0.50 µs | 0.691 MB | 1.38 TB/s | 17 % |
| 64 | 2.23 µs | ~0.77 µs | 5.53 MB | **7.18 TB/s** | **90 %** |
| 128 | 2.62 µs | ~1.13 µs | 11.06 MB | 9.8 TB/s | L2-assisted |
| 512 | 5.13 µs | ~3.64 µs | 44.24 MB | 12.2 TB/s | L2-assisted |

Ideal bytes = B × (14400×4 read + (1536 + 12288 + 576)×2 written) = B × 86 400.
B ≤ 64 is the ABI's range; B = 128/512 are only there to show the kernel keeps
scaling — beyond B ≈ 128 the working set is L2 resident in a repeated loop, so
those rows exceed HBM peak and are not a DRAM measurement (ncu with a cold L2
at B = 512 reads 29.52 MB from DRAM against an ideal 29.49 MB, i.e. ~8.1 TB/s
of real DRAM read over the same 3.64 µs).

So: at the batch sizes that matter the kernel reaches ~90 % of peak
(B = 64) and is a bare memory round trip (B = 1). The ≥ 60 %-of-peak target is
met at B = 64 and is not meaningful at B ≤ 8, where the whole row is 86 KB.

**Against `tools/k3-harness/naive/mla_prep.cubin`** (same back-to-back loop):

| | B=1 | B=64 |
|---|---|---|
| naive, grid (B,1,1) block 1024 | 5.58 µs | 5.73 µs |
| this kernel, same geometry (fallback path) | 4.61 µs | 4.68 µs |
| this kernel, grid (B,4,1) block 512 | **1.99 µs** | **2.23 µs** |

i.e. 2.8× / 2.6× launch-to-launch, or 8.2× / 5.5× once the 1.5 µs launch floor
is subtracted from both.

**Launch count.** This replaces 7 launches (2 land + 2 rms + rope land +
kv_append + gate land) with 1. Per MLA layer that is 6 launches saved; the
7 → 1 fusion is worth more than the kernel's own 0.5–0.8 µs, since each
saved launch costs ~1.5 µs of launch-to-launch time on this device.

## Things not done / caveats

1. **Harness build, early in the day.** The first drop of `harness.cu` did not
   compile under CUDA 13.1: `cuCtxCreate(&ctx, 0, dev)` resolves to
   `cuCtxCreate_v4(CUcontext*, CUctxCreateParams*, unsigned, CUdevice)` in 13.1
   and failed with "too few arguments in function call". That was fixed
   upstream while I was working; every number above is from the current,
   **unmodified** `tools/k3-harness/harness.cu` + `ref.h`, built into
   `tools/k3-scratch/k4/harness` (I did not write into `tools/k3-harness/`
   outside `notes/` and `reports/`).
2. **The harness's `TIME` line is not a kernel time.** `time_and_report`
   brackets *one* launch with two events and a `cuEventSynchronize` per rep,
   so it measures launch + sync latency: it reports ~6 µs for every B from 1
   to 64, i.e. its floor. The numbers in the table above come from a
   back-to-back launch loop instead. Nothing wrong with the harness for
   pass/fail, but its GB/s column should not be read as bandwidth for kernels
   this small.
3. **Negative `slot_mapping` entries** are treated as "no slot" and the append
   is skipped (`q_norm` and `mla_gate` are still written in full). The ABI does
   not say what a negative slot means; the documented formula would compute a
   negative row offset and write out of bounds, so the reference cannot be
   exercising it either. If the runtime never emits one this is dead code; if
   it does, this is the behaviour it gets.
4. **The `partial` row stride is assumed to be `MLA_FUSED` = 14400 and the
   valid columns to start at 0.** §0 describes f32 partials as `f32 [B, ldc]`
   with `n` valid columns from `off`, but the K4 signature carries neither
   `ldc` nor `off`, so the generic §0 rule ("row stride = width") applies. If
   the generator ever pads `ldc` for this GEMM the kernel needs the extra
   argument. The harness's reference makes the same assumption.
5. **The fallback path is ~25 % slower** than the documented shape (2.7 vs
   2.0 µs launch-to-launch at B = 1). It is a correctness net, not a second
   tuned configuration; the manifest should use grid (B, 4, 1) block 512.
6. Not attempted: a persistent/CUDA-graph-resident variant, or fusing this
   into the `w_q_b` GEMM epilogue. At 0.5–0.8 µs of kernel time against a
   ~1.5 µs launch cost, the remaining win in this kernel is the launch, not
   the kernel — the next real step is making the whole MLA layer one graph,
   which is generator-side work.
