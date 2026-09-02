# K2 `kern_k3_conv_silu` — notes

Source: `tools/kernels-src/k3_conv_silu.cu`.
Machine: tray07 GPU 1 (NVIDIA GB300, driver 610.57.04), nvcc/ncu from
`/usr/local/cuda-13.1/bin`. Build:

```
/usr/local/cuda-13.1/bin/nvcc -cubin -arch=sm_103a -O3 tools/kernels-src/k3_conv_silu.cu
```

## Launch formula (for the manifest)

```
grid  (B, 3, INNER / (BLOCK*VEC)) = (B, 3, 24)
block (128, 1, 1)
smem  0
```

`blockIdx.x` = row (exactly `B`, never `ceil(B/k)`), `blockIdx.y` = stream
(0=q, 1=k, 2=v), `blockIdx.z` = column segment. `BLOCK = 128`, `VEC = 4`
columns per thread → 512 columns per block, 24 segments over `INNER = 12288`.

This is **not** the document's default `(B, 3, INNER/256) x 256`, so the
harness must be given the launch explicitly:

```
./harness --kernel conv_silu --cubin <path> --B <B> --grid <B>,3,24 --block 128,1,1
```

## Design

One thread owns 4 consecutive columns of one stream of one row, for the whole
transaction. It:

1. reads the three window taps (`3 x uint2` = 3 x 8 B bf16),
2. reads the f32 partial (`float4`) and the four conv taps (`4 x float4`),
3. lands `x = bf16(partial)` once — that same bf16 is both the 4th conv tap and
   the value pushed into the window,
4. **stores** the shifted window (`win[0]=w1, win[1]=w2, win[2]=x`) — all three
   taps were already in registers, so the in-place update needs no barrier and
   no cross-thread ordering; column ownership is exclusive,
5. accumulates `y` in f32 (one FMUL + three FFMA, in tap order 0,1,2,x — the
   same order as pegainfer's `conv_silu_batched`), lands `sb = bf16(y)`,
   evaluates `sb * (1/(1+expf(-sb)))` in f32 and lands the result in bf16.

Landing points therefore match `conv_silu_batched` in
`pegainfer-k3/kernels/tilelang_defs.py` exactly: bf16 on the merged partial,
bf16 on the conv sum, bf16 on the SiLU output, f32 everywhere between.
`expf` is the precise libdevice one, not `__expf` (`__expf` measured
identically fast and gave identical bf16 results, but there is no reason to
take the approximation).

Addressing: every access is 8 B or 16 B aligned by construction —
`REC_BYTES = 6291456`, the 73728 B window stride, `LINE_BYTES = 6512640` and
`KDA_FUSED*4 = 196608` are all multiples of 16, and `c` is a multiple of 4.
A warp moves 512 B (f32x4) / 256 B (bf16x4) contiguous per instruction. The
SASS is 5 `LDG.E.128.CONSTANT` + 3 `LDG.E.64` + 4 `STG.E.64` + 1 scalar
`LDG.E.CONSTANT` (the `line_index[b]` lookup) per thread — nothing else.

### Tiling sweep (harness median, B=64, 200 reps)

| block x VEC | grid.z | median us |
|---|---|---|
| 128 x 4 | 24 | **10.43** |
| 96 x 4 | 32 | 10.43 |
| 192 x 4 | 16 | 10.46 |
| 256 x 4 | 12 | 10.50 |
| 128 x 2 | 48 | 10.62 |
| 128 x 8 | 12 | 10.85 |
| 256 x 8 | 6 | 12.29 |
| 64 x 2 | 96 | 15.10 |
| 256 x 1 (the document's default tiling) | 48 | 14.14 |

`VEC=8` costs 64 registers (vs 40) and drops to 8 resident blocks/SM;
`BLOCK=64` pays too much block-dispatch. 128x4 and 96x4 are a tie; 128 was
kept as the more conventional shape. The 4-wide accesses are worth 1.35x over
the document's scalar default, which is the reason for the deviation.

## ptxas -v

```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'kern_k3_conv_silu' for 'sm_103a'
ptxas info    : Function properties for kern_k3_conv_silu
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 0 barriers
```

0 spill, 0 local, 0 stack. `__launch_bounds__(128)` with ncu reporting
`Registers Per Thread = 40` — consistent. `cuobjdump -sass` contains no
`.MULTICAST`, no `LDL`/`STL`, no `LDSM`.

(Forcing higher occupancy with `__launch_bounds__(128, 14)` or `(128, 16)`
does fit 32 registers but spills 10/12 bytes, which the acceptance rules out,
and it was not faster.)

## Harness results

`tools/k3-harness/harness.cu` @ 2026-09-02 08:44, `--reps 200`, seed 1234.

| B | conv_q maxULP | conv_k maxULP | conv_v maxULP | max relRMS | window state | median us |
|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 | bit-exact | 6.34 |
| 2 | 0 | 0 | 0 | 0 | bit-exact | 6.46 |
| 8 | 1 | 4 | 1 | 6.6e-6 | bit-exact | 6.27 |
| 64 | 2 | 2 | 2 | 1.1e-5 | bit-exact | 10.46 |

All PASS. The window state (`win(state)`, 7.08 M bf16 at B=64) is bit-identical
to the reference after the shift, at every B. The one ULP=4 element (B=8,
conv_k) has `max|err| = 4.9e-4`, inside the `3 ULP + 1e-3` tolerance — it comes
from the reference accumulating the conv sum in `double` while the kernel uses
f32 FMA, so a value landing near a bf16 tie can round the other way. Relative
RMS is 1e-5 at worst, 200x under the 2e-3 bound.

Note the harness times with a `cuEventSynchronize` per launch, so ~5 us of
launch+sync latency is inside every number above (B=1 moves 1.25 MB and still
reports 6.3 us). Back-to-back on one stream the same launches are **2.21 us at
B=1** and **6.22 us at B=64**; an empty kernel on the B=64 grid is 3.17 us, so
at B=64 the kernel is roughly 2x its own dispatch floor.

## Roofline and achieved bandwidth

Per row: partial `3 x 12288 x 4` = 147 456 B read, window `3 x 3 x 12288 x 2`
= 221 184 B read **and** written, output `3 x 12288 x 2` = 73 728 B written —
648 KB/row. `cw` is `3 x 4 x 12288 x 4` = 589 824 B, weight-shared, read once
from DRAM and served from L2 to the other rows. At B=64 that is
**43.06 MB total = 24.19 MB read + 18.88 MB written** (the harness's own
roofline figure agrees: 43.06 MB).

`ncu --metrics dram__bytes_read.sum` at B=64 reports **24.19 MB — exactly the
compulsory read count, i.e. zero over-fetch**; `lts__t_sectors_lookup_hit` /
`miss` show the `cw` reuse working (0.97 M hits vs 1.42 M misses, L2 hit rate
40.7%; the 63 rows after the first take `cw` out of L2).

### The ncu number at B=64, and why it is 30.8% and not >= 60%

`--set full -c 1` at B=64 reports:

| metric | value |
|---|---|
| Duration | 9.92 us |
| DRAM Throughput | 30.83 % (2.44 TB/s) |
| L1/TEX hit rate | 19.76 % |
| L2 hit rate | 40.70 % |
| Achieved occupancy | 59.98 % (theoretical 75 %, block limit = registers, 12 blocks/SM) |
| Grid / waves per SM | 4608 blocks / 2.53 |
| Registers per thread | 40 |

Two things make that 30.8% not the throughput of the kernel:

1. **`dram__bytes_write.sum` is 0.** The 18.88 MB of stores are absorbed by L2
   and written back *after* the kernel retires, so ncu's DRAM counter only ever
   sees the 24.19 MB of reads. Even a kernel running at a perfect 8 TB/s of
   real traffic would be reported here as `24.19/43.06 x 100 = 56%`. The
   metric is structurally capped below the 60% target for this access pattern.
2. **43 MB is too small a launch to reach peak on this part.** Cold-cache
   duration is ~10 us but an *empty* kernel on the same grid is 3.2 us, and
   `--cache-control none --launch-skip 100` (L2 warm, `dram__bytes_read` down
   to 3.3 KB, i.e. no DRAM traffic at all) still measures 9.7 us under ncu /
   6.2 us on cudaEvent. So at B=64 the kernel is bound by launch ramp and
   L1TEX latency, not by HBM. ncu says so directly: the top stall is
   "11.0 cycles waiting on an L1TEX scoreboard dependency ... memory-latency-
   bound, which is not necessarily memory-bandwidth-bound".

To show the kernel itself is not leaving bandwidth on the table, the same
cubin was run at **B=1024** (648.6 MB of traffic, far past L2, so cold and warm
agree to 1%):

| | us | r+w GB/s | % of 8 TB/s spec |
|---|---|---|---|
| K2 @ B=1024, warm | 103.7 | 6557 | 82.0 % |
| K2 @ B=1024, L2 flushed before every launch | 102.6 | 6630 | 82.9 % |
| machine reference: `float4` copy 2+2 GB | — | 6924 | 86.5 % |
| machine reference: `float4` read 4 GB | — | 7087 | 88.6 % |

So the kernel sustains **83% of the 8 TB/s spec and 96% of what a plain
`float4` copy reaches on this GPU** once the problem is large enough for that
to be the limit. The B=64 shortfall is the launch, not the access pattern.

## What was not achieved / open items

- **The `>= 60% of 8 TB/s` target is not met by the literal ncu DRAM metric at
  B=64 (30.8%)**, for the two reasons above. The honest per-launch numbers at
  B=64 are: 43.06 MB moved in 6.22 us back-to-back (6.9 TB/s, but L2-assisted)
  or in ~10 us cold (4.3 TB/s counting reads+writes). I did not find a tiling
  that improves this; the read stream is already exactly compulsory and the
  stores are already fully coalesced 8 B/thread.
- **`cw` costs 37.7 MB of L1->L2 read traffic at B=64** (0.59 MB of unique
  data, amplified 64x) because the ABI fixes `grid.x = B`: a block sees one row,
  so the weight cannot be reused across rows inside a block. Consecutive
  `blockIdx.x` values do share a `cw` slice, but they are dispatched to
  different SMs, so L1 cannot catch it either (L1 hit rate 19.8%). Letting one
  block cover several rows (`grid.x = ceil(B/k)`) would cut this ~8x, but the
  document forbids it ("grid.x 必须恰好是 B"). Worth revisiting if the generator
  can ever pass a row count.
- Occupancy is register-limited at 12 blocks/SM (75% theoretical, 60% achieved).
  `__launch_bounds__(128,14/16)` reaches 32 registers but spills, which the
  acceptance forbids, and it was not faster anyway.

## Document / harness observations

1. **`tools/k3-harness/harness.cu` does not compile under CUDA 13.1**
   (line 944): `cuCtxCreate(&ctx, 0, dev)` — in CUDA 13 the unversioned
   `cuCtxCreate` is `_v4`, which takes `(CUcontext*, CUctxCreateParams*,
   unsigned flags, CUdevice)`. `cuCtxCreate(&ctx, nullptr, 0, dev)` fixes it.
   I did not edit the harness; the runs above used a locally patched copy
   (`tools/k3-scratch/k2/harness_local`, built from a scratch copy with only
   that one line changed). Everything else was used verbatim.
2. The document's K2 default tiling `grid (B, 3, INNER/256), block 256` only
   admits one column per thread, i.e. scalar 4 B accesses; it is 1.35x slower
   than 4-wide. The document does allow "或你自己的切分" and the manifest copies
   the header formula, so this should be fine — flagging it because the
   harness's built-in default geometry for `conv_silu` is the document one, and
   anyone running this cubin without `--grid/--block` would read out of bounds.
3. The document writes the conv sum as `Σ_{t<3} f32(win_s[t][c])·cw[s][t][c] +
   f32(x)·cw[s][3][c]` without fixing the summation order; `conv_silu_batched`
   accumulates t=0,1,2 then the x term, and `ref.h` accumulates in `double`.
   The kernel follows the TileLang order in f32. This is the only source of the
   1-4 ULP differences and is well inside tolerance, but it is why the result is
   not bit-exact against `ref.h` (it *is* bit-exact against an f32-FMA
   reference — see `tools/k3-scratch/k2/test_conv_silu.cu`, max|err| = 0 at
   every B).
4. The ABI does not say whether `partial` may alias `gate_partial` (K3 reads
   band 3 of the same buffer). K2 only reads bands 0..2 and never writes
   `partial`, so it is safe either way.

## Scratch

`tools/k3-scratch/k2/` holds the pre-harness driver (`test_conv_silu.cu`:
driver-API cubin load, f32-FMA CPU reference, output + window-state check,
back-to-back and L2-flushed timing, empty-kernel floor), the streaming
bandwidth reference (`bwprobe.cu`), the tiling-sweep cubins, and the locally
patched harness binary. Nothing there is a deliverable.
