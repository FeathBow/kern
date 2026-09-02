// K3 MegaMoE staging: the two kernels that fill a rank's slab before the
// fused kernel runs. Both take slab regions as raw bytes (the slab is one
// `u8` buffer; the offsets come from tools/k3-mega/layout_dump).
//
// Ported from pegainfer csrc/k3/k3_mega_moe_sm100.cu (mega_quant_x_kernel,
// mega_write_routing_kernel), bit-for-bit.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

// bf16 activations -> e4m3 + packed UE8M0 scale factors, written straight
// into the slab's `x` / `x_sf` regions. Per 32-element group: sf =
// ceil_to_ue8m0(max(amax, 1e-4) / 448), exponent clamped to 1..254; four
// groups' exponents pack LSB-first into one i32 per 128 elements. One warp
// per (token, 128-element word); lanes 8g..8g+7 own group g, 4 elements each.
//
//   x       [tokens, x_stride]      bf16
//   x_fp8   [tokens, hidden]        u8 (e4m3)
//   x_sf    [tokens, x_sf_stride]   i32
extern "C" __global__ void kern_k3_mega_quant_x(const __nv_bfloat16* __restrict__ x,
                                                unsigned char* __restrict__ x_fp8,
                                                unsigned char* __restrict__ x_sf_bytes,
                                                int num_tokens, int hidden, int x_stride,
                                                int x_sf_stride) {
  constexpr int kSfGroupK = 32;
  constexpr int kSfWordK = 128;
  int* __restrict__ x_sf = reinterpret_cast<int*>(x_sf_bytes);
  const int words_per_token = hidden / kSfWordK;
  const int warps_per_block = blockDim.x / 32;
  const int warp_id = (blockIdx.x * warps_per_block) + (threadIdx.x / 32);
  const int lane = threadIdx.x % 32;
  const long long total_warps = (long long)num_tokens * words_per_token;
  if (warp_id >= total_warps) return;

  const int token = warp_id / words_per_token;
  const int word = warp_id % words_per_token;
  const __nv_bfloat16* row = x + (size_t)token * x_stride + (size_t)word * kSfWordK;
  unsigned char* out_row = x_fp8 + (size_t)token * hidden + (size_t)word * kSfWordK;

  const int group = lane / 8;
  const int base = (lane % 8) * 4;
  float v[4];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[i] = __bfloat162float(row[group * kSfGroupK + base + i]);
    amax = fmaxf(amax, fabsf(v[i]));
  }
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
  }
  amax = fmaxf(amax, 1e-4f);

  const float raw = amax / 448.0f;
  unsigned int bits = __float_as_uint(raw) & 0x7fffffffu;
  int exp = (int)((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u ? 1 : 0);
  exp = exp < 1 ? 1 : (exp > 254 ? 254 : exp);
  const float sf = __uint_as_float((unsigned int)exp << 23);
  const float inv_sf = 1.0f / sf;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_fp8_storage_t q =
        __nv_cvt_float_to_fp8(v[i] * inv_sf, __NV_SATFINITE, __NV_E4M3);
    out_row[group * kSfGroupK + base + i] = (unsigned char)q;
  }
  const unsigned int my_exp = (unsigned int)exp;
  const unsigned int e0 = __shfl_sync(0xffffffffu, my_exp, 0);
  const unsigned int e1 = __shfl_sync(0xffffffffu, my_exp, 8);
  const unsigned int e2 = __shfl_sync(0xffffffffu, my_exp, 16);
  const unsigned int e3 = __shfl_sync(0xffffffffu, my_exp, 24);
  if (lane == 0) {
    x_sf[(size_t)token * x_sf_stride + word] = (int)(e0 | (e1 << 8) | (e2 << 16) | (e3 << 24));
  }
}

// Routing arrays into the slab: the router's i32 expert ids widen to the
// i64 the mega kernel reads; the weights copy through.
//
//   topk_idx    [entries] i32       dst_idx    [entries] i64 (slab bytes)
//   topk_weight [entries] f32       dst_weight [entries] f32 (slab bytes)
extern "C" __global__ void kern_k3_mega_write_routing(const int* __restrict__ topk_idx,
                                                      const float* __restrict__ topk_weight,
                                                      unsigned char* __restrict__ dst_idx_bytes,
                                                      unsigned char* __restrict__ dst_weight_bytes,
                                                      int entries) {
  long long* __restrict__ dst_idx = reinterpret_cast<long long*>(dst_idx_bytes);
  float* __restrict__ dst_weight = reinterpret_cast<float*>(dst_weight_bytes);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < entries;
       idx += gridDim.x * blockDim.x) {
    dst_idx[idx] = (long long)topk_idx[idx];
    dst_weight[idx] = topk_weight[idx];
  }
}
