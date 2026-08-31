// Gated attention output, bit-exact with vLLM's eager
// `attn_output * torch.sigmoid(gate)` (two ATen ops, two bf16 roundings):
//   g = bf16(1 / (1 + expf(-f32(gate))))     (ATen sigmoid, opmath f32)
//   out = bf16(f32(attn) * f32(g))           (ATen mul on bf16)
// attn / out are contiguous [tokens, heads*head_dim]; gate is a strided view
// (per-head [q | gate] halves of the fused qkv projection):
//   gate(t, h, d) = gate + t*gate_tstride + h*gate_hstride + d
//
//   nvcc -cubin -arch=sm_103a -o kernels/sigmoid_mul.cubin tools/kernels-src/sigmoid_mul.cu
#include <cuda_bf16.h>

extern "C" __global__ void kern_sigmoid_mul_bf16(
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ attn,
    const __nv_bfloat16* __restrict__ gate, int heads, int head_dim,
    int gate_tstride, int gate_hstride) {
  const long long t = blockIdx.x;
  const int n = heads * head_dim;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    const int h = j / head_dim, d = j % head_dim;
    const float x = __bfloat162float(gate[t * gate_tstride + (long long)h * gate_hstride + d]);
    const float sg = 1.0f / (1.0f + expf(-x));
    const float g = __bfloat162float(__float2bfloat16_rn(sg));
    const float a = __bfloat162float(attn[t * n + j]);
    out[t * n + j] = __float2bfloat16_rn(a * g);
  }
}
