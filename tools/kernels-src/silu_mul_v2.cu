// Hand-written replacement for the mined vLLM act_and_mul kernel.
// Interface contract (see manifest `silu_mul`): out[t][i] =
// silu(in[t][i]) * in[t][d+i], in laid out as [tokens, 2*d].
// This ABI drops the three trailing mined-ABI params (use_mup, scale,
// packed flag) the qwen3 call sites pin to constants anyway.
#include <cuda_bf16.h>

extern "C" __global__ void kern_silu_mul_bf16(__nv_bfloat16* __restrict__ out,
                                              const __nv_bfloat16* __restrict__ in,
                                              int d) {
    const long long t = blockIdx.x;
    const __nv_bfloat16* gate = in + t * 2LL * d;
    const __nv_bfloat16* up = gate + d;
    __nv_bfloat16* o = out + t * (long long)d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float g = __bfloat162float(gate[i]);
        // Match the mined vLLM kernel bit-for-bit: silu is rounded to bf16
        // first, then multiplied in bf16 (double rounding).
        __nv_bfloat16 s = __float2bfloat16(g / (1.0f + expf(-g)));
        o[i] = __hmul(s, up[i]);
    }
}
