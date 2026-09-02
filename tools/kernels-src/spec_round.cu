// The device-side glue of a fused speculative round (`round` = draft →
// splice → verify → draft_precompute → accept → advance, one program, one
// CUDA graph, one host sync per round): the two steps the host used to do
// between the programs. Per sequence i of the batch, `block` rows
// (anchor + block-1 drafts).
//
//   kern_splice_verify: verify's ids from draft's output —
//                       ids[i*block]         = anchor[i]
//                       ids[i*block + 1 + j] = drafts[i*(block-1) + j]
//                       grid.x = seqs, block-1 ≤ blockDim.x threads.
//   kern_spec_accept:   a = longest prefix of drafts[i] matched by
//                       verify[i] (row j predicts what follows draft j);
//                       nacc[i] = a + 1, and the sequence's line moves to
//                       entry a of its cell in every row of the line table:
//                       line_adv[(r*seqs_max + i)*w + e] = e == a ? line_in[(r*seqs_max + i)*w] : 0
//                       (entry 0 of `line_in` is the line the host staged
//                       for verify; 0 is the null line). grid.x = seqs,
//                       rows ≤ blockDim.x threads (one per table row).
//
//   nvcc -cubin -arch=sm_103a -o target/cubins/spec_round.cubin tools/kernels-src/spec_round.cu

extern "C" __global__ void kern_splice_verify(
    const long long* __restrict__ anchor, const long long* __restrict__ drafts,
    long long* __restrict__ ids, int block) {
  const int i = blockIdx.x;
  const int t = threadIdx.x;
  if (t == 0) ids[(long long)i * block] = anchor[i];
  if (t < block - 1) ids[(long long)i * block + 1 + t] = drafts[(long long)i * (block - 1) + t];
}

extern "C" __global__ void kern_spec_accept(
    const long long* __restrict__ drafts, const long long* __restrict__ verify,
    const int* __restrict__ line_in, int* __restrict__ nacc, int* __restrict__ line_adv,
    int block, int rows, int seqs_max) {
  const int i = blockIdx.x;
  __shared__ int s_a;
  if (threadIdx.x == 0) {
    const long long* d = drafts + (long long)i * (block - 1);
    const long long* v = verify + (long long)i * block;
    int a = 0;
    while (a < block - 1 && d[a] == v[a]) ++a;
    s_a = a;
    nacc[i] = a + 1;
  }
  __syncthreads();
  const int a = s_a;
  for (int r = threadIdx.x; r < rows; r += blockDim.x) {
    const long long cell = ((long long)r * seqs_max + i) * block;
    const int line = line_in[cell];
    for (int e = 0; e < block; ++e) line_adv[cell + e] = e == a ? line : 0;
  }
}
