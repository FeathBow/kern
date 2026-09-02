# K3 decode 核 ABI（kern 自写核集，取代 pegainfer 的 TileLang 桶）

状态：2026-09-02 任务书 / 合同。每个核一份 `.cu`、一个 `extern "C"` 入口、一份 cubin；
manifest 只认 **入口名 + 参数表 + grid/block/smem 公式**，语言无关。本轮全部用 CUDA C++
（nvcc 13.1，`-arch=sm_103a`），验收看 harness、SASS 和 ncu，不要求与 TileLang 逐位一致。

目标：(1) 每 rank B>1 —— B 是运行时参数，一 block 一行，没有桶、没有 line shim；
(2) 把只做搬运/逐元素的 launch 融进邻居 —— 93 层现在 3792 个 launch，目标 ≤ 1500；
(3) MLA decode attention 在长上下文下 KV 只读一次，不是每头读一次。

## 0. 通用约定

- 入口 `extern "C" __global__ void kern_k3_<name>(...)`，文件 `tools/kernels-src/k3_<name>.cu`，
  单文件，只 include CUDA 自带头（`cuda_bf16.h`、`cuda_fp16.h`、`mma.h`），
  `tools/build_kernels.sh` 会编成 `target/cubins/k3_<name>.cubin`。
- 类型：`bf16 = __nv_bfloat16`，`f32 = float`，`i32 = int`，`i64 = long long`。
  标量按值传；指针一律 `__restrict__`。**所有 buffer 行主序，行距 = 宽度**，除非参数表里给了 stride。
- **B 是运行时参数** `int B`（decode 里 tokens == seqs）。约定 `grid.x = B`（一行一个 block.x），
  `grid.y` 按核自己的切分；block 和动态 smem 在文档里给公式，manifest 会照抄。
  runtime 传 grid 时 `tokens` 是变量，所以 grid.x 必须恰好是 B，不能是 ceil(B/k)。
- 权重（gamma、bias、卷积核、`w_f_b`、`w_kv_b`…）所有行共享，没有 batch 维。
- 输出 buffer 必须整块写满；不允许"没变就不写"。原地（同一指针既读又写）只在明确写了"原地安全"的核允许。
- **f32 partial**：cuBLAS extern `cublas_bf16_tn_f32` 的输出，`f32 [B, ldc]`，有效列从 `off` 起 `n` 列。
  所有从 GEMM 接数的核直接读 f32 partial，自己 landing 成 bf16（"land"），不再有单独的 land 核。
- 舍入链：累加 f32，**landing 点**（f32→bf16）按每个核列出的位置放——这是 pegainfer 的链，
  照着做误差最小，但**不要求逐位**；harness 的容差是验收标准。
- 常量（全局，编进核里也行，但入口签名要保持文档形状）：

| 名 | 值 | 名 | 值 |
|---|---|---|---|
| H | 7168 | HEADS / HEAD_DIM | 96 / 128 |
| INNER = HEADS·HEAD_DIM | 12288 | Q_LORA / KV_LORA / ROPE | 1536 / 512 / 64 |
| KV_A = KV_LORA + ROPE | 576 | Q_B = HEADS·192 | 18432 |
| MLA_FUSED = Q_LORA + KV_A + INNER | 14400 | KDA_FUSED = 4·INNER | 49152 |
| WSM | 256（b_proj 96 \| f_a 128 \| pad） | EXPERTS / TOPK | 224 / 16 |
| LATENT / INTER / SHARED | 3584 / 3072 / 6144 | DENSE_I | 33792 |
| V | 163840 | NB_MAX（attnres 块数上限） | 8 |
| EPS（所有 rms） | 1e-5 | LB（KDA gate lower bound） | -5.0 |
| PAGE | 64 token | LATENT_ROW | 576 |

### 状态布局

**KDA line**（每序列每 KDA 层一条，`bytes_per_seq = n_kda · LINE_BYTES`）：

```
offset 0                         : rec   f32 [96 head][128 dv][128 k]      6291456 B
REC_BYTES = 6291456              : win_q bf16 [3 tap][12288]                73728 B
REC_BYTES + 73728                : win_k bf16 [3][12288]
REC_BYTES + 2*73728              : win_v bf16 [3][12288]
LINE_BYTES = 6512640
```

tap 0 最旧，tap 2 最新。行 b 这一层的 line 地址 = `kda_base + (i64)line_index[b] * LINE_BYTES`；
`line_index` 是 `i32 [B]`（manifest 里是 `kda.line_index[n_kda, seqs]` 的一行，runtime 按层
偏移传进来）。核签名里统一写成 `(void* kda_base, const int* line_index, long long line_bytes)`。

**MLA latent slab**（`bytes_per_token = n_mla · 576 · 2`）：页 p 起点 `p * page_stride`（元素），
本层切片 `+ layer_off`，token t `+ t * 576`，行 = `latent 512 | rope 64` bf16。
`page_stride = n_mla · 64 · 576`，`layer_off = k · 64 · 576`（k = 本层在 MLA 层里的序号）。
`block_table i32 [B, max_pages]` 逻辑页→物理页；`seq_lens i32 [B]` = 含本步 token 的上下文长度
（kv_append 在 attention 之前跑）；`slot_mapping i64 [B]` = 本步 token 的 slot，页 = slot/64，行 = slot%64。

**attnres 快照** `blocks bf16 [B, NB_MAX, H]`，`scores` 不再需要（融进核里）。

### 数学原语（CPU 参考按这个写）

- `rms(x, gamma)`（round-before-scale）：`y = bf16(x · rsqrt(mean(x²) + EPS))`，再 `y · gamma`（bf16×bf16 → bf16）。
- `rms_nw(x)`：无权重版，`x · rsqrt(mean(x²) + EPS)`，留 f32。
- `sigmoid(x) = 1/(1+exp(-x))`；`situ(g,u) = 4·tanh(g/4)·σ(g)·25·tanh(u/25)`。
- attnres（NB 个候选 + prefix）：`score_c = Σ_i rms_nw(cand_c)[i] · sw[i]`，c = 0..NB-1 取 `blocks[b,c]`，
  c = NB 取 prefix；`p = softmax(score)`；`mixed = bf16(Σ_c p_c · cand_c)`（f32 累加）。NB = 0 时 mixed = prefix。

## 1. 核清单

每层的 launch 序列（KDA 层 / MoE 层为例）改成：

```
attnres_rms(nb_in, snapshot)                → normed                    [K1]
gemm normed·wbig  → kda_partial f32 [B, 49152]  (q|k|v|gate 一个 GEMM)
gemm normed·wsm   → wsm_partial  f32 [B, 256]
conv_silu(kda_partial, line)                → conv_q/k/v                [K2]
kda_core(conv_qkv, wsm_partial, kda_partial band 3, line) → gated       [K3]
gemm gated·w_o    → hidden_partial f32
land_add_attnres_rms(hidden_partial, hidden, snapshot, nb_mlp) → prefix2, normed   [K1]
gemm normed·w_router → router_partial;  router_topk → idx, wts          [K6]
gemm normed·w_lat_down → latent_partial; land → latent                  [K7]
MegaMoE ×3 → routed_latent;  rms(gamma_lat) → routed_latent_norm       [K7]
gemm → routed_partial;  gemm normed·wsh → shared_partial
land_situ(shared_partial) → shared_act                                  [K7]
gemm shared_act·sh_down → shared_partial2
land_add2(routed_partial, shared_partial2, prefix2) → hidden            [K1]
```

MLA 层把 conv/kda 换成：

```
gemm normed·wfu → mla_fused_partial f32 [B, 14400]
mla_prep(mla_fused_partial, slot_mapping, slab) → q_norm, mla_gate, slab 追加   [K4]
gemm q_norm·w_q_b → q_partial f32 [B, 18432]
mla_paged_attn(q_partial, w_kv_b, slab, block_table, seq_lens, mla_gate) → gated   [K5]
```

尾部：`attnres_rms(8)` → `gemm w_lm` → `argmax_f32` [K6]。

### K1 残差流：`k3_attnres_rms` / `k3_land_add_attnres_rms` / `k3_land_add2`（一个 agent）

```c
// [K1a] mixed = attnres(blocks, prefix, nb); if (snapshot) blocks[b, nb] = prefix; normed = rms(mixed, gamma)
extern "C" __global__ void kern_k3_attnres_rms(
    const bf16* prefix,      // [B, H]  残差流（hidden）
    bf16*       blocks,      // [B, NB_MAX, H]  快照；snapshot != 0 时把 prefix 写进 blocks[b, nb]
    const f32*  sw,          // [H]  scoring 向量；nb == 0 时可为任意指针（不读）
    const bf16* gamma,       // [H]
    bf16*       normed,      // [B, H]
    int nb, int snapshot, int B);
// grid (B, 1, 1)，block 1024，smem 由你定（H=7168：每线程 7 个元素，NB+1 ≤ 9 个 score）

// [K1b] p = bf16(partial[b, :H]);  prefix2 = snapshot ? p : bf16(prefix + p);
//       mixed = attnres(blocks, prefix2, nb);  normed = rms(mixed, gamma)      （不写快照）
extern "C" __global__ void kern_k3_land_add_attnres_rms(
    const f32*  partial,     // [B, H]  o_proj 的 f32 partial
    const bf16* prefix,      // [B, H]
    const bf16* blocks,      // [B, NB_MAX, H]
    const f32*  sw, const bf16* gamma,
    bf16*       prefix2,     // [B, H]  必须写（层尾要用）
    bf16*       normed,      // [B, H]
    int nb, int snapshot, int B);

// [K1c] hidden = bf16( prefix2 + bf16(p1[b,:H]) + (two ? bf16(p2[b,:H]) : 0) )   （two == 0：dense 层，p2 不读，但传的是合法指针）
extern "C" __global__ void kern_k3_land_add2(
    const f32* p1, const f32* p2, const bf16* prefix2, bf16* hidden, int two, int B);
// grid (B, 1, 1)，block 1024
```

landing 点：attnres 的 mixed 落 bf16 一次；rms 内部 `bf16(x·rsqrt)` 再乘 gamma；K1b 的 p 先落 bf16
再与 prefix 相加落 bf16（两次舍入，与 pegainfer 同；愿意的话可以 f32 加完落一次，容差内都行）。
K1a 的 prefix 与 normed 可以是不同 buffer；`hidden` 在 K1c 之前不会被覆盖，所以生成器不再拷 prefix。

### K2 `k3_conv_silu`：三条流一次 launch，窗口在 KDA line 里

```c
// 对 s = 0,1,2（q,k,v）：x = bf16(partial[b, s*INNER + c]);
//   y = Σ_{t<3} f32(win_s[t][c])·cw[s][t][c] + f32(x)·cw[s][3][c];  sb = bf16(y);  out_s[b,c] = bf16(sb·σ(sb));
//   win_s[0..1] = win_s[1..2];  win_s[2] = x       （窗口原地更新）
extern "C" __global__ void kern_k3_conv_silu(
    const f32*  partial,     // [B, KDA_FUSED]  列 s*INNER.. 是流 s 的 f32 partial
    const f32*  cw,          // [3][4][INNER]   cw_q | cw_k | cw_v（生成器把三份权重拼成一块）
    void*       kda_base, const int* line_index, long long line_bytes,   // 窗口：line + REC_BYTES + s*73728
    bf16*       conv_q, bf16* conv_k, bf16* conv_v,   // [B, INNER] 各一
    int B);
// 交付：grid (B, 3, 24)（y = 流，z = 512 列一段），block 128，每线程 4 列，smem 0
```

### K3 `k3_kda_core`：delta rule，f_b 投影与 gate 的 landing 都在核内

```c
// 行 b、头 h（block = (b, h)，128 线程，线程 = dv）：
//   q,k,v = conv_q/k/v[b, h*128 .. +128]
//   qtot = Σ bf16(q·q) (f32)，kr 链全 bf16：qr = bf16(rsqrt(f32(bf16(qtot)) + 1e-6))
//   qs[d] = f32(bf16(q[d]·qr)) · 128^-0.5 ;  kn[d] = f32(bf16(k[d]·kr))
//   beta   = σ(f32(bf16(wsm_partial[b, h])))                          // 列 0..95
//   flow   = bf16(wsm_partial[b, 96 .. 224])                          // 列 96..223，128 个
//   ga[d]  = Σ_j f32(flow[j]) · f32(w_f_b[h*128+d, j])                // f_b GEMM 就地算，f32
//   raw[d] = f32(bf16(ga[d])) + dt_bias[h*128+d]
//   dec[d] = exp(LB · σ(exp(a_log[h]) · raw[d]))
//   m[dv]  = Σ_k S[h,dv,k]·dec[k]·kn[k];  dlt[dv] = (f32(v[dv]) - m[dv])·beta
//   S'[h,dv,k] = S[h,dv,k]·dec[k] + dlt[dv]·kn[k]      （原地写回 rec）
//   attn[dv] = bf16(Σ_k S'[h,dv,k]·qs[k])
//   o[d] = bf16( f32(attn[d]) · rsqrt(mean(attn²) + EPS) · gamma_o[d] ) · bf16(σ(f32(bf16(gate_partial[b, 3*INNER + h*128 + d]))))
extern "C" __global__ void kern_k3_kda_core(
    const bf16* conv_q, const bf16* conv_k, const bf16* conv_v,   // [B, INNER]
    const f32*  wsm_partial,   // [B, WSM]
    const f32*  gate_partial,  // [B, KDA_FUSED]  只读 band 3（列 3*INNER..）
    const bf16* w_f_b,         // [INNER, 128]
    const f32*  dt_bias,       // [INNER]
    const f32*  a_log,         // [HEADS]
    const f32*  gamma_o,       // [128]
    void*       kda_base, const int* line_index, long long line_bytes,   // rec 在 line 偏移 0
    bf16*       out,           // [B, INNER]
    int B);
// grid (B, HEADS, 1)，block 128
```

性能点：rec 每行每层读 + 写 6.3 MB，B=64 时一层 800 MB，这是 KDA 的带宽主项——访存模式（线程=dv、串行 k 是按行走，
warp 内 32 行同时读）请用 ncu 看 L1/L2 命中和 dram 吞吐，必要时换 k 分片 + shuffle 归约。
`w_f_b` 每 block 读 128×128 bf16 = 32 KB，B·96 个 block 共享，走 L2。

### K4 `k3_mla_prep`：MLA 融合投影的落地 + 两个 norm + 追加 latent

```c
// 从 P = mla_fused_partial[b, :]（14400 列）：
//   q_norm[b]   = rms(bf16(P[0 .. 1536]), gamma_q_a)                 // land 后 round-before-scale
//   kv_norm     = rms(bf16(P[1536 .. 2048]), gamma_kv_a)             // 512
//   rope        = bf16(P[2048 .. 2112])                              // 64
//   slab[slot]  = kv_norm | rope                                     // kv_append：行 = (slot/64)·page_stride + layer_off + (slot%64)·576
//   mla_gate[b] = bf16(P[2112 .. 14400])                             // 12288
extern "C" __global__ void kern_k3_mla_prep(
    const f32*  partial,       // [B, MLA_FUSED]
    const bf16* gamma_q_a,     // [Q_LORA]
    const bf16* gamma_kv_a,    // [KV_LORA]
    const i64*  slot_mapping,  // [B]
    bf16*       slab,          // state 基址
    long long layer_off, long long page_stride,   // 元素
    bf16*       q_norm,        // [B, Q_LORA]
    bf16*       mla_gate,      // [B, INNER]
    int B);
// 交付：grid (B, 4, 1)（y=0 做两个 norm + 追加，y=1..3 各落 1/3 的 gate），block 512，smem 12800（静态）
```

### K5 `k3_mla_paged_attn`：absorbed MLA decode，多头共享 KV 读，gate 在 epilogue

现有核（`tools/kernels-src/k3_mla_paged_attn.cu`，pegainfer 原版）一 block 一头，每头把整个上下文
读一遍：32k 上下文时每层每序列读 96 × 37 MB，24 层 ≈ 85 GB/步，这是长上下文的第一性能问题。

```c
// 行 b：q_h = bf16(q_partial[b, h*192 .. +192])（nope 128 | rope 64），h = 0..95
//   q_abs_h = [ bf16(Σ_d q_h[d]·W_UK_h[d, j]) for j<512 | q_h[128..192] ]        // W_UK_h = w_kv_b[h*256 + 0..128, :]
//   s_h[t]  = f32( bf16(q_abs_h · row_t) · scale )   t < seq_lens[b]，row_t 走 block_table   // bf16 landing 后乘 bf16 scale
//   p_h     = softmax_t(s_h)，概率 bf16 landing：p = f32(bf16(exp(s - m)/l))
//   lat_h   = bf16(Σ_t p_h[t] · row_t[0..512])                                       // f32 累加
//   o_h[dv] = bf16(Σ_j W_UV_h[dv, j] · f32(lat_h[j]))                                // W_UV_h = w_kv_b[h*256 + 128 + dv, :]
//   gated[b, h*128+dv] = o_h[dv] · bf16(σ(f32(mla_gate[b, h*128+dv])))              // mul_sigmoid 融进 epilogue
extern "C" __global__ void kern_k3_mla_paged_attn(
    const f32*  q_partial,     // [B, Q_B]
    const bf16* w_kv_b,        // [HEADS*256, KV_LORA]
    const bf16* cache,         // slab 基址 + layer_off（生成器传 state 偏移）
    const int*  block_table,   // [B, max_pages]
    int max_pages, long long page_stride,
    const int*  seq_lens,      // [B]
    const bf16* scale,         // [1]  bf16 softmax scale（192^-0.5）
    const bf16* mla_gate,      // [B, INNER]
    bf16*       gated,         // [B, INNER]
    int B);
// grid (B, HEAD_GROUPS, 1)：一个 block 处理一组头（8/16/32/96 由你定），KV 每页只读一次供整组用；
// 分数/累加可以用 mma（bf16 tensor core，[heads × 576] × [576 × 64]），也可以纯 FMA，看 ncu 说话
```

验收形状：ctx ∈ {1, 64, 65, 2048, 32768}，B ∈ {1, 8}。目标：ctx=32768、B=1 比现有核快 ≥ 3×
（现有核基线由 harness 给出）；短上下文不慢于现有核。也交付一份 split-KV（长上下文按页段切、再合并 (m, l, acc)）的分析，
不一定实现。

**交付（2026-09-02）**：grid `(B, 48, 1)` block 512，`__cluster_dims__(1, 8, 1)`，静态 smem 216 320 B：
6 个头组 × 16 头，每组 8 个 KV split 各占一个 cluster block，(m, l, acc) 经 DSMEM 合并；两遍 softmax，
分数与 P·V 走 `mma.m16n8k16` bf16。harness 全过；ctx=32768 B=1 **258 µs vs 旧核 10 047 µs（38.9×）**，
ctx=64 24.6 µs vs 78 µs。SPL=16 变体（155 µs）没上：runtime 的 `cuLaunchKernelEx` 不设
`NonPortableClusterSizeAllowed`；同理动态 smem > 48 KB 的 opt-in 也没有，所以是静态 smem。

### K6 `k3_router_topk` / `k3_argmax_f32`（一个 agent，顺带 embedding 与 rms）

```c
// sig = σ(S[b,e])；biased = sig + bias[e]；顺序扫描取 16 次 max（tie 取小 e）；
// wts[t] = sig[idx[t]] / (Σ_t sig[idx[t]] + 1e-20) · f32(rs[0])
extern "C" __global__ void kern_k3_router_topk(
    const f32* S,            // [B, EXPERTS]  f32 partial
    const f32* bias,         // [EXPERTS]
    const bf16* rs,          // [1]
    int* idx,                // [B, TOPK]
    f32* wts,                // [B, TOPK]
    int B);
// grid (B,1,1)，block 256

// argmax over f32 logits（不再 land 成 bf16）：两段
extern "C" __global__ void kern_k3_argmax_f32_partial(const f32* logits, f32* pmax, int* pidx, int n);  // grid (B, 64), block 1024
extern "C" __global__ void kern_k3_argmax_f32_final(const f32* pmax, const int* pidx, i64* out, int parts); // grid (B,1,1), block 64
// tie：取最小下标。

// 通用 rms（MoE 的 gamma_lat 用，h = 3584；也给 K1 之外任何地方）
extern "C" __global__ void kern_k3_rms(const bf16* x, const bf16* gamma, bf16* o, int h, int B);   // grid (B,1,1), block 1024
```

### K7 `k3_land` / `k3_land_situ`（归 K6 的 agent）

```c
// o[b, i] = bf16(p[b*ldc + off + i])，i < n
extern "C" __global__ void kern_k3_land(const f32* p, bf16* o, int n, int off, int ldc, int B);   // grid (B, ceil(n/1024)), block 1024
// act[b, i] = bf16( situ( f32(bf16(p[b*2n + i])), f32(bf16(p[b*2n + n + i])) ) )，i < n   （gate 在前 n 列，up 在后 n 列）
extern "C" __global__ void kern_k3_land_situ(const f32* p, bf16* act, int n, int B);               // grid (B, ceil(n/1024)), block 1024
```

## 2. 验收（每个核）

1. **harness 通过**：`tools/k3-harness/`（见其 README）——对每个核、每个规定形状，随机输入 + CPU 参考，
   B ∈ {1, 2, 8, 64}。容差：逐元素 |err| ≤ 3 bf16 ULP(|ref|) + 1e-3，且相对 RMS 误差 ≤ 2e-3。
   不许改 harness 与参考实现；觉得参考错了，写进 notes 找我。
2. **SASS**：`nvcc -Xptxas -v` 0 字节 spill、0 字节 local；`cuobjdump -sass` 里没有 `.MULTICAST`；
   寄存器数与 `__launch_bounds__` 一致；把 ptxas -v 输出贴进 notes。
3. **ncu**：`ncu --set full -k regex:kern_k3_<name> -c 1` 在 B=64（K5 在 ctx=32768，B=1）上，
   记录 dram 吞吐（GB/s 与占峰值 8 TB/s 的比例）、achieved occupancy、L2 命中率、总时长；报告存
   `tools/k3-harness/reports/k3_<name>.ncu.txt`。访存主导的核目标 ≥ 60% 峰值带宽；离目标远要在 notes 里说为什么。
4. **交付物**：`tools/kernels-src/k3_<name>.cu`（头注释 = 本文档的签名 + grid/block/smem 公式 + landing 点）、
   ncu 报告、`tools/k3-harness/notes/k3_<name>.md`（设计、访存模式、测得的数字、没做到的事）。
   不要动别的文件，不要 commit。
5. 机器：tray 由任务书指定，`CUDA_VISIBLE_DEVICES=<n>` 各用一张卡；`nvidia-smi` 先看一眼有没有人在跑。
   nvcc/ncu 在 `/usr/local/cuda-13.1/bin`。

## 3. 生成器侧配套（不归 agent）

- q|k|v|gate 一个 GEMM（wbig 全 49152 行）→ `kda_partial`；`cw_q/k/v` 拼成 `[3][4][INNER]` 一块权重。
- `prefix` / `mixed` / `mixed2` / `scores` / `conv_x` / `attn_out` / `mlp_out` / `logits` 等 workspace 删除。
- `seqs` 变成变量（max 64），`kda.line_index` 按层取行，`blocks` 变 `[seqs, 8, H]`。
- 每层 launch 数：KDA/MoE 层 3 + 2 + 1 + 1 + 1 + 3(MegaMoE) + 1 + 1 + 1 ≈ 14 + 8 GEMM；MLA 层少 1。

## 4. 交付状态与遗留（2026-09-02）

七族全部交付并入 master（`tools/kernels-src/k3_*.cu`，`tools/build_kernels.sh` 编成 `target/cubins/`）；
每个核在 harness 上 B ∈ {1, 2, 8, 64} 全过，0 spill，无 `.MULTICAST`；notes/ncu 报告在 `tools/k3-harness/`。
生成器 `tools/gen_k3_decode.py` 已切到这套核（manifest `examples/k3-*-v2.json`，93 层 1855 launch，其中 742 GEMM）。
门禁数字见 roadmap E2 行。

遗留（都是契约层面，核本身不用改）：

- K6 `k3_router_topk` 把 `EXPERTS=224` 烤进了核；满血 K3（896 expert）要另编一版或改成参数。
- K5 SPL=16 变体要 kern-runtime 的 launch 设 `CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE` /
  NonPortableClusterSizeAllowed；>48 KB 动态 smem 也要 `cuFuncSetAttribute` opt-in。两个都是 runtime 加一行属性的事。
- K1a `nb == 8` 与 snapshot 不能同时用（snapshot 写 blocks 的第 8 槽）；生成器只在 `nb < 8` 时带 snapshot。
- "snapshot" 在 K1a（把当前 hidden 存进 blocks[8]）和 K1b（把 landing 后的 hidden 存进 blocks[8]）里含义不同，
  签名同名不同义，改名的事等消融。
- manifest 传不了空指针：K1c 用 `int two` 标志代替 `p2 == NULL`。
