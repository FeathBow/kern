# DSpark 投机解码（`kern run --spec`）

draft 也是个 model，但**不是新的 schema 概念**：`examples/qwen3-4b-dspark.json`
是一份 manifest，target+draft 权重（`draft.` 前缀）、第二个 KV state、
六个 program（prefill/decode/decode_spec/verify/draft/draft_precompute）。
draft = deepseek-ai/dspark_qwen3_4b_block7（5 层 DFlash 并行 block draft +
Markov 顺序头，块长 7）。**新增手写核零个**（repo 手写核总数仍是 2：
embedding / argmax，都是 decode 路径原有的）——draft 与 target 几何完全
同构，grid 是 tokens 的表达式，5 层 forward 以 env tokens=7、verify 以
tokens=8 复用同一批 kernel 条目；新增条目全是布线/常量差异。

Markov 头怎么落到既有核上（`membed=markov_w1[prev]`、
`logits_i = base_logits[i] + markov_w2 @ membed`、`argmax`，
markov_w2 无 bias、scale=1）：gather 就是 token embedding 那个核
（`[V,D]` 表按 i64 下标取一行，D=256、grid 常量 1）；**GEMV 与那个
elementwise add 由 β=1 一次做完**（`C[1,V] += membed@markov_w2^T`，C 直接
是 logits_blk 第 t 行）——这才是要 `_acc` 变体的真正原因，否则此处要手写
一个 add 核；采样是既有两段式 argmax 的单行版。`embedding_row`/
`argmax_row` 只是 grid+scratch 取常量 1 的另一份 impl（同符号同 cubin），
避免 scratch 按 tokens 上界分配。7 步链的 `prev` 直接从 `draft_tokens`
的字节 offset 读，不回 host——所以整个 draft program（84 dispatch）能
一次 graph 捕获。若 dspark 走的是 top-k 的 `apply_bias_gathered`
（往 -inf 稠密缓冲 scatter），这里就真得手写核了；vanilla 全词表路径
正好躲过。

结构要点（全部由 spec capture 的断言证伪，`tools/capture_qwen3_spec.sh`）：
- **两个 28 参 unified 实例强制 cubin 钉定**：causal（prefill/verify）与
  non-causal（draft 的 7 query 互见）symbol、参数布局、block、smem 逐位
  相同，静态 ABI 无法消歧，唯一可见差异是 num_regs（94/86）。生成器从
  launch 流拿 regs、cuobjdump 定位 module 文件、按内容 sha256 钉进
  manifest——v2 的可插拔工件路径在真实场景里成为硬需求的实证。
- **draft 的 context KV 不来自 draft forward**，而是 target 隐状态投影：
  5 个 tap 点（layer 0/8/16/24/32 的 next_input_norm 之后，residual 恰是
  aux=hidden+residual）各放一个 β=1 累加 GEMM（`extern:cublaslt_bf16_tn_acc`，
  fc 权重按列切 5 块）——免 concat 免拷贝，vLLM 的一次 [n,12800] GEMM 数学
  等价。`fc_out` 是新 buffer class **`carry`**：verify/prefill 写、
  draft_precompute 读，跨 program 交接（顺序是 caller 契约）——program 级
  接口的第一块实料。
- **draft_precompute**：hidden_norm → 融合 KV GEMM `[n,10240]` → 逐层
  k_norm（打包写 k_n）→ K-only rope（num_kv=0 跳 key，等效 vLLM 的
  key=NULL——schema 无空指针）→ reshape_and_cache 进 5 层交织 draft_kv
  （20480 B/token）。positions/slot_mapping 直接沿用产生这批 aux 的那次
  forward 的输入，caller 无需重写。
- **Markov 头展开成 7 步链**（都在 manifest 里，可整图捕获）：
  embedding_row 取 `markov_w1[prev]` → gemm_acc 把 markov_w2 偏置累进该行
  base logits → argmax_row 出 draft token 喂下一步。argmax 核天然多行
  （grid.x=行号），verify 的 8 行 argmax 就是既有 kernel 换 env。
- caller 侧一轮（`kern run --spec`）：draft（graph）→ 读 7 token → verify
  （graph，[anchor,d0..d6]）→ 读 8 预测 → 前缀匹配接受 → precompute
  接受行（eager，17 dispatch）→ 滚动。回滚免费：paged KV 槽位=position，
  被拒绝的槽下一轮直接覆写。

**实测（GB300）**："The capital of France is" 32 token：**逐字节等于普通
decode**（无损 oracle：greedy 投机不改变输出，接错任何 tap/头只会掉接受
率）；3.44 token/轮、38% 接受率、3.56 ms/轮 ≈ **948 tok/s**（vs 非投机
388 → 2.4×）；eager 与 graph 两路逐字节一致。难 prompt（observatory 85
token，3 块 chunked prefill + spec）1.68 token/轮 vs vLLM 本尊同 prompt
1.78——draft 布线质量与 vLLM 持平。观测到一次输出分叉（" actions" vs
" trespass"）：HF 参考实现 top-2 logit 29.125/28.625，bf16 下 2–4 ulp 的
真平局，verify（m=8）与 decode（m=1）归约顺序不同翻了个 near-tie——vLLM
的批量 verify 有同样性质，无损保证 modulo bf16 平局。

```bash
# capture 投机路径（draft 非因果实例 + precompute + verify）
CUDA_VISIBLE_DEVICES=0 tools/capture_qwen3_spec.sh   # -> dumped-kernels/pid<M>/
# 生成两份 manifest（gen 会顺带把 non-causal cubin 拷进 kernels/ 并钉哈希）
.venv/bin/python tools/gen_qwen3_decode.py \
  dumped-kernels/pid<N>/launches.jsonl dumped-kernels/pid<M>
# 合并权重（target + draft.*，fc 按列切块、markov 头原样）
.venv/bin/python tools/export_weights.py             # -> weights/qwen3-4b-dspark.safetensors
./target/release/kern run --manifest examples/qwen3-4b-dspark.json \
  --weights weights/qwen3-4b-dspark.safetensors --spec --steps 320
```
