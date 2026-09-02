# `kern-serve`：continuous batching + OpenAI 兼容 endpoint

```bash
# 独立 workspace（serving 栈不进 runtime 的依赖图和 CI）；binary 在 crates/kern-serve/target/
cd crates/kern-serve && cargo build --release
target/release/kern-serve --model-path /mnt/shared/weights/Qwen3-4B --gpu 3 --port 8000
# state 池默认按显存自动定：权重/激活/scratch 分完后，剩余显存减 1 GiB 全给 state，
# KV 页与 state slot 共用这份预算、按需互换（runtime.md）。`--capacity <tokens>` 或
# kern.toml 的 `capacity` 显式给则照旧。
# /v1/completions、/v1/chat/completions（流式 + chat template）、/v1/models、/metrics
```

manifest / kernels / weights 来自 kern.toml 的 target；`--model-path` 是给
**前端**的 HF 目录（tokenizer、chat template、`generation_config.json` 的
eos）。前端整个来自 pegainfer（`pegainfer-frontend`，底下是 vLLM 官方的
Rust server crates，git dep 钉 pegainfer main 的一个 rev），kern 只贡献引擎：`crates/kern-serve`。

## 分工

- **`kern-serve::scheduler::KernScheduler`** 实现 pegainfer 的 `Scheduler`
  契约（`submit` / `step` / `metrics`），跑在 pegainfer 的 `drive` 轮询线程
  上，独占一个 `Runtime`。策略刻意简单：
  - prefill 优先、不混批：每步先把 waiting 里的请求逐个（bs=1、chunk 级）
    prefill 到预算（`--prefill-budget`，默认 2048 token），再对全部 running
    序列做一步 decode；最后一个 prompt token 作为首个 decode 步的输入。
  - 准入即预留：请求在准入时向 runtime 租下最坏情况 `prompt + max_tokens`
    的全部 KV 页（`Runtime::lease` → `Lease`，序列结束即 drop 归还），
    decode 永远不缺页、不抢占。超过单序列上限（最窄页表行长 × 页）→
    `ContextLength`，超过整池 → `KvBudget`。`slot_mapping` / `block_table`
    的值只能从 `Lease` 算出来，scheduler 不碰裸页号。
  - decode 按 bucket（1,2,4,8,16,24,32,48,64,96,128,192,256）pad，每个
    bucket 首次使用时 capture 一张图；pad 行写进 scheduler 自己租的一页。
  - greedy：采样就是 manifest 里的 `argmax`。非 greedy 参数 warn 一次后按
    greedy 服务。EOS 本身不发出（pegainfer 约定），仍计入 `max_tokens`。
- **manifest**（`tools/gen_qwen3_decode.py`）多了一个 var `seqs`（≤256）和
  一个 program：
  - `decode`：原样，bs=1 契约（3D split-KV unified + reduce_segments——挖到
    的 reduce 是 Triton 在 `num_seqs=1` 下特化出的实例，ABI 里没有
    num_seqs，只能 bs=1）；
  - `decode_batch`：`seqs` 个序列各一行（tokens = seqs），attention 用
    prefill 的那份 2D causal 实例（vLLM 自己在 num_seqs 超过 3D 阈值时 decode
    就走它），grid.x = `ceil(5·tokens/4)`——盖住 vLLM 的 q-block 索引空间
    `tokens//4 + num_seqs`（seqs ≤ tokens 下恒成立），多出的 block 核内提前
    返回；表达式集合因此不用加"两个 var 相加"。
  - 元数据按序列：`block_table [seqs, 256]`、`seq_lens [seqs]`、
    `cu_seqlens_q [257]`（shape 不能是表达式，按上界声明）、`logits [seqs,
    V]`、`next_token [seqs]`；lm_head 的 m = `seqs`。
  - 哪个 bs 走哪个核是 manifest 的选择（两个 program），caller 按 bucket
    选，runtime 不知情。以后补 bs 2–16 的 split-KV = 再捕一次 bs=2 的
    decode 拿未特化的 reduce，多加一个 program。
- **runtime** 只改一处：CUDA graph 按 `(program, var 值)` 键控（原来一个
  program 一张图），加 `is_captured`。

## 实测（GB300 单卡，Qwen3-4B，2026-09-01）

- bs=1 `kern run` 不变：2.6 ms/step，输出逐字一致。
- `vllm bench serve --backend openai --dataset-name random --random-input-len
  1024 --random-output-len 128 --num-prompts 256 --max-concurrency 64
  --ignore-eos`：256/256 成功，**5138 tok/s 输出吞吐**（总 46k tok/s），TPOT
  中位 11.7 ms / P99 11.9 ms，TTFT 中位 62 ms / P99 1.0 s，E2E 中位 1.55 s。
  引擎侧：decode 5.5 ms/step @ ~60 seq，prefill 79.5k tok/s（chunk 512 走图）。
- 一致性：8 个不同 prompt 并发两轮逐字相同（确定性）；同 batch 内 16 份
  相同 prompt 里同 cohort 的 14 份逐字相同（行没有错位）。并发 vs 串行有
  2/8 在 ~25 token 处的近平局分叉，两边都连贯——batch 大小改变 GEMM 选核
  和 attention 归约顺序，和 vLLM 一样不 batch-invariant。

## 投机解码（`--spec`，2026-09-01）

```bash
target/release/kern-serve qwen3-4b-dspark --model-path /mnt/shared/weights/Qwen3-4B --spec
```

manifest 得带 `draft` / `verify` / `draft_precompute` / `decode_spec`
（`examples/qwen3-4b-dspark.json`）。开了之后**每一步都是一轮**：

- admission：prefill 每个 chunk 后跑一次 `draft_precompute`（prompt 的 tap
  进 draft KV）；最后一个 prompt token 走 bs=1 `decode_spec` + precompute，
  它的输出是第一个 token，当轮的 anchor。租约是 `prompt + max_tokens +
  n_drafts`：最后一轮被拒的行也要有 slot（整页取整，通常就是多一页）。
- 一轮：`draft`（每序列 `[anchor, mask×6]` 一段，非因果，7·b 行）→ 读
  `draft_tokens [seqs, 7]` → `verify`（每序列 `[anchor, d0..d6]`，8·b 行）
  → 读 `verify_tokens [seqs, 8]` → `draft_precompute` 在 verify 的全部 8·b
  行上跑（被拒行落在各序列新 pos 之后，下一轮覆写，和 target KV 的免费
  回滚是一回事，所以不用按接受数 compact）→ host 逐序列前缀匹配，emit
  `a+1` 个 token。三段各按 bucket 捕成图；pad 序列的行写 pad 页。
- manifest 带 `round` 时（`qwen3.8-27b-dflash2`），整轮是**一个 program、
  一张图、一次 sync**：draft → `splice_verify`（device 上把 anchor +
  `draft_tokens` 拼成 verify 的 ids，`verify_ids` carry）→ verify →
  precompute → `spec_accept`（device 上前缀匹配，写 advance 自己的
  `nacc_adv` / `line_adv` carry——kernel 不能写 Input，所以是替身）→
  advance。host 只 stage 一次 8 行组（draft/verify 每序列行数相同，
  positions/slot_mapping 共用），轮末读 `draft_tokens` / `verify_tokens`
  照旧前缀匹配，emit 与分段路径一字不差。分段的 `draft`/`verify`/`advance`
  仍在 manifest 里，`kern run --spec --probe-dir` 逐轮 dump 靠它们。
- greedy only；`--spec` 是能力开关，不按 bs 自动切换——每轮 verify 是
  8·b 行的 target 前向，bs 大到算力瓶颈后一轮比一步贵得多，划不划算由
  用户按模型和负载定。

**验证**（GB300，Qwen3-4B + DSpark，docs 段落做 prompt，128 token）：
conc=1 与 `kern run --spec` 逐字节一致；接受率 conc=1 20.8%、conc=32
19.2%（2.4 / 2.3 tok/round）；不同 bs 之间的输出分叉率与普通模式的
`decode` vs `decode_batch` 对照组同量级（都是 bucket 变化 + bf16
near-tie）。吞吐：conc 1 / 8 / 32 ≈ 600 / 2560 / 5850 tok/s，普通模式
353 / 2048 / 6800——这组 prompt 上交叉点在 bs 16–32。

## 前缀缓存（K1，2026-09-02）

调度器持有一张 `Prefix` 表（kern-runtime，纯 host）：结束的序列留成 checkpoint，
新 prompt 从覆盖其真前缀的最长 checkpoint 起步（`Runtime::lease_from`），prefill
只补剩下的。两种模型一套机制，差别只在"何时留"：

- 纯 KV（qwen3-4b，页 16 token）：序列每填满一页就 `Runtime::checkpoint` 一次——页进
  共享链，不拷字节，所以任何早先的 prompt 或输出都按页粒度可复用；
- 带循环状态（qwen3.8-27b，页 784 token，GDN state 154 MB/序列）：只在请求结束时
  `Runtime::retire`——结束序列的 state slot 原样成为 checkpoint 的，不拷；因此只有
  "续着上一轮整段上下文"的 prompt 命中，同一 prompt 重发不命中（checkpoint 比它长）。
  slot 从 manifest 的 `seqs.max + 2` 个起，与 KV 页共用一份显存预算按需互换（K1b）：
  睡着的 session 的 checkpoint 拿着 slot，活跃请求再要 slot 就从空闲页拆，页不够
  再从空闲 slot 拆回来；租约 `Busy` 时按最久未命中淘汰，`Remapping` 时等它落地
  （stats 行的 `slots_used`/`slots`/`remaps`）。

门禁（本机 GB300，warm 服务与 cold 服务各一，greedy，`max_tokens 64`，prompt 为 46 KB
工程日志）：

| 模型 | prompt | cold prefill | 命中 | 命中后 prefill | 多轮 R3 warm vs cold |
|---|---:|---:|---:|---:|---|
| qwen3-4b | 13 983 tok | 888 ms（请求 1.23 s） | 13 968 tok | 27 ms（请求 0.36 s） | 64 token 逐字一致 |
| qwen3.8-27b | 14 256 tok | 请求 1.90 s | 14 319 tok（上一轮全文） | 请求 0.85 s | 64 token 逐字一致 |

qwen3.8 重发同一 prompt 不命中（唯一的 checkpoint 比它长），输出与首发一致。qwen3-4b
重发同一 prompt 命中 13 968，输出在第 10 个 token 后与首发分叉；两次命中之间完全一致。
分叉不是缓存的：同一 prompt 在 cold 服务上用 `--chunk 144`（末块同样是 15 个 token）
全量 prefill，得到第三种续写——三种切块三种 attention 归约顺序，这个位置是 bf16
近平局，与 `decode` / `decode_batch` 之间的分叉同类。
host 侧回放见 roadmap K1 / K1b 行（`crates/kern-run/examples/agentx_replay.rs`）。

K1b（页与 slot 共用预算）后的门禁（2026-09-02）：上表两行的输出逐字不变；qwen3.8-27b 单卡
预算 223.9 GiB = 9551 块 × 24 MiB，起步 4287 页 + 130 slot；200 个不同的短请求依次结束后
slot 长到 191（每个 checkpoint 拿一个，活跃请求再要就从空闲页拆，61 次 remap），此后重发
早先的 prompt 与首次输出一致、46 KB prompt 的 64 个 greedy token 与 cold 服务一致。
这条门禁第一次跑就抓到一个 manifest 的 bug：`gen_qwen35.py` 把 `(seqs.max + 2) × 48 = 6240`
当 vLLM conv kernel 的 `num_cache_lines` 写成字面量，kernel 用它做越界掩码，slot ≥ 130 的
conv state 静默丢掉（短 prompt 单块 prefill 从零态起步看不出来，14k 的 prompt 第二块起就错）。
按 slot 编号二分定位：页编号高没事、`kern run` 单序列没事，只有 kern-serve 里 slot ≥ 130
的多块 prefill 出错。manifest 里不许再出现 slot 数：conv kernel 现在拿 i32 最大值，掩码永不生效。

带状态模型的部分命中在算力上没有意义：state 快照之后的 token 必须整段重跑 forward
才能把状态推过去，attention 层的投影和注意力都省不掉，所以有效命中 = 最深的带
state 的 checkpoint，KV 页只是顺带共享（vLLM v1 也是这样：hit 取各层组最小值，
attention 的命中被截到 mamba 块边界）。快照放哪由调用方定——请求结束（agent 多轮）
和显式断点（roadmap K1b），不在每个块边界都存。

## 没做（按需要加）

混批（chunked prefill 进 decode 步）、抢占 / 动态页分配、
真采样（temperature/top-p 作为 manifest 内的 `sample` op；投机下是 rejection sampling）、logprobs / echo、
bs 2–16 的 split-KV decode、步间 host 空转（token 反馈进图）。
