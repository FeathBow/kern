# `kern-serve`：continuous batching + OpenAI 兼容 endpoint

```bash
# 独立 workspace（serving 栈不进 runtime 的依赖图和 CI）；binary 在 crates/kern-serve/target/
cd crates/kern-serve && cargo build --release
target/release/kern-serve --model-path /mnt/shared/weights/Qwen3-4B --gpu 3 --port 8000 --capacity 262144
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
  - 准入即预留：请求在准入时拿走最坏情况 `prompt + max_tokens` 的全部 KV
    页（`kern-serve::pages::PagePool`），decode 永远不缺页、不抢占。超过
    单序列上限（block_table 行长 × 页）→ `ContextLength`，超过整池 →
    `KvBudget`。
  - decode 按 bucket（1,2,4,8,16,24,32,48,64,96,128,192,256）pad，每个
    bucket 首次使用时 capture 一张图；pad 行指向池里留出的牺牲页。
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

## 没做（按需要加）

混批（chunked prefill 进 decode 步）、抢占 / 动态页分配、prefix cache、
真采样（temperature/top-p 作为 manifest 内的 `sample` op）、logprobs / echo、
bs 2–16 的 split-KV decode、步间 host 空转（token 反馈进图）。
