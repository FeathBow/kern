# tools/：从 vLLM 到可执行 manifest 的流水线

按执行顺序：

| # | 工具 | 输入 → 输出 |
|---|------|-------------|
| 1 | `capture_qwen3.sh` | vLLM 0.28（TRITON_ATTN，enforce_eager）跑 4 条递增 prompt → `dumped-kernels/pid<N>/`：全部 module cubin + `launches.jsonl`（每次 launch 的符号/grid/block/shmem/逐参数值 + `t_ns`） |
| 2 | `mine_capture.py` | `launches.jsonl` → 分析报告：按时间隙切 pass / 按核爆发切 forward、(range,offset) 指针稳定性分类、grid 表达式拟合（const/var/mul/ceil_div）。纯分析，无模型知识 |
| 2b | `capture_qwen3_spec.sh` | 同上但开 DSpark 投机（draft `weights/dspark_qwen3_4b_block7`，7 draft token，固定 k 验证）→ 第二个 dump：draft 的 non-causal unified 实例、context-KV precompute、verify pass |
| 3 | `gen_qwen3_decode.py` | 两个 `launches.jsonl` → `examples/qwen3-4b.json`（silu 用 HF hub 包）+ `qwen3-4b-silu-mined.json`（kern-test 的 A/B fixture，silu 用挖矿实例）+ `qwen3-4b-dspark.json`：真实 ABI + 手写连线，发射前用挖矿地址逐项断言证伪（q/k/v 视图偏移、KV 池布局、权重指针互异、precompute 的 K-only rope 与 grouped k_norm…）；顺带按 num_regs + cuobjdump 消歧两个同 ABI 的 unified 实例并钉哈希 |
| 3b | `kern_manifest.py` | 生成器共用件。`DumpIndex(dump).pin(symbol, regs, param_sizes)`：按内容索引 dump 的 module，给每个挖矿 launch 钉唯一的 module（寄存器数分不开的 Triton constexpr 实例再按 `.nv.info` 参数布局分）。`normalize()` 后处理：把 launch 内联的 cubin/sha256 提升到 `modules` 表、把每次 call 都相同的接口标量折进 impl 的 launch 字面量、抹掉恒等连线与重复 `params`、规范键序。生成器写长，wire form 写短 |
| 4 | `build_kernels.sh` + `handwritten.py` | `kernels-src/*.cu` → `target/cubins/`（nvcc，sm_103a）；生成器通过 `**hw("name")` 把当前 build 的 sha256 钉进 launch 的 module——换 nvcc / 换 flag / 改源码就是另一个核 |
| 5 | `extract_kernels.sh` | manifest + dump 目录 → `kernels/`：`modules` 表里每个 module 按 sha256 在 dump（递归）/ `target/cubins` 里找到文件，落地为 `<module>-<sha12>.cubin`；只增不减，同一目录可放每个版本，A/B 两份 manifest 共用（runtime 只装载各自点名的） |
| 6 | `export_weights.py` | HF checkpoint（+ draft checkpoint）→ `weights/`：qkv/gate_up 合并、rope cos_sin_cache 预计算、kv_scales 全 1、tied lm_head clone + tokenizer 文件；draft 侧另做 fc 按列切 5 块、融合 KV 权重 cat、markov 头原样 → `qwen3-4b-dspark.safetensors` |

## K3（多卡线，E1/E2）

kern 的 Kimi-K3 pruned decode 不是从 vLLM 挖的：vLLM 的 K3 路径是 fused
KDA / trtllm-gen MLA / DeepGEMM MegaMoE 的 struct ABI 与 torch.compile 混合，
不可 rebind。核来自 pegainfer 的认证 K3 核集，program 逐行照 pegainfer 的
`k3_step` 发射，oracle 是 pegainfer 的 golden fixture（`crates/kern-run/
examples/k3_golden.rs`）。

| 工具 | 输入 → 输出 |
|------|-------------|
| `k3-tilelang/` | pegainfer `pegainfer-k3/kernels/generate.py` 的 AOT TileLang 核源码（vendored，每族一个文件含全部 batch bucket） |
| `build_k3_kernels.sh` | `k3-tilelang/*.cu` → `target/cubins/k3_tl_<族>.cubin`（在 kernel-lab 里 nvcc，pegainfer 的 flag）；state 核经 `k3_line_shim.py` 生成 line 寻址的 `extern "C"` 包装（`kern_k3_kda_core_b1_..._line` / `kern_k3_conv_silu_b1_..._line`：body 原样内联为 `__device__`，包装算 `Base + Line[0]*stride + Off` 并**原地**读写）→ `k3_tl_<族>_line.cubin` |
| `k3-mega/` + `build_k3_mega.sh` | DeepGEMM MegaMoE fork（SymBuffer 在设备上读 peer 表）→ `k3_mega_moe.cubin` + `k3_mega_layout_dump`（E1） |
| `kernels-src/k3_mla_paged_attn.cu`、`k3_kv_append.cu`、`k3_mega_stage.cu` | pegainfer 的 absorbed paged-MLA decode 核原样（加 `extern "C"`）；latent 追加（token slot → 页/行）；MegaMoE 输入 staging |
| `export_k3.py` | HF checkpoint → `dense/bookends + dense/l<i>`（所有 rank 共用）+ `experts/ep<R>-r<r>-l<i>`（按 rank 分片，MegaMoE 布局，复用 `export_k3_moe.py` 的变换）；slot 布局照 pegainfer `model/plan.rs`。权重放数据盘（tray04 `/data/susun/kern-k3/`），跑在 vllm 镜像的 CPU 容器里 |
| `gen_k3_decode.py` | `--layers N --ranks R` → `examples/k3-<N>l-ep<R>.json` / `k3-ep4.json`：整条 decode program（3792 步 @93 层），几何与参数从 TileLang 源码解析，MoE 三步来自 `gen_k3_moe.mega_pieces` |
| `k3_oracle_dump.py` | 任一 OpenAI 兼容服务 → fixture（teacher-forced greedy）。vLLM 带 top-5 logprob，给 `k3_golden --margin-abs` 做 noise-floor 判定；pegainfer 的 K3 不出 logprob，用 `--no-logprobs`（`return_token_ids`，只记 argmax，逐步必须精确一致） |

支撑件：

- `kernel-capture/`：CUPTI 注入库（vendored from pegainfer PR #982 + `t_ns` patch），`CUDA_INJECTION64_PATH` 挂进目标进程。
- `kernels-src/`：手写核（embedding、argmax、gemma norm、copy_rows、sigmoid_mul、DFlash2 的 conv/select/topk…），其余全部来自 vLLM dump。
- `capture_abi_probe.sh`：诊断用。快速抓某个 attention backend 的 ABI（`ABI_PROBE_BACKEND=FLASH_ATTN` 等），当初用它实锤 FA4/trtllm-gen 不可 rebind。
- `capture_sglang.sh` + `capture_sglang.py`：跨框架演示——同一注入库 dump SGLang（docker 镜像里跑）。实测 GB300 上 SGLang 几乎全员 struct ABI（trtllm-gen attention + nvjet GEMM + 单 struct 参数的自家 JIT 核），可挖性远差于 vLLM，见 ../docs/kernel-mining.md。
