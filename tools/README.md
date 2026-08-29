# tools/：从 vLLM 到可执行 manifest 的流水线

按执行顺序：

| # | 工具 | 输入 → 输出 |
|---|------|-------------|
| 1 | `capture_qwen3.sh` | vLLM 0.28（TRITON_ATTN，enforce_eager）跑 4 条递增 prompt → `dumped-kernels/pid<N>/`：全部 module cubin + `launches.jsonl`（每次 launch 的符号/grid/block/shmem/逐参数值 + `t_ns`） |
| 2 | `mine_capture.py` | `launches.jsonl` → 分析报告：按时间隙切 pass / 按核爆发切 forward、(range,offset) 指针稳定性分类、grid 表达式拟合（const/sym/mul/ceil_div）。纯分析，无模型知识 |
| 2b | `capture_qwen3_spec.sh` | 同上但开 DSpark 投机（draft `weights/dspark_qwen3_4b_block7`，7 draft token，固定 k 验证）→ 第二个 dump：draft 的 non-causal unified 实例、context-KV precompute、verify pass |
| 3 | `gen_qwen3_decode.py` | 两个 `launches.jsonl` → `examples/qwen3-4b.json` + `examples/qwen3-4b-dspark.json`：真实 ABI + 手写连线，发射前用挖矿地址逐项断言证伪（q/k/v 视图偏移、KV 池布局、权重指针互异、precompute 的 K-only rope 与 grouped k_norm…）；顺带按 num_regs + cuobjdump 消歧两个同 ABI 的 unified 实例并把 non-causal cubin 拷进 `kernels/` 钉哈希 |
| 4 | `extract_kernels.sh` | dump 目录 → `kernels/`：拷出 manifest 用到的 module cubin + nvcc 编 `kernels-src/*.cu` |
| 5 | `export_weights.py` | HF checkpoint（+ draft checkpoint）→ `weights/`：qkv/gate_up 合并、rope cos_sin_cache 预计算、kv_scales 全 1、tied lm_head clone + tokenizer 文件；draft 侧另做 fc 按列切 5 块、融合 KV 权重 cat、markov 头原样 → `qwen3-4b-dspark.safetensors` |

支撑件：

- `kernel-capture/`：CUPTI 注入库（vendored from pegainfer PR #982 + `t_ns` patch），`CUDA_INJECTION64_PATH` 挂进目标进程。
- `kernels-src/`：仅有的两个自写核——`embedding.cu`（i64 gather）、`argmax.cu`（两段式 greedy 采样）。其余全部来自 vLLM dump。
- `capture_abi_probe.sh`：诊断用。快速抓某个 attention backend 的 ABI（`ABI_PROBE_BACKEND=FLASH_ATTN` 等），当初用它实锤 FA4/trtllm-gen 不可 rebind。
- `capture_sglang.sh` + `capture_sglang.py`：跨框架演示——同一注入库 dump SGLang（docker 镜像里跑）。实测 GB300 上 SGLang 几乎全员 struct ABI（trtllm-gen attention + nvjet GEMM + 单 struct 参数的自家 JIT 核），可挖性远差于 vLLM，见主 README。
