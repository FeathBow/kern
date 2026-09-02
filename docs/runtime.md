# Runtime：`kern-runtime` + `kern run`

## MVP 范围（已定）

- **只做 `decode` 一个 program**：prefill 视为特殊的 decode——prompt 逐 token
  过 decode 路径（tokens=1）。慢但正确，先让端到端闭环；真 prefill program
  之后加。
- **bs=1**，不考虑 batching。（已扩展：`kern-serve` 的 continuous batching 见 [serve.md](serve.md)——decode 多序列，prefill 仍 bs=1。）
- GEMM 走 runtime 特判（cublasLt extern op）；attention 与 reshape_and_cache
  收编 vLLM TRITON_ATTN backend 的 Triton 核；norm/rope/silu 收编 vLLM CUDA
  cubin。自己写的核只有两个（`tools/kernels-src/`）：embedding（trivial
  gather）和 argmax（greedy 采样）。

## 执行器（`crates/kern-runtime`）

`kern-runtime` 是模型无关的执行器（依赖只有 crates.io：cudarc/half/
safetensors/thiserror，可开源）：verify manifest → `kernels/` 下每个
cubin 算 sha256，**只装载 manifest `modules` 表点名的哈希**（其余文件不碰；
`tools/extract_kernels.sh` 按这些哈希从 dump / 手写核 build 里凑齐，落地名
`<module>-<sha12>.cubin`）→ 逐 op 逐 launch 在自己钉的模块里解析 entry
（文件名不参与），**同一模块里同名的 Triton 多 constexpr 实例靠
`cuFuncGetParamInfo` 参数布局与 manifest params 比对来消歧**（phase-2
ABI 校验兼做实例选择，绕开了 capture 缺 launch→module 映射的坑）→ 按
var max 分配全部 buffer / 按 bytes_per_token×capacity（或
bytes_per_seq×(seqs.max+2) 个序列 slot）分配 state →
safetensors 按名绑权重（scratch 按 impl 声明另行私有分配）→ 顺序重放
call 表：接口实参解析一次，逐 launch 按 `args` 连线转发/接 scratch/
填字面量后 raw `cuLaunchKernel`（实参 staging 成小端 u64 slot；>48KB
动态 shmem 自动 `cuFuncSetAttribute`）。state 的 token 容量（`--capacity`）
向下对齐到 manifest 里 `index_into` 该 state 的最大页单位（block table 的
`stride`），不会出现半页；不给（`Runtime::load(.., None)`，kern-serve 的默认）
则在 buffer 和 scratch 都分完之后 `cuMemGetInfo`，剩余显存减 `HEADROOM`
（1 GiB）与定长/per-seq state 之后按 Σ bytes_per_token 折成整页，再封顶到
`seqs.max × 行上限 + 1 页`。`Runtime::lease(tokens)` 一次租下 KV 页和每个
per-seq state 的一个 slot（租时在 stream 上清零），`Lease::seq_line(table,
r)` 给出 line 表的项（宽表 `[lines, seqs, w]` 的格宽由 `seq_width` 给出，
caller 决定 line 落在哪一项）；租约 drop 时一起归还。

**checkpoint（K1）**：`Runtime::checkpoint(&mut lease, len)` 把序列前 `len` 个
token 留成 `Checkpoint`——页进共享链（一页一个引用计数节点，一条序列每页一个
checkpoint 共用一条链，所以 checkpoint 深度再大也只是一个节点），有 per-seq
state 时把 state 拷进一个新 slot（没有空 slot 报 `Denied::Busy`）；
`Runtime::retire(lease, len)` 是请求结束时的零拷贝形式：`len` 之外的页归还，
slot 原样移交。`Runtime::lease_from(&checkpoint, tokens)` 从 checkpoint 起一条
新序列：整页共享，`len` 落在页中间时把那一页拷一份（新序列往里追加，
checkpoint 自己那页不动），state 拷进新 slot；租约的 `prefix()` = `len`，
`slot(pos)` 拒绝 `pos < prefix`。谁拿着句柄谁持有：页在最后一个 lease /
checkpoint drop 时回池，checkpoint 本身不会被 runtime 淘汰。
纯 host 的 `Prefix` 表（`prefix.rs`）按 token 哈希链索引 checkpoint：`lookup(tokens)`
给出覆盖 prompt 真前缀（不含最后一个 token）的最长 checkpoint；序列自己带一条
`Chain`（每 token 折一次，每页记一个头），`insert(&chain, cp)` 读链上对应长度的键，
每页留一个 checkpoint 也只把每个 token 哈希一次；`insert` 去重，
`evict` drop 最久未命中的一个（同一条链最深的先走，drop 叶子才真正还页）；
逻辑时钟计数，不读钟，同样的 token 序列给同样的判定。决策（共享哪些页、拷哪一页、
拷哪个 slot）由 `Pool` 在 host 上算成 `Copies`，runtime 只在 stream 上执行拷贝。
kern run / kern test 仍默认 4096（test 的 workload 抽样以 capacity 为界）。
`extern:cublaslt_bf16_tn` 特判：行主序 `C[m,n]=A[m,k]@W[n,k]^T` 映射成列
主序 `C'=W_cm^T×A_cm`（transa=T、lda=ldb=k、m'=n、ldc=n）；
`extern:cublaslt_bf16_tn_acc` 是同一条路径 β=1（`C += A@W^T`，c 参
`inout`），投机解码的 fc 分块累加与 markov 偏置都靠它，省掉 concat
缓冲和拷贝核。
`extern:cublas_bf16_tn_f32` 同一映射但结果落 **f32**（cublasGemmEx，
`CUBLAS_COMPUTE_32F` / `DEFAULT_TENSOR_OP`，独立 cuBLAS handle + 32 MiB
workspace，可捕获）：K3 的每条稠密投影都是 f32 partial 再由认证的 `k3_land`
核落成 bf16，所以 landing 的舍入链与 pegainfer 一致；权重行带 / 输出列带
用 arg 的字节 `offset` 加第 7 参 `ldc` 表达。

**多卡（E0/K0）**：state 一律 VMM 分配（`cuMemCreate` + `cuMemAddressReserve`
+ `cuMemMap` + `cuMemSetAccess`），设备报 `HANDLE_TYPE_FABRIC_SUPPORTED`
就带 fabric handle（driver 拒绝则退回本地映射并 warn）；`export: true`
的 buffer 同样，但拿不到 fabric handle 就报错。manifest 带 `topology` 时
`Runtime::load(.., Some(&Topology))` 给出本 rank 在每个组的下标（大小
必须与 manifest 一致），`{"rank": g}` 实参在 compile 时烧成常量。
`export_handles()` 返回每个 export buffer / 每个有 handle 的 state 的
`PeerHandle`（64 B fabric handle + 映射字节数，`to_bytes()` 72 B 一行，
传输是 caller 的事：共享盘、TCP 都行）；`import_peers(group, members)`
收全组每个成员的 handle 表（自己那份随意，用本地地址），
`cuMemImportFromShareableHandle` + reserve + map 后把 `u64[组大小]` 写进
该组的每个 `peer` buffer（同一目标只映射一次），映射与 runtime 同寿命。
所有 peer buffer 填满之前 `run`/`run_range`/`capture`/`time_*` 一律
`Api` 错误。装载时收到 peer buffer 的每个 kernel launch 都过
`cuobjdump -sass` 扫描，见 manifest.md 信任边界一节。门禁
`crates/kern-runtime/examples/peer_barrier.rs`：同进程 4 个 Runtime 各占一
卡，一份 SPMD manifest（`tools/kernels-src/peer_barrier.cu`：release 存到
每个成员的 flags、acquire 自旋 + globaltimer 超时、错误码落 output），
tray03 4×GB300 实测 captured burst **3.75 µs/barrier**、eager run+sync
15.8 µs；`--drop r` 让 r 缺席，其余 rank 2 s 内报"等 r 超时"而不是挂住；
换成 multicast bulk copy 的同名 kernel 装载即被拒。

**TMA 描述符与簇 launch**：launch 实参 `{"tensormap": {...}}` 在装载时对
finished 指针（buffer 基址 + call offset）`cuTensorMapEncodeTiled` 成
128 字节镜像（64 字节对齐），launch 时按值塞进参数槽（ABI 校验里它就
是一个 128 字节参数）；`cluster` 走 `cuLaunchKernelEx` +
`CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`（无 cluster 的 launch 也走同一
条路，attrs 为空）。E1 门禁 `crates/kern-runtime/examples/k3_moe_ep.rs`：
K3 pruned 第 1 层 MoE（224 expert，top-16，mxfp4 权重）作为三个 op 的
program（`tools/kernels-src/k3_mega_stage.cu` 的 quant_x / write_routing
往 slab 里写，然后 DeepGEMM MegaMoE 一个 launch：dispatch → L1 → situ →
L2 → combine），slab 是一个 `export` 的 carry buffer，peer 数组由 kernel
在设备上读（`tools/k3-mega/`：fork 只改签名，`SymBuffer` 的偏移表从
`__grid_constant__` 换成 `const int64_t*`）。权重与测试向量由
`tools/export_k3_moe.py` 导出（含 host 参考），manifest 由
`tools/gen_k3_moe.py` 生成（几何与 slab 偏移来自 `k3_mega_layout_dump`）。
tray04 4×GB300 实测（2026-09-02）：EP4 每 rank 64 token **227 µs/层**
（captured），四个 rank 的输出与 EP1（256 token 单卡，733 µs/层）对应行
**逐字节一致**；EP1 对 host 参考 max |err| 0.015、相对 RMS 1.7e-3，
917504 个元素无一超 5%+0.05。

**错误分类**（`kern_runtime::Error`，按"谁需要行动"分变体）：
`ManifestParse`/`ManifestVerify`/`Manifest`（provider 修生成器）、
`KernelArtifact`（cubin 缺失/哈希不符/ABI 不匹配/peer launch 含 multicast
指令，重新抽核）、
`WeightArtifact`（权重与 manifest 不符，重新导出）、`Api`（caller 用法
错误：未知 buffer/program、kind 不符、var 越界、graph env 不一致）、
`Call`（定位 call 表位置并包住底层错误）、`Cuda`/`Driver`/`Blas`
（执行期 CUDA 失败）。

## Caller 契约（`crates/kern-run`）

`kern run` 是 qwen3-4b 的 caller 契约（CLI 用 clap，日志走 tracing 到
stderr，`RUST_LOG` 控制级别，stdout 只出生成文本）：**chunked prefill**——
前 n-1 个 prompt token 按 `--chunk`（默认 512，clamp 到 tokens 上界）切块
连调 `prefill`（每块填 token_ids/positions/slot_mapping 前缀 + seq_lens=
已见数 + cu_seqlens_q=[0,块长]；`write_input` 支持前缀写，尾部 stale 字节
grid 界内永不被读），最后一个 prompt token 走 decode 出首个 logits；此后
每 step 填四个小 input（=pos），block_table 恒等；greedy argmax。权重由
`tools/export_weights.py` 从 HF checkpoint 导出（qkv/gate_up 合并、
cos_sin_cache 预计算、kv_scales 全 1、tied lm_head clone）。

**CUDA graph（默认开，`--eager` 回退）**：tokens=1 下 436 个 call 的
grid/标量实参全是常量，每步只有 4 个小 input buffer 的**内容**变、指针不变
→ 整个 call 表 stream-capture 成一张静态图，H2D 写留在图外，每步一次
`cuGraphLaunch`。graph 按 (program, env) 键控：decode 捕在 tokens=1，
prefill 捕在 tokens=chunk（整块走图、余数块 eager 一次）。要点：capture
不能用 legacy NULL stream（runtime 已改 `new_stream()`）；cublasLt 可被
捕获（workspace 预分配，算法启发式在捕获时定死，顺带省了每步的 CPU
开销）；`run_captured` 校验 env 与捕获时一致（var 值烧死在图里）。

**greedy 采样已下沉 GPU**：`tools/kernels-src/argmax.cu` 两段式行 argmax（64 block
分部归约 + 1 block 收尾；单 block 版 nsys 实测 55.7µs/步——单 SM 读 300KB
只有 5.5GB/s，两段式 5.5µs），平局取最小下标与 CPU 扫描语义一致，折叠成
一个两 launch 的 `argmax` op impl（partial 缓冲是私有 scratch）进
manifest 进 graph；`logits` 降级为 workspace，新增 output `next_token`
i64["tokens"]，每步回读从 300KB 变 8B。input 侧 H2D 走常驻 pinned
staging（pageable 会退化成驱动同步拷贝）。

## 实测（GB300）

输出连贯（"The capital of France is Paris. The capital of Germany is
Berlin. ..."；150 token 长文不劣化，KV 跨页正常），完整 step（含采样回读）
graph 2.7 ms ≈ 377 tok/s、eager ~3.0 ms，两路输出逐 token 一致。对照：
vLLM 0.28 本尊同卡 bs=1（TRITON_ATTN、graph 默认开）2.44 ms/token ≈
409 tok/s——kern ~92%。
**chunked prefill 实测**：709 token prompt 两块（512+197）58–60ms ≈
**12k tok/s**，vs 逐 token 假 prefill 2.18s——TTFT 提升 ~37×；三路交叉
验证（chunk=512 走图 / chunk=1 逐 token / eager）生成逐字节一致，2D 与
3D attention 实例在重叠输入上数值互证。

**nsys 定位（别猜，profile；纯 decode 窗口 + CUPTI 区间求并对账）**：
- GEMM 虽是唯一没从 vLLM 挖的核（nvjet ABI 挖不动，runtime 自己调
  cublasLt），但 heuristic 选出的 nvjet 内核和 vLLM 完全一致、逐核耗时
  持平（128x8 17.1 vs 17.7µs、splitK 7.9 vs 8.6、lm_head GEMV 124 vs
  126µs 已近带宽极限）——GEMM 不是差距。
- 每步 GPU busy：kern 2.25ms < vLLM 2.58ms（我们的 kernel 时间反而短，
  vLLM torch.compile 的 triton 小核并不更快）；差距全在每步边界 GPU
  空转：kern ~174µs vs vLLM ~71µs——我们 sync→8B 回读→4×H2D→graph
  launch 纯串行，vLLM async scheduling 把 host 活藏进 GPU 时间。

## 端到端流程（dump → manifest → run，宿主机裸跑即可，不需要 docker）

```bash
# 1) 挖：CUPTI 注入抓 vLLM（TRITON_ATTN）的全部 cubin + launch ABI 流水
#    （自动建 .venv 装 vLLM；挑张空卡跑，~几分钟）
CUDA_VISIBLE_DEVICES=0 tools/capture_qwen3.sh        # -> dumped-kernels/pid<N>/

# 2) 分析（可选，看切 pass/指针分类/表达式拟合报告）
.venv/bin/python tools/mine_capture.py dumped-kernels/pid<N>/launches.jsonl

# 3) 生成 manifest：真实 ABI + 手写连线，挖矿地址逐项断言证伪
.venv/bin/python tools/gen_qwen3_decode.py dumped-kernels/pid<N>/launches.jsonl
                                                     # -> examples/qwen3-4b.json

# 4) 抽核：按 manifest 钉的 sha256 从 dump 里拷 module、从 target/cubins 拷
#    手写核（tools/build_kernels.sh 编的），落地 <module>-<sha12>.cubin；目录只增不减
tools/extract_kernels.sh examples/qwen3-4b.json dumped-kernels/pid<N>   # -> kernels/
# 或 `kern kernels`：按 kern.toml 的 [kernels].dumps/.sources 给每个 target 的 manifest 与 reference 落 cubin

# 5) 权重：HF checkpoint 合并导出（qkv/gate_up 合并、rope cache 预计算）
.venv/bin/python tools/export_weights.py             # -> weights/

# 6) 跑（构建在 kernel-lab 容器里做；binary 宿主机 dlopen CUDA 直接跑）
./target/release/kern run            # 一切来自 kern.toml 的 target；flag 覆盖：
./target/release/kern run \
  --manifest examples/qwen3-4b.json --kernels kernels \
  --weights weights/qwen3-4b-decode.safetensors --tokenizer weights/tokenizer.json \
  --gpu 3 --capacity 4096 --chunk 512 --prompt "The capital of France is" --steps 320
```

启动输出即配置声明的展示面：manifest 元信息/var/state/buffer 分类统计、
逐 op 逐 launch 的 entry+参数布局+解析到哪个 module（gemm 显示 runtime
built-in）、权重绑定、graph 捕获（436 call → 每步 1 次 graph launch）。
