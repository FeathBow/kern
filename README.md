# kern

Model-agnostic GPU executable runtime（背景见 [design.md](design.md)）。

Model provider 交付 `manifest.json + kernels.cubin + weights`，runtime 负责：
加载时像 rustc 一样苛刻地校验 manifest，运行时按声明闭眼执行。runtime
不感知任何模型语义——它只会调度不透明的 kernel dispatch、按字节数供应不
透明的 state、以及对一个封闭的标量表达式集合求值算 launch 几何。

## Manifest 格式（version 2）

顶层五段，全部名字唯一、引用必须解析、不允许未知字段：

- `meta`：格式版本、模型标签（runtime 不赋予含义）。
- `symbols`：runtime 每次调用时提供的标量（如 `tokens`、`pos`），声明
  `min`/`max` 上下界，所有静态校验在界上进行，运行时拒绝越界值。
- `states`：不透明持久状态。**runtime 只知道 `bytes_per_token` 和对齐**；
  内部布局（每层偏移等）是 provider 生成器里的算式，以字面量参数传给
  provider 自己的 kernel。
- `buffers`：`dtype + shape + class`。shape 维度是常量或 symbol 名；
  class 为 `input`（runtime 写入）/ `output`（runtime 读回）/ `weight`
  （按名从权重文件绑定）/ `workspace`（runtime 规划，跨次执行不保留）/
  `carry`（一个 program 写、另一个 program 读的交接棒，跨次执行保留；
  谁先跑是 caller 契约，verifier 只要求它被某个 program 写到——投机解码
  的 aux 隐状态逼出来的）。
- `kernels`：**接口 + 实现分离**（v2 的核心，为 kernel 可插拔）。
  - **接口**是调用点契约：类型化参数列表——`"in buffer<bf16>"` /
    `"out buffer<fp8e4m3>"` / `"inout ptr"`（state 用）/ `"i32"`/`"u8"`
    等；buffer/ptr 必须声明方向，方向驱动数据流校验。
  - **实现（`impl`）是可整体替换的微程序**：`scratch`（impl 私有工作区，
    dtype+shape 声明，调用方看不见）+ `steps` 顺序 launch 列表。每个
    step 有自己的 cubin 符号、launch ABI（`params`）、block/grid 几何
    （grid 用下述表达式集合；可选 `shared_mem`，上限 227KB opt-in）、
    可选 `cubin` 文件钉定 + `sha256`，以及 `args` 连线：`{"arg": i}`
    转发接口第 i 参 / `{"scratch": name, offset}` 接私有工作区 /
    字面量标量（impl 私有常量）。多数 kernel 是单 step 恒等连线；
    两段式 argmax、vLLM attention（unified + reduce_segments）这类
    "一个逻辑核 = 多次 launch + 私有中间缓冲"整体折叠成一个 impl，
    不再向调用方泄漏。
- `programs`：每个 program（如 `prefill`/`decode`）是一段顺序 dispatch
  列表：`kernel` 名 + 接口实参（buffer/state 实参可带字节 `offset`，
  默认 0：kernel 收到 base+offset——provider 用它寻址融合 buffer 里的
  视图如 qkv 的 q/k/v 切片、state 里的逐层区域，offset 是 provider
  布局算术的字面量，runtime 只做加法）。launch 几何在 impl 里，不在
  dispatch 里。grid 表达式是封闭集合：常量、`{"sym": s}`、
  `{"ceil_div": [e, c]}`、`{"mul": [e, c]}`——这不是语言，是填空模板，
  永远不会加控制流。

**可插拔**：换一个 kernel 的实现 = 只改它的 `impl` 块（可能带上新 cubin
文件），接口、程序连线、其余 manifest 一字不动；verifier 静态把关新
impl 与接口的自洽（方向、dtype、scratch 数据流），runtime 加载时用
`cuFuncGetParamInfo` 比对每个 step 声明的 ABI，`sha256` 钉住工件。这
就是"kernel 市场"的交换单元。

Manifest 是**生成产物**（类比 `Cargo.lock`）：provider 手写的是生成器，
不是 manifest。样例见 `tools/gen_qwen3_decode.py` →
`examples/qwen3-4b.json`（Qwen3-4B，两个 program：`prefill` 433 dispatch
/ `decode` 436 dispatch，真实挖矿 ABI）与 `examples/qwen3-4b-dspark.json`
（同上 + DSpark 投机解码：六个 program，target+draft 权重同处一份
manifest，见下文）。

标量实参除 symbol 和字面量外还可以是表达式（`{"expr": {"mul":
["tokens", 32]}}`，同一封闭表达式语言）——prefill 逼出来的：head-norm
的"总 head 数"参数 = tokens×heads，decode 恒 tokens=1 时它伪装成字面量。

## Verifier（`kern-manifest`）

`verify()` 收集全部错误一次报告：

1. 格式版本；
2. symbol 界自洽；
3. state 尺寸非零、对齐为 2 的幂；
4. buffer shape 解析、字节数在 symbol 上界下不溢出；
5. kernel impl 逐 step：block 不超 CUDA 限制、grid 在 symbol 上界不超
   CUDA 限制/下界不为零、`sha256` 形状且必须伴随 `cubin`、step args 与
   step params 数量/逐位类型匹配；
6. impl 与接口的自洽：step 不得写穿接口 `in` 参、接口 `out` 参必须被某个
   step 写到、scratch dtype/offset 对界检查 + 跨 step 数据流（禁止读未写
   的 scratch）、未使用的 scratch 拒绝；
7. dispatch：kernel 引用、实参与**接口**参数逐位匹配（dtype 精确匹配、
   state 只能接 `ptr`、symbol/表达式的取值范围必须装进标量参数类型、
   offset 对齐且在界内）；
8. 逐 program 数据流：禁止读未写（read-before-write）、禁止写 input/
   weight；output / carry 必须被**某个** program 写到（prefill 这类只落
   state 的 program 合法地不写任何 output；carry 在每个 program 内视为
   已写——它的生产者是另一个 program）；
9. 拒绝一切未使用的声明。

反序列化层已拒绝：未知字段、重复名字、非法参数类型串。

**信任边界**：verifier 证明的是"声明自洽"，不是 kernel 行为。谎报自己
读写范围的 cubin 在边界之内被信任（debug 路径可用 compute-sanitizer 兜
底）。加载 cubin 后用 `cuKernelGetParamInfo` 比对参数个数/字节布局属于
runtime crate 的 phase-2 校验（`ParamType::size_bytes` 为此预留）。

## 构建 / 测试

host 无 cargo，在 kernel-lab 容器内构建：

```bash
docker exec kernel-lab bash -c 'export PATH=/root/.cargo/bin:$PATH && cd /work/kern && cargo test'
```

改 schema 后重新生成样例：`.venv/bin/python tools/gen_qwen3_decode.py`
（需要 capture dump）。tools/ 流水线全貌见 `tools/README.md`。

## Kernel 来源：从 vLLM 挖（kernel mining）

MVP 的 cubin 不自己写，从 vLLM 生产路径里挖——挖出来的是 GB300 上真正在
跑的一流核。工具 `tools/kernel-capture/`（vendored from pegainfer PR #982，
Apache-2.0）是一个 CUPTI 注入库：`CUDA_INJECTION64_PATH` 挂进任意 CUDA 进程，
把每个 module 的 cubin、每次 launch 的 grid/block/attrs 和完整 staged 参数
（含指针→device/host + 所属 allocation range 归类）dump 到
`dumped-kernels/pid<N>/`（`module_*.cubin` + `launches.jsonl`）。host `cc`
直编（CUPTI 在 `/usr/local/cuda/targets/sbsa-linux`）。

`tools/capture_qwen3.sh`：uv venv 装 vLLM 0.28.0，`enforce_eager=True`（每个
launch 都是真实 `cuLaunchKernel`、staged 参数可读），4 个长度递增的 prompt
逐个发（bs=1）、各出 4 token。两个必须的环境设置：
`VLLM_ENABLE_V1_MULTIPROCESSING=0`（engine 跑主进程，注入才作用到真正 launch
的 pid，stderr 也不被子进程吞），PATH 里要有 ninja/nvcc（FlashInfer 运行时
JIT attention 核）。Qwen3-4B 多 prompt 实测挖到 **79 cubin / 13042 launch**
（`dumped-kernels/pid5992`，带 `t_ns`；旧的 pid4185422 是单 prompt 无时间戳版）。

### dump 的结构：绝大多数 launch 是 dummy pass——而这是特性

单 prompt 实测里，rms_norm 的 grid 分布显示总共 ~8 个 forward pass，真实请求
只占 prefill(grid=2) + decode(grid=1)×2 三个；其余全是 vLLM 的
memory-profiling（16384 token 满配）和 warmup dummy pass。两个推论：

- 生成器第一步必须**切 pass**。capture 每条 launch 记 `t_ns`（CLOCK_MONOTONIC）。
  多 prompt dump 实测：**>5ms 空隙切到请求级**（4 个请求干净分离，final-norm
  的 grid.x 与 prompt token 数 1/34/85/175 逐一对上）；但请求内 prefill 与各
  decode step 之间最大空隙只有 ~1.7ms 且边界有偏移，**请求内的 forward pass
  要按锚点核（final-norm/采样簇，或按 launch 序列的周期性）切**，不能靠时间。
- dummy pass 不是噪声，是**表达式拟合的采样点**：真实 pass 的 shape 太退化
  （tokens=1/2 时 `ceil_div(tokens,c)` 和 `tokens` 无法区分），多个不同 token
  数的 pass 才能把 grid 表达式定下来。capture 脚本发多长度 prompt 就是为了
  加采样点。

另一个粒度陷阱：**allocation range ≠ 逻辑 buffer**。K/V 两个页池是同一个
3.2GB range 内的不同偏移（PyTorch allocator 一次 malloc 内部切分）；
slot_mapping/positions 等小 buffer 全挤在同一个 2MB arena。buffer 身份要用
`(range_start, offset)` + 跨 pass 指针稳定性（跨 pass 不变 = weight/持久
state，随 pass 变 = activation workspace）来推，不能只看 range。

### 关键发现：两个主力核挖得到、却 replay 不了

capture 按 kernelParams 逐参数拆分，对 flat ABI 的核完美——但恰好最重要的两个
核不是 flat ABI：

- **GEMM**（`nvjet_sm103_tst_*`，11 个变体）走 `cuLaunchKernel` 的 packed
  `extra` 缓冲区传参，不走 kernelParams → capture 只能记 `params: null`，
  参数指向抓不到。（cublasLt `splitKreduce` 反而是 flat ABI、21 参数全抓到，
  但它有两个 host 指针参数（pointer-mode alpha/beta）被当成普通 8 字节标量
  记录、pointee 没 dump——**host 指针不解引用是 capture 的系统性盲点**，值形如
  `0xffff...` 才能肉眼认出；且它只跟在 nvjet splitK 后面跑，孤立 replay 无意义。）
- **Attention** 是两个核：prefill 用 `fmhaSm103aKernel_...PersistentContext`，
  decode 用 `fmhaSm100fKernel_...SwapsAbForGen`（sm100f binary 跑在 sm103 上）。
  都只有 **1 个 1280 字节 packed struct 参数**，指针全埋在 struct 内部。字段
  布局在 flashinfer（trtllm-gen `KernelParams`）源码里是公开的，理论可 rebind，
  但版本锁死、随 flashinfer 升级漂移，不依赖。

flat ABI 的核全部可直接 replay：`rms_norm`(12 param)、`rotary_embedding`(13)、
`act_and_mul`/silu(6)、`reshape_and_cache_flash`(16)。注意模板实例化按
call-site 的**静态维度**选择（rms_norm 的 hidden=2560 与 head_dim=128 是两个
不同 symbol），不随 tokens 变——per-dispatch pin symbol 是安全的；但 vec-width
路径假设指针 16B 对齐，runtime 分配 buffer 对齐别低于 vLLM（保守按 256B）。

**结论**：mining 用作 ABI 参考 + 收编 flat 核；**GEMM 特判**——runtime 以内置
extern op 形式直接调 cublasLt（不啃 nvjet ABI，也是将来 collective extern op
的先例）；**attention 收编 vLLM TRITON_ATTN backend 的 Triton 核**（flat
kernelParams，可同法挖、可校验、可 rebind）。

### attention backend ABI 普查（探针实测，`tools/capture_abi_probe.sh`）

vLLM 0.28 起选 backend 只能用 `AttentionConfig(backend=...)`（
`VLLM_ATTENTION_BACKEND` 环境变量已删除，设了静默无效）。GB300 上三家的
launch ABI：

- **FLASHINFER（默认）**：trtllm-gen fmha，1 个 1280B struct → 不可 rebind。
- **FLASH_ATTN = FA4**（FA3 仅 sm90，FA2 无 Blackwell 核）：CuTe-DSL 核，
  19 参数全是 12–128B 的 packed struct/TMA descriptor，**0 个裸指针**——
  rebind 需要 host 侧按真实地址重编 TMA descriptor，比 trtllm-gen 更不可收编。
- **TRITON_ATTN**：`kernel_unified_attention`（prefill/decode 统一）+
  `reduce_segments`（decode split-KV 归并）+ Triton 版 reshape_and_cache，
  **全部 flat ABI**，KV 布局的 stride 全是普通标量参数。bs=1 decode 探针
  吞吐三家几乎持平（15.4/15.2/13.9 tok/s）——decode 是带宽活。

**Triton 同名 ≠ 同 ABI**：unified 的 2D(prefill) 实例 28 参数、3D(decode)
实例 31 参数；`reduce_segments` 的 `num_seqs` 无类型注解、值为 1 时被
Triton 特化进 binary（不再是运行时参数）。收编必须 pin 具体实例，且
capture 目前不记 launch→module 映射，同名多实例分不清出自哪个 cubin
（capture 待补 module id）。

### 跨框架实测：dump SGLang（capture 不挑框架）

同一注入库对 SGLang 直接可用（`tools/capture_sglang.sh`，docker 镜像里跑
——pip 装不通：pypi 的 sgl-kernel 轮子链 CUDA 12 库，与 aarch64 必需的
cu130 torch 冲突）。GB300 上 Qwen3-4B 抓到 55 cubin / 10079 launch，
`mine_capture.py` 无改动切出 3 个 forward、tokens 参照 34/85/175 全对。

但 SGLang 的 ABI 面貌比 vLLM 难挖得多——**几乎全员 struct 化**：

- attention：trtllm-gen fmha（`fmhaSm100f` decode / `fmhaSm103a` prefill），
  1 个 1152B packed struct，同 vLLM FLASHINFER 后端，不可 rebind；
- GEMM：全 nvjet（packed `extra` 缓冲区，`params:null`），与 vLLM 同款；
- 自家 JIT elementwise 核（fused_qknorm/fused_rope/store_kvcache/
  act_and_mul）：**单个 40–80B struct-by-value 参数**，指针全埋在 struct 里；
- 连 flashinfer 的 norm 核参数也是 16B tensorptr 结构（指针+元数据）。

结论：vLLM 至少有 TRITON_ATTN 全 flat 的收编路径，SGLang 只有 norm/
splitKreduce 是多参数 flat；struct 布局都在源码里公开，但逐版本漂移，
rebind 即锁版本。挖矿目标选 vLLM 是对的。

### KV 布局：vLLM 是 paged（实锤）

`reshape_and_cache_flash` 的 ABI 暴露了 vLLM 的 KV 模型：新 token 的 K/V 写进
两个 paged 池，靠 `slot_mapping` buffer 定位；`block_size=16`、`kv_heads=8`、
`head_dim=128`。attention 再吃页池 + block_table。写 KV 与读 KV 是两个独立核。

对我们的 schema：`state<KV>` 在边界处解构出来的不是单指针，而是「页池 ptr +
runtime 每 step 填的 `block_table`/`slot_mapping`/`seq_lens` 小 buffer」。state
声明从「`bytes_per_token` 一个数」演进为「runtime 要替我维护哪几样东西」的清单
——runtime 依旧不知道池子里是 K 还是 V，边界不破，只是更具体。

## MVP 范围（已定）

- **只做 `decode` 一个 program**：prefill 视为特殊的 decode——prompt 逐 token
  过 decode 路径（tokens=1）。慢但正确，先让端到端闭环；真 prefill program
  之后加。
- **bs=1**，不考虑 batching。
- GEMM 走 runtime 特判（cublasLt extern op）；attention 与 reshape_and_cache
  收编 vLLM TRITON_ATTN backend 的 Triton 核；norm/rope/silu 收编 vLLM CUDA
  cubin。自己写的核只有两个（`tools/kernels-src/`）：embedding（trivial
  gather）和 argmax（greedy 采样）。

## 生成器分析前端：`tools/mine_capture.py`（已做）

三步全自动，无模型知识输入：`launches.jsonl` → 人读报告 + `--json` 全量结果。

1. **切 pass**：时间空隙切请求窗口；窗口内按 core 核（全局最高频核，每
   forward 一个密集 burst）的 burst 空档中点切 forward。边界有 ±1 launch
   抖动，call-site 对齐按**众数出现次数**容错。
2. **tokens 参照自验证**：tokens = 某个每 forward 恰一次、grid.x 变化的核
   的 grid.x。选哪个不靠先验——对每个候选试拟合全部 site，取「拟合失败+
   欠定」最少者（选错参照会大面积拟不出/欠定，数据自己会说话）。实测自动
   选中 final-norm。
3. **对齐 + 分类 + 拟合**：call site = (symbol, forward 内出现序号)
   （enforce_eager 下 launch 序列确定）。指针按 (range_start, offset) 跨
   forward 稳定性分三类；grid 各轴与标量参数按封闭表达式集合拟合。

pid5992 实测结果：5 个 flat 核全部拟出（grid `tokens`/`tokens*32`/`tokens*8`，
7 个 token 采样点），指针分类与人工判读一致（逐层 KV 池、slot_mapping 全局
persistent、逐层权重、workspace）。仅有的拟不出标量是 **stride 且 prefill/
decode 取值不同**（reshape_and_cache 的 value stride：vLLM decode 1024=
连续、prefill 6144=qkv 融合视图）——在我们的布局里 v 恒在融合 qkv 中，
取 6144 即两个 program 共用的常量（decode tokens=1 时 stride 无效，当初
照抄 1024 也"对"，prefill 把真值逼了出来）。

## 生成器后端：`tools/gen_qwen3_decode.py`（已做）

产出 `examples/qwen3-4b.json`（310 buffer；`prefill` 433 dispatch +
`decode` 436 dispatch），**过 verifier**（`qwen3_decode_mined_verifies`
测试）。数据源是 TRITON_ATTN capture（`dumped-kernels/pid3977275`）。
bs=1，两个 program：

- 七个真核：rms/rms_head/rope/fused（vLLM CUDA cubin）+ Triton 版
  reshape_and_cache + `kernel_unified_attention`（decode 用 3D split-KV
  实例 31 参数 + `reduce_segments` 12 参数；prefill 用 2D 实例 28 参数）。
  真实 symbol、逐参数类型/方向表、标量字面量逐位取自代表性 decode/prefill
  forward。
- 连线是手写的（provider 知道模型），**挖矿数据负责证伪**：发射前断言
  q/k/v 在 qkv 融合 buffer 中的视图偏移（+0/+8192B/+10240B，q/k norm
  out-of-place）、residual 全程同址、6×36 个逐层权重指针互异、KV 池
  k/v 相距 256B、cache 与 attention 共享同一 KV 池与 k/v scale 指针、
  unified 与 reduce 共享 segm 部分和缓冲、eps 一致。
- KV state 从 vLLM 逐层池改为层交织 `[page][layer][16][8][2][128]`——
  同一批 kernel 靠 page-stride 参数 ×36 和 state offset 字面量
  （layer×65536）适配，bytes_per_token=147456，runtime 仍只知一个数。
- decode attention 是一个两 step impl：unified 3D 写 f32 segm 部分和
  （impl 私有 scratch），reduce_segments 归并到 attn_out——调用方只见
  28 参数的 `attn` 接口，一次 dispatch/层。**prefill attention
  （`attn_prefill`）是同一接口的另一份实现**：2D 实例单步无 scratch，
  其 28 参 launch ABI 恰好就是接口本身（接口切分正确的实证）；grid =
  `[ceil_div(tokens,4), 8, 1]`（两个真实 prompt 长度拟合证实）。
  seq_lens/cu_seqlens_q 是 caller 每次调用前填的 input buffer。
  rms_norm_head 因 q/k 调用点 grid 不同（×32 vs ×8，v2 里 grid 归
  impl）拆成 qhead/khead 两个 kernel 条目。
- **prefill program = decode 去掉 final_norm/lm_head/sample 尾巴**（只落
  KV），chunked prefill = caller 连调 prefill + 最后一个 prompt token 走
  decode 出首个 logits——"prefill_last"就是 decode，免掉 symbol 依赖的
  offset。三个 decode 时被 tokens=1 掩盖的字面量由 prefill capture 现形
  并修正：head-norm 输入行距=6144（融合 qkv）、head-norm 总 head 数 =
  tokens×heads（表达式标量）、cache value 行距=6144。logits/next_token
  定常形状 `[1,·]`、decode 专属 kernel（attn 3D/argmax）grid 与 scratch
  定常——否则按 CHUNK_MAX=2048 上界要多付 ~1GB。
- GEMM dispatch 用 `extern:cublaslt_bf16_tn` 符号（runtime 按前缀特判为
  cublasLt matmul）；embedding 是唯一手写核（`tools/kernels-src/embedding.cu`，
  `kern_embedding_i64_bf16`，20 行 gather）。

这一步倒逼出的 schema 演进（都已进 verifier）：buffer/state 实参字节
offset、`u8` 标量（rope 的 1 字节 bool 参数）、`shared_mem` 上限检查。

## Runtime：`crates/kern-runtime` + `kern-run`（已做，端到端通）

`kern-runtime` 是模型无关的执行器（依赖只有 crates.io：cudarc/half/
safetensors/tokenizers，可开源）：verify manifest → 加载 `kernels/` 下全部
cubin（`tools/extract_kernels.sh` 从 dump 抽 manifest 用到的 module +
nvcc 编 embedding）→ 逐 kernel 解析符号——**同名 Triton 多 constexpr 实例
靠 `cuFuncGetParamInfo` 参数布局与 manifest params 比对来消歧**（phase-2
ABI 校验兼做实例选择，绕开了 capture 缺 launch→module 映射的坑）→ 按
symbol max 分配全部 buffer / 按 bytes_per_token×capacity 分配 state →
safetensors 按名绑权重（scratch 按 impl 声明另行私有分配）→ 顺序重放
dispatch 表：接口实参解析一次，逐 step 按 `args` 连线转发/接 scratch/
填字面量后 raw `cuLaunchKernel`（实参 staging 成小端 u64 slot；>48KB
动态 shmem 自动 `cuFuncSetAttribute`）。step 钉了 `cubin` 就只在该文件
里解析，带 `sha256` 则加载时校验文件哈希——可插拔工件的完整性检查。
`extern:cublaslt_bf16_tn` 特判：行主序 `C[m,n]=A[m,k]@W[n,k]^T` 映射成列
主序 `C'=W_cm^T×A_cm`（transa=T、lda=ldb=k、m'=n、ldc=n）；
`extern:cublaslt_bf16_tn_acc` 是同一条路径 β=1（`C += A@W^T`，c 参
`inout`），投机解码的 fc 分块累加与 markov 偏置都靠它，省掉 concat
缓冲和拷贝核。

`kern-run` 是 qwen3-4b 的 caller 契约：**chunked prefill**——前 n-1 个
prompt token 按 `--chunk`（默认 512，clamp 到 tokens 上界）切块连调
`prefill`（每块填 token_ids/positions/slot_mapping 前缀 + seq_lens=已见
数 + cu_seqlens_q=[0,块长]；`write_input` 支持前缀写，尾部 stale 字节
grid 界内永不被读），最后一个 prompt token 走 decode 出首个 logits；此后
每 step 填四个小 input（=pos），block_table 恒等；greedy argmax。权重由
`tools/export_weights.py` 从 HF checkpoint 导出（qkv/gate_up 合并、
cos_sin_cache 预计算、kv_scales 全 1、tied lm_head clone）。

**CUDA graph（默认开，`--eager` 回退）**：tokens=1 下 436 个 dispatch 的
grid/标量实参全是常量，每步只有 4 个小 input buffer 的**内容**变、指针不变
→ 整个 dispatch 表 stream-capture 成一张静态图，H2D 写留在图外，每步一次
`cuGraphLaunch`。graph 按 (program, env) 键控：decode 捕在 tokens=1，
prefill 捕在 tokens=chunk（整块走图、余数块 eager 一次）。要点：capture
不能用 legacy NULL stream（runtime 已改 `new_stream()`）；cublasLt 可被
捕获（workspace 预分配，算法启发式在捕获时定死，顺带省了每步的 CPU
开销）；`run_captured` 校验 env 与捕获时一致（symbol 值烧死在图里）。

**greedy 采样已下沉 GPU**：`tools/kernels-src/argmax.cu` 两段式行 argmax（64 block
分部归约 + 1 block 收尾；单 block 版 nsys 实测 55.7µs/步——单 SM 读 300KB
只有 5.5GB/s，两段式 5.5µs），平局取最小下标与 CPU 扫描语义一致，折叠成
一个两 step 的 `argmax` kernel impl（partial 缓冲是私有 scratch）进
manifest 进 graph；`logits` 降级为 workspace，新增 output `next_token`
i64["tokens"]，每步回读从 300KB 变 8B。input 侧 H2D 走常驻 pinned
staging（pageable 会退化成驱动同步拷贝）。

**实测（GB300，`--gpu 3`）**：输出连贯（"The capital of France is
Paris. The capital of Germany is Berlin. ..."；150 token 长文不劣化，KV
跨页正常），完整 step（含采样回读）graph 2.7 ms ≈ 377 tok/s、eager
~3.0 ms，两路输出逐 token 一致。对照：vLLM 0.28 本尊同卡 bs=1
（TRITON_ATTN、graph 默认开）2.44 ms/token ≈ 409 tok/s——kern ~92%。
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

**端到端流程（dump → manifest → run，宿主机裸跑即可，不需要 docker）**：

```bash
# 1) 挖：CUPTI 注入抓 vLLM（TRITON_ATTN）的全部 cubin + launch ABI 流水
#    （自动建 .venv 装 vLLM；挑张空卡跑，~几分钟）
CUDA_VISIBLE_DEVICES=0 tools/capture_qwen3.sh        # -> dumped-kernels/pid<N>/

# 2) 分析（可选，看切 pass/指针分类/表达式拟合报告）
.venv/bin/python tools/mine_capture.py dumped-kernels/pid<N>/launches.jsonl

# 3) 生成 manifest：真实 ABI + 手写连线，挖矿地址逐项断言证伪
.venv/bin/python tools/gen_qwen3_decode.py dumped-kernels/pid<N>/launches.jsonl
                                                     # -> examples/qwen3-4b.json

# 4) 抽核：从 dump 的 module 里拷 manifest 用到的 cubin + nvcc 编两个自写核
tools/extract_kernels.sh dumped-kernels/pid<N>       # -> kernels/

# 5) 权重：HF checkpoint 合并导出（qkv/gate_up 合并、rope cache 预计算）
.venv/bin/python tools/export_weights.py             # -> weights/

# 6) 跑（构建在 kernel-lab 容器里做；binary 宿主机 dlopen CUDA 直接跑）
./target/release/kern-run \
  --manifest examples/qwen3-4b.json --kernels kernels \
  --weights weights/qwen3-4b-decode.safetensors --tokenizer weights/tokenizer.json \
  --gpu 3 --capacity 4096 --chunk 512 --prompt "The capital of France is" --steps 320
```

启动输出即配置声明的展示面：manifest 元信息/symbol/state/buffer 分类统计、
逐 kernel 逐 step 的符号+参数布局+解析到哪个 cubin（gemm 显示 runtime
built-in）、权重绑定、graph 捕获（436 dispatch → 每步 1 次 launch）。

## DSpark 投机解码（已做，`--spec`）

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
- caller 侧一轮（`kern-run --spec`）：draft（graph）→ 读 7 token → verify
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
# 合并权重（target + draft.*，fc 按列切块、融合 KV cat、markov 头原样）
.venv/bin/python tools/export_weights.py             # -> weights/qwen3-4b-dspark.safetensors
./target/release/kern-run --manifest examples/qwen3-4b-dspark.json \
  --weights weights/qwen3-4b-dspark.safetensors --spec --steps 320
```

## 后续（未做）
- capture 补 launch→module id 映射（unified 双实例现靠 num_regs+cuobjdump
  间接定位，capture 直接记 module id 更干净）；生成器给自写核（argmax/
  embedding）也填 `sha256`（unified 双实例已钉哈希）。
- workspace 静态规划（liveness + 贪心 offset 复用；现在逐 buffer 独立分配）。
- 性能收尾（vs vLLM 差的 ~0.25ms/step = 每步边界 GPU 空转）：token 反馈
  闭环进 graph——embedding 的 token_ids 直接由 next_token 喂（图内 D2D
  或 embedding 改读 next_token），positions/slot_mapping/seq_lens 可预知
  提前写，host 滞后一步异步取结果，步间不再 sync。kernel 时间本身已比
  vLLM 短（GEMM 打平、attention 更快），不用动。
- state 粒度目前仅 per-token；per-seq 定长（Mamba 类）是已知的 schema 扩展点。
- 多卡 collective 不是 kernel，将来以少量 runtime 内置 `extern op` 形式进入
  schema。
