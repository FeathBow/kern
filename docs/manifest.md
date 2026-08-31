# Manifest 格式（version 2）与 Verifier

Model provider 交付 `manifest.json + kernels.cubin + weights`，runtime 负责：
加载时像 rustc 一样苛刻地校验 manifest，运行时按声明闭眼执行。runtime
不感知任何模型语义——它只会调度不透明的 kernel dispatch、按字节数供应不
透明的 state、以及对一个封闭的标量表达式集合求值算 launch 几何。

## 顶层结构

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

**Registry ref**：step 的 `cubin` 除本地文件名外可写
`hf:<org>/<repo>/<path>[@revision]`（revision 默认 `main`），此时
`sha256` 必填——runtime 加载时把它物化进内容寻址缓存
（`$KERN_CACHE_DIR` 或 `~/.cache/kern` 下 `blobs/<sha256>`，命中免网
络），下载后先验哈希再落盘，**传输通道零信任**：名字只是 URL，身份是
哈希。工件可以是裸 cubin，也可以是 host 共享库（如 HF kernel hub 的
torch 扩展 .so）：runtime 剖开 ELF 取 `.nv_fatbin` 里的设备代码逐容器
装载，torch/python 绑定整个丢弃，符号 + ABI 逐位核对照旧。实证：
`examples/qwen3-4b.json` 的 `silu_mul` impl 直接指向
`hf:kernels-community/activation`（PyTorch 生态在用的原装包），输出与
挖矿基线逐字节一致。

Manifest 是**生成产物**（类比 `Cargo.lock`）：provider 手写的是生成器，
不是 manifest。样例见 `tools/gen_qwen3_decode.py` →
`examples/qwen3-4b.json`（Qwen3-4B，两个 program：`prefill` 433 dispatch
/ `decode` 436 dispatch，真实挖矿 ABI）与 `examples/qwen3-4b-dspark.json`
（同上 + DSpark 投机解码：六个 program，target+draft 权重同处一份
manifest，见 [spec-decode.md](spec-decode.md)）。

标量实参除 symbol 和字面量外还可以是表达式（`{"expr": {"mul":
["tokens", 32]}}`，同一封闭表达式语言）——prefill 逼出来的：head-norm
的"总 head 数"参数 = tokens×heads，decode 恒 tokens=1 时它伪装成字面量。

## Verifier（`kern-manifest`）

`verify()` 收集全部错误一次报告（`VerifyErrors`）：

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
底）。加载 cubin 后用 `cuFuncGetParamInfo` 比对参数个数/字节布局属于
runtime crate 的 phase-2 校验（`ParamType::size_bytes` 为此预留）。
