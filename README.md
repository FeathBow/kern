# kern

Model-agnostic GPU executable runtime（背景见 [design.md](design.md)）。

Model provider 交付 `manifest.json + kernels.cubin + weights`，runtime 负责：
加载时像 rustc 一样苛刻地校验 manifest，运行时按声明闭眼执行。runtime
不感知任何模型语义——它只会调度不透明的 kernel dispatch、按字节数供应不
透明的 state、以及对一个封闭的标量表达式集合求值算 launch 几何。

## Manifest 格式（version 1）

顶层五段，全部名字唯一、引用必须解析、不允许未知字段：

- `meta`：格式版本、模型标签（runtime 不赋予含义）、cubin 文件 + sha256。
- `symbols`：runtime 每次调用时提供的标量（如 `tokens`、`pos`），声明
  `min`/`max` 上下界，所有静态校验在界上进行，运行时拒绝越界值。
- `states`：不透明持久状态。**runtime 只知道 `bytes_per_token` 和对齐**；
  内部布局（每层偏移等）是 provider 生成器里的算式，以字面量参数传给
  provider 自己的 kernel。
- `buffers`：`dtype + shape + class`。shape 维度是常量或 symbol 名；
  class 为 `input`（runtime 写入）/ `output`（runtime 读回）/ `weight`
  （按名从权重文件绑定）/ `workspace`（runtime 规划，跨次执行不保留）。
- `kernels`：cubin 内的入口符号 + 类型化参数列表 + block 维度。参数写作
  `"in buffer<bf16>"` / `"out buffer<fp8e4m3>"` / `"inout ptr"`（state 用）
  / `"i32"` 等；buffer/ptr 必须声明方向，方向驱动数据流校验。
- `programs`：每个 program（如 `prefill`/`decode`）是一段顺序 dispatch
  列表。grid 维度用封闭表达式集合表达：常量、`{"sym": s}`、
  `{"ceil_div": [e, c]}`、`{"mul": [e, c]}`——这不是语言，是填空模板，
  永远不会加控制流。

Manifest 是**生成产物**（类比 `Cargo.lock`）：provider 手写的是生成器，
不是 manifest。样例见 `tools/gen_qwen3.py` → `examples/qwen3-4b.json`
（Qwen3-4B，双 program，1014 dispatches，535 KiB）。

## Verifier（`kern-manifest`）

`verify()` 收集全部错误一次报告：

1. 格式版本、hash 形状；
2. symbol 界自洽；
3. state 尺寸非零、对齐为 2 的幂；
4. buffer shape 解析、字节数在 symbol 上界下不溢出；
5. kernel block 不超 CUDA 限制；
6. dispatch：kernel 引用、实参/形参数量与逐位类型匹配（dtype 精确匹配、
   state 只能接 `ptr`、symbol 范围必须装进标量参数类型）、grid 在上界不
   超 CUDA 限制、在下界不为零、除零/乘零常量拒绝；
7. 逐 program 数据流：禁止读未写（read-before-write）、禁止写 input/
   weight、每个 output 必须被写到；
8. 拒绝一切未使用的声明。

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

改 schema 后重新生成样例：`python3 tools/gen_qwen3.py`。

## 后续（未做）

- runtime crate：cubin 加载 + phase-2 参数布局校验、workspace 静态规划
  （liveness + 贪心 offset）、单 stream 顺序执行器、contiguous KV state
  供应；之后换 paged 验证边界隔离论点（应只动 attention kernel 与 state
  声明，计算图不变）。
- state 粒度目前仅 per-token；per-seq 定长（Mamba 类）是已知的 schema
  扩展点。
- 多卡 collective 不是 kernel，将来以少量 runtime 内置 `extern op` 形式
  进入 schema。
