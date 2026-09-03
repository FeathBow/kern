# 多卡 runtime：TP / EP / 二者组合（设计 + 实测 + 待决）

状态：设计稿，2026-09-01；09-02 按 `dcp-bench.md` 的实测与拍板修订（见文末“已决”）。实测全部在 GB300 NVL72（pod4-gb300-3，driver
13030，CUDA 13.1）上做，bench 源码在
本地存档 `bench-results/2026-09-01-graph-launch/`（不在仓库）（`bench.cu` 图提交、`fabric.cu`
跨 tray P2P、`a2a.cu` dispatch 模拟、`hostmem.cu`/`hostnuma.cu`/
`egmverify.cu` 是 KV 分层那一半的，另见后续文档）。

## 结论先行

1. **runtime 不变：单设备、单线程、每卡一个 `Runtime`。** 多卡不是 runtime
   的属性，是 caller 拓扑（一个进程驱动 N 个 runtime）+ manifest 的属性。
2. **"谁提交 kernel"不是问题。** `cudaGraphLaunch` 的 host 开销 1–3 µs
   且与图大小无关；8 张 436-node 的图单线程顺序提交 13 µs，与 8 线程逐 µs
   打平。**不加线程。**
3. **collective 是 kernel，不是 runtime 功能。** all-reduce / dispatch /
   combine 是 manifest 里拿 peer 指针的普通 op；runtime 只多一种 buffer
   来源（`peer`）和一种分配方式（可导出的 VMM 分配）。tray 内 NVLink 和
   跨 tray MNNVL fabric 是同一种指针。
4. **step 边界的 host 串行路径**（sync → 8 B 回读 → 4×H2D → launch ≈ 174 µs）
   对小模型是每步的 15–20%，对 K3 decode（一步 13–30 ms）不到 1%。它不是多卡的
   第一刀；多卡的实际税是 collective 本身（跨卡 barrier 3.8–4.8 µs 一个，EP36
   dispatch+combine 外推每步 5–7 ms，DCP4 每步 5–8%，见 `dcp-bench.md`）。
5. **GPU 能自提交**（device graph tail-launch 在 CUDA 13 / GB300 上验证可
   跑），它省的是"host 不在环上"——这在多卡下的价值是**时钟**（launch count 由
   GPU 推进，见"修正"末段），不是 launch 开销。

## 修正（同日晚，对照 pegainfer `docs/models/k3/cp-lane-design.md` 之后）

上文以 TP 起手是错的顺序。K3 这类 MLA + 线性注意力混合的 MoE 模型里 attention-TP
一分不买（pegainfer 实测：kv_a@TP8 447 TF vs EP 2380；KCP4 比 TP4 快 2.7%；MLA
decode 切头不切 latent 读），pegainfer 已把 TP 从设计里去掉。kern 的多卡顺序改为：

1. **EP 固定拓扑**（expert ownership / graph 永不因请求变形；dispatch/combine 是
   拿 peer 指针的 kernel）；
2. **CP 作为 per-sequence 的弹性数据 lane**（一条序列的 context 分段住在多个
   rank；whale 长 prefill 是一个 gang，与 decode lane 在同一 EP superstep 共存；
   P/D 不分离，靠 lane 混合）；
3. TP 只给小 dense 模型垫底，不驱动设计。**（2026-09-03 修正：这条说重了。**
   attention-TP 仍然不买——切头不切 latent 读；但 K3 decode 每步的另一半是权重读
   （EP32 每 rank 147 GiB ≈ 30 ms 地板，`agent-workload.md` §2 重算表），tray 内
   dense（含 KDA）TP4 把它压到 ~18 ms，是量级上的收益。最终形态：**attention DP +
   dense/KDA tray 内 TP4 + expert 跨 tray EP**，manifest 用下文"TP × EP 组合"的两个
   组表达。）

**统一抽象：段的可结合 partial + merge。** softmax attention 的 partial 是
(O, LSE)，merge 是 online-softmax；线性注意力（KDA）的 partial 是仿射包 (M, D)，
merge 是复合 (MM', DM'+D')。两者可结合 ⇒ 段可以任意分布、任意顺序 merge。
CP prefill（条带 FMHA）、distributed-context decode（query 广播 + partial 回收）、
prefix-hit ingest、P/D 交接、跨卡 restore 全是同一个运算：**新 token 的 forward =
各段 partial 的 merge**，段在哪张卡只决定谁算 partial。这才是"整机架一样"的正确
形式——不是内存一样（decode 仍要本地 HBM），是**运算对位置不变**。

代价提醒：KDA 的 (M, D) 每段每层 12.6 MB × 69 层 ≈ 870 MB，比该段 MLA latent
（4k 段 × 24 层 × 576 B ≈ 56 MB）大 15×——线性层的段不能像 TokenLake 那样中段
复用，只能做**前缀对齐的 checkpoint（存 S，不存 M,D）**。agent 负载是 append-only
的分支树，前缀对齐够用；checkpoint（KDA 状态快照 + latent 页列表 + 位置 + commit
epoch）因此是 serve 层的一等抽象，不是优化。

**kern 与 pegainfer 的分工。** CP lane 里模型无关的"基底"——fabric slab +
doorbell（SM store 到远端、本地 wait；**GB300 stream memops 引擎拒绝 fabric
导入的 VA**，`cuStreamWriteValue` 报 INVALID_VALUE，SM store / DtoD 正常）、launch
count 当全局时钟、event 门控窗口、bucket 化 graph——就是本文的 `export`/`peer`/
flag + caller 的 clock。模型专属的"内容"——FlashKDA doctored 调用、M+D 融合核、
条带 FMHA、lse_merge、mega dispatch——全是 manifest 的 op；pegainfer 的 capsule
substrate（vendored cubin + `cuFuncGetParamInfo` fail-closed ABI walk）与 kern 的
module 表是同一个东西。**第一个多卡 target 定为 K3 pruned@EP4 单 tray 的一个
superstep**：program = 一个 superstep，段长是 per-rank var，stripe 掩码是
`{"rank"}` 派生标量，kernel 全部来自现成 cubin；gang / whale / leveling 留在
pegainfer 当 caller。

**阶段 2 的真正收益点**：pegainfer 的 ~270 ms 协调截距和 armed 期 decode-only 的
根因是"launch count 是时钟，但只有 host 能推进它"。图自 tail-launch、迭代计数器
在 GPU 上、描述符按迭代号投递到每 rank 的 descriptor slab + doorbell，则所有 rank
的计数天然每步 +1，arming 不再需要，slack 从 4 个 period 降到 1。这比 launch
开销大两个数量级，是 GPU 自转该被立项的理由。

**并行的命名，按 workload 分析（`agent-workload.md`）定稿**：一步 = 一批 span
（每序列一段 n ≥ 1 的连续新 token：decode 是 n=1，extend/whale 是 n=chunk），
attention 一律 = 本地自身三角 + 条带前缀 partial + merge。**context 切（DCP）对每个
span 都用，query 不切（不做空间上的 PCP）**；冷 whale（命中 0，占 whale 主体）的
新 token 侧 dense/MoE 工作若压垮 owner，先用"按 chunk 轮换 owner"这种时间上的
PCP（KDA 状态跟着走或留在 home），空间 PCP（pegainfer 的 gang）只在 TTFT SLO
逼到那一步才立项。

**落地梯子（EP 先行）**：
1. EP 机制在 DeepSeek-V2-Lite（pegainfer 有，16B/64 expert/MLA，单卡也放得下）上
   做 EP4 单 tray：`export`/`peer`/`{"rank"}` + dispatch/combine kernel（先写最简单的
   peer 指针版，不上 mega）+ 分 rank 权重导出；门禁 = EP4 vs EP1 逐位一致
   （pegainfer Gate 1 的形态）。
2. K3 pruned@EP4 单 tray 一个 superstep（capsule cubin），与 full@EP16 同构
   （56 expert/rank）。
3. 跨 tray EP8/16：只换 handle 交换，kernel/manifest 不动。
4. DCP：同一 peer 机制 + attention partial/merge op；然后 checkpoint、pool daemon。

下面的 TP 小节保留作 dense 模型参考；EP / CP 的 manifest 形状按上述抽象定。

## 实测

### 图提交（单卡，链式 1-block kernel）

| nodes | kernel 时长 | host launch (µs) | GPU 内每 node (µs) | roundtrip (µs) |
|---|---|---|---|---|
| 1 | 0 | 1.2 | 4.1 | 8.0 |
| 16 | 0 | 1.1 | 0.77 | 17.6 |
| 436 | 0 | 1.6 | 0.62 | 274 |
| 1024 | 0 | 2.8 | 0.61 | 633 |
| 436 | 5 µs | 1.4 | 5.65 | 2468 |
| eager 对照 436 launch | 0 | 628（1.44/launch） | — | 636 |

### 多卡提交策略（436 node × 5 µs，每 6 个 kernel 一个跨卡 P2P barrier = 72 个/step）

| 策略 | 3 rank step (µs) | 8 rank step (µs) | host 循环 (3 / 8) |
|---|---|---|---|
| 1 线程顺序 launch + 每步 sync | 2762 | 3013 | 6.7 / 16.8 µs |
| 1 线程顺序 launch，不 sync | 2748 | 2979 | 4.7 / 13.0 µs |
| 每卡一线程 + sync | 2754 | 2987 | — |
| 每卡一线程，不 sync | 2749 | 2980 | — |
| 一张跨设备图（capture 时 event fork/join），1 次 launch | 2768（1524 node） | 3010（4064 node） | 14.8 / 26.6 µs |

（8 rank 是 8 张图铺在 3 张卡上，GPU 时间偏高，host 侧数字是真的。）
无 barrier 对照 275 µs vs 有 72 个 barrier 551 µs（空核）：**一个跨卡
barrier ≈ 3.8 µs**，这是 TP 的固有成本，与提交方式无关。

### 跨 tray（MNNVL fabric，tray03 ↔ tray06）

集群 IMEX daemon 在跑，18 个 tray 同一 ClusterUUID / CliqueId 3366，即
**整个 pod 72 卡是一个 NVLink 域**。`cuMemCreate` +
`CU_MEM_HANDLE_TYPE_FABRIC` 导出 64 字节 handle，对面
`cuMemImportFromShareableHandle`，普通用户直接可用。

| | tray 内 (NV18) | 跨 tray (fabric) |
|---|---|---|
| P2P 原子 barrier | 3.8 µs | 4.8 µs |
| 单卡单向裸拷贝核 写 / 读 | — | 710 / 750 GB/s（NVLink5 单向峰值 900） |

### GPU 自提交

`cudaGraphInstantiateFlagDeviceLaunch` + 图尾 kernel
`cudaGraphLaunch(self, cudaStreamGraphTailLaunch)`：436 node 空核
270 µs/step vs host 流水 relaunch 277；5 µs 核 2465 vs 2471。限制：单设
备、只能 kernel/memcpy/memset/child 节点、无 host/event 节点、跨设备图不
能 device launch、尾核要 `-rdc` 链 cudadevrt。

## 设计

### 不变量

- `Runtime` = 一个 CUDA context、一个 stream、一份 manifest、一套
  buffer/state、一组图。**没有 rank 的概念以外的任何多卡知识。**
- 一个 caller 线程驱动它的全部 runtime（一 tray 4 个），顺序 launch。
- 跨 rank 的一切同步都发生在 kernel 里（release/acquire flag、原子计数
  器），host 不参与 step 内的任何同步。
- manifest 是 SPMD 的：**一份 manifest，全部 rank 共用**，rank 是 load 时
  给的常量；每 rank 的权重文件是自己的分片，名字相同。

### Manifest 扩展（进 v3，不升版）

```jsonc
"schema_version": 3,                                // 未发布，不升版
"topology": { "groups": { "tp": 8, "ep": 32 } },   // 只声明名字和大小；成员由 loader 给

"buffers": {
  // 可导出：VMM 分配（cuMemCreate + fabric handle），别的 rank 可映射
  "ar_in":  { "dtype": "bf16", "shape": ["tokens", 5120], "kind": "workspace", "export": true },
  "ar_flag": { "dtype": "i32", "shape": [64], "kind": "carry", "export": true },
  // peer：u64[组大小]，第 i 项是组内 rank i 的 `of` 的设备基址（本 rank 也在里面）
  "ar_in_peers":  { "dtype": "u64", "shape": [8], "kind": "peer", "of": "ar_in",  "group": "tp" },
  "ar_flag_peers": { "dtype": "u64", "shape": [8], "kind": "peer", "of": "ar_flag", "group": "tp" },
  "kv_peers":      { "dtype": "u64", "shape": [32], "kind": "peer", "of": "kv", "group": "ep" }
},
"states": {
  "kv": { "bytes_per_token": 147456 }        // state 一律 VMM + fabric handle，没有 export 字段；P/D push 的目标
},
"ops": {
  "allreduce": { "params": ["inout buffer<bf16>", "in buffer<u64>", "in buffer<u64>", "i32", "i32"],
                 "impl": { "launches": [{ "module": "ar", "entry": "oneshot_allreduce_bf16",
                            "block": [512,1,1], "grid": [{"ceil_div": ["tokens", 8]}, 1, 1],
                            "args": [{"param": 0}, {"param": 1}, {"param": 2}, {"rank": "tp"}, {"i32": 8}] }] } }
}
```

- `export`：buffer/state 的分配走 VMM（`cuMemCreate` +
  `cuMemAddressReserve` + `cuMemMap`），handle 类型 FABRIC；不导出的照旧
  `cuMemAlloc`。**P/D push、EGM、host 分层三件事都要求 state 走 VMM**，所
  以 state 一律 VMM，buffer 按需。
- `peer`：`kind: peer` 的 buffer 是 runtime 填的 `u64[group_size]`（写明
  `dtype: u64`、`shape: [组大小]`），verifier 要求 `of` 是已 `export` 的
  buffer 或任一 state、`group` 已声明、对 op 只读；ABI 上就是
  `in buffer<u64>`。kernel 拿到指针数组自己寻址（vLLM custom_allreduce /
  TRT-LLM / DeepEP intranode 都是这个形状）。
- `{"rank": "<group>"}` 是 launch/call 的标量实参来源，load 时烧死。
- **不加**：collective 类型、通信语义、组的成员关系（loader 给）。
  runtime 不知道 all-reduce 是什么。

### Runtime API 增量

```rust
Runtime::load(manifest, kernels, gpu, capacity, topo: Option<&Topology>)  // Topology { groups: {name: GroupRank { index, size }} }
fn export_handles(&self) -> Result<BTreeMap<String, PeerHandle>>  // 每个 export buffer / 每个带 handle 的 state 一个；PeerHandle = 64 B fabric handle + 映射字节数
fn import_peers(&mut self, group: &str, members: &[BTreeMap<String, PeerHandle>])  // 按 rank 序给全组的表；映射并填该组的 peer 数组
fn pending_peers(&self) -> Vec<&str>   // 还没填的 peer buffer；非空时 run/capture 拒绝
```

（已实现，分支 `ep0-k0-export-state`。）handle 交换的传输是 caller 的事
（共享盘文件、TCP，什么都行——`PeerHandle::to_bytes()` 72 B × buffer 数
× rank 数，一次性）。同进程内的 rank 也走同一条路（fabric
handle 同进程导入合法），不特判 P2P。

### 执行模型

**follower 没有 host 路径。** 早先设想的"follower host 轮询设备 flag 再 launch"
不成立：GB300 的 stream memops 拒绝 fabric 导入的 VA，host 可见的 step flag 只能
经 EGM 或额外 D2H，白白多一跳。所以 follower 一步到位就是自转形态：每 rank 的图尾
tail-launch 自己，图头一个 kernel 等 leader 写来的 step flag（SM store 经 fabric
到 follower 本地显存，follower 在本地 wait）；token 反馈、positions/slot_mapping/
seq_lens 推进都在图内（"advance" 小核）；host 只在 batch 成员变更时介入（写输入、
改 flag）。

**leader 分两阶段。** 阶段 1：leader tray 的调度线程决定 batch，把输入写进本 tray
4 个 runtime，再往每个 follower 的 `export` 过的 input buffer 里写（NVLink，µs 级）
并推 step flag；本 tray 的图由 host 顺序 launch。阶段 2：leader 自己的图也自转，
host 滞后一步异步取结果。TP 下各 rank 靠 collective 互锁，EP 下靠 dispatch/combine
互锁，**成员不变的步全在 GPU 上自转**。这是"scheduler GPU 化"的落点——不是把
admission/lease 搬上 GPU。

**自转是 runtime 行为，不进 manifest。** 图头等 flag、图尾 tail-launch、advance
都是普通 op；"这个 program 会自己循环"由 caller 在 `Runtime` API 上选（instantiate
时带 device-launch flag、给出 head flag 与 advance op 的名字），manifest 只描述一步。
verifier 因此只需检查 device-launch 的图约束（全 kernel/memcpy/memset 节点、无 host
节点、单设备）。

### TP

- 权重按 rank 切在导出时做（`tools/export_weights.py` 出 8 份），manifest
  的 GEMM 形状是切后的；每层两个 `allreduce` call。
- KV 按 head 切，每 rank 自己的 state；lease 决策只在 leader，block_table
  写给每个 rank。
- 图：每 rank 自己的图（不用跨设备图——它不能 device launch，且 fork/join
  只在同进程有意义）。
- decode 用 one-shot allreduce（延迟主导），prefill 用 reduce-scatter +
  all-gather（带宽主导）——manifest 里两个 op，prefill/decode program 各
  用各的。

### EP（含 wide EP）

- attention DP + experts EP：每 rank 自己的 batch、自己的 lease、自己的
  scheduler；只在 dispatch/combine 对齐。**每步所有 rank 必须到场**：没
  请求的 rank 跑 `tokens=1` 的 pad 步（var 下界是 1）。
- dispatch 缓冲按最坏情况分配（`max_tokens_per_rank × experts_per_rank`，
  DeepEP low-latency 模式的做法），正好是 kern"按 var max 分配、grid 按上
  界"的既有模型；实际 count 在 kernel 内走。
- 72 卡以内 dispatch/combine 就是 peer 指针 + flag 的 kernel，不需要
  NVSHMEM/RDMA；跨 NVL72 域才换 transport（symmetric 分配域 + IBGDA
  kernel），runtime 不变。
- GPU 自转对 EP 的收益比 TP 大：wide EP 的 pace 由最慢的 host 决定，
  host 不在环上就没有这个抖动。

### TP × EP 组合

manifest 声明多个组（`tp: 8, ep: 32`），peer buffer 各自指定 `group`，
loader 给每个组里本 rank 的 index 和成员。attention TP + experts EP 就是
allreduce 用 tp 组、dispatch 用 ep 组，同一份 manifest。组的成员映射
（哪 8 个全局 rank 是一个 TP 组）是 caller 的部署决策，manifest 不含。

### Prefill

已按 chunk 捕图，一次 launch 1–2 µs 对应几十 ms GPU；余数块 eager 630 µs
host 也远小于 GPU 时间——只要提交前不 sync 就被藏掉。TP 下同样。

### 故障模型

rank 死 → 组内所有图在 flag 上永久 spin。**所有跨 rank 等待都带超时**：一次等待
本来就是 3.8–4.8 µs 量级，多读一次 `globaltimer` 是纳秒级。超时的 kernel 把错误码
写进一个 `carry` buffer 后正常返回，图跑完，host 读到错误码就拆组重建；自转的图在
超时后不再 tail-launch。外部看门狗杀进程只作兜底。

### 硬约束（`dcp-bench.md` B3p / B2）

- **peer/fabric 映射的显存禁止 multicast TMA**（09-02 tray18 隔离实验，
  `moe-comm-survey.md` §5）：cluster `.multicast::cluster` 的 tensor TMA 第一次发出就把
  **发起方**的卡打进 "GPU requires reset"（被访问方无事）；cuBLAS/cuBLASLt/CUTLASS 的
  sm100 GEMM 都用它，昨天 tray13 的事故即此。其余全部可用且跨 tray bit-exact、
  690–750 GB/s：普通 ld/st、1-D `cp.async.bulk` 拉/推、`cp.reduce.async.bulk`、sys 原子、
  **非 multicast 的 tensor-map TMA**（1024 CTA 满并发也没事）。verifier 规则：拿到
  `peer` 派生指针的 module，SASS 里 UTMALDG/UTMASTG 带 `.MULTICAST` 即拒绝装载；
  extern op（cublasLt）永远不能接 peer buffer。attest 记录这条检查。
- **跨卡数据永远是 ship q / 回 partial，不读远端 KV**：远端流式读比本地慢 8×。
  远端读只用于迁移/staging。
- **DCP 宽度 w 是 per-span 的调度决策，不是 manifest 常量**：decode 上 DCP 是纯税
  （W=4 每步 5–8%），只对不平衡或超容量的 session 开；extend 上是真并行（W=4 得
  1.7–2.5×）。与"段长是 per-rank var、stripe 掩码是派生标量"一致：同一张图，w 由
  每步的 var 决定。

### Attest / 确定性

- rank-local：每 rank 对自己的 cut 做，peer buffer 的内容作为输入录下来
  （allreduce 的输入是本 rank 的 `ar_in` + 各 peer 的 `ar_in`），cut 边界
  在 collective 两侧。
- collective kernel 的归约顺序固定 → rank 间可复现；bucket 变化引起的
  near-tie 分叉和单卡一样存在。

## 待测

- [ ] （TP 已降级，低优先）真实 allreduce kernel 在 dense 形状跨 tray 的延迟——
  现在只有 barrier（4.8 µs）和裸拷贝（710 GB/s）两个端点。
- [x] **EP dispatch 模拟（`a2a.cu`）**：8 rank 跨 tray（tray14+17）一步 8.6 µs（0 B）
  … 22 µs（256 KB/peer）… 204 µs（4 MB/peer，egress 144 GB/s）。外推 EP36 每步
  dispatch+combine ≈ 5–7 ms，见 `dcp-bench.md` B4；W=36 真跑待做。
- [ ] **verifier 的 SASS 扫描**（peer 参数的 module 禁 multicast TMA）：`.MULTICAST` 的
  拼写要在 sm_103 上实际 cuobjdump 核对，见"硬约束"。
- [x] 跨 tray fabric VA 上的 1-D `cp.async.bulk`：通过（09-02，`moe-comm-survey.md` §5）。
- [ ] **多 writer 打一张卡的 ingress 聚合**（P/D push 8→1 的上限；估
  ~900 GB/s）。
- [ ] W=36 真实 EP fan-out（9 tray `a2a`），替换 B4 的外推。
- [ ] host 抖动：leader 线程在 serving 负载下 launch 8 个 runtime 的 p99。
- [ ] device tail-launch 下图内 `cublasLt` 节点是否全是 kernel 节点
  （capture 出的 memset/memcpy 节点允许，host 节点不允许）。

## 已决（2026-09-02）

1. **进程拓扑：一进程一 tray**（4 runtime，1 线程）。故障域本来就是 tray。
2. **控制通道：纯 GPU 内存。** follower 没有 host 路径（见"执行模型"）；TCP 只在
   启动 / 成员变更 / 故障时用。
3. **故障模型：spin 带超时**，错误码经 carry buffer 回 host（见"故障模型"）。
4. **rank：`{"rank": "<group>"}` 实参来源，load 时烧死，一份 manifest。** 不做每
   rank 一份 manifest。
5. **schema 不升版。** 未发布，`export` / `peer` / `topology` / `{"rank"}` 直接进
   `schema_version: 3`。
6. **自转不进 manifest。** 是 `Runtime` 的行为，caller 选；manifest 只描述一步。
7. **不为跨 NVL72 域留接口。**

8. **MoE 通信 = DeepGEMM MegaMoE**（pegainfer 的 AOT fork，一层一个 launch），源码级
   对比见 `moe-comm-survey.md`。TRT one-sided 作为 fallback 的事以后再说；不自写，NCCL
   不作为 extern 保留。
9. **K3 decode 核来源 = pegainfer 认证核集，不从 vLLM 挖。** vLLM 的 K3 路径
   （fused_kda_decode / flashinfer-trtllm MLA / DeepGEMM / torch.compile）拆不出可
   AOT 的核；kern 的 93 层 EP4 program（E2）逐行对照 pegainfer `k3_step`，正确性以
   pegainfer fixture + 三方 teacher-forced oracle 定案（见 roadmap E2 行）。导出的权重
   放 tray 本地数据盘（`/data/<user>/kern-k3/`），不进 Ceph。

## 待决

（无。）
