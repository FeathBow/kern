# Roadmap（2026-09-02 定稿；背景与实测见 multi-gpu.md / agent-workload.md / dcp-bench.md / moe-comm-survey.md）

**第一个多卡目标：K3 pruned（224 expert）@ EP4 单 tray 的 decode superstep**，与满血
K3 @ EP16 同构（56 expert/rank）。MoE 通信用 DeepGEMM MegaMoE（一层一个 launch）；
attention（MLA / KDA）、dense、norm 等 kernel 从 vLLM 的 K3 运行里挖（CUPTI capture →
manifest，与 qwen3 同一条流水）。两条线并行，每级带门禁。

## EP 线

| 级 | 内容 | 门禁 |
|---|---|---|
| E0 ✅ | runtime 原语：`export`（VMM + fabric handle）、`peer`（u64 数组）、`topology`、`{"rank"}`；`export_handles` / `import_peers` API；verifier 三条规则（peer 必须 of export、`.MULTICAST` SASS 扫描、extern 不接 peer） | 一进程 4 个 runtime，跨卡 barrier 作为 manifest op 跑通，≈3.8 µs —— **2026-09-02 tray03 实测 3.75 µs**（`ep0-k0-export-state`） |
| E1 ✅ | K3 pruned 的一个 MoE 层作为 program：quant → MegaMoE cubin → 输出，EP4 单 tray；`SymBuffer` 的偏移表由 kernel 在设备上从 peer 数组读；manifest 新增 `tensormap` launch 参与 `cluster` | EP4 每个 rank 与 EP1 逐位一致 —— **2026-09-02 tray04 实测通过**：EP4 227 µs/层（64 token/rank），EP1 733 µs/层（256 token），EP1 对 host 参考相对 RMS 1.7e-3（`k3_moe_ep`） |
| E2 ✅ | K3 pruned 完整 decode superstep @EP4：MLA + KDA + MoE 全 93 层一条 program（3792 launch），图捕获，每 rank 一条序列；核取自 pegainfer 认证 K3 核集（TileLang AOT + MegaMoE + 手写 MLA），稠密投影走新 extern `cublas_bf16_tn_f32` + 认证 `k3_land` | 与 pegainfer golden fixture 逐 token 一致（4 层 EP1/EP4 均 39/40 exact + 1 noise-floor，与 pegainfer 自身相同 —— **2026-09-02 tray04 通过**）。93 层 EP4 teacher-forced 三方对照（`tools/k3_oracle_dump.py`）：对 pegainfer EP4 prose 89/96、random 38/40 exact，对 vLLM prose 87/96 + 3 excused；所有不一致步都落在 vLLM top-1/top-2 margin ≤ 0.81 nats 的近平局上（kern 的答案均在 vLLM top-3，步 34/50 是 kern 与 vLLM 一致而 pegainfer 不一致），而 pegainfer 与 vLLM 自身也差 7/96 —— 三家核数值噪声，判为一致（用户 2026-09-02 定案）。步时 **37.2 ms/step**（4 rank × 1 seq，graph replay）vs pegainfer EP4 38.6 ms、vLLM DP4+EP4+MegaMoE 27.1 ms（vLLM TP4 12.5 ms 是另一种切法，不可比）；稠密权重每 rank 全量读的地板约 17 ms，步时对齐推后（用户：先不管性能）。已知限制：每 rank B=1（TileLang b1 桶 + 单 line shim）；lease 时 runtime 已把该序列的 KDA 槽清零 |
| E3 ✅ | 跨 tray EP8：一 tray 一进程，fabric handle 经 TCP rendezvous 交换（`k3_golden --world/--rank-base/--rendezvous`），MegaMoE 核加 R=8/16 世界；spin 超时 / 故障模型与 W=36 实测未做（用户：E3 只做简单验证） | **2026-09-02 tray07+tray08 通过**：4 层 EP8 跨 tray 8 个 rank 全部 39/40 exact + 1 excused，与单 tray EP4 同；93 层 EP8 跨 tray 对 pegainfer oracle 38/40，错的两步与单 tray EP4 完全相同（逐 token 一致）；步时 **36.8 ms/step** vs 单 tray EP4 37.2 ms —— 跨 tray dispatch+combine 税在 bs=1 下测不出（每 rank 少读一半 expert 抵掉了）|
| E4 | GPU 自转：图尾 tail-launch、图头等 step flag、advance 小核；follower 无 host | 成员不变的步 host 零参与 |

## KV 线

| 级 | 内容 | 门禁 |
|---|---|---|
| K0 ✅ | state 一律走 VMM（可导出）；per-seq 定长 state（KDA 状态）已有 `bytes_per_seq` | 现有 qwen3 / dspark 门禁不变（2026-09-02 过）；K3 的 KDA state 能装载 |
| K1 | 前缀缓存 = checkpoint 表：checkpoint = (精确长度, 前缀链哈希, 页表引用, 可选状态快照)，lease 池加页引用计数。纯 KV 模型的 checkpoint 不带状态、零成本，每个 64-token 块边界都留一个（退化成块哈希缓存）；KDA 模型每个 checkpoint 付 ~605 MB 快照，只在请求结束处留——MLA 页部分命中对 K3 没有价值，KDA 状态得从头重算 | AgentX trace 回放命中率 ≈ 98%（纯 host 侧页号模拟，不需要 GPU） |
| K2 | fork = 引用计数 +1 + 状态快照拷贝；子序列从父 checkpoint 起步，最后半页 CoW | 子 agent 从父 checkpoint 分叉，输出与重算一致（需要每 rank B>1，4 卡） |
| K3 | session 睡/醒到本 tray DRAM（C2C，与 HBM 不干扰） | 6 GB 唤醒 < 60 ms，并发 decode 步时抖动 ≤ 6% |
| K4 | DCP：ship q / 回 (O, LSE) 的 partial+merge op，w 按 span 定，flag-in-payload | 真 FlashMLA 复现 B2/B3：decode 税 ≤ 8%，extend W=4 ≥ 2× |
| K5 | extend 作为 decode superstep 的 filler，chunk 按每步预算与通信项派生 | 130k whale TTFT ≈ 2 s 量级，decode TPOT 不破 |

顺序：E0 + K0 一起先做（同一个分配器改动，K3 的 KDA state 也在关键路径上）；然后
E1–E2 与 K1–K2 并行；E3、K3；最后 E4、K4、K5。E4 依赖下面的 step 边界 GPU 化。

## 单卡遗留（未做）

- capture 补 launch→module id 映射（unified 双实例现靠 num_regs+cuobjdump
  间接定位，capture 直接记 module id 更干净）；生成器给自写核（argmax/
  embedding）也填 `sha256`（unified 双实例已钉哈希）。
- workspace 静态规划（liveness + 贪心 offset 复用；现在逐 buffer 独立分配）。
- step 边界 GPU 化（vs vLLM 差的 ~0.25ms/step）：token 反馈闭环进 graph——
  embedding 的 token_ids 直接由 next_token 喂，positions/slot_mapping/seq_lens
  可预知提前写，host 滞后一步异步取结果，步间不再 sync。E4 直接依赖它。
- attest 后续：kernel-as-package 目录里带上 attestation 当证据；bs>1 的
  workload（现在 bs=1 下 elementwise 核全是 launch 主导，roofline 列
  0.1%）；GEMM extern 的 FLOPs roofline（现在只算字节）；结构输入的
  domain 校验扩到 debug 模式下的设备侧 buffer（现在只查 host 写入）。
  多卡时：A、B 共用一个 runtime 装载；设备侧 compare op；rank-local 比较。
