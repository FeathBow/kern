# Roadmap（2026-09-02 定稿；背景与实测见 multi-gpu.md / agent-workload.md / dcp-bench.md / moe-comm-survey.md）

**第一个多卡目标：K3 pruned（224 expert）@ EP4 单 tray 的 decode superstep**，与满血
K3 @ EP16 同构（56 expert/rank）。MoE 通信用 DeepGEMM MegaMoE（一层一个 launch）；
attention（MLA / KDA）、dense、norm 等 kernel 从 vLLM 的 K3 运行里挖（CUPTI capture →
manifest，与 qwen3 同一条流水）。两条线并行，每级带门禁。

## EP 线

| 级 | 内容 | 门禁 |
|---|---|---|
| E0 ✅ | runtime 原语：`export`（VMM + fabric handle）、`peer`（u64 数组）、`topology`、`{"rank"}`；`export_handles` / `import_peers` API；verifier 三条规则（peer 必须 of export、`.MULTICAST` SASS 扫描、extern 不接 peer） | 一进程 4 个 runtime，跨卡 barrier 作为 manifest op 跑通，≈3.8 µs —— **2026-09-02 tray03 实测 3.75 µs**（`ep0-k0-export-state`） |
| E1 | K3 pruned 的一个 MoE 层作为 program：quant → MegaMoE cubin → 输出，EP4 单 tray；`SymBuffer` 由 host 从 peer 数组算出 | EP4 rank0 与 EP1 逐位一致 |
| E2 | K3 pruned 完整 decode superstep @EP4：MLA + KDA + MoE 全层，图捕获，leader host 驱动 | 与 vLLM 同 checkpoint 逐 token 一致；步时对齐 |
| E3 | 跨 tray EP8/16：只换 handle 交换；spin 超时进 MegaMoE fork；W=36 barrier 实测 | 跨 tray 与单 tray 逐位一致；每步 dispatch+combine 税在 5–7 ms 预估内 |
| E4 | GPU 自转：图尾 tail-launch、图头等 step flag、advance 小核；follower 无 host | 成员不变的步 host 零参与 |

## KV 线

| 级 | 内容 | 门禁 |
|---|---|---|
| K0 ✅ | state 一律走 VMM（可导出）；per-seq 定长 state（KDA 状态）已有 `bytes_per_seq` | 现有 qwen3 / dspark 门禁不变（2026-09-02 过）；K3 的 KDA state 能装载 |
| K1 | 前缀缓存 + 分支 CoW：lease 池加页引用计数，块哈希查找 | AgentX trace 回放命中率 ≈ 98% |
| K2 | checkpoint 一等对象：KDA 状态快照 + latent 页列表 + 位置 + epoch；fork = 引用计数 + 快照 | 子 agent 从父 checkpoint 分叉，输出与重算一致 |
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
