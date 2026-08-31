# 后续（未做）

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
