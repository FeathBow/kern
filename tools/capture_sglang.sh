#!/usr/bin/env bash
# Dump SGLang's kernels with the same CUPTI injection lib — capture 不挑框架
# （CUDA_INJECTION64_PATH 随环境进每个 CUDA 进程；SGLang 的 GPU 活在
# scheduler 子进程里，dump 落在它自己的 pid<N> 目录）。
#
# 跑在 docker 镜像里（pip 装 sglang 的路走不通：pypi 的 sgl-kernel 轮子
# 链接 CUDA 12 库，与 aarch64 上必须的 cu130 torch 冲突）。本机现成镜像
# `rfork-sglang:gb300-mnnvl-20260605`（`lmsysorg/sglang:latest` 也有 arm64）。
# 注入库 host 编译即可——容器内 libcupti.so.13 同版本直接解析。
#
# 两个坑：
# - payload 必须是真实文件（capture_sglang.py）：sglang 用 spawn 起
#   scheduler，spawn 重新 import __main__，stdin 脚本会挂。
# - disable_cuda_graph 之外还要 disable_piecewise_cuda_graph：该镜像的
#   piecewise 路径在 FusedAddRMSNorm 处 illegal memory access（与注入无关，
#   裸跑同样挂）。
#
#   CUDA_VISIBLE_DEVICES=2 tools/capture_sglang.sh
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${MODEL:-/mnt/shared/weights/Qwen3-4B}"
out="${KERNEL_CAPTURE_DIR:-$repo/dumped-kernels/sglang}"
image="${SGLANG_IMAGE:-rfork-sglang:gb300-mnnvl-20260605}"

[ -f "$repo/tools/kernel-capture/libkernelcapture.so" ] || "$repo/tools/kernel-capture/build.sh"
mkdir -p "$out"

docker run --rm --gpus all --ipc=host --shm-size=32g \
  -v /mnt/shared:/mnt/shared \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  -e CUDA_INJECTION64_PATH="$repo/tools/kernel-capture/libkernelcapture.so" \
  -e KERNEL_CAPTURE_DIR="$out" \
  -e HF_HUB_OFFLINE=1 \
  "$image" python3 "$repo/tools/capture_sglang.py" "$model"

echo "--- capture summary ---"
for d in "$out"/pid*/; do
  echo "$d: $(ls "$d" | grep -c cubin) cubins, $(wc -l < "$d/launches.jsonl") launches"
done
