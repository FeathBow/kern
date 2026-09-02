# pegainfer's AOT TileLang kernels for Kimi-K3 decode

Generated output of `pegainfer-k3/kernels/generate.py` (TileLang 0.1.12,
pegainfer `~/agent_code/pegainfer`), vendored so kern builds without
pegainfer's build tree. One file per family; each holds every batch bucket
`b1..b128` as an `extern "C" __global__` kernel plus pegainfer's host
launchers (unused here). `tools/build_k3_kernels.sh` compiles them —
inside the kernel-lab container, which has the `tl_templates` and CUTLASS
headers — and `tools/k3_line_shim.py` derives the line-indexed wrappers for
the two state kernels. Do not edit; regenerate from pegainfer instead.
