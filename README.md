# kern

**Models ship as verified GPU programs.**

[Website](https://kern-baa.pages.dev/) · [Design notes](docs/design.md) ·
[Manifest example](examples/qwen3-4b.json)

kern is a model-agnostic GPU runtime. A model provider ships
`manifest.json + kernels.cubin + weights`; the runtime verifies the manifest
as strictly as a compiler at load time, then executes it blindly. The runtime
assigns no meaning to any name in the manifest — it schedules opaque kernel
dispatches, provisions opaque per-token state bytes, and evaluates a closed
set of scalar expressions for launch geometry. All model semantics live on
the provider's side of that boundary.

Proof of concept: Qwen3-4B bs=1 greedy decode on GB300, with every kernel
mined from vLLM's production path (plus two trivial hand-written ones).
Decode runs at ~92% of vLLM's own throughput; chunked prefill at ~12k tok/s;
DSpark speculative decoding reaches 2.4× decode throughput while matching
plain decode byte-for-byte.

## Repository layout

| Path | What it is |
| --- | --- |
| `crates/kern-manifest` | Manifest types + static verifier (pure, no CUDA) |
| `crates/kern-runtime` | The executor: resolve cubins, allocate, replay dispatches, CUDA graphs |
| `crates/kern-run` | CLI caller contract for the qwen3-4b manifests |
| `examples/` | Generated manifests (`qwen3-4b.json`, `qwen3-4b-dspark.json`) |
| `kernels/` | Extracted cubin artifacts |
| `tools/` | Kernel-capture injector, capture analysis, manifest generator, weight export ([tools/README.md](tools/README.md)) |
| `docs/` | Design and development docs (below) |

## Quickstart

```bash
cargo build --release

./target/release/kern-run \
  --manifest examples/qwen3-4b.json --kernels kernels \
  --weights weights/qwen3-4b-decode.safetensors --tokenizer weights/tokenizer.json \
  --gpu 0 --prompt "The capital of France is" --steps 320

# DSpark speculative decoding
./target/release/kern-run --manifest examples/qwen3-4b-dspark.json \
  --weights weights/qwen3-4b-dspark.safetensors --spec --steps 320
```

`kern-run --help` lists all flags. Logs go to stderr via `tracing`
(filter with `RUST_LOG`, default `info`); stdout carries only the generated
text. The full pipeline that produces `kernels/` and `weights/` from a live
vLLM process is in [docs/runtime.md](docs/runtime.md).

## Documentation

- [docs/design.md](docs/design.md) — background and design exploration:
  why a model-agnostic runtime, the type/state boundary, open questions.
- [docs/manifest.md](docs/manifest.md) — manifest format v2 (interface/impl
  split, buffer classes, the closed expression language) and everything the
  verifier checks.
- [docs/kernel-mining.md](docs/kernel-mining.md) — mining kernels out of
  vLLM with CUPTI capture: what replays, what doesn't (nvjet/fmha packed
  ABIs), the attention-backend ABI survey, and the capture-analysis /
  manifest-generator tooling.
- [docs/runtime.md](docs/runtime.md) — the executor, the `kern-run` caller
  contract, CUDA graph capture, measured performance vs vLLM, and the
  end-to-end dump → manifest → run pipeline.
- [docs/spec-decode.md](docs/spec-decode.md) — DSpark speculative decoding
  as pure manifest wiring: draft KV via target hidden-state projection, the
  Markov head unrolled onto existing kernels, lossless-oracle results.
- [docs/roadmap.md](docs/roadmap.md) — known gaps and next steps.

## Development

```bash
cargo test --workspace     # verifier unit tests + mined-manifest round-trips
cargo clippy --workspace --all-targets
```

After a schema change, regenerate the example manifests with
`tools/gen_qwen3_decode.py` (needs a capture dump; see
[docs/kernel-mining.md](docs/kernel-mining.md)).
