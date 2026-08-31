//! End-to-end bs=1 greedy decode over a kern manifest.
//!
//! The runtime library is model-agnostic; this binary is the caller-side
//! contract for the qwen3-4b-decode manifest: which input buffers exist and
//! what to put in them each step (token_ids/positions/slot_mapping/seq_lens/
//! cu_seqlens_q/block_table), prefill expressed as repeated tokens=1 decode.
//!
//! Logging goes to stderr via `tracing` (filter with `RUST_LOG`, default
//! `info`); stdout carries only the generated text.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use kern_run::{
    env, i64_from_le, le_bytes_i32, le_bytes_i64, prefill_emits_next_token, Caller, STOP_TOKENS,
};
use kern_runtime::Runtime;
use tracing::info;

// DSpark draft fill token for the 6 non-anchor query slots (draft config).
const MASK_TOKEN: i64 = 151669;
// DSpark block size: 7 draft queries per round, verified as anchor + 7.
const DRAFT_TOKENS: usize = 7;

/// Greedy bs=1 decode over a kern manifest (qwen3-4b caller contract).
#[derive(Parser)]
#[command(version, about)]
struct Opts {
    /// Manifest JSON (must pass verification)
    #[arg(long, default_value = "examples/qwen3-4b.json")]
    manifest: PathBuf,

    /// Directory holding the .cubin modules
    #[arg(long, default_value = "kernels")]
    kernels: PathBuf,

    /// Safetensors weight artifact, tensors bound by name
    #[arg(long, default_value = "weights/qwen3-4b-decode.safetensors")]
    weights: PathBuf,

    /// HF tokenizer.json
    #[arg(long, default_value = "weights/tokenizer.json")]
    tokenizer: PathBuf,

    /// Raw (template-free) prompt
    #[arg(long, default_value = "The capital of France is")]
    prompt: String,

    /// Max new tokens to generate
    #[arg(long, default_value_t = 32)]
    steps: usize,

    /// CUDA device ordinal
    #[arg(long, default_value_t = 0)]
    gpu: usize,

    /// State capacity in tokens (KV pages etc.)
    #[arg(long, default_value_t = 4096)]
    capacity: u64,

    /// Chunked-prefill chunk size (clamped to the manifest's tokens bound)
    #[arg(long, default_value_t = 512)]
    chunk: u64,

    /// Skip CUDA graph capture, launch every dispatch eagerly
    #[arg(long)]
    eager: bool,

    /// DSpark speculative decoding (needs the dspark manifest programs)
    #[arg(long)]
    spec: bool,

    /// Token ids that end generation (comma-separated; default Qwen3's)
    #[arg(long, value_delimiter = ',', default_values_t = STOP_TOKENS)]
    stop_tokens: Vec<i64>,

    /// Debug: dump per-layer activations (`y` after every `*.down_proj`,
    /// embedding, logits) of the first prefill chunk and two decode steps
    /// into this directory, then exit. Programs run dispatch-range by
    /// dispatch-range so nothing executes twice.
    #[arg(long)]
    probe_dir: Option<PathBuf>,
}

fn human(bytes: u64) -> String {
    match bytes {
        b if b >= 1 << 30 => format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64),
        b if b >= 1 << 20 => format!("{:.1} MiB", b as f64 / (1u64 << 20) as f64),
        b if b >= 1 << 10 => format!("{:.1} KiB", b as f64 / 1024.0),
        b => format!("{b} B"),
    }
}

fn ellipsize(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n - 1).collect::<String>())
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .with_target(false)
        .without_time()
        .init();

    let o = Opts::parse();

    let manifest_json = std::fs::read_to_string(&o.manifest)
        .with_context(|| format!("reading manifest {}", o.manifest.display()))?;
    let t0 = Instant::now();
    let mut rt = Runtime::load(&manifest_json, &o.kernels, o.gpu, o.capacity)?;
    let load_t = t0.elapsed();

    let m = &rt.manifest;
    info!(
        "manifest `{}` (format v{}, {}): verified",
        m.meta.model,
        m.meta.version,
        o.manifest.display()
    );
    for (name, s) in &m.symbols {
        info!("  symbol   {name} ∈ [{}, {}] (runtime-provided per step)", s.min, s.max);
    }
    for (name, per_tok, alloc) in rt.state_sizes() {
        if per_tok > 0 {
            info!(
                "  state    {name}: opaque, {per_tok} B/token × capacity {} = {}",
                o.capacity,
                human(alloc)
            );
        } else {
            info!("  state    {name}: opaque, fixed {} (per-sequence)", human(alloc));
        }
    }
    let mut by_class: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
    for (_, class, bytes) in rt.buffer_sizes() {
        let e = by_class.entry(match class {
            kern_manifest::types::BufferClass::Input => "input",
            kern_manifest::types::BufferClass::Output => "output",
            kern_manifest::types::BufferClass::Weight => "weight",
            kern_manifest::types::BufferClass::Workspace => "workspace",
            kern_manifest::types::BufferClass::Carry => "carry",
        }).or_default();
        e.0 += 1;
        e.1 += bytes;
    }
    let classes = ["weight", "workspace", "carry", "input", "output"]
        .iter()
        .filter_map(|c| by_class.get(c).map(|(n, b)| format!("{c} {n} ({})", human(*b))))
        .collect::<Vec<_>>()
        .join(" | ");
    info!("  buffers  {classes}");
    for (name, p) in &m.programs {
        info!("  program  `{name}`: {} dispatches", p.dispatches.len());
    }

    info!(
        "kernel resolution: {} cubin modules in {}, matched by cuFuncGetParamInfo \
         layout vs declared params ({:?}):",
        rt.module_count(),
        o.kernels.display(),
        load_t
    );
    for (name, modules) in rt.kernel_resolution() {
        let k = &rt.manifest.kernels[&name];
        for (si, (st, module)) in k.imp.steps.iter().zip(&modules).enumerate() {
            let label = if si == 0 { name.clone() } else { format!("  ·step{si}") };
            let sm = match &st.shared_mem {
                Some(e) => format!(
                    ", shmem {:?}",
                    e.eval(&BTreeMap::from([("tokens".into(), 1)])).unwrap_or(0)
                ),
                None => String::new(),
            };
            info!(
                "  {label:<18} {:<44} {:>2} params, block {:?}{sm} <- {module}",
                ellipsize(&st.symbol, 44),
                st.params.len(),
                st.block,
            );
        }
    }

    let t0 = Instant::now();
    let blob = std::fs::read(&o.weights)
        .with_context(|| format!("reading weights {}", o.weights.display()))?;
    let blob_len = blob.len();
    rt.load_weights(&blob)?;
    drop(blob);
    let n_weights = by_class.get("weight").map_or(0, |e| e.0);
    info!(
        "weights: {n_weights} tensors bound by name from {} ({}) in {:?}",
        o.weights.display(),
        human(blob_len as u64),
        t0.elapsed()
    );

    let tokenizer = tokenizers::Tokenizer::from_file(&o.tokenizer)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let prompt_ids: Vec<i64> = tokenizer
        .encode(o.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .iter()
        .map(|&u| u as i64)
        .collect();
    ensure!(!prompt_ids.is_empty(), "empty prompt");
    info!("prompt: {} tokens {prompt_ids:?}", prompt_ids.len());

    let mut caller = Caller::new(rt)?;

    if let Some(dir) = &o.probe_dir {
        return probe(&mut caller, &prompt_ids, dir, o.chunk);
    }

    // Chunked prefill: repeated `prefill` calls. Two caller contracts:
    // - prefill writes state only (qwen3-4b): the first n-1 prompt tokens go
    //   through it and the final prompt token through `decode`, which
    //   produces the first logits — decode doubles as "prefill of the last
    //   token";
    // - prefill emits `next_token` (qwen3.8): every prompt token goes through
    //   it and the last chunk yields the first generated token. Hybrid GDN
    //   models need this — their chunked prefill kernels are a different
    //   arithmetic from the decode kernel, and the reference runs the last
    //   prompt token through the former.
    let chunk = o.chunk.min(caller.rt.manifest.symbols["tokens"].max).max(1);
    let n_prompt = prompt_ids.len();
    let prefill_all = prefill_emits_next_token(&caller.rt.manifest);
    let n_pre = if prefill_all { n_prompt } else { n_prompt - 1 };
    let mut generated: Vec<i64> = Vec::new();
    if n_pre > 0 {
        let t = Instant::now();
        let captured = if o.spec {
            // Each chunk's fc taps must be projected into the draft's context
            // KV while positions/slot_mapping still hold this chunk's rows.
            let mut captured = false;
            let mut i = 0;
            while i < n_pre {
                let c = (n_pre - i).min(chunk as usize);
                let e = caller.stage_prefill(&prompt_ids[i..i + c])?;
                if !o.eager && c == chunk as usize {
                    if !captured {
                        caller.rt.capture("prefill", &e)?;
                        captured = true;
                    }
                    caller.rt.run_captured("prefill", &e)?;
                } else {
                    caller.rt.run("prefill", &e)?;
                }
                caller.rt.run("draft_precompute", &e)?;
                caller.advance(c as u64);
                i += c;
            }
            captured
        } else {
            caller.prefill(&prompt_ids[..n_pre], chunk, o.eager)?
        };
        let dt = t.elapsed();
        let pos = caller.pos;
        let n_chunks = (pos as u64).div_ceil(chunk);
        info!(
            "prefill: {pos} tokens in {n_chunks} chunk(s) of <= {chunk} \
             ({dt:?}, {:.0} tok/s{}{})",
            pos as f64 / dt.as_secs_f64(),
            if captured { ", graph-captured" } else { ", eager" },
            if prefill_all { ", emits next_token" } else { "" }
        );
        if prefill_all {
            let first = caller.next_token()?;
            if o.stop_tokens.contains(&first) {
                info!("stop token {first} at pos {pos}");
                println!("{}", o.prompt);
                return Ok(());
            }
            generated.push(first);
        }
    }
    if o.spec {
        let pos = caller.pos;
        let generated = spec_decode(&mut caller.rt, &o, &prompt_ids, pos)?;
        let gen_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
        let text = tokenizer.decode(&gen_u32, false).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        println!("{}{}", o.prompt, text);
        return Ok(());
    }
    let env = env(1);
    if !o.eager {
        let t = Instant::now();
        caller.rt.capture("decode", &env)?;
        info!(
            "CUDA graph: `decode` stream-captured at tokens=1, {} dispatches -> \
             1 graph launch/step ({:?})",
            caller.rt.manifest.programs["decode"].dispatches.len(),
            t.elapsed()
        );
    }
    let mut decode_ns: u128 = 0;
    let mut decode_steps = 0u32;

    while generated.len() < o.steps {
        let pos = caller.pos as usize;
        let tok = if pos < prompt_ids.len() { prompt_ids[pos] } else { *generated.last().unwrap() };
        caller.stage_decode(tok)?;

        let t = Instant::now();
        if o.eager {
            caller.rt.run("decode", &env)?;
        } else {
            caller.rt.run_captured("decode", &env)?;
        }
        caller.advance(1);
        let pos = caller.pos;

        if (pos as usize) < prompt_ids.len() {
            continue; // prefill-as-decode: logits unused until the last prompt token
        }
        let next = caller.next_token()?;
        decode_ns += t.elapsed().as_nanos();
        decode_steps += 1;
        if o.stop_tokens.contains(&next) {
            info!("stop token {next} at pos {pos}");
            break;
        }
        generated.push(next);
        if pos as u64 + 1 >= o.capacity {
            break;
        }
    }

    let gen_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&gen_u32, false).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    info!("generated ids: {generated:?}");
    info!(
        "{} tokens generated, {:.1} ms/step ({:.1} tok/s)",
        generated.len(),
        decode_ns as f64 / 1e6 / decode_steps.max(1) as f64,
        decode_steps as f64 * 1e9 / decode_ns.max(1) as f64,
    );
    println!("{}{}", o.prompt, text);
    Ok(())
}

/// Activation probe for reference comparison (`--probe-dir`): the first
/// prefill chunk and two decode steps, each run as consecutive dispatch
/// ranges cut after `embed` and every `l<i>.down_proj`, dumping `residual`
/// / `y` (live `tokens` rows) and the final logits as raw little-endian
/// files `<tag>.<point>.bin`.
fn probe(caller: &mut Caller, prompt_ids: &[i64], dir: &std::path::Path, chunk: u64) -> Result<()> {
    use kern_manifest::types::Dim;
    std::fs::create_dir_all(dir)?;
    match caller.rt.manifest.buffers["y"].shape.as_slice() {
        [Dim::Sym(_), Dim::Const(_)] => {}
        s => bail!("unexpected `y` shape {s:?}"),
    }
    // KERN_PROBE_LAYER=<i>: additionally dump, after every dispatch of layer
    // `l<i>.`, the buffer its first `out` param writes (live `tokens` rows).
    let fine: Option<String> = std::env::var("KERN_PROBE_LAYER").ok().map(|l| format!("l{l}."));
    let row_bytes = |rt: &Runtime, name: &str| -> usize {
        let b = &rt.manifest.buffers[name];
        b.shape[1..]
            .iter()
            .map(|d| match d {
                Dim::Const(c) => *c as usize,
                _ => 1,
            })
            .product::<usize>()
            * b.dtype.bytes() as usize
    };
    let run_probed = |rt: &Runtime, program: &str, env: &BTreeMap<String, u64>, tokens: usize, tag: &str| -> Result<()> {
        let prog = &rt.manifest.programs[program];
        let labels: Vec<String> = prog.dispatches.iter().map(|d| d.label.clone().unwrap_or_default()).collect();
        let mut lo = 0;
        for (i, l) in labels.iter().enumerate() {
            let mut dumps: Vec<(String, String)> = Vec::new();
            if l == "embed" {
                dumps.push(("embed".into(), "residual".into()));
            } else if let Some(layer) = l.strip_suffix(".down_proj") {
                dumps.push((layer.to_string(), "y".into()));
            }
            if fine.as_deref().is_some_and(|p| l.starts_with(p)) {
                let d = &prog.dispatches[i];
                let k = &rt.manifest.kernels[&d.kernel];
                for (arg, p) in d.args.iter().zip(&k.params) {
                    if let (kern_manifest::types::Arg::Buf { buf, .. }, kern_manifest::types::ParamType::Buf { dir, .. }) = (arg, p) {
                        if matches!(dir, kern_manifest::types::Dir::Out | kern_manifest::types::Dir::InOut) {
                            dumps.push((format!("{l}.{buf}"), buf.clone()));
                            break;
                        }
                    }
                }
            }
            if dumps.is_empty() {
                continue;
            }
            rt.run_range(program, env, lo, i + 1)?;
            lo = i + 1;
            for (point, bufname) in dumps {
                let rows = match rt.manifest.buffers[&bufname].shape[0] {
                    Dim::Const(c) => c as usize,
                    _ => tokens,
                };
                let data = rt.read_buffer_prefix(&bufname, rows * row_bytes(rt, &bufname))?;
                std::fs::write(dir.join(format!("{tag}.{point}.bin")), data)?;
            }
        }
        rt.run_range(program, env, lo, labels.len())?;
        std::fs::write(dir.join(format!("{tag}.logits.bin")), rt.read_buffer("logits")?)?;
        std::fs::write(dir.join(format!("{tag}.next_token.bin")), rt.read_output("next_token")?)?;
        Ok(())
    };
    let chunk = chunk.min(caller.rt.manifest.symbols["tokens"].max).max(1) as usize;
    let prefill_all = prefill_emits_next_token(&caller.rt.manifest);
    let n_pre = if prefill_all { prompt_ids.len() } else { prompt_ids.len() - 1 };
    let c = n_pre.min(chunk);
    let e = caller.stage_prefill(&prompt_ids[..c])?;
    run_probed(&caller.rt, "prefill", &e, c, "prefill")?;
    caller.advance(c as u64);
    if c < n_pre {
        caller.prefill(&prompt_ids[c..n_pre], chunk as u64, true)?;
    }
    let mut tok = if prefill_all { caller.next_token()? } else { prompt_ids[n_pre] };
    for s in 0..2 {
        let e = caller.stage_decode(tok)?;
        run_probed(&caller.rt, "decode", &e, 1, &format!("decode{s}"))?;
        caller.advance(1);
        tok = caller.next_token()?;
    }
    info!("probe: wrote activations to {}", dir.display());
    Ok(())
}

/// DSpark speculative decoding, caller side. Per round: `draft` proposes 7
/// tokens (anchor + 6 mask queries, markov chain unrolled in-manifest),
/// `verify` runs the target once over [anchor, d0..d6] producing 8 greedy
/// predictions, the accept rule is plain prefix match (greedy spec decode is
/// lossless — output must byte-match plain decode), and `draft_precompute`
/// projects the accepted rows' target hidden states (fc taps in `fc_out`)
/// into the draft's context KV. Rollback is free: rejected slots are simply
/// overwritten by the next round (paged KV, position-identity slots).
fn spec_decode(
    rt: &mut Runtime,
    o: &Opts,
    prompt_ids: &[i64],
    mut pos: i64,
) -> Result<Vec<i64>> {
    for p in ["decode_spec", "verify", "draft", "draft_precompute"] {
        if !rt.manifest.programs.contains_key(p) {
            bail!("--spec needs program `{p}` (not in this manifest)");
        }
    }
    let env = |t: u64| BTreeMap::from([("tokens".to_string(), t)]);
    let verify_n = DRAFT_TOKENS + 1;

    // Last prompt token through decode_spec: first logits + its aux tap.
    rt.write_input("token_ids", &le_bytes_i64(&[prompt_ids[pos as usize]]))?;
    rt.write_input("positions", &le_bytes_i64(&[pos]))?;
    rt.write_input("slot_mapping", &le_bytes_i64(&[pos]))?;
    rt.write_input("seq_lens", &le_bytes_i32(&[pos as i32 + 1]))?;
    rt.write_input("cu_seqlens_q", &le_bytes_i32(&[0, 1]))?;
    rt.run("decode_spec", &env(1))?;
    rt.run("draft_precompute", &env(1))?;
    pos += 1;
    let first = i64::from_le_bytes(rt.read_output("next_token")?.try_into().unwrap());
    if o.stop_tokens.contains(&first) {
        info!("stop token {first} at pos {pos}");
        return Ok(Vec::new());
    }
    let mut generated = vec![first];

    if !o.eager {
        let t = Instant::now();
        rt.capture("draft", &env(DRAFT_TOKENS as u64))?;
        rt.capture("verify", &env(verify_n as u64))?;
        info!(
            "CUDA graphs: `draft` (tokens=7) + `verify` (tokens=8) captured ({:?})",
            t.elapsed()
        );
    }

    let t0 = Instant::now();
    let mut rounds = 0u32;
    let mut accepted = 0usize;
    'rounds: while generated.len() < o.steps && (pos + verify_n as i64) as u64 <= o.capacity {
        let anchor = *generated.last().unwrap();
        // Draft: 7 queries [anchor, mask x6] at pos..pos+6, non-causal.
        let mut ids = vec![anchor];
        ids.resize(DRAFT_TOKENS, MASK_TOKEN);
        let positions: Vec<i64> = (pos..pos + DRAFT_TOKENS as i64).collect();
        rt.write_input("token_ids", &le_bytes_i64(&ids))?;
        rt.write_input("positions", &le_bytes_i64(&positions))?;
        rt.write_input("slot_mapping", &le_bytes_i64(&positions))?;
        rt.write_input("seq_lens", &le_bytes_i32(&[pos as i32 + DRAFT_TOKENS as i32]))?;
        rt.write_input("cu_seqlens_q", &le_bytes_i32(&[0, DRAFT_TOKENS as i32]))?;
        rt.write_input("anchor_token", &le_bytes_i64(&[anchor]))?;
        if o.eager {
            rt.run("draft", &env(DRAFT_TOKENS as u64))?;
        } else {
            rt.run_captured("draft", &env(DRAFT_TOKENS as u64))?;
        }
        let drafts = i64_from_le(&rt.read_output("draft_tokens")?);

        // Verify: one causal target pass over [anchor, d0..d6] -> 8 greedy
        // predictions; row i answers "what follows position pos+i".
        let mut vids = vec![anchor];
        vids.extend_from_slice(&drafts);
        let positions: Vec<i64> = (pos..pos + verify_n as i64).collect();
        rt.write_input("token_ids", &le_bytes_i64(&vids))?;
        rt.write_input("positions", &le_bytes_i64(&positions))?;
        rt.write_input("slot_mapping", &le_bytes_i64(&positions))?;
        rt.write_input("seq_lens", &le_bytes_i32(&[pos as i32 + verify_n as i32]))?;
        rt.write_input("cu_seqlens_q", &le_bytes_i32(&[0, verify_n as i32]))?;
        if o.eager {
            rt.run("verify", &env(verify_n as u64))?;
        } else {
            rt.run_captured("verify", &env(verify_n as u64))?;
        }
        let vt = i64_from_le(&rt.read_output("verify_tokens")?);

        // Accept the longest matching prefix; vt[a] is the correction (or the
        // bonus token when everything matched).
        let mut a = 0;
        while a < DRAFT_TOKENS && drafts[a] == vt[a] {
            a += 1;
        }
        // Project the accepted rows' aux states into the draft context KV
        // (rows 0..=a of fc_out; positions/slot_mapping still hold them).
        rt.run("draft_precompute", &env(a as u64 + 1))?;
        pos += a as i64 + 1;
        rounds += 1;
        accepted += a;
        for &tok in &vt[..=a] {
            if o.stop_tokens.contains(&tok) {
                info!("stop token {tok} at pos {pos}");
                break 'rounds;
            }
            generated.push(tok);
            if generated.len() >= o.steps {
                break 'rounds;
            }
        }
    }
    let dt = t0.elapsed();
    let in_rounds = generated.len() - 1; // first token came from decode_spec
    info!(
        "spec: {in_rounds} tokens in {rounds} rounds ({:.2} tokens/round, \
         {:.1}% drafts accepted), {:.2} ms/round ({:.1} tok/s)",
        in_rounds as f64 / rounds.max(1) as f64,
        accepted as f64 * 100.0 / (rounds.max(1) as usize * DRAFT_TOKENS) as f64,
        dt.as_millis() as f64 / rounds.max(1) as f64,
        in_rounds as f64 / dt.as_secs_f64().max(1e-9),
    );
    Ok(generated)
}
