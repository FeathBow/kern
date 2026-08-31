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
use kern_runtime::Runtime;
use tracing::info;

// Qwen3 stop tokens for raw (template-free) completion.
const STOP_TOKENS: [i64; 2] = [151643, 151645]; // <|endoftext|>, <|im_end|>
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

fn le_bytes_i64(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i64_from_le(b: &[u8]) -> Vec<i64> {
    b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
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
        info!(
            "  state    {name}: opaque, {per_tok} B/token × capacity {} = {}",
            o.capacity,
            human(alloc)
        );
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

    // One-time inputs: identity page table (pages allocated linearly by
    // position), single-sequence query bounds.
    let n_pages = match rt.manifest.buffers["block_table"].shape.as_slice() {
        [kern_manifest::types::Dim::Const(n)] => *n as i32,
        s => bail!("unexpected block_table shape {s:?}"),
    };
    rt.write_input("block_table", &le_bytes_i32(&(0..n_pages).collect::<Vec<_>>()))?;

    // Chunked prefill over the first n-1 prompt tokens: repeated `prefill`
    // calls (KV writes only, no logits). The final prompt token goes through
    // `decode`, which produces the first logits — decode doubles as
    // "prefill of the last token".
    let chunk = o.chunk.min(rt.manifest.symbols["tokens"].max).max(1);
    let n_prompt = prompt_ids.len();
    let mut pos: i64 = 0;
    if n_prompt > 1 {
        let t = Instant::now();
        let mut captured = false;
        while (pos as usize) < n_prompt - 1 {
            let c = ((n_prompt - 1 - pos as usize) as u64).min(chunk);
            let ids = &prompt_ids[pos as usize..pos as usize + c as usize];
            let positions: Vec<i64> = (pos..pos + c as i64).collect();
            rt.write_input("token_ids", &le_bytes_i64(ids))?;
            rt.write_input("positions", &le_bytes_i64(&positions))?;
            rt.write_input("slot_mapping", &le_bytes_i64(&positions))?;
            rt.write_input("seq_lens", &le_bytes_i32(&[pos as i32 + c as i32]))?;
            rt.write_input("cu_seqlens_q", &le_bytes_i32(&[0, c as i32]))?;
            let env = BTreeMap::from([("tokens".to_string(), c)]);
            if !o.eager && c == chunk {
                if !captured {
                    rt.capture("prefill", &env)?;
                    captured = true;
                }
                rt.run_captured("prefill", &env)?;
            } else {
                rt.run("prefill", &env)?; // remainder chunk: eager
            }
            if o.spec {
                // The chunk's fc taps are in fc_out; project them into the
                // draft's context KV while positions/slot_mapping still hold
                // this chunk's rows.
                rt.run("draft_precompute", &env)?;
            }
            pos += c as i64;
        }
        let dt = t.elapsed();
        let n_chunks = (pos as u64).div_ceil(chunk);
        info!(
            "prefill: {pos} tokens in {n_chunks} chunk(s) of <= {chunk} \
             ({dt:?}, {:.0} tok/s{})",
            pos as f64 / dt.as_secs_f64(),
            if captured { ", graph-captured" } else { ", eager" }
        );
    }
    if o.spec {
        let generated = spec_decode(&mut rt, &o, &prompt_ids, pos)?;
        let gen_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
        let text = tokenizer.decode(&gen_u32, false).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        println!("{}{}", o.prompt, text);
        return Ok(());
    }
    rt.write_input("cu_seqlens_q", &le_bytes_i32(&[0, 1]))?;

    let env = BTreeMap::from([("tokens".to_string(), 1u64)]);
    if !o.eager {
        let t = Instant::now();
        rt.capture("decode", &env)?;
        info!(
            "CUDA graph: `decode` stream-captured at tokens=1, {} dispatches -> \
             1 graph launch/step ({:?})",
            rt.manifest.programs["decode"].dispatches.len(),
            t.elapsed()
        );
    }
    let mut generated: Vec<i64> = Vec::new();
    let mut decode_ns: u128 = 0;
    let mut decode_steps = 0u32;

    loop {
        let tok = if (pos as usize) < prompt_ids.len() {
            prompt_ids[pos as usize]
        } else {
            *generated.last().unwrap()
        };
        rt.write_input("token_ids", &le_bytes_i64(&[tok]))?;
        rt.write_input("positions", &le_bytes_i64(&[pos]))?;
        rt.write_input("slot_mapping", &le_bytes_i64(&[pos]))?;
        rt.write_input("seq_lens", &le_bytes_i32(&[pos as i32 + 1]))?;

        let t = Instant::now();
        if o.eager {
            rt.run("decode", &env)?;
        } else {
            rt.run_captured("decode", &env)?;
        }
        pos += 1;

        if (pos as usize) < prompt_ids.len() {
            continue; // prefill-as-decode: logits unused until the last prompt token
        }
        let next = i64::from_le_bytes(rt.read_output("next_token")?.try_into().unwrap());
        decode_ns += t.elapsed().as_nanos();
        decode_steps += 1;
        if STOP_TOKENS.contains(&next) {
            info!("stop token {next} at pos {pos}");
            break;
        }
        generated.push(next);
        if generated.len() >= o.steps || pos as u64 + 1 >= o.capacity {
            break;
        }
    }

    let gen_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&gen_u32, false).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    info!(
        "{} tokens generated, {:.1} ms/step ({:.1} tok/s)",
        generated.len(),
        decode_ns as f64 / 1e6 / decode_steps.max(1) as f64,
        decode_steps as f64 * 1e9 / decode_ns.max(1) as f64,
    );
    println!("{}{}", o.prompt, text);
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
    if STOP_TOKENS.contains(&first) {
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
            if STOP_TOKENS.contains(&tok) {
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
