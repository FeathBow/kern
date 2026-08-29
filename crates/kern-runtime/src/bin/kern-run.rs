//! End-to-end bs=1 greedy decode over a kern manifest.
//!
//! The runtime library is model-agnostic; this binary is the caller-side
//! contract for the qwen3-4b-decode manifest: which input buffers exist and
//! what to put in them each step (token_ids/positions/slot_mapping/seq_lens/
//! cu_seqlens_q/block_table), prefill expressed as repeated tokens=1 decode.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use kern_runtime::{Result, Runtime};

// Qwen3 stop tokens for raw (template-free) completion.
const STOP_TOKENS: [i64; 2] = [151643, 151645]; // <|endoftext|>, <|im_end|>

struct Opts {
    manifest: PathBuf,
    kernels: PathBuf,
    weights: PathBuf,
    tokenizer: PathBuf,
    prompt: String,
    steps: usize,
    gpu: usize,
    capacity: u64,
    eager: bool,
}

fn parse_opts() -> Opts {
    let mut o = Opts {
        manifest: "examples/qwen3-4b-decode.json".into(),
        kernels: "kernels".into(),
        weights: "weights/qwen3-4b-decode.safetensors".into(),
        tokenizer: "weights/tokenizer.json".into(),
        prompt: "The capital of France is".into(),
        steps: 32,
        gpu: 0,
        capacity: 4096,
        eager: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| panic!("missing value for {a}"));
        match a.as_str() {
            "--manifest" => o.manifest = val().into(),
            "--kernels" => o.kernels = val().into(),
            "--weights" => o.weights = val().into(),
            "--tokenizer" => o.tokenizer = val().into(),
            "--prompt" => o.prompt = val(),
            "--steps" => o.steps = val().parse().expect("--steps"),
            "--gpu" => o.gpu = val().parse().expect("--gpu"),
            "--capacity" => o.capacity = val().parse().expect("--capacity"),
            "--eager" => o.eager = true,
            _ => panic!("unknown flag {a}"),
        }
    }
    o
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


fn main() -> Result<()> {
    let o = parse_opts();

    let manifest_json = std::fs::read_to_string(&o.manifest)
        .map_err(|e| format!("reading manifest {}: {e}", o.manifest.display()))?;
    let t0 = Instant::now();
    let mut rt = Runtime::load(&manifest_json, &o.kernels, o.gpu, o.capacity)?;
    let load_t = t0.elapsed();

    let m = &rt.manifest;
    eprintln!(
        "[kern-run] manifest `{}` (format v{}, {}): verified",
        m.meta.model,
        m.meta.version,
        o.manifest.display()
    );
    for (name, s) in &m.symbols {
        eprintln!("[kern-run]   symbol   {name} ∈ [{}, {}] (runtime-provided per step)", s.min, s.max);
    }
    for (name, per_tok, alloc) in rt.state_sizes() {
        eprintln!(
            "[kern-run]   state    {name}: opaque, {per_tok} B/token × capacity {} = {}",
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
        }).or_default();
        e.0 += 1;
        e.1 += bytes;
    }
    let classes = ["weight", "workspace", "input", "output"]
        .iter()
        .filter_map(|c| by_class.get(c).map(|(n, b)| format!("{c} {n} ({})", human(*b))))
        .collect::<Vec<_>>()
        .join(" | ");
    eprintln!("[kern-run]   buffers  {classes}");
    for (name, p) in &m.programs {
        eprintln!("[kern-run]   program  `{name}`: {} dispatches", p.dispatches.len());
    }

    eprintln!(
        "[kern-run] kernel resolution: {} cubin modules in {}, matched by cuFuncGetParamInfo \
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
            eprintln!(
                "[kern-run]   {label:<18} {:<44} {:>2} params, block {:?}{sm} <- {module}",
                ellipsize(&st.symbol, 44),
                st.params.len(),
                st.block,
            );
        }
    }

    let t0 = Instant::now();
    let blob = std::fs::read(&o.weights)
        .map_err(|e| format!("reading weights {}: {e}", o.weights.display()))?;
    let blob_len = blob.len();
    rt.load_weights(&blob)?;
    drop(blob);
    let n_weights = by_class.get("weight").map_or(0, |e| e.0);
    eprintln!(
        "[kern-run] weights: {n_weights} tensors bound by name from {} ({}) in {:?}",
        o.weights.display(),
        human(blob_len as u64),
        t0.elapsed()
    );

    let tokenizer = tokenizers::Tokenizer::from_file(&o.tokenizer)
        .map_err(|e| format!("tokenizer: {e}"))?;
    let prompt_ids: Vec<i64> = tokenizer
        .encode(o.prompt.as_str(), false)
        .map_err(|e| format!("encode: {e}"))?
        .get_ids()
        .iter()
        .map(|&u| u as i64)
        .collect();
    assert!(!prompt_ids.is_empty(), "empty prompt");
    eprintln!("[kern-run] prompt: {} tokens {prompt_ids:?}", prompt_ids.len());

    // One-time inputs: identity page table (pages allocated linearly by
    // position), single-sequence query bounds.
    let n_pages = match rt.manifest.buffers["block_table"].shape.as_slice() {
        [kern_manifest::types::Dim::Const(n)] => *n as i32,
        s => panic!("unexpected block_table shape {s:?}"),
    };
    rt.write_input("block_table", &le_bytes_i32(&(0..n_pages).collect::<Vec<_>>()))?;
    rt.write_input("cu_seqlens_q", &le_bytes_i32(&[0, 1]))?;

    let env = BTreeMap::from([("tokens".to_string(), 1u64)]);
    if !o.eager {
        let t = Instant::now();
        rt.capture("decode", &env)?;
        eprintln!(
            "[kern-run] CUDA graph: `decode` stream-captured at tokens=1, {} dispatches -> \
             1 graph launch/step ({:?})",
            rt.manifest.programs["decode"].dispatches.len(),
            t.elapsed()
        );
    }
    let mut generated: Vec<i64> = Vec::new();
    let mut pos: i64 = 0;
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
            eprintln!("[kern-run] stop token {next} at pos {pos}");
            break;
        }
        generated.push(next);
        if generated.len() >= o.steps || pos as u64 + 1 >= o.capacity {
            break;
        }
    }

    let gen_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&gen_u32, false).map_err(|e| format!("decode: {e}"))?;
    eprintln!(
        "[kern-run] {} tokens generated, {:.1} ms/step ({:.1} tok/s)",
        generated.len(),
        decode_ns as f64 / 1e6 / decode_steps.max(1) as f64,
        decode_steps as f64 * 1e9 / decode_ns.max(1) as f64,
    );
    println!("{}{}", o.prompt, text);
    Ok(())
}
