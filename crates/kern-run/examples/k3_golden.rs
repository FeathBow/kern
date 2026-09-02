//! Kimi-K3 decode gate: replay pegainfer's golden fixture through kern's
//! `decode` program, one sequence per rank, and compare every step's argmax.
//!
//!   cargo run --release -p kern-run --example k3_golden -- \
//!       --manifest examples/k3-4l-ep4.json --weights /data/susun/kern-k3/4l \
//!       --fixture <pegainfer>/pegainfer-k3/tests/fixtures/k3_4l_greedy.json \
//!       --gpus 0,1,2,3 [--graph] [--iters 50] [--margin-abs 0.125]
//!
//! The fixture feeds `prompt + greedy continuation` one token at a time from
//! position 0 (pure decode, no prefill) and records the reference's argmax
//! and top-5 logits after each. A step whose reference margin is inside the
//! measured noise floor (2 bf16 ULP at the logit's magnitude) is excused when
//! the sampled token is one of the reference's top 5 — pegainfer's own rule
//! (tests/golden_decode.rs). Every rank runs the same sequence, so an EP
//! world must also agree with itself token for token.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier, Mutex};

use kern_manifest::types::Dim;
use kern_run::Caller;
use kern_runtime::{PeerHandle, Runtime, Topology};

const NOISE_FLOOR_ULP: f32 = 2.0;

struct Golden {
    feed: Vec<i64>,
    argmax: Vec<i64>,
    top5: Vec<Vec<i64>>,
    top5_logits: Vec<Vec<f32>>,
    num_layers: usize,
    /// An absolute top-1/top-2 margin that counts as a coin flip, for
    /// fixtures whose `top5_logits` are logprobs (tools/k3_oracle_dump.py);
    /// `None` uses the bf16-ULP rule on logits.
    noise_abs: Option<f32>,
}

impl Golden {
    fn load(path: &Path) -> Golden {
        let j: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("fixture json");
        let steps = j["steps"].as_array().expect("steps");
        let ints = |v: &serde_json::Value| v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect::<Vec<_>>();
        Golden {
            feed: steps.iter().map(|s| s["feed"].as_i64().unwrap()).collect(),
            argmax: steps.iter().map(|s| s["argmax"].as_i64().unwrap()).collect(),
            top5: steps.iter().map(|s| ints(&s["top5_ids"])).collect(),
            top5_logits: steps
                .iter()
                .map(|s| s["top5_logits"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect())
                .collect(),
            num_layers: j["num_layers"].as_u64().expect("num_layers") as usize,
            noise_abs: None,
        }
    }

    fn margin_ulp(&self, step: usize) -> f32 {
        let top = self.top5_logits[step][0];
        let ulp = f32::from_bits((top.abs().to_bits() & 0x7f80_0000).max(1)) / 128.0;
        (top - self.top5_logits[step][1]) / ulp
    }

    /// Exact match, or a coin flip the reference itself decided inside the
    /// noise floor with our pick among its top 5.
    fn coin_flip(&self, step: usize) -> bool {
        match self.noise_abs {
            Some(m) => self.top5_logits[step][0] - self.top5_logits[step][1] <= m,
            None => self.margin_ulp(step) <= NOISE_FLOOR_ULP,
        }
    }

    fn accept(&self, step: usize, got: i64) -> (bool, bool) {
        let exact = got == self.argmax[step];
        let excused = !exact && self.coin_flip(step) && self.top5[step].contains(&got);
        (exact, excused)
    }
}

fn stage_cubins(cubins: &Path, kernels: &Path) {
    std::fs::create_dir_all(kernels).unwrap();
    for entry in std::fs::read_dir(cubins).expect("cubins dir") {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "cubin") {
            let bytes = std::fs::read(&path).unwrap();
            let sha = format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(&bytes));
            let stem = path.file_stem().unwrap().to_string_lossy();
            std::fs::write(kernels.join(format!("{stem}-{}.cubin", &sha[..12])), &bytes).unwrap();
        }
    }
}

/// The weight blobs one rank needs: the shared dense files plus its expert
/// shard, all memory-mapped.
fn weight_files(weights: &Path, layers: usize, ranks: usize, rank: usize) -> Vec<PathBuf> {
    let mut files = vec![weights.join("dense/bookends.safetensors")];
    for i in 0..layers {
        files.push(weights.join(format!("dense/l{i}.safetensors")));
    }
    for i in 1..layers {
        files.push(weights.join(format!("experts/ep{ranks}-r{rank}-l{i}.safetensors")));
    }
    files
}

struct Outcome {
    tokens: Vec<i64>,
    exact: usize,
    excused: usize,
    failures: Vec<String>,
    step_ms: Option<f64>,
}

#[allow(clippy::too_many_arguments)]
fn run_rank(
    json: &str,
    kernels: &Path,
    gpu: usize,
    topo: &Topology,
    files: &[PathBuf],
    golden: &Golden,
    graph: bool,
    iters: usize,
    rendezvous: &dyn Fn(&mut Runtime) -> kern_runtime::Result<()>,
) -> anyhow::Result<Outcome> {
    let manifest = kern_manifest::types::Manifest::from_json(json)?;
    let table = &manifest.buffers["block_table"];
    let capacity = match table.shape.as_slice() {
        [_, Dim::Const(pages)] => pages * table.domain.as_ref().map(|d| d.stride).unwrap_or(1),
        s => anyhow::bail!("unexpected block_table shape {s:?}"),
    };
    let mut rt = Runtime::load(json, kernels, gpu, Some(capacity), Some(topo))?;
    let maps: Vec<memmap2::Mmap> = files
        .iter()
        .map(|f| {
            let file = std::fs::File::open(f).map_err(|e| anyhow::anyhow!("{}: {e}", f.display()))?;
            Ok(unsafe { memmap2::Mmap::map(&file)? })
        })
        .collect::<anyhow::Result<_>>()?;
    let blobs: Vec<&[u8]> = maps.iter().map(|m| &m[..]).collect();
    rt.load_weights(&blobs)?;
    rendezvous(&mut rt)?;
    let mut caller = Caller::new(rt)?;

    let mut out = Outcome { tokens: Vec::new(), exact: 0, excused: 0, failures: Vec::new(), step_ms: None };
    let mut captured = false;
    let mut env = BTreeMap::new();
    for (step, &tok) in golden.feed.iter().enumerate() {
        env = caller.stage_decode(tok)?;
        if graph {
            if !captured {
                caller.rt.capture("decode", &env)?;
                captured = true;
            }
            caller.rt.run_captured("decode", &env)?;
        } else {
            caller.rt.run("decode", &env)?;
        }
        caller.advance(1);
        let bytes = caller.rt.read_output("next_token")?;
        let got = i64::from_le_bytes(bytes[..8].try_into().unwrap());
        out.tokens.push(got);
        let (exact, excused) = golden.accept(step, got);
        if exact {
            out.exact += 1;
        } else if excused {
            out.excused += 1;
        } else {
            out.failures.push(format!(
                "step {step}: got {got}, reference {} (margin {:.1} ulp, top5 {:?})",
                golden.argmax[step],
                golden.margin_ulp(step),
                golden.top5[step]
            ));
        }
    }
    if iters > 0 {
        if !captured {
            caller.rt.capture("decode", &env)?;
        }
        // Median ms per replay.
        out.step_ms = Some(caller.rt.time_captured("decode", &env, iters)? as f64);
    }
    Ok(out)
}

fn main() {
    let mut manifest = PathBuf::from("examples/k3-4l-ep1.json");
    let mut weights = PathBuf::from("/data/susun/kern-k3/4l");
    let mut fixture = PathBuf::from("tests/fixtures/k3_4l_greedy.json");
    let mut cubins = PathBuf::from("target/cubins");
    let mut gpus: Vec<usize> = vec![0];
    let mut graph = false;
    let mut iters = 0usize;
    let mut margin_abs: Option<f32> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().expect("value");
        match a.as_str() {
            "--manifest" => manifest = PathBuf::from(v()),
            "--weights" => weights = PathBuf::from(v()),
            "--fixture" => fixture = PathBuf::from(v()),
            "--cubins" => cubins = PathBuf::from(v()),
            "--gpus" => gpus = v().split(',').map(|s| s.parse().unwrap()).collect(),
            "--graph" => graph = true,
            "--iters" => iters = v().parse().unwrap(),
            "--margin-abs" => margin_abs = Some(v().parse().unwrap()),
            _ => panic!("unknown arg {a}"),
        }
    }
    let json = std::fs::read_to_string(&manifest).expect("manifest");
    let mut golden = Golden::load(&fixture);
    golden.noise_abs = margin_abs;
    let golden = Arc::new(golden);
    let n = gpus.len();
    let kernels = std::env::temp_dir().join(format!("kern-k3-golden-{}", std::process::id()));
    stage_cubins(&cubins, &kernels);
    println!(
        "{}: {} layers, EP{n} on gpus {gpus:?}, {} fixture steps, {}",
        manifest.display(),
        golden.num_layers,
        golden.feed.len(),
        if graph { "graph replay" } else { "eager" }
    );

    let posted: Arc<Mutex<Vec<Option<BTreeMap<String, PeerHandle>>>>> = Arc::new(Mutex::new(vec![None; n]));
    let gate = Arc::new(Barrier::new(n));
    let results: Arc<Mutex<Vec<Option<Result<Outcome, String>>>>> = Arc::new(Mutex::new((0..n).map(|_| None).collect()));
    let mut threads = Vec::new();
    for (rank, &gpu) in gpus.iter().enumerate() {
        let (json, kernels, posted, gate, results, golden, weights) =
            (json.clone(), kernels.clone(), posted.clone(), gate.clone(), results.clone(), golden.clone(), weights.clone());
        threads.push(std::thread::spawn(move || {
            let files = weight_files(&weights, golden.num_layers, n, rank);
            let rendezvous = |rt: &mut Runtime| -> kern_runtime::Result<()> {
                let mine = rt.export_handles()?;
                posted.lock().unwrap()[rank] = Some(mine);
                gate.wait();
                let members: Vec<_> = posted.lock().unwrap().iter().map(|m| m.clone().unwrap()).collect();
                rt.import_peers("ep", &members)
            };
            let r = run_rank(
                &json,
                &kernels,
                gpu,
                &Topology::one("ep", rank as u64, n as u64),
                &files,
                &golden,
                graph,
                iters,
                &rendezvous,
            );
            results.lock().unwrap()[rank] = Some(r.map_err(|e| format!("{e:#}")));
        }));
    }
    for th in threads {
        th.join().unwrap();
    }
    let results = results.lock().unwrap();
    let mut ok = true;
    let mut first: Option<Vec<i64>> = None;
    for (rank, r) in results.iter().enumerate() {
        match r {
            Some(Ok(o)) => {
                let steps = golden.feed.len();
                println!(
                    "rank {rank} gpu {}: {}/{steps} exact, {} excused inside the noise floor, {} wrong{}",
                    gpus[rank],
                    o.exact,
                    o.excused,
                    o.failures.len(),
                    o.step_ms.map(|ms| format!("; {ms:.3} ms/step (captured, {iters} iters)")).unwrap_or_default()
                );
                for f in &o.failures {
                    println!("  {f}");
                }
                ok &= o.failures.is_empty();
                match &first {
                    None => first = Some(o.tokens.clone()),
                    Some(t) if *t != o.tokens => {
                        println!("  rank {rank} disagrees with rank 0 on the sampled tokens");
                        ok = false;
                    }
                    _ => {}
                }
            }
            Some(Err(e)) => {
                println!("rank {rank} gpu {}: FAILED: {e}", gpus[rank]);
                ok = false;
            }
            None => unreachable!(),
        }
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
}
