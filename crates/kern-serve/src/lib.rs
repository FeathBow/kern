//! `kern serve`: an OpenAI-compatible HTTP endpoint over a kern manifest.
//!
//! The HTTP/protocol stack is pegainfer's frontend (vLLM's Rust server
//! crates underneath: completions, chat completions, streaming, chat
//! templates, stop strings). This crate contributes the engine behind it:
//! [`scheduler::KernScheduler`], the pegainfer `Scheduler` contract over a
//! `kern_runtime::Runtime`, and [`pages::PagePool`], the KV page owner.

pub mod pages;
pub mod scheduler;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use kern_runtime::Runtime;
use pegainfer_frontend::engine::{
    drive, scheduler_pair, Engine, EngineInfo, KvCapacity, LaunchedEngine, LiveScheduler,
};
use pegainfer_frontend::vllm;
use tracing::info;

use scheduler::{KernScheduler, Policy};

/// The manifest and its artifacts (from kern.toml's target or flags).
pub struct Artifacts {
    pub manifest: PathBuf,
    pub kernels: PathBuf,
    pub weights: Vec<PathBuf>,
}

/// Defaults a kern.toml may supply.
#[derive(Default)]
pub struct Defaults {
    pub gpu: Option<usize>,
    pub capacity: Option<u64>,
    pub chunk: Option<u64>,
}

#[derive(Args, Debug, Clone)]
pub struct ServeOpts {
    /// HF-layout model directory for the frontend: config.json, tokenizer,
    /// chat template, generation_config (stop tokens)
    #[arg(long)]
    pub model_path: PathBuf,

    /// Model id served by the API (default: the manifest's `model`)
    #[arg(long)]
    pub served_model_name: Option<String>,

    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    /// CUDA device ordinal
    #[arg(long)]
    pub gpu: Option<usize>,

    /// KV pool in tokens (rounded down to the page); every request reserves
    /// its worst case `prompt + max_tokens` at admission
    #[arg(long)]
    pub capacity: Option<u64>,

    /// Prefill chunk in tokens
    #[arg(long)]
    pub chunk: Option<u64>,

    /// Prompt tokens one step may prefill before it decodes
    #[arg(long, default_value_t = 2048)]
    pub prefill_budget: usize,

    /// Cap on concurrently running sequences (≤ the manifest's `seqs` bound)
    #[arg(long, default_value_t = 256)]
    pub max_seqs: usize,

    /// Skip CUDA graph capture, launch every call eagerly
    #[arg(long)]
    pub eager: bool,

    /// Extra stop token ids (generation_config.json's eos ids always apply)
    #[arg(long, value_delimiter = ',')]
    pub stop_tokens: Vec<u32>,
}

/// Stop tokens from the HF directory: generation_config.json `eos_token_id`
/// (int or list), else config.json's.
fn hf_stop_tokens(model_path: &Path) -> Vec<u32> {
    let read = |f: &str| -> Option<serde_json::Value> {
        serde_json::from_str(&std::fs::read_to_string(model_path.join(f)).ok()?).ok()
    };
    let ids = |v: &serde_json::Value| -> Vec<u32> {
        match v.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => n.as_u64().into_iter().map(|x| x as u32).collect(),
            Some(serde_json::Value::Array(a)) => a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect(),
            _ => Vec::new(),
        }
    };
    let mut out = read("generation_config.json").map(|v| ids(&v)).unwrap_or_default();
    if out.is_empty() {
        out = read("config.json").map(|v| ids(&v)).unwrap_or_default();
    }
    out
}

pub fn serve(o: ServeOpts, art: Artifacts, d: Defaults) -> Result<()> {
    // The frontend logs through `log`; route it into tracing.
    let _ = tracing_log::LogTracer::init();
    let gpu = o.gpu.or(d.gpu).unwrap_or(0);
    let capacity = o.capacity.or(d.capacity).unwrap_or(65536);
    let chunk = o.chunk.or(d.chunk).unwrap_or(512) as usize;
    let mut stop_tokens = hf_stop_tokens(&o.model_path);
    stop_tokens.extend(&o.stop_tokens);
    stop_tokens.sort_unstable();
    stop_tokens.dedup();
    anyhow::ensure!(!stop_tokens.is_empty(), "no stop tokens: none in {}/generation_config.json or config.json and no --stop-tokens", o.model_path.display());
    info!("stop tokens {stop_tokens:?}");

    let manifest_json = std::fs::read_to_string(&art.manifest)
        .with_context(|| format!("reading manifest {}", art.manifest.display()))?;
    let served_name = o.served_model_name.clone().unwrap_or_else(|| {
        serde_json::from_str::<serde_json::Value>(&manifest_json)
            .ok()
            .and_then(|v| v.get("model")?.as_str().map(str::to_owned))
            .unwrap_or_else(|| "kern".to_owned())
    });

    // The scheduler thread owns the runtime for its whole life: load there,
    // report readiness, then drive.
    let (handle, backend) = scheduler_pair();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<scheduler::Facts>>();
    let policy = Policy {
        chunk,
        prefill_budget: o.prefill_budget,
        eager: o.eager,
        max_seqs: o.max_seqs,
        stop_tokens,
    };
    let join = std::thread::Builder::new()
        .name("kern-scheduler".into())
        .spawn(move || {
            let load = || -> Result<(KernScheduler, scheduler::Facts)> {
                let t0 = Instant::now();
                let mut rt = Runtime::load(&manifest_json, &art.kernels, gpu, capacity)?;
                info!("manifest `{}` verified, {} modules loaded ({:?})", rt.manifest.model, rt.module_count(), t0.elapsed());
                let t0 = Instant::now();
                let blobs = art
                    .weights
                    .iter()
                    .map(|p| std::fs::read(p).with_context(|| format!("reading weights {}", p.display())))
                    .collect::<Result<Vec<_>>>()?;
                rt.load_weights(&blobs.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
                drop(blobs);
                info!("weights bound ({:?})", t0.elapsed());
                KernScheduler::new(rt, policy)
            };
            match load() {
                Ok((sched, facts)) => {
                    let _ = ready_tx.send(Ok(facts));
                    drive(sched, backend);
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            }
        })
        .context("spawning the scheduler thread")?;

    let engine = async move {
        let facts = tokio::task::spawn_blocking(move || ready_rx.recv())
            .await
            .context("scheduler thread died before reporting readiness")?
            .context("scheduler thread died before reporting readiness")??;
        Ok(LaunchedEngine::Stepped(Engine {
            schedulers: vec![LiveScheduler { handle, join }],
            info: EngineInfo {
                kv_capacity: Some(KvCapacity { total_blocks: facts.total_blocks, block_size: facts.block_size }),
                servable_len: Some(facts.max_request_tokens as u32),
            },
            lora: None,
        }))
    };

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    info!("serving `{served_name}` on 0.0.0.0:{} (frontend model dir {})", o.port, o.model_path.display());
    rt.block_on(async move {
        // Needs the runtime: it spawns the signal listener.
        let shutdown = vllm::shutdown_token_from_ctrl_c();
        vllm::serve_with_engine_count(engine, &o.model_path, vec![served_name], o.port, None, 1, shutdown).await
    })
}
