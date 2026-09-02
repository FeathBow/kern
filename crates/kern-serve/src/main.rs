//! `kern-serve`: OpenAI-compatible HTTP over a kern.toml target.
//!
//! Same lookup as `kern run`: the manifest, kernels and weights come from
//! the target in the nearest `kern.toml`; `--model-path` is the HF directory
//! the frontend reads (tokenizer, chat template, stop tokens).

use std::path::PathBuf;

use anyhow::{bail, ensure, Result};
use clap::Parser;
use kern_run::config::Config;

#[derive(Parser)]
#[command(name = "kern-serve", version, about = "serve a kern manifest over an OpenAI-compatible HTTP endpoint")]
struct Cli {
    /// kern.toml to use (default: the nearest one at or above the cwd)
    #[arg(long)]
    config: Option<PathBuf>,
    /// Target in kern.toml (needed when it declares several)
    target: Option<String>,
    #[command(flatten)]
    opts: kern_serve::ServeOpts,
}

fn main() -> Result<()> {
    kern_serve::logline::init();
    let cli = Cli::parse();
    let Some(c) = Config::find(cli.config.as_deref())?.filter(|c| !c.targets.is_empty()) else {
        bail!("kern-serve needs a kern.toml target (manifest, kernels, weights)");
    };
    let (_, t) = c.one(cli.target.as_deref())?;
    ensure!(!t.weights.is_empty(), "target declares no weights");
    kern_serve::serve(
        cli.opts,
        kern_serve::Artifacts { manifest: t.manifest.clone(), kernels: t.kernels.clone(), weights: t.weights.clone() },
        kern_serve::Defaults { gpu: c.gpu, capacity: c.capacity, chunk: c.run.chunk },
    )
}
