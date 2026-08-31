//! kern-attest: evidence for a kernel swap.
//!
//! Given two manifests A (the reference — assumed correct) and B (the
//! candidate), attest:
//!   1. diffs them structurally: which kernels changed (interface / impl /
//!      added / removed) and, per program, the aligned dispatch segments
//!      that differ ("cuts") — everything else is shared;
//!   2. taps a real workload once: A and B run prefill + one decode step in
//!      lockstep; at every cut the frontier inputs and A's outputs are
//!      snapshotted and B's outputs compared (local);
//!   3. measures the noise floor per cut: A's cut re-run from its own
//!      snapshot against its own output;
//!   4. fuzzes each cut from the snapshot: frontier inputs synthesized from
//!      several value distributions (integers only where the buffer
//!      declares a domain), both sides run, outputs compared and checked
//!      against declared domains;
//!   5. times the cuts in isolation (plus, opt-in, graph-level step time
//!      and a prefill symbol sweep) and computes a static bytes-moved
//!      roofline for the changed kernels.
//!
//! Everything after the tap is cut-local: cost scales with the cut, not the
//! model. There is deliberately no end-to-end generation: bit-identity at
//! every cut implies it, and differences beyond the noise floor are
//! reported as INCONCLUSIVE rather than adjudicated here.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use kern_manifest::types::{Arg, BufferClass, DType, Dim, Dir, Manifest, ParamType, Program};
use kern_manifest::verify;
use kern_run::{env, Caller, DRIVEN, TOKENS};
use kern_runtime::{values, Runtime};
use serde_json::{json, Value};

/// A/B evidence for a kernel swap (qwen3-4b caller contract).
#[derive(Parser)]
#[command(version, about)]
struct Opts {
    /// Reference manifest (assumed correct)
    #[arg(long, default_value = "examples/qwen3-4b.json")]
    a: PathBuf,
    /// Candidate manifest
    #[arg(long)]
    b: PathBuf,
    /// Directory holding the .cubin modules (both manifests)
    #[arg(long, default_value = "kernels")]
    kernels: PathBuf,
    /// Safetensors artifact(s); repeat for a target + draft pair
    #[arg(long, default_value = "weights/qwen3-4b-decode.safetensors")]
    weights: Vec<PathBuf>,
    #[arg(long, default_value = "weights/tokenizer.json")]
    tokenizer: PathBuf,
    /// Prompt for the real-workload tap
    #[arg(long, default_value = DEFAULT_PROMPT)]
    prompt: String,
    /// Fuzz rounds per cut (0 disables); rounds cycle through the value
    /// distributions
    #[arg(long, default_value_t = 6)]
    fuzz: usize,
    #[arg(long, default_value_t = 0)]
    gpu: usize,
    #[arg(long, default_value_t = 4096)]
    capacity: u64,
    #[arg(long, default_value_t = 512)]
    chunk: u64,
    /// Replays for cut timing (minimum is reported)
    #[arg(long, default_value_t = 20)]
    iters: usize,
    /// Skip capturing both decode programs as CUDA graphs for the step time
    #[arg(long)]
    no_graph_step: bool,
    /// Skip sweeping prefill over the `tokens` symbol range
    #[arg(long)]
    no_sweep: bool,
    /// Device peak memory bandwidth in GB/s, for the roofline column
    #[arg(long, default_value_t = 8000.0)]
    peak_bw: f64,
    /// Write the attestation as JSON here
    #[arg(long)]
    out: Option<PathBuf>,
    /// Only print the structural diff
    #[arg(long)]
    diff_only: bool,
    /// Skip the timing section
    #[arg(long)]
    no_perf: bool,
    /// Skip the noise-floor re-runs
    #[arg(long)]
    no_noise: bool,
    /// Seed for the fuzz generator
    #[arg(long, default_value_t = 0x5eed)]
    seed: u64,
    /// Report format on stdout
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,
    /// ANSI color (text format only)
    #[arg(long, value_enum, default_value_t = Color::Auto)]
    color: Color,
}

const DEFAULT_PROMPT: &str =
    "The lighthouse keeper had not spoken to another person in eleven days when the boat appeared";

// ---------------------------------------------------------------- static diff

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Same,
    Changed,
}

#[derive(Clone, Debug)]
struct Segment {
    kind: Kind,
    a: (usize, usize),
    b: (usize, usize),
}

fn kernel_changes(ma: &Manifest, mb: &Manifest) -> BTreeMap<String, &'static str> {
    let mut out = BTreeMap::new();
    let names: BTreeSet<&String> = ma.kernels.keys().chain(mb.kernels.keys()).collect();
    for n in names {
        let kind = match (ma.kernels.get(n), mb.kernels.get(n)) {
            (Some(_), None) => "removed",
            (None, Some(_)) => "added",
            (Some(x), Some(y)) => {
                if json!(x.params) != json!(y.params) {
                    "interface"
                } else if json!(x.imp) != json!(y.imp) {
                    "impl"
                } else {
                    continue;
                }
            }
            (None, None) => unreachable!(),
        };
        out.insert(n.clone(), kind);
    }
    out
}

/// Align two dispatch lists (LCS over canonical dispatch keys; a dispatch of
/// a changed kernel never matches across sides) into Same/Changed segments.
fn align(pa: &Program, pb: &Program, changed: &BTreeMap<String, &str>) -> Vec<Segment> {
    let key = |d: &kern_manifest::types::Dispatch, side: &str| {
        if changed.contains_key(&d.kernel) {
            format!("{}@{side}#{}", d.kernel, json!(d.args))
        } else {
            format!("{}#{}", d.kernel, json!(d.args))
        }
    };
    let ka: Vec<String> = pa.dispatches.iter().map(|d| key(d, "A")).collect();
    let kb: Vec<String> = pb.dispatches.iter().map(|d| key(d, "B")).collect();
    let (n, m) = (ka.len(), kb.len());
    let mut lcs = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if ka[i] == kb[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }
    let mut pairs = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if ka[i] == kb[j] {
            pairs.push((i, j));
            i += 1;
            j += 1;
        } else if lcs[i + 1][j] >= lcs[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    let mut segs: Vec<Segment> = Vec::new();
    let (mut ia, mut ib) = (0, 0);
    for (i, j) in pairs {
        if i > ia || j > ib {
            segs.push(Segment { kind: Kind::Changed, a: (ia, i), b: (ib, j) });
        }
        match segs.last_mut() {
            Some(s) if s.kind == Kind::Same && s.a.1 == i && s.b.1 == j => {
                s.a.1 = i + 1;
                s.b.1 = j + 1;
            }
            _ => segs.push(Segment { kind: Kind::Same, a: (i, i + 1), b: (j, j + 1) }),
        }
        ia = i + 1;
        ib = j + 1;
    }
    if ia < n || ib < m {
        segs.push(Segment { kind: Kind::Changed, a: (ia, n), b: (ib, m) });
    }
    segs
}

// ------------------------------------------------------------- dataflow view

#[derive(Default, Clone)]
struct Access {
    reads: BTreeSet<String>,
    writes: BTreeSet<String>,
    state_reads: BTreeSet<String>,
    state_writes: BTreeSet<String>,
}

fn access(m: &Manifest, prog: &str, lo: usize, hi: usize) -> Access {
    let mut acc = Access::default();
    for d in &m.programs[prog].dispatches[lo..hi] {
        let k = &m.kernels[&d.kernel];
        for (arg, p) in d.args.iter().zip(&k.params) {
            match (arg, p) {
                (Arg::Buf { buf, .. }, ParamType::Buf { dir, .. }) => {
                    if matches!(dir, Dir::In | Dir::InOut) {
                        acc.reads.insert(buf.clone());
                    }
                    if matches!(dir, Dir::Out | Dir::InOut) {
                        acc.writes.insert(buf.clone());
                    }
                }
                (Arg::State { state, .. }, ParamType::Ptr { dir }) => {
                    if matches!(dir, Dir::In | Dir::InOut) {
                        acc.state_reads.insert(state.clone());
                    }
                    if matches!(dir, Dir::Out | Dir::InOut) {
                        acc.state_writes.insert(state.clone());
                    }
                }
                _ => {}
            }
        }
    }
    acc
}

/// Buffers a range reads before it writes them: what the cut consumes from
/// outside.
fn frontier_inputs(m: &Manifest, prog: &str, lo: usize, hi: usize) -> BTreeSet<String> {
    let mut written = BTreeSet::new();
    let mut inputs = BTreeSet::new();
    for d in &m.programs[prog].dispatches[lo..hi] {
        let k = &m.kernels[&d.kernel];
        for (arg, p) in d.args.iter().zip(&k.params) {
            if let (Arg::Buf { buf, .. }, ParamType::Buf { dir, .. }) = (arg, p) {
                if matches!(dir, Dir::In | Dir::InOut) && !written.contains(buf) {
                    inputs.insert(buf.clone());
                }
                if matches!(dir, Dir::Out | Dir::InOut) {
                    written.insert(buf.clone());
                }
            }
        }
    }
    inputs
}

fn live_elems(m: &Manifest, name: &str, e: &BTreeMap<String, u64>) -> usize {
    m.buffers[name]
        .shape
        .iter()
        .map(|d| match d {
            Dim::Const(c) => *c as usize,
            Dim::Sym(s) => e[s] as usize,
        })
        .product()
}

fn live_bytes(m: &Manifest, name: &str, e: &BTreeMap<String, u64>) -> usize {
    live_elems(m, name, e) * m.buffers[name].dtype.bytes() as usize
}

fn is_float(dt: DType) -> bool {
    matches!(dt, DType::Bf16 | DType::F16 | DType::F32 | DType::Fp8E4m3)
}

// ---------------------------------------------------------------- comparison

#[derive(Clone, Debug, Default)]
struct Cmp {
    n: usize,
    n_diff: usize,
    max_ulp: Option<u64>,
    max_abs: f64,
    nan_only_one_side: usize,
    /// Bit-different but value-equal: +0 vs -0.
    signed_zero: usize,
}

fn compare(dt: DType, a: &[u8], b: &[u8]) -> Cmp {
    let w = dt.bytes() as usize;
    let mut c = Cmp { n: a.len() / w, ..Default::default() };
    for (x, y) in a.chunks_exact(w).zip(b.chunks_exact(w)) {
        if x == y {
            continue;
        }
        c.n_diff += 1;
        let (fx, fy) = (values::to_f64(dt, x)[0], values::to_f64(dt, y)[0]);
        if fx.is_nan() != fy.is_nan() {
            c.nan_only_one_side += 1;
            continue;
        }
        if fx == fy {
            c.signed_zero += 1; // +0 vs -0: bit-different, value-equal
            continue;
        }
        c.max_abs = c.max_abs.max((fx - fy).abs());
        if is_float(dt) {
            if let Some(u) = values::ulp_distance(dt, x, y) {
                c.max_ulp = Some(c.max_ulp.unwrap_or(0).max(u));
            }
        }
    }
    c
}

impl Cmp {
    fn identical(&self) -> bool {
        self.n_diff == 0
    }
    /// Every difference is a signed zero.
    fn value_identical(&self) -> bool {
        self.n_diff == self.signed_zero
    }
    fn to_json(&self) -> Value {
        json!({"n": self.n, "n_diff": self.n_diff, "max_ulp": self.max_ulp, "max_abs": self.max_abs,
               "nan_mismatch": self.nan_only_one_side, "signed_zero": self.signed_zero})
    }
}

/// Compare the buffers a segment wrote on both sides (live prefix at `e`),
/// plus any state it wrote. Buffers written by one side only are internal
/// to that side's implementation.
fn compare_written(
    a: &Caller,
    b: &Caller,
    acc_a: &Access,
    acc_b: &Access,
    e: &BTreeMap<String, u64>,
    with_states: bool,
) -> Result<(BTreeMap<String, Cmp>, BTreeMap<String, usize>, Vec<String>)> {
    let mut bufs = BTreeMap::new();
    let mut states = BTreeMap::new();
    let mut one_sided = Vec::new();
    for name in acc_a.writes.union(&acc_b.writes) {
        if !(acc_a.writes.contains(name) && acc_b.writes.contains(name)) {
            one_sided.push(name.clone());
            continue;
        }
        let bytes = live_bytes(&a.rt.manifest, name, e);
        let x = a.rt.read_buffer_prefix(name, bytes)?;
        let y = b.rt.read_buffer_prefix(name, bytes)?;
        bufs.insert(name.clone(), compare(a.rt.manifest.buffers[name].dtype, &x, &y));
    }
    if with_states {
        for name in acc_a.state_writes.union(&acc_b.state_writes) {
            let x = a.rt.read_state(name)?;
            let y = b.rt.read_state(name)?;
            let n = x.iter().zip(&y).filter(|(p, q)| p != q).count();
            states.insert(name.clone(), n);
        }
    }
    Ok((bufs, states, one_sided))
}

// ------------------------------------------------------------------- fuzzing

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit().max(1e-300), self.unit());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

const DISTS: [&str; 6] = ["uniform", "normal", "laplace", "outliers", "edge", "special"];

fn gen_float(rng: &mut Rng, dist: usize, n: usize, dt: DType) -> Vec<f64> {
    let max = match dt {
        DType::Bf16 => 3.0e38,
        DType::F16 => 65504.0,
        DType::F32 => 3.0e38,
        DType::Fp8E4m3 => 448.0,
        _ => unreachable!(),
    };
    let tiny = match dt {
        DType::Bf16 | DType::F32 => 1e-39,
        DType::F16 => 1e-6,
        DType::Fp8E4m3 => 0.002,
        _ => unreachable!(),
    };
    (0..n)
        .map(|_| match DISTS[dist % DISTS.len()] {
            "uniform" => rng.unit() * 2.0 - 1.0,
            "normal" => rng.normal(),
            "laplace" => {
                let u = rng.unit() - 0.5;
                -u.signum() * (1.0 - 2.0 * u.abs()).max(1e-300).ln()
            }
            "outliers" => {
                let x = rng.normal();
                if rng.below(100) == 0 { x * 100.0 } else { x }
            }
            "edge" => {
                let choices = [0.0, -0.0, tiny, -tiny, max / 2.0, -max / 2.0, 1.0, -1.0, 0.5, -0.5];
                choices[rng.below(choices.len() as u64) as usize]
            }
            _ => match rng.below(200) {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                _ => rng.normal(),
            },
        })
        .collect()
}

fn gen_int(rng: &mut Rng, n: usize, lo: f64, hi: f64, monotone: bool) -> Vec<f64> {
    let span = (hi - lo + 1.0).max(1.0) as u64;
    let mut v: Vec<f64> = (0..n).map(|_| lo + rng.below(span) as f64).collect();
    if monotone {
        v.sort_by(|a, b| a.total_cmp(b));
    }
    v
}

// ------------------------------------------------------------------- driver

struct Sides {
    a: Caller,
    b: Caller,
}

/// One cut with a real-workload snapshot: what it consumed (frontier inputs)
/// and what A produced from them.
struct Snap {
    program: String,
    seg: Segment,
    env: BTreeMap<String, u64>,
    inputs: Vec<(String, Vec<u8>)>,
    ref_out: BTreeMap<String, Vec<u8>>,
    ref_states: BTreeMap<String, Vec<u8>>,
}

fn load_side(json: &str, o: &Opts, blobs: &[&[u8]]) -> Result<Caller> {
    let mut rt = Runtime::load(json, &o.kernels, o.gpu, o.capacity)?;
    rt.load_weights(blobs)?;
    Caller::new(rt)
}

fn tokens_of(tok: &tokenizers::Tokenizer, s: &str) -> Result<Vec<i64>> {
    let ids: Vec<i64> = tok
        .encode(s, false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .iter()
        .map(|&u| u as i64)
        .collect();
    if ids.len() < 2 {
        bail!("prompt too short: {s:?}");
    }
    Ok(ids)
}

fn seg_label(s: &Segment) -> String {
    format!("A[{}..{}) B[{}..{})", s.a.0, s.a.1, s.b.0, s.b.1)
}

// ------------------------------------------------------------------ report
//
// Sections of aligned tables, rendered as colored text (tty) or GitHub
// markdown. Everything goes to stdout; the JSON report is `--out`.

#[derive(Clone, Copy, PartialEq, Debug, Default)]
enum Style {
    #[default]
    Plain,
    Bold,
    Dim,
    Good,
    Warn,
    Bad,
}

#[derive(Clone, Debug, Default)]
struct Cell {
    s: String,
    st: Style,
}

impl Cell {
    fn new(s: impl Into<String>, st: Style) -> Cell {
        Cell { s: s.into(), st }
    }
    fn good(s: impl Into<String>) -> Cell {
        Cell::new(s, Style::Good)
    }
    fn warn(s: impl Into<String>) -> Cell {
        Cell::new(s, Style::Warn)
    }
    fn bad(s: impl Into<String>) -> Cell {
        Cell::new(s, Style::Bad)
    }
    fn dim(s: impl Into<String>) -> Cell {
        Cell::new(s, Style::Dim)
    }
    fn bold(s: impl Into<String>) -> Cell {
        Cell::new(s, Style::Bold)
    }
}

impl From<String> for Cell {
    fn from(s: String) -> Cell {
        Cell { s, st: Style::Plain }
    }
}
impl From<&str> for Cell {
    fn from(s: &str) -> Cell {
        Cell { s: s.into(), st: Style::Plain }
    }
}

macro_rules! row {
    ($($x:expr),* $(,)?) => { vec![$(Cell::from($x)),*] };
}

enum Block {
    Table { header: bool, rows: Vec<Vec<Cell>> },
    Note(Cell),
}

struct Section {
    title: String,
    subtitle: String,
    blocks: Vec<Block>,
    started: Instant,
    timed: bool,
}

impl Section {
    fn new(title: &str, subtitle: &str) -> Section {
        Section { title: title.into(), subtitle: subtitle.into(), blocks: Vec::new(), started: Instant::now(), timed: true }
    }
    fn untimed(mut self) -> Section {
        self.timed = false;
        self
    }
    fn table(&mut self, rows: Vec<Vec<Cell>>) {
        self.blocks.push(Block::Table { header: false, rows });
    }
    /// First row is a header.
    fn table_h(&mut self, rows: Vec<Vec<Cell>>) {
        self.blocks.push(Block::Table { header: true, rows });
    }
    fn note(&mut self, c: Cell) {
        self.blocks.push(Block::Note(c));
    }
}

#[derive(Clone, Copy, PartialEq, clap::ValueEnum)]
enum Format {
    Text,
    Md,
}

#[derive(Clone, Copy, PartialEq, clap::ValueEnum)]
enum Color {
    Auto,
    Always,
    Never,
}

struct Renderer {
    format: Format,
    color: bool,
}

impl Renderer {
    fn new(format: Format, color: Color) -> Renderer {
        use std::io::IsTerminal;
        let color = match color {
            Color::Always => true,
            Color::Never => false,
            Color::Auto => {
                std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none()
            }
        } && format == Format::Text;
        Renderer { format, color }
    }

    fn paint(&self, c: &Cell) -> String {
        if !self.color || c.st == Style::Plain || c.s.is_empty() {
            return c.s.clone();
        }
        let code = match c.st {
            Style::Bold => "1",
            Style::Dim => "2",
            Style::Good => "32",
            Style::Warn => "33",
            Style::Bad => "1;31",
            Style::Plain => unreachable!(),
        };
        format!("\x1b[{code}m{}\x1b[0m", c.s)
    }

    fn md_cell(c: &Cell) -> String {
        let esc = c.s.replace('|', "\\|");
        match c.st {
            Style::Bad | Style::Bold => format!("**{esc}**"),
            Style::Warn => format!("*{esc}*"),
            Style::Dim => format!("<sub>{esc}</sub>"),
            _ => esc,
        }
    }

    fn header(&self, a: &str, b: &str) {
        match self.format {
            Format::Text => println!(
                "{}   {} {}   {} {}",
                self.paint(&Cell::bold("kern-attest")),
                self.paint(&Cell::dim("A")),
                a,
                self.paint(&Cell::dim("B")),
                b
            ),
            Format::Md => println!("# kern-attest\n\n`A` {a} → `B` {b}"),
        }
    }

    fn section(&self, sec: &Section) {
        println!();
        match self.format {
            Format::Text => {
                let t = self.paint(&Cell::bold(&sec.title));
                let took = if sec.timed { format!("   {:.1?}", sec.started.elapsed()) } else { String::new() };
                if sec.subtitle.is_empty() {
                    println!("{t}{}", self.paint(&Cell::dim(&took)));
                } else {
                    // Pad on the raw title so escape codes don't skew it.
                    let pad = " ".repeat(15usize.saturating_sub(sec.title.chars().count()));
                    println!("{t}{pad}{}", self.paint(&Cell::dim(&format!("{}{took}", sec.subtitle))));
                }
                for (bi, b) in sec.blocks.iter().enumerate() {
                    match b {
                        Block::Table { header, rows } => {
                            if bi > 0 {
                                println!();
                            }
                            self.text_table(*header, rows)
                        }
                        Block::Note(c) => println!("  {}", self.paint(c)),
                    }
                }
            }
            Format::Md => {
                println!("## {}", sec.title);
                let took = if sec.timed { format!(" ({:.1?})", sec.started.elapsed()) } else { String::new() };
                if !sec.subtitle.is_empty() || sec.timed {
                    println!("\n*{}{took}*", sec.subtitle);
                }
                for b in &sec.blocks {
                    match b {
                        Block::Table { header, rows } => self.md_table(*header, rows),
                        Block::Note(c) => println!("\n{}", Self::md_cell(c)),
                    }
                }
            }
        }
    }

    fn text_table(&self, header: bool, rows: &[Vec<Cell>]) {
        let ncol = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        let width = |s: &str| s.chars().count();
        let mut w = vec![0usize; ncol];
        for r in rows {
            for (i, c) in r.iter().enumerate() {
                w[i] = w[i].max(width(&c.s));
            }
        }
        for (ri, r) in rows.iter().enumerate() {
            let mut line = String::from("  ");
            for (i, c) in r.iter().enumerate() {
                let c = if header && ri == 0 && c.st == Style::Plain { Cell::dim(&c.s) } else { c.clone() };
                line.push_str(&self.paint(&c));
                if i + 1 < r.len() {
                    line.push_str(&" ".repeat(w[i] - width(&c.s) + 3));
                }
            }
            println!("{}", line.trim_end());
        }
    }

    fn md_table(&self, header: bool, rows: &[Vec<Cell>]) {
        let ncol = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        let line = |r: &[Cell]| {
            let mut cells: Vec<String> = r.iter().map(Self::md_cell).collect();
            cells.resize(ncol, String::new());
            format!("| {} |", cells.join(" | "))
        };
        println!();
        let (head, body) = if header { (line(&rows[0]), &rows[1..]) } else { (line(&vec![Cell::default(); ncol]), rows) };
        println!("{head}");
        println!("|{}", " --- |".repeat(ncol));
        for r in body {
            println!("{}", line(r));
        }
    }

    fn verdict(&self, code: i32, summary: &str, elapsed: std::time::Duration, out: Option<&std::path::Path>) {
        println!();
        let tag = match code {
            0 => Cell::good("PASS"),
            1 => Cell::bad("FAIL"),
            _ => Cell::warn("INCONCLUSIVE"),
        };
        match self.format {
            Format::Text => {
                println!(
                    "{}   {}   {}   {}",
                    self.paint(&Cell::bold("VERDICT")),
                    self.paint(&tag),
                    summary,
                    self.paint(&Cell::dim(&format!("{elapsed:.1?}")))
                );
                if let Some(p) = out {
                    println!("{}", self.paint(&Cell::dim(&format!("          attestation written to {}", p.display()))));
                }
            }
            Format::Md => {
                println!("## Verdict\n\n**{}** — {summary} ({elapsed:.1?})", tag.s);
                if let Some(p) = out {
                    println!("\n<sub>attestation: `{}`</sub>", p.display());
                }
            }
        }
    }
}

fn us(ms: f32) -> String {
    if ms >= 1.0 { format!("{ms:.3} ms") } else { format!("{:.1} µs", ms * 1e3) }
}

/// B − A as a styled cell: faster is good.
fn delta(a: f32, b: f32) -> Cell {
    let d = (b - a) * 1e3;
    let pct = (b - a) / a.max(1e-9) * 100.0;
    let s = format!("{}{:.1} µs  {}{:.1}%", if d < 0.0 { "−" } else { "+" }, d.abs(), if pct < 0.0 { "−" } else { "+" }, pct.abs());
    if pct <= -1.0 { Cell::good(s) } else if pct >= 1.0 { Cell::bad(s) } else { Cell::from(s) }
}

fn pct_cell(a: f32, b: f32) -> Cell {
    let pct = (b - a) / a.max(1e-9) * 100.0;
    let s = format!("{}{:.1}%", if pct < 0.0 { "−" } else { "+" }, pct.abs());
    if pct <= -1.0 { Cell::good(s) } else if pct >= 1.0 { Cell::bad(s) } else { Cell::from(s) }
}

fn kb(bytes: usize) -> String {
    match bytes {
        b if b >= 1 << 20 => format!("{:.1} MB", b as f64 / 1e6),
        b if b >= 1 << 10 => format!("{:.0} KB", b as f64 / 1e3),
        b => format!("{b} B"),
    }
}

/// Short styled cell for one comparison.
fn cell(c: &Cmp) -> Cell {
    if c.identical() {
        Cell::good("bit-identical")
    } else if c.value_identical() {
        Cell::warn(format!("±0 only ({})", c.signed_zero))
    } else {
        let mut s = format!("{}/{} differ", c.n_diff - c.signed_zero, c.n);
        if let Some(u) = c.max_ulp {
            s += &format!(" · max {u} ulp");
        } else {
            s += &format!(" · max |Δ| {:.2e}", c.max_abs);
        }
        if c.nan_only_one_side > 0 {
            s += &format!(" · {} nan", c.nan_only_one_side);
        }
        Cell::bad(s)
    }
}

/// Summarize one buffer's comparisons across many cuts.
fn summarize(results: &[(String, Cmp)]) -> Cell {
    let n = results.len();
    let bit = results.iter().filter(|(_, c)| c.identical()).count();
    let val = results.iter().filter(|(_, c)| !c.identical() && c.value_identical()).count();
    if bit == n {
        return Cell::good("bit-identical");
    }
    if bit + val == n {
        let z: usize = results.iter().map(|(_, c)| c.signed_zero).sum();
        return Cell::warn(format!("value-identical · ±0 only ({z} elements)"));
    }
    let worst = results
        .iter()
        .filter(|(_, c)| !c.value_identical())
        .max_by_key(|(_, c)| (c.max_ulp, c.n_diff - c.signed_zero))
        .unwrap();
    Cell::bad(format!("{}/{n} cuts differ · worst {} at {}", n - bit - val, cell(&worst.1).s, worst.0))
}

fn main() -> Result<()> {
    let o = Opts::parse();
    let t_start = Instant::now();
    let ja = std::fs::read_to_string(&o.a).with_context(|| format!("reading {}", o.a.display()))?;
    let jb = std::fs::read_to_string(&o.b).with_context(|| format!("reading {}", o.b.display()))?;
    let ma = Manifest::from_json(&ja)?;
    let mb = Manifest::from_json(&jb)?;
    verify(&ma).with_context(|| format!("A ({}) failed verification", o.a.display()))?;
    verify(&mb).with_context(|| format!("B ({}) failed verification", o.b.display()))?;
    let mut report = json!({
        "a": o.a.display().to_string(), "b": o.b.display().to_string(),
    });
    let r = Renderer::new(o.format, o.color);
    r.header(&o.a.display().to_string(), &o.b.display().to_string());

    // ---- 1. static diff
    let mut sec = Section::new("DIFF", "").untimed();
    let changed = kernel_changes(&ma, &mb);
    let mut rows: Vec<Vec<Cell>> = Vec::new();
    if changed.is_empty() {
        rows.push(row!["kernels", "no interface or implementation differs"]);
    }
    let step_desc = |k: &kern_manifest::types::Kernel| -> String {
        let s = &k.imp.steps;
        let mut names: Vec<String> = s
            .iter()
            .map(|st| match &st.cubin {
                Some(c) => c.rsplit('/').next().unwrap_or(c).to_string(),
                None => "(unpinned module)".to_string(),
            })
            .collect();
        names.dedup();
        let n = names.join(" + ");
        if s.len() > 1 { format!("{} steps: {n}", s.len()) } else { n }
    };
    for (k, kind) in &changed {
        let detail = match *kind {
            "interface" => format!("{} → {} params", ma.kernels[k].params.len(), mb.kernels[k].params.len()),
            "impl" => format!("{}  →  {}", step_desc(&ma.kernels[k]), step_desc(&mb.kernels[k])),
            "added" => step_desc(&mb.kernels[k]),
            _ => step_desc(&ma.kernels[k]),
        };
        rows.push(row![Cell::bold(k), Cell::warn(*kind), detail]);
    }
    for name in ma.buffers.keys().chain(mb.buffers.keys()).collect::<BTreeSet<_>>() {
        let what = match (ma.buffers.get(name), mb.buffers.get(name)) {
            (Some(x), Some(y)) if json!(x) != json!(y) => "changed",
            (Some(_), None) => "removed",
            (None, Some(_)) => "added",
            _ => continue,
        };
        rows.push(row![format!("buffer {name}"), Cell::warn(what), ""]);
    }
    let mut segments: BTreeMap<String, Vec<Segment>> = BTreeMap::new();
    let mut frontier_warn = false;
    for (pname, pa) in &ma.programs {
        let Some(pb) = mb.programs.get(pname) else {
            rows.push(row![pname.clone(), Cell::warn("only in A"), "skipped"]);
            continue;
        };
        let segs = align(pa, pb, &changed);
        let cuts: Vec<&Segment> = segs.iter().filter(|s| s.kind == Kind::Changed).collect();
        if cuts.is_empty() {
            rows.push(row![pname.clone(), Cell::dim("identical"), format!("{} dispatches", pa.dispatches.len())]);
            continue;
        }
        // Group cuts by shape: (A kernels, B kernels, reads, writes).
        let mut groups: BTreeMap<(String, String, String, String), usize> = BTreeMap::new();
        for s in &cuts {
            let ka = pa.dispatches[s.a.0..s.a.1].iter().map(|d| d.kernel.as_str()).collect::<Vec<_>>().join("+");
            let kb = pb.dispatches[s.b.0..s.b.1].iter().map(|d| d.kernel.as_str()).collect::<Vec<_>>().join("+");
            let ia = frontier_inputs(&ma, pname, s.a.0, s.a.1);
            let ib = frontier_inputs(&mb, pname, s.b.0, s.b.1);
            let wa = access(&ma, pname, s.a.0, s.a.1).writes;
            let wb = access(&mb, pname, s.b.0, s.b.1).writes;
            if ia != ib || wa != wb {
                frontier_warn = true;
            }
            let reads = ia.iter().cloned().collect::<Vec<_>>().join(", ");
            let writes = wa.iter().cloned().collect::<Vec<_>>().join(", ");
            *groups.entry((ka, kb, reads, writes)).or_default() += 1;
        }
        let shared = pa.dispatches.len() - cuts.iter().map(|s| s.a.1 - s.a.0).sum::<usize>();
        for (gi, ((ka, kb, reads, writes), n)) in groups.iter().enumerate() {
            let what = if ka == kb { ka.clone() } else if ka.is_empty() { format!("∅ → {kb}") } else if kb.is_empty() { format!("{ka} → ∅") } else { format!("{ka} → {kb}") };
            rows.push(row![
                if gi == 0 { pname.clone() } else { String::new() },
                Cell::bold(format!("{n} cut{}", if *n == 1 { "" } else { "s" })),
                format!("{what}   {reads} → {writes}"),
                Cell::dim(if gi == 0 { format!("{} dispatches, {shared} shared", pa.dispatches.len()) } else { String::new() }),
            ]);
        }
        segments.insert(pname.clone(), segs);
    }
    for pname in mb.programs.keys() {
        if !ma.programs.contains_key(pname) {
            rows.push(row![pname.clone(), Cell::warn("only in B"), "skipped"]);
        }
    }
    sec.table(rows);
    if frontier_warn {
        sec.note(Cell::warn("⚠ some cuts read or write different buffers on the two sides — not a cut-internal replacement"));
    }
    r.section(&sec);
    report["diff"] = json!({
        "kernels": changed,
        "programs": segments.iter().map(|(p, segs)| (p.clone(), json!(segs.iter()
            .filter(|s| s.kind == Kind::Changed)
            .map(|s| json!({"a": [s.a.0, s.a.1], "b": [s.b.0, s.b.1]})).collect::<Vec<_>>())))
            .collect::<serde_json::Map<_, _>>(),
    });
    if o.diff_only {
        return Ok(());
    }
    if segments.is_empty() {
        println!("\nnothing to attest: the programs are identical.");
        return Ok(());
    }

    // ---- load A + B
    let blobs = o
        .weights
        .iter()
        .map(|p| std::fs::read(p).with_context(|| format!("reading {}", p.display())))
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&[u8]> = blobs.iter().map(Vec::as_slice).collect();
    let t = Instant::now();
    let mut s = Sides { a: load_side(&ja, &o, &refs)?, b: load_side(&jb, &o, &refs)? };
    drop(blobs);
    let load_t = t.elapsed();
    let tokenizer = tokenizers::Tokenizer::from_file(&o.tokenizer)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let ids = tokens_of(&tokenizer, &o.prompt)?;
    let e1 = env(1);
    let cuts_of = |p: &str| -> Vec<Segment> {
        segments.get(p).map(|sg| sg.iter().filter(|s| s.kind == Kind::Changed).cloned().collect()).unwrap_or_default()
    };

    // ---- 2. tap: A and B in lockstep on the prompt, snapshot at every cut
    let mut sec = Section::new("TAP", &format!("A and B lockstep · prompt {} tokens · snapshot + compare at every cut · runtimes loaded in {:.1?}", ids.len(), load_t));
    let mut snaps: Vec<Snap> = Vec::new();
    // program -> buffer -> [(cut label, cmp)]
    let mut local_res: BTreeMap<String, BTreeMap<String, Vec<(String, Cmp)>>> = BTreeMap::new();
    let mut local_states: BTreeMap<String, BTreeMap<String, Vec<(String, usize)>>> = BTreeMap::new();
    let mut one_sided_all: BTreeSet<String> = BTreeSet::new();
    let mut local_json = Vec::new();
    let mut n_chunks = 0usize;
    s.a.reset();
    s.b.reset();
    // Run one program in lockstep; at each cut snapshot A's frontier inputs
    // (from A, before the cut), A's outputs after, and compare B's outputs.
    let mut lockstep = |s: &mut Sides, pname: &str, e: &BTreeMap<String, u64>, label_prefix: &str, keep: bool| -> Result<()> {
        let Some(segs) = segments.get(pname) else {
            s.a.rt.run(pname, e)?;
            s.b.rt.run(pname, e)?;
            return Ok(());
        };
        for seg in segs {
            if seg.kind != Kind::Changed {
                s.a.rt.run_range(pname, e, seg.a.0, seg.a.1)?;
                s.b.rt.run_range(pname, e, seg.b.0, seg.b.1)?;
                continue;
            }
            let label = format!("{label_prefix}{}", seg_label(seg));
            let input_names: BTreeSet<String> = frontier_inputs(&ma, pname, seg.a.0, seg.a.1)
                .union(&frontier_inputs(&mb, pname, seg.b.0, seg.b.1))
                .cloned()
                .collect();
            let mut inputs = Vec::new();
            for n in &input_names {
                if ma.buffers[n].class != BufferClass::Weight {
                    inputs.push((n.clone(), s.a.rt.read_buffer_prefix(n, live_bytes(&ma, n, e))?));
                }
            }
            s.a.rt.run_range(pname, e, seg.a.0, seg.a.1)?;
            s.b.rt.run_range(pname, e, seg.b.0, seg.b.1)?;
            let (aa, ab) = (access(&ma, pname, seg.a.0, seg.a.1), access(&mb, pname, seg.b.0, seg.b.1));
            let (bufs, states, one_sided) = compare_written(&s.a, &s.b, &aa, &ab, e, true)?;
            for (n, cmp) in &bufs {
                local_res.entry(pname.into()).or_default().entry(n.clone()).or_default().push((label.clone(), cmp.clone()));
            }
            for (st, n) in &states {
                local_states.entry(pname.into()).or_default().entry(st.clone()).or_default().push((label.clone(), *n));
            }
            one_sided_all.extend(one_sided.iter().cloned());
            local_json.push(json!({"program": pname, "cut": label,
                "buffers": bufs.iter().map(|(n, c)| (n.clone(), c.to_json())).collect::<serde_json::Map<_, _>>(),
                "states": states, "one_sided": one_sided}));
            if keep {
                let mut ref_out = BTreeMap::new();
                for n in aa.writes.intersection(&ab.writes) {
                    ref_out.insert(n.clone(), s.a.rt.read_buffer_prefix(n, live_bytes(&ma, n, e))?);
                }
                let mut ref_states = BTreeMap::new();
                for st in aa.state_writes.union(&ab.state_writes) {
                    ref_states.insert(st.clone(), s.a.rt.read_state(st)?);
                }
                snaps.push(Snap { program: pname.into(), seg: seg.clone(), env: e.clone(), inputs, ref_out, ref_states });
            }
        }
        Ok(())
    };
    let pre = &ids[..ids.len() - 1];
    let chunk = o.chunk.min(ma.symbols[TOKENS].max).max(1) as usize;
    let mut i = 0;
    while i < pre.len() {
        let c = (pre.len() - i).min(chunk);
        let e = s.a.stage_prefill(&pre[i..i + c])?;
        s.b.stage_prefill(&pre[i..i + c])?;
        lockstep(&mut s, "prefill", &e, &format!("chunk {n_chunks} "), n_chunks == 0)?;
        s.a.advance(c as u64);
        s.b.advance(c as u64);
        i += c;
        n_chunks += 1;
    }
    s.a.stage_decode(*ids.last().unwrap())?;
    s.b.stage_decode(*ids.last().unwrap())?;
    lockstep(&mut s, "decode", &e1, "", true)?;
    let mut local_identical = true;
    let mut local_bit = true;
    let mut rows = Vec::new();
    let undriven: Vec<String> = segments.keys().filter(|p| !DRIVEN.contains(&p.as_str())).cloned().collect();
    for pname in ma.programs.keys().map(String::as_str) {
        let n_cuts = cuts_of(pname).len();
        if n_cuts == 0 || undriven.iter().any(|u| u == pname) {
            continue;
        }
        let count = if pname == "prefill" && n_chunks > 1 { format!("{n_cuts} cuts × {n_chunks} chunks") } else { format!("{n_cuts} cuts") };
        let mut first = true;
        if let Some(bufs) = local_res.get(pname) {
            for (buf, res) in bufs {
                local_identical &= res.iter().all(|(_, c)| c.value_identical());
                local_bit &= res.iter().all(|(_, c)| c.identical());
                rows.push(row![if first { pname.to_string() } else { String::new() }, count.clone(), buf.clone(), summarize(res)]);
                first = false;
            }
        }
        if let Some(sts) = local_states.get(pname) {
            for (st, res) in sts {
                let bad: Vec<&(String, usize)> = res.iter().filter(|(_, n)| *n > 0).collect();
                local_identical &= bad.is_empty();
                local_bit &= bad.is_empty();
                let txt = if bad.is_empty() { Cell::good("bit-identical") } else { Cell::bad(format!("{}/{} cuts differ · {} bytes at {}", bad.len(), res.len(), bad[0].1, bad[0].0)) };
                rows.push(row![if first { pname.to_string() } else { String::new() }, count.clone(), format!("state {st}"), txt]);
                first = false;
            }
        }
    }
    let mut out_cmp = BTreeMap::new();
    for (name, b) in &ma.buffers {
        if b.class == BufferClass::Output && mb.buffers.contains_key(name) {
            let bytes = live_bytes(&ma, name, &e1);
            let c = compare(b.dtype, &s.a.rt.read_buffer_prefix(name, bytes)?, &s.b.rt.read_buffer_prefix(name, bytes)?);
            rows.push(row!["output", "", name.clone(), cell(&c)]);
            out_cmp.insert(name.clone(), c.to_json());
        }
    }
    sec.table(rows);
    let snap_bytes: usize = snaps.iter().map(|sn| sn.inputs.iter().map(|(_, b)| b.len()).sum::<usize>() + sn.ref_out.values().map(|b| b.len()).sum::<usize>()).sum();
    sec.note(Cell::dim(format!("{} cuts snapshotted ({} of frontier inputs + reference outputs)", snaps.len(), kb(snap_bytes))));
    if !one_sided_all.is_empty() {
        sec.note(Cell::dim(format!("written on one side only (implementation-internal, not compared): {:?}", one_sided_all)));
    }
    for p in &undriven {
        sec.note(Cell::bad(format!("{p}: changed but not tapped — the workload driver only stages {DRIVEN:?}")));
    }
    r.section(&sec);
    report["local"] = json!({"value_identical": local_identical, "bit_identical": local_bit, "cuts": local_json, "outputs": out_cmp});

    // Replay one snapshot's cut on a side from its inputs; returns the
    // written buffers (live prefix) and states.
    let replay = |c: &mut Caller, m: &Manifest, sn: &Snap, side_b: bool, inputs: &[(String, Vec<u8>)]| -> Result<(BTreeMap<String, Vec<u8>>, BTreeMap<String, Vec<u8>>)> {
        for (n, bytes) in inputs {
            c.rt.write_buffer(n, bytes)?;
        }
        let r = if side_b { sn.seg.b } else { sn.seg.a };
        c.rt.run_range(&sn.program, &sn.env, r.0, r.1)?;
        let mut out = BTreeMap::new();
        for n in sn.ref_out.keys() {
            out.insert(n.clone(), c.rt.read_buffer_prefix(n, live_bytes(m, n, &sn.env))?);
        }
        let mut st = BTreeMap::new();
        for n in sn.ref_states.keys() {
            st.insert(n.clone(), c.rt.read_state(n)?);
        }
        Ok((out, st))
    };
    let cmp_out = |m: &Manifest, sn: &Snap, out: &BTreeMap<String, Vec<u8>>| -> BTreeMap<String, Cmp> {
        out.iter().map(|(n, b)| (n.clone(), compare(m.buffers[n].dtype, &sn.ref_out[n], b))).collect()
    };

    // ---- 3. noise floor: A's cut re-run from its own snapshot
    let mut noise_res: BTreeMap<String, BTreeMap<String, Vec<(String, Cmp)>>> = BTreeMap::new();
    let mut noise_clean = true;
    if !o.no_noise {
        let mut sec = Section::new("NOISE FLOOR", &format!("A re-run from each snapshot vs A's own output · {} cuts", snaps.len()));
        let mut state_noise = Vec::new();
        for sn in &snaps {
            let (out, st) = replay(&mut s.a, &ma, sn, false, &sn.inputs)?;
            for (n, c) in cmp_out(&ma, sn, &out) {
                noise_clean &= c.identical();
                noise_res.entry(sn.program.clone()).or_default().entry(n).or_default().push((seg_label(&sn.seg), c));
            }
            for (name, bytes) in st {
                let d = bytes.iter().zip(&sn.ref_states[&name]).filter(|(p, q)| p != q).count();
                if d > 0 {
                    noise_clean = false;
                    state_noise.push(format!("state {name}: {d} bytes at {} {}", sn.program, seg_label(&sn.seg)));
                }
            }
        }
        let mut rows = Vec::new();
        if noise_clean {
            rows.push(row![Cell::good("clean"), "every cut reproduces its output bit for bit"]);
        } else {
            for (pname, bufs) in &noise_res {
                for (buf, res) in bufs {
                    rows.push(row![pname.clone(), format!("{} cuts", res.len()), buf.clone(), summarize(res)]);
                }
            }
            for t in &state_noise {
                rows.push(row![Cell::warn("⚠"), t.clone()]);
            }
            rows.push(row!["", Cell::warn("A is not deterministic at these cuts; B is judged against this band")]);
        }
        sec.table(rows);
        r.section(&sec);
    }
    report["noise_floor"] = json!({"clean": noise_clean,
        "cuts": noise_res.iter().map(|(p, bufs)| (p.clone(), json!(bufs.iter().map(|(b, res)| (b.clone(), json!(res.iter().map(|(l, c)| json!({"cut": l, "cmp": c.to_json()})).collect::<Vec<_>>()))).collect::<serde_json::Map<_, _>>()))).collect::<serde_json::Map<_, _>>()});

    // ---- 4. fuzz the cuts from their snapshots
    let mut fuzz_ok = true;
    let mut fuzz_identical = true; // value-identical under every distribution
    let mut fuzz_bit = true;
    if o.fuzz > 0 {
        let mut sec = Section::new("FUZZ", &format!("{} rounds per cut · {} cuts · frontier inputs synthesized, integers from their domains", o.fuzz, snaps.len()));
        let mut rng = Rng(o.seed);
        let progs: Vec<String> = segments.keys().cloned().collect();
        // round -> program -> worst cell
        let mut grid: BTreeMap<usize, BTreeMap<String, Cell>> = BTreeMap::new();
        let mut unfuzzed_all: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut rounds_json = Vec::new();
        for round in 0..o.fuzz {
            let dist = round % DISTS.len();
            let mut worst: BTreeMap<String, Cmp> = BTreeMap::new();
            let mut violations: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for sn in &snaps {
                let mut inputs = Vec::new();
                for (name, _) in &sn.inputs {
                    let decl = &ma.buffers[name];
                    let n = live_elems(&ma, name, &sn.env);
                    let vals = if is_float(decl.dtype) {
                        gen_float(&mut rng, dist, n, decl.dtype)
                    } else {
                        match &decl.domain {
                            Some(d) => {
                                let r = d.resolve(&ma, &sn.env, o.capacity)?;
                                let (lo, hi) = (r.lo.unwrap_or(0.0), r.hi.unwrap_or(r.lo.unwrap_or(0.0) + 1024.0));
                                gen_int(&mut rng, n, lo, hi, r.monotone)
                            }
                            None => {
                                unfuzzed_all.entry(sn.program.clone()).or_default().insert(name.clone());
                                continue; // keeps the tapped value
                            }
                        }
                    };
                    inputs.push((name.clone(), values::from_f64(decl.dtype, &vals)));
                }
                let (out_a, _) = match replay(&mut s.a, &ma, sn, false, &inputs) {
                    Ok(x) => x,
                    Err(err) => bail!("A crashed under fuzz ({}) at {} cut {}: {err}", DISTS[dist], sn.program, seg_label(&sn.seg)),
                };
                let (out_b, _) = match replay(&mut s.b, &mb, sn, true, &inputs) {
                    Ok(x) => x,
                    Err(err) => {
                        println!("  ✗ B crashed under `{}` at {} cut {}: {err}", DISTS[dist], sn.program, seg_label(&sn.seg));
                        bail!("B crashed under fuzz; the CUDA context is unusable past this point");
                    }
                };
                for (name, b) in &out_b {
                    let c = compare(ma.buffers[name].dtype, &out_a[name], b);
                    let w = worst.entry(sn.program.clone()).or_default();
                    if (c.n_diff - c.signed_zero, c.max_ulp, c.signed_zero) > (w.n_diff - w.signed_zero, w.max_ulp, w.signed_zero) {
                        *w = c;
                    }
                    // Post-condition: produced values must lie in the
                    // buffer's declared domain (A is checked too — a
                    // violation there is the reference misbehaving).
                    if let Some(d) = &mb.buffers[name].domain {
                        let r = d.resolve(&mb, &sn.env, o.capacity)?;
                        for (side, bytes) in [("A", &out_a[name]), ("B", b)] {
                            let v = values::to_f64(mb.buffers[name].dtype, bytes);
                            if let Some(i) = v.iter().position(|x| !r.contains(*x)) {
                                violations.entry(sn.program.clone()).or_default().push(format!("{side} {name}[{i}] = {} outside domain", v[i]));
                            }
                        }
                    }
                }
            }
            for p in &progs {
                let w = worst.get(p).cloned().unwrap_or_default();
                fuzz_identical &= w.value_identical();
                fuzz_bit &= w.identical();
                let txt = match violations.get(p) {
                    Some(v) => {
                        fuzz_ok = false;
                        Cell::bad(format!("✗ {}", v.join("; ")))
                    }
                    None => cell(&w),
                };
                grid.entry(round).or_default().insert(p.clone(), txt);
                rounds_json.push(json!({"dist": DISTS[dist], "program": p, "worst": w.to_json(), "violations": violations.get(p)}));
            }
        }
        let mut rows = vec![std::iter::once(Cell::default()).chain(progs.iter().map(|p| Cell::from(p.as_str()))).collect::<Vec<_>>()];
        for (round, cells) in &grid {
            let name = if o.fuzz > DISTS.len() { format!("{} #{}", DISTS[round % DISTS.len()], round / DISTS.len()) } else { DISTS[round % DISTS.len()].to_string() };
            rows.push(std::iter::once(Cell::from(name)).chain(progs.iter().map(|p| cells.get(p).cloned().unwrap_or_default())).collect());
        }
        sec.table_h(rows);
        for (p, u) in &unfuzzed_all {
            sec.note(Cell::warn(format!("{p}: not fuzzed (integer buffers without a domain, tapped values kept): {}", u.iter().cloned().collect::<Vec<_>>().join(", "))));
        }
        r.section(&sec);
        report["fuzz"] = json!({"ok": fuzz_ok, "value_identical": fuzz_identical, "bit_identical": fuzz_bit, "rounds": rounds_json,
            "unfuzzed": unfuzzed_all});
    }

    // ---- 5. perf: eager step attribution, graph step, sweep, roofline
    if !o.no_perf {
        let sweep_iters = o.iters.min(10);
        let mut sec = Section::new("PERF", &format!("eager per-dispatch timing, min of {} (sweep: {sweep_iters}){}", o.iters, if o.no_graph_step { "" } else { " · graph step median of 100" }));
        let mut rows = vec![row!["", "", "A", "B measured", "B derived", "Δ measured (B − A)"]];
        let mut per_kernel: BTreeMap<String, [(usize, f32, usize); 2]> = BTreeMap::new(); // kernel -> per side (bytes, ms, count)
        let mut state_traffic = false;
        let mut perf_json = serde_json::Map::new();
        // Time a whole program on both sides at `e`; returns (step, Σ cuts)
        // per side and feeds the roofline accumulator.
        let mut step = |s: &mut Sides, pname: &str, e: &BTreeMap<String, u64>, iters: usize, roof: bool| -> Result<[(f32, f32); 2]> {
            let ta = s.a.rt.time_range(pname, e, 0, s.a.rt.dispatch_count(pname)?, iters)?;
            let tb = s.b.rt.time_range(pname, e, 0, s.b.rt.dispatch_count(pname)?, iters)?;
            let mut out = [(ta.iter().sum::<f32>(), 0f32), (tb.iter().sum::<f32>(), 0f32)];
            for sg in cuts_of(pname) {
                for (si, m, r, t) in [(0usize, &ma, sg.a, &ta), (1, &mb, sg.b, &tb)] {
                    for i in r.0..r.1 {
                        out[si].1 += t[i];
                        if !roof {
                            continue;
                        }
                        let acc = access(m, pname, i, i + 1);
                        let bytes: usize = acc.reads.iter().map(|n| live_bytes(m, n, e)).sum::<usize>()
                            + acc.writes.iter().map(|n| live_bytes(m, n, e)).sum::<usize>();
                        state_traffic |= !(acc.state_reads.is_empty() && acc.state_writes.is_empty());
                        let ent = per_kernel.entry(format!("{pname} · {}", m.programs[pname].dispatches[i].kernel)).or_default();
                        ent[si].0 += bytes;
                        ent[si].1 += t[i];
                        ent[si].2 += 1;
                    }
                }
            }
            Ok(out)
        };
        // derived = A's step with A's cuts swapped for B's cuts (both timed
        // eager); the gap to the measurement is launch-gap / L2 interaction.
        let derived = |a_step: f32, st: [(f32, f32); 2]| a_step - st[0].1 + st[1].1;
        let push_step = |rows: &mut Vec<Vec<Cell>>, label: &str, n_cuts: usize, st: [(f32, f32); 2], graph: Option<(f32, f32)>| {
            let d = derived(st[0].0, st);
            rows.push(row![Cell::bold(label), "step, eager", us(st[0].0), us(st[1].0), Cell::bold(us(d)), delta(st[0].0, st[1].0),
                Cell::dim(format!("measured − derived {:+.1} µs", (st[1].0 - d) * 1e3))]);
            if let Some((ga, gb)) = graph {
                let d = derived(ga, st);
                rows.push(row!["", "step, graph (TPOT)", us(ga), us(gb), Cell::bold(us(d)), delta(ga, gb),
                    Cell::dim(format!("{:.0} → {:.0} tok/s", 1e3 / ga, 1e3 / gb))]);
            }
            rows.push(row!["", format!("Σ {n_cuts} cuts (the swap)"), us(st[0].1), us(st[1].1), "", delta(st[0].1, st[1].1)]);
        };
        let n_cuts = |p: &str| cuts_of(p).len();
        // decode at the tapped position
        if n_cuts("decode") > 0 && !undriven.iter().any(|u| u == "decode") {
            s.a.stage_decode(*ids.last().unwrap())?;
            s.b.stage_decode(*ids.last().unwrap())?;
            let st = step(&mut s, "decode", &e1, o.iters, true)?;
            let mut graph = None;
            if !o.no_graph_step {
                s.a.rt.capture("decode", &e1)?;
                s.b.rt.capture("decode", &e1)?;
                graph = Some((s.a.rt.time_captured("decode", &e1, 100)?, s.b.rt.time_captured("decode", &e1, 100)?));
            }
            push_step(&mut rows, &format!("decode  {TOKENS}=1"), n_cuts("decode"), st, graph);
            let mut j = json!({"tokens": 1, "eager_ms": {"a": st[0].0, "b": st[1].0, "b_derived": derived(st[0].0, st)}, "cut_ms": {"a": st[0].1, "b": st[1].1}});
            if let Some((ga, gb)) = graph {
                j["graph_ms"] = json!({"a": ga, "b": gb, "b_derived": derived(ga, st)});
            }
            perf_json.insert("decode".into(), j);
        }
        // prefill: the tapped chunk length plus a sweep over the symbol range
        if n_cuts("prefill") > 0 && !undriven.iter().any(|u| u == "prefill") {
            let max = ma.symbols[TOKENS].max;
            let tap_len = pre.len().min(chunk) as u64;
            let mut points: BTreeSet<u64> = [tap_len].into();
            if !o.no_sweep {
                points.extend([1u64, 16, 128, 512, 2048, 4096, max].into_iter().filter(|&t| t <= max));
            }
            let vocab = s.a.vocab();
            let mut rng = Rng(o.seed);
            let mut sweep = Vec::new();
            let mut sw = vec![row!["step A, eager"], row!["step B, measured"], row![Cell::bold("step B, derived")], row!["Δ measured"], row![Cell::dim("Σ cuts A")], row![Cell::dim("Σ cuts B")], row![Cell::dim("Δ")]];
            for &t in &points {
                let tid: Vec<i64> = (0..t).map(|_| rng.below(vocab) as i64).collect();
                s.a.reset();
                s.b.reset();
                let e = s.a.stage_prefill(&tid)?;
                s.b.stage_prefill(&tid)?;
                let st = step(&mut s, "prefill", &e, if t == tap_len { o.iters } else { sweep_iters }, t == tap_len)?;
                if t == tap_len {
                    push_step(&mut rows, &format!("prefill  {TOKENS}={t}"), n_cuts("prefill"), st, None);
                }
                sw[0].push(Cell::from(us(st[0].0)));
                sw[1].push(Cell::from(us(st[1].0)));
                sw[2].push(Cell::bold(us(derived(st[0].0, st))));
                sw[3].push(pct_cell(st[0].0, st[1].0));
                sw[4].push(Cell::dim(us(st[0].1)));
                sw[5].push(Cell::dim(us(st[1].1)));
                sw[6].push(pct_cell(st[0].1, st[1].1));
                sweep.push(json!({"tokens": t, "eager_ms": {"a": st[0].0, "b": st[1].0, "b_derived": derived(st[0].0, st)}, "cut_ms": {"a": st[0].1, "b": st[1].1}}));
            }
            sec.table_h(std::mem::take(&mut rows));
            if points.len() > 1 {
                let mut hdr = row![format!("prefill · {TOKENS} =")];
                hdr.extend(points.iter().map(|t| Cell::from(t.to_string())));
                sw.insert(0, hdr);
                sec.table_h(sw);
            }
            perf_json.insert("prefill".into(), json!(sweep));
        }
        if !rows.is_empty() {
            sec.table_h(rows);
        }
        let mut rows = vec![row!["roofline", "moved / dispatch", "A", "B"]];
        let mut roof = Vec::new();
        for (k, sides) in &per_kernel {
            let fmt = |(bytes, ms, n): (usize, f32, usize)| -> String {
                if n == 0 {
                    return "—".into();
                }
                let gbs = bytes as f64 / 1e9 / (ms as f64 / 1e3);
                format!("{} · {:.1} GB/s · {:.2}% of peak", us(ms / n as f32), gbs, gbs / o.peak_bw * 100.0)
            };
            let n = sides[0].2.max(sides[1].2);
            let per = sides.iter().find(|s| s.2 > 0).map_or(0, |s| s.0 / s.2);
            rows.push(row![Cell::bold(format!("{k} ×{n}")), format!("{}{}", kb(per), if state_traffic { " + opaque state" } else { "" }), fmt(sides[0]), fmt(sides[1])]);
            roof.push(json!({"kernel": k, "bytes_per_dispatch": per, "a": {"ms": sides[0].1, "n": sides[0].2}, "b": {"ms": sides[1].1, "n": sides[1].2}}));
        }
        sec.table_h(rows);
        perf_json.insert("roofline".into(), json!(roof));
        perf_json.insert("peak_bw_gbs".into(), json!(o.peak_bw));
        r.section(&sec);
        report["perf"] = Value::Object(perf_json);
    }

    // Is every local difference inside A's own noise band?
    let within_noise = !local_identical && !noise_clean && local_res.iter().all(|(p, bufs)| {
        bufs.iter().all(|(b, res)| {
            let worst_local = res.iter().filter_map(|(_, c)| if c.value_identical() { None } else { c.max_ulp.or(Some(u64::MAX)) }).max();
            let worst_noise = noise_res.get(p).and_then(|nb| nb.get(b)).and_then(|nr| nr.iter().filter_map(|(_, c)| if c.value_identical() { None } else { c.max_ulp.or(Some(u64::MAX)) }).max());
            match (worst_local, worst_noise) {
                (None, _) => true,
                (Some(l), Some(n)) => l <= n,
                (Some(_), None) => false,
            }
        })
    });

    // ---- verdict
    let (code, verdict) = if !fuzz_ok {
        (1, "B violates a declared domain (or crashed) under fuzz")
    } else if !undriven.is_empty() {
        (2, "a changed program was not tapped — the workload driver can't stage it")
    } else if local_bit && fuzz_bit {
        (0, "bit-identical at every cut, real and synthesized inputs")
    } else if local_identical && fuzz_identical {
        (0, "value-identical at every cut (only signed zeros differ)")
    } else if within_noise && fuzz_identical {
        (0, "differences at every cut lie within A's own noise floor")
    } else {
        (2, "cuts differ beyond bit/value identity — no end-to-end oracle in this harness")
    };
    r.verdict(code, verdict, t_start.elapsed(), o.out.as_deref());
    report["verdict"] = json!({"code": code, "pass": code == 0, "summary": verdict,
        "noise_clean": noise_clean, "within_noise": within_noise, "local_value_identical": local_identical, "local_bit_identical": local_bit,
        "fuzz_ok": fuzz_ok, "fuzz_value_identical": fuzz_identical, "fuzz_bit_identical": fuzz_bit});
    if let Some(p) = &o.out {
        std::fs::write(p, serde_json::to_string_pretty(&report)?)?;
    }
    if code != 0 {
        std::process::exit(code);
    }
    Ok(())
}
