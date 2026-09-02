//! The kern scheduler: one `Runtime`, many sequences.
//!
//! Implements the pegainfer frontend's [`Scheduler`] contract — `submit`,
//! `step`, `metrics` — over a kern manifest with the qwen3-4b caller
//! contract: input buffers `token_ids` / `positions` / `slot_mapping` /
//! `seq_lens` / `cu_seqlens_q` / `block_table`, programs `prefill`
//! (single sequence, chunked), `decode` (single sequence, the manifest's
//! bs=1 microprogram) and `decode_batch` (`seqs` sequences, one row each).
//! Two prefill contracts: state only (the last prompt token is the first
//! decode step's input) or, when `prefill` writes `next_token` (hybrid
//! GDN models, whose chunked kernels must see every prompt token), every
//! prompt token through prefill and the first generated token from it.
//! A manifest with per-sequence states (`bytes_per_seq`) also has line
//! tables (`[lines, seqs]` inputs indexing them); every call stages them
//! from the leases, column i = sequence i of the batch.
//!
//! Policy, deliberately simple:
//! - prefill first: each step admits waiting requests (up to a token
//!   budget) and prefills them one at a time, then runs one decode step
//!   over every running sequence;
//! - a request leases every KV page its worst case (`prompt + max_tokens`)
//!   needs at admission (`Runtime::lease`), so decode never runs out of
//!   pages and nothing is ever preempted; the lease drops with the sequence;
//! - decode batches are padded up to a bucket size and each bucket's
//!   program is CUDA-graph-captured once; padding rows write into a page
//!   the scheduler leases for them and nobody reads;
//! - greedy only: the manifest's `argmax` is the sampler. Non-greedy
//!   sampling params are logged once and served greedily;
//! - `--spec` (a manifest with `draft` / `verify` / `draft_precompute` /
//!   `decode_spec`): every step is one speculative round over the batch —
//!   `draft` proposes `n` tokens per sequence, `verify` runs the target
//!   over `[anchor, drafts]` per sequence, `draft_precompute` projects the
//!   target's taps into the draft's context KV for every row (rejected
//!   rows land past the sequence's position and are overwritten next
//!   round, exactly like the target KV's free rollback), and the host
//!   accepts each sequence's longest matching prefix. The lease grows by
//!   `n` tokens so the last round's rejected rows have slots. A manifest
//!   with a `round` program runs the whole round as one graph: draft,
//!   verify's ids spliced on device, verify, precompute, accept on device,
//!   `advance` from the device's `num_accepted` — one launch and one sync
//!   per round instead of four; the host reads `draft_tokens` /
//!   `verify_tokens` and accepts the same prefix. Whether a round beats a
//!   plain step at a given batch size is the operator's call, not the
//!   scheduler's: the flag picks the mode for the process.

use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use kern_manifest::types::{Arg, Dim, Manifest};
use kern_run::{first_i64, i64_from_le, le_bytes_i32, le_bytes_i64};
use kern_runtime::{Denied, Error, Lease, Runtime};
use pegainfer_frontend::engine::{
    FinishReason, QueuedRequest, RejectReason, RequestId, RequestLedger, Scheduler,
    SchedulerMetrics, SpecDecodeCounters, MAX_SPEC_TOKENS,
};
use tracing::{debug, info, warn};

use crate::logline;

/// `Runtime` holds raw CUDA handles (graph execs, functions) and is used
/// from the scheduler thread only; every entry point rebinds the context
/// to the calling thread, so moving it there once is sound.
struct Rt(Runtime);
unsafe impl Send for Rt {}

/// Decode batch buckets; a batch is padded up to the smallest one that
/// fits and each bucket owns one captured graph.
const BUCKETS: [usize; 13] = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256];

pub struct Policy {
    /// Prefill chunk (tokens per `prefill` call), clamped to the manifest's
    /// `tokens` bound.
    pub chunk: usize,
    /// Prompt tokens one step may prefill before it runs decode (at least
    /// one request is always admitted when one fits).
    pub prefill_budget: usize,
    /// Launch every call eagerly instead of capturing graphs.
    pub eager: bool,
    /// Cap on concurrently running sequences (≤ the manifest's `seqs` bound).
    pub max_seqs: usize,
    /// Token ids that end a request unless it asked `ignore_eos`.
    pub stop_tokens: Vec<u32>,
    /// Speculative rounds instead of decode steps (needs the spec programs).
    pub spec: bool,
}

/// A line table over a per-sequence state, shaped `[lines, seqs]` or
/// `[lines, seqs, w]`: `rows` lines per sequence, `width` entries per
/// (line, sequence) cell.
struct LineTable {
    name: String,
    rows: usize,
    width: usize,
}

impl LineTable {
    /// `name` is a line table over a per-sequence state; batched decode
    /// needs its `seqs` dimension (the runtime also allows `[lines]`).
    fn check(m: &Manifest, name: &str) -> Result<LineTable> {
        let (rows, width) = match shape(m, name)? {
            [Dim::Const(rows), Dim::Var(v)] if v == "seqs" => (*rows, 1),
            [Dim::Const(rows), Dim::Var(v), Dim::Const(w)] if v == "seqs" => (*rows, *w),
            s => bail!("line table `{name}` shaped {s:?}, expected [lines, seqs] or [lines, seqs, w]"),
        };
        Ok(LineTable { name: name.to_string(), rows: rows as usize, width: width as usize })
    }
}

/// DSpark's mask token when the manifest declares no `spec` block.
const DSPARK_MASK_TOKEN: i64 = 151669;

/// The manifest's speculative contract, one row-group per sequence.
struct SpecPlan {
    /// Tokens `draft` proposes per sequence (`draft_tokens` is `[seqs, n]`).
    n_drafts: usize,
    /// Rows per sequence in `draft`: `[anchor, mask, ...]`.
    draft_rows: usize,
    /// Rows per sequence in `verify`: `[anchor, drafts...]` = n + 1.
    verify_rows: usize,
    mask_token: i64,
    /// The target resumes a recurrent state from `num_accepted_tokens`
    /// (one per sequence) and commits the accepted rows with `advance`.
    advance: bool,
    /// The manifest has `round`: the whole round is one program (draft
    /// and verify rows per sequence coincide, so one staging serves both).
    fused: bool,
    counters: SpecDecodeCounters,
}

impl SpecPlan {
    /// The manifest's speculative contract; a round's rows per sequence
    /// must fit the `page`-token pad page.
    fn check(m: &Manifest, page: usize) -> Result<SpecPlan> {
        let fused = m.programs.contains_key("round");
        need_programs(m, if fused { &["decode_spec", "draft_precompute"] } else { &["decode_spec", "draft_precompute", "draft", "verify"] })?;
        let n_drafts = seqs_rows(m, "draft_tokens")?;
        let verify_rows = seqs_rows(m, "verify_tokens")?;
        if verify_rows != n_drafts + 1 {
            bail!("verify_tokens has {verify_rows} rows per sequence, expected {} (anchor + drafts)", n_drafts + 1);
        }
        if n_drafts > MAX_SPEC_TOKENS {
            bail!("{n_drafts} drafts per round, the frontend's metrics hold {MAX_SPEC_TOKENS}");
        }
        per_seq(m, "anchor_token")?;
        // The target resumes a recurrent state from `num_accepted_tokens`
        // and commits the accepted rows with `advance`: both or neither.
        let advance = m.programs.contains_key("advance");
        match (m.buffers.contains_key("num_accepted_tokens"), advance) {
            (true, true) => per_seq(m, "num_accepted_tokens")?,
            (true, false) => bail!("`num_accepted_tokens` resumes a recurrent state but no `advance` program commits the accepted rows"),
            (false, true) => bail!("program `advance` without a `num_accepted_tokens` input"),
            (false, false) => {}
        }
        let (draft_rows, mask_token) = match &m.spec {
            Some(s) => (s.block as usize, s.mask_token),
            None => (n_drafts, DSPARK_MASK_TOKEN),
        };
        if fused && draft_rows != verify_rows {
            bail!("`round` needs draft and verify rows per sequence to coincide, got {draft_rows} and {verify_rows}");
        }
        let plan = SpecPlan {
            n_drafts,
            draft_rows,
            verify_rows,
            mask_token,
            advance,
            fused,
            counters: SpecDecodeCounters { num_spec_tokens: n_drafts as u64, ..Default::default() },
        };
        if plan.rows() > page {
            bail!("{} rows per sequence per round exceed the {page}-token pad page", plan.rows());
        }
        Ok(plan)
    }

    /// Rows per sequence a round stages: the wider of draft's and verify's.
    fn rows(&self) -> usize {
        self.draft_rows.max(self.verify_rows)
    }
}

struct Seq {
    id: RequestId,
    /// Tokens already in the KV state.
    pos: usize,
    /// The token the next decode step feeds at `pos`.
    next: u32,
    generated: usize,
    max_tokens: usize,
    ignore_eos: bool,
    /// Its KV pages; returned to the runtime when the sequence drops.
    pages: Lease,
    prompt_len: usize,
    admitted: Instant,
}

impl Seq {
    /// Account `toks` as generated, in order: each is emitted until a stop
    /// token (itself not emitted, pegainfer convention; it still counts
    /// against `max_tokens` like vLLM's) or `max_tokens`. Finishes the
    /// request in the ledger when it is done, otherwise the last token is
    /// the next step's input. Returns how many tokens were emitted and
    /// whether the sequence finished.
    fn emit(&mut self, toks: &[u32], stop: &[u32], ledger: &mut RequestLedger) -> (u64, bool) {
        let mut out = Vec::with_capacity(toks.len());
        let mut reason = None;
        for &tok in toks {
            self.generated += 1;
            if !self.ignore_eos && stop.contains(&tok) {
                reason = Some(FinishReason::Stop);
                break;
            }
            out.push(tok);
            if self.generated >= self.max_tokens {
                reason = Some(FinishReason::Length);
                break;
            }
        }
        if !out.is_empty() {
            ledger.push_tokens(self.id, &out, &[]);
        }
        let done = match reason {
            Some(r) => {
                debug!(
                    request = %self.id,
                    reason = ?r,
                    prompt = self.prompt_len,
                    generated = self.generated,
                    elapsed_s = logline::secs(self.admitted.elapsed()),
                    "finished"
                );
                ledger.finish(self.id, r);
                true
            }
            None => {
                if let Some(&t) = toks.last() {
                    self.next = t;
                }
                false
            }
        };
        (out.len() as u64, done)
    }
}

pub struct KernScheduler {
    rt: Rt,
    /// One page (and sequence slot) no sequence owns: the padding rows of
    /// a decode batch write their junk here.
    pad: Lease,
    policy: Policy,
    spec: Option<SpecPlan>,
    /// `prefill` emits `next_token` itself (see the module doc).
    prefill_emits: bool,
    /// The manifest's line tables over per-sequence states.
    line_tables: Vec<LineTable>,
    /// The page tables, each shaped `[seqs, n]`: `block_table` and, under
    /// speculation, the draft's.
    page_tables: Vec<String>,
    /// The `seqs` bound: a line table's row width.
    seqs_max: usize,
    waiting: VecDeque<QueuedRequest>,
    running: Vec<Seq>,
    warned_sampling: bool,
    stats: Stats,
}

/// Rolling counters for the periodic log line, reset when it prints.
struct Stats {
    since: Instant,
    /// Decode steps (speculative rounds under `--spec`) and their time.
    steps: u64,
    step_ns: u128,
    /// Tokens emitted to the ledger.
    tokens: u64,
    prefill_tokens: u64,
    prefill_ns: u128,
    /// The speculative counters at the window's start, so the window's
    /// acceptance is reported rather than the process's.
    spec_at: (u64, u64, u64),
}

impl Stats {
    fn new(spec: &Option<SpecPlan>) -> Stats {
        let spec_at = spec.as_ref().map_or((0, 0, 0), |p| {
            let c = &p.counters;
            (c.num_drafts, c.num_draft_tokens, c.num_accepted_tokens)
        });
        Stats { since: Instant::now(), steps: 0, step_ns: 0, tokens: 0, prefill_tokens: 0, prefill_ns: 0, spec_at }
    }
}

/// Public facts the frontend wants at launch.
pub struct Facts {
    pub total_blocks: usize,
    pub block_size: usize,
    /// Longest request (prompt + completion) one sequence can hold.
    pub max_request_tokens: usize,
}

/// The manifest's fit to this caller contract, settled before the GPU is
/// touched: pure over the manifest and the runtime's table names, so a
/// synthetic manifest exercises every rejection.
struct Contract {
    seqs_max: usize,
    tokens_max: usize,
    prefill_emits: bool,
    line_tables: Vec<LineTable>,
    page_tables: Vec<String>,
    spec: Option<SpecPlan>,
}

impl Contract {
    /// `seq_tables` / `page_tables` are the runtime's, `page` its page in
    /// tokens; `spec` asks for the speculative contract too.
    fn check(m: &Manifest, seq_tables: &[&str], page_tables: &[&str], page: usize, spec: bool) -> Result<Contract> {
        // `decode_batch` drives plain batched decode; a speculative run
        // only ever runs `prefill` and the spec round's programs.
        need_programs(m, if spec { &["prefill", "decode"] } else { &["prefill", "decode", "decode_batch"] })?;
        need_buffers(m, &["token_ids", "positions", "slot_mapping", "seq_lens", "cu_seqlens_q"])?;
        if !page_tables.contains(&"block_table") {
            bail!("`block_table` is not a page table (an input indexing a paged state)");
        }
        let seqs_max = var_max(m, "seqs")?;
        let tokens_max = var_max(m, "tokens")?;
        let prefill_emits =
            m.programs["prefill"].iter().flat_map(|c| &c.args).any(|a| matches!(a, Arg::Buf { buf, .. } if buf == "next_token"));
        let line_tables = seq_tables.iter().map(|name| LineTable::check(m, name)).collect::<Result<Vec<_>>>()?;
        let page_tables =
            page_tables.iter().map(|name| seqs_rows(m, name).map(|_| name.to_string())).collect::<Result<Vec<_>>>()?;
        let spec = spec.then(|| SpecPlan::check(m, page).context("--spec: the manifest's speculative contract")).transpose()?;
        Ok(Contract { seqs_max, tokens_max, prefill_emits, line_tables, page_tables, spec })
    }

    /// Concurrent sequences: `want` within the `seqs` bound and what one
    /// call's rows — one per sequence, a round's row-group under
    /// speculation — fit in `tokens`.
    fn max_seqs(&self, want: usize) -> usize {
        let rows = self.spec.as_ref().map_or(1, SpecPlan::rows);
        want.clamp(1, self.seqs_max).min(self.tokens_max / rows).max(1)
    }
}

impl KernScheduler {
    /// Wrap a loaded runtime (weights bound): check the manifest against
    /// this caller contract, settle the policy within its bounds, lease
    /// the padding page.
    pub fn new(mut rt: Runtime, policy: Policy) -> Result<KernScheduler> {
        let seq_tables: Vec<&str> = rt.seq_tables().collect();
        let page_tables: Vec<&str> = rt.page_tables().collect();
        let c = Contract::check(&rt.manifest, &seq_tables, &page_tables, rt.page() as usize, policy.spec)?;
        let policy = Policy { max_seqs: c.max_seqs(policy.max_seqs), chunk: policy.chunk.clamp(1, c.tokens_max), ..policy };
        let pad = rt.lease(1).map_err(|e| anyhow::anyhow!("no page for the padding rows: {e}"))?;
        if rt.pages_used() == rt.pages_total() {
            bail!("capacity {} tokens holds one page; nothing left to serve from", rt.capacity());
        }
        let stats = Stats::new(&c.spec);
        let s = KernScheduler {
            rt: Rt(rt),
            pad,
            policy,
            spec: c.spec,
            prefill_emits: c.prefill_emits,
            line_tables: c.line_tables,
            page_tables: c.page_tables,
            seqs_max: c.seqs_max,
            waiting: VecDeque::new(),
            running: Vec::new(),
            warned_sampling: false,
            stats,
        };
        s.log_ready();
        Ok(s)
    }

    /// What the frontend advertises at launch.
    pub fn facts(&self) -> Facts {
        // What admit() leases is `prompt + max_tokens + headroom`;
        // advertise the request-shaped remainder so the frontend clamps
        // `max_tokens` to something admissible instead of the scheduler
        // bouncing it (the wire turns a scheduler reject into a 500).
        Facts {
            total_blocks: self.kv_total(),
            block_size: self.rt.0.page() as usize,
            max_request_tokens: self.rt.0.max_seq_tokens() - self.headroom(),
        }
    }

    /// Slots a lease holds past `prompt + max_tokens`: a speculative
    /// round writes `n_drafts` rows past the token it may still have to
    /// emit, and the last round's rejects need slots.
    fn headroom(&self) -> usize {
        self.spec.as_ref().map_or(0, |s| s.n_drafts)
    }

    fn log_ready(&self) {
        let (rt, policy, spec, facts) = (&self.rt.0, &self.policy, self.spec.as_ref(), self.facts());
        // Pages: what requests can lease (one more holds the padding rows).
        // Graphs are captured per bucket on first use unless `eager`.
        info!(
            pages = facts.total_blocks,
            page = facts.block_size,
            seq_slots = (rt.seq_slots() > 0).then(|| rt.seq_slots()),
            max_seqs = policy.max_seqs,
            max_request_tokens = facts.max_request_tokens,
            chunk = policy.chunk,
            buckets = ?BUCKETS.iter().filter(|&&b| b <= policy.max_seqs).collect::<Vec<_>>(),
            eager = policy.eager,
            prefill_emits = self.prefill_emits,
            drafts = spec.map(|s| s.n_drafts),
            draft_rows = spec.map(|s| s.draft_rows),
            verify_rows = spec.map(|s| s.verify_rows),
            fused_round = spec.map(|s| s.fused),
            "scheduler ready"
        );
    }

    fn bucket(&self, n: usize) -> usize {
        BUCKETS.iter().copied().find(|&b| b >= n).unwrap_or(n)
    }

    /// Admit waiting requests in order and prefill each (single sequence,
    /// chunked) up to the step's token budget.
    fn admit(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        let mut budget_used = 0usize;
        while let Some(q) = self.waiting.front() {
            let id = q.id;
            if ledger.is_aborted(id) {
                ledger.retire(id);
                self.waiting.pop_front();
                continue;
            }
            if self.running.len() >= self.policy.max_seqs {
                break;
            }
            let prompt = q.request.prompt_tokens.len();
            let max_tokens = q.request.max_tokens;
            let worst = prompt + max_tokens + self.headroom();
            if prompt == 0 {
                let limit = self.rt.0.max_seq_tokens();
                ledger.reject(id, RejectReason::ContextLength { prompt_tokens: prompt, max_tokens, limit });
                self.waiting.pop_front();
                continue;
            }
            if budget_used > 0 && budget_used + prompt - 1 > self.policy.prefill_budget {
                break; // enough prefill for this step; decode must run
            }
            let pages = match self.rt.0.lease(worst) {
                Ok(pages) => pages,
                Err(Error::Denied(Denied::Busy)) => break, // wait for pages / a slot
                Err(Error::Denied(Denied::ExceedsRow { limit })) => {
                    ledger.reject(id, RejectReason::ContextLength { prompt_tokens: prompt, max_tokens, limit });
                    self.waiting.pop_front();
                    continue;
                }
                Err(Error::Denied(Denied::ExceedsPool)) => {
                    ledger.reject(id, RejectReason::KvBudget { prompt_tokens: prompt, worst_case_tokens: worst });
                    self.waiting.pop_front();
                    continue;
                }
                Err(e) => return Err(e.into()),
            };
            let q = self.waiting.pop_front().unwrap();
            if !q.request.params.is_greedy() && !self.warned_sampling {
                warn!(
                    request = %id,
                    temperature = q.request.params.temperature,
                    top_p = q.request.params.top_p,
                    top_k = q.request.params.top_k,
                    "non-greedy sampling params; this engine samples greedily (argmax in the manifest); further requests are not warned about"
                );
                self.warned_sampling = true;
            }
            ledger.admit(id);
            let ids: Vec<i64> = q.request.prompt_tokens.iter().map(|&t| t as i64).collect();
            let t0 = Instant::now();
            // Every prompt token goes through `prefill` when it emits the
            // first generated token itself; otherwise everything but the
            // last, which is the first decode step's input.
            let n_pre = if self.prefill_emits { prompt } else { prompt - 1 };
            let first = self.prefill(&pages, &ids[..n_pre])?;
            self.stats.prefill_ns += t0.elapsed().as_nanos();
            self.stats.prefill_tokens += n_pre as u64;
            budget_used += n_pre;
            debug!(request = %id, prompt, max_tokens, pages = ?pages, prefill_ms = logline::ms(t0.elapsed()), "admitted");
            let mut seq = Seq {
                id,
                pos: n_pre,
                next: *q.request.prompt_tokens.last().unwrap(),
                generated: 0,
                max_tokens,
                ignore_eos: q.request.params.ignore_eos,
                pages,
                prompt_len: prompt,
                admitted: t0,
            };
            let first = match first {
                Some(tok) => Some(tok),
                None if self.spec.is_some() => {
                    // The last prompt token goes through `decode_spec` now
                    // (a round needs an anchor and its tap in the draft
                    // KV); its token is the first one emitted.
                    let tok = self.first_token(&seq)?;
                    seq.pos += 1;
                    Some(tok)
                }
                None => None,
            };
            if let Some(tok) = first {
                let (emitted, done) = seq.emit(&[tok], &self.policy.stop_tokens, ledger);
                self.stats.tokens += emitted;
                if done {
                    continue;
                }
            }
            self.running.push(seq);
        }
        Ok(())
    }

    /// Drop aborted sequences before a step so they neither pad nor compute.
    fn drop_aborted(&mut self, ledger: &mut RequestLedger) {
        self.running.retain(|s| {
            let live = !ledger.is_aborted(s.id);
            if !live {
                ledger.retire(s.id);
            }
            live
        });
    }

    /// Pages available to requests / held by them (the pad page excluded).
    fn kv_total(&self) -> usize {
        self.rt.0.pages_total() - self.pad.pages()
    }

    fn kv_used(&self) -> usize {
        self.rt.0.pages_used() - self.pad.pages()
    }

    /// Chunked single-sequence prefill of `ids` starting at position 0 of
    /// a sequence holding `pages`; the first generated token when the
    /// manifest's prefill emits it.
    fn prefill(&mut self, pages: &Lease, ids: &[i64]) -> Result<Option<u32>> {
        let chunk = self.policy.chunk;
        self.stage_lines(&[pages], &[])?;
        let mut pos = 0usize;
        while pos < ids.len() {
            let c = (ids.len() - pos).min(chunk);
            let env = env(c, 1);
            let positions: Vec<i64> = (pos..pos + c).map(|p| p as i64).collect();
            let slots = pages.slots(pos..pos + c);
            let rt = &mut self.rt.0;
            rt.write_input_at("token_ids", &le_bytes_i64(&ids[pos..pos + c]), &env)?;
            rt.write_input_at("positions", &le_bytes_i64(&positions), &env)?;
            rt.write_input_at("slot_mapping", &le_bytes_i64(&slots), &env)?;
            rt.write_input_at("seq_lens", &le_bytes_i32(&[(pos + c) as i32]), &env)?;
            rt.write_input_at("cu_seqlens_q", &le_bytes_i32(&[0, c as i32]), &env)?;
            stage_tables(rt, &self.page_tables, &[pages], &self.pad, 1, &env)?;
            let eager = self.policy.eager || c != chunk;
            run_program(rt, "prefill", &env, eager)?;
            if self.spec.is_some() {
                // The chunk's taps (`fc_out`) into the draft's context KV;
                // positions/slot_mapping are still the chunk's.
                run_program(rt, "draft_precompute", &env, eager)?;
            }
            pos += c;
        }
        if !self.prefill_emits || ids.is_empty() {
            return Ok(None);
        }
        Ok(Some(first_i64(&self.rt.0.read_output("next_token")?) as u32))
    }

    /// Stage every line table: column i names sequence i's lines, the
    /// columns past the batch the pad's (a manifest without per-sequence
    /// states has no line tables; nothing is written). `cols[i]` picks the
    /// entry of a wide table's cell that carries sequence i's line.
    fn stage_lines(&mut self, seqs: &[&Lease], cols: &[usize]) -> Result<()> {
        stage_lines(&mut self.rt.0, &self.line_tables, self.seqs_max, &self.pad, seqs, cols)
    }

    /// Speculative admission: the last prompt token through `decode_spec`
    /// (bs=1, taps) and its row into the draft KV; returns the first
    /// generated token.
    fn first_token(&mut self, s: &Seq) -> Result<u32> {
        let env = env(1, 1);
        self.stage_lines(&[&s.pages], &[])?;
        let rt = &mut self.rt.0;
        rt.write_input_at("token_ids", &le_bytes_i64(&[s.next as i64]), &env)?;
        rt.write_input_at("positions", &le_bytes_i64(&[s.pos as i64]), &env)?;
        rt.write_input_at("slot_mapping", &le_bytes_i64(&[s.pages.slot(s.pos)]), &env)?;
        rt.write_input_at("seq_lens", &le_bytes_i32(&[s.pos as i32 + 1]), &env)?;
        rt.write_input_at("cu_seqlens_q", &le_bytes_i32(&[0, 1]), &env)?;
        stage_tables(rt, &self.page_tables, &[&s.pages], &self.pad, 1, &env)?;
        run_program(rt, "decode_spec", &env, self.policy.eager)?;
        run_program(rt, "draft_precompute", &env, self.policy.eager)?;
        Ok(first_i64(&rt.read_output("next_token")?) as u32)
    }

    /// One speculative round over every running sequence: draft, verify,
    /// precompute, accept — the `round` program, or the four phased ones.
    fn spec_round(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        self.drop_aborted(ledger);
        let n = self.running.len();
        if n == 0 {
            return Ok(());
        }
        let b = self.bucket(n);
        let plan = self.spec.as_ref().unwrap();
        let (nd, dr, vr, mask, advance, fused) =
            (plan.n_drafts, plan.draft_rows, plan.verify_rows, plan.mask_token, plan.advance, plan.fused);
        let t0 = Instant::now();

        // A batch of row-groups: `rows` per sequence, padding sequences
        // write their rows into the pad page.
        struct Group<'a> {
            ids: Vec<i64>,
            positions: Vec<i64>,
            slots: Vec<i64>,
            seq_lens: Vec<i32>,
            cu: Vec<i32>,
            pad: &'a Lease,
        }
        impl<'a> Group<'a> {
            fn new(rows: usize, pad: &'a Lease) -> Group<'a> {
                Group {
                    ids: Vec::with_capacity(rows),
                    positions: Vec::with_capacity(rows),
                    slots: Vec::with_capacity(rows),
                    seq_lens: Vec::new(),
                    cu: vec![0],
                    pad,
                }
            }
            fn push(&mut self, ids: &[i64], pos: usize, pages: &Lease) {
                let rows = ids.len();
                self.ids.extend_from_slice(ids);
                self.positions.extend((pos..pos + rows).map(|p| p as i64));
                self.slots.extend(pages.slots(pos..pos + rows));
                self.seq_lens.push((pos + rows) as i32);
                self.cu.push(self.cu.last().unwrap() + rows as i32);
            }
            fn pad_to(&mut self, b: usize, rows: usize) {
                let pad = self.pad;
                while self.seq_lens.len() < b {
                    self.push(&vec![0; rows], 0, pad);
                }
            }
            fn stage(&self, rt: &mut Runtime, env: &BTreeMap<String, u64>) -> Result<()> {
                rt.write_input_at("token_ids", &le_bytes_i64(&self.ids), env)?;
                rt.write_input_at("positions", &le_bytes_i64(&self.positions), env)?;
                rt.write_input_at("slot_mapping", &le_bytes_i64(&self.slots), env)?;
                rt.write_input_at("seq_lens", &le_bytes_i32(&self.seq_lens), env)?;
                rt.write_input_at("cu_seqlens_q", &le_bytes_i32(&self.cu), env)?;
                Ok(())
            }
        }

        // Draft: [anchor, mask × (dr-1)] per sequence at pos.., non-causal.
        let mut g = Group::new(b * dr, &self.pad);
        let mut anchors = Vec::with_capacity(b);
        let mut ids = vec![mask; dr];
        for s in &self.running {
            ids[0] = s.next as i64;
            g.push(&ids, s.pos, &s.pages);
            anchors.push(s.next as i64);
        }
        g.pad_to(b, dr);
        anchors.resize(b, 0);
        let env_d = env(b * dr, b);
        let eager = self.policy.eager;
        // The batch's page tables and line tables (the line in entry 0 of
        // a wide table's cell: verify resumes from the committed state).
        let leases: Vec<&Lease> = self.running.iter().map(|s| &s.pages).collect();
        stage_lines(&mut self.rt.0, &self.line_tables, self.seqs_max, &self.pad, &leases, &[])?;
        let rt = &mut self.rt.0;
        stage_tables(rt, &self.page_tables, &leases, &self.pad, b, &env_d)?;
        g.stage(rt, &env_d)?;
        rt.write_input_at("anchor_token", &le_bytes_i64(&anchors), &env_d)?;
        let (drafts, vt) = if fused {
            // Verify resumes from the committed state; the round's accept
            // writes advance's own `num_accepted` and line table.
            if advance {
                rt.write_input_at("num_accepted_tokens", &le_bytes_i32(&vec![1; b]), &env_d)?;
            }
            run_program(rt, "round", &env_d, eager)?;
            (i64_from_le(&rt.read_output("draft_tokens")?), i64_from_le(&rt.read_output("verify_tokens")?))
        } else {
            run_program(rt, "draft", &env_d, eager)?;
            let drafts = i64_from_le(&rt.read_output("draft_tokens")?);

            // Verify: [anchor, d0..] per sequence at pos.., causal; row i of
            // a group answers "what follows position pos+i".
            let mut g = Group::new(b * vr, &self.pad);
            let mut ids = vec![0i64; vr];
            for (i, s) in self.running.iter().enumerate() {
                ids[0] = s.next as i64;
                ids[1..].copy_from_slice(&drafts[i * nd..i * nd + nd]);
                g.push(&ids, s.pos, &s.pages);
            }
            g.pad_to(b, vr);
            let env_v = env(b * vr, b);
            g.stage(rt, &env_v)?;
            if advance {
                rt.write_input_at("num_accepted_tokens", &le_bytes_i32(&vec![1; b]), &env_v)?;
            }
            run_program(rt, "verify", &env_v, eager)?;
            let vt = i64_from_le(&rt.read_output("verify_tokens")?);
            // Every row's tap into the draft KV (positions/slot_mapping are
            // still verify's): rejected rows land past the sequence's new
            // position and the next round overwrites them.
            run_program(rt, "draft_precompute", &env_v, eager)?;
            (drafts, vt)
        };

        // Accept the longest matching prefix; vt[a] is the correction (or
        // the bonus token when everything matched).
        let accepted: Vec<usize> = (0..n)
            .map(|i| {
                let d = &drafts[i * nd..i * nd + nd];
                let v = &vt[i * vr..i * vr + vr];
                d.iter().zip(v).take_while(|(x, y)| x == y).count()
            })
            .collect();
        if advance && !fused {
            // Commit the accepted rows into the recurrent state: the
            // target re-runs verify's rows from the state after the anchor
            // and stores after the last accepted one — the line moves to
            // entry `a` of its cell, `num_accepted_tokens` = a + 1.
            let mut nacc: Vec<i32> = accepted.iter().map(|&a| a as i32 + 1).collect();
            nacc.resize(b, 1);
            stage_lines(&mut self.rt.0, &self.line_tables, self.seqs_max, &self.pad, &leases, &accepted)?;
            let rt = &mut self.rt.0;
            let env_v = env(b * vr, b);
            rt.write_input_at("num_accepted_tokens", &le_bytes_i32(&nacc), &env_v)?;
            run_program(rt, "advance", &env_v, eager)?;
        }
        self.stats.step_ns += t0.elapsed().as_nanos();
        self.stats.steps += 1;

        let plan = self.spec.as_mut().unwrap();
        let stop = &self.policy.stop_tokens;
        let mut emitted = 0u64;
        let mut i = 0;
        self.running.retain_mut(|s| {
            let v = &vt[i * vr..i * vr + vr];
            let a = accepted[i];
            i += 1;
            for p in &mut plan.counters.num_accepted_tokens_per_pos[..a] {
                *p += 1;
            }
            plan.counters.num_drafts += 1;
            plan.counters.num_draft_tokens += nd as u64;
            plan.counters.num_accepted_tokens += a as u64;
            s.pos += a + 1;
            let toks: Vec<u32> = v[..=a].iter().map(|&t| t as u32).collect();
            let (n, done) = s.emit(&toks, stop, ledger);
            emitted += n;
            !done
        });
        self.stats.tokens += emitted;
        Ok(())
    }

    /// One decode step over every running sequence.
    fn decode(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        self.drop_aborted(ledger);
        let n = self.running.len();
        if n == 0 {
            return Ok(());
        }
        let b = self.bucket(n);
        let env = env(b, b);
        let pad_slot = self.pad.slot(0);
        let mut token_ids = Vec::with_capacity(b);
        let mut positions = Vec::with_capacity(b);
        let mut slots = Vec::with_capacity(b);
        let mut seq_lens = Vec::with_capacity(b);
        for s in &self.running {
            token_ids.push(s.next as i64);
            positions.push(s.pos as i64);
            slots.push(s.pages.slot(s.pos));
            seq_lens.push(s.pos as i32 + 1);
        }
        for _ in n..b {
            token_ids.push(0);
            positions.push(0);
            slots.push(pad_slot);
            seq_lens.push(1);
        }
        let cu: Vec<i32> = (0..=b as i32).collect();
        let program = if b == 1 { "decode" } else { "decode_batch" };
        let t0 = Instant::now();
        let leases: Vec<&Lease> = self.running.iter().map(|s| &s.pages).collect();
        stage_lines(&mut self.rt.0, &self.line_tables, self.seqs_max, &self.pad, &leases, &[])?;
        let rt = &mut self.rt.0;
        stage_tables(rt, &self.page_tables, &leases, &self.pad, b, &env)?;
        rt.write_input_at("token_ids", &le_bytes_i64(&token_ids), &env)?;
        rt.write_input_at("positions", &le_bytes_i64(&positions), &env)?;
        rt.write_input_at("slot_mapping", &le_bytes_i64(&slots), &env)?;
        rt.write_input_at("seq_lens", &le_bytes_i32(&seq_lens), &env)?;
        rt.write_input_at("cu_seqlens_q", &le_bytes_i32(&cu), &env)?;
        run_program(rt, program, &env, self.policy.eager)?;
        let out = i64_from_le(&rt.read_output("next_token")?);
        self.stats.step_ns += t0.elapsed().as_nanos();
        self.stats.steps += 1;

        let mut i = 0;
        let stop = &self.policy.stop_tokens;
        let mut emitted = 0u64;
        self.running.retain_mut(|s| {
            let tok = out[i] as u32;
            i += 1;
            s.pos += 1;
            let (n, done) = s.emit(&[tok], stop, ledger);
            emitted += n;
            !done
        });
        self.stats.tokens += emitted;
        Ok(())
    }

    /// One line per 5 s window in which anything happened; a window that
    /// only idled is dropped (and restarted, so the next line's rates are
    /// over its own window). `steps` are speculative rounds under
    /// `--spec`, and `accepted` / `accept_pct` are the window's.
    fn log_stats(&mut self) {
        let st = &self.stats;
        let dt = st.since.elapsed();
        if dt.as_secs() < 5 {
            return;
        }
        if st.tokens > 0 || st.prefill_tokens > 0 {
            let round = |x: f64, d: f64| (x * d).round() / d;
            let (drafts, draft_tokens, accepted) = self.spec.as_ref().map_or((0, 0, 0), |p| {
                let c = &p.counters;
                (c.num_drafts - st.spec_at.0, c.num_draft_tokens - st.spec_at.1, c.num_accepted_tokens - st.spec_at.2)
            });
            info!(
                running = self.running.len(),
                waiting = self.waiting.len(),
                kv_pct = round(self.kv_used() as f64 * 100.0 / self.kv_total().max(1) as f64, 10.0),
                steps = st.steps,
                step_ms = round(st.step_ns as f64 / 1e6 / st.steps.max(1) as f64, 100.0),
                tok_s = round(st.tokens as f64 / dt.as_secs_f64(), 1.0),
                prefill_tokens = st.prefill_tokens,
                prefill_tok_s = round(st.prefill_tokens as f64 / (st.prefill_ns as f64 / 1e9).max(1e-9), 1.0),
                accepted = self.spec.as_ref().map(|_| round((accepted + drafts) as f64 / drafts.max(1) as f64, 100.0)),
                accept_pct = self.spec.as_ref().map(|_| round(accepted as f64 * 100.0 / draft_tokens.max(1) as f64, 1.0)),
                "stats"
            );
        }
        self.stats = Stats::new(&self.spec);
    }
}

impl Scheduler for KernScheduler {
    fn submit(&mut self, request: QueuedRequest) {
        self.waiting.push_back(request);
    }

    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        self.admit(ledger)?;
        if self.spec.is_some() {
            self.spec_round(ledger)?;
        } else {
            self.decode(ledger)?;
        }
        self.log_stats();
        Ok(())
    }

    fn metrics(&self) -> SchedulerMetrics {
        SchedulerMetrics {
            kv_used_blocks: self.kv_used() as u64,
            kv_total_blocks: self.kv_total() as u64,
            num_running_reqs: self.running.len() as u64,
            num_waiting_reqs: self.waiting.len() as u64,
            spec_decode: self.spec.as_ref().map(|p| p.counters),
        }
    }
}

/// Run `program` at `env`: eagerly, or through its CUDA graph, captured
/// on first use.
fn run_program(rt: &mut Runtime, program: &str, env: &BTreeMap<String, u64>, eager: bool) -> Result<()> {
    if eager {
        return Ok(rt.run(program, env)?);
    }
    if !rt.is_captured(program, env) {
        let t = Instant::now();
        rt.capture(program, env)?;
        info!(program, seqs = env.get("seqs"), tokens = env.get("tokens"), capture_ms = logline::ms(t.elapsed()), "captured");
    }
    Ok(rt.run_captured(program, env)?)
}

/// See [`KernScheduler::stage_lines`].
/// Every line table for a batch: cell `[r, i]` carries sequence i's line
/// r — in entry `cols[i]` (0 past `cols`) of a wide table's cell, the
/// null line 0 in the rest — and the pad's line past the batch.
fn stage_lines(
    rt: &mut Runtime,
    tables: &[LineTable],
    seqs_max: usize,
    pad: &Lease,
    seqs: &[&Lease],
    cols: &[usize],
) -> Result<()> {
    for LineTable { name, rows, width: w } in tables {
        let mut t = vec![0i32; rows * seqs_max * w];
        for r in 0..*rows {
            let fill = pad.seq_line(name, r)?;
            for i in 0..seqs_max {
                let (line, col) = match seqs.get(i) {
                    Some(l) => (l.seq_line(name, r)?, cols.get(i).copied().unwrap_or(0)),
                    None => (fill, 0),
                };
                if col >= *w {
                    bail!("line table `{name}`: entry {col} of a {w}-wide cell");
                }
                t[(r * seqs_max + i) * w + col] = line;
            }
        }
        rt.write_input(name, &le_bytes_i32(&t))?;
    }
    Ok(())
}

/// Every page table for a batch of `b` rows: sequence i's row, the pad's
/// past the batch.
fn stage_tables(
    rt: &mut Runtime,
    tables: &[String],
    seqs: &[&Lease],
    pad: &Lease,
    b: usize,
    env: &BTreeMap<String, u64>,
) -> Result<()> {
    for name in tables {
        let mut t = Vec::new();
        for i in 0..b {
            seqs.get(i).copied().unwrap_or(pad).extend_row(name, &mut t)?;
        }
        rt.write_input_at(name, &le_bytes_i32(&t), env)?;
    }
    Ok(())
}

fn env(tokens: usize, seqs: usize) -> BTreeMap<String, u64> {
    BTreeMap::from([("tokens".to_string(), tokens as u64), ("seqs".to_string(), seqs as u64)])
}

fn need_programs(m: &Manifest, names: &[&str]) -> Result<()> {
    match names.iter().find(|p| !m.programs.contains_key(**p)) {
        Some(p) => bail!("manifest has no program `{p}`"),
        None => Ok(()),
    }
}

fn need_buffers(m: &Manifest, names: &[&str]) -> Result<()> {
    match names.iter().find(|b| !m.buffers.contains_key(**b)) {
        Some(b) => bail!("manifest has no input buffer `{b}`"),
        None => Ok(()),
    }
}

fn var_max(m: &Manifest, name: &str) -> Result<usize> {
    m.vars.get(name).map(|v| v.max as usize).with_context(|| format!("manifest has no var `{name}`"))
}

fn shape<'m>(m: &'m Manifest, name: &str) -> Result<&'m [Dim]> {
    m.buffers.get(name).map(|b| b.shape.as_slice()).with_context(|| format!("manifest has no buffer `{name}`"))
}

/// A buffer shaped `[seqs, n]`, one row per sequence: `n`.
fn seqs_rows(m: &Manifest, name: &str) -> Result<usize> {
    match shape(m, name)? {
        [Dim::Var(v), Dim::Const(n)] if v == "seqs" => Ok(*n as usize),
        s => bail!("`{name}` shaped {s:?}, expected [seqs, n]"),
    }
}

/// A buffer shaped `[seqs]`, one entry per sequence.
fn per_seq(m: &Manifest, name: &str) -> Result<()> {
    match shape(m, name)? {
        [Dim::Var(v)] if v == "seqs" => Ok(()),
        s => bail!("`{name}` shaped {s:?}, expected [seqs]"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kern_manifest::types::{Buffer, Spec};

    /// The plain contract: 8 tokens, 4 sequences, a 3-page `block_table`
    /// and a 3-line table over a recurrent state.
    fn plain() -> Manifest {
        Manifest::from_json(
            r#"{
            "schema_version": 3, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 4}},
            "states": {"kv": {"bytes_per_token": 1}, "gdn": {"bytes_per_seq": 24}},
            "buffers": {
                "token_ids": {"kind": "input", "dtype": "i64", "shape": ["tokens"]},
                "positions": {"kind": "input", "dtype": "i64", "shape": ["tokens"]},
                "slot_mapping": {"kind": "input", "dtype": "i64", "shape": ["tokens"], "domain": {"index_into": "kv"}},
                "seq_lens": {"kind": "input", "dtype": "i32", "shape": ["seqs"]},
                "cu_seqlens_q": {"kind": "input", "dtype": "i32", "shape": ["seqs"]},
                "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 3], "domain": {"index_into": "kv", "stride": 16}},
                "line_index": {"kind": "input", "dtype": "i32", "shape": [3, "seqs"], "domain": {"index_into": "gdn", "stride": 8}}
            },
            "modules": {}, "ops": {}, "programs": {"prefill": [], "decode": [], "decode_batch": []}
        }"#,
        )
        .unwrap()
    }

    fn buffer(kind: &str, shape: &str) -> Buffer {
        serde_json::from_str(&format!(r#"{{"kind": "{kind}", "dtype": "i64", "shape": {shape}}}"#)).unwrap()
    }

    /// The same plus DSpark's contract: 3 drafts, `spec.block` 4.
    fn speculative() -> Manifest {
        let mut m = plain();
        m.spec = Some(Spec { block: 4, mask_token: 7 });
        m.buffers.insert("draft_tokens".into(), buffer("output", r#"["seqs", 3]"#));
        m.buffers.insert("verify_tokens".into(), buffer("output", r#"["seqs", 4]"#));
        m.buffers.insert("anchor_token".into(), buffer("input", r#"["seqs"]"#));
        for p in ["decode_spec", "draft_precompute", "draft", "verify"] {
            m.programs.insert(p.into(), vec![]);
        }
        m
    }

    fn check(m: &Manifest, spec: bool) -> Result<Contract> {
        Contract::check(m, &["line_index"], &["block_table"], 16, spec)
    }

    fn rejects(m: &Manifest, spec: bool, what: &str) {
        let Err(e) = check(m, spec) else { panic!("accepted, expected `{what}`") };
        let e = format!("{e:#}");
        assert!(e.contains(what), "{e}");
    }

    #[test]
    fn plain_contract() {
        let c = check(&plain(), false).unwrap();
        assert!(!c.prefill_emits && c.spec.is_none());
        assert_eq!((c.seqs_max, c.tokens_max), (4, 8));
        let lines: Vec<_> = c.line_tables.iter().map(|t| (t.name.as_str(), t.rows, t.width)).collect();
        assert_eq!((lines, c.page_tables.clone()), (vec![("line_index", 3, 1)], vec!["block_table".to_string()]));
        assert_eq!((c.max_seqs(0), c.max_seqs(3), c.max_seqs(100)), (1, 3, 4));
    }

    #[test]
    fn prefill_emits_when_it_writes_next_token() {
        let mut m = plain();
        let call = serde_json::from_str(r#"{"op": "head", "args": [{"buf": "next_token"}]}"#).unwrap();
        m.programs.insert("prefill".into(), vec![call]);
        assert!(check(&m, false).unwrap().prefill_emits);
    }

    #[test]
    fn plain_rejections() {
        let mut m = plain();
        m.programs.remove("decode_batch");
        rejects(&m, false, "no program `decode_batch`");
        let mut m = plain();
        m.buffers.remove("cu_seqlens_q");
        rejects(&m, false, "no input buffer `cu_seqlens_q`");
        let mut m = plain();
        m.buffers.get_mut("line_index").unwrap().shape = vec![Dim::Const(3)];
        rejects(&m, false, "expected [lines, seqs]");
        let mut m = plain();
        m.buffers.get_mut("block_table").unwrap().shape = vec![Dim::Var("seqs".into())];
        rejects(&m, false, "`block_table` shaped [Var(\"seqs\")], expected [seqs, n]");
        let Err(e) = Contract::check(&plain(), &[], &[], 16, false) else { panic!("no page table accepted") };
        assert!(e.to_string().contains("block_table"), "{e}");
    }

    #[test]
    fn speculative_contract() {
        let c = check(&speculative(), true).unwrap();
        let s = c.spec.as_ref().unwrap();
        assert_eq!((s.n_drafts, s.draft_rows, s.verify_rows, s.mask_token, s.advance, s.fused), (3, 4, 4, 7, false, false));
        assert_eq!(s.counters.num_spec_tokens, 3);
        // Four rows per sequence per round fit twice in 8 tokens.
        assert_eq!((c.max_seqs(1), c.max_seqs(4)), (1, 2));
        // Without a `spec` block the draft rows are the drafts and the
        // mask is DSpark's.
        let mut m = speculative();
        m.spec = None;
        let c = check(&m, true).unwrap();
        let s = c.spec.as_ref().unwrap();
        assert_eq!((s.draft_rows, s.mask_token), (3, DSPARK_MASK_TOKEN));
        // A plain manifest has no `decode_batch` to miss under --spec.
        let mut m = speculative();
        m.programs.remove("decode_batch");
        assert!(check(&m, true).is_ok());
    }

    #[test]
    fn speculative_rejections() {
        let mut m = speculative();
        m.programs.remove("verify");
        rejects(&m, true, "--spec: the manifest's speculative contract: manifest has no program `verify`");
        let mut m = speculative();
        m.buffers.get_mut("verify_tokens").unwrap().shape = vec![Dim::Var("seqs".into()), Dim::Const(5)];
        rejects(&m, true, "verify_tokens has 5 rows per sequence, expected 4");
        let mut m = speculative();
        m.buffers.get_mut("anchor_token").unwrap().shape = vec![Dim::Const(4)];
        rejects(&m, true, "`anchor_token` shaped [Const(4)], expected [seqs]");
        let mut m = speculative();
        m.programs.insert("advance".into(), vec![]);
        rejects(&m, true, "`advance` without a `num_accepted_tokens`");
        let mut m = speculative();
        m.buffers.insert("num_accepted_tokens".into(), buffer("input", r#"["seqs"]"#));
        rejects(&m, true, "no `advance` program");
        m.programs.insert("advance".into(), vec![]);
        assert!(check(&m, true).unwrap().spec.unwrap().advance);
        // `round` fuses draft and verify: their rows must coincide.
        let mut m = speculative();
        m.programs.insert("round".into(), vec![]);
        m.programs.remove("draft");
        assert!(check(&m, true).unwrap().spec.unwrap().fused);
        m.spec = Some(Spec { block: 3, mask_token: 7 });
        rejects(&m, true, "coincide, got 3 and 4");
        // A round's rows must fit the pad page.
        let Err(e) = Contract::check(&speculative(), &["line_index"], &["block_table"], 2, true) else { panic!("4 rows in a 2-token page") };
        assert!(format!("{e:#}").contains("exceed the 2-token pad page"), "{e:#}");
    }
}
