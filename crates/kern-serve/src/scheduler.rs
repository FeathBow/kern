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
//!   `n` tokens so the last round's rejected rows have slots. Whether a
//!   round beats a plain step at a given batch size is the operator's
//!   call, not the scheduler's: the flag picks the mode for the process.

use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use kern_manifest::types::{Arg, Dim};
use kern_runtime::{Denied, Error, Lease, Runtime};
use pegainfer_frontend::engine::{
    FinishReason, QueuedRequest, RejectReason, RequestId, RequestLedger, Scheduler,
    SchedulerMetrics, SpecDecodeCounters, MAX_SPEC_TOKENS,
};
use tracing::{debug, info, warn};

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
    counters: SpecDecodeCounters,
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

pub struct KernScheduler {
    rt: Rt,
    /// One page (and sequence slot) no sequence owns: the padding rows of
    /// a decode batch write their junk here.
    pad: Lease,
    policy: Policy,
    spec: Option<SpecPlan>,
    /// `prefill` emits `next_token` itself (see the module doc).
    prefill_emits: bool,
    /// The manifest's line tables over per-sequence states: (name, lines
    /// per sequence, entries per cell), shaped `[lines, seqs]` or
    /// `[lines, seqs, w]`.
    line_tables: Vec<(String, usize, usize)>,
    /// The page tables, each shaped `[seqs, n]`: `block_table` and, under
    /// speculation, the draft's.
    page_tables: Vec<String>,
    /// The `seqs` bound: a line table's row width.
    seqs_max: usize,
    waiting: VecDeque<QueuedRequest>,
    running: Vec<Seq>,
    warned_sampling: bool,
    // Rolling stats for the periodic log line.
    stat_since: Instant,
    stat_steps: u64,
    stat_tokens: u64,
    stat_decode_ns: u128,
    stat_prefill_tokens: u64,
    stat_prefill_ns: u128,
}

/// Public facts the frontend wants at launch.
pub struct Facts {
    pub total_blocks: usize,
    pub block_size: usize,
    /// Longest request (prompt + completion) one sequence can hold.
    pub max_request_tokens: usize,
}

impl KernScheduler {
    /// Wrap a loaded runtime (weights bound). Validates the manifest against
    /// this caller contract.
    pub fn new(mut rt: Runtime, mut policy: Policy) -> Result<(KernScheduler, Facts)> {
        let m = &rt.manifest;
        for p in ["prefill", "decode", "decode_batch"] {
            if !m.programs.contains_key(p) {
                bail!("manifest has no program `{p}`");
            }
        }
        for b in ["token_ids", "positions", "slot_mapping", "seq_lens", "cu_seqlens_q", "block_table"] {
            if !m.buffers.contains_key(b) {
                bail!("manifest has no input buffer `{b}`");
            }
        }
        let seqs_max = m.vars.get("seqs").map(|v| v.max as usize).context("manifest has no `seqs` var")?;
        let tokens_max = m.vars["tokens"].max as usize;
        match m.buffers["block_table"].shape.as_slice() {
            [Dim::Var(v), Dim::Const(_)] if v == "seqs" => {}
            s => bail!("block_table shape {s:?}, expected [seqs, n]"),
        }
        let prefill_emits = m.programs["prefill"]
            .iter()
            .any(|c| c.args.iter().any(|a| matches!(a, Arg::Buf { buf, .. } if buf == "next_token")));
        let line_tables = rt
            .seq_tables()
            .map(|name| match m.buffers[name].shape.as_slice() {
                [Dim::Const(rows), Dim::Var(v)] if v == "seqs" => Ok((name.to_string(), *rows as usize, 1)),
                [Dim::Const(rows), Dim::Var(v), Dim::Const(w)] if v == "seqs" => {
                    Ok((name.to_string(), *rows as usize, *w as usize))
                }
                s => bail!("line table `{name}` shaped {s:?}, expected [lines, seqs] or [lines, seqs, w]"),
            })
            .collect::<Result<Vec<_>>>()?;
        let page_tables = rt
            .page_tables()
            .map(|name| match m.buffers[name].shape.as_slice() {
                [Dim::Var(v), Dim::Const(_)] if v == "seqs" => Ok(name.to_string()),
                s => bail!("page table `{name}` shaped {s:?}, expected [seqs, n]"),
            })
            .collect::<Result<Vec<_>>>()?;
        policy.max_seqs = policy.max_seqs.clamp(1, seqs_max);
        policy.chunk = policy.chunk.clamp(1, tokens_max);
        let spec = if policy.spec {
            for p in ["decode_spec", "draft", "verify", "draft_precompute"] {
                if !m.programs.contains_key(p) {
                    bail!("--spec needs program `{p}` (not in this manifest)");
                }
            }
            let rows_per_seq = |name: &str| -> Result<usize> {
                match m.buffers.get(name).map(|b| b.shape.as_slice()) {
                    Some([Dim::Var(v), Dim::Const(n)]) if v == "seqs" => Ok(*n as usize),
                    s => bail!("--spec needs `{name}` shaped [seqs, n] (one row per sequence), got {s:?}"),
                }
            };
            let n_drafts = rows_per_seq("draft_tokens")?;
            let verify_rows = rows_per_seq("verify_tokens")?;
            if verify_rows != n_drafts + 1 {
                bail!("--spec: verify_tokens has {verify_rows} per sequence, expected {} (anchor + drafts)", n_drafts + 1);
            }
            match m.buffers.get("anchor_token").map(|b| b.shape.as_slice()) {
                Some([Dim::Var(v)]) if v == "seqs" => {}
                s => bail!("--spec needs `anchor_token` shaped [seqs], got {s:?}"),
            }
            let advance = m.programs.contains_key("advance");
            if let Some(b) = m.buffers.get("num_accepted_tokens") {
                match b.shape.as_slice() {
                    [Dim::Var(v)] if v == "seqs" => {}
                    s => bail!("--spec needs `num_accepted_tokens` shaped [seqs] (one per sequence), got {s:?}"),
                }
                if !advance {
                    bail!("--spec: this manifest resumes a recurrent state from `num_accepted_tokens` but has no `advance` program to commit the accepted rows");
                }
            } else if advance {
                bail!("--spec: program `advance` without a `num_accepted_tokens` input");
            }
            if n_drafts > MAX_SPEC_TOKENS {
                bail!("--spec: {n_drafts} drafts per round, the frontend's metrics hold {MAX_SPEC_TOKENS}");
            }
            let (draft_rows, mask_token) = match &m.spec {
                Some(s) => (s.block as usize, s.mask_token),
                None => (n_drafts, DSPARK_MASK_TOKEN),
            };
            let rows = draft_rows.max(verify_rows);
            if rows > rt.page() as usize {
                bail!("--spec: {rows} rows per sequence per round exceed the {}-token pad page", rt.page());
            }
            // A round's rows: every sequence's row-group must fit `tokens`.
            policy.max_seqs = policy.max_seqs.min(tokens_max / rows).max(1);
            Some(SpecPlan {
                n_drafts,
                draft_rows,
                verify_rows,
                mask_token,
                advance,
                counters: SpecDecodeCounters {
                    num_spec_tokens: n_drafts as u64,
                    num_drafts: 0,
                    num_draft_tokens: 0,
                    num_accepted_tokens: 0,
                    num_accepted_tokens_per_pos: [0; MAX_SPEC_TOKENS],
                },
            })
        } else {
            None
        };
        let pad = rt.lease(1).map_err(|e| anyhow::anyhow!("no page for the padding rows: {e}"))?;
        if rt.pages_used() == rt.pages_total() {
            bail!("capacity {} tokens holds one page; nothing left to serve from", rt.capacity());
        }
        let facts = Facts {
            total_blocks: rt.pages_total() - pad.pages(),
            block_size: rt.page() as usize,
            max_request_tokens: rt.max_seq_tokens(),
        };
        info!(
            "scheduler: {} pages × {} tokens (+1 pad page){}, ≤{} sequences, ≤{} tokens/sequence, chunk {}, buckets {:?}{}{}{}",
            facts.total_blocks,
            facts.block_size,
            if rt.seq_slots() > 0 { format!(", {} sequence slots", rt.seq_slots()) } else { String::new() },
            policy.max_seqs,
            facts.max_request_tokens,
            policy.chunk,
            BUCKETS.iter().filter(|&&b| b <= policy.max_seqs).collect::<Vec<_>>(),
            if policy.eager { ", eager" } else { ", graphs captured per bucket on first use" },
            if prefill_emits { ", prefill emits next_token" } else { "" },
            match &spec {
                Some(s) => format!(", speculative: {} drafts/round ({} draft + {} verify rows per sequence)", s.n_drafts, s.draft_rows, s.verify_rows),
                None => String::new(),
            }
        );
        Ok((
            KernScheduler {
                rt: Rt(rt),
                pad,
                policy,
                spec,
                prefill_emits,
                line_tables,
                page_tables,
                seqs_max,
                waiting: VecDeque::new(),
                running: Vec::new(),
                warned_sampling: false,
                stat_since: Instant::now(),
                stat_steps: 0,
                stat_tokens: 0,
                stat_decode_ns: 0,
                stat_prefill_tokens: 0,
                stat_prefill_ns: 0,
            },
            facts,
        ))
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
            // A speculative round writes `n_drafts` rows past the token it
            // may still have to emit; the last round's rejects need slots.
            let worst = prompt + max_tokens + self.spec.as_ref().map_or(0, |s| s.n_drafts);
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
                    "request {id}: non-greedy sampling params (temperature {}, top_p {}, top_k {}) — this engine samples greedily (argmax in the manifest); further requests are not warned about",
                    q.request.params.temperature, q.request.params.top_p, q.request.params.top_k
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
            self.stat_prefill_ns += t0.elapsed().as_nanos();
            self.stat_prefill_tokens += n_pre as u64;
            budget_used += n_pre;
            debug!(
                "admitted {id}: {prompt} prompt tokens, max_tokens {max_tokens}, {:?}, prefill {:?}",
                pages,
                t0.elapsed()
            );
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
                seq.generated += 1;
                let reason = if !seq.ignore_eos && self.policy.stop_tokens.contains(&tok) {
                    Some(FinishReason::Stop)
                } else {
                    ledger.push_tokens(id, &[tok], &[]);
                    self.stat_tokens += 1;
                    (seq.generated >= seq.max_tokens).then_some(FinishReason::Length)
                };
                if let Some(r) = reason {
                    ledger.finish(id, r);
                    continue;
                }
                seq.next = tok;
            }
            self.running.push(seq);
        }
        Ok(())
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
            rt.write_input_at("token_ids", &le_i64(&ids[pos..pos + c]), &env)?;
            rt.write_input_at("positions", &le_i64(&positions), &env)?;
            rt.write_input_at("slot_mapping", &le_i64(&slots), &env)?;
            rt.write_input_at("seq_lens", &le_i32(&[(pos + c) as i32]), &env)?;
            rt.write_input_at("cu_seqlens_q", &le_i32(&[0, c as i32]), &env)?;
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
        let out = self.rt.0.read_output("next_token")?;
        Ok(Some(i64::from_le_bytes(out[..8].try_into().unwrap()) as u32))
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
        rt.write_input_at("token_ids", &le_i64(&[s.next as i64]), &env)?;
        rt.write_input_at("positions", &le_i64(&[s.pos as i64]), &env)?;
        rt.write_input_at("slot_mapping", &le_i64(&[s.pages.slot(s.pos)]), &env)?;
        rt.write_input_at("seq_lens", &le_i32(&[s.pos as i32 + 1]), &env)?;
        rt.write_input_at("cu_seqlens_q", &le_i32(&[0, 1]), &env)?;
        stage_tables(rt, &self.page_tables, &[&s.pages], &self.pad, 1, &env)?;
        run_program(rt, "decode_spec", &env, self.policy.eager)?;
        run_program(rt, "draft_precompute", &env, self.policy.eager)?;
        let out = rt.read_output("next_token")?;
        Ok(i64::from_le_bytes(out[..8].try_into().unwrap()) as u32)
    }

    /// One speculative round over every running sequence: draft, verify,
    /// precompute, accept.
    fn spec_round(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        self.running.retain(|s| {
            let live = !ledger.is_aborted(s.id);
            if !live {
                ledger.retire(s.id);
            }
            live
        });
        let n = self.running.len();
        if n == 0 {
            return Ok(());
        }
        let b = self.bucket(n);
        let plan = self.spec.as_ref().unwrap();
        let (nd, dr, vr, mask, advance) = (plan.n_drafts, plan.draft_rows, plan.verify_rows, plan.mask_token, plan.advance);
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
            fn push(&mut self, ids: &[i64], pos: usize, pages: &Lease) -> Result<()> {
                let rows = ids.len();
                self.ids.extend_from_slice(ids);
                self.positions.extend((pos..pos + rows).map(|p| p as i64));
                self.slots.extend(pages.slots(pos..pos + rows));
                self.seq_lens.push((pos + rows) as i32);
                self.cu.push(self.cu.last().unwrap() + rows as i32);
                Ok(())
            }
            fn pad_to(&mut self, b: usize, rows: usize) -> Result<()> {
                let pad = self.pad;
                while self.seq_lens.len() < b {
                    self.push(&vec![0; rows], 0, pad)?;
                }
                Ok(())
            }
            fn stage(&self, rt: &mut Runtime, env: &BTreeMap<String, u64>) -> Result<()> {
                rt.write_input_at("token_ids", &le_i64(&self.ids), env)?;
                rt.write_input_at("positions", &le_i64(&self.positions), env)?;
                rt.write_input_at("slot_mapping", &le_i64(&self.slots), env)?;
                rt.write_input_at("seq_lens", &le_i32(&self.seq_lens), env)?;
                rt.write_input_at("cu_seqlens_q", &le_i32(&self.cu), env)?;
                Ok(())
            }
        }

        // Draft: [anchor, mask × (dr-1)] per sequence at pos.., non-causal.
        let mut g = Group::new(b * dr, &self.pad);
        let mut anchors = Vec::with_capacity(b);
        let mut ids = vec![mask; dr];
        for s in &self.running {
            ids[0] = s.next as i64;
            g.push(&ids, s.pos, &s.pages)?;
            anchors.push(s.next as i64);
        }
        g.pad_to(b, dr)?;
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
        rt.write_input_at("anchor_token", &le_i64(&anchors), &env_d)?;
        run_program(rt, "draft", &env_d, eager)?;
        let drafts = i64s(&rt.read_output("draft_tokens")?);

        // Verify: [anchor, d0..] per sequence at pos.., causal; row i of a
        // group answers "what follows position pos+i".
        let mut g = Group::new(b * vr, &self.pad);
        let mut ids = vec![0i64; vr];
        for (i, s) in self.running.iter().enumerate() {
            ids[0] = s.next as i64;
            ids[1..].copy_from_slice(&drafts[i * nd..i * nd + nd]);
            g.push(&ids, s.pos, &s.pages)?;
        }
        g.pad_to(b, vr)?;
        let env_v = env(b * vr, b);
        let rt = &mut self.rt.0;
        g.stage(rt, &env_v)?;
        if advance {
            rt.write_input_at("num_accepted_tokens", &le_i32(&vec![1; b]), &env_v)?;
        }
        run_program(rt, "verify", &env_v, eager)?;
        let vt = i64s(&rt.read_output("verify_tokens")?);
        // Every row's tap into the draft KV (positions/slot_mapping are
        // still verify's): rejected rows land past the sequence's new
        // position and the next round overwrites them.
        run_program(rt, "draft_precompute", &env_v, eager)?;

        // Accept the longest matching prefix; vt[a] is the correction (or
        // the bonus token when everything matched).
        let accepted: Vec<usize> = (0..n)
            .map(|i| {
                let d = &drafts[i * nd..i * nd + nd];
                let v = &vt[i * vr..i * vr + vr];
                d.iter().zip(v).take_while(|(x, y)| x == y).count()
            })
            .collect();
        if advance {
            // Commit the accepted rows into the recurrent state: the
            // target re-runs verify's rows from the state after the anchor
            // and stores after the last accepted one — the line moves to
            // entry `a` of its cell, `num_accepted_tokens` = a + 1.
            let mut nacc: Vec<i32> = accepted.iter().map(|&a| a as i32 + 1).collect();
            nacc.resize(b, 1);
            stage_lines(&mut self.rt.0, &self.line_tables, self.seqs_max, &self.pad, &leases, &accepted)?;
            let rt = &mut self.rt.0;
            rt.write_input_at("num_accepted_tokens", &le_i32(&nacc), &env_v)?;
            run_program(rt, "advance", &env_v, eager)?;
        }
        self.stat_decode_ns += t0.elapsed().as_nanos();
        self.stat_steps += 1;

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
            let mut out = Vec::with_capacity(a + 1);
            let mut reason = None;
            for &tok in &v[..=a] {
                let tok = tok as u32;
                s.generated += 1;
                // The stop token itself is not emitted (pegainfer
                // convention); it still counts against max_tokens.
                if !s.ignore_eos && stop.contains(&tok) {
                    reason = Some(FinishReason::Stop);
                    break;
                }
                out.push(tok);
                if s.generated >= s.max_tokens {
                    reason = Some(FinishReason::Length);
                    break;
                }
            }
            if !out.is_empty() {
                ledger.push_tokens(s.id, &out, &[]);
                emitted += out.len() as u64;
            }
            match reason {
                Some(r) => {
                    debug!(
                        "finished {}: {r:?}, {} prompt + {} generated in {:?}",
                        s.id, s.prompt_len, s.generated, s.admitted.elapsed()
                    );
                    ledger.finish(s.id, r);
                    false
                }
                None => {
                    s.next = v[a] as u32;
                    true
                }
            }
        });
        self.stat_tokens += emitted;
        Ok(())
    }

    /// One decode step over every running sequence.
    fn decode(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        // Drop aborted sequences first so they neither pad nor compute.
        self.running.retain(|s| {
            let live = !ledger.is_aborted(s.id);
            if !live {
                ledger.retire(s.id);
            }
            live
        });
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
        rt.write_input_at("token_ids", &le_i64(&token_ids), &env)?;
        rt.write_input_at("positions", &le_i64(&positions), &env)?;
        rt.write_input_at("slot_mapping", &le_i64(&slots), &env)?;
        rt.write_input_at("seq_lens", &le_i32(&seq_lens), &env)?;
        rt.write_input_at("cu_seqlens_q", &le_i32(&cu), &env)?;
        run_program(rt, program, &env, self.policy.eager)?;
        let out = rt.read_output("next_token")?;
        self.stat_decode_ns += t0.elapsed().as_nanos();
        self.stat_steps += 1;
        self.stat_tokens += n as u64;

        let mut i = 0;
        let stop = &self.policy.stop_tokens;
        self.running.retain_mut(|s| {
            let tok = i64::from_le_bytes(out[i * 8..i * 8 + 8].try_into().unwrap()) as u32;
            i += 1;
            s.pos += 1;
            s.generated += 1;
            // The stop token itself is not emitted (pegainfer convention);
            // it still counts against max_tokens like vLLM's.
            let reason = if !s.ignore_eos && stop.contains(&tok) {
                Some(FinishReason::Stop)
            } else {
                ledger.push_tokens(s.id, &[tok], &[]);
                (s.generated >= s.max_tokens).then_some(FinishReason::Length)
            };
            match reason {
                Some(r) => {
                    debug!(
                        "finished {}: {r:?}, {} prompt + {} generated in {:?}",
                        s.id, s.prompt_len, s.generated, s.admitted.elapsed()
                    );
                    ledger.finish(s.id, r);
                    false
                }
                None => {
                    s.next = tok;
                    true
                }
            }
        });
        Ok(())
    }

    fn log_stats(&mut self) {
        let dt = self.stat_since.elapsed();
        if dt.as_secs() < 5 || self.stat_steps == 0 && self.stat_prefill_tokens == 0 {
            return;
        }
        let spec = match &self.spec {
            Some(p) => {
                let c = &p.counters;
                format!(
                    " | spec {:.2} tok/round, {:.0}% drafts accepted (cumulative)",
                    (c.num_accepted_tokens + c.num_drafts) as f64 / c.num_drafts.max(1) as f64,
                    c.num_accepted_tokens as f64 * 100.0 / c.num_draft_tokens.max(1) as f64,
                )
            }
            None => String::new(),
        };
        info!(
            "{} running, {} waiting, {}/{} pages | {} {} {}, {:.2} ms each, {:.0} tok/s | prefill {} tokens, {:.0} tok/s{spec}",
            self.running.len(),
            self.waiting.len(),
            self.kv_used(),
            self.kv_total(),
            if self.spec.is_some() { "spec" } else { "decode" },
            self.stat_steps,
            if self.spec.is_some() { "rounds" } else { "steps" },
            self.stat_decode_ns as f64 / 1e6 / self.stat_steps.max(1) as f64,
            self.stat_tokens as f64 / dt.as_secs_f64(),
            self.stat_prefill_tokens,
            self.stat_prefill_tokens as f64 / (self.stat_prefill_ns as f64 / 1e9).max(1e-9),
        );
        self.stat_since = Instant::now();
        self.stat_steps = 0;
        self.stat_tokens = 0;
        self.stat_decode_ns = 0;
        self.stat_prefill_tokens = 0;
        self.stat_prefill_ns = 0;
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
        info!("captured `{program}` at {env:?} ({:?})", t.elapsed());
    }
    Ok(rt.run_captured(program, env)?)
}

/// See [`KernScheduler::stage_lines`].
/// Every line table for a batch: cell `[r, i]` carries sequence i's line
/// r — in entry `cols[i]` (0 past `cols`) of a wide table's cell, the
/// null line 0 in the rest — and the pad's line past the batch.
fn stage_lines(
    rt: &mut Runtime,
    tables: &[(String, usize, usize)],
    seqs_max: usize,
    pad: &Lease,
    seqs: &[&Lease],
    cols: &[usize],
) -> Result<()> {
    for (name, rows, w) in tables {
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
        rt.write_input(name, &le_i32(&t))?;
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
        rt.write_input_at(name, &le_i32(&t), env)?;
    }
    Ok(())
}

fn i64s(bytes: &[u8]) -> Vec<i64> {
    bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn env(tokens: usize, seqs: usize) -> BTreeMap<String, u64> {
    BTreeMap::from([("tokens".to_string(), tokens as u64), ("seqs".to_string(), seqs as u64)])
}

fn le_i64(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
