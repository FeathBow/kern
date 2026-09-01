//! The kern scheduler: one `Runtime`, many sequences.
//!
//! Implements the pegainfer frontend's [`Scheduler`] contract — `submit`,
//! `step`, `metrics` — over a kern manifest with the qwen3-4b caller
//! contract: input buffers `token_ids` / `positions` / `slot_mapping` /
//! `seq_lens` / `cu_seqlens_q` / `block_table`, programs `prefill`
//! (single sequence, chunked), `decode` (single sequence, the manifest's
//! bs=1 microprogram) and `decode_batch` (`seqs` sequences, one row each).
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
//!   sampling params are logged once and served greedily.

use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use kern_manifest::types::Dim;
use kern_runtime::{Denied, Lease, Runtime};
use pegainfer_frontend::engine::{
    FinishReason, QueuedRequest, RejectReason, RequestId, RequestLedger, Scheduler,
    SchedulerMetrics,
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
    /// One page no sequence owns: the padding rows of a decode batch write
    /// their junk here.
    pad: Lease,
    policy: Policy,
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
    pub fn new(rt: Runtime, mut policy: Policy) -> Result<(KernScheduler, Facts)> {
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
        let pad = rt.lease(1).map_err(|d| anyhow::anyhow!("no page for the padding rows: {d}"))?;
        if rt.pages_used() == rt.pages_total() {
            bail!("capacity {} tokens holds one page; nothing left to serve from", rt.capacity());
        }
        policy.max_seqs = policy.max_seqs.clamp(1, seqs_max);
        policy.chunk = policy.chunk.clamp(1, tokens_max);
        let facts = Facts {
            total_blocks: rt.pages_total() - pad.pages(),
            block_size: rt.page() as usize,
            max_request_tokens: rt.max_seq_tokens(),
        };
        info!(
            "scheduler: {} pages × {} tokens (+1 pad page), ≤{} sequences, ≤{} tokens/sequence, chunk {}, buckets {:?}{}",
            facts.total_blocks,
            facts.block_size,
            policy.max_seqs,
            facts.max_request_tokens,
            policy.chunk,
            BUCKETS.iter().filter(|&&b| b <= policy.max_seqs).collect::<Vec<_>>(),
            if policy.eager { ", eager" } else { ", graphs captured per bucket on first use" }
        );
        Ok((
            KernScheduler {
                rt: Rt(rt),
                pad,
                policy,
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
            let worst = prompt + max_tokens;
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
                Err(Denied::Busy) => break, // wait for pages
                Err(Denied::ExceedsRow { limit }) => {
                    ledger.reject(id, RejectReason::ContextLength { prompt_tokens: prompt, max_tokens, limit });
                    self.waiting.pop_front();
                    continue;
                }
                Err(Denied::ExceedsPool) => {
                    ledger.reject(id, RejectReason::KvBudget { prompt_tokens: prompt, worst_case_tokens: worst });
                    self.waiting.pop_front();
                    continue;
                }
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
            // Everything but the last prompt token goes through `prefill`
            // (state only); the last one is the first decode step's input,
            // whose logits yield the first generated token.
            self.prefill(&pages, &ids[..prompt - 1])?;
            self.stat_prefill_ns += t0.elapsed().as_nanos();
            self.stat_prefill_tokens += (prompt - 1) as u64;
            budget_used += prompt - 1;
            debug!(
                "admitted {id}: {prompt} prompt tokens, max_tokens {max_tokens}, {} pages, prefill {:?}",
                pages.pages(),
                t0.elapsed()
            );
            self.running.push(Seq {
                id,
                pos: prompt - 1,
                next: *q.request.prompt_tokens.last().unwrap(),
                generated: 0,
                max_tokens,
                ignore_eos: q.request.params.ignore_eos,
                pages,
                prompt_len: prompt,
                admitted: t0,
            });
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
    /// a sequence holding `pages`.
    fn prefill(&mut self, pages: &Lease, ids: &[i64]) -> Result<()> {
        let chunk = self.policy.chunk;
        let mut row = Vec::new();
        pages.extend_row("block_table", &mut row)?;
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
            rt.write_input_at("block_table", &le_i32(&row), &env)?;
            if !self.policy.eager && c == chunk {
                if !rt.is_captured("prefill", &env) {
                    let t = Instant::now();
                    rt.capture("prefill", &env)?;
                    info!("captured `prefill` at tokens={c} ({:?})", t.elapsed());
                }
                rt.run_captured("prefill", &env)?;
            } else {
                rt.run("prefill", &env)?;
            }
            pos += c;
        }
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
        let mut table = Vec::new();
        for s in &self.running {
            token_ids.push(s.next as i64);
            positions.push(s.pos as i64);
            slots.push(s.pages.slot(s.pos));
            seq_lens.push(s.pos as i32 + 1);
            s.pages.extend_row("block_table", &mut table)?;
        }
        for _ in n..b {
            token_ids.push(0);
            positions.push(0);
            slots.push(pad_slot);
            seq_lens.push(1);
            self.pad.extend_row("block_table", &mut table)?;
        }
        let cu: Vec<i32> = (0..=b as i32).collect();
        let program = if b == 1 { "decode" } else { "decode_batch" };
        let t0 = Instant::now();
        let rt = &mut self.rt.0;
        rt.write_input_at("token_ids", &le_i64(&token_ids), &env)?;
        rt.write_input_at("positions", &le_i64(&positions), &env)?;
        rt.write_input_at("slot_mapping", &le_i64(&slots), &env)?;
        rt.write_input_at("seq_lens", &le_i32(&seq_lens), &env)?;
        rt.write_input_at("cu_seqlens_q", &le_i32(&cu), &env)?;
        rt.write_input_at("block_table", &le_i32(&table), &env)?;
        if self.policy.eager {
            rt.run(program, &env)?;
        } else {
            if !rt.is_captured(program, &env) {
                let t = Instant::now();
                rt.capture(program, &env)?;
                info!("captured `{program}` at seqs={b} ({:?})", t.elapsed());
            }
            rt.run_captured(program, &env)?;
        }
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
        info!(
            "{} running, {} waiting, {}/{} pages | decode {} steps, {:.2} ms/step, {:.0} tok/s | prefill {} tokens, {:.0} tok/s",
            self.running.len(),
            self.waiting.len(),
            self.kv_used(),
            self.kv_total(),
            self.stat_steps,
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
        self.decode(ledger)?;
        self.log_stats();
        Ok(())
    }

    fn metrics(&self) -> SchedulerMetrics {
        SchedulerMetrics {
            kv_used_blocks: self.kv_used() as u64,
            kv_total_blocks: self.kv_total() as u64,
            num_running_reqs: self.running.len() as u64,
            num_waiting_reqs: self.waiting.len() as u64,
            spec_decode: None,
        }
    }
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
