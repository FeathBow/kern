//! Caller-side contract for the qwen3-4b manifests: which input buffers
//! exist and what goes in them for a chunked-prefill call and a bs=1
//! decode step. The runtime is model-agnostic; this is the one place that
//! knows there is a single sequence, holding one [`Lease`] of the runtime's
//! token slots for its whole life. `kern-run` (generation) and
//! `kern-attest` (A/B evidence) both drive the runtime through it.

#![forbid(unsafe_code)]

pub mod attest;
pub mod config;
pub mod run;

use std::collections::BTreeMap;

use anyhow::{bail, ensure, Result};
use kern_manifest::types::{Arg, BufferKind, Dim, Manifest};
use kern_runtime::{Lease, Runtime};

/// Default stop tokens (Qwen3 <|endoftext|>, <|im_end|>) for raw
/// (template-free) completion; `kern-run --stop-tokens` overrides.
pub const STOP_TOKENS: [i64; 2] = [151643, 151645];

/// Whether `prefill` produces `next_token` itself. A manifest whose prefill
/// arithmetic differs from decode's (hybrid GDN models: chunked FLA kernels
/// vs the recurrent kernel) must run the last prompt token through prefill
/// to match the reference; the driver then prefills every prompt token and
/// takes the first generated token from the prefill call.
pub fn prefill_emits_next_token(m: &Manifest) -> bool {
    m.programs.get("prefill").is_some_and(|calls| {
        calls.iter().any(|c| c.args.iter().any(|a| matches!(a, Arg::Buf { buf, .. } if buf == "next_token")))
    })
}

/// Programs this driver knows how to stage. A manifest may declare others;
/// the driver can't produce a workload for them. `decode_batch` takes the
/// same single-sequence inputs as `decode` at `seqs=1`, so the decode
/// workload alternates between the two when both exist.
pub const DRIVEN: [&str; 3] = ["prefill", "decode", "decode_batch"];
/// The decode-step programs of `DRIVEN`, in the order the workload rotates
/// through those a manifest declares.
pub const DECODE_LIKE: [&str; 2] = ["decode", "decode_batch"];
/// The var a prefill call is sized by.
pub const TOKENS: &str = "tokens";

pub fn le_bytes_i64(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn le_bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn i64_from_le(b: &[u8]) -> Vec<i64> {
    b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Var env of a single-sequence call: `tokens` rows of one sequence. A
/// manifest without a `seqs` var ignores the extra key (the runtime only
/// looks up the vars it declares).
pub fn env(tokens: u64) -> BTreeMap<String, u64> {
    BTreeMap::from([("tokens".to_string(), tokens), ("seqs".to_string(), 1)])
}

/// First element of an i64 output buffer (the buffer may be allocated for
/// more rows than this single-sequence caller uses).
pub fn first_i64(bytes: &[u8]) -> i64 {
    i64::from_le_bytes(bytes[..8].try_into().unwrap())
}

/// Every line table gets this sequence's lines in every column; a wide
/// table `[lines, seqs, w]` carries the line in entry `col` of each cell
/// and the null line 0 in the rest.
fn stage_lines(rt: &mut Runtime, lease: &Lease, col: usize) -> Result<()> {
    let tables: Vec<String> = rt.seq_tables().map(str::to_string).collect();
    for name in tables {
        let seqs = match rt.manifest.buffers[&name].shape.as_slice() {
            [Dim::Const(_)] => 1,
            [Dim::Const(_), Dim::Var(v)] | [Dim::Const(_), Dim::Var(v), Dim::Const(_)] => rt.manifest.vars[v].max,
            s => bail!("unexpected {name} shape {s:?}"),
        };
        let w = lease.seq_width(&name)?;
        ensure!(col < w, "`{name}`: column {col} of a {w}-wide line table");
        let mut table = Vec::new();
        for r in 0..lease.seq_lines(&name)? {
            let line = lease.seq_line(&name, r)?;
            for _ in 0..seqs {
                table.extend((0..w).map(|j| if j == col { line } else { 0 }));
            }
        }
        rt.write_input(&name, &le_bytes_i32(&table))?;
    }
    Ok(())
}

/// A runtime plus the single sequence: its token slots and position cursor.
pub struct Caller {
    pub rt: Runtime,
    /// The sequence's slots: as many as one page-table row (or the whole
    /// state) holds, leased once for the caller's life.
    lease: Lease,
    /// Tokens already in the KV state (next slot to fill).
    pub pos: i64,
}

impl Caller {
    /// Leases the sequence's slots and writes its row into every page table
    /// once (a speculative manifest has one per paged state, e.g.
    /// `draft_block_table`). A 2-D table (`[seqs, n]`) gets the row in every
    /// slot: this caller uses row 0, but every row must hold valid page ids.
    /// Line tables of a per-sequence state (`[lines, seqs]`) likewise get
    /// this sequence's lines in every column.
    pub fn new(mut rt: Runtime) -> Result<Caller> {
        ensure!(rt.manifest.buffers.contains_key("block_table"), "manifest has no `block_table` input");
        let lease = rt.lease(rt.max_seq_tokens().min(rt.capacity() as usize))?;
        let tables: Vec<String> = rt.page_tables().map(str::to_string).collect();
        for name in tables {
            let rows = match rt.manifest.buffers[&name].shape.as_slice() {
                [Dim::Const(_)] => 1,
                [Dim::Var(v), Dim::Const(_)] => rt.manifest.vars[v].max,
                s => bail!("unexpected {name} shape {s:?}"),
            };
            let mut table = Vec::new();
            for _ in 0..rows {
                lease.extend_row(&name, &mut table)?;
            }
            rt.write_input(&name, &le_bytes_i32(&table))?;
        }
        stage_lines(&mut rt, &lease, 0)?;
        // A recurrent state resumed from the last accepted row: 1 (the
        // anchor) everywhere until the spec driver says otherwise.
        if rt.manifest.buffers.get("num_accepted_tokens").is_some_and(|b| b.kind == BufferKind::Input) {
            let n = rt.manifest.buffers["num_accepted_tokens"]
                .shape
                .iter()
                .map(|d| match d {
                    Dim::Const(c) => *c,
                    Dim::Var(v) => rt.manifest.vars[v].max,
                })
                .product::<u64>() as usize;
            rt.write_input("num_accepted_tokens", &le_bytes_i32(&vec![1; n]))?;
        }
        Ok(Caller { rt, lease, pos: 0 })
    }

    /// Re-stage every line table with the line in column `col` of each
    /// cell (wide `[lines, seqs, w]` tables; the others ignore `col`).
    pub fn set_line_column(&mut self, col: usize) -> Result<()> {
        stage_lines(&mut self.rt, &self.lease, col)
    }

    /// Token slots the sequence can hold.
    pub fn limit(&self) -> usize {
        self.lease.tokens()
    }

    /// Stage one call's rows: `ids` at the cursor as one causal sequence
    /// (`token_ids` / `positions` / `slot_mapping` / `seq_lens` /
    /// `cu_seqlens_q`); does not advance. Returns the var env for the call.
    pub fn stage(&mut self, ids: &[i64]) -> Result<BTreeMap<String, u64>> {
        let c = ids.len();
        let pos = self.pos as usize;
        let positions: Vec<i64> = (self.pos..self.pos + c as i64).collect();
        let e = env(c as u64);
        self.rt.write_input_at("token_ids", &le_bytes_i64(ids), &e)?;
        self.rt.write_input_at("slot_mapping", &le_bytes_i64(&self.lease.slots(pos..pos + c)), &e)?;
        self.rt.write_input_at("seq_lens", &le_bytes_i32(&[(pos + c) as i32]), &e)?;
        // Optional: a NoPE model has no positions, a decode-only manifest no
        // query offsets.
        if self.rt.manifest.buffers.contains_key("positions") {
            self.rt.write_input_at("positions", &le_bytes_i64(&positions), &e)?;
        }
        if self.rt.manifest.buffers.contains_key("cu_seqlens_q") {
            self.rt.write_input_at("cu_seqlens_q", &le_bytes_i32(&[0, c as i32]), &e)?;
        }
        Ok(e)
    }

    /// Reset the cursor (a new prompt reuses the slots from position 0).
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Stage a `prefill` call over `ids` at the cursor.
    pub fn stage_prefill(&mut self, ids: &[i64]) -> Result<BTreeMap<String, u64>> {
        self.stage(ids)
    }

    /// Stage one `decode` step of token `tok` at the cursor.
    pub fn stage_decode(&mut self, tok: i64) -> Result<BTreeMap<String, u64>> {
        self.stage(&[tok])
    }

    pub fn advance(&mut self, n: u64) {
        self.pos += n as i64;
    }

    /// Chunked prefill of `ids` (eager or graph-captured full chunks),
    /// advancing the cursor past them. State writes only — call `decode` on
    /// the following token for logits — unless the manifest's prefill emits
    /// `next_token` (see [`prefill_emits_next_token`]), in which case the
    /// last chunk's output is the first generated token.
    pub fn prefill(&mut self, ids: &[i64], chunk: u64, eager: bool) -> Result<bool> {
        let chunk = chunk.min(self.rt.manifest.vars["tokens"].max).max(1);
        let mut captured = false;
        let mut i = 0usize;
        while i < ids.len() {
            let c = ((ids.len() - i) as u64).min(chunk) as usize;
            let e = self.stage_prefill(&ids[i..i + c])?;
            if !eager && c as u64 == chunk {
                if !captured {
                    self.rt.capture("prefill", &e)?;
                    captured = true;
                }
                self.rt.run_captured("prefill", &e)?;
            } else {
                self.rt.run("prefill", &e)?;
            }
            self.advance(c as u64);
            i += c;
        }
        Ok(captured)
    }

    /// Vocabulary size as declared by `token_ids`' domain (1000 if none).
    pub fn vocab(&self) -> u64 {
        let m = &self.rt.manifest;
        m.buffers["token_ids"]
            .domain
            .as_ref()
            .and_then(|d| d.resolve(m, &env(1), &self.rt.provision()).ok())
            .and_then(|r| r.hi)
            .map_or(1000, |hi| hi as u64 + 1)
    }

    pub fn next_token(&self) -> Result<i64> {
        Ok(first_i64(&self.rt.read_output("next_token")?))
    }
}
