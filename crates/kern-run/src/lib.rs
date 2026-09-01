//! Caller-side contract for the qwen3-4b manifests: which input buffers
//! exist and what goes in them for a chunked-prefill call and a bs=1
//! decode step. The runtime is model-agnostic; this is the one place that
//! knows `slot_mapping` is position-identity and `seq_lens` is a single
//! sequence. `kern-run` (generation) and `kern-attest` (A/B evidence) both
//! drive the runtime through it.

pub mod attest;
pub mod config;
pub mod run;

use std::collections::BTreeMap;

use anyhow::{bail, ensure, Result};
use kern_manifest::types::{Arg, Dim, Manifest};
use kern_runtime::Runtime;

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
        calls.iter().any(|c| {
            c.args.iter().any(|a| matches!(a, Arg::Buf { buf, .. } if buf == "next_token"))
        })
    })
}

/// Programs this driver knows how to stage. A manifest may declare others;
/// the driver can't produce a workload for them.
pub const DRIVEN: [&str; 2] = ["prefill", "decode"];
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

pub fn env(tokens: u64) -> BTreeMap<String, u64> {
    BTreeMap::from([("tokens".to_string(), tokens)])
}

/// A runtime plus the single sequence's position cursor.
pub struct Caller {
    pub rt: Runtime,
    /// Tokens already in the KV state (next slot to fill).
    pub pos: i64,
}

impl Caller {
    /// Writes the one-time identity page table (pages allocated linearly by
    /// position). Entries past the state's capacity are never read, but
    /// they must still be valid page ids — the buffer's domain says so.
    pub fn new(mut rt: Runtime) -> Result<Caller> {
        // Every `*block_table` input (a speculative manifest has one per
        // paged state, e.g. `draft_block_table`); page size from its domain.
        let tables: Vec<String> = rt
            .manifest
            .buffers
            .keys()
            .filter(|n| n.ends_with("block_table"))
            .cloned()
            .collect();
        ensure!(tables.iter().any(|n| n == "block_table"), "manifest has no `block_table` input");
        for name in tables {
            let b = &rt.manifest.buffers[&name];
            let n_entries = match b.shape.as_slice() {
                [Dim::Const(n)] => *n as i64,
                s => bail!("unexpected {name} shape {s:?}"),
            };
            let stride = b.domain.as_ref().map_or(16, |d| d.stride) as i64;
            let n_pages = (rt.capacity() as i64 / stride).max(1);
            let table: Vec<i32> = (0..n_entries).map(|i| i.min(n_pages - 1) as i32).collect();
            rt.write_input(&name, &le_bytes_i32(&table))?;
        }
        Ok(Caller { rt, pos: 0 })
    }

    /// Reset the cursor (a new prompt reuses the slots from position 0).
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Stage the inputs for a `prefill` call over `ids` at the cursor; does
    /// not advance. Returns the var env for the call.
    pub fn stage_prefill(&mut self, ids: &[i64]) -> Result<BTreeMap<String, u64>> {
        let c = ids.len() as u64;
        let positions: Vec<i64> = (self.pos..self.pos + c as i64).collect();
        let e = env(c);
        self.rt.write_input_at("token_ids", &le_bytes_i64(ids), &e)?;
        self.rt.write_input_at("positions", &le_bytes_i64(&positions), &e)?;
        self.rt.write_input_at("slot_mapping", &le_bytes_i64(&positions), &e)?;
        self.rt.write_input_at("seq_lens", &le_bytes_i32(&[self.pos as i32 + c as i32]), &e)?;
        self.rt.write_input_at("cu_seqlens_q", &le_bytes_i32(&[0, c as i32]), &e)?;
        Ok(e)
    }

    /// Stage the inputs for one `decode` step of token `tok` at the cursor;
    /// does not advance.
    pub fn stage_decode(&mut self, tok: i64) -> Result<BTreeMap<String, u64>> {
        let e = env(1);
        self.rt.write_input_at("token_ids", &le_bytes_i64(&[tok]), &e)?;
        self.rt.write_input_at("positions", &le_bytes_i64(&[self.pos]), &e)?;
        self.rt.write_input_at("slot_mapping", &le_bytes_i64(&[self.pos]), &e)?;
        self.rt.write_input_at("seq_lens", &le_bytes_i32(&[self.pos as i32 + 1]), &e)?;
        self.rt.write_input_at("cu_seqlens_q", &le_bytes_i32(&[0, 1]), &e)?;
        Ok(e)
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
        m.buffers["token_ids"].domain.as_ref()
            .and_then(|d| d.resolve(m, &env(1), self.rt.capacity()).ok())
            .and_then(|r| r.hi)
            .map_or(1000, |hi| hi as u64 + 1)
    }

    pub fn next_token(&self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.rt.read_output("next_token")?.try_into().unwrap()))
    }
}
