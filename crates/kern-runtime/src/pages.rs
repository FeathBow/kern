//! Token-slot and sequence-slot ownership of the states.
//!
//! The runtime provisions every paged state as `capacity` tokens and every
//! per-sequence state as `seq_slots` slots; the kernels address them
//! through the manifest's tables — inputs whose domain `index_into`s a
//! state: page tables (`stride` tokens per entry) and slot lists (stride 1)
//! over a paged state, line tables (`stride` bytes per line) over a
//! per-sequence one. Which slot holds what is the caller's business, but
//! the only way to name a slot is a [`Lease`]: pages and a sequence slot
//! come out of the pool as a lease, slots, table rows and line indices are
//! computed from it, and everything goes back when it drops. Nothing can
//! free a page twice, free a page it never leased, or address a slot past
//! its lease.
//!
//! Pages are in the runtime's page unit — the lcm of every page table's
//! stride — so one lease serves every paged state at once (a 16-token
//! draft table sees 49 entries per 784-token page of the target's table).
//! A lease is all-or-nothing: a caller takes the pages its worst case
//! needs and holds them, so the pool never fragments.
//!
//! A per-sequence state is `seq_slots` slots of `bytes_per_seq`; slot 0 is
//! never leased (a kernel may read line index 0 as the null line), the
//! rest go one per lease. A line table is shaped `[lines, seqs]` (or
//! `[lines]`): row `r` names, for every sequence of the batch, line `r` of
//! its slot — `slot × lines_per_slot + r`. A wide table `[lines, seqs, w]`
//! has `w` entries per (line, sequence) cell for kernels that take a
//! per-sequence list of lines: the caller puts the line in one of them (the
//! contract of the program says which) and 0, the null line, in the rest.
//!
//! # Checkpoints
//!
//! A [`Checkpoint`] is the first `len` tokens of a sequence kept after the
//! sequence is gone, so a later sequence with the same prefix starts at
//! `len` instead of 0: the pages holding those tokens and, when the
//! manifest has per-sequence states, a slot holding the recurrent state as
//! it was after token `len - 1`. Shared pages live in chains — a page and
//! the chain before it, reference-counted — so a checkpoint is one node
//! however deep it sits, a sequence's checkpoints at every page share one
//! chain, and a page returns to the pool when the last lease or checkpoint
//! holding its node drops. A paged state alone makes a checkpoint free — a
//! node, no bytes move — so a caller leaves one at every page boundary; a
//! recurrent state makes it cost a slot, so a caller leaves one where a
//! request ends ([`Pool::retire`] hands the finished sequence's slot over
//! without a copy) and only there.
//!
//! Restoring ([`Pool::restore`]) shares the checkpoint's whole pages, copies
//! its last page when `len` ends inside one (the new sequence appends into
//! that page, the checkpoint keeps its own), copies the state slot, and
//! hands out a lease whose first `len` positions are read-only — the lease
//! refuses to name a slot inside its prefix. Positions past a checkpoint's
//! `len` in its last page belong to whoever writes them next; a checkpoint
//! claims positions, not the page's tail. The pool decides all of this on
//! the host and returns the byte moves as [`Copies`]; the runtime is the
//! shell that runs them on the stream.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use kern_manifest::types::{BufferKind, Dim, Manifest};

use crate::error::{bail, Result};

/// A page table input: `stride` tokens per entry, `width` entries per row.
struct Table {
    stride: u64,
    width: usize,
}

/// A line table input over a per-sequence state: `rows` lines per
/// sequence it names, out of `per_slot` lines a slot holds, `width`
/// entries per (line, sequence) cell.
struct SeqTable {
    rows: usize,
    per_slot: i32,
    width: usize,
}

/// Shared between the runtime and every live lease and checkpoint so a
/// drop returns pages directly. One caller thread; the mutexes are for
/// `Send`, never contended.
pub struct Pool {
    /// Tokens per page.
    unit: u64,
    total: usize,
    /// Pages one sequence may hold: what the narrowest table row fits.
    max_pages: usize,
    tables: BTreeMap<String, Table>,
    free: Mutex<Vec<i32>>,
    /// Sequence slots (0 when no state is per-sequence), slot 0 reserved.
    slots: usize,
    seq_tables: BTreeMap<String, SeqTable>,
    free_slots: Mutex<Vec<i32>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The page tables: every input whose domain `index_into`s a paged state
/// (an index a kernel writes — a carry — is the manifest's business, not
/// the host's).
fn tables(m: &Manifest) -> BTreeMap<String, Table> {
    m.buffers
        .iter()
        .filter_map(|(name, b)| {
            if b.kind != BufferKind::Input {
                return None;
            }
            let d = b.domain.as_ref()?;
            if !m.states.get(d.index_into.as_deref()?).is_some_and(|s| !s.is_per_seq()) {
                return None;
            }
            let Some(Dim::Const(width)) = b.shape.last() else { return None };
            Some((name.clone(), Table { stride: d.stride.max(1), width: *width as usize }))
        })
        .collect()
}

/// The line tables: every input whose domain `index_into`s a per-sequence
/// state, shaped `[lines]`, `[lines, seqs]` or `[lines, seqs, w]`.
fn seq_tables(m: &Manifest) -> Result<BTreeMap<String, SeqTable>> {
    let mut out = BTreeMap::new();
    for (name, b) in &m.buffers {
        if b.kind != BufferKind::Input {
            continue;
        }
        let Some(d) = b.domain.as_ref() else { continue };
        let Some(st) = d.index_into.as_deref().and_then(|s| m.states.get(s)) else { continue };
        if !st.is_per_seq() {
            continue;
        }
        let (rows, width) = match b.shape.as_slice() {
            [Dim::Const(rows)] | [Dim::Const(rows), Dim::Var(_)] => (*rows, 1),
            [Dim::Const(rows), Dim::Var(_), Dim::Const(w)] => (*rows, *w as usize),
            s => bail!(
                Manifest,
                "`{name}` indexes a per-sequence state: expected shape [lines], [lines, seqs] or [lines, seqs, w], got {s:?}"
            ),
        };
        let per_slot = st.bytes_per_seq / d.stride.max(1);
        if rows > per_slot {
            bail!(Manifest, "`{name}` names {rows} lines per sequence, the state holds {per_slot}");
        }
        out.insert(name.clone(), SeqTable { rows: rows as usize, per_slot: per_slot as i32, width });
    }
    Ok(out)
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn lcm(a: u64, b: u64) -> u64 {
    a / gcd(a, b) * b
}

/// The page unit: the lcm of every page table's stride (a per-sequence
/// state's stride is bytes per line, not tokens: not a page).
pub fn page_unit(m: &Manifest) -> u64 {
    m.buffers
        .values()
        .filter_map(|b| b.domain.as_ref())
        .filter(|d| d.index_into.as_deref().and_then(|s| m.states.get(s)).is_some_and(|s| !s.is_per_seq()))
        .map(|d| d.stride.max(1))
        .fold(1u64, lcm)
}

/// Tokens one sequence can hold, in whole pages of `unit`: what the
/// narrowest page-table row references. `None` when nothing is paged.
pub(crate) fn row_tokens(m: &Manifest, unit: u64) -> Option<u64> {
    tables(m).values().map(|t| t.width as u64 * t.stride / unit * unit).min()
}

/// Device copies that realize a pool decision, in page and slot numbers:
/// `pages` are (from, to) for every paged state, `slot` is (from, to) for
/// every per-sequence state. Empty when nothing moves.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Copies {
    pub pages: Vec<(i32, i32)>,
    pub slot: Option<(i32, i32)>,
}

/// A shared page and the chain before it. Holding a node holds every page
/// up to it; the page returns when its last holder lets go.
struct Node {
    page: i32,
    parent: Option<Arc<Node>>,
    pool: Arc<Pool>,
}

impl Drop for Node {
    /// Unwind the chain in a loop: a recursive drop of a 65k-page chain
    /// would overflow the stack.
    fn drop(&mut self) {
        self.pool.release(&[self.page], None);
        let mut next = self.parent.take();
        while let Some(n) = next {
            match Arc::try_unwrap(n) {
                Ok(mut node) => next = node.parent.take(),
                Err(_) => break,
            }
        }
    }
}

/// The pages of a chain, root first.
fn chain_pages(chain: &Option<Arc<Node>>) -> Vec<i32> {
    let mut out = Vec::new();
    let mut cur = chain.as_ref();
    while let Some(n) = cur {
        out.push(n.page);
        cur = n.parent.as_ref();
    }
    out.reverse();
    out
}

/// The node `depth` pages up the chain from `chain` (0: `chain` itself).
fn ancestor(chain: &Option<Arc<Node>>, depth: usize) -> Option<Arc<Node>> {
    let mut cur = chain.clone();
    for _ in 0..depth {
        cur = cur.and_then(|n| n.parent.clone());
    }
    cur
}

impl Pool {
    /// The pool of `m` at `capacity` tokens, in whole pages of
    /// [`page_unit`].
    pub fn new(m: &Manifest, capacity: u64) -> Result<Pool> {
        let unit = page_unit(m);
        let total = (capacity / unit) as usize;
        let tables = tables(m);
        let max_pages = row_tokens(m, unit).map_or(total, |t| (t / unit) as usize);
        let slots = if m.states.values().any(|s| s.is_per_seq()) { m.seq_slots() as usize } else { 0 };
        Ok(Pool {
            unit,
            total,
            max_pages,
            tables,
            free: Mutex::new((0..total as i32).collect()),
            slots,
            seq_tables: seq_tables(m)?,
            free_slots: Mutex::new((1..slots as i32).collect()),
        })
    }

    pub fn unit(&self) -> u64 {
        self.unit
    }

    pub fn total(&self) -> usize {
        self.total
    }

    /// Pages held by a lease or a checkpoint.
    pub fn used(&self) -> usize {
        self.total - lock(&self.free).len()
    }

    pub fn max_seq_tokens(&self) -> usize {
        self.max_pages * self.unit as usize
    }

    pub fn tables(&self) -> impl Iterator<Item = &str> {
        self.tables.keys().map(String::as_str)
    }

    /// Sequence slots provisioned (0 when no state is per-sequence).
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Sequence slots held by a lease or a checkpoint.
    pub fn slots_used(&self) -> usize {
        self.slots.saturating_sub(1) - lock(&self.free_slots).len()
    }

    pub fn seq_tables(&self) -> impl Iterator<Item = &str> {
        self.seq_tables.keys().map(String::as_str)
    }

    fn pages_for(&self, tokens: usize) -> std::result::Result<usize, Denied> {
        let need = tokens.div_ceil(self.unit as usize);
        if need > self.max_pages {
            return Err(Denied::ExceedsRow { limit: self.max_seq_tokens() });
        }
        if need > self.total {
            return Err(Denied::ExceedsPool);
        }
        Ok(need)
    }

    /// `fresh` free pages and, when the manifest has per-sequence states,
    /// a slot; all or nothing.
    fn take(&self, fresh: usize) -> std::result::Result<(Vec<i32>, Option<i32>), Denied> {
        let mut free = lock(&self.free);
        let mut free_slots = lock(&self.free_slots);
        if fresh > free.len() || (self.slots > 0 && free_slots.is_empty()) {
            return Err(Denied::Busy);
        }
        let at = free.len() - fresh;
        let taken = free.split_off(at);
        let slot = if self.slots > 0 { free_slots.pop() } else { None };
        Ok((taken, slot))
    }

    fn release(&self, pages: &[i32], slot: Option<i32>) {
        lock(&self.free).extend_from_slice(pages);
        if let Some(s) = slot {
            lock(&self.free_slots).push(s);
        }
    }

    /// A fresh sequence: the pages `tokens` need and a sequence slot.
    pub fn lease(self: &Arc<Pool>, tokens: usize) -> std::result::Result<Lease, Denied> {
        let need = self.pages_for(tokens)?;
        let (pages, slot) = self.take(need)?;
        Ok(Lease { chain: None, shared: 0, pages, slot, prefix: 0, pool: Arc::clone(self) })
    }

    /// The first `len` tokens of `lease` as a checkpoint the lease's
    /// sequence keeps running past: its pages up to there become shared,
    /// its state slot (when the manifest has one) is copied into a fresh
    /// slot — the [`Copies`] say which. `len` is 1 to the lease's tokens.
    pub fn checkpoint(
        self: &Arc<Pool>,
        lease: &mut Lease,
        len: usize,
    ) -> std::result::Result<(Checkpoint, Copies), Denied> {
        assert!(len >= 1 && len <= lease.tokens(), "checkpoint of {len} tokens out of a lease of {}", lease.tokens());
        let (_, slot) = self.take(0)?;
        let chain = lease.share(len.div_ceil(self.unit as usize));
        let copies = Copies { pages: Vec::new(), slot: lease.slot.zip(slot) };
        Ok((Checkpoint { len, chain, slot, pool: Arc::clone(self) }, copies))
    }

    /// The first `len` tokens of a finished sequence as a checkpoint:
    /// the lease's pages past `len` return, the rest and its state slot
    /// move over as they are. Nothing is copied.
    pub fn retire(self: &Arc<Pool>, mut lease: Lease, len: usize) -> Checkpoint {
        assert!(len >= 1 && len <= lease.tokens(), "retiring {len} tokens out of a lease of {}", lease.tokens());
        let chain = lease.share(len.div_ceil(self.unit as usize));
        Checkpoint { len, chain, slot: lease.slot.take(), pool: Arc::clone(self) }
    }

    /// A sequence continuing from `cp` with room for `tokens` (more than
    /// the checkpoint's `len`): the checkpoint's whole pages shared, a copy
    /// of the page `len` ends inside when it does, fresh pages for the
    /// rest, a fresh slot with the checkpoint's state copied in. The lease
    /// names positions from `len` on.
    pub fn restore(self: &Arc<Pool>, cp: &Checkpoint, tokens: usize) -> std::result::Result<(Lease, Copies), Denied> {
        assert!(tokens > cp.len, "restoring {} tokens into room for {tokens}", cp.len);
        let need = self.pages_for(tokens)?;
        let unit = self.unit as usize;
        let full = cp.len / unit;
        let partial = if cp.len.is_multiple_of(unit) { None } else { Some(cp.chain.page) };
        let (fresh, slot) = self.take(need - full)?;
        // The chain through the whole pages: the checkpoint's own node, or
        // its parent when the checkpoint's last page is the partial one.
        let chain = if partial.is_some() { cp.chain.parent.clone() } else { Some(Arc::clone(&cp.chain)) };
        let mut pages = chain_pages(&chain);
        pages.extend_from_slice(&fresh);
        let copies = Copies { pages: partial.map(|p| (p, fresh[0])).into_iter().collect(), slot: cp.slot.zip(slot) };
        Ok((Lease { chain, shared: full, pages, slot, prefix: cp.len, pool: Arc::clone(self) }, copies))
    }
}

/// Why [`Runtime::lease`](crate::Runtime::lease) said no.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Denied {
    /// More tokens than one page-table row can reference; never fits.
    ExceedsRow { limit: usize },
    /// More pages than the pool has, even empty; never fits.
    ExceedsPool,
    /// Fits, but not right now: pages or sequence slots all held.
    Busy,
}

impl fmt::Display for Denied {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Denied::ExceedsRow { limit } => write!(f, "longer than a page-table row ({limit} tokens)"),
            Denied::ExceedsPool => write!(f, "more pages than the state capacity holds"),
            Denied::Busy => write!(f, "pages or sequence slots busy"),
        }
    }
}

impl std::error::Error for Denied {}

/// Pages and a sequence slot leased to one sequence: the only handle to
/// its token slots and its per-sequence state. Dropping it returns them
/// to the runtime. The first `shared` pages are held through a chain
/// (checkpoints hold them too), the rest are the lease's own. A lease
/// restored from a checkpoint starts with `prefix` positions already
/// filled; it never names a slot inside them.
pub struct Lease {
    chain: Option<Arc<Node>>,
    shared: usize,
    /// Every page in order: the chain's, then the lease's own.
    pages: Vec<i32>,
    slot: Option<i32>,
    prefix: usize,
    pool: Arc<Pool>,
}

impl Lease {
    /// The chain through the first `keep` pages, moving own pages into
    /// nodes as needed.
    fn share(&mut self, keep: usize) -> Arc<Node> {
        while self.shared < keep {
            let page = self.pages[self.shared];
            self.chain = Some(Arc::new(Node { page, parent: self.chain.take(), pool: Arc::clone(&self.pool) }));
            self.shared += 1;
        }
        ancestor(&self.chain, self.shared - keep).expect("keep is at least 1")
    }

    /// Pages held.
    pub fn pages(&self) -> usize {
        self.pages.len()
    }

    /// Token slots held (whole pages, so at least what was asked for).
    pub fn tokens(&self) -> usize {
        self.pages.len() * self.pool.unit as usize
    }

    /// Positions already filled when the lease was handed out: 0 for a
    /// fresh sequence, the checkpoint's length for a restored one.
    pub fn prefix(&self) -> usize {
        self.prefix
    }

    /// The token slot of position `pos` of the sequence, to write into;
    /// `pos` is past the shared prefix.
    pub fn slot(&self, pos: usize) -> i64 {
        assert!(pos >= self.prefix, "position {pos} is inside the shared prefix of {} tokens", self.prefix);
        let unit = self.pool.unit as usize;
        let page = *self.pages.get(pos / unit).expect("position past the lease") as i64;
        page * unit as i64 + (pos % unit) as i64
    }

    /// The token slots of consecutive positions (a `slot_mapping` list).
    pub fn slots(&self, positions: Range<usize>) -> Vec<i64> {
        positions.map(|p| self.slot(p)).collect()
    }

    /// Append the sequence's row of page table `table`: one entry per
    /// `stride` tokens of every page held, then the first entry repeated to
    /// the row's width (entries past the sequence length are never
    /// dereferenced, but the domain wants valid page ids in them).
    pub fn extend_row(&self, table: &str, out: &mut Vec<i32>) -> Result<()> {
        let Some(t) = self.pool.tables.get(table) else {
            bail!(Api, "`{table}` is not a page table of this manifest");
        };
        let per_page = (self.pool.unit / t.stride) as i32;
        let start = out.len();
        for &p in &self.pages {
            out.extend((0..per_page).map(|k| p * per_page + k));
        }
        let fill = out.get(start).copied().unwrap_or(0);
        out.resize(start + t.width, fill);
        Ok(())
    }

    /// The sequence slot held in every per-sequence state (`None` when
    /// the manifest has none). Slot 0 is never handed out.
    pub fn seq_slot(&self) -> Option<i32> {
        self.slot
    }

    /// Byte range of this sequence's slot in a per-sequence state of
    /// `bytes_per_seq`.
    pub(crate) fn seq_bytes(&self, bytes_per_seq: u64) -> Option<Range<usize>> {
        let s = self.slot? as u64;
        Some((s * bytes_per_seq) as usize..((s + 1) * bytes_per_seq) as usize)
    }

    /// Line `row` of this sequence in line table `table`: the index its
    /// entry `[row, i]` holds when the sequence is column `i` of a batch.
    pub fn seq_line(&self, table: &str, row: usize) -> Result<i32> {
        let Some(t) = self.pool.seq_tables.get(table) else {
            bail!(Api, "`{table}` is not a line table of this manifest");
        };
        if row >= t.rows {
            bail!(Api, "`{table}` has {} lines per sequence, asked for line {row}", t.rows);
        }
        let Some(slot) = self.slot else {
            bail!(Api, "lease holds no sequence slot");
        };
        Ok(slot * t.per_slot + row as i32)
    }

    /// Lines per sequence line table `table` names.
    pub fn seq_lines(&self, table: &str) -> Result<usize> {
        match self.pool.seq_tables.get(table) {
            Some(t) => Ok(t.rows),
            None => bail!(Api, "`{table}` is not a line table of this manifest"),
        }
    }

    /// Entries per (line, sequence) cell of line table `table`: 1, or the
    /// `w` of a wide `[lines, seqs, w]` table.
    pub fn seq_width(&self, table: &str) -> Result<usize> {
        match self.pool.seq_tables.get(table) {
            Some(t) => Ok(t.width),
            None => bail!(Api, "`{table}` is not a line table of this manifest"),
        }
    }
}

impl fmt::Debug for Lease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lease({} pages, {} shared", self.pages.len(), self.shared)?;
        if let Some(s) = self.slot {
            write!(f, ", slot {s}")?;
        }
        write!(f, ", prefix {})", self.prefix)
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.pool.release(&self.pages[self.shared..], self.slot.take());
    }
}

/// The first `len` tokens of a sequence that is gone: the pages holding
/// them (shared with whoever else holds them) and, when the manifest has
/// per-sequence states, a slot with the state after those tokens. Made
/// by [`Pool::checkpoint`] / [`Pool::retire`], spent by [`Pool::restore`];
/// dropping it releases what it alone holds.
pub struct Checkpoint {
    len: usize,
    chain: Arc<Node>,
    slot: Option<i32>,
    pool: Arc<Pool>,
}

impl Checkpoint {
    /// Tokens the checkpoint holds; never 0.
    pub fn tokens(&self) -> usize {
        self.len
    }

    /// Pages held.
    pub fn pages(&self) -> usize {
        self.len.div_ceil(self.pool.unit as usize)
    }

    /// The sequence slot holding the state, when the manifest has one.
    pub fn seq_slot(&self) -> Option<i32> {
        self.slot
    }
}

impl fmt::Debug for Checkpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.slot {
            Some(s) => write!(f, "Checkpoint({} tokens, {} pages, slot {s})", self.len, self.pages()),
            None => write!(f, "Checkpoint({} tokens, {} pages)", self.len, self.pages()),
        }
    }
}

impl Drop for Checkpoint {
    fn drop(&mut self) {
        self.pool.release(&[], self.slot.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// kv paged in 16-token entries (row of 3), a draft state paged in
    /// 4-token entries (row of 16): page unit 16, 4 pages on 64 tokens.
    fn manifest() -> Manifest {
        Manifest::from_json(
            r#"{
            "schema_version": 3, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 2}},
            "states": {"kv": {"bytes_per_token": 1}, "draft_kv": {"bytes_per_token": 1}},
            "buffers": {
                "slot_mapping": {"kind": "input", "dtype": "i64", "shape": ["tokens"], "domain": {"index_into": "kv"}},
                "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 3], "domain": {"index_into": "kv", "stride": 16}},
                "draft_block_table": {"kind": "input", "dtype": "i32", "shape": [16], "domain": {"index_into": "draft_kv", "stride": 4}}
            },
            "modules": {}, "ops": {}, "programs": {}
        }"#,
        )
        .unwrap()
    }

    /// The same, plus a recurrent state of 3 lines of 8 bytes per
    /// sequence and its line table.
    fn hybrid() -> Manifest {
        Manifest::from_json(
            r#"{
            "schema_version": 3, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 2}},
            "states": {"kv": {"bytes_per_token": 1}, "gdn": {"bytes_per_seq": 24}},
            "buffers": {
                "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 3], "domain": {"index_into": "kv", "stride": 16}},
                "line_index": {"kind": "input", "dtype": "i32", "shape": [3, "seqs"], "domain": {"index_into": "gdn", "stride": 8}}
            },
            "modules": {}, "ops": {}, "programs": {}
        }"#,
        )
        .unwrap()
    }

    fn pool() -> Arc<Pool> {
        Arc::new(Pool::new(&manifest(), 64).unwrap())
    }

    #[test]
    fn geometry() {
        let p = pool();
        assert_eq!((p.total(), p.unit(), p.max_seq_tokens()), (4, 16, 48));
        assert_eq!(p.tables().collect::<Vec<_>>(), ["block_table", "draft_block_table"]);
        // The draft table's 16 entries × 4 tokens hold 4 pages; kv's 3 × 16 hold 3.
        assert_eq!(p.max_pages, 3);
        assert_eq!(p.slots(), 0);
        assert_eq!(p.lease(1).unwrap().seq_slot(), None);
        // Capacity rounds down to whole pages.
        assert_eq!(Pool::new(&manifest(), 70).unwrap().total(), 4);
    }

    #[test]
    fn drop_returns_pages() {
        let p = pool();
        let a = p.lease(17).unwrap(); // 2 pages
        let b = p.lease(1).unwrap();
        assert_eq!((a.pages(), a.tokens(), b.pages(), p.used()), (2, 32, 1, 3));
        drop(a);
        assert_eq!(p.used(), 1);
        drop(b);
        assert_eq!(p.used(), 0);
        let all: Vec<Lease> = (0..4).map(|_| p.lease(1).unwrap()).collect();
        let mut ids: Vec<i32> = all.iter().flat_map(|l| l.pages.clone()).collect();
        ids.sort();
        assert_eq!((ids, p.used()), (vec![0, 1, 2, 3], 4));
        drop(all);
        assert_eq!(p.used(), 0);
    }

    #[test]
    fn denials() {
        let p = pool();
        assert_eq!(p.lease(49).unwrap_err(), Denied::ExceedsRow { limit: 48 });
        let a = p.lease(48).unwrap();
        assert_eq!(p.lease(17).unwrap_err(), Denied::Busy);
        let _b = p.lease(16).unwrap();
        assert_eq!(p.lease(1).unwrap_err(), Denied::Busy);
        drop(a);
        assert!(p.lease(33).is_ok());
        // Two pages of capacity can never hold a three-page sequence.
        let small = Arc::new(Pool::new(&manifest(), 32).unwrap());
        assert_eq!(small.lease(48).unwrap_err(), Denied::ExceedsPool);
    }

    #[test]
    fn slots_and_rows() {
        let p = pool();
        let l = p.lease(20).unwrap(); // 2 pages
        let [p0, p1] = l.pages[..] else { panic!() };
        assert_eq!(l.slot(0), p0 as i64 * 16);
        assert_eq!(l.slot(15), p0 as i64 * 16 + 15);
        assert_eq!(l.slot(16), p1 as i64 * 16);
        assert_eq!(l.slots(15..17), [p0 as i64 * 16 + 15, p1 as i64 * 16]);
        let mut t = Vec::new();
        l.extend_row("block_table", &mut t).unwrap();
        assert_eq!(t, [p0, p1, p0]);
        let mut d = Vec::new();
        l.extend_row("draft_block_table", &mut d).unwrap();
        let mut want: Vec<i32> = (0..4).map(|k| p0 * 4 + k).chain((0..4).map(|k| p1 * 4 + k)).collect();
        want.resize(16, p0 * 4);
        assert_eq!(d, want);
        // Position 17's draft entry (row index 17/4) names the draft page its slot falls in.
        assert_eq!(d[17 / 4], (l.slot(17) / 4) as i32);
        assert!(l.extend_row("slot_mapping", &mut t).is_err());
        assert!(l.seq_line("block_table", 0).is_err());
    }

    #[test]
    #[should_panic(expected = "past the lease")]
    fn slot_past_lease() {
        let p = pool();
        p.lease(16).unwrap().slot(16);
    }

    #[test]
    fn seq_slots_and_lines() {
        let m = hybrid();
        // seqs 2 + pad + null: slots 0..4, three of them leasable.
        assert_eq!(m.seq_slots(), 4);
        let p = Arc::new(Pool::new(&m, 64).unwrap());
        assert_eq!((p.slots(), p.seq_tables().collect::<Vec<_>>()), (4, vec!["line_index"]));
        let a = p.lease(16).unwrap();
        let b = p.lease(16).unwrap();
        let c = p.lease(16).unwrap();
        let (sa, sb, sc) = (a.seq_slot().unwrap(), b.seq_slot().unwrap(), c.seq_slot().unwrap());
        let mut got = vec![sa, sb, sc];
        got.sort();
        assert_eq!((got, p.slots_used()), (vec![1, 2, 3], 3));
        // A fourth page is free but no slot is: busy, not exhausted.
        assert_eq!(p.lease(16).unwrap_err(), Denied::Busy);
        // Line r of a slot: slot × 3 + r; the null line 0 belongs to no lease.
        assert_eq!(a.seq_line("line_index", 2).unwrap(), sa * 3 + 2);
        assert_eq!(a.seq_lines("line_index").unwrap(), 3);
        assert!(a.seq_line("line_index", 3).is_err());
        assert_ne!(a.seq_line("line_index", 0).unwrap(), 0);
        assert_eq!(a.seq_bytes(24), Some(sa as usize * 24..sa as usize * 24 + 24));
        drop(b);
        let d = p.lease(16).unwrap();
        assert_eq!(d.seq_slot().unwrap(), sb);
    }

    #[test]
    fn line_table_shape_is_checked() {
        let mut m = hybrid();
        let b = m.buffers.get_mut("line_index").unwrap();
        b.shape = vec![Dim::Const(4), Dim::Var("seqs".into())];
        let Err(e) = Pool::new(&m, 64) else { panic!("4 lines of 3 accepted") };
        assert!(e.to_string().contains("state holds 3"), "{e}");
        let b = m.buffers.get_mut("line_index").unwrap();
        b.shape = vec![Dim::Var("seqs".into())];
        let Err(e) = Pool::new(&m, 64) else { panic!("[seqs] accepted") };
        assert!(e.to_string().contains("[lines, seqs, w]"), "{e}");
        let b = m.buffers.get_mut("line_index").unwrap();
        b.shape = vec![Dim::Const(3), Dim::Var("seqs".into()), Dim::Const(8)];
        let p = Arc::new(Pool::new(&m, 64).unwrap());
        let a = p.lease(16).unwrap();
        assert_eq!(a.seq_width("line_index").unwrap(), 8);
        assert_eq!(a.seq_lines("line_index").unwrap(), 3);
    }

    // ---- checkpoints

    #[test]
    fn checkpoint_shares_pages_and_outlives_the_lease() {
        let p = pool();
        let mut a = p.lease(40).unwrap(); // 3 pages
        let (cp, copies) = p.checkpoint(&mut a, 32).unwrap(); // the first 2
        assert_eq!((cp.tokens(), cp.pages(), cp.seq_slot(), copies), (32, 2, None, Copies::default()));
        assert_eq!((a.shared, a.pages(), p.used()), (2, 3, 3));
        drop(a);
        // The checkpoint keeps its 2 pages; the lease's third came back.
        assert_eq!(p.used(), 2);
        assert_eq!(p.lease(32).unwrap().pages(), 2);
        drop(cp);
        assert_eq!(p.used(), 0);
    }

    #[test]
    fn checkpoints_along_one_lease_share_one_chain() {
        let p = pool();
        let mut a = p.lease(48).unwrap();
        let c1 = p.checkpoint(&mut a, 16).unwrap().0;
        let c2 = p.checkpoint(&mut a, 32).unwrap().0;
        let c3 = p.checkpoint(&mut a, 48).unwrap().0;
        // A shallower checkpoint taken after a deeper one finds its node up the chain.
        let c2b = p.checkpoint(&mut a, 20).unwrap().0;
        assert!(Arc::ptr_eq(&c2.chain, &c2b.chain));
        assert!(Arc::ptr_eq(&c1.chain, c2.chain.parent.as_ref().unwrap()));
        assert_eq!((a.shared, p.used()), (3, 3));
        drop(a);
        drop(c3);
        assert_eq!(p.used(), 2);
        drop(c2);
        assert_eq!(p.used(), 2); // c2b still holds page 2's node
        drop(c2b);
        assert_eq!(p.used(), 1);
        drop(c1);
        assert_eq!(p.used(), 0);
    }

    #[test]
    fn restore_shares_whole_pages_and_copies_a_partial_one() {
        let p = pool();
        let mut a = p.lease(40).unwrap();
        let [a0, a1, _] = a.pages[..] else { panic!() };
        // 20 tokens: page a0 whole, a1 holds positions 16..20.
        let (cp, _) = p.checkpoint(&mut a, 20).unwrap();
        drop(a);
        let (b, copies) = p.restore(&cp, 40).unwrap();
        let [b0, b1, b2] = b.pages[..] else { panic!() };
        // Shares a0, gets a fresh copy of a1 (a1 itself stays the checkpoint's), one more fresh page.
        assert_eq!((b0, b.shared, b.prefix(), b.tokens()), (a0, 1, 20, 48));
        assert_ne!(b1, a1);
        assert_eq!(copies, Copies { pages: vec![(a1, b1)], slot: None });
        assert_eq!(p.used(), 4);
        // Positions from 20 on are the lease's to write, into its own copy.
        assert_eq!(b.slot(20), b1 as i64 * 16 + 4);
        assert_eq!(b.slots(31..33), [b1 as i64 * 16 + 15, b2 as i64 * 16]);
        drop(cp);
        // a1 is only the checkpoint's: freed with it; a0 is still b's.
        assert_eq!(p.used(), 3);
        drop(b);
        assert_eq!(p.used(), 0);
    }

    #[test]
    fn restore_at_a_page_boundary_copies_nothing() {
        let p = pool();
        let mut a = p.lease(32).unwrap();
        let (cp, _) = p.checkpoint(&mut a, 32).unwrap();
        drop(a);
        let (b, copies) = p.restore(&cp, 33).unwrap();
        assert_eq!((b.prefix(), b.shared, b.pages(), copies), (32, 2, 3, Copies::default()));
        assert_eq!(&b.pages[..2], &chain_pages(&Some(Arc::clone(&cp.chain)))[..]);
        assert_eq!(p.used(), 3);
    }

    #[test]
    #[should_panic(expected = "inside the shared prefix")]
    fn restored_lease_refuses_its_prefix() {
        let p = pool();
        let mut a = p.lease(32).unwrap();
        let (cp, _) = p.checkpoint(&mut a, 20).unwrap();
        let (b, _) = p.restore(&cp, 40).unwrap();
        b.slot(19);
    }

    #[test]
    fn restore_denials() {
        let p = pool();
        let mut a = p.lease(16).unwrap();
        let (cp, _) = p.checkpoint(&mut a, 16).unwrap();
        assert_eq!(p.restore(&cp, 49).unwrap_err(), Denied::ExceedsRow { limit: 48 });
        let _b = p.lease(48).unwrap();
        // 3 more pages held: the one fresh page a 17-token restore needs is gone.
        assert_eq!(p.restore(&cp, 17).unwrap_err(), Denied::Busy);
        drop(a);
        assert_eq!(p.restore(&cp, 17).unwrap_err(), Denied::Busy);
    }

    #[test]
    fn hybrid_checkpoint_copies_the_slot_and_retire_moves_it() {
        let p = Arc::new(Pool::new(&hybrid(), 64).unwrap());
        let mut a = p.lease(16).unwrap();
        let sa = a.seq_slot().unwrap();
        let (cp, copies) = p.checkpoint(&mut a, 10).unwrap();
        let sc = cp.seq_slot().unwrap();
        assert_ne!(sc, sa);
        assert_eq!((copies, p.slots_used()), (Copies { pages: vec![], slot: Some((sa, sc)) }, 2));
        // A slot each for a and cp: one left; a restore takes it and copies the state in.
        let (b, copies) = p.restore(&cp, 17).unwrap();
        let sb = b.seq_slot().unwrap();
        assert_eq!((copies.slot, b.prefix(), p.slots_used()), (Some((sc, sb)), 10, 3));
        assert_eq!(p.restore(&cp, 17).unwrap_err(), Denied::Busy);
        drop(b);
        // Retiring a moves its slot to the checkpoint: no copy; its one page is the same one cp shares.
        let a2 = p.retire(a, 10);
        assert_eq!((a2.seq_slot(), a2.pages(), p.slots_used(), p.used()), (Some(sa), 1, 2, 1));
        drop(cp);
        drop(a2);
        assert_eq!((p.slots_used(), p.used()), (0, 0));
    }

    #[test]
    fn checkpoint_without_a_free_slot_is_busy() {
        let p = Arc::new(Pool::new(&hybrid(), 64).unwrap());
        let mut a = p.lease(16).unwrap();
        let _b = p.lease(16).unwrap();
        let _c = p.lease(16).unwrap();
        assert_eq!(p.checkpoint(&mut a, 16).unwrap_err(), Denied::Busy);
        // Retiring never needs a slot.
        let cp = p.retire(a, 16);
        assert!(cp.seq_slot().is_some());
    }

    #[test]
    fn retire_returns_pages_past_len() {
        let p = pool();
        let a = p.lease(48).unwrap();
        let cp = p.retire(a, 17); // pages 0..2 kept
        assert_eq!((cp.tokens(), cp.pages(), p.used()), (17, 2, 2));
    }

    #[test]
    fn a_long_chain_drops_without_recursion() {
        let m = Manifest::from_json(
            r#"{
            "schema_version": 3, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 2}},
            "states": {"kv": {"bytes_per_token": 1}},
            "buffers": {
                "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 200000], "domain": {"index_into": "kv", "stride": 1}}
            },
            "modules": {}, "ops": {}, "programs": {}
        }"#,
        )
        .unwrap();
        let p = Arc::new(Pool::new(&m, 200_000).unwrap());
        let mut a = p.lease(200_000).unwrap();
        let cp = p.checkpoint(&mut a, 200_000).unwrap().0;
        drop(a);
        assert_eq!(p.used(), 200_000);
        drop(cp);
        assert_eq!(p.used(), 0);
    }

    /// Random leases, checkpoints, restores, retirements and drops against
    /// a model that only asks which pages are reachable from a live
    /// handle: those and the free list partition the pool.
    #[test]
    fn ownership_partitions_the_pool() {
        let m = hybrid();
        let p = Arc::new(Pool::new(&m, 64).unwrap());
        let mut leases: Vec<Lease> = Vec::new();
        let mut cps: Vec<Checkpoint> = Vec::new();
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut rand = move |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 33) as usize % n
        };
        for _ in 0..4000 {
            match rand(6) {
                0 => {
                    if let Ok(l) = p.lease(1 + rand(48)) {
                        leases.push(l);
                    }
                }
                1 if !leases.is_empty() => {
                    let at = rand(leases.len());
                    let l = &mut leases[at];
                    let len = 1 + rand(l.tokens());
                    if let Ok((cp, c)) = p.checkpoint(l, len) {
                        assert_eq!(c.slot.map(|(a, _)| a), l.seq_slot());
                        cps.push(cp);
                    }
                }
                2 if !cps.is_empty() => {
                    let cp = &cps[rand(cps.len())];
                    if cp.tokens() < 48 {
                        if let Ok((l, c)) = p.restore(cp, cp.tokens() + 1 + rand(48 - cp.tokens())) {
                            assert_eq!(
                                (l.prefix(), c.pages.len()),
                                (cp.tokens(), (!cp.tokens().is_multiple_of(16)) as usize)
                            );
                            leases.push(l);
                        }
                    }
                }
                3 if !leases.is_empty() => {
                    let l = leases.swap_remove(rand(leases.len()));
                    let len = 1 + rand(l.tokens());
                    cps.push(p.retire(l, len));
                }
                4 if !leases.is_empty() => {
                    leases.swap_remove(rand(leases.len()));
                }
                5 if !cps.is_empty() => {
                    cps.swap_remove(rand(cps.len()));
                }
                _ => {}
            }
            let mut held: Vec<i32> = Vec::new();
            for l in &leases {
                held.extend(&l.pages[l.shared..]);
                held.extend(chain_pages(&l.chain));
            }
            for cp in &cps {
                held.extend(chain_pages(&Some(Arc::clone(&cp.chain))));
            }
            held.sort();
            held.dedup();
            let mut free = lock(&p.free).clone();
            free.sort();
            let mut all = held.clone();
            all.extend(&free);
            all.sort();
            assert_eq!(all, (0..4).collect::<Vec<i32>>());
            // A page is either held or free, never both; each slot has one holder.
            assert!(held.iter().all(|pg| !free.contains(pg)));
            let mut slots: Vec<i32> = leases.iter().map(|l| l.seq_slot().unwrap()).collect();
            slots.extend(cps.iter().map(|c| c.seq_slot().unwrap()));
            let mut sorted = slots.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), slots.len());
            let mut free_slots = lock(&p.free_slots).clone();
            free_slots.sort();
            assert_eq!(free_slots, (1..4).filter(|i| !slots.contains(i)).collect::<Vec<i32>>());
        }
    }
}
