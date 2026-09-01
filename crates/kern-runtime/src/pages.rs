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
//! its slot — `slot × lines_per_slot + r`.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use kern_manifest::types::{Dim, Manifest};

use crate::error::{bail, Result};

/// A page table input: `stride` tokens per entry, `width` entries per row.
struct Table {
    stride: u64,
    width: usize,
}

/// A line table input over a per-sequence state: `rows` lines per
/// sequence it names, out of `per_slot` lines a slot holds.
struct SeqTable {
    rows: usize,
    per_slot: i32,
}

/// Shared between the runtime and every live lease so a drop returns pages
/// directly. One caller thread; the mutex is for `Send`, never contended.
pub(crate) struct Pool {
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

fn lock(free: &Mutex<Vec<i32>>) -> MutexGuard<'_, Vec<i32>> {
    free.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The page tables: every input whose domain `index_into`s a paged state.
fn tables(m: &Manifest) -> BTreeMap<String, Table> {
    m.buffers
        .iter()
        .filter_map(|(name, b)| {
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
/// state, shaped `[lines, ...]`.
fn seq_tables(m: &Manifest) -> Result<BTreeMap<String, SeqTable>> {
    let mut out = BTreeMap::new();
    for (name, b) in &m.buffers {
        let Some(d) = b.domain.as_ref() else { continue };
        let Some(st) = d.index_into.as_deref().and_then(|s| m.states.get(s)) else { continue };
        if !st.is_per_seq() {
            continue;
        }
        let Some(Dim::Const(rows)) = b.shape.first() else {
            bail!(Manifest, "`{name}` indexes a per-sequence state: expected shape [lines, ...], got {:?}", b.shape);
        };
        let per_slot = st.bytes_per_seq / d.stride.max(1);
        if *rows > per_slot {
            bail!(Manifest, "`{name}` names {rows} lines per sequence, the state holds {per_slot}");
        }
        out.insert(name.clone(), SeqTable { rows: *rows as usize, per_slot: per_slot as i32 });
    }
    Ok(out)
}

/// Tokens one sequence can hold, in whole pages of `unit`: what the
/// narrowest page-table row references. `None` when nothing is paged.
pub(crate) fn row_tokens(m: &Manifest, unit: u64) -> Option<u64> {
    tables(m).values().map(|t| t.width as u64 * t.stride / unit * unit).min()
}

impl Pool {
    /// `capacity` is a multiple of `unit`, the lcm of every table stride.
    pub(crate) fn new(m: &Manifest, capacity: u64, unit: u64) -> Result<Pool> {
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

    pub(crate) fn unit(&self) -> u64 {
        self.unit
    }

    pub(crate) fn total(&self) -> usize {
        self.total
    }

    pub(crate) fn used(&self) -> usize {
        self.total - lock(&self.free).len()
    }

    pub(crate) fn max_seq_tokens(&self) -> usize {
        self.max_pages * self.unit as usize
    }

    pub(crate) fn tables(&self) -> impl Iterator<Item = &str> {
        self.tables.keys().map(String::as_str)
    }

    /// Sequence slots provisioned (0 when no state is per-sequence).
    pub(crate) fn slots(&self) -> usize {
        self.slots
    }

    pub(crate) fn seq_tables(&self) -> impl Iterator<Item = &str> {
        self.seq_tables.keys().map(String::as_str)
    }

    pub(crate) fn lease(self: &Arc<Pool>, tokens: usize) -> std::result::Result<Lease, Denied> {
        let need = tokens.div_ceil(self.unit as usize);
        if need > self.max_pages {
            return Err(Denied::ExceedsRow { limit: self.max_seq_tokens() });
        }
        if need > self.total {
            return Err(Denied::ExceedsPool);
        }
        let mut free = lock(&self.free);
        let mut free_slots = lock(&self.free_slots);
        if need > free.len() || (self.slots > 0 && free_slots.is_empty()) {
            return Err(Denied::Busy);
        }
        let at = free.len() - need;
        let slot = if self.slots > 0 { free_slots.pop() } else { None };
        Ok(Lease { pages: free.split_off(at), slot, pool: Arc::clone(self) })
    }
}

/// Why [`Runtime::lease`](crate::Runtime::lease) said no.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Denied {
    /// More tokens than one page-table row can reference; never fits.
    ExceedsRow { limit: usize },
    /// More pages than the pool has, even empty; never fits.
    ExceedsPool,
    /// Fits, but not right now: pages or sequence slots all leased.
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
/// to the runtime.
pub struct Lease {
    pages: Vec<i32>,
    slot: Option<i32>,
    pool: Arc<Pool>,
}

impl Lease {
    /// Pages held.
    pub fn pages(&self) -> usize {
        self.pages.len()
    }

    /// Token slots held (whole pages, so at least what was asked for).
    pub fn tokens(&self) -> usize {
        self.pages.len() * self.pool.unit as usize
    }

    /// The token slot of position `pos` of the sequence.
    pub fn slot(&self, pos: usize) -> i64 {
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
}

impl fmt::Debug for Lease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.slot {
            Some(s) => write!(f, "Lease({} pages, slot {s})", self.pages.len()),
            None => write!(f, "Lease({} pages)", self.pages.len()),
        }
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        lock(&self.pool.free).append(&mut self.pages);
        if let Some(s) = self.slot.take() {
            lock(&self.pool.free_slots).push(s);
        }
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
        Arc::new(Pool::new(&manifest(), 64, 16).unwrap())
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
        let small = Arc::new(Pool::new(&manifest(), 32, 16).unwrap());
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
        let p = Arc::new(Pool::new(&m, 64, 16).unwrap());
        assert_eq!((p.slots(), p.seq_tables().collect::<Vec<_>>()), (4, vec!["line_index"]));
        let a = p.lease(16).unwrap();
        let b = p.lease(16).unwrap();
        let c = p.lease(16).unwrap();
        let (sa, sb, sc) = (a.seq_slot().unwrap(), b.seq_slot().unwrap(), c.seq_slot().unwrap());
        let mut got = vec![sa, sb, sc];
        got.sort();
        assert_eq!(got, [1, 2, 3]);
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
        let Err(e) = Pool::new(&m, 64, 16) else { panic!("4 lines of 3 accepted") };
        assert!(e.to_string().contains("state holds 3"), "{e}");
        let b = m.buffers.get_mut("line_index").unwrap();
        b.shape = vec![Dim::Var("seqs".into())];
        let Err(e) = Pool::new(&m, 64, 16) else { panic!("[seqs] accepted") };
        assert!(e.to_string().contains("[lines, ...]"), "{e}");
    }
}
