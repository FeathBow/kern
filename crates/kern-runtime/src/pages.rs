//! Token-slot ownership of the paged states.
//!
//! The runtime provisions every paged state as `capacity` tokens; the
//! kernels address them through the manifest's page tables (an input whose
//! domain `index_into`s a state, `stride` tokens per entry) and slot lists
//! (stride 1). Which slot holds what is the caller's business, but the
//! only way to name a slot is a [`Lease`]: pages come out of the pool as a
//! lease, slots and table rows are computed from it, and the pages go back
//! when it drops. Nothing can free a page twice, free a page it never
//! leased, or address a slot past its lease.
//!
//! Pages are in the runtime's page unit — the lcm of every table's stride —
//! so one lease serves every paged state at once (a 16-token draft table
//! sees 49 entries per 784-token page of the target's table). A lease is
//! all-or-nothing: a caller takes the pages its worst case needs and holds
//! them, so the pool never fragments.

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
}

fn lock(free: &Mutex<Vec<i32>>) -> MutexGuard<'_, Vec<i32>> {
    free.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The page tables: every input whose domain `index_into`s a state.
fn tables(m: &Manifest) -> BTreeMap<String, Table> {
    m.buffers
        .iter()
        .filter_map(|(name, b)| {
            let d = b.domain.as_ref()?;
            if !m.states.contains_key(d.index_into.as_deref()?) {
                return None;
            }
            let Some(Dim::Const(width)) = b.shape.last() else { return None };
            Some((name.clone(), Table { stride: d.stride.max(1), width: *width as usize }))
        })
        .collect()
}

/// Tokens one sequence can hold, in whole pages of `unit`: what the
/// narrowest page-table row references. `None` when nothing is paged.
pub(crate) fn row_tokens(m: &Manifest, unit: u64) -> Option<u64> {
    tables(m).values().map(|t| t.width as u64 * t.stride / unit * unit).min()
}

impl Pool {
    /// `capacity` is a multiple of `unit`, the lcm of every table stride.
    pub(crate) fn new(m: &Manifest, capacity: u64, unit: u64) -> Pool {
        let total = (capacity / unit) as usize;
        let tables = tables(m);
        let max_pages = row_tokens(m, unit).map_or(total, |t| (t / unit) as usize);
        Pool { unit, total, max_pages, tables, free: Mutex::new((0..total as i32).collect()) }
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

    pub(crate) fn lease(self: &Arc<Pool>, tokens: usize) -> std::result::Result<Lease, Denied> {
        let need = tokens.div_ceil(self.unit as usize);
        if need > self.max_pages {
            return Err(Denied::ExceedsRow { limit: self.max_seq_tokens() });
        }
        if need > self.total {
            return Err(Denied::ExceedsPool);
        }
        let mut free = lock(&self.free);
        if need > free.len() {
            return Err(Denied::Busy);
        }
        let at = free.len() - need;
        Ok(Lease { pages: free.split_off(at), pool: Arc::clone(self) })
    }
}

/// Why [`Runtime::lease`](crate::Runtime::lease) said no.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Denied {
    /// More tokens than one page-table row can reference; never fits.
    ExceedsRow { limit: usize },
    /// More pages than the pool has, even empty; never fits.
    ExceedsPool,
    /// Fits, but not right now.
    Busy,
}

impl fmt::Display for Denied {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Denied::ExceedsRow { limit } => write!(f, "longer than a page-table row ({limit} tokens)"),
            Denied::ExceedsPool => write!(f, "more pages than the state capacity holds"),
            Denied::Busy => write!(f, "pages busy"),
        }
    }
}

impl std::error::Error for Denied {}

/// Pages leased to one sequence: the only handle to its token slots.
/// Dropping it returns the pages to the runtime.
pub struct Lease {
    pages: Vec<i32>,
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
}

impl fmt::Debug for Lease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lease({} pages)", self.pages.len())
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        lock(&self.pool.free).append(&mut self.pages);
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

    fn pool() -> Arc<Pool> {
        Arc::new(Pool::new(&manifest(), 64, 16))
    }

    #[test]
    fn geometry() {
        let p = pool();
        assert_eq!((p.total(), p.unit(), p.max_seq_tokens()), (4, 16, 48));
        assert_eq!(p.tables().collect::<Vec<_>>(), ["block_table", "draft_block_table"]);
        // The draft table's 16 entries × 4 tokens hold 4 pages; kv's 3 × 16 hold 3.
        assert_eq!(p.max_pages, 3);
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
        let small = Arc::new(Pool::new(&manifest(), 32, 16));
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
    }

    #[test]
    #[should_panic(expected = "past the lease")]
    fn slot_past_lease() {
        let p = pool();
        p.lease(16).unwrap().slot(16);
    }
}
