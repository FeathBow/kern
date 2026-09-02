//! The checkpoint table: which prefixes of past sequences are kept, and the
//! longest one a new prompt can start from.
//!
//! A [`Checkpoint`] is bytes; what makes it findable is the tokens it
//! holds. The table keys every checkpoint by a hash chain over its tokens
//! in blocks of the page unit — the chain at depth `d` covers the first
//! `d` pages, a tail hash covers what ends inside the next one — so a
//! lookup hashes the prompt once and probes the depths from the deepest
//! down; the first depth with a checkpoint whose tail also matches is the
//! longest usable prefix. A prompt never uses its last token: that token
//! must still go through a program to produce the next one.
//!
//! A sequence carries its own [`Chain`] and grows it as tokens enter the
//! state, so checkpointing every page hashes each token once; the table
//! reads the chain's key at the checkpoint's length instead of rehashing.
//!
//! Eviction is the caller's answer to a `Busy` lease: [`Prefix::evict`]
//! drops the checkpoint hit least recently, and dropping it is all that
//! frees anything (pages shared with a live sequence or a deeper checkpoint
//! stay). Recency is a counter, not a clock. A hit touches every
//! checkpoint on its chain, deepest first, so the chain ages together and
//! its leaf — the only entry whose drop returns a page — goes first.
//!
//! Same tokens, same hashes, same choices: the table has no clock and no
//! hash map, so a replay makes the same decisions in the same order.

use std::collections::BTreeMap;

use crate::pages::Checkpoint;

/// A checkpoint a prompt can continue from: the first `len` prompt
/// tokens are already in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hit {
    pub id: u64,
    pub len: usize,
}

struct Entry {
    checkpoint: Checkpoint,
    key: Key,
    used: u64,
}

pub struct Prefix {
    unit: usize,
    entries: BTreeMap<u64, Entry>,
    /// (depth, chain through it) → entries at that depth.
    at_depth: BTreeMap<(usize, u64), Vec<u64>>,
    /// Recency stamp → entry, the eviction order.
    lru: BTreeMap<u64, u64>,
    next_id: u64,
    clock: u64,
}

const SEED: u64 = 0x243F_6A88_85A3_08D3;

/// splitmix64's finalizer: a bijection, so a chain never collides with a
/// shorter one by absorbing a zero.
fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn fold(h: u64, token: i64) -> u64 {
    mix(h ^ (token as u64).wrapping_add(0x9E37_79B9_7F4A_7C15))
}

fn hash(h: u64, tokens: &[i64]) -> u64 {
    tokens.iter().fold(h, |h, &t| fold(h, t))
}

/// What identifies a checkpoint's tokens: whole pages and the chain
/// through them, then the tokens past them and the chain continued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Key {
    depth: usize,
    chain: u64,
    tail_len: usize,
    tail: u64,
}

/// The hash chain of one sequence, grown a token at a time: one hash per
/// whole page, one over the tokens past the last whole page. Pure data;
/// the same tokens in any grouping give the same chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chain {
    unit: usize,
    /// `heads[d]` covers the first `d` pages; `heads[0]` is the seed.
    heads: Vec<u64>,
    tail: u64,
    len: usize,
}

impl Chain {
    pub fn new(unit: usize) -> Chain {
        assert!(unit >= 1);
        Chain { unit, heads: vec![SEED], tail: SEED, len: 0 }
    }

    /// The chain of `tokens`.
    pub fn over(unit: usize, tokens: &[i64]) -> Chain {
        let mut c = Chain::new(unit);
        c.extend(tokens.iter().copied());
        c
    }

    pub fn push(&mut self, token: i64) {
        self.tail = fold(self.tail, token);
        self.len += 1;
        if self.len.is_multiple_of(self.unit) {
            self.heads.push(self.tail);
        }
    }

    pub fn extend(&mut self, tokens: impl IntoIterator<Item = i64>) {
        tokens.into_iter().for_each(|t| self.push(t));
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The key of the first `len` tokens: known at every whole page and
    /// at the chain's own length, nowhere else.
    fn key(&self, len: usize) -> Option<Key> {
        let depth = len / self.unit;
        let chain = *self.heads.get(depth)?;
        let tail_len = len % self.unit;
        match tail_len {
            0 => Some(Key { depth, chain, tail_len, tail: chain }),
            _ if len == self.len => Some(Key { depth, chain, tail_len, tail: self.tail }),
            _ => None,
        }
    }
}

impl Prefix {
    /// A table over sequences paged in `unit` tokens.
    pub fn new(unit: usize) -> Prefix {
        assert!(unit >= 1);
        Prefix { unit, entries: BTreeMap::new(), at_depth: BTreeMap::new(), lru: BTreeMap::new(), next_id: 0, clock: 0 }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn touch(&mut self, id: u64) {
        let e = self.entries.get_mut(&id).expect("entry");
        self.lru.remove(&e.used);
        self.clock += 1;
        e.used = self.clock;
        self.lru.insert(self.clock, id);
    }

    /// Keep `checkpoint`, whose tokens are the first `checkpoint.tokens()`
    /// of `chain`. A checkpoint of the same tokens is already here: the
    /// new one is dropped and the old one counts as used.
    pub fn insert(&mut self, chain: &Chain, checkpoint: Checkpoint) -> u64 {
        assert_eq!(chain.unit, self.unit, "chain unit and table unit differ");
        let key = chain.key(checkpoint.tokens()).expect("checkpoint length is on the chain");
        if let Some(&id) =
            self.at_depth.get(&(key.depth, key.chain)).and_then(|ids| ids.iter().find(|id| self.entries[id].key == key))
        {
            self.touch(id);
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.clock += 1;
        self.entries.insert(id, Entry { checkpoint, key, used: self.clock });
        self.at_depth.entry((key.depth, key.chain)).or_default().push(id);
        self.lru.insert(self.clock, id);
        id
    }

    /// The longest checkpoint holding a proper prefix of `tokens` (the
    /// last token is never covered). A hit touches it and every
    /// checkpoint on the chain above it.
    pub fn lookup(&mut self, tokens: &[i64]) -> Option<Hit> {
        let usable = tokens.len().checked_sub(1)?;
        let heads = Chain::over(self.unit, &tokens[..usable]).heads;
        let mut hit: Option<Hit> = None;
        let mut above: Vec<u64> = Vec::new();
        for (d, &head) in heads.iter().enumerate().rev() {
            let Some(ids) = self.at_depth.get(&(d, head)) else { continue };
            let room = usable - d * self.unit;
            let best = ids
                .iter()
                .map(|&id| (id, self.entries[&id].key))
                .filter(|(_, k)| {
                    k.tail_len <= room && k.tail == hash(head, &tokens[d * self.unit..d * self.unit + k.tail_len])
                })
                .max_by_key(|(_, k)| k.tail_len);
            let Some((id, k)) = best else { continue };
            match hit {
                None => hit = Some(Hit { id, len: d * self.unit + k.tail_len }),
                Some(_) => above.push(id),
            }
        }
        let hit = hit?;
        self.touch(hit.id);
        for id in above {
            self.touch(id);
        }
        Some(hit)
    }

    pub fn get(&self, id: u64) -> Option<&Checkpoint> {
        self.entries.get(&id).map(|e| &e.checkpoint)
    }

    /// Drop the checkpoint used least recently; `false` when there is none.
    pub fn evict(&mut self) -> bool {
        let Some((&stamp, &id)) = self.lru.iter().next() else { return false };
        self.lru.remove(&stamp);
        let e = self.entries.remove(&id).expect("entry");
        let ids = self.at_depth.get_mut(&(e.key.depth, e.key.chain)).expect("depth");
        ids.retain(|&i| i != id);
        if ids.is_empty() {
            self.at_depth.remove(&(e.key.depth, e.key.chain));
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::pages::Pool;
    use kern_manifest::types::Manifest;

    /// kv paged in 4 tokens, 8 pages.
    fn pool() -> Arc<Pool> {
        let m = Manifest::from_json(
            r#"{
            "schema_version": 3, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 2}},
            "states": {"kv": {"bytes_per_token": 1}},
            "buffers": {
                "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 8], "domain": {"index_into": "kv", "stride": 4}}
            },
            "modules": {}, "ops": {}, "programs": {}
        }"#,
        )
        .unwrap();
        Arc::new(Pool::new(&m, 32).unwrap())
    }

    fn toks(n: usize) -> Vec<i64> {
        (0..n as i64).map(|i| i * 7 + 3).collect()
    }

    #[test]
    fn longest_proper_prefix_wins() {
        let p = pool();
        let mut t = Prefix::new(4);
        let mut l = p.lease(12).unwrap();
        let a = t.insert(&Chain::over(4, &toks(4)), p.checkpoint(&mut l, 4).unwrap().0);
        let b = t.insert(&Chain::over(4, &toks(8)), p.checkpoint(&mut l, 8).unwrap().0);
        let c = t.insert(&Chain::over(4, &toks(10)), p.checkpoint(&mut l, 10).unwrap().0);
        assert_eq!(t.len(), 3);
        // A 12-token prompt of the same tokens may use 11: the 10-token checkpoint.
        assert_eq!(t.lookup(&toks(12)), Some(Hit { id: c, len: 10 }));
        // A 10-token prompt may use 9: the 8-token one.
        assert_eq!(t.lookup(&toks(10)), Some(Hit { id: b, len: 8 }));
        // Diverging inside the third page: the 8-token one.
        let mut d = toks(12);
        d[9] = -1;
        assert_eq!(t.lookup(&d), Some(Hit { id: b, len: 8 }));
        // Diverging inside the second page: the 4-token one.
        d[5] = -1;
        assert_eq!(t.lookup(&d), Some(Hit { id: a, len: 4 }));
        d[0] = -1;
        assert_eq!(t.lookup(&d), None);
        assert_eq!(t.lookup(&toks(1)), None);
        assert_eq!(t.lookup(&[]), None);
    }

    #[test]
    fn same_tokens_share_one_entry() {
        let p = pool();
        let mut t = Prefix::new(4);
        let mut l = p.lease(8).unwrap();
        let a = t.insert(&Chain::over(4, &toks(8)), p.checkpoint(&mut l, 8).unwrap().0);
        assert_eq!(p.used(), 2);
        let b = t.insert(&Chain::over(4, &toks(8)), p.checkpoint(&mut l, 8).unwrap().0);
        assert_eq!((a, b, t.len(), p.used()), (a, a, 1, 2));
        // Same pages but different tokens is a different entry.
        let mut other = toks(8);
        other[7] = -1;
        let c = t.insert(&Chain::over(4, &other), p.checkpoint(&mut l, 8).unwrap().0);
        assert_ne!(c, a);
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn eviction_is_least_recent_leaf_first() {
        let p = pool();
        let mut t = Prefix::new(4);
        let mut l = p.lease(12).unwrap();
        let a = t.insert(&Chain::over(4, &toks(4)), p.checkpoint(&mut l, 4).unwrap().0);
        let b = t.insert(&Chain::over(4, &toks(8)), p.checkpoint(&mut l, 8).unwrap().0);
        let mut other = toks(8);
        other[6] = -1;
        let c = t.insert(&Chain::over(4, &other), p.checkpoint(&mut l, 8).unwrap().0);
        drop(l);
        // A hit on b touches a too; c is now the least recent.
        assert_eq!(t.lookup(&toks(9)).map(|h| h.id), Some(b));
        assert!(t.evict());
        assert_eq!((t.get(c).is_none(), t.get(a).is_some(), t.get(b).is_some()), (true, true, true));
        // Of a chain aged together, the leaf goes before its parent.
        assert!(t.evict());
        assert_eq!((t.get(b).is_none(), t.get(a).is_some()), (true, true));
        assert_eq!(p.used(), 1);
        assert!(t.evict());
        assert_eq!((t.is_empty(), p.used()), (true, 0));
        assert!(!t.evict());
    }

    #[test]
    fn a_page_the_lease_still_holds_survives_eviction() {
        let p = pool();
        let mut t = Prefix::new(4);
        let mut l = p.lease(8).unwrap();
        t.insert(&Chain::over(4, &toks(8)), p.checkpoint(&mut l, 8).unwrap().0);
        assert!(t.evict());
        assert_eq!(p.used(), 2);
        drop(l);
        assert_eq!(p.used(), 0);
    }

    #[test]
    fn restore_then_checkpoint_deeper() {
        let p = pool();
        let mut t = Prefix::new(4);
        let mut l = p.lease(8).unwrap();
        t.insert(&Chain::over(4, &toks(8)), p.checkpoint(&mut l, 8).unwrap().0);
        drop(l);
        let hit = t.lookup(&toks(16)).unwrap();
        let (mut l2, _) = p.restore(t.get(hit.id).unwrap(), 16).unwrap();
        assert_eq!(l2.prefix(), 8);
        // The sequence grows one chain and checkpoints off it as it goes.
        let mut c = Chain::over(4, &toks(12));
        t.insert(&c, p.checkpoint(&mut l2, 12).unwrap().0);
        c.extend(toks(16)[12..].iter().copied());
        t.insert(&c, p.checkpoint(&mut l2, 16).unwrap().0);
        drop(l2);
        assert_eq!(t.lookup(&toks(17)).map(|h| h.len), Some(16));
        assert_eq!(t.lookup(&toks(13)).map(|h| h.len), Some(12));
        assert_eq!(p.used(), 4);
    }

    /// A chain grown a token at a time is the chain over the tokens, its
    /// key at every whole page is the shorter chain's, and it has no key
    /// inside a page it has grown past.
    #[test]
    fn a_chain_is_the_same_however_it_grows() {
        let mut x = 0x9E37_79B9u64;
        let mut rand = |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n as u64) as usize
        };
        for _ in 0..200 {
            let unit = 1 + rand(6);
            let n = rand(40);
            let tokens: Vec<i64> = (0..n).map(|_| rand(3) as i64).collect();
            let mut grown = Chain::new(unit);
            for (i, &t) in tokens.iter().enumerate() {
                assert_eq!(grown, Chain::over(unit, &tokens[..i]));
                grown.push(t);
            }
            assert_eq!((grown.len(), &grown), (n, &Chain::over(unit, &tokens)));
            for len in 0..=n {
                let short = Chain::over(unit, &tokens[..len]);
                let expected = (len.is_multiple_of(unit) || len == n).then(|| short.key(len).unwrap());
                assert_eq!(grown.key(len), expected, "unit {unit} len {len} of {n}");
            }
            assert_eq!(grown.key(n + 1), None);
        }
    }
}
