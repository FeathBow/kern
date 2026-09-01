//! KV page pool: the caller-side owner of a paged state's token slots.
//!
//! The runtime provisions a state as `capacity` tokens and knows nothing
//! else; which page holds which sequence's tokens is the caller's business
//! (the manifest's `block_table` domain says how many tokens one page id
//! covers). This pool hands out page ids, keeps one page back as the
//! sacrificial target of padding rows, and never fragments: a request takes
//! all the pages its worst-case length needs at admission and gives them
//! back at the end.

pub struct PagePool {
    /// Tokens per page (the block table's `stride`).
    page: usize,
    free: Vec<i32>,
    total: usize,
    /// Page every padding row points at (its slots are written with junk
    /// each step and never read back as anyone's context).
    pad: i32,
}

impl PagePool {
    /// `capacity_tokens` is a multiple of `page` (the runtime rounds down).
    pub fn new(capacity_tokens: u64, page: u64) -> Option<PagePool> {
        let n = (capacity_tokens / page) as usize;
        if n < 2 {
            return None;
        }
        let pad = (n - 1) as i32;
        // Low ids first: a sequence's pages come out in ascending order.
        let free: Vec<i32> = (0..pad).rev().collect();
        Some(PagePool { page: page as usize, free, total: n - 1, pad })
    }

    pub fn page(&self) -> usize {
        self.page
    }

    pub fn pad(&self) -> i32 {
        self.pad
    }

    /// Pages available to requests (the pad page excluded).
    pub fn total(&self) -> usize {
        self.total
    }

    pub fn available(&self) -> usize {
        self.free.len()
    }

    pub fn used(&self) -> usize {
        self.total - self.free.len()
    }

    /// Pages `tokens` slots need.
    pub fn pages_for(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.page)
    }

    /// All-or-nothing allocation.
    pub fn alloc(&mut self, n: usize) -> Option<Vec<i32>> {
        if n > self.free.len() {
            return None;
        }
        let at = self.free.len() - n;
        Some(self.free.split_off(at))
    }

    pub fn release(&mut self, pages: &[i32]) {
        self.free.extend_from_slice(pages);
    }

    /// The token slot of position `pos` of a sequence holding `pages`.
    pub fn slot(&self, pages: &[i32], pos: usize) -> i64 {
        pages[pos / self.page] as i64 * self.page as i64 + (pos % self.page) as i64
    }
}
