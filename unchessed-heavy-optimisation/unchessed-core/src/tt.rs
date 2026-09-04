//! Transposition table — lock-free, shared across search threads.
//!
//! Each slot is two `AtomicU64` words: a packed `data` word (move, score,
//! depth, bound) and `key ^ data` in place of the raw key. A probe reads
//! both words and re-derives the key; if it doesn't match, the entry is
//! either genuinely a different position or was torn by a concurrent write
//! — either way it's treated as a miss, never as corrupted data. This is
//! the standard technique that lets Lazy SMP threads share one table with
//! no locking (relaxed ordering; occasional lost/overwritten entries are
//! an accepted cost, not a correctness issue, for a cache like this).

use std::sync::atomic::{AtomicU64, Ordering};

use crate::board::Move;

pub const BOUND_EXACT: u8 = 0;
pub const BOUND_LOWER: u8 = 1;
pub const BOUND_UPPER: u8 = 2;

#[derive(Clone, Copy)]
pub struct Entry {
    pub mv: u16,
    pub score: i16,
    pub depth: i8,
    pub bound: u8,
}

#[inline]
fn pack(mv: u16, score: i16, depth: i8, bound: u8) -> u64 {
    // depth stored +1 so the "empty" sentinel (depth=-1) is all-zero data,
    // matching a freshly-allocated (zeroed) table.
    (mv as u64)
        | ((score as u16 as u64) << 16)
        | (((depth as i32 + 1) as u16 as u64) << 32)
        | ((bound as u64) << 48)
}

#[inline]
fn unpack(data: u64) -> Entry {
    Entry {
        mv: (data & 0xFFFF) as u16,
        score: ((data >> 16) & 0xFFFF) as u16 as i16,
        depth: (((data >> 32) & 0xFFFF) as i32 - 1) as i8,
        bound: ((data >> 48) & 0xFF) as u8,
    }
}

struct Slot {
    xor_key: AtomicU64,
    data: AtomicU64,
}

pub struct TT {
    table: Vec<Slot>,
    mask: usize,
}

impl TT {
    pub fn new(mb: usize) -> TT {
        let bytes = mb.max(1) * 1024 * 1024;
        let mut n = bytes / std::mem::size_of::<Slot>();
        if !n.is_power_of_two() {
            n = n.next_power_of_two() >> 1;
        }
        n = n.max(1024);
        let mut table = Vec::with_capacity(n);
        table.resize_with(n, || Slot {
            xor_key: AtomicU64::new(0),
            data: AtomicU64::new(0),
        });
        TT { table, mask: n - 1 }
    }

    pub fn resize(&mut self, mb: usize) {
        *self = TT::new(mb);
    }

    /// Safe to call while other threads hold `&TT` for probe/store (a clear
    /// is only ever issued between searches, but doesn't need exclusivity).
    pub fn clear(&self) {
        for slot in &self.table {
            slot.xor_key.store(0, Ordering::Relaxed);
            slot.data.store(0, Ordering::Relaxed);
        }
    }

    /// Ask the CPU to start pulling this hash's slot into cache.
    ///
    /// The table is megabytes wide, so `probe` is a near-guaranteed cache
    /// miss taken at the very top of a node. Issuing this right after the
    /// child position is made overlaps that memory latency with the
    /// legality/accumulator work that runs before the child actually
    /// probes.
    ///
    /// This is a pure hint: prefetch has no architectural effect, cannot
    /// fault, and does not change what `probe` returns.
    #[inline]
    pub fn prefetch(&self, hash: u64) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let slot = &self.table[(hash as usize) & self.mask] as *const Slot as *const i8;
            // SAFETY: `slot` is derived from a live element of `self.table`,
            // and `_mm_prefetch` only issues a cache hint -- it never
            // dereferences the pointer in an observable way.
            unsafe { _mm_prefetch(slot, _MM_HINT_T0) };
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = hash;
        }
    }

    #[inline]
    /// Permille of the table currently occupied, for UCI `info hashfull`.
    ///
    /// This exists because hash pressure is invisible without it, and
    /// Stockfish's published measurements show it matters: at LTC, going
    /// from 131 permille hashfull to 591 costs about 12 Elo, and to 931
    /// costs about 52. The advice that follows from that data is to keep
    /// average hashfull under ~300 permille, which an operator cannot act on
    /// unless the engine reports the number.
    ///
    /// Sampled over the first 1000 slots rather than scanned in full: the
    /// UCI specification defines this field as approximate, and a full scan
    /// of a 2 GB table during search would cost more than the information is
    /// worth. Small tables are scanned exactly, since 1000 slots may be the
    /// whole table.
    ///
    /// Occupancy is `data != 0`, which is exact rather than heuristic here:
    /// `pack` stores `depth + 1`, so a live entry can never have all-zero
    /// data, and `new`/`clear` leave exactly zero.
    pub fn hashfull(&self) -> usize {
        let sample = self.table.len().min(1000);
        if sample == 0 {
            return 0;
        }
        let used = self.table[..sample]
            .iter()
            .filter(|slot| slot.data.load(Ordering::Relaxed) != 0)
            .count();
        used * 1000 / sample
    }

    pub fn probe(&self, hash: u64) -> Option<Entry> {
        let slot = &self.table[(hash as usize) & self.mask];
        let xor_key = slot.xor_key.load(Ordering::Relaxed);
        let data = slot.data.load(Ordering::Relaxed);
        if xor_key ^ data != hash {
            return None; // miss, or a torn read racing a concurrent writer
        }
        let e = unpack(data);
        if e.depth < 0 {
            None
        } else {
            Some(e)
        }
    }

    #[inline]
    pub fn store(&self, hash: u64, mv: Move, score: i32, depth: i32, bound: u8) {
        let slot = &self.table[(hash as usize) & self.mask];
        // depth-preferred replacement; always claim slots holding other keys.
        // The read-then-write here isn't atomic as a whole under concurrent
        // access — a rare suboptimal replacement is fine, never a corrupt one.
        let existing = {
            let xor_key = slot.xor_key.load(Ordering::Relaxed);
            let data = slot.data.load(Ordering::Relaxed);
            if xor_key ^ data == hash {
                Some(unpack(data))
            } else {
                None
            }
        };
        let same_key = existing.is_some();
        if let Some(e) = existing {
            if depth as i8 <= e.depth && bound != BOUND_EXACT {
                return;
            }
        }
        let keep_move = match (mv == Move::NONE, same_key, existing) {
            (true, true, Some(e)) => e.mv,
            _ => mv.0,
        };
        let data = pack(
            keep_move,
            score.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            depth.clamp(0, 127) as i8,
            bound,
        );
        slot.data.store(data, Ordering::Relaxed);
        slot.xor_key.store(hash ^ data, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `hashfull` must read 0 on a fresh table and rise as entries land.
    ///
    /// The occupancy test is `data != 0`, which is only exact because `pack`
    /// stores `depth + 1` -- so a real entry, even one at depth 0, can never
    /// produce all-zero data. If that encoding ever changed, a stored entry
    /// would read as empty and this metric would silently report 0 forever.
    #[test]
    fn hashfull_is_zero_when_empty_and_grows_when_filled() {
        let tt = TT::new(1);
        assert_eq!(tt.hashfull(), 0, "a fresh table must be empty");

        // Fill the sampled window. Slot index is `hash & mask`, so hashes
        // 0..1000 land in distinct slots for any table of at least 1024.
        for hash in 0..1000u64 {
            tt.store(hash, Move::NONE, 0, 0, BOUND_EXACT);
        }
        assert_eq!(tt.hashfull(), 1000, "every sampled slot should be used");

        tt.clear();
        assert_eq!(tt.hashfull(), 0, "clear must reset occupancy");
    }

    /// A depth-0 entry must still count as occupied.
    ///
    /// This is the specific case the `depth + 1` offset protects, and the
    /// one a naive `depth != 0` check would get wrong.
    #[test]
    fn hashfull_counts_depth_zero_entries() {
        let tt = TT::new(1);
        tt.store(0, Move::NONE, 0, 0, BOUND_EXACT);
        assert!(tt.hashfull() > 0, "a depth-0 entry is still an entry");
    }

    /// Partial occupancy must land between the extremes, not saturate.
    #[test]
    fn hashfull_reports_partial_occupancy() {
        let tt = TT::new(1);
        for hash in 0..250u64 {
            tt.store(hash, Move::NONE, 0, 0, BOUND_EXACT);
        }
        let full = tt.hashfull();
        assert!(
            (200..=300).contains(&full),
            "expected roughly 250 permille, got {full}"
        );
    }

    /// The value is a permille and must never exceed 1000.
    #[test]
    fn hashfull_is_bounded() {
        let tt = TT::new(1);
        for hash in 0..5000u64 {
            tt.store(hash, Move::NONE, 0, 1, BOUND_EXACT);
        }
        assert!(tt.hashfull() <= 1000);
    }

    /// `prefetch` is a cache hint and must be observationally inert: it may
    /// not fault for any hash, and may not change what a later probe sees.
    #[test]
    fn prefetch_is_side_effect_free() {
        let tt = TT::new(1);
        let hash = 0x0123_4567_89ab_cdef;

        tt.prefetch(hash);
        assert!(tt.probe(hash).is_none(), "prefetch must not create entries");

        tt.store(hash, Move(0x1234), 42, 7, BOUND_EXACT);
        let before = tt.probe(hash).expect("stored entry");

        // Prefetching arbitrary hashes, including ones never stored and the
        // extremes of the index space, must not disturb existing data.
        for probe in [0u64, u64::MAX, hash, hash ^ 0xffff, 1] {
            tt.prefetch(probe);
        }

        let after = tt.probe(hash).expect("entry survives prefetch");
        assert_eq!(before.mv, after.mv);
        assert_eq!(before.score, after.score);
        assert_eq!(before.depth, after.depth);
        assert_eq!(before.bound, after.bound);
    }
}

// TT is Sync automatically (all fields are atomics), so `&TT` can be shared
// freely across scoped search threads.
