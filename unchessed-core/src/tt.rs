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
    pub generation: u8,
}

#[inline]
fn pack(mv: u16, score: i16, depth: i8, bound: u8, generation: u8) -> u64 {
    // depth stored +1 so the "empty" sentinel (depth=-1) is all-zero data,
    // matching a freshly-allocated (zeroed) table.
    (mv as u64)
        | ((score as u16 as u64) << 16)
        | (((depth as i32 + 1) as u16 as u64) << 32)
        | ((bound as u64) << 48)
        | ((generation as u64) << 56)
}

#[inline]
fn unpack(data: u64) -> Entry {
    Entry {
        mv: (data & 0xFFFF) as u16,
        score: ((data >> 16) & 0xFFFF) as u16 as i16,
        depth: (((data >> 32) & 0xFFFF) as i32 - 1) as i8,
        bound: ((data >> 48) & 0xFF) as u8,
        generation: (data >> 56) as u8,
    }
}

struct Slot {
    xor_key: AtomicU64,
    data: AtomicU64,
}

pub struct TT {
    table: Vec<Slot>,
    mask: usize,
    generation: AtomicU64,
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
        TT {
            table,
            mask: n - 1,
            generation: AtomicU64::new(1),
        }
    }

    fn generation(&self) -> u8 {
        self.generation.load(Ordering::Relaxed) as u8
    }

    /// Start a new root generation. Helpers share the same generation and old
    /// entries become freely replaceable without clearing the whole table.
    pub fn new_search(&self) {
        let next = self.generation().wrapping_add(1).max(1);
        self.generation.store(next as u64, Ordering::Relaxed);
    }

    /// Approximate occupancy of the current generation, in permill as required
    /// by UCI `hashfull`.
    pub fn hashfull(&self) -> u16 {
        let generation = self.generation();
        let sample = self.table.len().min(1_000);
        if sample == 0 {
            return 0;
        }
        let used = self.table[..sample]
            .iter()
            .filter(|slot| unpack(slot.data.load(Ordering::Relaxed)).generation == generation)
            .count();
        (used * 1_000 / sample) as u16
    }

    pub fn resize(&mut self, mb: usize) {
        *self = TT::new(mb);
    }

    /// Safe to call while other threads hold `&TT` for probe/store (a clear
    /// is only ever issued between searches, but doesn't need exclusivity).
    pub fn clear(&self) {
        self.generation.store(1, Ordering::Relaxed);
        for slot in &self.table {
            slot.xor_key.store(0, Ordering::Relaxed);
            slot.data.store(0, Ordering::Relaxed);
        }
    }

    #[inline]
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
            if e.generation == self.generation() && depth as i8 <= e.depth && bound != BOUND_EXACT {
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
            self.generation(),
        );
        slot.data.store(data, Ordering::Relaxed);
        slot.xor_key.store(hash ^ data, Ordering::Relaxed);
    }
}

// TT is Sync automatically (all fields are atomics), so `&TT` can be shared
// freely across scoped search threads.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_entry_roundtrips_generation() {
        let entry = unpack(pack(123, -45, 12, BOUND_LOWER, 9));
        assert_eq!(entry.mv, 123);
        assert_eq!(entry.score, -45);
        assert_eq!(entry.depth, 12);
        assert_eq!(entry.bound, BOUND_LOWER);
        assert_eq!(entry.generation, 9);
    }

    #[test]
    fn old_generation_is_replaceable_and_hashfull_tracks_current() {
        let table = TT::new(1);
        let hash = 0; // sampled slot zero
        table.store(hash, Move(7), 10, 20, BOUND_LOWER);
        assert_eq!(table.probe(hash).unwrap().depth, 20);
        assert!(table.hashfull() > 0);
        table.new_search();
        assert_eq!(table.hashfull(), 0);
        table.store(hash, Move(8), 20, 1, BOUND_UPPER);
        let entry = table.probe(hash).unwrap();
        assert_eq!(entry.depth, 1);
        assert_eq!(entry.mv, 8);
    }
}
