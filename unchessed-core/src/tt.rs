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

// TT is Sync automatically (all fields are atomics), so `&TT` can be shared
// freely across scoped search threads.
