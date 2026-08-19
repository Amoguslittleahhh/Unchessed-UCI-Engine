//! Move generation: const attack tables, magic-bitboard slider attacks,
//! pseudo-legal generation + legality filtering.

use crate::board::*;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Attack tables (built at compile time)
// ---------------------------------------------------------------------------

// Direction indices for RAYS. Positive = target index increases along ray.
pub const DIR_N: usize = 0; // +8
pub const DIR_NE: usize = 1; // +9
pub const DIR_E: usize = 2; // +1
pub const DIR_SE: usize = 3; // -7
pub const DIR_S: usize = 4; // -8
pub const DIR_SW: usize = 5; // -9
pub const DIR_W: usize = 6; // -1
pub const DIR_NW: usize = 7; // +7

const DIR_DELTAS: [(i32, i32); 8] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
];

const fn build_rays() -> [[Bitboard; 64]; 8] {
    let mut rays = [[0u64; 64]; 8];
    let mut d = 0;
    while d < 8 {
        let (df, dr) = DIR_DELTAS[d];
        let mut s = 0i32;
        while s < 64 {
            let mut f = (s & 7) + df;
            let mut r = (s >> 3) + dr;
            let mut bb = 0u64;
            while f >= 0 && f < 8 && r >= 0 && r < 8 {
                bb |= 1u64 << (r * 8 + f);
                f += df;
                r += dr;
            }
            rays[d][s as usize] = bb;
            s += 1;
        }
        d += 1;
    }
    rays
}

const fn build_leaper(deltas: &[(i32, i32)]) -> [Bitboard; 64] {
    let mut t = [0u64; 64];
    let mut s = 0i32;
    while s < 64 {
        let f0 = s & 7;
        let r0 = s >> 3;
        let mut bb = 0u64;
        let mut i = 0;
        while i < deltas.len() {
            let f = f0 + deltas[i].0;
            let r = r0 + deltas[i].1;
            if f >= 0 && f < 8 && r >= 0 && r < 8 {
                bb |= 1u64 << (r * 8 + f);
            }
            i += 1;
        }
        t[s as usize] = bb;
        s += 1;
    }
    t
}

const fn build_pawn_attacks() -> [[Bitboard; 64]; 2] {
    let mut t = [[0u64; 64]; 2];
    let mut s = 0i32;
    while s < 64 {
        let f = s & 7;
        let r = s >> 3;
        let mut w = 0u64;
        let mut b = 0u64;
        if r + 1 < 8 {
            if f - 1 >= 0 {
                w |= 1u64 << ((r + 1) * 8 + f - 1);
            }
            if f + 1 < 8 {
                w |= 1u64 << ((r + 1) * 8 + f + 1);
            }
        }
        if r - 1 >= 0 {
            if f - 1 >= 0 {
                b |= 1u64 << ((r - 1) * 8 + f - 1);
            }
            if f + 1 < 8 {
                b |= 1u64 << ((r - 1) * 8 + f + 1);
            }
        }
        t[0][s as usize] = w;
        t[1][s as usize] = b;
        s += 1;
    }
    t
}

pub static RAYS: [[Bitboard; 64]; 8] = build_rays();
pub static KNIGHT_ATT: [Bitboard; 64] = build_leaper(&[
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
]);
pub static KING_ATT: [Bitboard; 64] = build_leaper(&[
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
]);
pub static PAWN_ATT: [[Bitboard; 64]; 2] = build_pawn_attacks();

pub const FILE_A: Bitboard = 0x0101_0101_0101_0101;
pub const FILE_H: Bitboard = FILE_A << 7;
pub const RANK_1: Bitboard = 0xFF;
pub const RANK_3: Bitboard = RANK_1 << 16;
pub const RANK_6: Bitboard = RANK_1 << 40;
pub const RANK_8: Bitboard = RANK_1 << 56;

// ---------------------------------------------------------------------------
// Slider attacks: classical ray approach (kept as the correctness oracle
// magic-table construction is verified against, and as a fallback reference
// in tests) plus the magic-bitboard tables actually used by search.
// ---------------------------------------------------------------------------

#[inline]
fn ray_pos(d: usize, s: u8, occ: Bitboard) -> Bitboard {
    let ray = RAYS[d][s as usize];
    let blockers = ray & occ;
    if blockers != 0 {
        let first = blockers.trailing_zeros() as usize;
        ray ^ RAYS[d][first]
    } else {
        ray
    }
}

#[inline]
fn ray_neg(d: usize, s: u8, occ: Bitboard) -> Bitboard {
    let ray = RAYS[d][s as usize];
    let blockers = ray & occ;
    if blockers != 0 {
        let first = (63 - blockers.leading_zeros()) as usize;
        ray ^ RAYS[d][first]
    } else {
        ray
    }
}

#[inline]
fn bishop_att_classical(s: u8, occ: Bitboard) -> Bitboard {
    ray_pos(DIR_NE, s, occ)
        | ray_pos(DIR_NW, s, occ)
        | ray_neg(DIR_SE, s, occ)
        | ray_neg(DIR_SW, s, occ)
}

#[inline]
fn rook_att_classical(s: u8, occ: Bitboard) -> Bitboard {
    ray_pos(DIR_N, s, occ)
        | ray_pos(DIR_E, s, occ)
        | ray_neg(DIR_S, s, occ)
        | ray_neg(DIR_W, s, occ)
}

// ---------------------------------------------------------------------------
// Magic bitboards: O(1) table lookup instead of ray scanning. Magic numbers
// are found at first use (not embedded as literals) by random search,
// verified against the classical functions above as ground truth for every
// occupancy subset -- this makes the search self-checking: it is
// structurally impossible for a "found" magic to produce a wrong attack set,
// since a collision against a differing attack value is rejected and
// retried. Mirrors Stockfish's own approach (it also searches at startup
// rather than embedding a fixed magic table).
// ---------------------------------------------------------------------------

/// Relevant occupancy mask for a slider on `s`: the full ray in each
/// direction with the outermost (board-edge) square removed, since a piece
/// there can never change the attack set (the ray already terminates at the
/// board edge regardless of whether it's occupied).
fn edge_trim(positive_dir: bool, ray: Bitboard) -> Bitboard {
    if ray == 0 {
        return 0;
    }
    if positive_dir {
        let hi = 63 - ray.leading_zeros();
        ray & !(1u64 << hi)
    } else {
        let lo = ray.trailing_zeros();
        ray & !(1u64 << lo)
    }
}

fn relevant_bishop_mask(s: u8) -> Bitboard {
    edge_trim(true, RAYS[DIR_NE][s as usize])
        | edge_trim(true, RAYS[DIR_NW][s as usize])
        | edge_trim(false, RAYS[DIR_SE][s as usize])
        | edge_trim(false, RAYS[DIR_SW][s as usize])
}

fn relevant_rook_mask(s: u8) -> Bitboard {
    edge_trim(true, RAYS[DIR_N][s as usize])
        | edge_trim(true, RAYS[DIR_E][s as usize])
        | edge_trim(false, RAYS[DIR_S][s as usize])
        | edge_trim(false, RAYS[DIR_W][s as usize])
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// AND of three random draws biases toward sparse bit patterns, which tend
/// to make better magics (standard trick, same one Stockfish uses).
#[inline]
fn sparse_rand(state: &mut u64) -> u64 {
    splitmix64(state) & splitmix64(state) & splitmix64(state)
}

/// Enumerate every subset of `mask` via the carry-rippler trick (returns
/// 2^popcount(mask) subsets, including 0, terminating back at 0).
fn subsets_of(mask: Bitboard) -> Vec<Bitboard> {
    let mut out = Vec::with_capacity(1 << mask.count_ones());
    let mut subset: Bitboard = 0;
    loop {
        out.push(subset);
        subset = subset.wrapping_sub(mask) & mask;
        if subset == 0 {
            break;
        }
    }
    out
}

#[derive(Clone, Copy)]
struct MagicEntry {
    mask: Bitboard,
    magic: u64,
    shift: u32,
    offset: usize,
}

struct MagicTables {
    rook: [MagicEntry; 64],
    bishop: [MagicEntry; 64],
    table: Vec<Bitboard>,
}

/// Search for a working magic for one square + mask, verified against the
/// classical attack function for every occupancy subset of the mask. Two
/// subsets are allowed to collide on the same table index only if they
/// produce the identical attack set; any other collision rejects the magic
/// and retries with a new random candidate.
fn find_magic(
    sq: u8,
    mask: Bitboard,
    classical: fn(u8, Bitboard) -> Bitboard,
    rng: &mut u64,
) -> (u64, Vec<Bitboard>) {
    let bits = mask.count_ones();
    let size = 1usize << bits;
    let shift = 64 - bits;
    let subsets = subsets_of(mask);
    let attacks: Vec<Bitboard> = subsets.iter().map(|&occ| classical(sq, occ)).collect();

    loop {
        let magic = sparse_rand(rng);
        // quality filter: reject magics that don't spread bits well into the
        // top byte of mask*magic (same heuristic Stockfish uses)
        if (mask.wrapping_mul(magic) >> 56).count_ones() < 6 {
            continue;
        }
        let mut table = vec![0u64; size];
        let mut used = vec![false; size];
        let mut ok = true;
        for (i, &occ) in subsets.iter().enumerate() {
            let idx = ((occ.wrapping_mul(magic)) >> shift) as usize;
            if used[idx] {
                if table[idx] != attacks[i] {
                    ok = false;
                    break;
                }
            } else {
                used[idx] = true;
                table[idx] = attacks[i];
            }
        }
        if ok {
            return (magic, table);
        }
    }
}

fn build_magics() -> MagicTables {
    let mut rng: u64 = 0x00C0_FFEE_D00D_2025;
    let mut table: Vec<Bitboard> = Vec::new();
    let mut rook = [MagicEntry {
        mask: 0,
        magic: 0,
        shift: 0,
        offset: 0,
    }; 64];
    let mut bishop = [MagicEntry {
        mask: 0,
        magic: 0,
        shift: 0,
        offset: 0,
    }; 64];

    for s in 0u8..64 {
        let mask = relevant_rook_mask(s);
        let (magic, attacks) = find_magic(s, mask, rook_att_classical, &mut rng);
        let offset = table.len();
        table.extend_from_slice(&attacks);
        rook[s as usize] = MagicEntry {
            mask,
            magic,
            shift: 64 - mask.count_ones(),
            offset,
        };
    }
    for s in 0u8..64 {
        let mask = relevant_bishop_mask(s);
        let (magic, attacks) = find_magic(s, mask, bishop_att_classical, &mut rng);
        let offset = table.len();
        table.extend_from_slice(&attacks);
        bishop[s as usize] = MagicEntry {
            mask,
            magic,
            shift: 64 - mask.count_ones(),
            offset,
        };
    }

    MagicTables {
        rook,
        bishop,
        table,
    }
}

static MAGICS: OnceLock<MagicTables> = OnceLock::new();

#[inline]
fn magics() -> &'static MagicTables {
    MAGICS.get_or_init(build_magics)
}

#[inline]
pub fn bishop_att(s: u8, occ: Bitboard) -> Bitboard {
    let m = magics();
    let e = &m.bishop[s as usize];
    let idx = ((occ & e.mask).wrapping_mul(e.magic) >> e.shift) as usize;
    m.table[e.offset + idx]
}

#[inline]
pub fn rook_att(s: u8, occ: Bitboard) -> Bitboard {
    let m = magics();
    let e = &m.rook[s as usize];
    let idx = ((occ & e.mask).wrapping_mul(e.magic) >> e.shift) as usize;
    m.table[e.offset + idx]
}

#[inline]
pub fn queen_att(s: u8, occ: Bitboard) -> Bitboard {
    bishop_att(s, occ) | rook_att(s, occ)
}

/// Is `s` attacked by side `by`?
pub fn attacked(pos: &Position, s: u8, by: Color) -> bool {
    let b = by.idx();
    // A square is attacked by a `by`-pawn if such a pawn stands on the squares
    // a defender pawn of the other color would attack from `s`.
    if PAWN_ATT[by.flip().idx()][s as usize] & pos.bb[b][PAWN] != 0 {
        return true;
    }
    if KNIGHT_ATT[s as usize] & pos.bb[b][KNIGHT] != 0 {
        return true;
    }
    if KING_ATT[s as usize] & pos.bb[b][KING] != 0 {
        return true;
    }
    let diag = pos.bb[b][BISHOP] | pos.bb[b][QUEEN];
    if diag != 0 && bishop_att(s, pos.occ) & diag != 0 {
        return true;
    }
    let ortho = pos.bb[b][ROOK] | pos.bb[b][QUEEN];
    if ortho != 0 && rook_att(s, pos.occ) & ortho != 0 {
        return true;
    }
    false
}

/// Is the side to move in check?
#[inline]
pub fn in_check(pos: &Position) -> bool {
    attacked(pos, pos.king_sq(pos.side), pos.side.flip())
}

/// After `make`, was the move legal for the side that just moved?
#[inline]
pub fn king_safe_after(pos_after: &Position, mover: Color) -> bool {
    !attacked(pos_after, pos_after.king_sq(mover), pos_after.side)
}

/// Squares strictly between `a` and `b` (exclusive) if they share a rank,
/// file, or diagonal; 0 otherwise. Used by `pinned_blockers` below to find
/// the single piece (if any) sitting between a king and an aligned enemy
/// slider.
const fn build_between() -> [[u64; 64]; 64] {
    let mut t = [[0u64; 64]; 64];
    let mut a = 0usize;
    while a < 64 {
        let af = (a % 8) as i32;
        let ar = (a / 8) as i32;
        let mut b = 0usize;
        while b < 64 {
            if a != b {
                let bf = (b % 8) as i32;
                let br = (b / 8) as i32;
                let df = bf - af;
                let dr = br - ar;
                if df == 0 || dr == 0 || df == dr || df == -dr {
                    let step_f = if df == 0 {
                        0
                    } else if df > 0 {
                        1
                    } else {
                        -1
                    };
                    let step_r = if dr == 0 {
                        0
                    } else if dr > 0 {
                        1
                    } else {
                        -1
                    };
                    let mut bb = 0u64;
                    let mut f = af + step_f;
                    let mut r = ar + step_r;
                    while f != bf || r != br {
                        bb |= 1u64 << ((r * 8 + f) as usize);
                        f += step_f;
                        r += step_r;
                    }
                    t[a][b] = bb;
                }
            }
            b += 1;
        }
        a += 1;
    }
    t
}

static BETWEEN: [[u64; 64]; 64] = build_between();

/// For `us`'s king: the set of `us`'s own pieces that, if moved, would
/// expose the king to a slider check (`blockers`), and the enemy sliders
/// doing the pinning (`pinners`). Candidate "snipers" are found via a
/// zero-occupancy ray from the king square (so they see through everything,
/// matching every reference engine's real `attacks_bb(ksq, 0)` pattern);
/// a sniper is a genuine pinner only if exactly one piece (of either color)
/// sits strictly between it and the king.
pub fn pinned_blockers(pos: &Position, us: Color) -> (Bitboard, Bitboard) {
    let them = us.flip();
    let ksq = pos.king_sq(us) as usize;
    let mut blockers = 0u64;
    let mut pinners = 0u64;

    let snipers = (rook_att(ksq as u8, 0) & (pos.bb[them.idx()][ROOK] | pos.bb[them.idx()][QUEEN]))
        | (bishop_att(ksq as u8, 0) & (pos.bb[them.idx()][BISHOP] | pos.bb[them.idx()][QUEEN]));
    let occ_without_snipers = pos.occ & !snipers;

    let mut s = snipers;
    while s != 0 {
        let sq = s.trailing_zeros() as usize;
        s &= s - 1;
        let between = BETWEEN[ksq][sq] & occ_without_snipers;
        if between != 0 && (between & (between - 1)) == 0 {
            blockers |= between;
            pinners |= 1u64 << sq;
        }
    }
    (blockers, pinners)
}

// ---------------------------------------------------------------------------
// Move list
// ---------------------------------------------------------------------------

pub struct MoveList {
    pub moves: [Move; 256],
    pub len: usize,
}

impl MoveList {
    #[inline]
    pub fn new() -> MoveList {
        MoveList {
            moves: [Move::NONE; 256],
            len: 0,
        }
    }
    #[inline]
    pub fn push(&mut self, m: Move) {
        self.moves[self.len] = m;
        self.len += 1;
    }
    #[inline]
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

#[inline]
fn push_promos(list: &mut MoveList, from: u8, to: u8) {
    list.push(Move::new_promo(from, to, QUEEN));
    list.push(Move::new_promo(from, to, KNIGHT));
    list.push(Move::new_promo(from, to, ROOK));
    list.push(Move::new_promo(from, to, BISHOP));
}

/// Pseudo-legal move generation. `caps_only` = captures + promotions (for qsearch).
pub fn generate(pos: &Position, caps_only: bool, list: &mut MoveList) {
    let us = pos.side;
    let u = us.idx();
    let them = us.flip();
    let own = pos.occ_side[u];
    let enemy = pos.occ_side[them.idx()];
    let occ = pos.occ;
    let targets = if caps_only { enemy } else { !own };

    // --- pawns ---
    let pawns = pos.bb[u][PAWN];
    if let Color::White = us {
        let singles = (pawns << 8) & !occ;
        let promo_push = singles & RANK_8;
        let mut b = promo_push;
        while b != 0 {
            let to = b.trailing_zeros() as u8;
            b &= b - 1;
            push_promos(list, to - 8, to);
        }
        if !caps_only {
            let mut b = singles & !RANK_8;
            while b != 0 {
                let to = b.trailing_zeros() as u8;
                b &= b - 1;
                list.push(Move::new(to - 8, to, MK_NORMAL));
            }
            let mut b = ((singles & RANK_3) << 8) & !occ;
            while b != 0 {
                let to = b.trailing_zeros() as u8;
                b &= b - 1;
                list.push(Move::new(to - 16, to, MK_NORMAL));
            }
        }
        let left = ((pawns & !FILE_A) << 7) & enemy;
        let mut b = left & !RANK_8;
        while b != 0 {
            let to = b.trailing_zeros() as u8;
            b &= b - 1;
            list.push(Move::new(to - 7, to, MK_NORMAL));
        }
        let mut b = left & RANK_8;
        while b != 0 {
            let to = b.trailing_zeros() as u8;
            b &= b - 1;
            push_promos(list, to - 7, to);
        }
        let right = ((pawns & !FILE_H) << 9) & enemy;
        let mut b = right & !RANK_8;
        while b != 0 {
            let to = b.trailing_zeros() as u8;
            b &= b - 1;
            list.push(Move::new(to - 9, to, MK_NORMAL));
        }
        let mut b = right & RANK_8;
        while b != 0 {
            let to = b.trailing_zeros() as u8;
            b &= b - 1;
            push_promos(list, to - 9, to);
        }
    } else {
        let singles = (pawns >> 8) & !occ;
        let promo_push = singles & RANK_1;
        let mut b = promo_push;
        while b != 0 {
            let to = b.trailing_zeros() as u8;
            b &= b - 1;
            push_promos(list, to + 8, to);
        }
        if !caps_only {
            let mut b = singles & !RANK_1;
            while b != 0 {
                let to = b.trailing_zeros() as u8;
                b &= b - 1;
                list.push(Move::new(to + 8, to, MK_NORMAL));
            }
            let mut b = ((singles & RANK_6) >> 8) & !occ;
            while b != 0 {
                let to = b.trailing_zeros() as u8;
                b &= b - 1;
                list.push(Move::new(to + 16, to, MK_NORMAL));
            }
        }
        let left = ((pawns & !FILE_H) >> 7) & enemy;
        let mut b = left & !RANK_1;
        while b != 0 {
            let to = b.trailing_zeros() as u8;
            b &= b - 1;
            list.push(Move::new(to + 7, to, MK_NORMAL));
        }
        let mut b = left & RANK_1;
        while b != 0 {
            let to = b.trailing_zeros() as u8;
            b &= b - 1;
            push_promos(list, to + 7, to);
        }
        let right = ((pawns & !FILE_A) >> 9) & enemy;
        let mut b = right & !RANK_1;
        while b != 0 {
            let to = b.trailing_zeros() as u8;
            b &= b - 1;
            list.push(Move::new(to + 9, to, MK_NORMAL));
        }
        let mut b = right & RANK_1;
        while b != 0 {
            let to = b.trailing_zeros() as u8;
            b &= b - 1;
            push_promos(list, to + 9, to);
        }
    }

    // en passant (counts as a capture)
    if pos.ep != NO_EP {
        let mut b = PAWN_ATT[them.idx()][pos.ep as usize] & pawns;
        while b != 0 {
            let from = b.trailing_zeros() as u8;
            b &= b - 1;
            list.push(Move::new(from, pos.ep, MK_EP));
        }
    }

    // --- knights ---
    let mut b = pos.bb[u][KNIGHT];
    while b != 0 {
        let from = b.trailing_zeros() as u8;
        b &= b - 1;
        let mut att = KNIGHT_ATT[from as usize] & targets;
        while att != 0 {
            let to = att.trailing_zeros() as u8;
            att &= att - 1;
            list.push(Move::new(from, to, MK_NORMAL));
        }
    }

    // --- sliders ---
    let mut b = pos.bb[u][BISHOP];
    while b != 0 {
        let from = b.trailing_zeros() as u8;
        b &= b - 1;
        let mut att = bishop_att(from, occ) & targets;
        while att != 0 {
            let to = att.trailing_zeros() as u8;
            att &= att - 1;
            list.push(Move::new(from, to, MK_NORMAL));
        }
    }
    let mut b = pos.bb[u][ROOK];
    while b != 0 {
        let from = b.trailing_zeros() as u8;
        b &= b - 1;
        let mut att = rook_att(from, occ) & targets;
        while att != 0 {
            let to = att.trailing_zeros() as u8;
            att &= att - 1;
            list.push(Move::new(from, to, MK_NORMAL));
        }
    }
    let mut b = pos.bb[u][QUEEN];
    while b != 0 {
        let from = b.trailing_zeros() as u8;
        b &= b - 1;
        let mut att = queen_att(from, occ) & targets;
        while att != 0 {
            let to = att.trailing_zeros() as u8;
            att &= att - 1;
            list.push(Move::new(from, to, MK_NORMAL));
        }
    }

    // --- king ---
    let ksq = pos.king_sq(us);
    let mut att = KING_ATT[ksq as usize] & targets;
    while att != 0 {
        let to = att.trailing_zeros() as u8;
        att &= att - 1;
        list.push(Move::new(ksq, to, MK_NORMAL));
    }

    // --- castling ---
    if !caps_only {
        if let Color::White = us {
            if pos.castling & WK != 0
                && pos.piece_on(4) == Some((Color::White, KING))
                && pos.piece_on(7) == Some((Color::White, ROOK))
                && occ & 0x60 == 0
                && !attacked(pos, 4, them)
                && !attacked(pos, 5, them)
                && !attacked(pos, 6, them)
            {
                list.push(Move::new(4, 6, MK_CASTLE));
            }
            if pos.castling & WQ != 0
                && pos.piece_on(4) == Some((Color::White, KING))
                && pos.piece_on(0) == Some((Color::White, ROOK))
                && occ & 0x0E == 0
                && !attacked(pos, 4, them)
                && !attacked(pos, 3, them)
                && !attacked(pos, 2, them)
            {
                list.push(Move::new(4, 2, MK_CASTLE));
            }
        } else {
            if pos.castling & BK != 0
                && pos.piece_on(60) == Some((Color::Black, KING))
                && pos.piece_on(63) == Some((Color::Black, ROOK))
                && occ & (0x60u64 << 56) == 0
                && !attacked(pos, 60, them)
                && !attacked(pos, 61, them)
                && !attacked(pos, 62, them)
            {
                list.push(Move::new(60, 62, MK_CASTLE));
            }
            if pos.castling & BQ != 0
                && pos.piece_on(60) == Some((Color::Black, KING))
                && pos.piece_on(56) == Some((Color::Black, ROOK))
                && occ & (0x0Eu64 << 56) == 0
                && !attacked(pos, 60, them)
                && !attacked(pos, 59, them)
                && !attacked(pos, 58, them)
            {
                list.push(Move::new(60, 58, MK_CASTLE));
            }
        }
    }
}

/// Fully legal moves.
pub fn legal(pos: &Position) -> MoveList {
    let mut pseudo = MoveList::new();
    generate(pos, false, &mut pseudo);
    let mut out = MoveList::new();
    let us = pos.side;
    for &m in pseudo.as_slice() {
        let next = pos.make(m);
        if king_safe_after(&next, us) {
            out.push(m);
        }
    }
    out
}

/// Parse a UCI move string against the legal moves of `pos`.
pub fn parse_uci_move(pos: &Position, s: &str) -> Option<Move> {
    let ml = legal(pos);
    ml.as_slice().iter().copied().find(|m| m.uci() == s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Magic-bitboard attacks must agree with the classical ray-scan
    /// attacks for every square, across many random occupancies -- this is
    /// the correctness net for the magic search: find_magic() already
    /// rejects any magic that disagrees with the classical function on any
    /// occupancy subset of the *relevant mask*, but this test additionally
    /// exercises full-board occupancies (bits outside the mask too), which
    /// must be masked off correctly by `occ & e.mask` in bishop_att/rook_att.
    #[test]
    fn magic_attacks_match_classical_ray_scan() {
        let mut rng: u64 = 0xABCD_1234_5678_9EF0;
        for s in 0u8..64 {
            for _ in 0..2000 {
                let occ = splitmix64(&mut rng);
                assert_eq!(
                    bishop_att(s, occ),
                    bishop_att_classical(s, occ),
                    "bishop attacks mismatch at square {s} for occ {occ:#018x}"
                );
                assert_eq!(
                    rook_att(s, occ),
                    rook_att_classical(s, occ),
                    "rook attacks mismatch at square {s} for occ {occ:#018x}"
                );
            }
        }
    }

    #[test]
    fn magic_table_sizes_match_expected_ballpark() {
        // sanity check the table isn't wildly oversized/undersized -- rook
        // is ~800KiB, bishop ~41KiB at 8 bytes/entry, per the known
        // reference figures for this masking scheme.
        let m = magics();
        let rook_entries: usize = m.rook.iter().map(|e| 1usize << (64 - e.shift)).sum();
        let bishop_entries: usize = m.bishop.iter().map(|e| 1usize << (64 - e.shift)).sum();
        assert!(
            rook_entries >= 90_000 && rook_entries <= 110_000,
            "rook table size {rook_entries} out of expected range"
        );
        assert!(
            bishop_entries >= 4_000 && bishop_entries <= 6_000,
            "bishop table size {bishop_entries} out of expected range"
        );
    }
}
