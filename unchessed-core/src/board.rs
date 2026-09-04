//! Board representation: bitboards + mailbox, copy-make, Zobrist hashing.
//!
//! Square mapping: a1 = 0, b1 = 1, ... h8 = 63 (little-endian rank-file).

pub type Bitboard = u64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }
    #[inline]
    pub fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

pub const PAWN: usize = 0;
pub const KNIGHT: usize = 1;
pub const BISHOP: usize = 2;
pub const ROOK: usize = 3;
pub const QUEEN: usize = 4;
pub const KING: usize = 5;

pub const NO_PIECE: u8 = 0xFF;

// Castling right bits.
pub const WK: u8 = 1;
pub const WQ: u8 = 2;
pub const BK: u8 = 4;
pub const BQ: u8 = 8;

pub const NO_EP: u8 = 64;

#[inline]
pub fn sq(file: u8, rank: u8) -> u8 {
    rank * 8 + file
}

#[inline]
pub fn file_of(s: u8) -> u8 {
    s & 7
}

#[inline]
pub fn rank_of(s: u8) -> u8 {
    s >> 3
}

pub fn sq_name(s: u8) -> String {
    let f = (b'a' + file_of(s)) as char;
    let r = (b'1' + rank_of(s)) as char;
    format!("{}{}", f, r)
}

pub fn parse_sq(s: &str) -> Option<u8> {
    let b = s.as_bytes();
    if b.len() < 2 {
        return None;
    }
    let f = b[0].wrapping_sub(b'a');
    let r = b[1].wrapping_sub(b'1');
    if f < 8 && r < 8 {
        Some(sq(f, r))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Moves
// ---------------------------------------------------------------------------

pub const MK_NORMAL: u16 = 0;
pub const MK_PROMO: u16 = 1;
pub const MK_EP: u16 = 2;
pub const MK_CASTLE: u16 = 3;

/// bits 0-5 from, 6-11 to, 12-13 promo piece (0=N .. 3=Q), 14-15 kind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move(pub u16);

impl Move {
    pub const NONE: Move = Move(0);

    #[inline]
    pub fn new(from: u8, to: u8, kind: u16) -> Move {
        Move(from as u16 | (to as u16) << 6 | kind << 14)
    }

    #[inline]
    pub fn new_promo(from: u8, to: u8, piece: usize) -> Move {
        Move(from as u16 | (to as u16) << 6 | ((piece - KNIGHT) as u16) << 12 | MK_PROMO << 14)
    }

    #[inline]
    pub fn from(self) -> u8 {
        (self.0 & 63) as u8
    }
    #[inline]
    pub fn to(self) -> u8 {
        ((self.0 >> 6) & 63) as u8
    }
    #[inline]
    pub fn kind(self) -> u16 {
        self.0 >> 14
    }
    #[inline]
    pub fn promo_piece(self) -> usize {
        KNIGHT + ((self.0 >> 12) & 3) as usize
    }
    #[inline]
    pub fn is_promo(self) -> bool {
        self.kind() == MK_PROMO
    }

    /// UCI long algebraic string, e.g. "e2e4", "e7e8q".
    pub fn uci(self) -> String {
        if self == Move::NONE {
            return "0000".to_string();
        }
        let mut s = format!("{}{}", sq_name(self.from()), sq_name(self.to()));
        if self.is_promo() {
            s.push(match self.promo_piece() {
                KNIGHT => 'n',
                BISHOP => 'b',
                ROOK => 'r',
                _ => 'q',
            });
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Zobrist keys (const-generated with splitmix64)
// ---------------------------------------------------------------------------

pub struct Zobrist {
    pub psq: [[[u64; 64]; 6]; 2],
    pub side: u64,
    pub castle: [u64; 16],
    pub ep_file: [u64; 8],
}

const fn sm_next(state: u64) -> (u64, u64) {
    let s = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (s, z ^ (z >> 31))
}

const fn build_zobrist() -> Zobrist {
    let mut st: u64 = 0x00C0_FFEE_D00D_2024;
    let mut psq = [[[0u64; 64]; 6]; 2];
    let mut c = 0;
    while c < 2 {
        let mut p = 0;
        while p < 6 {
            let mut s = 0;
            while s < 64 {
                let (ns, v) = sm_next(st);
                st = ns;
                psq[c][p][s] = v;
                s += 1;
            }
            p += 1;
        }
        c += 1;
    }
    let (ns, side) = sm_next(st);
    st = ns;
    let mut castle = [0u64; 16];
    let mut i = 0;
    while i < 16 {
        let (ns, v) = sm_next(st);
        st = ns;
        castle[i] = v;
        i += 1;
    }
    // castle[0] = 0 so "no rights" contributes nothing; keeps hashes tidy.
    castle[0] = 0;
    let mut ep_file = [0u64; 8];
    let mut i = 0;
    while i < 8 {
        let (ns, v) = sm_next(st);
        st = ns;
        ep_file[i] = v;
        i += 1;
    }
    Zobrist {
        psq,
        side,
        castle,
        ep_file,
    }
}

pub static ZOBRIST: Zobrist = build_zobrist();

/// castling &= CASTLE_MASK[from] & CASTLE_MASK[to]
const fn build_castle_mask() -> [u8; 64] {
    let mut m = [15u8; 64];
    m[4] = 15 & !(WK | WQ); // e1
    m[0] = 15 & !WQ; // a1
    m[7] = 15 & !WK; // h1
    m[60] = 15 & !(BK | BQ); // e8
    m[56] = 15 & !BQ; // a8
    m[63] = 15 & !BK; // h8
    m
}

pub static CASTLE_MASK: [u8; 64] = build_castle_mask();

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Position {
    pub bb: [[Bitboard; 6]; 2],
    pub occ_side: [Bitboard; 2],
    pub occ: Bitboard,
    /// mailbox: color*6 + piece, or NO_PIECE
    pub board: [u8; 64],
    pub side: Color,
    pub castling: u8,
    pub ep: u8, // NO_EP if none
    pub halfmove: u16,
    pub fullmove: u16,
    pub hash: u64,
}

impl Position {
    pub fn empty() -> Position {
        Position {
            bb: [[0; 6]; 2],
            occ_side: [0; 2],
            occ: 0,
            board: [NO_PIECE; 64],
            side: Color::White,
            castling: 0,
            ep: NO_EP,
            halfmove: 0,
            fullmove: 1,
            hash: 0,
        }
    }

    #[inline]
    pub fn pieces(&self, c: Color, p: usize) -> Bitboard {
        self.bb[c.idx()][p]
    }

    #[inline]
    pub fn king_sq(&self, c: Color) -> u8 {
        self.bb[c.idx()][KING].trailing_zeros() as u8
    }

    /// (color, piece) on a square, if any.
    #[inline]
    pub fn piece_on(&self, s: u8) -> Option<(Color, usize)> {
        let v = self.board[s as usize];
        if v == NO_PIECE {
            None
        } else {
            let c = if v < 6 { Color::White } else { Color::Black };
            Some((c, (v % 6) as usize))
        }
    }

    #[inline]
    pub fn has_non_pawn(&self, c: Color) -> bool {
        self.occ_side[c.idx()] != self.bb[c.idx()][PAWN] | self.bb[c.idx()][KING]
    }

    #[inline]
    fn put(&mut self, c: Color, p: usize, s: u8) {
        let b = 1u64 << s;
        self.bb[c.idx()][p] |= b;
        self.occ_side[c.idx()] |= b;
        self.occ |= b;
        self.board[s as usize] = (c.idx() * 6 + p) as u8;
        self.hash ^= ZOBRIST.psq[c.idx()][p][s as usize];
    }

    #[inline]
    fn take(&mut self, c: Color, p: usize, s: u8) {
        let b = 1u64 << s;
        self.bb[c.idx()][p] &= !b;
        self.occ_side[c.idx()] &= !b;
        self.occ &= !b;
        self.board[s as usize] = NO_PIECE;
        self.hash ^= ZOBRIST.psq[c.idx()][p][s as usize];
    }

    /// Recompute the hash from scratch (used after FEN parsing and in tests).
    /// The Zobrist contribution `ep` should make to the hash, given who's
    /// to move: zero unless a real en-passant capture is actually available
    /// (a real pawn of `side_to_move`'s color sits on an adjacent file, on
    /// the rank the double-moved pawn landed on). `ep` itself is still
    /// recorded on the position (and in FEN/UCI output) whenever a double
    /// push happens, per normal chess notation -- this only controls
    /// whether that flag also perturbs the hash. Without this, two
    /// positions that are identical in every rules-relevant way except one
    /// having a dead ep flag (no pawn able to actually use it) get
    /// different hashes, which can fragment TT identity and hide a real
    /// repetition behind an irrelevant difference.
    fn ep_hash_contribution(bb: &[[Bitboard; 6]; 2], side_to_move: Color, ep: u8) -> u64 {
        let ep_rank = rank_of(ep);
        let mover_rank = if let Color::White = side_to_move {
            ep_rank - 1
        } else {
            ep_rank + 1
        };
        let ep_file_idx = file_of(ep);
        let capturer_pawns = bb[side_to_move.idx()][PAWN];
        let has_capture = [ep_file_idx.wrapping_sub(1), ep_file_idx + 1]
            .into_iter()
            .any(|f| f < 8 && capturer_pawns & (1u64 << sq(f, mover_rank)) != 0);
        if has_capture {
            ZOBRIST.ep_file[ep_file_idx as usize]
        } else {
            0
        }
    }

    pub fn compute_hash(&self) -> u64 {
        let mut h = 0u64;
        for s in 0..64u8 {
            if let Some((c, p)) = self.piece_on(s) {
                h ^= ZOBRIST.psq[c.idx()][p][s as usize];
            }
        }
        h ^= ZOBRIST.castle[self.castling as usize];
        if self.ep != NO_EP {
            h ^= Self::ep_hash_contribution(&self.bb, self.side, self.ep);
        }
        if let Color::Black = self.side {
            h ^= ZOBRIST.side;
        }
        h
    }

    /// Copy-make: returns the position after the move (assumed pseudo-legal).
    pub fn make(&self, mv: Move) -> Position {
        let mut p = *self;
        let us = p.side;
        let them = us.flip();
        let from = mv.from();
        let to = mv.to();
        let (_, pt) = p.piece_on(from).expect("make: no piece on from-square");

        // clear old en passant (board/side unchanged from `self` so far --
        // recompute the same contribution compute_hash() would have made)
        if p.ep != NO_EP {
            p.hash ^= Self::ep_hash_contribution(&p.bb, us, p.ep);
            p.ep = NO_EP;
        }

        let mut capture = false;
        match mv.kind() {
            MK_CASTLE => {
                // king from->to; move the rook too
                let (rf, rt) = if to > from {
                    (to + 1, to - 1)
                } else {
                    (to - 2, to + 1)
                };
                p.take(us, KING, from);
                p.put(us, KING, to);
                p.take(us, ROOK, rf);
                p.put(us, ROOK, rt);
            }
            MK_EP => {
                let cap_sq = if let Color::White = us { to - 8 } else { to + 8 };
                p.take(them, PAWN, cap_sq);
                p.take(us, PAWN, from);
                p.put(us, PAWN, to);
                capture = true;
            }
            _ => {
                if let Some((cc, cp)) = p.piece_on(to) {
                    debug_assert!(cc == them);
                    p.take(cc, cp, to);
                    capture = true;
                }
                p.take(us, pt, from);
                if mv.is_promo() {
                    p.put(us, mv.promo_piece(), to);
                } else {
                    p.put(us, pt, to);
                }
                // double pawn push sets ep square
                if pt == PAWN && (from as i16 - to as i16).abs() == 16 {
                    let ep_sq = ((from as u16 + to as u16) / 2) as u8;
                    p.ep = ep_sq;
                    // `p.bb` already reflects this pawn's move (placed above),
                    // matching what compute_hash() would see for this
                    // resulting position.
                    p.hash ^= Self::ep_hash_contribution(&p.bb, them, ep_sq);
                }
            }
        }

        // castling rights
        let old = p.castling;
        p.castling &= CASTLE_MASK[from as usize] & CASTLE_MASK[to as usize];
        if p.castling != old {
            p.hash ^= ZOBRIST.castle[old as usize] ^ ZOBRIST.castle[p.castling as usize];
        }

        if pt == PAWN || capture {
            p.halfmove = 0;
        } else {
            p.halfmove += 1;
        }
        if let Color::Black = us {
            p.fullmove += 1;
        }
        p.side = them;
        p.hash ^= ZOBRIST.side;
        p
    }

    /// Null move: pass the turn (for null-move pruning).
    pub fn make_null(&self) -> Position {
        let mut p = *self;
        if p.ep != NO_EP {
            p.hash ^= Self::ep_hash_contribution(&p.bb, p.side, p.ep);
            p.ep = NO_EP;
        }
        p.halfmove += 1;
        p.side = p.side.flip();
        p.hash ^= ZOBRIST.side;
        p
    }

    /// ASCII board for debugging.
    pub fn pretty(&self) -> String {
        let chars = ['P', 'N', 'B', 'R', 'Q', 'K'];
        let mut out = String::new();
        for r in (0..8).rev() {
            out.push_str(&format!("{} ", r + 1));
            for f in 0..8 {
                let ch = match self.piece_on(sq(f, r)) {
                    Some((Color::White, p)) => chars[p],
                    Some((Color::Black, p)) => chars[p].to_ascii_lowercase(),
                    None => '.',
                };
                out.push(ch);
                out.push(' ');
            }
            out.push('\n');
        }
        out.push_str("  a b c d e f g h\n");
        out
    }
}
