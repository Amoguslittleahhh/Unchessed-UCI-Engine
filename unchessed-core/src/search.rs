//! Iterative-deepening alpha-beta search with quiescence, transposition table,
//! null-move pruning, LMR, killers/history ordering, MultiPV and time management.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::board::*;
use crate::eval::{Eval, EvalState, MG_VALUE};
use crate::movegen::{generate, in_check, king_safe_after, legal, MoveList};
use crate::see::{see_with_pins, LazyPins};
use crate::tt::{BOUND_EXACT, BOUND_LOWER, BOUND_UPPER, TT};

pub const MATE: i32 = 30_000;
pub const MATE_IN_MAX: i32 = MATE - 512;
pub const MAX_PLY: usize = 96;

/// Tunable search constants, exposed as UCI options so they can be adjusted
/// without a rebuild (and eventually driven by automated tuning, e.g. SPSA).
/// Defaults match the values that were previously hard-coded, so behavior is
/// unchanged unless a caller explicitly overrides them.
#[derive(Clone, Copy)]
pub struct SearchParams {
    /// reverse futility pruning: prune when static_eval - rfp_margin*depth >= beta
    pub rfp_margin: i32,
    /// null-move reduction: r = nm_base + depth / nm_divisor
    pub nm_base: i32,
    pub nm_divisor: i32,
    /// late move reductions apply only from this depth
    pub lmr_min_depth: i32,
    /// ...and only after this many legal moves have been tried
    pub lmr_min_movenum: i32,
    /// an extra ply of reduction once this many moves have been tried
    pub lmr_big_movenum: i32,
    /// initial aspiration window half-width in centipawns
    pub aspiration_delta: i32,
    /// aspiration windows only apply from this depth (shallow scores are too
    /// volatile for a narrow window to pay off)
    pub aspiration_min_depth: i32,
    /// ProbCut: margin added to beta for the reduced-depth verification
    /// search (fixed, not adjusted by an "improving" term — a deliberate
    /// simplification of the technique for a first implementation)
    pub probcut_margin: i32,
    /// depth reduction for ProbCut's verification search
    pub probcut_reduction: i32,
    /// ProbCut only applies from this depth (needs depth - reduction >= 1)
    pub probcut_min_depth: i32,
    /// plain futility pruning: skip a quiet move when
    /// static_eval + futility_margin*depth <= alpha (per-move, complements
    /// reverse futility pruning above which is a whole-node decision)
    pub futility_margin: i32,
    /// plain futility pruning only applies at or below this depth
    pub futility_max_depth: i32,
    /// Skip ProbCut verification of captures that SEE already scores as
    /// material-losing.
    ///
    /// Default `false`: this removes searches, so it can change when ProbCut
    /// fires and therefore changes the search tree. Per this project's
    /// discipline a tree-changing pruning rule stays off until a paired-game
    /// SPRT says otherwise; the plumbing is here so that gate can actually be
    /// run (`ProbcutSeeFilter` UCI option).
    pub probcut_see_filter: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            rfp_margin: 90,
            nm_base: 3,
            nm_divisor: 6,
            lmr_min_depth: 3,
            lmr_min_movenum: 3,
            lmr_big_movenum: 12,
            aspiration_delta: 25,
            aspiration_min_depth: 4,
            probcut_margin: 200,
            probcut_reduction: 4,
            probcut_min_depth: 5,
            futility_margin: 150,
            futility_max_depth: 8,
            probcut_see_filter: false,
        }
    }
}

#[derive(Clone, Default)]
pub struct Limits {
    pub depth: Option<i32>,
    pub movetime: Option<u64>,
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    pub movestogo: Option<u64>,
    pub nodes: Option<u64>,
    pub infinite: bool,
    /// `go searchmoves <m1> <m2> ...`: raw UCI move strings restricting the
    /// root move set. Stored as strings (not `Move`) since parsing a UCI
    /// move string requires the position, which isn't known at `go`-line
    /// parse time -- resolved against the actual root move list in
    /// `go_with_root_hints`. Empty means unrestricted (the default for
    /// every internally-constructed `Limits`, e.g. the adapter's opponent-
    /// observation and book-troll-check probes, which must never be
    /// accidentally restricted by an outer command's searchmoves).
    pub searchmoves: Vec<String>,
    /// `go ponder`: the UCI-line "ponder" flag. `Limits` itself does nothing
    /// with this -- it's read by the UCI command loop to decide whether to
    /// defer starting the search until `ponderhit`, per this engine's
    /// pondering model (see uci.rs's "go"/"ponderhit" handling).
    pub ponder: bool,
}

impl Limits {
    pub fn depth(d: i32) -> Limits {
        Limits {
            depth: Some(d),
            ..Default::default()
        }
    }

    pub fn movetime(ms: u64) -> Limits {
        Limits {
            movetime: Some(ms),
            ..Default::default()
        }
    }

    /// Is this a game move (clocks, movetime, or a fixed depth/node match as
    /// GUIs use for engine-vs-engine), as opposed to analysis? Analysis is
    /// signalled by `go infinite` (what En Croissant sends when analyzing).
    pub fn is_game_mode(&self) -> bool {
        !self.infinite
            && (self.movetime.is_some()
                || self.wtime.is_some()
                || self.btime.is_some()
                || self.depth.is_some()
                || self.nodes.is_some())
    }

    /// Remaining clock time for `side`, if this is a clock game.
    pub fn my_time(&self, side: Color) -> Option<u64> {
        match side {
            Color::White => self.wtime,
            Color::Black => self.btime,
        }
    }

    /// (soft_ms, hard_ms) budgets; None = unlimited.
    ///
    /// Urgency scales with the remaining clock: with plenty of time the
    /// engine invests in depth; as the clock drains the budgets shrink
    /// faster than linearly, and in real time trouble it moves near-
    /// instantly on the increment.
    pub(crate) fn budget(&self, side: Color) -> (Option<u64>, Option<u64>) {
        if self.infinite {
            return (None, None);
        }
        if let Some(mt) = self.movetime {
            let t = mt.saturating_sub(25).max(5);
            return (Some(t), Some(t));
        }
        let inc = match side {
            Color::White => self.winc.unwrap_or(0),
            Color::Black => self.binc.unwrap_or(0),
        };
        if let Some(t) = self.my_time(side) {
            let mtg = self.movestogo.unwrap_or(30).max(1);
            let mut soft = t / mtg + inc * 3 / 4;
            let mut hard = (t / 5 + inc / 2).max(soft);
            // low-clock urgency tiers: naturally quicker as time runs down
            if t < 20_000 {
                soft = soft.min(t / 35 + inc / 2);
                hard = hard.min(t / 10 + inc / 2);
            }
            if t < 6_000 {
                soft = soft.min(t / 60 + inc / 2);
                hard = hard.min(t / 16 + inc / 2);
            }
            if t < 2_000 {
                // panic mode: play on the increment, keep a reserve
                soft = soft.min((inc / 2).max(30));
                hard = hard.min(t / 8);
            }
            let ceiling = t.saturating_sub(60).max(5);
            let hard = hard.min(ceiling).max(5);
            let soft = soft.min(hard).max(3);
            return (Some(soft), Some(hard));
        }
        (None, None)
    }
}

#[derive(Clone, Debug)]
pub struct Line {
    pub mv: Move,
    pub score: i32,
    pub depth: i32,
    pub pv: Vec<Move>,
}

/// Optional policy-only root ordering signal. It can change only the order of
/// the first iterative-deepening pass: every legal move is still searched and
/// all completed alpha-beta scores remain authoritative.
#[derive(Clone, Copy, Debug)]
pub struct RootHint {
    pub mv: Move,
    pub policy_score: f32,
}

pub struct InfoEvent<'a> {
    pub depth: i32,
    pub multipv: usize,
    pub score: i32,
    pub nodes: u64,
    pub time_ms: u64,
    /// Transposition-table occupancy in permille, for UCI `info hashfull`.
    pub hashfull: usize,
    pub pv: &'a [Move],
}

#[inline]
pub fn is_mate_score(s: i32) -> bool {
    s.abs() >= MATE_IN_MAX
}

/// Moves until mate (signed), for "score mate N" output.
pub fn mate_in(s: i32) -> i32 {
    if s > 0 {
        (MATE - s + 1) / 2
    } else {
        -(MATE + s + 1) / 2
    }
}

#[inline]
fn to_tt(s: i32, ply: usize) -> i32 {
    if s >= MATE_IN_MAX {
        s + ply as i32
    } else if s <= -MATE_IN_MAX {
        s - ply as i32
    } else {
        s
    }
}

#[inline]
fn from_tt(s: i32, ply: usize) -> i32 {
    if s >= MATE_IN_MAX {
        s - ply as i32
    } else if s <= -MATE_IN_MAX {
        s + ply as i32
    } else {
        s
    }
}

struct Searcher<'a> {
    tt: &'a TT,
    eval: &'a dyn Eval,
    params: SearchParams,
    stop: &'a AtomicBool,
    start: Instant,
    hard_ms: Option<u64>,
    node_limit: Option<u64>,
    nodes: u64,
    abort: bool,
    /// score of a drawn line from the ROOT side's perspective (contempt)
    root_draw: i32,
    killers: [[Move; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    /// hashes of game positions + current search path (ancestors of the node)
    path: Vec<u64>,
    pv_table: [[Move; MAX_PLY]; MAX_PLY],
    pv_len: [usize; MAX_PLY],
    /// Ply-indexed evaluator state (NNUE's incremental accumulators; empty
    /// for HCE). `eval_states[ply]` is always the state for whatever
    /// position was passed as `pos` to the negamax/qsearch call running at
    /// that ply -- the parent computes and writes `eval_states[ply + 1]`
    /// via `Eval::update_state` right before recursing, so no unmake is
    /// needed even though `Position` is copy-make with no explicit pop.
    eval_states: Vec<EvalState>,
}

impl<'a> Searcher<'a> {
    /// Return true before entering a node that would exceed an explicit UCI
    /// `go nodes N` budget. The counter therefore remains the exact number of
    /// visited nodes and never rounds up to the periodic stop/time poll.
    #[inline]
    fn node_limit_reached(&mut self) -> bool {
        if let Some(n) = self.node_limit {
            if self.nodes >= n {
                self.abort = true;
                return true;
            }
        }
        false
    }

    #[inline]
    fn check_limits(&mut self) {
        // Checked on every node, not masked to a periodic cadence. The old
        // "every 2,048 nodes" cadence is fine for ordinary time controls
        // (that's only ~2ms of wall time at this engine's typical NPS), but
        // for a genuinely short `go movetime`/`go nodes` budget a shallow
        // search can complete entirely -- several iterative-deepening passes
        // -- using fewer than 2,048 total nodes, so the deadline is never
        // consulted again after the very first (trivially-passing) check at
        // node 0. An AtomicBool load and an Instant::elapsed() read are both
        // cheap enough per node that checking every time costs no measurable
        // NPS (verified: ~975k nodes/s before and after on a real search).
        if self.stop.load(Ordering::Relaxed) {
            self.abort = true;
            return;
        }
        if let Some(h) = self.hard_ms {
            if self.start.elapsed().as_millis() as u64 >= h {
                self.abort = true;
            }
        }
    }

    /// Has `hash` already occurred on the path from the game start to this
    /// node?
    ///
    /// Only the most recent `halfmove` entries can possibly match: the
    /// halfmove clock resets to zero on every capture and pawn push, and
    /// those moves are irreversible, so no position before the last one can
    /// ever recur. Scanning the entire path (which is seeded with the whole
    /// game history) therefore compared a long tail of entries that are
    /// provably non-matching -- at move 40 that is ~80 wasted comparisons on
    /// every single node.
    ///
    /// This is a strict narrowing of the search space, not a heuristic: any
    /// entry it now skips could not have compared equal anyway, so the
    /// returned value -- and hence the search tree -- is unchanged.
    ///
    /// Note the bound uses `halfmove` from the node being tested, and
    /// `make_null` increments `halfmove` without clearing the path, so the
    /// window can only ever be too generous (safe), never too tight.
    #[inline]
    fn is_repetition(&self, hash: u64, halfmove: u16) -> bool {
        let window = (halfmove as usize).min(self.path.len());
        self.path[self.path.len() - window..]
            .iter()
            .rev()
            .any(|&h| h == hash)
    }

    /// Draw value from the perspective of the side to move at `ply`.
    #[inline]
    fn draw(&self, ply: usize) -> i32 {
        if ply % 2 == 0 {
            self.root_draw
        } else {
            -self.root_draw
        }
    }

    fn update_pv(&mut self, ply: usize, m: Move) {
        self.pv_table[ply][0] = m;
        let child_len = if ply + 1 < MAX_PLY {
            self.pv_len[ply + 1]
        } else {
            0
        };
        for i in 0..child_len {
            self.pv_table[ply][i + 1] = self.pv_table[ply + 1][i];
        }
        self.pv_len[ply] = child_len + 1;
    }

    #[inline]
    fn move_score(&self, pos: &Position, m: Move, tt_mv: Move, ply: usize, pins: &LazyPins) -> i32 {
        if m == tt_mv {
            return 1 << 22;
        }
        let is_ep = m.kind() == MK_EP;
        let is_cap = is_ep || pos.piece_on(m.to()).is_some();
        // SEE-based ordering: a capture/promotion that nets material (or is
        // at worst equal) is searched before quiets; one that loses material
        // once the full recapture sequence plays out is deferred behind them
        // (MVV-LVA alone can't tell a winning capture from a losing one —
        // e.g. QxP defended by a pawn scores high under MVV-LVA but is a
        // material-losing move).
        if is_cap || m.is_promo() {
            let sc = see_with_pins(pos, m, pins.get());
            if sc >= 0 {
                return 1_000_000 + sc;
            }
            return -1_000_000 + sc;
        }
        if m == self.killers[ply][0] {
            return 800_000;
        }
        if m == self.killers[ply][1] {
            return 790_000;
        }
        self.history[pos.side.idx()][m.from() as usize][m.to() as usize]
    }

    fn qsearch(&mut self, pos: &Position, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        if self.abort || self.node_limit_reached() {
            return 0;
        }
        self.nodes += 1;
        self.check_limits();
        if self.abort {
            return 0;
        }
        if ply >= MAX_PLY {
            return self.eval.eval_with_state(pos, &self.eval_states[ply]);
        }
        let in_chk = in_check(pos);
        let us = pos.side;

        let mut best;
        let mut ml = MoveList::new();
        if in_chk {
            // evasions: search everything; mate detection needs full gen
            best = -MATE + ply as i32;
            generate(pos, false, &mut ml);
        } else {
            best = self.eval.eval_with_state(pos, &self.eval_states[ply]);
            if best >= beta {
                return best;
            }
            if best > alpha {
                alpha = best;
            }
            generate(pos, true, &mut ml);
        }

        // MVV-LVA ordering
        let mut scores = [0i32; 256];
        // Pin scan is position-wide, not move-specific: compute it at most
        // once per node instead of once per capture inside `see`, and only
        // if this node actually scores a capture.
        let pins = LazyPins::new(pos);
        for i in 0..ml.len {
            scores[i] = self.move_score(pos, ml.moves[i], Move::NONE, ply.min(MAX_PLY - 1), &pins);
        }

        let mut any_legal = false;
        for i in 0..ml.len {
            // selection sort step
            let mut bi = i;
            for j in (i + 1)..ml.len {
                if scores[j] > scores[bi] {
                    bi = j;
                }
            }
            ml.moves.swap(i, bi);
            scores.swap(i, bi);
            let m = ml.moves[i];

            // SEE pruning: a capture that loses material after the full
            // recapture sequence is essentially never worth searching in
            // quiescence (we're only here to resolve tactics, not to play a
            // losing trade). `scores[i]` already encodes this: move_score
            // puts losing captures below -1_000_000.
            if !in_chk && scores[i] < -1_000_000 {
                continue;
            }

            // delta pruning (skip when in check)
            if !in_chk {
                let victim_val = if m.kind() == MK_EP {
                    MG_VALUE[PAWN]
                } else {
                    pos.piece_on(m.to()).map(|(_, p)| MG_VALUE[p]).unwrap_or(0)
                };
                let promo_gain = if m.is_promo() {
                    MG_VALUE[m.promo_piece()] - MG_VALUE[PAWN]
                } else {
                    0
                };
                if best + victim_val + promo_gain + 180 < alpha {
                    continue;
                }
            }

            let next = pos.make(m);
            if !king_safe_after(&next, us) {
                continue;
            }
            any_legal = true;
            let child_state = self
                .eval
                .update_state(pos, &next, m, &self.eval_states[ply]);
            self.eval_states[ply + 1] = child_state;
            let sc = -self.qsearch(&next, -beta, -alpha, ply + 1);
            if self.abort {
                return 0;
            }
            if sc > best {
                best = sc;
                if sc > alpha {
                    alpha = sc;
                    if sc >= beta {
                        break;
                    }
                }
            }
        }

        if in_chk && !any_legal {
            return -MATE + ply as i32;
        }
        best
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        pos: &Position,
        depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        is_pv: bool,
        allow_null: bool,
    ) -> i32 {
        if self.abort || self.node_limit_reached() {
            return 0;
        }
        if ply < MAX_PLY {
            self.pv_len[ply] = 0;
        }
        self.nodes += 1;
        self.check_limits();
        if self.abort {
            return 0;
        }
        if ply >= MAX_PLY {
            return self.eval.eval_with_state(pos, &self.eval_states[ply]);
        }

        let in_chk = in_check(pos);

        // draws
        //
        // Checkmate takes precedence over the 50-move rule: a position
        // reached at halfmove 100 with the side to move in check and no
        // legal reply is a completed game (checkmate), not a draw --
        // mirrors Stockfish's own is_draw() guard for this exact edge
        // case. Generating the legal move list here is only paid in that
        // rare in-check-at-halfmove-100 case, not on the hot path; the
        // ordinary (not-in-check, or in-check-with-a-reply) cases below
        // still return immediately without it.
        if pos.halfmove >= 100 && (!in_chk || legal(pos).len != 0) {
            return self.draw(ply);
        }
        if self.is_repetition(pos.hash, pos.halfmove) {
            return self.draw(ply);
        }

        let depth = if in_chk { depth.max(1) } else { depth };
        if depth <= 0 {
            self.nodes -= 1; // qsearch counts it
            return self.qsearch(pos, alpha, beta, ply);
        }

        // TT probe
        let mut tt_mv = Move::NONE;
        if let Some(e) = self.tt.probe(pos.hash) {
            tt_mv = Move(e.mv);
            if !is_pv && e.depth as i32 >= depth {
                let sc = from_tt(e.score as i32, ply);
                match e.bound {
                    BOUND_EXACT => return sc,
                    BOUND_LOWER if sc >= beta => return sc,
                    BOUND_UPPER if sc <= alpha => return sc,
                    _ => {}
                }
            }
        }

        let static_eval = self.eval.eval_with_state(pos, &self.eval_states[ply]);

        // reverse futility pruning
        if !is_pv
            && !in_chk
            && depth <= 6
            && beta.abs() < MATE_IN_MAX
            && static_eval - self.params.rfp_margin * depth >= beta
        {
            return static_eval;
        }

        // null-move pruning
        if allow_null
            && !is_pv
            && !in_chk
            && depth >= 3
            && static_eval >= beta
            && beta.abs() < MATE_IN_MAX
            && pos.has_non_pawn(pos.side)
        {
            let r = self.params.nm_base + depth / self.params.nm_divisor;
            self.path.push(pos.hash);
            // A null move changes no piece positions (only side-to-move and
            // en-passant state), so the evaluator's accumulators are
            // unaffected -- carry the state forward unchanged instead of
            // running it through update_state for no reason.
            self.eval_states[ply + 1] = self.eval_states[ply];
            let sc = -self.negamax(
                &pos.make_null(),
                depth - 1 - r,
                -beta,
                -beta + 1,
                ply + 1,
                false,
                false,
            );
            self.path.pop();
            if self.abort {
                return 0;
            }
            if sc >= beta {
                return beta;
            }
        }

        // ProbCut: before committing to the full-depth search, check whether
        // a handful of captures/promotions already confirm a fail-high well
        // past beta at reduced depth; if so, trust it and cut. A known-risky
        // (not fully sound) pruning technique — the margin trades a little
        // tactical accuracy for speed, same tradeoff class as null-move.
        if !is_pv && !in_chk && depth >= self.params.probcut_min_depth && beta.abs() < MATE_IN_MAX {
            let beta_cut = beta + self.params.probcut_margin;
            let rdepth = depth - self.params.probcut_reduction;
            if rdepth >= 1 {
                let mut pc_ml = MoveList::new();
                generate(pos, true, &mut pc_ml);
                let mut pc_scores = [0i32; 256];
                let pc_pins = LazyPins::new(pos);
                for i in 0..pc_ml.len {
                    pc_scores[i] = self.move_score(
                        pos,
                        pc_ml.moves[i],
                        Move::NONE,
                        ply.min(MAX_PLY - 1),
                        &pc_pins,
                    );
                }
                let mut found = false;
                self.path.push(pos.hash);
                for i in 0..pc_ml.len {
                    let mut bi = i;
                    for j in (i + 1)..pc_ml.len {
                        if pc_scores[j] > pc_scores[bi] {
                            bi = j;
                        }
                    }
                    pc_ml.moves.swap(i, bi);
                    pc_scores.swap(i, bi);

                    // Optional: skip captures that SEE already says lose
                    // material. `move_score` scores those below -1_000_000,
                    // so this reuses a value we just computed rather than
                    // paying for a second SEE. A capture that loses material
                    // is very unlikely to beat `beta + probcut_margin`, so
                    // verifying it is near-pure waste. Because the list is
                    // sorted best-first, every remaining entry is also
                    // losing -- hence `break`, not `continue`.
                    //
                    // Default-off: this changes which nodes ProbCut cuts, so
                    // it is a tree change and needs an SPRT before shipping.
                    if self.params.probcut_see_filter && pc_scores[i] < -1_000_000 {
                        break;
                    }

                    let m = pc_ml.moves[i];
                    let next = pos.make(m);
                    self.tt.prefetch(next.hash);
                    if !king_safe_after(&next, pos.side) {
                        continue;
                    }
                    let child_state = self
                        .eval
                        .update_state(pos, &next, m, &self.eval_states[ply]);
                    self.eval_states[ply + 1] = child_state;
                    let sc = -self.negamax(
                        &next,
                        rdepth - 1,
                        -beta_cut,
                        -beta_cut + 1,
                        ply + 1,
                        false,
                        true,
                    );
                    if self.abort {
                        break;
                    }
                    if sc >= beta_cut {
                        found = true;
                        break;
                    }
                }
                self.path.pop();
                if self.abort {
                    return 0;
                }
                if found {
                    return beta;
                }
            }
        }

        let mut ml = MoveList::new();
        generate(pos, false, &mut ml);
        let mut scores = [0i32; 256];
        let pins = LazyPins::new(pos);
        for i in 0..ml.len {
            scores[i] = self.move_score(pos, ml.moves[i], tt_mv, ply, &pins);
        }

        let us = pos.side;
        let mut best = -MATE;
        let mut best_mv = Move::NONE;
        let mut bound = BOUND_UPPER;
        let mut legal_count = 0;

        self.path.push(pos.hash);
        for i in 0..ml.len {
            let mut bi = i;
            for j in (i + 1)..ml.len {
                if scores[j] > scores[bi] {
                    bi = j;
                }
            }
            ml.moves.swap(i, bi);
            scores.swap(i, bi);
            let m = ml.moves[i];

            let next = pos.make(m);
            // Start the child's TT line moving toward cache now; the child
            // won't probe until after the legality check, `in_check`, and
            // (for non-pruned moves) the accumulator update have run, so the
            // miss latency overlaps real work. Pure hint, no semantic effect.
            self.tt.prefetch(next.hash);
            if !king_safe_after(&next, us) {
                continue;
            }
            legal_count += 1;

            let gives_check = in_check(&next);
            let is_cap = m.kind() == MK_EP || pos.board[m.to() as usize] != NO_PIECE;
            let ext = if gives_check { 1 } else { 0 };
            let nd = depth - 1 + ext;

            // plain futility pruning: unlike reverse futility pruning above
            // (a whole-node decision based on beta), this skips individual
            // late quiet moves when even a generous margin can't bring the
            // position back up to alpha — a quiet move rarely swings eval by
            // more than the margin, so it's not worth a full recursive
            // search just to confirm it fails low.
            if !is_pv
                && !in_chk
                && !gives_check
                && !is_cap
                && !m.is_promo()
                && depth <= self.params.futility_max_depth
                && legal_count > 1
                && alpha.abs() < MATE_IN_MAX
                && static_eval + self.params.futility_margin * depth <= alpha
            {
                // Fail-soft floor: this node's true score can't be proven
                // below the futility bound itself, so `best` (and whatever
                // gets stored in the TT) shouldn't be either -- matches
                // every reference implementation (Stockfish, Heinz's 1998
                // pseudocode), which raise the returned score the same way
                // before skipping. A bare `continue` here previously let a
                // node that pruned several quiets return an unnecessarily
                // pessimistic fail-soft bound.
                let futility_floor = static_eval + self.params.futility_margin * depth;
                if futility_floor > best {
                    best = futility_floor;
                }
                continue;
            }

            // Deferred until after futility pruning on purpose: this is a
            // full NNUE accumulator update plus a 2KB `EvalState` write, and
            // a futility-pruned move never recurses, so doing it above the
            // pruning test spent that work on a move whose result was thrown
            // away. Nothing between the old and new position of this line
            // reads `eval_states[ply + 1]`, and the futility test itself
            // depends only on `static_eval`/`depth`/`alpha`/move flags, so
            // the pruning decision -- and the tree -- is unchanged.
            let child_state = self
                .eval
                .update_state(pos, &next, m, &self.eval_states[ply]);
            self.eval_states[ply + 1] = child_state;

            let mut sc;
            if legal_count == 1 {
                sc = -self.negamax(&next, nd, -beta, -alpha, ply + 1, is_pv, true);
            } else {
                // late move reductions for quiet moves
                let mut r = 0;
                if depth >= self.params.lmr_min_depth
                    && legal_count > self.params.lmr_min_movenum
                    && !is_cap
                    && !m.is_promo()
                    && !in_chk
                    && !gives_check
                {
                    r =
                        1 + if legal_count > self.params.lmr_big_movenum {
                            1
                        } else {
                            0
                        } + if !is_pv { 1 } else { 0 };
                    r = r.min(nd - 1).max(0);
                }
                sc = -self.negamax(&next, nd - r, -(alpha + 1), -alpha, ply + 1, false, true);
                if sc > alpha && r > 0 {
                    sc = -self.negamax(&next, nd, -(alpha + 1), -alpha, ply + 1, false, true);
                }
                if sc > alpha && sc < beta && is_pv {
                    sc = -self.negamax(&next, nd, -beta, -alpha, ply + 1, true, true);
                }
            }
            if self.abort {
                self.path.pop();
                return 0;
            }

            if sc > best {
                best = sc;
                best_mv = m;
                if sc > alpha {
                    alpha = sc;
                    bound = BOUND_EXACT;
                    if is_pv {
                        self.update_pv(ply, m);
                    }
                    if sc >= beta {
                        bound = BOUND_LOWER;
                        if !is_cap && !m.is_promo() {
                            if self.killers[ply][0] != m {
                                self.killers[ply][1] = self.killers[ply][0];
                                self.killers[ply][0] = m;
                            }
                            let h = &mut self.history[us.idx()][m.from() as usize][m.to() as usize];
                            *h += depth * depth;
                            if *h > 1 << 20 {
                                *h /= 2;
                            }
                        }
                        break;
                    }
                }
            }
        }
        self.path.pop();

        if legal_count == 0 {
            return if in_chk {
                -MATE + ply as i32
            } else {
                self.draw(ply)
            };
        }

        self.tt
            .store(pos.hash, best_mv, to_tt(best, ply), depth, bound);
        best
    }
}

/// Run a search. `eval` is the static evaluator (HCE or NNUE). `history` =
/// Zobrist hashes of all game positions before `pos` (used for repetition
/// detection). `draw_score` is the value of a drawn line from the root side's
/// perspective (0 = neutral; negative = contempt, we want to avoid draws).
/// `tt` is a shared reference: safe to call `go` concurrently from multiple
/// threads against the same table (see tt.rs) — this is what Lazy SMP helper
/// threads do. `start_depth` skips the (cheap, mostly-redundant-across-
/// threads) shallow iterations; Lazy SMP helpers stagger this per-thread so
/// they diverge from the main thread's search sooner instead of all threads
/// retracing the same shallow tree in lockstep. Pass 1 for a normal search.
/// Returns MultiPV lines sorted best-first.
#[allow(clippy::too_many_arguments)]
pub fn go(
    pos: &Position,
    eval: &dyn Eval,
    limits: &Limits,
    multipv: usize,
    tt: &TT,
    stop: &AtomicBool,
    history: &[u64],
    draw_score: i32,
    params: SearchParams,
    start_depth: i32,
    info: &mut dyn FnMut(&InfoEvent),
) -> Vec<Line> {
    go_with_root_hints(
        pos,
        eval,
        limits,
        multipv,
        tt,
        stop,
        history,
        draw_score,
        params,
        start_depth,
        &[],
        std::time::Duration::ZERO,
        info,
    )
}

/// Experimental, default-unreachable root-hint trial entry point.
///
/// `preprocessing_elapsed` is charged against the same soft/hard deadline as
/// search. This prevents a synchronous caller from treating neural inference
/// as free time. The normal UCI path calls [`go`] and supplies no hints.
/// Static evaluation and quiescence evaluation of `pos`, from the side to
/// move's perspective.
///
/// Exposed for training-data generation. Tan & Watkinson Medina,
/// *Study of the Proper NNUE Dataset* (arXiv:2412.17948), define a "quiet"
/// position partly as one where these two values agree closely: a large gap
/// means a capture sequence is available that will swing the evaluation, so
/// the static score is not a label worth training on.
///
/// Runs no search bookkeeping beyond quiescence itself -- no time limit, no
/// node limit, no repetition path -- so it is deterministic and cheap.
pub fn static_and_quiescence(pos: &Position, eval: &dyn Eval, tt: &TT) -> (i32, i32) {
    let stop = AtomicBool::new(false);
    let mut searcher = Searcher {
        tt,
        eval,
        params: SearchParams::default(),
        stop: &stop,
        start: Instant::now(),
        hard_ms: None,
        node_limit: None,
        nodes: 0,
        abort: false,
        root_draw: 0,
        killers: [[Move::NONE; 2]; MAX_PLY],
        history: [[[0; 64]; 64]; 2],
        path: Vec::new(),
        pv_table: [[Move::NONE; MAX_PLY]; MAX_PLY],
        pv_len: [0; MAX_PLY],
        eval_states: vec![eval.initial_state(pos); MAX_PLY + 1],
    };
    let static_eval = eval.eval_with_state(pos, &searcher.eval_states[0]);
    let quiet_eval = searcher.qsearch(pos, -MATE, MATE, 0);
    (static_eval, quiet_eval)
}

pub fn go_with_root_hints(
    pos: &Position,
    eval: &dyn Eval,
    limits: &Limits,
    multipv: usize,
    tt: &TT,
    stop: &AtomicBool,
    history: &[u64],
    draw_score: i32,
    params: SearchParams,
    start_depth: i32,
    root_hints: &[RootHint],
    preprocessing_elapsed: std::time::Duration,
    info: &mut dyn FnMut(&InfoEvent),
) -> Vec<Line> {
    let now = Instant::now();
    let start = now.checked_sub(preprocessing_elapsed).unwrap_or(now);
    let (base_soft, hard_ms) = limits.budget(pos.side);
    let max_depth = limits
        .depth
        .unwrap_or(MAX_PLY as i32 - 1)
        .clamp(1, MAX_PLY as i32 - 1);

    let root_moves_list = legal(pos);
    if root_moves_list.len == 0 {
        return Vec::new();
    }
    // `go searchmoves`: restrict to the requested root moves, matched by
    // their own UCI notation against the already-legal move list (so
    // castling/en-passant/promotion notation is handled exactly like move
    // printing already does, no separate position-aware parsing needed).
    // Falls back to the full legal set if every requested move turns out
    // illegal here, rather than searching nothing or panicking on a stale
    // or malformed list.
    let restricted: Vec<Move> = if limits.searchmoves.is_empty() {
        Vec::new()
    } else {
        root_moves_list
            .as_slice()
            .iter()
            .copied()
            .filter(|m| limits.searchmoves.iter().any(|s| s == &m.uci()))
            .collect()
    };
    let root_slice: &[Move] = if restricted.is_empty() {
        root_moves_list.as_slice()
    } else {
        &restricted
    };
    let multipv = multipv.max(1).min(root_slice.len());

    // situation-based allocation: sharp or wide positions deserve more of
    // the clock, simple ones less; a single legal move needs almost none
    let root_in_check = in_check(pos);
    let situation = {
        let width = (0.65 + root_slice.len() as f64 / 45.0).clamp(0.75, 1.3);
        let sharp = if root_in_check { 1.25 } else { 1.0 };
        width * sharp
    };
    let mut soft_ms =
        base_soft.map(|s| ((s as f64 * situation) as u64).clamp(3, hard_ms.unwrap_or(u64::MAX)));

    struct RootMove {
        mv: Move,
        score: i32,
        policy_hint: f32,
        pv: Vec<Move>,
        depth: i32,
    }
    let mut roots: Vec<RootMove> = root_slice
        .iter()
        .map(|&m| RootMove {
            mv: m,
            score: -MATE,
            policy_hint: root_hints
                .iter()
                .find(|hint| hint.mv == m && hint.policy_score.is_finite())
                .map(|hint| hint.policy_score)
                .unwrap_or(f32::NEG_INFINITY),
            pv: vec![m],
            depth: 0,
        })
        .collect();

    let mut s = Searcher {
        tt,
        eval,
        params,
        stop,
        start,
        hard_ms,
        node_limit: limits.nodes,
        nodes: 0,
        abort: false,
        root_draw: draw_score.clamp(-100, 100),
        killers: [[Move::NONE; 2]; MAX_PLY],
        history: [[[0; 64]; 64]; 2],
        path: history.to_vec(),
        pv_table: [[Move::NONE; MAX_PLY]; MAX_PLY],
        pv_len: [0; MAX_PLY],
        // eval_states[ply + 1] can be written for ply up to MAX_PLY - 1
        // (the negamax/qsearch ply>=MAX_PLY guard bails before indexing
        // further), so this needs MAX_PLY + 1 slots, not MAX_PLY.
        eval_states: vec![eval.initial_state(pos); MAX_PLY + 1],
    };

    let mut completed: Vec<Line> = Vec::new();
    // in-search time feedback
    let mut stable_best: Move = Move::NONE;
    let mut stable_count = 0u32;
    let mut prev_iter_score: Option<i32> = None;
    let mut extended = false;

    let start_depth = start_depth.clamp(1, max_depth);
    'deepening: for depth in start_depth..=max_depth {
        // A policy hint can order only the first pass. From the second pass
        // onward completed alpha-beta scores are the sole ordering signal.
        if depth == start_depth && !root_hints.is_empty() {
            roots.sort_by(|left, right| right.policy_hint.total_cmp(&left.policy_hint));
        } else {
            roots.sort_by_key(|root| -root.score);
        }
        let mut chosen: Vec<Move> = Vec::new();

        for pv_idx in 0..multipv {
            // aspiration window: narrow around the previous iteration's
            // score for this slot, widening geometrically on fail-low/high.
            // Shallow depths and mate-range scores skip straight to a full
            // window since a narrow guess isn't reliable there.
            let prev_score = roots[pv_idx].score;
            let (mut window_lo, mut window_hi) =
                if depth >= s.params.aspiration_min_depth && prev_score.abs() < MATE_IN_MAX {
                    (
                        (prev_score - s.params.aspiration_delta).max(-MATE),
                        (prev_score + s.params.aspiration_delta).min(MATE),
                    )
                } else {
                    (-MATE, MATE)
                };
            let mut delta = s.params.aspiration_delta.max(1);

            let mut best_idx: Option<usize>;
            loop {
                let mut alpha = window_lo;
                let beta = window_hi;
                let mut searched = 0;
                best_idx = None;

                s.path.push(pos.hash);
                for ri in 0..roots.len() {
                    let m = roots[ri].mv;
                    if chosen.contains(&m) {
                        continue;
                    }
                    let next = pos.make(m);
                    searched += 1;
                    let child_state = s.eval.update_state(pos, &next, m, &s.eval_states[0]);
                    s.eval_states[1] = child_state;
                    let mut sc;
                    if searched == 1 {
                        sc = -s.negamax(&next, depth - 1, -beta, -alpha, 1, true, true);
                    } else {
                        sc = -s.negamax(&next, depth - 1, -(alpha + 1), -alpha, 1, false, true);
                        if sc > alpha && !s.abort {
                            sc = -s.negamax(&next, depth - 1, -beta, -alpha, 1, true, true);
                        }
                    }
                    if s.abort {
                        break;
                    }
                    if sc > alpha || searched == 1 {
                        alpha = alpha.max(sc);
                        roots[ri].score = sc;
                        roots[ri].depth = depth;
                        let mut pv = vec![m];
                        pv.extend_from_slice(&s.pv_table[1][..s.pv_len[1]]);
                        roots[ri].pv = pv;
                        best_idx = Some(ri);
                    } else {
                        // keep ordering info without overwriting a real score
                        roots[ri].score = roots[ri].score.min(sc);
                    }
                }
                s.path.pop();

                if s.abort {
                    break 'deepening;
                }

                let result = best_idx.map(|bi| roots[bi].score);
                let failed_low =
                    window_lo > -MATE && result.map(|r| r <= window_lo).unwrap_or(true);
                let failed_high =
                    window_hi < MATE && result.map(|r| r >= window_hi).unwrap_or(false);
                if !failed_low && !failed_high {
                    break;
                }
                delta *= 4;
                if failed_low {
                    window_lo = (window_lo - delta).max(-MATE);
                }
                if failed_high {
                    window_hi = (window_hi + delta).min(MATE);
                }
            }

            if let Some(bi) = best_idx {
                chosen.push(roots[bi].mv);
                let elapsed = start.elapsed().as_millis() as u64;
                info(&InfoEvent {
                    depth,
                    multipv: pv_idx + 1,
                    score: roots[bi].score,
                    nodes: s.nodes,
                    time_ms: elapsed,
                    hashfull: s.tt.hashfull(),
                    pv: &roots[bi].pv,
                });
            }
        }

        // snapshot completed iteration
        let mut iter_lines: Vec<Line> = Vec::new();
        for m in &chosen {
            if let Some(r) = roots.iter().find(|r| r.mv == *m) {
                iter_lines.push(Line {
                    mv: r.mv,
                    score: r.score,
                    depth: r.depth,
                    pv: r.pv.clone(),
                });
            }
        }
        if !iter_lines.is_empty() {
            completed = iter_lines;
        }

        // stop deepening on forced mate found (with a little margin)
        if let Some(first) = completed.first() {
            if is_mate_score(first.score) && depth >= 12 {
                break;
            }
        }

        if let Some(first) = completed.first() {
            // best-move stability tracking (easy-move detection)
            if first.mv == stable_best {
                stable_count += 1;
            } else {
                stable_best = first.mv;
                stable_count = 0;
            }
            // trouble detection: score fell hard between iterations -> spend
            // more of the clock, once, up to near the hard limit
            if let (Some(prev), Some(soft), Some(hard)) = (prev_iter_score, soft_ms, hard_ms) {
                if !extended && first.score < prev - 45 {
                    soft_ms = Some(((soft as f64 * 1.7) as u64).min(hard * 9 / 10));
                    extended = true;
                }
            }
            prev_iter_score = Some(first.score);
        }

        if let Some(soft) = soft_ms {
            let elapsed = start.elapsed().as_millis() as u64;
            // forced move: no point searching deep
            if root_moves_list.len == 1 && depth >= 4 {
                break;
            }
            // easy move: the choice hasn't changed in ages, move naturally
            if stable_count >= 5 && depth >= 11 && elapsed > soft / 3 {
                break;
            }
            if elapsed >= soft {
                break;
            }
        }
    }

    if completed.is_empty() {
        // never finished depth 1 (extreme time pressure): fall back to first legal
        completed.push(Line {
            mv: roots[0].mv,
            score: 0,
            depth: 0,
            pv: vec![roots[0].mv],
        });
    }
    completed.sort_by_key(|l| -l.score);
    completed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Hce;
    use crate::fen;
    use std::sync::atomic::AtomicBool;

    /// The halfmove-bounded repetition scan must agree with a full scan for
    /// every hash that could legally still be a repetition.
    ///
    /// The bound is a strict narrowing: entries older than the halfmove
    /// clock sit behind an irreversible move and can never recur, so
    /// skipping them cannot change the answer. This checks both directions
    /// -- it must never invent a repetition, and never miss one inside the
    /// window.
    #[test]
    fn bounded_repetition_scan_matches_full_scan_within_window() {
        fn full_scan(path: &[u64], hash: u64) -> bool {
            path.iter().rev().any(|&h| h == hash)
        }

        let mut state = 0x2026_0824_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let tt = TT::new(1);
        let stop = AtomicBool::new(false);
        for _ in 0..2000 {
            let len = (next() % 120) as usize + 1;
            let halfmove = (next() % 60) as u16;
            // Small alphabet so repetitions actually occur.
            let path: Vec<u64> = (0..len).map(|_| next() % 40).collect();
            let probe = next() % 40;

            let searcher = Searcher {
                tt: &tt,
                eval: &Hce::default(),
                params: SearchParams::default(),
                stop: &stop,
                start: Instant::now(),
                hard_ms: None,
                node_limit: None,
                nodes: 0,
                abort: false,
                root_draw: 0,
                killers: [[Move::NONE; 2]; MAX_PLY],
                history: [[[0; 64]; 64]; 2],
                path: path.clone(),
                pv_table: [[Move::NONE; MAX_PLY]; MAX_PLY],
                pv_len: [0; MAX_PLY],
                eval_states: Vec::new(),
            };

            let bounded = searcher.is_repetition(probe, halfmove);
            let window = (halfmove as usize).min(path.len());
            let inside = path[path.len() - window..].iter().any(|&h| h == probe);

            assert_eq!(
                bounded, inside,
                "bounded scan must exactly cover the halfmove window"
            );
            if bounded {
                assert!(
                    full_scan(&path, probe),
                    "bounded scan must never invent a repetition the full scan misses"
                );
            }
        }
    }

    /// `static_and_quiescence` must actually separate quiet from noisy
    /// positions -- that is the entire basis of the arXiv:2412.17948 filter.
    #[test]
    fn static_and_quiescence_separates_quiet_from_hanging_positions() {
        let tt = TT::new(1);
        let hce = Hce::default();

        // Start position: nothing to capture, so quiescence has nothing to
        // resolve and must agree with the static evaluation exactly.
        let quiet = fen::parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let (s, q) = static_and_quiescence(&quiet, &hce, &tt);
        assert_eq!(s, q, "a position with no captures must be its own qsearch");

        // Black queen en prise to a pawn with no compensation: quiescence
        // must find the win and diverge sharply from the static score.
        let hanging = fen::parse("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1").unwrap();
        let (s2, q2) = static_and_quiescence(&hanging, &hce, &tt);
        assert!(
            (s2 - q2).abs() > 60,
            "hanging queen must exceed the 60cp quiet margin: static={s2} qsearch={q2}"
        );
        assert!(q2 > s2, "winning the queen must improve the score");
    }

    fn searcher_with_node_limit<'a>(
        tt: &'a TT,
        eval: &'a Hce,
        stop: &'a AtomicBool,
        node_limit: u64,
    ) -> Searcher<'a> {
        let pos = fen::startpos();
        Searcher {
            tt,
            eval,
            params: SearchParams::default(),
            stop,
            start: Instant::now(),
            hard_ms: None,
            node_limit: Some(node_limit),
            nodes: 0,
            abort: false,
            root_draw: 0,
            killers: [[Move::NONE; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            path: Vec::new(),
            pv_table: [[Move::NONE; MAX_PLY]; MAX_PLY],
            pv_len: [0; MAX_PLY],
            eval_states: vec![eval.initial_state(&pos); MAX_PLY + 1],
        }
    }

    /// Exact explicit budgets must not be rounded to the next 2,048-node
    /// asynchronous stop/time polling boundary.
    #[test]
    fn node_limits_abort_at_each_requested_node() {
        let pos = fen::startpos();
        for limit in [1, 2_047, 2_048, 2_049, 25_000] {
            let tt = TT::new(16);
            let eval = Hce::default();
            let stop = AtomicBool::new(false);
            let mut searcher = searcher_with_node_limit(&tt, &eval, &stop, limit);

            // A fresh TT and deliberately deep search ensure every requested
            // budget is reached through the real negamax/qsearch accounting.
            let _ = searcher.negamax(&pos, 32, -MATE, MATE, 0, true, true);

            assert!(searcher.abort, "limit {limit} did not abort the search");
            assert_eq!(
                searcher.nodes, limit,
                "limit {limit} must stop at exactly that visited-node count"
            );
        }
    }

    /// A tiny public `go` node budget can abort before depth 1 completes; it
    /// must still return the documented first-legal-move fallback.
    #[test]
    fn node_limited_go_returns_a_legal_fallback_move() {
        let pos = fen::startpos();
        let tt = TT::new(16);
        let stop = AtomicBool::new(false);
        let lines = go(
            &pos,
            &Hce::default(),
            &Limits {
                nodes: Some(1),
                ..Default::default()
            },
            1,
            &tt,
            &stop,
            &[],
            0,
            SearchParams::default(),
            1,
            &mut |_| {},
        );

        assert_eq!(lines.len(), 1, "node-limited search must return a line");
        assert!(
            legal(&pos).as_slice().contains(&lines[0].mv),
            "fallback move {} must be legal",
            lines[0].mv.uci()
        );
        assert_eq!(lines[0].depth, 0, "one node cannot complete depth 1");
    }

    fn best_move(fen: &str, depth: i32) -> (String, i32) {
        let pos = fen::parse(fen).unwrap();
        let mut tt = TT::new(16);
        let stop = AtomicBool::new(false);
        let lines = go(
            &pos,
            &Hce::default(),
            &Limits::depth(depth),
            1,
            &mut tt,
            &stop,
            &[],
            0,
            SearchParams::default(),
            1,
            &mut |_| {},
        );
        let l = &lines[0];
        (l.mv.uci(), l.score)
    }

    #[test]
    fn finds_mate_in_one() {
        // Back-rank mate: Ra8#
        let (mv, sc) = best_move("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 4);
        assert_eq!(mv, "a1a8");
        assert!(is_mate_score(sc), "score {}", sc);
        assert_eq!(mate_in(sc), 1);
    }

    /// Matetrack-style suite: every forced mate must be found, every time.
    ///
    /// Stockfish gates mate-finding behaviour this way -- a fixed set of
    /// positions with known forced mates, run deterministically, with no
    /// games involved. It catches a failure SPRT is bad at detecting: an
    /// engine that is fine on average but has lost the ability to see a
    /// specific forced win. Their `seekMate` work is validated exactly like
    /// this ("61 FENs, 61 found mates").
    ///
    /// It matters here specifically because mate handling is this engine's
    /// weakest measured area. The theme breakdown put `mate_available` last
    /// of nine categories (top-1 0.2105, mean regret 408.7cp against 38.9cp
    /// for captures), and the real checkpoint ranks a forced back-rank mate
    /// 10th of 17. That is the policy rather than the search, but it is
    /// precisely why the search's own mate finding needs a standing test
    /// rather than four scattered assertions.
    ///
    /// The positions mirror `benchmarks/matetrack.epd`, which is generated
    /// and verified by `tools/build_matetrack_suite.py`: each `bm` is
    /// checked to be legal, mating, and the *unique* mate, so the expected
    /// move is unambiguous. `tools/test_matetrack_suite.py` asserts this
    /// list stays in sync with that file.
    #[test]
    fn matetrack_suite_finds_every_forced_mate() {
        // (id, fen, expected uci)
        let suite = [
            ("backrank-rook", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"),
            (
                "backrank-rook-black",
                "r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1",
                "a8a1",
            ),
            (
                "backrank-full-shield",
                "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
                "a1a8",
            ),
            (
                "smothered-knight",
                "6rk/6pp/8/6N1/8/8/8/6K1 w - - 0 1",
                "g5f7",
            ),
            (
                "queen-support-king",
                "7k/5K2/Q7/8/8/8/8/8 w - - 0 1",
                "a6h6",
            ),
            ("corner-queen", "k7/8/2K5/8/8/8/8/1Q6 w - - 0 1", "b1b7"),
            (
                "ladder-two-rooks",
                "7k/R7/1R6/8/8/8/8/6K1 w - - 0 1",
                "b6b8",
            ),
        ];

        for (id, fen, expected) in suite {
            let (mv, sc) = best_move(fen, 4);
            assert_eq!(mv, expected, "{id}: wrong move for {fen}");
            assert!(is_mate_score(sc), "{id}: score {sc} is not a mate score");
            assert_eq!(mate_in(sc), 1, "{id}: expected mate in 1, got {sc}");
        }
    }

    #[test]
    fn root_hints_cannot_override_a_forced_mate() {
        let pos = fen::parse("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1").unwrap();
        let moves = legal(&pos);
        let hints = moves
            .as_slice()
            .iter()
            .map(|&mv| RootHint {
                mv,
                // Deliberately put Ra8# last.
                policy_score: if mv.uci() == "a1a8" { -1000.0 } else { 1000.0 },
            })
            .collect::<Vec<_>>();
        let tt = TT::new(4);
        let stop = AtomicBool::new(false);
        let lines = go_with_root_hints(
            &pos,
            &Hce::default(),
            &Limits::depth(4),
            1,
            &tt,
            &stop,
            &[],
            0,
            SearchParams::default(),
            1,
            &hints,
            std::time::Duration::ZERO,
            &mut |_| {},
        );
        assert_eq!(lines[0].mv.uci(), "a1a8");
        assert!(is_mate_score(lines[0].score));
    }

    #[test]
    fn root_hints_cannot_override_black_back_rank_mate() {
        let pos = fen::parse("r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1").unwrap();
        let moves = legal(&pos);
        let hints = moves
            .as_slice()
            .iter()
            .map(|&mv| RootHint {
                mv,
                policy_score: if mv.uci() == "a8a1" { -1000.0 } else { 1000.0 },
            })
            .collect::<Vec<_>>();
        let tt = TT::new(4);
        let stop = AtomicBool::new(false);
        let lines = go_with_root_hints(
            &pos,
            &Hce::default(),
            &Limits::depth(4),
            1,
            &tt,
            &stop,
            &[],
            0,
            SearchParams::default(),
            1,
            &hints,
            std::time::Duration::ZERO,
            &mut |_| {},
        );
        assert_eq!(lines[0].mv.uci(), "a8a1");
        assert!(is_mate_score(lines[0].score));
    }

    #[test]
    fn stale_or_nonfinite_hints_cannot_remove_legal_moves() {
        let pos = fen::startpos();
        let stop = AtomicBool::new(false);
        let baseline_tt = TT::new(4);
        let baseline = go(
            &pos,
            &Hce::default(),
            &Limits::depth(3),
            1,
            &baseline_tt,
            &stop,
            &[],
            0,
            SearchParams::default(),
            1,
            &mut |_| {},
        );
        let hinted_tt = TT::new(4);
        let hinted = go_with_root_hints(
            &pos,
            &Hce::default(),
            &Limits::depth(3),
            1,
            &hinted_tt,
            &stop,
            &[],
            0,
            SearchParams::default(),
            1,
            &[
                RootHint {
                    mv: Move(0xffff),
                    policy_score: 1000.0,
                },
                RootHint {
                    mv: legal(&pos).as_slice()[0],
                    policy_score: f32::NAN,
                },
            ],
            std::time::Duration::ZERO,
            &mut |_| {},
        );
        assert_eq!(hinted[0].mv, baseline[0].mv);
        assert_eq!(hinted[0].score, baseline[0].score);
    }

    #[test]
    fn stalemate_position_ignores_root_hints() {
        let pos = fen::parse("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        assert_eq!(legal(&pos).len, 0);
        let tt = TT::new(4);
        let stop = AtomicBool::new(false);
        let lines = go_with_root_hints(
            &pos,
            &Hce::default(),
            &Limits::depth(4),
            1,
            &tt,
            &stop,
            &[],
            0,
            SearchParams::default(),
            1,
            &[RootHint {
                mv: Move(0xffff),
                policy_score: 1000.0,
            }],
            std::time::Duration::ZERO,
            &mut |_| {},
        );
        assert!(lines.is_empty());
    }

    #[test]
    fn precharged_root_hints_keep_only_move_legal_and_bounded() {
        let pos = fen::parse("R5k1/6pp/8/8/8/8/8/6K1 b - - 0 1").unwrap();
        let moves = legal(&pos);
        assert_eq!(moves.len, 1);
        let tt = TT::new(4);
        let stop = AtomicBool::new(false);
        let started = Instant::now();
        let lines = go_with_root_hints(
            &pos,
            &Hce::default(),
            &Limits::movetime(10),
            1,
            &tt,
            &stop,
            &[],
            0,
            SearchParams::default(),
            1,
            &[RootHint {
                mv: moves.as_slice()[0],
                policy_score: -1000.0,
            }],
            std::time::Duration::from_millis(20),
            &mut |_| {},
        );
        assert_eq!(lines[0].mv, moves.as_slice()[0]);
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }

    /// Only-legal-move-under-check, broadened past the rook check already
    /// covered above to a knight check (which, unlike a rook/bishop/queen
    /// check, can never be blocked -- only captured or escaped), with a
    /// hostile hint trying to rank the sole legal move last.
    #[test]
    fn precharged_root_hints_keep_only_move_under_knight_check() {
        let pos = fen::parse("k7/1p6/1N6/8/5B2/8/8/7K b - - 0 1").unwrap();
        let moves = legal(&pos);
        assert_eq!(
            moves.len, 1,
            "expected exactly one legal move under this knight check"
        );
        let tt = TT::new(4);
        let stop = AtomicBool::new(false);
        let started = Instant::now();
        let lines = go_with_root_hints(
            &pos,
            &Hce::default(),
            &Limits::movetime(10),
            1,
            &tt,
            &stop,
            &[],
            0,
            SearchParams::default(),
            1,
            &[RootHint {
                mv: moves.as_slice()[0],
                policy_score: -1000.0,
            }],
            std::time::Duration::from_millis(20),
            &mut |_| {},
        );
        assert_eq!(lines[0].mv, moves.as_slice()[0]);
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }

    /// Same as above, but the checking piece is a bishop (a sliding check
    /// that theoretically could be blocked, unlike the knight case, but
    /// isn't here because nothing can reach the checking diagonal in time).
    #[test]
    fn precharged_root_hints_keep_only_move_under_bishop_check() {
        let pos = fen::parse("7k/8/8/8/8/2B5/8/1K4R1 b - - 0 1").unwrap();
        let moves = legal(&pos);
        assert_eq!(
            moves.len, 1,
            "expected exactly one legal move under this bishop check"
        );
        let tt = TT::new(4);
        let stop = AtomicBool::new(false);
        let started = Instant::now();
        let lines = go_with_root_hints(
            &pos,
            &Hce::default(),
            &Limits::movetime(10),
            1,
            &tt,
            &stop,
            &[],
            0,
            SearchParams::default(),
            1,
            &[RootHint {
                mv: moves.as_slice()[0],
                policy_score: -1000.0,
            }],
            std::time::Duration::from_millis(20),
            &mut |_| {},
        );
        assert_eq!(lines[0].mv, moves.as_slice()[0]);
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }

    /// Same again, checking piece is a queen (the piece with the widest
    /// attack coverage, so the widest range of squares a hostile hint could
    /// plausibly try to steer toward instead of the one legal escape).
    #[test]
    fn precharged_root_hints_keep_only_move_under_queen_check() {
        let pos = fen::parse("7k/8/8/8/8/8/8/QK4R1 b - - 0 1").unwrap();
        let moves = legal(&pos);
        assert_eq!(
            moves.len, 1,
            "expected exactly one legal move under this queen check"
        );
        let tt = TT::new(4);
        let stop = AtomicBool::new(false);
        let started = Instant::now();
        let lines = go_with_root_hints(
            &pos,
            &Hce::default(),
            &Limits::movetime(10),
            1,
            &tt,
            &stop,
            &[],
            0,
            SearchParams::default(),
            1,
            &[RootHint {
                mv: moves.as_slice()[0],
                policy_score: -1000.0,
            }],
            std::time::Duration::from_millis(20),
            &mut |_| {},
        );
        assert_eq!(lines[0].mv, moves.as_slice()[0]);
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }

    /// A mate pattern using a different mating piece (queen, edge-supported
    /// by its own king) than the rook back-rank mates already covered above,
    /// with a hostile hint ranking the mating move last.
    #[test]
    fn root_hints_cannot_override_king_and_queen_mate() {
        let pos = fen::parse("7k/5K2/Q7/8/8/8/8/8 w - - 0 1").unwrap();
        let moves = legal(&pos);
        let hints = moves
            .as_slice()
            .iter()
            .map(|&mv| RootHint {
                mv,
                policy_score: if mv.uci() == "a6h6" { -1000.0 } else { 1000.0 },
            })
            .collect::<Vec<_>>();
        let tt = TT::new(4);
        let stop = AtomicBool::new(false);
        let lines = go_with_root_hints(
            &pos,
            &Hce::default(),
            &Limits::depth(4),
            1,
            &tt,
            &stop,
            &[],
            0,
            SearchParams::default(),
            1,
            &hints,
            std::time::Duration::ZERO,
            &mut |_| {},
        );
        assert_eq!(lines[0].mv.uci(), "a6h6");
        assert!(is_mate_score(lines[0].score));
    }

    /// Apply a recorded real-model policy ranking as root hints.
    ///
    /// `scored` holds the actual logits the exported checkpoint produced for
    /// this position. Any legal move missing from the list gets the minimum
    /// logit, so the mapping stays total even if movegen order differs.
    fn hints_from_recorded_logits(pos: &Position, scored: &[(&str, f32)]) -> Vec<RootHint> {
        let floor = scored.iter().map(|&(_, s)| s).fold(f32::INFINITY, f32::min);
        legal(pos)
            .as_slice()
            .iter()
            .map(|&mv| {
                let uci = mv.uci();
                let policy_score = scored
                    .iter()
                    .find(|&&(m, _)| m == uci.as_str())
                    .map(|&(_, s)| s)
                    .unwrap_or(floor);
                RootHint { mv, policy_score }
            })
            .collect()
    }

    /// The real checkpoint ranks this forced mate **10th of 17**, and the
    /// search must still play it.
    ///
    /// Every other hint test in this file feeds a hand-built adversarial
    /// ranking (`-1000.0` on the mate, `+1000.0` on everything else). That
    /// proves robustness against uniform hostile noise, but it is not the
    /// input most likely to expose a real ordering leak: a synthetic hint is
    /// obviously wrong, whereas a real policy is *confidently* wrong in a
    /// structured way, preferring natural-looking moves.
    ///
    /// These logits are not invented. They were measured by running
    /// `artifacts/unarchitectured-v1-final.unarchv1` through
    /// `tools/find_unarchitectured_v1_hint_disagreements.py`; the full output
    /// is committed at
    /// `benchmarks/unarchitectured-v1/hint-disagreements-2026-08-24.json`.
    /// The model puts Ka2/Kg2 and four rook moves ahead of Ra8#.
    #[test]
    fn real_checkpoint_ranking_cannot_suppress_back_rank_mate() {
        let pos = fen::parse("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1").unwrap();
        let scored = [
            ("g1h2", -0.917403f32),
            ("g1g2", -0.567702),
            ("g1f2", -0.635615),
            ("g1h1", -1.234828),
            ("g1f1", -1.019693),
            ("a1a8", -1.049943), // the mate, ranked 10th
            ("a1a7", -0.576414),
            ("a1a6", -1.010350),
            ("a1a5", -1.355014),
            ("a1a4", -1.240723),
            ("a1a3", -1.369275),
            ("a1a2", -1.337878),
            ("a1f1", -0.858250),
            ("a1e1", -0.806330),
            ("a1d1", -0.894320),
            ("a1c1", -0.909317),
            ("a1b1", -1.101056),
        ];
        let hints = hints_from_recorded_logits(&pos, &scored);

        // Sanity-check the premise: if the model ever starts ranking the mate
        // first, this test silently stops testing anything.
        let best_hint = hints
            .iter()
            .max_by(|a, b| a.policy_score.partial_cmp(&b.policy_score).unwrap())
            .unwrap();
        assert_ne!(
            best_hint.mv.uci(),
            "a1a8",
            "fixture no longer represents a disagreement"
        );

        let tt = TT::new(4);
        let stop = AtomicBool::new(false);
        let lines = go_with_root_hints(
            &pos,
            &Hce::default(),
            &Limits::depth(4),
            1,
            &tt,
            &stop,
            &[],
            0,
            SearchParams::default(),
            1,
            &hints,
            std::time::Duration::ZERO,
            &mut |_| {},
        );
        assert_eq!(lines[0].mv.uci(), "a1a8");
        assert!(is_mate_score(lines[0].score));
    }

    /// A real, non-synthetic disagreement in a full middlegame position.
    ///
    /// The checkpoint ranks the Greek-gift sacrifice Bxf7+ **18th of 38**,
    /// strongly preferring quiet castling (the only move it scores above
    /// zero). This is the widest real logit gap found (1.91), and unlike the
    /// mate fixtures the position is a normal opening tabiya rather than a
    /// constructed endgame -- so root ordering is under maximum realistic
    /// pressure from a plausible-looking wrong move.
    ///
    /// This asserts the *search stays sound*, not that it finds Bxf7 -- at
    /// shallow depth with HCE the objectively best move here is genuinely
    /// debatable, and asserting a specific move would be encoding an opinion
    /// rather than a correctness property. What must hold is that a hostile
    /// real ranking cannot make the search return an illegal or absurd move.
    #[test]
    fn real_checkpoint_ranking_keeps_middlegame_search_sound() {
        let pos =
            fen::parse("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1")
                .unwrap();
        let scored = [
            ("c4f7", -1.774796f32), // the sacrifice, ranked 18th
            ("c4e6", -2.502336),
            ("c4a6", -3.061207),
            ("c4d5", -2.111246),
            ("c4b5", -1.818687),
            ("c4b3", -1.717077),
            ("f3g5", -1.821042),
            ("f3e5", -1.468683),
            ("f3h4", -2.010518),
            ("f3d4", -2.138683),
            ("f3d2", -2.010799),
            ("f3g1", -2.458552),
            ("h1g1", -2.137744),
            ("h1f1", -2.031782),
            ("e1e2", -2.309721),
            ("e1d2", -2.288298),
            ("e1f1", -2.188198),
            ("d1e2", -1.428620),
            ("d1d2", -1.474914),
            ("c1h6", -3.256310),
            ("c1g5", -0.725913),
            ("c1f4", -2.460410),
            ("c1e3", -1.295096),
            ("c1d2", -1.444858),
            ("b1c3", -0.614645),
            ("b1a3", -1.860586),
            ("b1d2", -1.233407),
            ("e1g1", 0.136316), // model's top choice: castle
            ("d3d4", -1.338036),
            ("h2h3", -1.322452),
            ("g2g3", -1.904789),
            ("c2c3", -1.422695),
            ("b2b3", -1.633388),
            ("a2a3", -1.312639),
            ("h2h4", -1.591149),
            ("g2g4", -1.988203),
            ("b2b4", -1.341225),
            ("a2a4", -1.749212),
        ];
        let hints = hints_from_recorded_logits(&pos, &scored);
        let tt = TT::new(4);
        let stop = AtomicBool::new(false);
        let lines = go_with_root_hints(
            &pos,
            &Hce::default(),
            &Limits::depth(6),
            1,
            &tt,
            &stop,
            &[],
            0,
            SearchParams::default(),
            1,
            &hints,
            std::time::Duration::ZERO,
            &mut |_| {},
        );

        // The returned move must be one of the position's real legal moves.
        let legal_ucis: Vec<String> = legal(&pos).as_slice().iter().map(|m| m.uci()).collect();
        assert!(
            legal_ucis.contains(&lines[0].mv.uci()),
            "search returned {} which is not legal here",
            lines[0].mv.uci()
        );
        // Nobody is getting mated in this quiet opening position.
        assert!(!is_mate_score(lines[0].score));
        // A sane evaluation: this is a balanced tabiya, not a lost position.
        assert!(
            lines[0].score.abs() < 500,
            "implausible score {} for a balanced opening",
            lines[0].score
        );
    }

    #[test]
    fn finds_mate_in_two() {
        // Classic: 1.Qh7+!? no — use a known M2: white Qg7#? Position:
        // k7/8/2K5/8/8/8/8/1Q6 w - - 0 1 : 1.Qb7# is mate in 1 actually.
        let (mv, sc) = best_move("k7/8/2K5/8/8/8/8/1Q6 w - - 0 1", 6);
        assert!(is_mate_score(sc));
        assert_eq!(mv, "b1b7");
    }

    #[test]
    fn takes_hanging_queen() {
        // Black queen hangs on d5, white knight on c3 can take? Use rook takes.
        let (mv, _) = best_move("6k1/8/8/3q4/8/8/8/3R2K1 w - - 0 1", 5);
        assert_eq!(mv, "d1d5");
    }

    #[test]
    fn game_mode_detection() {
        assert!(Limits::movetime(500).is_game_mode());
        assert!(
            Limits::depth(10).is_game_mode(),
            "fixed-depth matches are games"
        );
        assert!(Limits {
            wtime: Some(60_000),
            btime: Some(60_000),
            ..Default::default()
        }
        .is_game_mode());
        assert!(
            !Limits {
                infinite: true,
                ..Default::default()
            }
            .is_game_mode(),
            "go infinite is analysis"
        );
        assert!(!Limits::default().is_game_mode());
    }

    #[test]
    fn budget_speeds_up_as_clock_drains() {
        fn soft_for(ms: u64) -> u64 {
            let l = Limits {
                wtime: Some(ms),
                btime: Some(ms),
                winc: Some(0),
                binc: Some(0),
                ..Default::default()
            };
            l.budget(Color::White).0.unwrap()
        }
        let full = soft_for(180_000);
        let mid = soft_for(20_000);
        let low = soft_for(5_000);
        let panic = soft_for(1_000);
        assert!(
            full > mid && mid > low && low >= panic,
            "budgets must shrink: {} {} {} {}",
            full,
            mid,
            low,
            panic
        );
        // low clock spends a much smaller *fraction* of remaining time too
        assert!((low as f64) / 5_000.0 < (full as f64) / 180_000.0 * 0.8);
        // panic mode is near-instant
        assert!(panic <= 40, "panic soft budget {}", panic);
        // hard limit always leaves a reserve
        let l = Limits {
            wtime: Some(300),
            btime: Some(300),
            ..Default::default()
        };
        let (_, hard) = l.budget(Color::White);
        assert!(hard.unwrap() <= 300 - 60 + 5, "hard {}", hard.unwrap());
    }

    #[test]
    fn forced_move_returns_quickly() {
        // rook check on the back rank; Kf7 is the only legal move
        let pos = fen::parse("R5k1/6pp/8/8/8/8/8/6K1 b - - 0 1").unwrap();
        let ml = crate::movegen::legal(&pos);
        assert_eq!(ml.len, 1, "test position must have exactly one legal move");
        let mut tt = TT::new(4);
        let stop = AtomicBool::new(false);
        let limits = Limits {
            wtime: Some(60_000),
            btime: Some(60_000),
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let lines = go(
            &pos,
            &Hce::default(),
            &limits,
            1,
            &mut tt,
            &stop,
            &[],
            0,
            SearchParams::default(),
            1,
            &mut |_| {},
        );
        assert_eq!(lines[0].mv.uci(), ml.as_slice()[0].uci());
        assert!(
            t0.elapsed().as_millis() < 1000,
            "forced move took {} ms",
            t0.elapsed().as_millis()
        );
    }

    #[test]
    fn multipv_returns_distinct_moves() {
        let pos = fen::startpos();
        let mut tt = TT::new(16);
        let stop = AtomicBool::new(false);
        let lines = go(
            &pos,
            &Hce::default(),
            &Limits::depth(6),
            4,
            &mut tt,
            &stop,
            &[],
            0,
            SearchParams::default(),
            1,
            &mut |_| {},
        );
        assert_eq!(lines.len(), 4);
        let mut mvs: Vec<String> = lines.iter().map(|l| l.mv.uci()).collect();
        mvs.dedup();
        assert_eq!(mvs.len(), 4, "multipv lines must be distinct: {:?}", mvs);
        // scores should be non-increasing
        for w in lines.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }
}
