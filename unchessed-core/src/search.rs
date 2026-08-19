//! Iterative-deepening alpha-beta search with quiescence, transposition table,
//! null-move pruning, LMR, killers/history ordering, MultiPV and time management.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::board::*;
use crate::eval::{Eval, EvalState, MG_VALUE};
use crate::movegen::{generate, in_check, king_safe_after, legal, MoveList};
use crate::see::see;
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
    fn budget(&self, side: Color) -> (Option<u64>, Option<u64>) {
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

pub struct InfoEvent<'a> {
    pub depth: i32,
    pub multipv: usize,
    pub score: i32,
    pub nodes: u64,
    pub time_ms: u64,
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
    #[inline]
    fn check_limits(&mut self) {
        if self.nodes & 2047 == 0 {
            if self.stop.load(Ordering::Relaxed) {
                self.abort = true;
                return;
            }
            if let Some(h) = self.hard_ms {
                if self.start.elapsed().as_millis() as u64 >= h {
                    self.abort = true;
                    return;
                }
            }
            if let Some(n) = self.node_limit {
                if self.nodes >= n {
                    self.abort = true;
                }
            }
        }
    }

    #[inline]
    fn is_repetition(&self, hash: u64) -> bool {
        self.path.iter().rev().any(|&h| h == hash)
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
    fn move_score(&self, pos: &Position, m: Move, tt_mv: Move, ply: usize) -> i32 {
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
            let sc = see(pos, m);
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
        for i in 0..ml.len {
            scores[i] = self.move_score(pos, ml.moves[i], Move::NONE, ply.min(MAX_PLY - 1));
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
            let child_state = self.eval.update_state(pos, &next, m, &self.eval_states[ply]);
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

        // draws
        if pos.halfmove >= 100 {
            return self.draw(ply);
        }
        if self.is_repetition(pos.hash) {
            return self.draw(ply);
        }

        let in_chk = in_check(pos);
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
        if !is_pv
            && !in_chk
            && depth >= self.params.probcut_min_depth
            && beta.abs() < MATE_IN_MAX
        {
            let beta_cut = beta + self.params.probcut_margin;
            let rdepth = depth - self.params.probcut_reduction;
            if rdepth >= 1 {
                let mut pc_ml = MoveList::new();
                generate(pos, true, &mut pc_ml);
                let mut pc_scores = [0i32; 256];
                for i in 0..pc_ml.len {
                    pc_scores[i] =
                        self.move_score(pos, pc_ml.moves[i], Move::NONE, ply.min(MAX_PLY - 1));
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
                    let m = pc_ml.moves[i];
                    let next = pos.make(m);
                    if !king_safe_after(&next, pos.side) {
                        continue;
                    }
                    let child_state = self.eval.update_state(pos, &next, m, &self.eval_states[ply]);
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
        for i in 0..ml.len {
            scores[i] = self.move_score(pos, ml.moves[i], tt_mv, ply);
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
            if !king_safe_after(&next, us) {
                continue;
            }
            legal_count += 1;
            let child_state = self.eval.update_state(pos, &next, m, &self.eval_states[ply]);
            self.eval_states[ply + 1] = child_state;

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
                    r = 1
                        + if legal_count > self.params.lmr_big_movenum { 1 } else { 0 }
                        + if !is_pv { 1 } else { 0 };
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
    let start = Instant::now();
    let (base_soft, hard_ms) = limits.budget(pos.side);
    let max_depth = limits.depth.unwrap_or(MAX_PLY as i32 - 1).clamp(1, MAX_PLY as i32 - 1);

    let root_moves_list = legal(pos);
    if root_moves_list.len == 0 {
        return Vec::new();
    }
    let multipv = multipv.max(1).min(root_moves_list.len);

    // situation-based allocation: sharp or wide positions deserve more of
    // the clock, simple ones less; a single legal move needs almost none
    let root_in_check = in_check(pos);
    let situation = {
        let width = (0.65 + root_moves_list.len as f64 / 45.0).clamp(0.75, 1.3);
        let sharp = if root_in_check { 1.25 } else { 1.0 };
        width * sharp
    };
    let mut soft_ms = base_soft.map(|s| {
        ((s as f64 * situation) as u64)
            .clamp(3, hard_ms.unwrap_or(u64::MAX))
    });

    struct RootMove {
        mv: Move,
        score: i32,
        pv: Vec<Move>,
        depth: i32,
    }
    let mut roots: Vec<RootMove> = root_moves_list
        .as_slice()
        .iter()
        .map(|&m| RootMove {
            mv: m,
            score: -MATE,
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
        // order by previous scores, best first
        roots.sort_by_key(|r| -r.score);
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
                let failed_low = window_lo > -MATE && result.map(|r| r <= window_lo).unwrap_or(true);
                let failed_high = window_hi < MATE && result.map(|r| r >= window_hi).unwrap_or(false);
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
        assert!(Limits::depth(10).is_game_mode(), "fixed-depth matches are games");
        assert!(Limits {
            wtime: Some(60_000),
            btime: Some(60_000),
            ..Default::default()
        }
        .is_game_mode());
        assert!(!Limits {
            infinite: true,
            ..Default::default()
        }
        .is_game_mode(), "go infinite is analysis");
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
        assert!(full > mid && mid > low && low >= panic,
            "budgets must shrink: {} {} {} {}", full, mid, low, panic);
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
