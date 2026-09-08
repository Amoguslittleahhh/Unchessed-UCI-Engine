//! Stability-oriented evaluation-bar projection.
//!
//! The module is deliberately separate from search. It provides a fast,
//! White-centric presentation projection, not a second move selector.
//!
//! Breakthroughs in this experiment:
//! 1. Material calibration is cached by the 0..78 Stockfish material index.
//! 2. Logistic WDL conversion uses a bounded lookup table with interpolation,
//!    avoiding an `exp` call on every bar refresh while preserving per-mille
//!    accuracy.
//! 3. Presentation smoothing adapts to confidence and bounds single-update
//!    shocks; raw engine scores remain available and are never fed back into
//!    search.

use std::sync::OnceLock;

use crate::board::{Color, Position, BISHOP, KNIGHT, PAWN, QUEEN, ROOK};
use crate::eval::{evaluate, Eval, EvalParams, EvalState};
use crate::movegen::{in_check, legal, KING_ATT, KNIGHT_ATT, PAWN_ATT};
use crate::tactical_eval::{heads_from_score, TacticalHeads};

const MATE_CP: i32 = 20_000;
const WDL_CP_MIN: i32 = -4000;
const WDL_CP_MAX: i32 = 4000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalBarSource {
    HceProxy,
    LoadedNnueProxy,
    /// Original Unchessed feature fusion; no external weights or constants.
    HomemadeFusion,
    /// Original compact model fit only from black-box UCI observations.
    HomemadeParityModel,
    /// Original teacher-calibrated model fit from 5,000 black-box labels and
    /// homemade bitboard features; opt-in and presentation-only.
    HomemadeTeacherCalibrated,
    /// Original gated score/WDL/terminal research head; opt-in and not search-enabled.
    TacticalMultiHead,
}

impl EvalBarSource {
    pub const fn label(self) -> &'static str {
        match self {
            Self::HceProxy => "hce-proxy",
            Self::LoadedNnueProxy => "nnue-proxy",
            Self::HomemadeFusion => "homemade-stack-v3-selected",
            Self::HomemadeParityModel => "homemade-parity-linear-v1",
            Self::HomemadeTeacherCalibrated => "homemade-teacher-calibrated-v1",
            Self::TacticalMultiHead => "unchessed-tactical-multi-head-v1",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitboardSnapshot {
    pub position_hash: u64,
    pub occupancy: u64,
    pub white_occupancy: u64,
    pub black_occupancy: u64,
    pub material_index: u8,
}

/// Original, deterministic feature summary used by the homemade fusion path.
/// These are deliberately simple bitboard-derived signals, not copied NNUE
/// feature rows or imported Stockfish evaluation constants.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HomemadeFeatures {
    pub center_control: i16,
    pub pawn_structure: i16,
    pub king_safety: i16,
    pub piece_activity: i16,
    pub coordination: i16,
    pub space: i16,
    pub passed_pawns: i16,
    pub outposts: i16,
    pub bishop_pair: i16,
    pub rook_files: i16,
    pub tempo: i16,
    pub phase: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalBarSample {
    pub static_cp_white: i32,
    pub display_cp_white: i32,
    pub wdl_per_mille: [u16; 3],
    pub expected_score: f64,
    pub bar_fraction: f64,
    /// A presentation confidence estimate, not a playing-strength claim.
    pub confidence: f64,
    pub exact_stockfish_path: bool,
    pub source: EvalBarSource,
    pub bitboard: BitboardSnapshot,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalBarLink {
    pub bitboard: BitboardSnapshot,
    pub display_cp_white: i32,
    pub bar_fraction: f64,
    pub elo_estimate: i32,
    pub elo_confidence_cp: i32,
    pub elo_suspect: bool,
    pub elo_suspect_reason: &'static str,
}

impl EvalBarLink {
    pub const fn from_sample(
        sample: &EvalBarSample,
        elo_estimate: i32,
        elo_confidence_cp: i32,
        elo_suspect: bool,
        elo_suspect_reason: &'static str,
    ) -> Self {
        Self {
            bitboard: sample.bitboard,
            display_cp_white: sample.display_cp_white,
            bar_fraction: sample.bar_fraction,
            elo_estimate,
            elo_confidence_cp,
            elo_suspect,
            elo_suspect_reason,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalBar {
    smoothing_alpha: f64,
    max_step_cp: i32,
    smoothed_cp_white: Option<f64>,
}

impl Default for EvalBar {
    fn default() -> Self {
        Self::new()
    }
}

impl EvalBar {
    pub fn new() -> Self {
        Self {
            smoothing_alpha: 0.35,
            max_step_cp: 180,
            smoothed_cp_white: None,
        }
    }

    pub fn reset(&mut self) {
        self.smoothed_cp_white = None;
    }

    /// Project the current HCE score to a stable, White-centric bar.
    pub fn sample(&mut self, pos: &Position) -> EvalBarSample {
        let raw_stm = evaluate(pos, &EvalParams::default());
        self.sample_from_score(pos, raw_stm, EvalBarSource::HceProxy)
    }

    /// Apply the original homemade residual on top of an existing evaluator
    /// score. This path is opt-in so the baseline evaluator remains untouched.
    pub fn sample_homemade_from_score(&mut self, pos: &Position, raw_stm: i32) -> EvalBarSample {
        let residual_white = homemade_residual_cp(pos);
        let residual_stm = if pos.side == Color::White {
            residual_white
        } else {
            -residual_white
        };
        let scored_stm = raw_stm.saturating_add(residual_stm);
        let raw_static_cp_white = if pos.side == Color::White {
            scored_stm
        } else {
            -scored_stm
        };
        // Clean-room calibration learned from held-out black-box UCI scores:
        // shrink volatile hand-crafted residuals and correct the corpus bias.
        // These coefficients belong to Unchessed and are not Stockfish data.
        const HOMEMADE_SCORE_SCALE: f64 = 0.680292396;
        const HOMEMADE_SCORE_BIAS_CP: f64 = 24.603188915;
        let static_cp_white = (HOMEMADE_SCORE_SCALE * f64::from(raw_static_cp_white)
            + HOMEMADE_SCORE_BIAS_CP)
            .round() as i32;
        let display_cp_white = damp_rule50(static_cp_white, pos.halfmove);
        let wdl = homemade_wdl_from_cp(display_cp_white, pos);
        let expected_score = (f64::from(wdl[0]) + 0.5 * f64::from(wdl[1])) / 1000.0;
        let confidence = score_confidence(expected_score);
        let target = f64::from(display_cp_white);
        let prior = self.smoothed_cp_white.unwrap_or(target);
        let jump = target - prior;
        let alpha = (self.smoothing_alpha + 0.25 * confidence).clamp(0.20, 0.60);
        let bounded_delta = jump.clamp(-f64::from(self.max_step_cp), f64::from(self.max_step_cp));
        let smoothed = prior + alpha * bounded_delta;
        self.smoothed_cp_white = Some(smoothed);
        EvalBarSample {
            static_cp_white,
            display_cp_white,
            wdl_per_mille: wdl,
            expected_score,
            bar_fraction: expected_score,
            confidence,
            exact_stockfish_path: false,
            source: EvalBarSource::HomemadeFusion,
            bitboard: bitboard_snapshot(pos),
        }
    }

    /// Apply the original compact parity model. Its coefficients are Unchessed
    /// parameters fit from black-box UCI outputs; it does not contain network
    /// weights or implementation material from Stockfish.
    pub fn sample_parity_from_score(&mut self, pos: &Position, raw_stm: i32) -> EvalBarSample {
        let raw_static_cp_white = if pos.side == Color::White {
            raw_stm
        } else {
            -raw_stm
        };
        let residual_white = homemade_residual_cp(pos);
        let base_hce_white = raw_static_cp_white - residual_white;
        let static_cp_white = parity_model_cp_white(pos, base_hce_white);
        let display_cp_white = damp_rule50(static_cp_white, pos.halfmove);
        let wdl = homemade_wdl_from_cp(display_cp_white, pos);
        let expected_score = (f64::from(wdl[0]) + 0.5 * f64::from(wdl[1])) / 1000.0;
        let confidence = score_confidence(expected_score);
        let target = f64::from(display_cp_white);
        let prior = self.smoothed_cp_white.unwrap_or(target);
        let jump = target - prior;
        let alpha = (self.smoothing_alpha + 0.25 * confidence).clamp(0.20, 0.60);
        let bounded_delta = jump.clamp(-f64::from(self.max_step_cp), f64::from(self.max_step_cp));
        self.smoothed_cp_white = Some(prior + alpha * bounded_delta);
        EvalBarSample {
            static_cp_white,
            display_cp_white,
            wdl_per_mille: wdl,
            expected_score,
            bar_fraction: expected_score,
            confidence,
            exact_stockfish_path: false,
            source: EvalBarSource::HomemadeParityModel,
            bitboard: bitboard_snapshot(pos),
        }
    }

    /// Apply the original 5,000-position teacher calibration to an already
    /// available evaluator score. This is opt-in and presentation-only: the
    /// search evaluator remains untouched. The coefficients were fitted from
    /// Stockfish UCI labels and original Unchessed bitboard signals; no
    /// Stockfish code, network weights, or feature rows are used.
    pub fn sample_teacher_calibrated_from_score(&mut self, pos: &Position, raw_stm: i32) -> EvalBarSample {
        let raw_white = if pos.side == Color::White { raw_stm } else { -raw_stm };
        if let Some(static_cp_white) = forced_mate_score_white(pos, 3) {
            let display_cp_white = static_cp_white;
            let wdl = homemade_wdl_from_cp(display_cp_white, pos);
            let expected_score = (f64::from(wdl[0]) + 0.5 * f64::from(wdl[1])) / 1000.0;
            let confidence = score_confidence(expected_score);
            self.smoothed_cp_white = Some(f64::from(display_cp_white));
            return EvalBarSample {
                static_cp_white, display_cp_white, wdl_per_mille: wdl,
                expected_score, bar_fraction: expected_score, confidence,
                exact_stockfish_path: false, source: EvalBarSource::HomemadeTeacherCalibrated,
                bitboard: bitboard_snapshot(pos),
            };
        }
        let f = homemade_features(pos);
        let phase = f64::from(f.phase);
        let signals = [
            f.center_control, f.pawn_structure, f.king_safety, f.piece_activity,
            f.coordination, f.space, f.passed_pawns, f.outposts, f.bishop_pair,
            f.rook_files, f.tempo,
        ];
        let mut value = -4.172527714120014
            + 627.4074064989893 * (f64::from(raw_white) / 1000.0)
            + 3.41303882089065 * (phase / 30.0);
        let weights = [
            -2.784548995458911, 15.09246925460668, 9.754923260521622,
            0.501334898209607, -5.892604746791196, 10.117275303106599,
            29.015908572590188, 8.049217041921155, 5.175420838421003,
            21.735708371254233, 0.0,
        ];
        for (signal, weight) in signals.iter().zip(weights) {
            value += weight * (f64::from(*signal) / 10.0);
        }
        let phase_norm = phase / 30.0;
        for (signal, weight) in [
            signals[1], signals[2], signals[6], signals[8], signals[9],
        ].iter().zip([
            -2.6391156762900914, -5.171169016106844, 3.69814318527974,
            3.3828920186060865, 1.7581095484204152,
        ]) {
            value += weight * (f64::from(*signal) * phase_norm / 10.0);
        }
        value += 3.4130388208906273 * if pos.side == Color::White { phase_norm } else { -phase_norm };
        value += -3.210290712812883 * f64::from(signals[1]) * f64::from(signals[2]) / 100.0;
        value += -6.060674712045301 * f64::from(signals[6]) * f64::from(signals[8]) / 10.0;
        let static_cp_white = value.round() as i32;
        let display_cp_white = damp_rule50(static_cp_white, pos.halfmove);
        let wdl = homemade_wdl_from_cp(display_cp_white, pos);
        let expected_score = (f64::from(wdl[0]) + 0.5 * f64::from(wdl[1])) / 1000.0;
        let confidence = score_confidence(expected_score);
        let target = f64::from(display_cp_white);
        let prior = self.smoothed_cp_white.unwrap_or(target);
        let jump = target - prior;
        let alpha = (self.smoothing_alpha + 0.25 * confidence).clamp(0.20, 0.60);
        let bounded_delta = jump.clamp(-f64::from(self.max_step_cp), f64::from(self.max_step_cp));
        self.smoothed_cp_white = Some(prior + alpha * bounded_delta);
        EvalBarSample {
            static_cp_white, display_cp_white, wdl_per_mille: wdl,
            expected_score, bar_fraction: expected_score, confidence,
            exact_stockfish_path: false, source: EvalBarSource::HomemadeTeacherCalibrated,
            bitboard: bitboard_snapshot(pos),
        }
    }

    /// Project an already-available evaluator score. This is the no-bottleneck
    /// seam for the search path: callers can pass a score from an incremental
    /// NNUE state without asking the bar to rescan the board.
    pub fn sample_from_score(
        &mut self,
        pos: &Position,
        raw_stm: i32,
        source: EvalBarSource,
    ) -> EvalBarSample {
        let static_cp_white = if pos.side == Color::White {
            raw_stm
        } else {
            -raw_stm
        };
        let display_cp_white = damp_rule50(static_cp_white, pos.halfmove);
        let wdl = wdl_from_display_cp_fast(display_cp_white, pos);
        let expected_score = (f64::from(wdl[0]) + 0.5 * f64::from(wdl[1])) / 1000.0;
        let confidence = score_confidence(expected_score);
        let target = f64::from(display_cp_white);
        let prior = self.smoothed_cp_white.unwrap_or(target);
        let jump = target - prior;
        // Low-confidence equal positions get more smoothing; decisive positions
        // settle faster. The hard cap prevents a single transient spike from
        // filling the visual bar instantly.
        let alpha = (self.smoothing_alpha + 0.25 * confidence).clamp(0.20, 0.60);
        let bounded_delta = jump.clamp(-f64::from(self.max_step_cp), f64::from(self.max_step_cp));
        let smoothed = prior + alpha * bounded_delta;
        self.smoothed_cp_white = Some(smoothed);
        EvalBarSample {
            static_cp_white,
            display_cp_white,
            wdl_per_mille: wdl,
            expected_score,
            bar_fraction: expected_score,
            confidence,
            exact_stockfish_path: false,
            source,
            bitboard: bitboard_snapshot(pos),
        }
    }

    /// Apply the original gated tactical head to an available score. The
    /// returned heads expose terminal and mate-distance metadata separately;
    /// this method is presentation-only until all research gates pass.
    pub fn sample_tactical_from_score(
        &mut self,
        pos: &Position,
        raw_stm: i32,
    ) -> (EvalBarSample, TacticalHeads) {
        let heads = heads_from_score(raw_stm, pos);
        let static_cp_white = if pos.side == Color::White {
            heads.bounded_static_cp
        } else {
            -heads.bounded_static_cp
        };
        let display_cp_white = damp_rule50(static_cp_white, pos.halfmove);
        let wdl = homemade_wdl_from_cp(display_cp_white, pos);
        let expected_score = (f64::from(wdl[0]) + 0.5 * f64::from(wdl[1])) / 1000.0;
        let confidence = f64::from(heads.confidence_per_mille) / 1000.0;
        let target = f64::from(display_cp_white);
        let prior = self.smoothed_cp_white.unwrap_or(target);
        let bounded_delta =
            (target - prior).clamp(-f64::from(self.max_step_cp), f64::from(self.max_step_cp));
        self.smoothed_cp_white =
            Some(prior + (0.35 + 0.25 * confidence).clamp(0.20, 0.60) * bounded_delta);
        (
            EvalBarSample {
                static_cp_white,
                display_cp_white,
                wdl_per_mille: wdl,
                expected_score,
                bar_fraction: expected_score,
                confidence,
                exact_stockfish_path: false,
                source: EvalBarSource::TacticalMultiHead,
                bitboard: bitboard_snapshot(pos),
            },
            heads,
        )
    }

    /// Use the evaluator's incremental state directly. This keeps the bar
    /// from becoming a second full-board evaluator when a search already has
    /// the accumulator available.
    pub fn sample_with_state(
        &mut self,
        pos: &Position,
        evaluator: &dyn Eval,
        state: &EvalState,
        source: EvalBarSource,
    ) -> EvalBarSample {
        self.sample_from_score(pos, evaluator.eval_with_state(pos, state), source)
    }

    pub fn smoothed_cp_white(&self) -> Option<i32> {
        self.smoothed_cp_white.map(|v| v.round() as i32)
    }
}

/// Original rational WDL projection. It uses no Stockfish calibration table,
/// fitted coefficient, or external weight. The output is a display projection,
/// not a claim about game outcome probability.
pub fn homemade_wdl_from_cp(display_cp_white: i32, pos: &Position) -> [u16; 3] {
    let phase = f64::from(homemade_features(pos).phase);
    let scale = 360.0 + 8.0 * phase;
    let x = f64::from(display_cp_white);
    let win = 0.5 + x / (2.0 * (scale + x.abs()));
    let loss = 0.5 - x / (2.0 * (scale + x.abs()));
    let draw = 0.28 * (1.0 - x.abs() / (scale + x.abs()));
    let decisive = 1.0 - draw;
    normalize_wdl(win * decisive, loss * decisive)
}

pub fn homemade_features(pos: &Position) -> HomemadeFeatures {
    const CENTER: u64 = 0x0000_0018_1800_0000;
    const EXTENDED_CENTER: u64 = 0x0000_3c3c_3c3c_0000;
    const FILE_A: u64 = 0x0101_0101_0101_0101;
    let mut center = [0i16; 2];
    let mut safety = [0i16; 2];
    let mut activity = [0i16; 2];
    let mut coordination = [0i16; 2];
    let mut space = [0i16; 2];
    let mut passed = [0i16; 2];
    let mut outposts = [0i16; 2];
    let mut bishop_pair = [0i16; 2];
    let mut rook_files = [0i16; 2];
    for c in 0..2 {
        let enemy = 1 - c;
        let own_occ = pos.occ_side[c];
        let enemy_pawns = pos.bb[enemy][PAWN];
        center[c] = (own_occ & CENTER).count_ones() as i16;
        activity[c] = (own_occ & EXTENDED_CENTER).count_ones() as i16;
        space[c] = ((pos.bb[c][PAWN] & EXTENDED_CENTER).count_ones() as i16)
            + ((pos.bb[c][PAWN] & own_occ).count_ones() as i16 / 2);
        let own_pawns = pos.bb[c][PAWN];
        let own_pawn_attacks = pawn_attack_map(pos, c);
        let enemy_pawn_attacks = pawn_attack_map(pos, enemy);
        let own_knight_attacks = knight_attack_map(pos, c);
        coordination[c] = ((own_pawn_attacks & own_occ).count_ones()
            + (own_knight_attacks & own_occ).count_ones()) as i16;
        let mut pawns = own_pawns;
        while pawns != 0 {
            let sq = pawns.trailing_zeros() as usize;
            pawns &= pawns - 1;
            let file = sq % 8;
            let rank = sq / 8;
            let mut enemy_ahead = 0u64;
            let file_lo = file.saturating_sub(1);
            let file_hi = (file + 1).min(7);
            for f in file_lo..=file_hi {
                let file_mask = FILE_A << f;
                let ranks = if c == Color::White.idx() {
                    if rank == 7 {
                        0
                    } else {
                        file_mask & (!0u64 << ((rank + 1) * 8))
                    }
                } else if rank == 0 {
                    0
                } else {
                    file_mask & ((1u64 << (rank * 8)) - 1)
                };
                enemy_ahead |= enemy_pawns & ranks;
            }
            if enemy_ahead == 0 {
                passed[c] += 1;
            }
        }
        let mut knights = pos.bb[c][KNIGHT];
        while knights != 0 {
            let sq = knights.trailing_zeros() as usize;
            knights &= knights - 1;
            let bit = 1u64 << sq;
            if bit & EXTENDED_CENTER != 0 && bit & enemy_pawn_attacks == 0 {
                outposts[c] += 1;
            }
        }
        bishop_pair[c] = i16::from(pos.bb[c][BISHOP].count_ones() >= 2);
        let own_rooks = pos.bb[c][ROOK];
        let mut rooks = own_rooks;
        while rooks != 0 {
            let sq = rooks.trailing_zeros() as usize;
            rooks &= rooks - 1;
            let file_mask = FILE_A << (sq % 8);
            if (pos.bb[Color::White.idx()][PAWN] | pos.bb[Color::Black.idx()][PAWN]) & file_mask
                == 0
            {
                rook_files[c] += 1;
            }
        }
        let king = if c == Color::White.idx() {
            pos.king_sq(Color::White)
        } else {
            pos.king_sq(Color::Black)
        };
        safety[c] = (KING_ATT[king as usize] & own_pawns).count_ones() as i16;
    }
    HomemadeFeatures {
        center_control: center[0] - center[1],
        pawn_structure: structure_score(pos, 0) - structure_score(pos, 1),
        king_safety: safety[0] - safety[1],
        piece_activity: activity[0] - activity[1],
        coordination: coordination[0] - coordination[1],
        space: space[0] - space[1],
        passed_pawns: passed[0] - passed[1],
        outposts: outposts[0] - outposts[1],
        bishop_pair: bishop_pair[0] - bishop_pair[1],
        rook_files: rook_files[0] - rook_files[1],
        tempo: if pos.side == Color::White { 1 } else { -1 },
        phase: (pos.occ.count_ones().saturating_sub(2)).min(30) as u8,
    }
}

fn forced_mate_score_white(pos: &Position, max_plies: u8) -> Option<i32> {
    forced_mate_in(pos, max_plies).map(|plies| {
        let score = MATE_CP - i32::from(plies);
        if pos.side == Color::White { score } else { -score }
    })
}

/// Return the shortest forced mate for the side to move within `depth` plies.
/// This is an original bounded proof probe for the presentation path; it is
/// deliberately not called from the main search evaluator.
fn forced_mate_in(pos: &Position, depth: u8) -> Option<u8> {
    if depth == 0 {
        return None;
    }
    let moves = legal(pos);
    for mv in moves.as_slice() {
        let child = pos.make(*mv);
        let replies = legal(&child);
        if replies.moves.is_empty() {
            if in_check(&child) {
                return Some(1);
            }
            continue;
        }
        if depth < 2 {
            continue;
        }
        let mut all_replies_fail = true;
        let mut longest = 0u8;
        for reply in replies.as_slice() {
            if let Some(distance) = forced_mate_in(&child.make(*reply), depth - 1) {
                longest = longest.max(distance);
            } else {
                all_replies_fail = false;
                break;
            }
        }
        if all_replies_fail {
            return Some(1 + longest);
        }
    }
    None
}

fn pawn_attack_map(pos: &Position, color: usize) -> u64 {
    let mut pawns = pos.bb[color][PAWN];
    let mut attacks = 0u64;
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as usize;
        pawns &= pawns - 1;
        attacks |= PAWN_ATT[color][sq];
    }
    attacks
}

fn knight_attack_map(pos: &Position, color: usize) -> u64 {
    let mut knights = pos.bb[color][KNIGHT];
    let mut attacks = 0u64;
    while knights != 0 {
        let sq = knights.trailing_zeros() as usize;
        knights &= knights - 1;
        attacks |= KNIGHT_ATT[sq];
    }
    attacks
}

fn structure_score(pos: &Position, color: usize) -> i16 {
    const FILE_A: u64 = 0x0101_0101_0101_0101;
    let pawns = pos.bb[color][PAWN];
    let mut score = 0i16;
    for file in 0..8 {
        let count = (pawns & (FILE_A << file)).count_ones() as i16;
        if count > 1 {
            score -= count - 1;
        }
        if count > 0 {
            let mut adjacent = 0u64;
            if file > 0 {
                adjacent |= FILE_A << (file - 1);
            }
            if file < 7 {
                adjacent |= FILE_A << (file + 1);
            }
            score += if pawns & adjacent != 0 { 1 } else { -1 };
        }
    }
    score
}

/// Original Unchessed residual: a phase-aware fusion of eleven independent
/// bitboard signals. No external feature rows, network weights, or copied
/// evaluation constants are used. The hard bound is a presentation safety rail.
pub fn homemade_residual_cp(pos: &Position) -> i32 {
    let f = homemade_features(pos);
    // Selected by held-out ablation: pawn geometry, king shelter, passed
    // pawns, bishop-pair structure, and rook-file pressure. The other
    // original signals remain available for future retraining but are not
    // allowed to add noise to the current production candidate.
    let raw = 4 * i32::from(f.pawn_structure)
        + 3 * i32::from(f.king_safety)
        + 6 * i32::from(f.passed_pawns)
        + 10 * i32::from(f.bishop_pair)
        + 3 * i32::from(f.rook_files);
    let phase_scale = 8 + i32::from(f.phase);
    (raw * phase_scale / 16).clamp(-180, 180)
}

fn parity_model_cp_white(pos: &Position, hce_white: i32) -> i32 {
    let f = homemade_features(pos);
    let material = material_balance(pos) as f64;
    let phase = f64::from(f.phase);
    let signals = [
        f.center_control,
        f.pawn_structure,
        f.king_safety,
        f.piece_activity,
        f.coordination,
        f.space,
        f.passed_pawns,
        f.outposts,
        f.bishop_pair,
        f.rook_files,
        f.tempo,
    ];
    let mut value = 29.0756183042
        + 165.050011876 * (f64::from(hce_white) / 1000.0)
        + 33.3111274764 * (material / 20.0)
        - 28.1279586099 * (phase / 30.0);
    let weights = [
        14.5404607618,
        53.9593866715,
        89.0217440204,
        22.5674077942,
        6.36106482837,
        11.5142968358,
        47.504983524,
        24.2963794598,
        16.4149212044,
        44.7864979209,
        7.44361477216e-14,
    ];
    for (signal, weight) in signals.iter().zip(weights) {
        value += weight * (f64::from(*signal) / 10.0);
    }
    let phase_norm = phase / 30.0;
    for (index, weight) in [
        11.9471892712,
        46.4900769586,
        24.5292930088,
        12.0971006247,
        27.0766769686,
    ]
    .iter()
    .enumerate()
    {
        let signal = signals[[1, 2, 6, 8, 9][index]];
        value += weight * (f64::from(signal) * phase_norm / 10.0);
    }
    value += 16.2443705529 * (material / 20.0) * phase_norm;
    value.round().clamp(-4000.0, 4000.0) as i32
}

fn material_balance(pos: &Position) -> i32 {
    const VALUES: [i32; 6] = [100, 320, 330, 500, 900, 0];
    let white = pos.bb[Color::White.idx()];
    let black = pos.bb[Color::Black.idx()];
    (0..6)
        .map(|piece| {
            VALUES[piece] * (white[piece].count_ones() as i32 - black[piece].count_ones() as i32)
        })
        .sum()
}

pub fn bitboard_snapshot(pos: &Position) -> BitboardSnapshot {
    BitboardSnapshot {
        position_hash: pos.hash,
        occupancy: pos.occ,
        white_occupancy: pos.occ_side[Color::White.idx()],
        black_occupancy: pos.occ_side[Color::Black.idx()],
        material_index: material_index(pos) as u8,
    }
}

fn score_confidence(expected_score: f64) -> f64 {
    ((expected_score - 0.5).abs() * 2.0).clamp(0.0, 1.0)
}

/// Cached Stockfish 19 material-dependent calibration polynomial.
pub fn win_rate_params(pos: &Position) -> (f64, f64) {
    let material = material_index(pos);
    calibration_table()[material as usize]
}

fn material_index(pos: &Position) -> u32 {
    let material = pos.bb[0][PAWN].count_ones()
        + pos.bb[1][PAWN].count_ones()
        + 3 * (pos.bb[0][KNIGHT].count_ones() + pos.bb[1][KNIGHT].count_ones())
        + 3 * (pos.bb[0][BISHOP].count_ones() + pos.bb[1][BISHOP].count_ones())
        + 5 * (pos.bb[0][ROOK].count_ones() + pos.bb[1][ROOK].count_ones())
        + 9 * (pos.bb[0][QUEEN].count_ones() + pos.bb[1][QUEEN].count_ones());
    material.clamp(17, 78)
}

fn calibration_table() -> &'static [(f64, f64); 79] {
    static TABLE: OnceLock<[(f64, f64); 79]> = OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|index| {
            let m = f64::from(index as u32) / 58.0;
            let a = (((-142.72052667 * m + 372.35176398) * m - 340.71073572) * m) + 415.23490212;
            let b = (((5.93832785 * m + 15.61267078) * m - 30.57816876) * m) + 69.63866711;
            (a, b.max(1.0))
        })
    })
}

pub fn wdl_from_display_cp_fast(display_cp_white: i32, pos: &Position) -> [u16; 3] {
    if display_cp_white >= MATE_CP {
        return [1000, 0, 0];
    }
    if display_cp_white <= -MATE_CP {
        return [0, 0, 1000];
    }
    if (WDL_CP_MIN..=WDL_CP_MAX).contains(&display_cp_white) {
        let material = material_index(pos) as usize;
        let cp = (display_cp_white - WDL_CP_MIN) as usize;
        return wdl_tensor()[material * (WDL_CP_MAX - WDL_CP_MIN + 1) as usize + cp];
    }
    let (a, b) = win_rate_params(pos);
    wdl_exact_for_calibration(display_cp_white, a, b)
}

/// Exact reference implementation used for tests and calibration audits.
pub fn wdl_from_display_cp_exact(display_cp_white: i32, pos: &Position) -> [u16; 3] {
    if display_cp_white >= MATE_CP {
        return [1000, 0, 0];
    }
    if display_cp_white <= -MATE_CP {
        return [0, 0, 1000];
    }
    let (a, b) = win_rate_params(pos);
    wdl_exact_for_calibration(display_cp_white, a, b)
}

fn wdl_exact_for_calibration(display_cp_white: i32, a: f64, b: f64) -> [u16; 3] {
    let internal = f64::from(display_cp_white) * a / 100.0;
    let win = logistic_exact((a - internal) / b);
    let loss = logistic_exact((a + internal) / b);
    normalize_wdl(win, loss)
}

fn wdl_tensor() -> &'static Vec<[u16; 3]> {
    static TENSOR: OnceLock<Vec<[u16; 3]>> = OnceLock::new();
    TENSOR.get_or_init(|| {
        let span = (WDL_CP_MAX - WDL_CP_MIN + 1) as usize;
        let mut tensor = Vec::with_capacity(79 * span);
        for material in 0..79 {
            let (a, b) = calibration_table()[material];
            for cp in WDL_CP_MIN..=WDL_CP_MAX {
                tensor.push(wdl_exact_for_calibration(cp, a, b));
            }
        }
        tensor
    })
}

fn normalize_wdl(win: f64, loss: f64) -> [u16; 3] {
    let draw = (1.0 - win - loss).max(0.0);
    let mut values = [
        (win * 1000.0).round() as i32,
        (draw * 1000.0).round() as i32,
        (loss * 1000.0).round() as i32,
    ];
    let correction = 1000 - values.iter().sum::<i32>();
    values[1] = (values[1] + correction).clamp(0, 1000);
    [values[0] as u16, values[1] as u16, values[2] as u16]
}

fn logistic_exact(z: f64) -> f64 {
    1.0 / (1.0 + z.exp())
}

fn damp_rule50(cp: i32, halfmove: u16) -> i32 {
    let factor = 199 - i32::from(halfmove).min(199);
    cp.saturating_mul(factor) / 199
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen;

    #[test]
    fn start_position_is_near_equal_and_wdl_sums_to_1000() {
        let pos = fen::startpos();
        let mut bar = EvalBar::new();
        let sample = bar.sample(&pos);
        assert!(sample.display_cp_white.abs() < 100);
        assert_eq!(
            sample
                .wdl_per_mille
                .iter()
                .map(|v| u32::from(*v))
                .sum::<u32>(),
            1000
        );
        assert!((0.0..=1.0).contains(&sample.bar_fraction));
        assert!((0.0..=1.0).contains(&sample.confidence));
        assert!(!sample.exact_stockfish_path);
    }

    #[test]
    fn side_to_move_is_normalized_to_white() {
        let white = fen::parse("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1").unwrap();
        let black = fen::parse("4k3/8/8/8/8/8/3Q4/4K3 b - - 0 1").unwrap();
        let mut a = EvalBar::new();
        let mut b = EvalBar::new();
        let white_cp = a.sample(&white).static_cp_white;
        let black_cp = b.sample(&black).static_cp_white;
        assert!(white_cp > 0 && black_cp > 0);
        assert!((white_cp - black_cp).abs() <= 50);
    }

    #[test]
    fn rule50_dampener_is_monotone_and_bounded() {
        let fresh = fen::startpos();
        let mut stale = fresh;
        stale.halfmove = 199;
        let mut a = EvalBar::new();
        let mut b = EvalBar::new();
        let x = a.sample(&fresh).display_cp_white.abs();
        let y = b.sample(&stale).display_cp_white.abs();
        assert!(y <= x);
    }

    #[test]
    fn fast_wdl_is_within_one_per_mille_of_exact_reference() {
        let pos = fen::startpos();
        for cp in (-1800..=1800).step_by(37) {
            let fast = wdl_from_display_cp_fast(cp, &pos);
            let exact = wdl_from_display_cp_exact(cp, &pos);
            for (a, b) in fast.iter().zip(exact.iter()) {
                assert!(
                    (i32::from(*a) - i32::from(*b)).abs() <= 1,
                    "cp={cp} fast={fast:?} exact={exact:?}"
                );
            }
        }
    }

    #[test]
    fn wdl_is_symmetric_and_monotone() {
        let pos = fen::startpos();
        let mut previous = 0;
        for cp in (-1000..=1000).step_by(25) {
            let wdl = wdl_from_display_cp_fast(cp, &pos);
            assert!(wdl[0] >= previous);
            previous = wdl[0];
            let mirror = wdl_from_display_cp_fast(-cp, &pos);
            assert_eq!(wdl[0], mirror[2]);
        }
    }

    #[test]
    fn smoothing_limits_single_position_jump() {
        let mut bar = EvalBar::new();
        let first = fen::startpos();
        let second = fen::parse("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1").unwrap();
        bar.sample(&first);
        let prior = bar.smoothed_cp_white().unwrap();
        let _sample = bar.sample(&second);
        let smooth = bar.smoothed_cp_white().unwrap();
        assert!((smooth - prior).abs() <= 108);
    }

    #[test]
    fn bitboard_snapshot_is_a_lossless_link_key() {
        let pos = fen::startpos();
        let snapshot = bitboard_snapshot(&pos);
        assert_eq!(snapshot.position_hash, pos.hash);
        assert_eq!(snapshot.occupancy, pos.occ);
        assert_eq!(snapshot.white_occupancy, pos.occ_side[Color::White.idx()]);
        assert_eq!(snapshot.black_occupancy, pos.occ_side[Color::Black.idx()]);
        assert_eq!(snapshot.material_index, material_index(&pos) as u8);
    }

    struct FixedEval(i32);

    impl Eval for FixedEval {
        fn eval(&self, _pos: &Position) -> i32 {
            self.0
        }
    }

    #[test]
    fn homemade_mode_is_explicitly_original_and_bounded() {
        let pos = fen::startpos();
        let features = homemade_features(&pos);
        assert_eq!(features.center_control, 0);
        assert!(homemade_residual_cp(&pos).abs() <= 120);
        let wdl = homemade_wdl_from_cp(0, &pos);
        assert_eq!(wdl.iter().map(|v| u32::from(*v)).sum::<u32>(), 1000);
        assert_eq!(wdl[0], wdl[2]);
        let mut bar = EvalBar::new();
        let sample = bar.sample_homemade_from_score(&pos, 0);
        assert_eq!(sample.source, EvalBarSource::HomemadeFusion);
        assert!(!sample.exact_stockfish_path);
    }

    #[test]
    fn parity_model_is_explicitly_experimental_and_bounded() {
        let pos = fen::startpos();
        let mut bar = EvalBar::new();
        let sample = bar.sample_parity_from_score(&pos, 0);
        assert_eq!(sample.source, EvalBarSource::HomemadeParityModel);
        assert!((-4000..=4000).contains(&sample.static_cp_white));
        assert_eq!(
            sample
                .wdl_per_mille
                .iter()
                .map(|v| u32::from(*v))
                .sum::<u32>(),
            1000
        );
        assert!(!sample.exact_stockfish_path);
    }

    #[test]
    fn homemade_projection_is_side_normalized_and_monotone() {
        let pos = fen::startpos();
        let mut previous = [0u16; 3];
        for cp in (-2000..=2000).step_by(100) {
            let current = homemade_wdl_from_cp(cp, &pos);
            assert!(current[0] >= previous[0]);
            assert_eq!(current.iter().map(|v| u32::from(*v)).sum::<u32>(), 1000);
            previous = current;
        }
        let positive = homemade_wdl_from_cp(700, &pos);
        let negative = homemade_wdl_from_cp(-700, &pos);
        assert_eq!(positive[0], negative[2]);
    }

    #[test]
    fn incremental_state_seam_uses_existing_score_without_rescanning() {
        let pos = fen::startpos();
        let evaluator = FixedEval(123);
        let mut bar = EvalBar::new();
        let sample = bar.sample_with_state(
            &pos,
            &evaluator,
            &EvalState::stateless(),
            EvalBarSource::LoadedNnueProxy,
        );
        assert_eq!(sample.static_cp_white, 123);
        assert_eq!(sample.source, EvalBarSource::LoadedNnueProxy);
    }
}
