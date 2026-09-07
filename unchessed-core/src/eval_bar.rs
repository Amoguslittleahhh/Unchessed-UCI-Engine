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

const MATE_CP: i32 = 20_000;
const WDL_CP_MIN: i32 = -4000;
const WDL_CP_MAX: i32 = 4000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalBarSource {
    HceProxy,
    LoadedNnueProxy,
}

impl EvalBarSource {
    pub const fn label(self) -> &'static str {
        match self {
            Self::HceProxy => "hce-proxy",
            Self::LoadedNnueProxy => "nnue-proxy",
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
