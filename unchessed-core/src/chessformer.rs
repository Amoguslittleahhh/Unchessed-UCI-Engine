//! CPU-oriented Chessformer input contract and persona/search router.
//!
//! This is deliberately model-agnostic infrastructure. It fixes the 64-square
//! tokenization, dynamic geometric relation keys, continuous-Elo context, and
//! safe backend-routing semantics before any trained transformer asset exists.
//! A later `UNCHFORM` loader can implement a small encoder-only model without
//! changing datasets or persona decisions.

use crate::adapt::Mode;
use crate::board::*;
use crate::movegen::legal;
use crate::threat_features::attacks_from;

pub const BOARD_TOKENS: usize = 64;
pub const EMPTY_TOKEN: u8 = 0;
pub const NO_EP_FILE: u8 = 8;
pub const POLICY_HISTORY_PLIES: usize = 8;
pub const AEGIS_MODEL_UUID_BYTES: usize = 16;
pub const POLICY_PROMOTION_CLASSES: usize = 5;
pub const POLICY_ACTION_VOCABULARY: usize = 64 * 64 * POLICY_PROMOTION_CLASSES;
pub const MAX_LEGAL_POLICY_ACTIONS: usize = 218;

/// Discrete square token consumed by the proposed tiny Chessformer.
///
/// `piece` is perspective-relative: 0 empty, 1..6 own P..K, 7..12 opponent
/// P..K. Black-to-move positions are rank-flipped so the mover always advances
/// toward increasing ranks. Global state is repeated per token to keep the
/// inference format simple and friendly to int8 embedding lookups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SquareToken {
    pub piece: u8,
    pub square: u8,
    pub castling: u8,
    pub ep_file: u8,
    pub halfmove_bucket: u8,
    pub elo_context: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChessformerInput {
    pub tokens: [SquareToken; BOARD_TOKENS],
    pub flipped: bool,
}

/// Aegis v3 keeps temporal/rating information in a private human-policy
/// adapter. It is intentionally absent from the board-state value trunk, so
/// transpositions cannot acquire different values from player history.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TemporalPolicyContext {
    /// Most recent normalized moves, newest first. Each entry retains the full
    /// internal move kind/promotion bits; unused slots are zero.
    pub history: [u16; POLICY_HISTORY_PLIES],
    pub history_len: u8,
    /// 0 bullet, 1 blitz, 2 rapid, 3 classical, 4 unknown.
    pub time_class: u8,
    pub elo_context: u8,
}

impl TemporalPolicyContext {
    pub fn new(history: &[Move], perspective: Color, time_class: u8, target_elo: i32) -> Self {
        let mut normalized = [0u16; POLICY_HISTORY_PLIES];
        for (slot, mv) in history.iter().rev().take(POLICY_HISTORY_PLIES).enumerate() {
            let flip = perspective == Color::Black;
            let from = orient_square(mv.from() as usize, flip) as u16;
            let to = orient_square(mv.to() as usize, flip) as u16;
            normalized[slot] = from | (to << 6) | (mv.0 & 0xf000);
        }
        Self {
            history: normalized,
            history_len: history.len().min(POLICY_HISTORY_PLIES) as u8,
            time_class: time_class.min(4),
            elo_context: (((target_elo.clamp(100, 3650) - 100) * 255) / 3550) as u8,
        }
    }
}

/// Exact board-value cache key. `ep_square` is explicit because canonical
/// repetition hashing omits pseudo-uncapturable en-passant targets while the
/// square-token encoder still exposes them. A v3 value never keys on persona,
/// Elo, clock class, or move history.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BoardValueCacheKey {
    pub position_hash: u64,
    pub model_uuid: [u8; AEGIS_MODEL_UUID_BYTES],
    pub ep_square: u8,
    pub halfmove_bucket: u8,
}

/// Exact human/guide-policy cache key. The full temporal context is stored,
/// not reduced to a collision-prone digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PolicyCacheKey {
    pub board: BoardValueCacheKey,
    pub context: TemporalPolicyContext,
    pub persona: u8,
}

pub fn board_value_cache_key(
    pos: &Position,
    model_uuid: [u8; AEGIS_MODEL_UUID_BYTES],
) -> BoardValueCacheKey {
    BoardValueCacheKey {
        position_hash: pos.hash,
        model_uuid,
        ep_square: pos.ep,
        halfmove_bucket: (pos.halfmove / 8).min(15) as u8,
    }
}

pub fn policy_cache_key(
    pos: &Position,
    model_uuid: [u8; AEGIS_MODEL_UUID_BYTES],
    context: TemporalPolicyContext,
    persona: u8,
) -> PolicyCacheKey {
    PolicyCacheKey {
        board: board_value_cache_key(pos, model_uuid),
        context,
        persona,
    }
}

/// One exact legal policy action. The action vocabulary is
/// `(promotion_class * 4096) + (to * 64) + from`, where promotion class is
/// 0 for an ordinary move and 1/2/3/4 for knight/bishop/rook/queen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyActionEntry {
    pub action: u16,
    pub mv: Move,
}

const EMPTY_POLICY_ACTION: PolicyActionEntry = PolicyActionEntry {
    action: 0,
    mv: Move::NONE,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegalPolicyActions {
    entries: [PolicyActionEntry; MAX_LEGAL_POLICY_ACTIONS],
    len: usize,
    overflowed: bool,
}

impl Default for LegalPolicyActions {
    fn default() -> Self {
        Self {
            entries: [EMPTY_POLICY_ACTION; MAX_LEGAL_POLICY_ACTIONS],
            len: 0,
            overflowed: false,
        }
    }
}

impl LegalPolicyActions {
    pub fn as_slice(&self) -> &[PolicyActionEntry] {
        &self.entries[..self.len]
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    pub fn find_action(&self, action: u16) -> Option<Move> {
        self.as_slice()
            .binary_search_by_key(&action, |entry| entry.action)
            .ok()
            .map(|index| self.entries[index].mv)
    }
}

#[inline]
pub fn encode_policy_action(mv: Move, perspective: Color) -> u16 {
    let flip = perspective == Color::Black;
    let from = orient_square(mv.from() as usize, flip) as u16;
    let to = orient_square(mv.to() as usize, flip) as u16;
    let promotion = if mv.is_promo() {
        mv.promo_piece() as u16
    } else {
        0
    };
    debug_assert!((promotion as usize) < POLICY_PROMOTION_CLASSES);
    from | (to << 6) | (promotion << 12)
}

/// Enumerate, promotion-disambiguate, and sort every legal move for the policy
/// head. The theoretical legal maximum is 218; overflow is explicit and must
/// disable the model rather than train/infer on a truncated action set.
pub fn legal_policy_actions(pos: &Position) -> LegalPolicyActions {
    let mut output = LegalPolicyActions::default();
    for &mv in legal(pos).as_slice() {
        if output.len == MAX_LEGAL_POLICY_ACTIONS {
            output.overflowed = true;
            continue;
        }
        output.entries[output.len] = PolicyActionEntry {
            action: encode_policy_action(mv, pos.side),
            mv,
        };
        output.len += 1;
    }
    output.entries[..output.len].sort_unstable_by_key(|entry| entry.action);
    output
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyCandidateScore {
    pub entry: PolicyActionEntry,
    pub logit: f32,
    /// Conformal upper bound on common-budget regret in centipawns.
    pub regret_upper_cp: f32,
}

const EMPTY_CANDIDATE_SCORE: PolicyCandidateScore = PolicyCandidateScore {
    entry: EMPTY_POLICY_ACTION,
    logit: f32::NEG_INFINITY,
    regret_upper_cp: f32::INFINITY,
};

/// Search ordering plan produced by Aegis v4. Priority moves are searched
/// first, but all legal moves remain in the plan: statistical coverage is not
/// a proof that omitted chess moves are bad.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicySearchPlan {
    ordered: [PolicyCandidateScore; MAX_LEGAL_POLICY_ACTIONS],
    len: usize,
    priority_len: usize,
    pub certificate_valid: bool,
    pub full_legal_fallback_required: bool,
}

impl PolicySearchPlan {
    pub fn ordered(&self) -> &[PolicyCandidateScore] {
        &self.ordered[..self.len]
    }

    pub fn priority(&self) -> &[PolicyCandidateScore] {
        &self.ordered[..self.priority_len]
    }
}

/// Build a conformal-regret candidate set without granting it pruning
/// authority. Non-finite model values fail to the non-priority partition.
pub fn build_policy_search_plan(
    candidates: &[PolicyCandidateScore],
    regret_tolerance_cp: f32,
    maximum_priority_moves: usize,
    observed_coverage_bps: u16,
    required_coverage_bps: u16,
) -> Option<PolicySearchPlan> {
    if candidates.is_empty() || candidates.len() > MAX_LEGAL_POLICY_ACTIONS {
        return None;
    }
    let mut sorted = [EMPTY_CANDIDATE_SCORE; MAX_LEGAL_POLICY_ACTIONS];
    sorted[..candidates.len()].copy_from_slice(candidates);
    sorted[..candidates.len()].sort_by(|a, b| {
        b.logit
            .partial_cmp(&a.logit)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.entry.action.cmp(&b.entry.action))
    });
    let cap = maximum_priority_moves.clamp(1, candidates.len());
    let tolerance = regret_tolerance_cp.max(0.0);
    let mut ordered = [EMPTY_CANDIDATE_SCORE; MAX_LEGAL_POLICY_ACTIONS];
    let mut selected = [false; MAX_LEGAL_POLICY_ACTIONS];
    let mut priority_len = 0usize;
    for (index, candidate) in sorted[..candidates.len()].iter().enumerate() {
        if priority_len < cap
            && candidate.logit.is_finite()
            && candidate.regret_upper_cp.is_finite()
            && candidate.regret_upper_cp <= tolerance
        {
            ordered[priority_len] = *candidate;
            selected[index] = true;
            priority_len += 1;
        }
    }
    if priority_len == 0 {
        ordered[0] = sorted[0];
        selected[0] = true;
        priority_len = 1;
    }
    let mut cursor = priority_len;
    for (index, candidate) in sorted[..candidates.len()].iter().enumerate() {
        if !selected[index] {
            ordered[cursor] = *candidate;
            cursor += 1;
        }
    }
    Some(PolicySearchPlan {
        ordered,
        len: candidates.len(),
        priority_len,
        certificate_valid: observed_coverage_bps.min(10_000) >= required_coverage_bps.min(10_000),
        // Calibration is empirical coverage, not a formal chess proof.
        full_legal_fallback_required: true,
    })
}

/// Bit flags for a dynamic source/destination attention bias.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeometryRelation {
    pub file_delta: i8,
    pub rank_delta: i8,
    pub chebyshev_distance: u8,
    pub flags: u16,
}

impl GeometryRelation {
    pub const SAME_RANK: u16 = 1 << 0;
    pub const SAME_FILE: u16 = 1 << 1;
    pub const SAME_DIAGONAL: u16 = 1 << 2;
    pub const KNIGHT_GEOMETRY: u16 = 1 << 3;
    pub const KING_NEIGHBOR: u16 = 1 << 4;
    pub const SOURCE_ATTACKS_TARGET: u16 = 1 << 5;
    pub const TARGET_ATTACKS_SOURCE: u16 = 1 << 6;
    pub const TARGET_OCCUPIED: u16 = 1 << 7;
    pub const SAME_OWNER: u16 = 1 << 8;

    pub fn has(self, flag: u16) -> bool {
        self.flags & flag != 0
    }
}

fn normalized_castling(pos: &Position) -> u8 {
    if pos.side == Color::White {
        pos.castling & 0x0f
    } else {
        let mut rights = 0;
        if pos.castling & BK != 0 {
            rights |= WK;
        }
        if pos.castling & BQ != 0 {
            rights |= WQ;
        }
        if pos.castling & WK != 0 {
            rights |= BK;
        }
        if pos.castling & WQ != 0 {
            rights |= BQ;
        }
        rights
    }
}

#[inline]
fn orient_square(square: usize, flip: bool) -> usize {
    if flip {
        square ^ 56
    } else {
        square
    }
}

#[inline]
fn original_square(oriented: usize, flip: bool) -> usize {
    orient_square(oriented, flip)
}

/// Encode one position into exactly 64 board-square tokens.
pub fn encode_position(pos: &Position, target_elo: i32) -> ChessformerInput {
    let flip = pos.side == Color::Black;
    let mover = pos.side;
    let castling = normalized_castling(pos);
    let ep_file = if pos.ep == NO_EP {
        NO_EP_FILE
    } else {
        file_of(pos.ep)
    };
    let halfmove_bucket = (pos.halfmove / 8).min(15) as u8;
    let elo_context = (((target_elo.clamp(100, 3650) - 100) * 255) / 3550) as u8;
    let empty = SquareToken {
        piece: EMPTY_TOKEN,
        square: 0,
        castling,
        ep_file,
        halfmove_bucket,
        elo_context,
    };
    let mut tokens = [empty; BOARD_TOKENS];
    for (oriented, token) in tokens.iter_mut().enumerate() {
        let original = original_square(oriented, flip);
        let piece = match pos.piece_on(original as u8) {
            Some((color, kind)) => 1 + kind as u8 + if color == mover { 0 } else { 6 },
            None => EMPTY_TOKEN,
        };
        *token = SquareToken {
            piece,
            square: oriented as u8,
            castling,
            ep_file,
            halfmove_bucket,
            elo_context,
        };
    }
    ChessformerInput {
        tokens,
        flipped: flip,
    }
}

/// Board-only Aegis v3 trunk input. Elo is deliberately zeroed and supplied
/// separately through `TemporalPolicyContext` to a private policy adapter.
pub fn encode_board_state_v3(pos: &Position) -> ChessformerInput {
    encode_position(pos, 100)
}

/// Dynamic geometry key for Geometric-Attention-Bias-style lookup/MLP input.
pub fn geometric_relation(
    pos: &Position,
    source_oriented: usize,
    target_oriented: usize,
) -> GeometryRelation {
    debug_assert!(source_oriented < 64 && target_oriented < 64);
    let flip = pos.side == Color::Black;
    let source = original_square(source_oriented, flip);
    let target = original_square(target_oriented, flip);
    let sf = (source_oriented & 7) as i8;
    let sr = (source_oriented >> 3) as i8;
    let tf = (target_oriented & 7) as i8;
    let tr = (target_oriented >> 3) as i8;
    let df = tf - sf;
    let dr = tr - sr;
    let adf = df.unsigned_abs();
    let adr = dr.unsigned_abs();
    let mut flags = 0u16;
    if dr == 0 {
        flags |= GeometryRelation::SAME_RANK;
    }
    if df == 0 {
        flags |= GeometryRelation::SAME_FILE;
    }
    if adf == adr {
        flags |= GeometryRelation::SAME_DIAGONAL;
    }
    if (adf == 1 && adr == 2) || (adf == 2 && adr == 1) {
        flags |= GeometryRelation::KNIGHT_GEOMETRY;
    }
    if adf <= 1 && adr <= 1 && source != target {
        flags |= GeometryRelation::KING_NEIGHBOR;
    }
    let source_piece = pos.piece_on(source as u8);
    let target_piece = pos.piece_on(target as u8);
    if let Some((color, piece)) = source_piece {
        if attacks_from(pos, color, piece, source) & (1u64 << target) != 0 {
            flags |= GeometryRelation::SOURCE_ATTACKS_TARGET;
        }
    }
    if let Some((color, piece)) = target_piece {
        flags |= GeometryRelation::TARGET_OCCUPIED;
        if attacks_from(pos, color, piece, target) & (1u64 << source) != 0 {
            flags |= GeometryRelation::TARGET_ATTACKS_SOURCE;
        }
    }
    if matches!((source_piece, target_piece), (Some((a, _)), Some((b, _))) if a == b) {
        flags |= GeometryRelation::SAME_OWNER;
    }
    GeometryRelation {
        file_delta: df,
        rank_delta: dr,
        chebyshev_distance: adf.max(adr),
        flags,
    }
}

/// Search architecture selected once per move by the persona layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HybridBackend {
    /// Conventional NNUE alpha-beta; authoritative for engines, GMs, defense,
    /// tactical conversion, unknown opponents, and all model-missing cases.
    AlphaBeta,
    /// Chessformer policy orders root/top-ply moves; alpha-beta remains the
    /// value authority and verifies every chosen move.
    PolicyGuidedAlphaBeta,
    /// Sample a human policy at the target Elo, then apply an alpha-beta safety
    /// veto for forced mate and catastrophic tactical loss.
    HumanPolicyGuarded,
}

impl HybridBackend {
    pub fn name(self) -> &'static str {
        match self {
            Self::AlphaBeta => "ALPHABETA",
            Self::PolicyGuidedAlphaBeta => "POLICY_GUIDED_AB",
            Self::HumanPolicyGuarded => "HUMAN_POLICY_GUARDED",
        }
    }
}

/// Conservative routing contract for the future trained Chessformer asset.
///
/// This function has no effect on production play until a caller explicitly
/// wires a verified model. It codifies the safety boundary now so a missing or
/// uncertain model can never weaken play against an engine.
pub fn choose_backend(
    model_available: bool,
    adaptive: bool,
    fixed_strength: bool,
    target_elo: i32,
    mode: Mode,
    confident_human: bool,
    engine_or_uncertain: bool,
) -> HybridBackend {
    if !model_available
        || !adaptive
        || engine_or_uncertain
        || matches!(mode, Mode::Full | Mode::Punish | Mode::Defend)
        || target_elo >= 2300
    {
        return HybridBackend::AlphaBeta;
    }
    if fixed_strength || (confident_human && mode == Mode::Match && target_elo <= 2100) {
        return HybridBackend::HumanPolicyGuarded;
    }
    if confident_human && matches!(mode, Mode::Match | Mode::Clinch) {
        return HybridBackend::PolicyGuidedAlphaBeta;
    }
    HybridBackend::AlphaBeta
}

/// Elastic Chessformer exit selected from calibrated evidential uncertainty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChessformerExit {
    Layer2Width128,
    Layer4Width192,
    Layer8Width256,
}

impl ChessformerExit {
    pub fn layers(self) -> usize {
        match self {
            Self::Layer2Width128 => 2,
            Self::Layer4Width192 => 4,
            Self::Layer8Width256 => 8,
        }
    }

    pub fn width(self) -> usize {
        match self {
            Self::Layer2Width128 => 128,
            Self::Layer4Width192 => 192,
            Self::Layer8Width256 => 256,
        }
    }
}

/// Convert non-negative WDL evidence to expected probabilities and Dirichlet
/// uncertainty `u = 3 / sum(alpha)`, where `alpha = evidence + 1`.
pub fn evidential_wdl(evidence: [f32; 3]) -> ([f32; 3], f32) {
    let alpha = evidence.map(|value| value.max(0.0) + 1.0);
    let strength = alpha.iter().sum::<f32>().max(3.0);
    (
        alpha.map(|value| value / strength),
        (3.0 / strength).clamp(0.0, 1.0),
    )
}

/// Select adaptive transformer depth. Alpha-beta backends do not invoke the
/// transformer. In-check or high-uncertainty positions always use full depth.
pub fn choose_chessformer_exit(
    backend: HybridBackend,
    uncertainty: f32,
    in_check: bool,
    remaining_ms: u64,
) -> Option<ChessformerExit> {
    if backend == HybridBackend::AlphaBeta {
        return None;
    }
    let uncertainty = uncertainty.clamp(0.0, 1.0);
    if in_check || uncertainty > 0.16 {
        return Some(ChessformerExit::Layer8Width256);
    }
    match backend {
        HybridBackend::HumanPolicyGuarded if uncertainty <= 0.08 => {
            Some(ChessformerExit::Layer2Width128)
        }
        HybridBackend::PolicyGuidedAlphaBeta if uncertainty <= 0.08 && remaining_ms < 2_000 => {
            Some(ChessformerExit::Layer4Width192)
        }
        HybridBackend::HumanPolicyGuarded | HybridBackend::PolicyGuidedAlphaBeta => {
            Some(ChessformerExit::Layer4Width192)
        }
        HybridBackend::AlphaBeta => None,
    }
}

/// Holdout-derived coverage for early exits, in basis points. Thresholds are
/// fitted on a calibration split and frozen before final testing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExitCalibration {
    pub layer2_coverage_bps: u16,
    pub layer4_coverage_bps: u16,
    pub minimum_coverage_bps: u16,
}

/// Aegis v3 refuses an otherwise cheap exit when its calibration certificate
/// misses the configured coverage floor. Missing/impossible basis-point values
/// therefore escalate compute rather than silently trusting confidence.
pub fn choose_calibrated_chessformer_exit(
    backend: HybridBackend,
    uncertainty: f32,
    in_check: bool,
    remaining_ms: u64,
    calibration: ExitCalibration,
) -> Option<ChessformerExit> {
    let selected = choose_chessformer_exit(backend, uncertainty, in_check, remaining_ms)?;
    let minimum = calibration.minimum_coverage_bps.min(10_000);
    Some(match selected {
        ChessformerExit::Layer2Width128
            if calibration.layer2_coverage_bps.min(10_000) < minimum =>
        {
            if calibration.layer4_coverage_bps.min(10_000) >= minimum {
                ChessformerExit::Layer4Width192
            } else {
                ChessformerExit::Layer8Width256
            }
        }
        ChessformerExit::Layer4Width192
            if calibration.layer4_coverage_bps.min(10_000) < minimum =>
        {
            ChessformerExit::Layer8Width256
        }
        exit => exit,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicySafetyDecision {
    AcceptPolicyMove,
    UseAlphaBetaMove,
}

/// Final guard for direct human-policy play. Alpha-beta remains authoritative
/// for forced-mate status and rejects policy candidates whose common-budget
/// score loss exceeds the persona's explicit allowance.
pub fn policy_safety_veto(
    best_is_forced_mate: bool,
    candidate_allows_forced_mate: bool,
    best_score_cp: i32,
    candidate_score_cp: i32,
    maximum_loss_cp: i32,
) -> PolicySafetyDecision {
    if best_is_forced_mate
        || candidate_allows_forced_mate
        || best_score_cp.saturating_sub(candidate_score_cp) > maximum_loss_cp.max(0)
    {
        PolicySafetyDecision::UseAlphaBetaMove
    } else {
        PolicySafetyDecision::AcceptPolicyMove
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen;

    #[test]
    fn black_to_move_is_normalized_as_the_mover() {
        let pos = fen::parse("4k3/p7/8/8/8/8/7P/4K3 b - - 3 20").unwrap();
        let input = encode_position(&pos, 1500);
        assert!(input.flipped);
        // Black pawn a7 is mover-owned and rank-flips to a2.
        assert_eq!(input.tokens[8].piece, 1 + PAWN as u8);
        // White pawn h2 becomes the opponent pawn on h7.
        assert_eq!(input.tokens[55].piece, 7 + PAWN as u8);
    }

    #[test]
    fn geometry_captures_dynamic_piece_attacks() {
        let pos = fen::parse("4k3/8/8/3p4/2B5/8/8/4K3 w - - 0 1").unwrap();
        let relation = geometric_relation(&pos, 26, 35); // c4 -> d5
        assert_eq!((relation.file_delta, relation.rank_delta), (1, 1));
        assert!(relation.has(GeometryRelation::SAME_DIAGONAL));
        assert!(relation.has(GeometryRelation::SOURCE_ATTACKS_TARGET));
        assert!(relation.has(GeometryRelation::TARGET_OCCUPIED));
        assert!(!relation.has(GeometryRelation::SAME_OWNER));
    }

    #[test]
    fn v4_policy_actions_are_legal_sorted_and_promotion_complete() {
        let start = fen::startpos();
        let actions = legal_policy_actions(&start);
        assert_eq!(actions.len(), 20);
        assert!(!actions.overflowed());
        assert!(actions
            .as_slice()
            .windows(2)
            .all(|pair| pair[0].action < pair[1].action));

        let promotion = fen::parse("7k/P7/8/8/8/8/8/7K w - - 0 1").unwrap();
        let actions = legal_policy_actions(&promotion);
        let from_to = sq(0, 6) as u16 | ((sq(0, 7) as u16) << 6);
        for class in 1..=4u16 {
            let action = from_to | (class << 12);
            let mv = actions
                .find_action(action)
                .expect("missing promotion class");
            assert!(mv.is_promo());
            assert_eq!(mv.promo_piece() as u16, class);
        }
    }

    #[test]
    fn v4_candidate_set_orders_without_pruning_omitted_moves() {
        let pos = fen::startpos();
        let legal = legal_policy_actions(&pos);
        let candidates: Vec<_> = legal
            .as_slice()
            .iter()
            .enumerate()
            .map(|(index, &entry)| PolicyCandidateScore {
                entry,
                logit: -(index as f32),
                regret_upper_cp: if index < 3 { 20.0 } else { 200.0 },
            })
            .collect();
        let plan = build_policy_search_plan(&candidates, 50.0, 8, 9_970, 9_950).unwrap();
        assert_eq!(plan.ordered().len(), legal.len());
        assert_eq!(plan.priority().len(), 3);
        assert!(plan.certificate_valid);
        assert!(plan.full_legal_fallback_required);
        let planned_actions: std::collections::HashSet<_> = plan
            .ordered()
            .iter()
            .map(|candidate| candidate.entry.action)
            .collect();
        assert_eq!(planned_actions.len(), legal.len());
    }

    #[test]
    fn router_never_uses_transformer_for_engines_or_gms() {
        assert_eq!(
            choose_backend(true, true, false, 1800, Mode::Match, false, true),
            HybridBackend::AlphaBeta
        );
        assert_eq!(
            choose_backend(true, true, false, 2450, Mode::Match, true, false),
            HybridBackend::AlphaBeta
        );
    }

    #[test]
    fn router_uses_guarded_human_policy_only_in_safe_contexts() {
        assert_eq!(
            choose_backend(true, true, false, 1500, Mode::Match, true, false),
            HybridBackend::HumanPolicyGuarded
        );
        assert_eq!(
            choose_backend(true, true, false, 2200, Mode::Clinch, true, false),
            HybridBackend::PolicyGuidedAlphaBeta
        );
        assert_eq!(
            choose_backend(false, true, false, 1500, Mode::Match, true, false),
            HybridBackend::AlphaBeta
        );
    }

    #[test]
    fn evidential_uncertainty_distinguishes_no_evidence() {
        let (uniform, uncertainty) = evidential_wdl([0.0, 0.0, 0.0]);
        assert_eq!(uniform, [1.0 / 3.0; 3]);
        assert_eq!(uncertainty, 1.0);
        let (confident, uncertainty) = evidential_wdl([0.0, 2.0, 20.0]);
        assert!(confident[2] > 0.8);
        assert!(uncertainty <= 0.120_001);
    }

    #[test]
    fn elastic_exit_is_safe_and_backend_aware() {
        assert_eq!(
            choose_chessformer_exit(HybridBackend::AlphaBeta, 0.01, false, 500),
            None
        );
        assert_eq!(
            choose_chessformer_exit(HybridBackend::HumanPolicyGuarded, 0.05, false, 500),
            Some(ChessformerExit::Layer2Width128)
        );
        assert_eq!(
            choose_chessformer_exit(HybridBackend::HumanPolicyGuarded, 0.30, false, 5_000),
            Some(ChessformerExit::Layer8Width256)
        );
        assert_eq!(
            choose_chessformer_exit(HybridBackend::PolicyGuidedAlphaBeta, 0.01, true, 5_000),
            Some(ChessformerExit::Layer8Width256)
        );
    }

    #[test]
    fn v3_history_is_private_to_the_policy_cache() {
        let pos = fen::startpos();
        let model_uuid = [7u8; AEGIS_MODEL_UUID_BYTES];
        let first = Move::new(sq(4, 1), sq(4, 3), MK_NORMAL);
        let second = Move::new(sq(3, 1), sq(3, 3), MK_NORMAL);
        let context_a = TemporalPolicyContext::new(&[first], Color::White, 2, 1500);
        let context_b = TemporalPolicyContext::new(&[second], Color::White, 2, 1500);
        assert_eq!(
            board_value_cache_key(&pos, model_uuid),
            board_value_cache_key(&pos, model_uuid)
        );
        assert_ne!(
            policy_cache_key(&pos, model_uuid, context_a, 1),
            policy_cache_key(&pos, model_uuid, context_b, 1)
        );
        let board_input = encode_board_state_v3(&pos);
        assert!(board_input
            .tokens
            .iter()
            .all(|token| token.elo_context == 0));
    }

    #[test]
    fn calibration_failure_escalates_but_never_downgrades() {
        let failed = ExitCalibration {
            layer2_coverage_bps: 9_700,
            layer4_coverage_bps: 9_850,
            minimum_coverage_bps: 9_900,
        };
        assert_eq!(
            choose_calibrated_chessformer_exit(
                HybridBackend::HumanPolicyGuarded,
                0.05,
                false,
                500,
                failed,
            ),
            Some(ChessformerExit::Layer8Width256)
        );
        let passed = ExitCalibration {
            layer2_coverage_bps: 9_950,
            layer4_coverage_bps: 9_950,
            minimum_coverage_bps: 9_900,
        };
        assert_eq!(
            choose_calibrated_chessformer_exit(
                HybridBackend::HumanPolicyGuarded,
                0.05,
                false,
                500,
                passed,
            ),
            Some(ChessformerExit::Layer2Width128)
        );
    }

    #[test]
    fn alpha_beta_veto_is_authoritative() {
        assert_eq!(
            policy_safety_veto(false, false, 80, 20, 75),
            PolicySafetyDecision::AcceptPolicyMove
        );
        assert_eq!(
            policy_safety_veto(false, false, 80, -20, 75),
            PolicySafetyDecision::UseAlphaBetaMove
        );
        assert_eq!(
            policy_safety_veto(false, true, 0, 0, 10_000),
            PolicySafetyDecision::UseAlphaBetaMove
        );
    }
}
