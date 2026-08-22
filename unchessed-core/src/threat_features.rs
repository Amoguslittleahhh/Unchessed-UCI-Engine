//! Compact, project-native threat-relation features for the next NNUE format.
//!
//! Stockfish SFNNv10 demonstrated that attacked-piece relations can improve a
//! high-end NNUE. Copying its 79,856 x 1,024 transformer would be a poor fit
//! for Unchessed's latency and model-size budget. This module instead defines
//! a 32,400-dimensional multiset of relative threat relations suitable for a
//! low-rank 32-wide residual accumulator:
//!
//!   attacker class (12) x target class (12) x relative delta (15 x 15)
//!
//! Classes are perspective-relative `(piece_type, own/opponent)`. Squares use
//! the same vertical and king-file orientation as HalfKAv2_hm. Repeated indices
//! are intentional: two geometrically identical relations count twice. At a
//! 32-wide int16 embedding this table occupies ~2.0 MiB and a full refresh is
//! at most 256 x 32 additions, small enough to benchmark before implementing
//! complex dirty-ray updates. It is research infrastructure only; the shipped
//! v3 evaluator does not consume these features yet.

use crate::board::*;
use crate::movegen::{bishop_att, queen_att, rook_att, KING_ATT, KNIGHT_ATT, PAWN_ATT};

pub const THREAT_CLASSES: usize = 12;
pub const THREAT_RELATIONS: usize = 15 * 15;
pub const THREAT_DIMENSIONS: usize = THREAT_CLASSES * THREAT_CLASSES * THREAT_RELATIONS;
pub const THREAT_LATENT_WIDTH: usize = 32;
pub const MAX_ACTIVE_THREATS: usize = 256;

/// Aegis v3 x-ray triples use perspective-relative piece classes and one of
/// eight canonical ray directions.
pub const XRAY_DIRECTIONS: usize = 8;
pub const XRAY_DIMENSIONS: usize =
    THREAT_CLASSES * THREAT_CLASSES * THREAT_CLASSES * XRAY_DIRECTIONS;
pub const XRAY_LATENT_WIDTH: usize = 16;
pub const MAX_ACTIVE_XRAYS: usize = 256;

/// Hashed local pawn/king topology rows. Collisions are part of the feature
/// definition and must be identical in data generation, training, and runtime.
pub const PAWN_TOPOLOGY_DIMENSIONS: usize = 4096;
pub const PAWN_TOPOLOGY_WIDTH: usize = 16;
pub const MAX_ACTIVE_TOPOLOGIES: usize = 34; // 32 legal pawns plus two kings

/// Adaptive XT-NNUE compute tier. V2 used only the fast/full endpoints. Aegis
/// v3 inserts a direct-relation tier so ordinary tactical structure can be
/// evaluated without refreshing x-ray and pawn topology accumulators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualTier {
    FastPositionOnly,
    DirectRelations,
    FullHypergraph,
}

pub fn choose_residual_tier(
    is_root: bool,
    is_pv: bool,
    in_check: bool,
    fast_uncertainty: f32,
    threshold: f32,
) -> ResidualTier {
    if is_root || is_pv || in_check || fast_uncertainty.clamp(0.0, 1.0) > threshold {
        ResidualTier::FullHypergraph
    } else {
        ResidualTier::FastPositionOnly
    }
}

/// Inputs to the three-stage Aegis v3 router. `direct_uncertainty` is `None`
/// before the direct stage has run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualRouteContext {
    pub is_root: bool,
    pub is_pv: bool,
    pub in_check: bool,
    pub tactical_pressure: bool,
    pub fast_uncertainty: f32,
    pub direct_uncertainty: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualThresholds {
    pub fast_to_direct: f32,
    pub direct_to_full: f32,
}

/// Hard tactical contexts always take the full path. Otherwise the
/// position-only stage may terminate at low uncertainty; a direct evaluation
/// may terminate only after its own calibrated uncertainty passes. Non-finite
/// uncertainty is treated as maximally unsafe.
pub fn choose_residual_tier_v3(
    context: ResidualRouteContext,
    thresholds: ResidualThresholds,
) -> ResidualTier {
    if context.is_root || context.is_pv || context.in_check || context.tactical_pressure {
        return ResidualTier::FullHypergraph;
    }
    let fast = if context.fast_uncertainty.is_finite() {
        context.fast_uncertainty.clamp(0.0, 1.0)
    } else {
        1.0
    };
    if fast <= thresholds.fast_to_direct.clamp(0.0, 1.0) {
        return ResidualTier::FastPositionOnly;
    }
    let Some(direct) = context.direct_uncertainty else {
        return ResidualTier::DirectRelations;
    };
    let direct = if direct.is_finite() {
        direct.clamp(0.0, 1.0)
    } else {
        1.0
    };
    if direct <= thresholds.direct_to_full.clamp(0.0, 1.0) {
        ResidualTier::DirectRelations
    } else {
        ResidualTier::FullHypergraph
    }
}

/// Calibrated score interval used by uncertainty-aware pruning. The quantiles
/// are fitted on a calibration split for a material-phase/depth bucket; they
/// are not learned on, or tuned against, the final test holdout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConformalBounds {
    pub lower_cp: f32,
    pub upper_cp: f32,
}

pub fn conformal_score_bounds(
    mean_cp: f32,
    aleatoric_sigma_cp: f32,
    ensemble_means_cp: [f32; 2],
    lower_quantile: f32,
    upper_quantile: f32,
) -> ConformalBounds {
    if !mean_cp.is_finite()
        || !aleatoric_sigma_cp.is_finite()
        || ensemble_means_cp.iter().any(|value| !value.is_finite())
        || !lower_quantile.is_finite()
        || !upper_quantile.is_finite()
    {
        return ConformalBounds {
            lower_cp: f32::NEG_INFINITY,
            upper_cp: f32::INFINITY,
        };
    }
    let disagreement = (ensemble_means_cp[0] - ensemble_means_cp[1]).abs() * 0.5;
    let scale = aleatoric_sigma_cp.max(0.0).hypot(disagreement).max(1.0);
    ConformalBounds {
        lower_cp: mean_cp - lower_quantile.max(0.0) * scale,
        upper_cp: mean_cp + upper_quantile.max(0.0) * scale,
    }
}

/// Fail-high is safe only when the calibrated lower bound clears beta.
pub fn conformal_fail_high(bounds: ConformalBounds, beta_cp: f32, margin_cp: f32) -> bool {
    bounds.lower_cp.is_finite() && bounds.lower_cp >= beta_cp + margin_cp.max(0.0)
}

/// Fail-low is safe only when the calibrated upper bound clears alpha.
pub fn conformal_fail_low(bounds: ConformalBounds, alpha_cp: f32, margin_cp: f32) -> bool {
    bounds.upper_cp.is_finite() && bounds.upper_cp <= alpha_cp - margin_cp.max(0.0)
}

#[derive(Clone)]
pub struct ThreatFeatureList {
    indices: [u16; MAX_ACTIVE_THREATS],
    len: usize,
    overflowed: bool,
}

impl Default for ThreatFeatureList {
    fn default() -> Self {
        Self {
            indices: [0; MAX_ACTIVE_THREATS],
            len: 0,
            overflowed: false,
        }
    }
}

impl ThreatFeatureList {
    pub fn as_slice(&self) -> &[u16] {
        &self.indices[..self.len]
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

    fn push(&mut self, index: usize) {
        debug_assert!(index < THREAT_DIMENSIONS);
        if self.len == MAX_ACTIVE_THREATS {
            self.overflowed = true;
            return;
        }
        self.indices[self.len] = index as u16;
        self.len += 1;
    }
}

#[inline]
pub fn attacks_from(pos: &Position, color: Color, piece: usize, square: usize) -> Bitboard {
    match piece {
        PAWN => PAWN_ATT[color.idx()][square],
        KNIGHT => KNIGHT_ATT[square],
        BISHOP => bishop_att(square as u8, pos.occ),
        ROOK => rook_att(square as u8, pos.occ),
        QUEEN => queen_att(square as u8, pos.occ),
        KING => KING_ATT[square],
        _ => 0,
    }
}

/// HalfKAv2_hm-compatible square orientation for one perspective.
fn orientation(pos: &Position, perspective: Color) -> impl Fn(usize) -> usize {
    let white = perspective == Color::White;
    let king_raw = pos.king_sq(perspective) as usize;
    let king_vertical = if white { king_raw } else { king_raw ^ 56 };
    let mirror_file = king_vertical % 8 < 4;
    move |square: usize| {
        let vertical = if white { square } else { square ^ 56 };
        if mirror_file {
            vertical ^ 7
        } else {
            vertical
        }
    }
}

#[inline]
fn piece_class(piece: usize, color: Color, perspective: Color) -> usize {
    piece * 2 + usize::from(color != perspective)
}

#[inline]
fn relation_index(
    attacker_class: usize,
    target_class: usize,
    attacker_square: usize,
    target_square: usize,
) -> usize {
    let af = (attacker_square & 7) as i32;
    let ar = (attacker_square >> 3) as i32;
    let tf = (target_square & 7) as i32;
    let tr = (target_square >> 3) as i32;
    let delta_file = (tf - af + 7) as usize;
    let delta_rank = (tr - ar + 7) as usize;
    let relation = delta_rank * 15 + delta_file;
    (attacker_class * THREAT_CLASSES + target_class) * THREAT_RELATIONS + relation
}

/// Enumerate attacks and defenses whose target square is occupied.
///
/// The returned list is a multiset. Callers must reject `overflowed()` rather
/// than silently training/inferencing on truncated features. No legal position
/// in the regression corpus reaches the 256-relation bound.
pub fn active_threat_features(pos: &Position, perspective: Color) -> ThreatFeatureList {
    let orient = orientation(pos, perspective);
    let mut output = ThreatFeatureList::default();

    for attacker_color in [Color::White, Color::Black] {
        for attacker_piece in 0..6 {
            let mut attackers = pos.bb[attacker_color.idx()][attacker_piece];
            while attackers != 0 {
                let from = attackers.trailing_zeros() as usize;
                attackers &= attackers - 1;
                let mut targets = attacks_from(pos, attacker_color, attacker_piece, from) & pos.occ;
                while targets != 0 {
                    let to = targets.trailing_zeros() as usize;
                    targets &= targets - 1;
                    if let Some((target_color, target_piece)) = pos.piece_on(to as u8) {
                        let index = relation_index(
                            piece_class(attacker_piece, attacker_color, perspective),
                            piece_class(target_piece, target_color, perspective),
                            orient(from),
                            orient(to),
                        );
                        output.push(index);
                    }
                }
            }
        }
    }
    output
}

#[derive(Clone)]
pub struct XrayFeatureList {
    indices: [u16; MAX_ACTIVE_XRAYS],
    len: usize,
    overflowed: bool,
}

impl Default for XrayFeatureList {
    fn default() -> Self {
        Self {
            indices: [0; MAX_ACTIVE_XRAYS],
            len: 0,
            overflowed: false,
        }
    }
}

impl XrayFeatureList {
    pub fn as_slice(&self) -> &[u16] {
        &self.indices[..self.len]
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

    fn push(&mut self, index: usize) {
        debug_assert!(index < XRAY_DIMENSIONS);
        if self.len == MAX_ACTIVE_XRAYS {
            self.overflowed = true;
            return;
        }
        self.indices[self.len] = index as u16;
        self.len += 1;
    }
}

const RAY_DELTAS: [(i8, i8); XRAY_DIRECTIONS] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
];

#[inline]
fn slider_uses_direction(piece: usize, direction: usize) -> bool {
    match piece {
        BISHOP => !direction.is_multiple_of(2),
        ROOK => direction.is_multiple_of(2),
        QUEEN => true,
        _ => false,
    }
}

#[inline]
fn canonical_ray_direction(from: usize, to: usize) -> usize {
    let df = ((to & 7) as i8 - (from & 7) as i8).signum();
    let dr = ((to >> 3) as i8 - (from >> 3) as i8).signum();
    match (df, dr) {
        (0, 1) => 0,
        (1, 1) => 1,
        (1, 0) => 2,
        (1, -1) => 3,
        (0, -1) => 4,
        (-1, -1) => 5,
        (-1, 0) => 6,
        (-1, 1) => 7,
        _ => unreachable!("x-ray endpoints must be distinct and collinear"),
    }
}

/// Enumerate `(slider, first blocker, second occupied target, direction)`
/// hyperedges. The first blocker need not be an enemy: friendly blockers are
/// essential for batteries and discovered attacks. The second occupied square
/// terminates the ray, so the extractor is exact and never sees through three
/// pieces.
pub fn active_xray_features(pos: &Position, perspective: Color) -> XrayFeatureList {
    let orient = orientation(pos, perspective);
    let mut output = XrayFeatureList::default();
    for attacker_color in [Color::White, Color::Black] {
        for attacker_piece in [BISHOP, ROOK, QUEEN] {
            let mut attackers = pos.bb[attacker_color.idx()][attacker_piece];
            while attackers != 0 {
                let from = attackers.trailing_zeros() as usize;
                attackers &= attackers - 1;
                let source_file = (from & 7) as i8;
                let source_rank = (from >> 3) as i8;
                for (direction, &(df, dr)) in RAY_DELTAS.iter().enumerate() {
                    if !slider_uses_direction(attacker_piece, direction) {
                        continue;
                    }
                    let mut first: Option<(Color, usize)> = None;
                    let mut file = source_file + df;
                    let mut rank = source_rank + dr;
                    while (0..8).contains(&file) && (0..8).contains(&rank) {
                        let square = (rank as usize) * 8 + file as usize;
                        if let Some((color, piece)) = pos.piece_on(square as u8) {
                            if let Some((blocker_color, blocker_piece)) = first {
                                let canonical_direction =
                                    canonical_ray_direction(orient(from), orient(square));
                                let index =
                                    (((piece_class(attacker_piece, attacker_color, perspective)
                                        * THREAT_CLASSES
                                        + piece_class(blocker_piece, blocker_color, perspective))
                                        * THREAT_CLASSES
                                        + piece_class(piece, color, perspective))
                                        * XRAY_DIRECTIONS)
                                        + canonical_direction;
                                output.push(index);
                                break;
                            }
                            first = Some((color, piece));
                        }
                        file += df;
                        rank += dr;
                    }
                }
            }
        }
    }
    output
}

#[derive(Clone)]
pub struct PawnTopologyFeatureList {
    indices: [u16; MAX_ACTIVE_TOPOLOGIES],
    len: usize,
    overflowed: bool,
}

impl Default for PawnTopologyFeatureList {
    fn default() -> Self {
        Self {
            indices: [0; MAX_ACTIVE_TOPOLOGIES],
            len: 0,
            overflowed: false,
        }
    }
}

impl PawnTopologyFeatureList {
    pub fn as_slice(&self) -> &[u16] {
        &self.indices[..self.len]
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

    fn push(&mut self, index: usize) {
        debug_assert!(index < PAWN_TOPOLOGY_DIMENSIONS);
        if self.len == MAX_ACTIVE_TOPOLOGIES {
            self.overflowed = true;
            return;
        }
        self.indices[self.len] = index as u16;
        self.len += 1;
    }
}

#[inline]
fn orient_bitboard(mut bitboard: Bitboard, orient: &impl Fn(usize) -> usize) -> Bitboard {
    let mut output = 0u64;
    while bitboard != 0 {
        let square = bitboard.trailing_zeros() as usize;
        bitboard &= bitboard - 1;
        output |= 1u64 << orient(square);
    }
    output
}

/// Stable 32-bit integer finalizer shared with the v3 GPU trainer. Arithmetic
/// is explicitly wrapping and the low twelve bits select one of 4,096 rows.
pub fn pawn_topology_hash(mut key: u64) -> usize {
    key ^= key >> 16;
    key = key.wrapping_mul(0x045d_9f3b) & 0xffff_ffff;
    key ^= key >> 16;
    key = key.wrapping_mul(0x045d_9f3b) & 0xffff_ffff;
    key ^= key >> 16;
    key as usize & (PAWN_TOPOLOGY_DIMENSIONS - 1)
}

fn pawn_structure_flags(
    square: usize,
    forward: i8,
    friendly_pawns: Bitboard,
    enemy_pawns: Bitboard,
    occupied: Bitboard,
) -> u8 {
    let file = (square & 7) as i8;
    let rank = (square >> 3) as i8;
    let mut enemy_ahead = false;
    let mut adjacent_friend = false;
    let mut connected = false;
    let mut lever = false;
    let mut doubled = false;
    for other in 0..64usize {
        let bit = 1u64 << other;
        let other_file = (other & 7) as i8;
        let other_rank = (other >> 3) as i8;
        if enemy_pawns & bit != 0
            && (other_file - file).abs() <= 1
            && (other_rank - rank) * forward > 0
        {
            enemy_ahead = true;
        }
        if friendly_pawns & bit != 0 && other != square {
            if (other_file - file).abs() == 1 {
                adjacent_friend = true;
                if (other_rank - rank).abs() <= 1 {
                    connected = true;
                }
            }
            if other_file == file {
                doubled = true;
            }
        }
    }
    let next_rank = rank + forward;
    if (0..8).contains(&next_rank) {
        for next_file in [file - 1, file + 1] {
            if (0..8).contains(&next_file)
                && enemy_pawns & (1u64 << (next_rank as usize * 8 + next_file as usize)) != 0
            {
                lever = true;
            }
        }
    }
    let blocked = (0..8).contains(&next_rank)
        && occupied & (1u64 << (next_rank as usize * 8 + file as usize)) != 0;
    u8::from(!enemy_ahead)
        | (u8::from(!adjacent_friend) << 1)
        | (u8::from(connected) << 2)
        | (u8::from(lever) << 3)
        | (u8::from(doubled) << 4)
        | (u8::from(blocked) << 5)
}

/// Hash a canonical 3-file by 4-rank pawn window around every pawn and king.
/// Bits 0..11 are perspective-owned pawns, bits 12..23 opponent pawns,
/// bits 24..25 identify own/enemy pawn/king anchors, and bits 26..31 carry
/// passed/isolated/connected/lever/doubled/blocked flags for pawn anchors.
pub fn active_pawn_topology_features(
    pos: &Position,
    perspective: Color,
) -> PawnTopologyFeatureList {
    let orient = orientation(pos, perspective);
    let own_pawns = orient_bitboard(pos.bb[perspective.idx()][PAWN], &orient);
    let enemy_pawns = orient_bitboard(pos.bb[perspective.flip().idx()][PAWN], &orient);
    let occupied = orient_bitboard(pos.occ, &orient);
    let mut output = PawnTopologyFeatureList::default();

    for color in [perspective, perspective.flip()] {
        for piece in [PAWN, KING] {
            let mut anchors = pos.bb[color.idx()][piece];
            while anchors != 0 {
                let original = anchors.trailing_zeros() as usize;
                anchors &= anchors - 1;
                let square = orient(original);
                let file = (square & 7) as i8;
                let rank = (square >> 3) as i8;
                let mut own_window = 0u64;
                let mut enemy_window = 0u64;
                let mut window_bit = 0usize;
                for rank_offset in [-1i8, 0, 1, 2] {
                    for file_offset in [-1i8, 0, 1] {
                        let wf = file + file_offset;
                        let wr = rank + rank_offset;
                        if (0..8).contains(&wf) && (0..8).contains(&wr) {
                            let bit = 1u64 << (wr as usize * 8 + wf as usize);
                            if own_pawns & bit != 0 {
                                own_window |= 1u64 << window_bit;
                            }
                            if enemy_pawns & bit != 0 {
                                enemy_window |= 1u64 << window_bit;
                            }
                        }
                        window_bit += 1;
                    }
                }
                let anchor_class = match (piece, color == perspective) {
                    (PAWN, true) => 0u64,
                    (PAWN, false) => 1u64,
                    (KING, true) => 2u64,
                    (KING, false) => 3u64,
                    _ => unreachable!(),
                };
                let flags = if piece == PAWN {
                    let own_anchor = color == perspective;
                    pawn_structure_flags(
                        square,
                        if own_anchor { 1 } else { -1 },
                        if own_anchor { own_pawns } else { enemy_pawns },
                        if own_anchor { enemy_pawns } else { own_pawns },
                        occupied,
                    )
                } else {
                    0
                };
                let key = own_window
                    | (enemy_window << 12)
                    | (anchor_class << 24)
                    | ((flags as u64) << 26);
                output.push(pawn_topology_hash(key));
            }
        }
    }
    output
}

/// Full relation state for one perspective. Aegis v4 uses this as the exact
/// oracle for validating future dirty-square kernels before they are allowed
/// into search.
#[derive(Clone)]
pub struct HypergraphSnapshot {
    pub direct: ThreatFeatureList,
    pub xray: XrayFeatureList,
    pub topology: PawnTopologyFeatureList,
}

pub fn hypergraph_snapshot(pos: &Position, perspective: Color) -> HypergraphSnapshot {
    HypergraphSnapshot {
        direct: active_threat_features(pos, perspective),
        xray: active_xray_features(pos, perspective),
        topology: active_pawn_topology_features(pos, perspective),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureDeltaList<const N: usize> {
    added: [u16; N],
    added_len: usize,
    removed: [u16; N],
    removed_len: usize,
    overflowed: bool,
}

impl<const N: usize> Default for FeatureDeltaList<N> {
    fn default() -> Self {
        Self {
            added: [0; N],
            added_len: 0,
            removed: [0; N],
            removed_len: 0,
            overflowed: false,
        }
    }
}

impl<const N: usize> FeatureDeltaList<N> {
    pub fn added(&self) -> &[u16] {
        &self.added[..self.added_len]
    }

    pub fn removed(&self) -> &[u16] {
        &self.removed[..self.removed_len]
    }

    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    fn push_added(&mut self, value: u16) {
        if self.added_len == N {
            self.overflowed = true;
        } else {
            self.added[self.added_len] = value;
            self.added_len += 1;
        }
    }

    fn push_removed(&mut self, value: u16) {
        if self.removed_len == N {
            self.overflowed = true;
        } else {
            self.removed[self.removed_len] = value;
            self.removed_len += 1;
        }
    }
}

fn exact_multiset_delta<const N: usize>(before: &[u16], after: &[u16]) -> FeatureDeltaList<N> {
    let mut output = FeatureDeltaList::default();
    if before.len() > N || after.len() > N {
        output.overflowed = true;
        return output;
    }
    let mut old = [0u16; N];
    let mut new = [0u16; N];
    old[..before.len()].copy_from_slice(before);
    new[..after.len()].copy_from_slice(after);
    old[..before.len()].sort_unstable();
    new[..after.len()].sort_unstable();
    let (mut i, mut j) = (0usize, 0usize);
    while i < before.len() || j < after.len() {
        if i == before.len() {
            output.push_added(new[j]);
            j += 1;
        } else if j == after.len() || old[i] < new[j] {
            output.push_removed(old[i]);
            i += 1;
        } else if new[j] < old[i] {
            output.push_added(new[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    output
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HypergraphDelta {
    pub direct: FeatureDeltaList<MAX_ACTIVE_THREATS>,
    pub xray: FeatureDeltaList<MAX_ACTIVE_XRAYS>,
    pub topology: FeatureDeltaList<MAX_ACTIVE_TOPOLOGIES>,
    pub requires_full_refresh: bool,
}

/// Exact, full-refresh-backed multiset delta. This is intentionally a
/// correctness oracle rather than the eventual optimized dirty closure. Any
/// bounded dirty updater must match this result over random legal trees before
/// replacing it.
pub fn exact_hypergraph_delta(
    before: &HypergraphSnapshot,
    after: &HypergraphSnapshot,
) -> HypergraphDelta {
    let direct = exact_multiset_delta(before.direct.as_slice(), after.direct.as_slice());
    let xray = exact_multiset_delta(before.xray.as_slice(), after.xray.as_slice());
    let topology = exact_multiset_delta(before.topology.as_slice(), after.topology.as_slice());
    let requires_full_refresh = before.direct.overflowed()
        || before.xray.overflowed()
        || before.topology.overflowed()
        || after.direct.overflowed()
        || after.xray.overflowed()
        || after.topology.overflowed()
        || direct.overflowed()
        || xray.overflowed()
        || topology.overflowed();
    HypergraphDelta {
        direct,
        xray,
        topology,
        requires_full_refresh,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen;

    #[test]
    fn threat_indices_are_bounded_and_stack_resident() {
        let pos = fen::startpos();
        for perspective in [Color::White, Color::Black] {
            let features = active_threat_features(&pos, perspective);
            assert!(!features.is_empty());
            assert!(!features.overflowed());
            assert!(features.len() <= MAX_ACTIVE_THREATS);
            assert!(features
                .as_slice()
                .iter()
                .all(|&index| (index as usize) < THREAT_DIMENSIONS));
        }
    }

    #[test]
    fn bishop_attack_relation_has_expected_relative_encoding() {
        let pos = fen::parse("4k3/8/8/3p4/2B5/8/8/4K3 w - - 0 1").unwrap();
        let features = active_threat_features(&pos, Color::White);
        // White bishop (class 4) on c4 attacks black pawn (class 1) on d5.
        // Relative delta (+1,+1) maps to row 8, column 8 in the 15x15 grid.
        let expected = ((4 * 12 + 1) * 225 + 8 * 15 + 8) as u16;
        assert!(features.as_slice().contains(&expected));
    }

    #[test]
    fn feature_multiset_is_deterministic() {
        let pos =
            fen::parse("r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1BN2/PPP2PPP/R2Q1RK1 b kq - 7 13").unwrap();
        let a = active_threat_features(&pos, Color::White);
        let b = active_threat_features(&pos, Color::White);
        assert_eq!(a.as_slice(), b.as_slice());
        assert!(!a.overflowed());
    }

    #[test]
    fn residual_tier_preserves_tactical_nodes() {
        assert_eq!(
            choose_residual_tier(true, false, false, 0.0, 0.18),
            ResidualTier::FullHypergraph
        );
        assert_eq!(
            choose_residual_tier(false, true, false, 0.0, 0.18),
            ResidualTier::FullHypergraph
        );
        assert_eq!(
            choose_residual_tier(false, false, true, 0.0, 0.18),
            ResidualTier::FullHypergraph
        );
        assert_eq!(
            choose_residual_tier(false, false, false, 0.25, 0.18),
            ResidualTier::FullHypergraph
        );
        assert_eq!(
            choose_residual_tier(false, false, false, 0.05, 0.18),
            ResidualTier::FastPositionOnly
        );
    }

    #[test]
    fn xray_triples_encode_slider_blocker_and_target() {
        let pos = fen::parse("4k3/8/8/4r3/4B3/8/4R3/4K3 w - - 0 1").unwrap();
        let features = active_xray_features(&pos, Color::White);
        // White rook e2 (class 6), white bishop e4 (class 4), black rook e5
        // (class 7), canonical north ray (0).
        let expected = ((((6 * 12 + 4) * 12 + 7) * 8) + 0) as u16;
        assert!(features.as_slice().contains(&expected));
        assert!(!features.overflowed());
        assert!(features
            .as_slice()
            .iter()
            .all(|&index| (index as usize) < XRAY_DIMENSIONS));
    }

    #[test]
    fn pawn_topology_is_bounded_deterministic_and_anchor_complete() {
        let pos = fen::startpos();
        let a = active_pawn_topology_features(&pos, Color::White);
        let b = active_pawn_topology_features(&pos, Color::White);
        assert_eq!(a.as_slice(), b.as_slice());
        assert_eq!(a.len(), 18); // sixteen pawns and two kings
        assert!(!a.overflowed());
        assert!(a
            .as_slice()
            .iter()
            .all(|&index| (index as usize) < PAWN_TOPOLOGY_DIMENSIONS));

        // White pawn d4 sees itself at window bit 4 and black pawn e5 at
        // enemy-window bit 8. It is isolated and has a lever (flags 0b001010).
        let lever = fen::parse("4k3/8/8/4p3/3P4/8/8/4K3 w - - 0 1").unwrap();
        let features = active_pawn_topology_features(&lever, Color::White);
        assert!(features.as_slice().contains(&1373));
    }

    fn apply_delta<const N: usize>(before: &[u16], delta: &FeatureDeltaList<N>) -> Vec<u16> {
        let mut values = before.to_vec();
        for removed in delta.removed() {
            let index = values
                .iter()
                .position(|value| value == removed)
                .expect("delta removed absent multiset member");
            values.swap_remove(index);
        }
        values.extend_from_slice(delta.added());
        values.sort_unstable();
        values
    }

    #[test]
    fn v4_reference_deltas_reconstruct_full_refresh_for_legal_moves() {
        let fens = [
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "7k/P7/8/8/8/8/8/7K w - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ];
        for fen_value in fens {
            let pos = fen::parse(fen_value).unwrap();
            for mv in crate::movegen::legal(&pos).as_slice().iter().copied() {
                let next = pos.make(mv);
                for perspective in [Color::White, Color::Black] {
                    let before = hypergraph_snapshot(&pos, perspective);
                    let after = hypergraph_snapshot(&next, perspective);
                    let delta = exact_hypergraph_delta(&before, &after);
                    assert!(!delta.requires_full_refresh, "{fen_value} {}", mv.uci());

                    let mut expected = after.direct.as_slice().to_vec();
                    expected.sort_unstable();
                    assert_eq!(
                        apply_delta(before.direct.as_slice(), &delta.direct),
                        expected
                    );
                    let mut expected = after.xray.as_slice().to_vec();
                    expected.sort_unstable();
                    assert_eq!(apply_delta(before.xray.as_slice(), &delta.xray), expected);
                    let mut expected = after.topology.as_slice().to_vec();
                    expected.sort_unstable();
                    assert_eq!(
                        apply_delta(before.topology.as_slice(), &delta.topology),
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn v3_router_escalates_one_stage_at_a_time_and_fails_safe() {
        let thresholds = ResidualThresholds {
            fast_to_direct: 0.10,
            direct_to_full: 0.18,
        };
        let route = |tactical_pressure, fast_uncertainty, direct_uncertainty| {
            choose_residual_tier_v3(
                ResidualRouteContext {
                    is_root: false,
                    is_pv: false,
                    in_check: false,
                    tactical_pressure,
                    fast_uncertainty,
                    direct_uncertainty,
                },
                thresholds,
            )
        };
        assert_eq!(route(false, 0.05, None), ResidualTier::FastPositionOnly);
        assert_eq!(route(false, 0.14, None), ResidualTier::DirectRelations);
        assert_eq!(
            route(false, 0.14, Some(0.12)),
            ResidualTier::DirectRelations
        );
        assert_eq!(
            route(false, f32::NAN, Some(f32::NAN)),
            ResidualTier::FullHypergraph
        );
        assert_eq!(route(true, 0.0, None), ResidualTier::FullHypergraph);
    }

    #[test]
    fn conformal_pruning_uses_the_conservative_tail() {
        let bounds = conformal_score_bounds(120.0, 10.0, [118.0, 122.0], 2.0, 3.0);
        assert!(bounds.lower_cp < 100.0);
        assert!(bounds.upper_cp > 150.0);
        assert!(!conformal_fail_high(bounds, 110.0, 0.0));
        assert!(!conformal_fail_low(bounds, 100.0, 0.0));

        let invalid = conformal_score_bounds(f32::NAN, 1.0, [0.0, 0.0], 2.0, 2.0);
        assert!(!conformal_fail_high(invalid, -10_000.0, 0.0));
        assert!(!conformal_fail_low(invalid, 10_000.0, 0.0));
    }

    #[test]
    #[ignore = "microbenchmark; run explicitly in release mode"]
    fn aegis_v3_feature_microbench() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r2q1rk1/pp2bppp/2npbn2/2p1p3/4P3/2NP1N2/PPPQBPPP/R1B2RK1 w - - 4 9",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "8/5pk1/4p1p1/3pP2p/3P3P/5KP1/5P2/4R3 w - - 0 40",
            "3r2k1/pp3ppp/2p1b3/4P3/1q1P1Q2/2R5/PP3PPP/4R1K1 w - - 0 22",
        ];
        let positions: Vec<_> = fens
            .iter()
            .map(|value| fen::parse(value).unwrap())
            .collect();
        let direct_weights: Vec<i16> = (0..THREAT_DIMENSIONS * THREAT_LATENT_WIDTH)
            .map(|index| ((index.wrapping_mul(17) % 31) as i16) - 15)
            .collect();
        let xray_weights: Vec<i16> = (0..XRAY_DIMENSIONS * XRAY_LATENT_WIDTH)
            .map(|index| ((index.wrapping_mul(13) % 23) as i16) - 11)
            .collect();
        let pawn_weights: Vec<i16> = (0..PAWN_TOPOLOGY_DIMENSIONS * PAWN_TOPOLOGY_WIDTH)
            .map(|index| ((index.wrapping_mul(7) % 19) as i16) - 9)
            .collect();
        let mut relation_counts = [0usize; 3];
        for pos in &positions {
            for perspective in [Color::White, Color::Black] {
                relation_counts[0] += active_threat_features(pos, perspective).len();
                relation_counts[1] += active_xray_features(pos, perspective).len();
                relation_counts[2] += active_pawn_topology_features(pos, perspective).len();
            }
        }
        let iterations = 10_000usize;
        let calls = iterations * positions.len() * 2;
        let started = std::time::Instant::now();
        let mut checksum = 0i16;
        for _ in 0..iterations {
            for pos in &positions {
                for perspective in [Color::White, Color::Black] {
                    let direct = active_threat_features(pos, perspective);
                    let xray = active_xray_features(pos, perspective);
                    let pawn = active_pawn_topology_features(pos, perspective);
                    assert!(!direct.overflowed() && !xray.overflowed() && !pawn.overflowed());
                    let mut direct_acc = [0i16; THREAT_LATENT_WIDTH];
                    let mut xray_acc = [0i16; XRAY_LATENT_WIDTH];
                    let mut pawn_acc = [0i16; PAWN_TOPOLOGY_WIDTH];
                    for &index in direct.as_slice() {
                        let row = &direct_weights[index as usize * THREAT_LATENT_WIDTH
                            ..(index as usize + 1) * THREAT_LATENT_WIDTH];
                        for (value, &weight) in direct_acc.iter_mut().zip(row) {
                            *value = value.wrapping_add(weight);
                        }
                    }
                    for &index in xray.as_slice() {
                        let row = &xray_weights[index as usize * XRAY_LATENT_WIDTH
                            ..(index as usize + 1) * XRAY_LATENT_WIDTH];
                        for (value, &weight) in xray_acc.iter_mut().zip(row) {
                            *value = value.wrapping_add(weight);
                        }
                    }
                    for &index in pawn.as_slice() {
                        let row = &pawn_weights[index as usize * PAWN_TOPOLOGY_WIDTH
                            ..(index as usize + 1) * PAWN_TOPOLOGY_WIDTH];
                        for (value, &weight) in pawn_acc.iter_mut().zip(row) {
                            *value = value.wrapping_add(weight);
                        }
                    }
                    checksum = checksum
                        .wrapping_add(std::hint::black_box(direct_acc)[0])
                        .wrapping_add(std::hint::black_box(xray_acc)[0])
                        .wrapping_add(std::hint::black_box(pawn_acc)[0]);
                }
            }
        }
        println!(
            "Aegis v3 full feature refresh + synthetic accumulation: {:.1} ns/call; mean active direct/xray/topology {:.1}/{:.1}/{:.1}; checksum {checksum}",
            started.elapsed().as_nanos() as f64 / calls as f64,
            relation_counts[0] as f64 / (positions.len() * 2) as f64,
            relation_counts[1] as f64 / (positions.len() * 2) as f64,
            relation_counts[2] as f64 / (positions.len() * 2) as f64,
        );
    }

    #[test]
    #[ignore = "microbenchmark; run explicitly in release mode"]
    fn threat_feature_microbench() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r2q1rk1/pp2bppp/2npbn2/2p1p3/4P3/2NP1N2/PPPQBPPP/R1B2RK1 w - - 4 9",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "r1bq1rk1/pp2bppp/2np1n2/2p1p3/4P3/2PP1N2/PP1N1PPP/R1BQR1K1 w - - 4 9",
            "3r2k1/pp3ppp/2p1b3/4P3/1q1P1Q2/2R5/PP3PPP/4R1K1 w - - 0 22",
            "8/5pk1/4p1p1/3pP2p/3P3P/5KP1/5P2/4R3 w - - 0 40",
            "8/5pk1/4p1p1/3pP2p/3P3P/5KP1/5P2/8 w - - 0 40",
            "r3k2r/ppp2ppp/2n1bn2/3pp3/8/2N1PN2/PPPPBPPP/R3K2R w KQkq - 6 9",
            "r1bq1rk1/ppp2ppp/2np1n2/4p1B1/2B1P3/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 9",
        ];
        let positions: Vec<_> = fens
            .iter()
            .enumerate()
            .map(|(index, value)| {
                fen::parse(value).unwrap_or_else(|error| panic!("FEN {index}: {error}"))
            })
            .collect();
        let iterations = 20_000usize;
        let mut relations = 0usize;
        for pos in &positions {
            for perspective in [Color::White, Color::Black] {
                let features = active_threat_features(pos, perspective);
                assert!(!features.overflowed());
                relations += features.len();
            }
        }
        let calls = iterations * positions.len() * 2;

        let weights: Vec<i16> = (0..THREAT_DIMENSIONS * THREAT_LATENT_WIDTH)
            .map(|index| ((index.wrapping_mul(17) % 31) as i16) - 15)
            .collect();
        let fused_started = std::time::Instant::now();
        let mut checksum = 0i16;
        for _ in 0..iterations {
            for pos in &positions {
                for perspective in [Color::White, Color::Black] {
                    let features = std::hint::black_box(active_threat_features(pos, perspective));
                    let mut accumulator = [0i16; THREAT_LATENT_WIDTH];
                    for &index in features.as_slice() {
                        let row = &weights[index as usize * THREAT_LATENT_WIDTH
                            ..(index as usize + 1) * THREAT_LATENT_WIDTH];
                        for (value, &weight) in accumulator.iter_mut().zip(row) {
                            *value = value.wrapping_add(weight);
                        }
                    }
                    let accumulator = std::hint::black_box(accumulator);
                    checksum = checksum.wrapping_add(accumulator[0]);
                }
            }
        }
        let fused_elapsed = fused_started.elapsed();
        println!(
            "threat residual prototype: {:.1} ns/call, {:.1} active relations/call ({checksum})",
            fused_elapsed.as_nanos() as f64 / calls as f64,
            relations as f64 / (positions.len() * 2) as f64,
        );
    }
}
