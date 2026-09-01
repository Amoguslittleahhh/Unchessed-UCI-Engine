//! The Adapter brain: live offline opponent-Elo estimation, the
//! MATCH / PUNISH / CLINCH / DEFEND persona state machine, and
//! human-plausible move selection over MultiPV candidates.

use crate::board::*;
use crate::movegen::{in_check, legal, KING_ATT};
use crate::search::Line;

// ---------------------------------------------------------------------------
// Tiny deterministic-seedable RNG (xorshift*)
// ---------------------------------------------------------------------------

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }
    pub fn from_time() -> Rng {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x1234_5678);
        Rng::new(t)
    }
    #[inline]
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// uniform in [0, 1)
    #[inline]
    pub fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// Known-engine table (seeds the model when UCI_Opponent identifies a computer)
// ---------------------------------------------------------------------------

const KNOWN_ENGINES: &[(&str, i32)] = &[
    ("stockfish", 3600),
    ("dragon", 3500),
    ("komodo", 3400),
    ("lc0", 3500),
    ("leela", 3500),
    ("berserk", 3400),
    ("ethereal", 3350),
    ("koivisto", 3300),
    ("rubichess", 3300),
    ("fairy", 3200),
    ("maia", 1600),
    ("torch", 3550),
    ("obsidian", 3400),
    ("alexandria", 3350),
];

// ---------------------------------------------------------------------------
// Live opponent model
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct OpponentModel {
    /// running Elo estimate
    mean: f64,
    /// evidence weight (grows with samples; prior counts as 1)
    weight: f64,
    pub samples: u32,
    pub last_cp_loss: Option<i32>,
    pub opponent_name: Option<String>,
    pub is_computer: bool,
    pub declared_elo: Option<i32>,
    /// depth into book before opponent's first deviation (a prep signal)
    pub book_depth_plies: u32,
    prev_mean: f64,
    /// exponentially-weighted variance of the per-move Elo samples: high =
    /// erratic play (brilliancies mixed with blunders — sandbagger pattern)
    var_accum: f64,
    /// grows when the opponent replies near-instantly with strong moves in
    /// non-trivial positions — the classic engine tell
    suspicion: f64,
    /// consecutive observations with cp-loss ≤ 40 (quality streak). Used
    /// for the ceiling tell so the climb from the 1500 prior does not
    /// inflate `var_accum` into a veto.
    low_loss_streak: u32,
}

impl Default for OpponentModel {
    fn default() -> Self {
        Self::new()
    }
}

impl OpponentModel {
    pub fn new() -> OpponentModel {
        OpponentModel {
            mean: 1500.0,
            weight: 1.0,
            samples: 0,
            last_cp_loss: None,
            opponent_name: None,
            is_computer: false,
            declared_elo: None,
            book_depth_plies: 0,
            prev_mean: 1500.0,
            var_accum: 90_000.0, // start wide (~300 sd)
            suspicion: 0.0,
            low_loss_streak: 0,
        }
    }

    /// Parse "setoption name UCI_Opponent value <title> <elo> <computer|human> <name...>".
    /// Returns a log line describing what was detected.
    pub fn seed_from_uci_opponent(&mut self, value: &str) -> String {
        let mut toks = value.split_whitespace();
        let _title = toks.next().unwrap_or("-");
        let elo_tok = toks.next().unwrap_or("-");
        let kind = toks.next().unwrap_or("human");
        let name: String = toks.collect::<Vec<_>>().join(" ");
        self.opponent_name = if name.is_empty() { None } else { Some(name.clone()) };
        self.is_computer = kind.eq_ignore_ascii_case("computer");
        self.declared_elo = elo_tok.parse::<i32>().ok().filter(|e| *e > 0);

        if self.is_computer {
            let lower = name.to_lowercase();
            let known = KNOWN_ENGINES
                .iter()
                .find(|(k, _)| lower.contains(k))
                .map(|&(k, e)| (k, e));
            let seed = self
                .declared_elo
                .or(known.map(|(_, e)| e))
                .unwrap_or(2800);
            self.mean = seed as f64;
            self.weight = 6.0; // strong prior for engines
            self.prev_mean = self.mean;
            match known {
                Some((k, e)) => format!(
                    "opponent: {} (computer, known engine '{}', seeded ~{})",
                    name, k, seed.max(e.min(seed))
                ),
                None => format!("opponent: {} (computer, seeded ~{})", name, seed),
            }
        } else {
            // Humans: declared ratings are ignored as a source of truth; the
            // live model judges pure skill from the moves. Declared Elo only
            // nudges the starting prior slightly.
            if let Some(e) = self.declared_elo {
                self.mean = 1500.0 * 0.5 + e as f64 * 0.5;
                self.prev_mean = self.mean;
            }
            format!(
                "opponent: {} (human) — rating will be estimated live from move quality",
                if name.is_empty() { "unknown" } else { &name }
            )
        }
    }

    /// cp-loss → Elo sample curve. Calibration constants are a declared
    /// tuning target for the training phase.
    fn elo_sample(cp_loss: f64) -> f64 {
        (2950.0 - 850.0 * (1.0 + cp_loss / 20.0).ln()).clamp(400.0, 3200.0)
    }

    /// Update with one observed opponent move.
    /// `difficulty_weight`: 1.0 normal, lower for forced/book/trivial moves.
    pub fn observe(&mut self, cp_loss: i32, difficulty_weight: f64) {
        self.prev_mean = self.mean;
        self.last_cp_loss = Some(cp_loss);
        let opening = if self.samples < 8 { 0.5 } else { 1.0 };
        let w = (difficulty_weight * opening).clamp(0.05, 2.0);
        let sample = Self::elo_sample(cp_loss as f64);
        let dev = sample - self.mean;
        self.var_accum = 0.88 * self.var_accum + 0.12 * dev * dev;
        self.mean = (self.mean * self.weight + sample * w) / (self.weight + w);
        self.weight = (self.weight + w).min(14.0);
        // slow decay keeps the model tracking (fatigue, sandbagging)
        self.weight *= 0.985;
        self.samples += 1;
        if cp_loss <= 40 {
            self.low_loss_streak = self.low_loss_streak.saturating_add(1);
        } else {
            self.low_loss_streak = 0;
        }
    }

    /// Feed the opponent's clock usage for their last move. Near-instant,
    /// near-perfect replies in positions with real choice are an engine tell.
    ///
    /// Opening/premove discount: the first few observed moves are often
    /// booked or pre-clicked even for 2000+ humans. Counting those as
    /// engine tells is the high-level misfire. Require `samples >= 8`
    /// (real middlegame evidence) before suspicion can grow.
    pub fn observe_time(&mut self, used_ms: u64, position_had_choice: bool) {
        let strong = matches!(self.last_cp_loss, Some(l) if l <= 60);
        let in_opening = self.samples < 8;
        if used_ms < 300 && strong && position_had_choice && !in_opening {
            self.suspicion += 1.0;
        } else if used_ms > 1500 {
            self.suspicion = (self.suspicion - 0.5).max(0.0);
        }
    }

    /// Engine detection without any GUI help, two signals:
    ///  - clock tell: repeated instant-strong replies in real positions;
    ///  - ceiling tell: our cp-loss yardstick (quick shallow analysis) cannot
    ///    measure above ~2700, so an estimate *pinned* near that ceiling with
    ///    low volatility over many moves is not a casual human. (And if it
    ///    somehow is a 2500+ human, giving them full strength is correct too.)
    ///
    /// Gated on accumulated evidence WEIGHT, not raw move count: `samples`
    /// increments once per move regardless of how forced/trivial it was,
    /// but `weight` is discounted by `difficulty_weight()` for near-forced
    /// positions. Threshold set high (close to weight's own 14.0 cap): a
    /// real low-rated player can absolutely play a clean 6-8 move stretch,
    /// since openings are the most memorized/intuitive part of chess even
    /// for weak players -- that's normal human variance, not an engine
    /// tell, and a short streak alone (whether gated by move count or by
    /// lightly-discounted weight) isn't enough to tell them apart. Only a
    /// genuinely long sustained run of low cp-loss should count. Calibrated
    /// against real chess.com games (100-1500 rated): weight>=6.0 still
    /// false-triggered on 3 of 7 real sub-1600 games by move 8-9; weight
    /// >=10.0 was chosen to require most of a full game's worth of
    /// evidence before concluding "too strong/consistent to be this
    /// opponent's declared level," while still comfortably tripping for a
    /// genuinely sustained 2500+ performance (own stress-test confirmed
    /// this separately).
    pub fn engine_suspect(&self) -> bool {
        // GUI-labelled computers: only *strong* engines force FULL.
        // Maia (table 1600) and other human-like engines are the MATCH
        // target of this project — treating them as Stockfish was the
        // ecosystem misfire (persona never ran vs the labelled opponent
        // we actually train to imitate).
        if self.is_computer {
            return self.mean >= 2400.0;
        }
        // Clock tell: 4 instant-strong middlegame replies. 3 was too
        // hungry in blitz vs 2200+ humans who just play fast.
        if self.suspicion >= 4.0 {
            return true;
        }
        // Ceiling tell: anonymous opponent pinned at the measurement
        // ceiling, *and* low volatility (a 2400 human still dribbles
        // 50–120 cp). Declared humans never take this path — a 2500
        // titled player is allowed to be strong; MATCH at target~2560
        // is already near-ceiling play without flipping to FULL.
        if self.declared_elo.is_some() {
            return false;
        }
        self.weight >= 11.0
            && self.samples >= 16
            && self.mean >= 2500.0
            && self.low_loss_streak >= 12
    }

    /// Spread of recent per-move Elo samples.
    pub fn volatility(&self) -> i32 {
        self.var_accum.sqrt().round() as i32
    }

    pub fn observe_book_move(&mut self, plies: u32) {
        self.book_depth_plies = self.book_depth_plies.max(plies);
        // theory carries little signal, but deep prep nudges the estimate up
        if plies >= 10 {
            self.observe(10, 0.25);
        }
    }

    pub fn estimate(&self) -> i32 {
        self.mean.round() as i32
    }

    /// rough +- confidence band; erratic opponents stay uncertain
    pub fn confidence(&self) -> i32 {
        let base = 600.0 / self.weight.sqrt();
        let volatility_widening = (self.var_accum.sqrt() / 400.0).clamp(0.6, 2.0);
        (base * volatility_widening).round() as i32
    }

    pub fn trend(&self) -> &'static str {
        if !self.is_computer && self.engine_suspect() {
            return if self.suspicion >= 4.0 {
                "instant strong replies — engine suspected"
            } else {
                "pinned at measurement ceiling — engine suspected"
            };
        }
        if self.volatility() > 380 && self.samples >= 6 {
            return "erratic (sandbagging?)";
        }
        let d = self.mean - self.prev_mean;
        if d > 25.0 {
            "trending up"
        } else if d < -25.0 {
            "trending down"
        } else {
            "steady"
        }
    }

    /// Did the last observed move look like a blunder?
    pub fn last_was_blunder(&self) -> bool {
        matches!(self.last_cp_loss, Some(l) if l >= 180)
    }
}

// ---------------------------------------------------------------------------
// Persona state machine
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// full strength, no adaptation (analysis, Adaptive=off, big-game)
    Full,
    /// blend to the opponent's level with human-plausible moves
    Match,
    /// opponent blundered or is far weaker: convert with forcing best moves
    Punish,
    /// close late game we want to win: venomous, trap-laden choices
    Clinch,
    /// we are clearly worse: maximum resistance
    Defend,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Full => "FULL",
            Mode::Match => "MATCH",
            Mode::Punish => "PUNISH",
            Mode::Clinch => "CLINCH",
            Mode::Defend => "DEFEND",
        }
    }
}

#[derive(Clone)]
pub struct AdaptConfig {
    pub adaptive: bool,
    pub limit_strength: bool,
    pub elo_cap: i32,
    pub contempt: i32,
}

impl Default for AdaptConfig {
    fn default() -> Self {
        AdaptConfig {
            adaptive: true,
            limit_strength: false,
            elo_cap: 2400,
            contempt: 25,
        }
    }
}

/// Nominal full playing strength of the engine itself (used to scale
/// selection temperature; not a claim, a tuning anchor).
pub const ENGINE_CEILING: i32 = 2600;

/// Decide the persona for this move, with hysteresis against `prev` so the
/// engine commits to a plan instead of flapping between personalities.
///
/// Enter/exit thresholds per mode:
///   DEFEND enters below -180 cp, exits above -80.
///   PUNISH enters on a fresh blunder (while better) or a huge skill gap with
///     a big lead; once punishing, it stays until the advantage is either
///     converted (still > +200) or fizzled.
///   CLINCH enters in drawish late middlegames (contempt > 0), and holds
///     while the game stays within +-100 cp.
/// Live persona with an EMA on search eval and a 2-ply dwell before
/// non-emergency switches. Raw `decide_mode` is kept for the threshold
/// contract and unit tests; the UCI worker uses this so noisy MultiPV
/// scores cannot flap MATCH/CLINCH/PUNISH every move.
///
/// Emergencies (no dwell): engine-suspect → FULL, eval EMA below −220 →
/// DEFEND, a fresh opponent blunder while we are better → PUNISH.
#[derive(Clone, Debug)]
pub struct PersonaState {
    pub mode: Mode,
    eval_ema: f64,
    ema_init: bool,
    dwell: u8,
    candidate: Mode,
}

impl Default for PersonaState {
    fn default() -> Self {
        PersonaState {
            mode: Mode::Match,
            eval_ema: 0.0,
            ema_init: false,
            dwell: 0,
            candidate: Mode::Match,
        }
    }
}

impl PersonaState {
    /// EMA coefficient: 0.35 on the new search score, 0.65 on history.
    /// Chosen so a one-move ±80 cp spike moves the filter ~28 cp — inside
    /// the CLINCH ±60 band's hysteresis, not across it.
    pub const ALPHA: f64 = 0.35;
    /// Consecutive agreeing votes required to leave the current mode,
    /// except emergencies.
    pub const DWELL: u8 = 2;
    pub const DEFEND_EMERGENCY: i32 = -220;

    pub fn smoothed_eval(&self) -> i32 {
        self.eval_ema.round() as i32
    }

    pub fn update(
        &mut self,
        cfg: &AdaptConfig,
        model: &OpponentModel,
        raw_eval_cp: i32,
        fullmove: u16,
    ) -> Mode {
        if !self.ema_init {
            self.eval_ema = raw_eval_cp as f64;
            self.ema_init = true;
        } else {
            self.eval_ema =
                Self::ALPHA * raw_eval_cp as f64 + (1.0 - Self::ALPHA) * self.eval_ema;
        }
        let smoothed = self.eval_ema.round() as i32;

        // Low confidence → require a clearer eval before leaving MATCH.
        // `confidence()` is a ± band; ~150 after a few moves, ~400 early.
        let pad = (model.confidence() / 20).clamp(0, 40);

        let emergency_defend = smoothed < Self::DEFEND_EMERGENCY;
        let emergency_punish = model.last_was_blunder() && smoothed > 60;
        let emergency_full = model.engine_suspect() && !cfg.limit_strength && cfg.adaptive;

        if emergency_full || emergency_defend || emergency_punish {
            let mode = if emergency_full {
                Mode::Full
            } else if emergency_defend {
                Mode::Defend
            } else {
                Mode::Punish
            };
            self.mode = mode;
            self.dwell = 0;
            self.candidate = mode;
            return mode;
        }

        let mut proposed = decide_mode(cfg, model, smoothed, fullmove, self.mode);
        // Widen the CLINCH deadband when we are unsure of the opponent:
        // accidental CLINCH is the noisiest switch (thin |eval|<60 window).
        if proposed == Mode::Clinch && self.mode != Mode::Clinch && smoothed.abs() + pad >= 60 {
            proposed = Mode::Match;
        }

        if proposed == self.mode {
            self.dwell = 0;
            self.candidate = proposed;
            return self.mode;
        }
        if proposed == self.candidate {
            self.dwell = self.dwell.saturating_add(1);
        } else {
            self.candidate = proposed;
            self.dwell = 1;
        }
        if self.dwell >= Self::DWELL {
            self.mode = proposed;
            self.dwell = 0;
        }
        self.mode
    }
}

pub fn decide_mode(
    cfg: &AdaptConfig,
    model: &OpponentModel,
    our_eval_cp: i32,
    fullmove: u16,
    prev: Mode,
) -> Mode {
    if !cfg.adaptive {
        // UCI semantics: with UCI_LimitStrength the engine plays AT UCI_Elo
        // even when adaptation is off.
        return if cfg.limit_strength { Mode::Match } else { Mode::Full };
    }
    // a suspected engine gets our best chess, not a blended-down imitation
    if model.engine_suspect() && !cfg.limit_strength {
        return Mode::Full;
    }

    if our_eval_cp < -180 || (prev == Mode::Defend && our_eval_cp < -80) {
        return Mode::Defend;
    }

    let target = target_elo(cfg, model);
    let punish_trigger = (model.last_was_blunder() && our_eval_cp > 60)
        || (target + 500 < ENGINE_CEILING.min(cfg_effective_cap(cfg)) && our_eval_cp > 250);
    if punish_trigger || (prev == Mode::Punish && our_eval_cp > 200) {
        return Mode::Punish;
    }

    let clinch_enter = fullmove > 28 && our_eval_cp.abs() < 60;
    let clinch_hold = prev == Mode::Clinch && our_eval_cp.abs() < 100;
    if cfg.contempt > 0 && (clinch_enter || clinch_hold) {
        return Mode::Clinch;
    }
    Mode::Match
}

fn cfg_effective_cap(cfg: &AdaptConfig) -> i32 {
    if cfg.limit_strength {
        cfg.elo_cap
    } else {
        ENGINE_CEILING
    }
}

/// The strength we aim to play at in MATCH mode.
pub fn target_elo(cfg: &AdaptConfig, model: &OpponentModel) -> i32 {
    if cfg.limit_strength && !cfg.adaptive {
        // pure fixed-strength play at the requested rating
        return cfg.elo_cap.max(500);
    }
    // aim a touch above the estimate: competitive but beatable
    let t = model.estimate() + 60;
    t.min(cfg_effective_cap(cfg)).max(500)
}

/// Draw score for the search (contempt wiring): when we are chasing a win a
/// draw is mildly bad for us; when we are defending, a draw is a rescue and
/// must not be repelled.
pub fn draw_score_for(cfg: &AdaptConfig, prev: Mode) -> i32 {
    if !cfg.adaptive || prev == Mode::Defend {
        return 0;
    }
    match prev {
        Mode::Clinch => -(cfg.contempt / 2).clamp(0, 50),
        _ => -(cfg.contempt / 3).clamp(0, 33),
    }
}

// ---------------------------------------------------------------------------
// Human-plausibility priors (heuristic MovePrior; a trained Maia-style
// policy net implements this same trait in the training phase)
// ---------------------------------------------------------------------------

pub trait MovePrior: Send + Sync {
    /// Relative plausibility weights for the candidate moves (need not be
    /// normalized; only ratios matter to the selector).
    fn priors(&self, pos: &Position, moves: &[Move], target_elo: i32) -> Vec<f64>;

    fn describe(&self) -> String;
}

pub struct HeuristicPrior;

impl MovePrior for HeuristicPrior {
    fn priors(&self, pos: &Position, moves: &[Move], target_elo: i32) -> Vec<f64> {
        moves
            .iter()
            .map(|&m| self.single(pos, m, target_elo))
            .collect()
    }

    fn describe(&self) -> String {
        "heuristic move priors".to_string()
    }
}

/// Maia-style trained human policy net (per-rating buckets).
pub struct MaiaPrior(pub std::sync::Arc<crate::policy::PolicyNet>);

impl MovePrior for MaiaPrior {
    fn priors(&self, pos: &Position, moves: &[Move], target_elo: i32) -> Vec<f64> {
        // small floor keeps eval-good moves alive even when the net hates them
        self.0
            .priors(pos, moves, target_elo)
            .into_iter()
            .map(|p| p + 0.005)
            .collect()
    }

    fn describe(&self) -> String {
        format!("human policy net ({})", self.0.describe())
    }
}

impl HeuristicPrior {
    fn single(&self, pos: &Position, mv: Move, target_elo: i32) -> f64 {
        // how strongly human habits shape choice: strong at low Elo, fades out
        let strength = ((2200 - target_elo).max(0) as f64 / 1700.0).clamp(0.0, 1.0);
        if strength == 0.0 {
            return 1.0;
        }
        let mut w: f64 = 1.0;
        let from = mv.from();
        let to = mv.to();
        let (us, piece) = match pos.piece_on(from) {
            Some(x) => x,
            None => return 1.0,
        };
        let opening = pos.fullmove <= 12;
        let is_capture = pos.board[to as usize] != NO_PIECE || mv.kind() == MK_EP;
        let next = pos.make(mv);
        let gives_check = in_check(&next);

        if is_capture {
            w *= 1.35; // humans love captures
        }
        if gives_check {
            w *= 1.25; // ...and checks
        }
        if mv.kind() == MK_CASTLE {
            w *= 1.6;
        }
        if mv.is_promo() {
            w *= if mv.promo_piece() == QUEEN { 1.4 } else { 0.5 };
        }
        if opening {
            let home_rank = if let Color::White = us { 0u8 } else { 7u8 };
            if (piece == KNIGHT || piece == BISHOP) && rank_of(from) == home_rank {
                w *= 1.25; // development
            }
            if piece == QUEEN {
                w *= 0.85; // early queen wandering (a bit) discouraged
            }
            if piece == KING && mv.kind() != MK_CASTLE {
                w *= 0.45; // manual king walks look weird
            }
        }
        // pushing edge pawns is a classic low-priority move
        if piece == PAWN && (file_of(from) == 0 || file_of(from) == 7) && !is_capture {
            w *= 0.85;
        }
        // pawn moves in front of our castled king feel scary to humans
        if piece == PAWN {
            let ksq = pos.king_sq(us);
            if KING_ATT[ksq as usize] & (1u64 << from) != 0 && !is_capture {
                w *= 0.8;
            }
        }
        // "engine-weird" backward retreats
        let fwd = if let Color::White = us {
            rank_of(to) as i32 - rank_of(from) as i32
        } else {
            rank_of(from) as i32 - rank_of(to) as i32
        };
        if fwd < 0 && !is_capture && piece != KING {
            w *= 0.8;
        }

        // blend toward neutrality as target rises
        let blended = 1.0 * (1.0 - strength) + w * strength;

        // Voluntarily forfeiting castling rights is a near-universal red
        // flag that doesn't fade with (estimated) opponent strength the
        // way ordinary stylistic tells do -- real humans essentially never
        // do this without a concrete tactical reason, at almost any rating
        // above complete beginner. Applied as a final hard multiplier
        // AFTER the blend above (not folded into `w`), because the blend
        // itself dilutes any in-`w` penalty too much at the moderate
        // target Elo (~1800-2200) where MATCH mode's blunder simulator is
        // most active -- confirmed via a live 3-min game vs full-strength
        // RubiChess where the old flat 0.45 king-walk penalty (diluted by
        // the blend down to ~0.87 net weight at target~1838) still let
        // 3.Kd2 get sampled as a "human blunder", forfeiting all castling
        // rights on move 3 for no tactical reason and losing the game.
        if piece == KING && mv.kind() != MK_CASTLE {
            let own_castle = if let Color::White = us { WK | WQ } else { BK | BQ };
            if (pos.castling & own_castle) != 0 && (next.castling & own_castle) == 0 {
                return blended * 0.15;
            }
        }
        blended
    }
}

// ---------------------------------------------------------------------------
// Move selection
// ---------------------------------------------------------------------------

pub struct Selection {
    pub mv: Move,
    pub reason: String,
}

/// Max centipawns we are willing to give up vs. the best move at a target Elo.
fn max_loss_for(target: i32) -> f64 {
    ((ENGINE_CEILING - target).max(0) as f64 * 0.35).max(12.0)
}

/// Pick the move to play from MultiPV lines according to the persona.
///
/// `probe` lets CLINCH measure trap potential: it analyzes a position
/// (the one after our candidate move) and returns the opponent's best
/// reply lines; implemented by the UCI layer with a small search budget.
pub fn select_move(
    pos: &Position,
    lines: &[Line],
    mode: Mode,
    cfg: &AdaptConfig,
    model: &OpponentModel,
    prior: &dyn MovePrior,
    rng: &mut Rng,
    probe: &mut dyn FnMut(&Position) -> Vec<Line>,
) -> Selection {
    assert!(!lines.is_empty());
    let best = &lines[0];

    match mode {
        Mode::Full => Selection {
            mv: best.mv,
            reason: "best move (full strength)".to_string(),
        },
        Mode::Defend => Selection {
            mv: best.mv,
            reason: "maximum resistance".to_string(),
        },
        Mode::Punish => {
            // a found mate is played, period
            if crate::search::is_mate_score(best.score) && best.score > 0 {
                return Selection {
                    mv: best.mv,
                    reason: "punishing (mate found)".to_string(),
                };
            }
            // among near-best lines, prefer forcing moves; when far ahead,
            // prefer simplifying captures over checks (convert, don't stunt)
            let margin = 25;
            let far_ahead = best.score > 500;
            let mut pick = best.mv;
            let mut why = "best move";
            for l in lines {
                if best.score - l.score > margin {
                    break;
                }
                let is_cap = pos.board[l.mv.to() as usize] != NO_PIECE || l.mv.kind() == MK_EP;
                let gives_check = in_check(&pos.make(l.mv));
                let preferred = if far_ahead { is_cap } else { is_cap || gives_check };
                if preferred {
                    pick = l.mv;
                    why = if is_cap { "forcing capture" } else { "forcing check" };
                    break;
                }
            }
            Selection {
                mv: pick,
                reason: format!("punishing ({})", why),
            }
        }
        Mode::Clinch => {
            // probe top candidates: a trap-laden move is one where the
            // opponent's best reply is far better than their second-best
            // (narrow path), while our eval stays acceptable.
            let budget_loss = 40;
            let both_queens_now =
                pos.bb[0][QUEEN] != 0 && pos.bb[1][QUEEN] != 0;
            let mut best_score = f64::MIN;
            let mut pick = best.mv;
            let mut picked_gap = 0;
            for l in lines.iter().take(3) {
                let loss = best.score - l.score;
                if loss > budget_loss {
                    continue;
                }
                let after = pos.make(l.mv);
                let replies = probe(&after);
                let gap = if replies.len() >= 2 {
                    (replies[0].score - replies[1].score).max(0)
                } else {
                    0
                };
                let mut s = -(loss as f64) + gap as f64 * 0.6;
                // dirty chess wants pieces on the board: reward lines that
                // keep both queens alive in a drawish position
                if both_queens_now && after.bb[0][QUEEN] != 0 && after.bb[1][QUEEN] != 0 {
                    s += 12.0;
                }
                if s > best_score {
                    best_score = s;
                    pick = l.mv;
                    picked_gap = gap;
                }
            }
            Selection {
                mv: pick,
                reason: format!("venom line (only-move gap {} cp for opponent)", picked_gap),
            }
        }
        Mode::Match => {
            let target = target_elo(cfg, model);
            let max_loss = max_loss_for(target);
            let temp = (max_loss / 2.0).max(8.0);

            // The accurate multipv `lines` are always included, but at low
            // targets they are NOT the whole candidate pool: `lines` is only
            // ever the engine's own top-K BEST moves (K = MultiPV, min 5),
            // so a real weak human's characteristic errors (hung pieces,
            // missed one-movers) are structurally never present in it,
            // regardless of how low `target`/`max_loss` is set -- reweighting
            // among only-good options can't produce weak play. Below 2200
            // (HeuristicPrior's own weakening cutoff -- no point widening
            // where nothing downstream uses it), score every other legal
            // move too, via the same shallow `probe` search CLINCH uses
            // (not a raw static eval): a naive 1-ply eval can't see that a
            // move hangs a piece, since the material loss only shows up
            // after the opponent's reply -- `probe`'s search (which bottoms
            // out in quiescence) correctly resolves that capture sequence.
            // Confirmed necessary via a 64-level Elo-ladder stress test
            // (own play vs Stockfish at every UCI_Elo from 500-3200) that
            // showed zero measurable correlation between the declared
            // target and actual move quality before this fix.
            let mut cand: Vec<(Move, i32)> = lines.iter().map(|l| (l.mv, l.score)).collect();
            if target < 2200 {
                let ml = legal(pos);
                for &m in ml.as_slice() {
                    if cand.iter().any(|&(cm, _)| cm == m) {
                        continue;
                    }
                    let after = pos.make(m);
                    let sc = match probe(&after).first() {
                        Some(l) => -l.score,
                        // no time budget for the probe right now -- skip
                        // rather than trust an un-vetted candidate
                        None => continue,
                    };
                    cand.push((m, sc));
                }
            }

            // Never blend into a move that walks into mate, and never
            // decline a mate we have found -- filter once, up front, so
            // both the "normal" and "blunder" sampling below share it.
            let viable: Vec<(Move, i32)> = cand
                .iter()
                .copied()
                .filter(|&(_, score)| {
                    let walks_into_mate =
                        crate::search::is_mate_score(score) && score < 0 && best.score > score;
                    !walks_into_mate
                        && !((best.score - score) > 0 && crate::search::is_mate_score(best.score))
                })
                .collect();
            if viable.is_empty() {
                return Selection {
                    mv: best.mv,
                    reason: "best move (no viable alternatives)".to_string(),
                };
            }
            let cand_moves: Vec<Move> = viable.iter().map(|&(m, _)| m).collect();
            let priors = prior.priors(pos, &cand_moves, target);

            // Real human error isn't "somewhat worse with some randomness"
            // -- it's a genuinely different regime: usually fine, but with
            // occasional QUALITATIVE blunders (hung pieces). A single
            // smoothly-decaying softmax over a wide pool structurally can't
            // reproduce that: the many "slightly worse" legal moves dilute
            // any individual real blunder's selection probability down to
            // near zero, even once it's included in the pool (confirmed
            // empirically -- widening the pool alone, without this, showed
            // ~0 correlation between target Elo and actual move quality
            // across a 64-level ladder test). Model it as an explicit
            // two-mode mixture instead: with `blunder_prob(target)`,
            // deliberately sample from the WORSE end of the viable pool
            // (weight grows with loss, not against it); otherwise sample
            // among the better options as before. Rate calibrated loosely
            // against real blunder-frequency-by-rating references (roughly
            // 30-35% of moves at ~500 Elo, ~0% at 2200+, matching
            // HeuristicPrior's own weakening cutoff).
            let blunder_prob = ((2200 - target).max(0) as f64 / 1700.0).clamp(0.0, 1.0) * 0.35;
            if blunder_prob > 0.0 && rng.f64() < blunder_prob {
                let lo = max_loss * 0.25;
                let mut weights: Vec<f64> = Vec::with_capacity(viable.len());
                for (i, &(_, score)) in viable.iter().enumerate() {
                    let loss = (best.score - score) as f64;
                    if loss > max_loss || loss < lo {
                        weights.push(0.0);
                        continue;
                    }
                    weights.push(loss * priors.get(i).copied().unwrap_or(1.0).max(0.1));
                }
                let total: f64 = weights.iter().sum();
                if total > 0.0 {
                    let mut roll = rng.f64() * total;
                    for (&(mv, score), w) in viable.iter().zip(&weights) {
                        roll -= w;
                        if roll <= 0.0 {
                            let loss = best.score - score;
                            return Selection {
                                mv,
                                reason: format!("human blunder at ~{} (loss {} cp)", target, loss),
                            };
                        }
                    }
                }
                // nothing in the "genuinely bad" band this move (e.g. every
                // legal option is close in value) -- fall through to normal
            }

            let mut weights: Vec<f64> = Vec::with_capacity(viable.len());
            for (i, &(_, score)) in viable.iter().enumerate() {
                let loss = (best.score - score) as f64;
                if loss > max_loss {
                    weights.push(0.0);
                    continue;
                }
                let w = (-loss / temp).exp() * priors.get(i).copied().unwrap_or(1.0);
                weights.push(w);
            }
            let total: f64 = weights.iter().sum();
            if total <= 0.0 {
                return Selection {
                    mv: best.mv,
                    reason: "best move (no viable alternatives)".to_string(),
                };
            }
            let mut roll = rng.f64() * total;
            for (&(mv, score), w) in viable.iter().zip(&weights) {
                roll -= w;
                if roll <= 0.0 {
                    let loss = best.score - score;
                    return Selection {
                        mv,
                        reason: format!("human-plausible at ~{} (loss {} cp)", target, loss),
                    };
                }
            }
            Selection {
                mv: best.mv,
                reason: "best move".to_string(),
            }
        }
    }
}

/// Weight applied to an opponent-move observation given the position's
/// character: forced or trivial situations carry little Elo signal.
pub fn difficulty_weight(pre_lines: &[Line], legal_count: usize, was_book: bool) -> f64 {
    if was_book {
        return 0.3;
    }
    if legal_count <= 2 {
        return 0.4;
    }
    if pre_lines.len() >= 2 {
        let gap = (pre_lines[0].score - pre_lines[1].score).abs();
        if gap > 300 {
            // one obvious move (e.g. forced recapture): finding it proves little
            return 0.5;
        }
        if gap < 40 {
            // many decent moves: also low signal
            return 0.8;
        }
    }
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_converges_downward_on_blunders() {
        let mut m = OpponentModel::new();
        for _ in 0..10 {
            m.observe(220, 1.0);
        }
        assert!(m.estimate() < 1000, "estimate {}", m.estimate());
        assert!(m.confidence() < 300);
        assert!(m.last_was_blunder());
    }

    #[test]
    fn model_converges_upward_on_strong_play() {
        let mut m = OpponentModel::new();
        for _ in 0..12 {
            m.observe(8, 1.0);
        }
        assert!(m.estimate() > 2300, "estimate {}", m.estimate());
    }

    #[test]
    fn engine_seed_from_uci_opponent() {
        let mut m = OpponentModel::new();
        let log = m.seed_from_uci_opponent("GM 3644 computer Stockfish 16.1");
        assert!(m.is_computer);
        assert_eq!(m.estimate(), 3644);
        assert!(log.contains("Stockfish"));
    }

    #[test]
    fn human_declared_elo_only_nudges() {
        let mut m = OpponentModel::new();
        m.seed_from_uci_opponent("- 2800 human MagnusFan");
        assert!(m.estimate() < 2300, "human declared elo must not dominate");
    }

    #[test]
    fn persona_hysteresis_latches() {
        let cfg = AdaptConfig::default();
        let m = OpponentModel::new();
        // -120 cp: not bad enough to ENTER defend...
        assert_ne!(decide_mode(&cfg, &m, -120, 20, Mode::Match), Mode::Defend);
        // ...but bad enough to STAY in defend once there
        assert_eq!(decide_mode(&cfg, &m, -120, 20, Mode::Defend), Mode::Defend);
        // recovered: defend releases
        assert_ne!(decide_mode(&cfg, &m, -20, 20, Mode::Defend), Mode::Defend);
        // punish latches while the advantage holds
        assert_eq!(decide_mode(&cfg, &m, 350, 20, Mode::Punish), Mode::Punish);
        // clinch holds inside the +-100 band
        assert_eq!(decide_mode(&cfg, &m, 90, 40, Mode::Clinch), Mode::Clinch);
        assert_ne!(decide_mode(&cfg, &m, 150, 40, Mode::Clinch), Mode::Clinch);
    }

    #[test]
    fn engine_suspicion_from_ceiling() {
        // anonymous opponent pinned at the ceiling, long enough, low vol
        let mut m = OpponentModel::new();
        for _ in 0..16 {
            m.observe(5, 1.0);
        }
        assert!(m.estimate() >= 2500, "estimate {}", m.estimate());
        assert!(m.engine_suspect());
        // 10 clean opening moves is *not* enough (high-level human theory)
        let mut short = OpponentModel::new();
        for _ in 0..10 {
            short.observe(5, 1.0);
        }
        assert!(
            !short.engine_suspect(),
            "10 clean moves must not ceiling-flag, estimate {}",
            short.estimate()
        );
        // merely-good erratic human is not flagged
        let mut h = OpponentModel::new();
        for i in 0..16 {
            h.observe(if i % 3 == 0 { 150 } else { 20 }, 1.0);
        }
        assert!(!h.engine_suspect(), "estimate {}", h.estimate());
    }

    #[test]
    fn engine_suspicion_from_clock() {
        let mut m = OpponentModel::new();
        // opening instants must not count
        for _ in 0..7 {
            m.observe(5, 1.0);
            m.observe_time(80, true);
        }
        assert!(!m.engine_suspect(), "opening premoves are not an engine tell");
        // middlegame: 4 instant-strong replies after sample>=8
        for _ in 0..4 {
            m.observe(5, 1.0);
            m.observe_time(80, true);
        }
        assert!(m.engine_suspect());
        let cfg = AdaptConfig::default();
        assert_eq!(decide_mode(&cfg, &m, 0, 20, Mode::Match), Mode::Full);
    }

    #[test]
    fn maia_computer_does_not_force_full() {
        let mut m = OpponentModel::new();
        m.seed_from_uci_opponent("- 1600 computer Maia 2");
        assert!(m.is_computer);
        assert!(!m.engine_suspect(), "Maia is the MATCH target, estimate {}", m.estimate());
        let cfg = AdaptConfig::default();
        assert_ne!(decide_mode(&cfg, &m, 0, 10, Mode::Match), Mode::Full);
    }

    #[test]
    fn stockfish_computer_does_force_full() {
        let mut m = OpponentModel::new();
        m.seed_from_uci_opponent("GM 3644 computer Stockfish 16.1");
        assert!(m.engine_suspect());
        let cfg = AdaptConfig::default();
        assert_eq!(decide_mode(&cfg, &m, 0, 10, Mode::Match), Mode::Full);
    }

    #[test]
    fn declared_human_master_is_not_ceiling_flagged() {
        let mut m = OpponentModel::new();
        m.seed_from_uci_opponent("- 2500 human IM_Player");
        for _ in 0..20 {
            m.observe(8, 1.0);
        }
        assert!(
            !m.engine_suspect(),
            "a labelled 2500 human playing well is MATCH, not FULL, estimate {}",
            m.estimate()
        );
    }

    #[test]
    fn limit_strength_plays_at_uci_elo() {
        let cfg = AdaptConfig {
            adaptive: false,
            limit_strength: true,
            elo_cap: 1400,
            contempt: 25,
        };
        let m = OpponentModel::new();
        assert_eq!(decide_mode(&cfg, &m, 0, 10, Mode::Match), Mode::Match);
        assert_eq!(target_elo(&cfg, &m), 1400);
    }

    #[test]
    fn draw_score_respects_defend() {
        let cfg = AdaptConfig::default();
        assert!(draw_score_for(&cfg, Mode::Clinch) < 0);
        assert!(draw_score_for(&cfg, Mode::Match) < 0);
        assert_eq!(draw_score_for(&cfg, Mode::Defend), 0);
    }

    #[test]
    fn match_never_samples_into_mate() {
        use crate::fen;
        use crate::search::{Line, MATE};
        let pos = fen::startpos();
        let ml = crate::movegen::legal(&pos);
        let mvs: Vec<crate::board::Move> = ml.as_slice().to_vec();
        // candidate 1 is "getting mated" — must never be picked
        let lines = vec![
            Line { mv: mvs[0], score: 20, depth: 8, pv: vec![mvs[0]] },
            Line { mv: mvs[1], score: -(MATE - 6), depth: 8, pv: vec![mvs[1]] },
        ];
        let cfg = AdaptConfig::default();
        let mut m = OpponentModel::new();
        for _ in 0..8 {
            m.observe(300, 1.0); // very weak opponent: temperature is huge
        }
        let prior = HeuristicPrior;
        let mut rng = Rng::new(1234);
        for _ in 0..200 {
            let sel = select_move(
                &pos, &lines, Mode::Match, &cfg, &m, &prior, &mut rng,
                &mut |_p| Vec::new(),
            );
            assert_eq!(sel.mv, mvs[0], "sampled a move that walks into mate");
        }
    }

    #[test]
    fn king_move_that_forfeits_castling_scores_far_below_a_normal_king_walk() {
        // Regression: a live 3-min game vs full-strength RubiChess had
        // MATCH mode's blunder simulator pick 3.Kd2 (forfeiting all
        // castling rights on move 3) because the old flat 0.45 king-walk
        // penalty wasn't nearly low enough relative to an ordinary
        // developing move. White has both castling rights here; Ke1d2
        // forfeits them both, Nb1c3 doesn't touch them at all.
        use crate::fen;
        let pos = fen::parse("4k3/8/8/8/8/8/8/RN2K2R w KQ - 0 1").unwrap();
        let king_walk = crate::movegen::legal(&pos)
            .as_slice()
            .iter()
            .copied()
            .find(|m| m.from() == 4 && m.to() == 11) // e1 -> d2
            .expect("Ke1-d2 should be legal here");
        let knight_dev = crate::movegen::legal(&pos)
            .as_slice()
            .iter()
            .copied()
            .find(|m| m.from() == 1 && m.to() == 18) // b1 -> c3
            .expect("Nb1-c3 should be legal here");
        let prior = HeuristicPrior;
        let target = 1800; // comfortably inside MATCH mode's active range
        let w_king = prior.single(&pos, king_walk, target);
        let w_knight = prior.single(&pos, knight_dev, target);
        assert!(
            w_king < w_knight * 0.5,
            "castling-forfeiting king walk (weight {}) should score well below an ordinary \
             developing move (weight {})",
            w_king,
            w_knight
        );
    }

    #[test]
    fn persona_state_dwell_ignores_one_move_clinch_spike() {
        let cfg = AdaptConfig::default();
        let m = OpponentModel::new();
        let mut s = PersonaState::default();
        // quiet equal game, then one noisy +10 that would enter CLINCH at move 40
        s.update(&cfg, &m, 5, 40);
        let after_spike = s.update(&cfg, &m, 8, 40);
        // first CLINCH vote must not switch yet
        assert_eq!(after_spike, Mode::Match, "single CLINCH vote must dwell");
        let held = s.update(&cfg, &m, 4, 40);
        assert_eq!(held, Mode::Match);
    }

    #[test]
    fn persona_state_punish_on_blunder_is_immediate() {
        let cfg = AdaptConfig::default();
        let mut m = OpponentModel::new();
        m.observe(220, 1.0);
        assert!(m.last_was_blunder());
        let mut s = PersonaState::default();
        s.update(&cfg, &m, 80, 20);
        let mode = s.update(&cfg, &m, 90, 20);
        assert_eq!(mode, Mode::Punish);
    }

    #[test]
    fn persona_state_ema_rejects_single_eval_spike_across_defend() {
        let cfg = AdaptConfig::default();
        let m = OpponentModel::new();
        let mut s = PersonaState::default();
        s.update(&cfg, &m, 0, 15);
        // one −190 blip: raw decide_mode would enter DEFEND; EMA ~ −66 stays MATCH
        let mode = s.update(&cfg, &m, -190, 15);
        assert_ne!(mode, Mode::Defend, "one-ply −190 must not enter DEFEND");
        // sustained collapse still enters via emergency (−220) or dwell
        let mut hard = PersonaState::default();
        hard.update(&cfg, &m, 0, 15);
        assert_eq!(hard.update(&cfg, &m, -400, 15), Mode::Defend);
    }
}
