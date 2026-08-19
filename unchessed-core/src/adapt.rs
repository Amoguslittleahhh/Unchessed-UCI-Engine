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

    /// Standard normal via Box--Muller, deterministic for a fixed seed.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.f64().clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Opponent identity, strength posterior, and live evidence
// ---------------------------------------------------------------------------

const MIN_TRACKED_ELO: i32 = 100;
const MAX_TRACKED_ELO: i32 = 3650;
const ELO_BUCKETS: usize = (MAX_TRACKED_ELO - MIN_TRACKED_ELO + 1) as usize;

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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DeclaredAgent {
    Unknown,
    Human,
    Computer,
}

/// Persistent identity metadata supplied by the GUI. This is deliberately
/// separate from per-game observations: `ucinewgame` resets evidence but must
/// not erase who the opponent is.
#[derive(Clone, Debug)]
pub struct OpponentDescriptor {
    pub title: Option<String>,
    pub declared_elo: Option<i32>,
    pub agent: DeclaredAgent,
    pub name: Option<String>,
    pub known_engine: Option<&'static str>,
    pub known_engine_elo: Option<i32>,
}

impl Default for OpponentDescriptor {
    fn default() -> Self {
        OpponentDescriptor {
            title: None,
            declared_elo: None,
            agent: DeclaredAgent::Unknown,
            name: None,
            known_engine: None,
            known_engine_elo: None,
        }
    }
}

impl OpponentDescriptor {
    /// Parse the UCI convention:
    /// `<title> <elo> <computer|human> <name...>`.
    pub fn parse(value: &str) -> OpponentDescriptor {
        let mut toks = value.split_whitespace();
        let title = toks.next().filter(|s| *s != "-").map(str::to_string);
        let elo_tok = toks.next().unwrap_or("-");
        let kind = toks.next().unwrap_or("unknown");
        let name = toks.collect::<Vec<_>>().join(" ");
        let name = (!name.is_empty()).then_some(name);
        let declared_elo = elo_tok
            .parse::<i32>()
            .ok()
            .filter(|e| *e > 0)
            .map(|e| e.clamp(MIN_TRACKED_ELO, MAX_TRACKED_ELO));
        let agent = if kind.eq_ignore_ascii_case("computer") {
            DeclaredAgent::Computer
        } else if kind.eq_ignore_ascii_case("human") {
            DeclaredAgent::Human
        } else {
            DeclaredAgent::Unknown
        };
        let lower = name.as_deref().unwrap_or("").to_lowercase();
        let known = KNOWN_ENGINES
            .iter()
            .find(|(key, _)| lower.contains(key))
            .copied();
        OpponentDescriptor {
            title,
            declared_elo,
            agent,
            name,
            known_engine: known.map(|(key, _)| key),
            known_engine_elo: known.map(|(_, elo)| elo),
        }
    }

    /// Known/computer identity is an anti-troll fact, independent of whether
    /// that engine is deliberately limited to a low playing strength.
    pub fn anti_troll_lock(&self) -> bool {
        self.agent == DeclaredAgent::Computer || self.known_engine.is_some()
    }

    fn initial_strength(&self) -> (f64, f64, f64) {
        match self.agent {
            DeclaredAgent::Computer => {
                let mean = self.declared_elo.or(self.known_engine_elo).unwrap_or(2800) as f64;
                (
                    mean,
                    6.0,
                    if self.declared_elo.is_some() {
                        220.0
                    } else {
                        350.0
                    },
                )
            }
            DeclaredAgent::Human => {
                // A declared human rating is useful as a prior, not truth.
                let mean = self
                    .declared_elo
                    .map(|e| (1500.0 + e as f64) * 0.5)
                    .unwrap_or(1500.0);
                (mean, 1.0, 650.0)
            }
            DeclaredAgent::Unknown => (1500.0, 1.0, 750.0),
        }
    }

    pub fn describe(&self, seed: i32) -> String {
        let name = self.name.as_deref().unwrap_or("unknown");
        match self.agent {
            DeclaredAgent::Computer => match self.known_engine {
                Some(engine) => format!(
                    "opponent: {} (computer, known engine '{}', playing-strength prior ~{}, anti-troll locked)",
                    name, engine, seed
                ),
                None => format!(
                    "opponent: {} (computer, playing-strength prior ~{}, anti-troll locked)",
                    name, seed
                ),
            },
            DeclaredAgent::Human => format!(
                "opponent: {} (human metadata; strength/type will be verified from play)",
                name
            ),
            DeclaredAgent::Unknown => {
                "opponent: unknown type; using conservative no-troll classification".to_string()
            }
        }
    }
}

#[derive(Clone)]
struct RatingPosterior {
    /// One bucket for every integer Elo from 100 through 3650 inclusive.
    /// This is output granularity, not a claim of one-Elo statistical precision;
    /// callers must use the credible interval for decisions.
    probabilities: Vec<f64>,
}

impl RatingPosterior {
    fn new(mean: f64, sigma: f64) -> RatingPosterior {
        let mut probabilities = vec![0.0; ELO_BUCKETS];
        for (i, probability) in probabilities.iter_mut().enumerate() {
            let elo = MIN_TRACKED_ELO as f64 + i as f64;
            let z = (elo - mean) / sigma.max(100.0);
            *probability = (-0.5 * z * z).exp().max(1e-12);
        }
        let mut posterior = RatingPosterior { probabilities };
        posterior.normalize();
        posterior
    }

    fn normalize(&mut self) {
        let sum: f64 = self.probabilities.iter().sum();
        if sum > 0.0 && sum.is_finite() {
            for probability in &mut self.probabilities {
                *probability /= sum;
            }
        } else {
            self.probabilities.fill(1.0 / ELO_BUCKETS as f64);
        }
    }

    fn observe(&mut self, elo_sample: f64, evidence: f64) {
        // Difficulty controls the likelihood width. Forced/book moves are
        // intentionally broad and therefore barely move the posterior.
        let sigma = (430.0 / evidence.max(0.05).sqrt()).clamp(180.0, 1600.0);
        for (i, probability) in self.probabilities.iter_mut().enumerate() {
            let elo = MIN_TRACKED_ELO as f64 + i as f64;
            let z = (elo - elo_sample) / sigma;
            let likelihood = (-0.5 * z * z).exp().max(1e-8);
            // Tiny forgetting keeps the model able to track fatigue, a
            // deliberately limited engine, or a changing opponent.
            *probability = probability.powf(0.997) * likelihood;
        }
        self.normalize();
    }

    fn quantile(&self, q: f64) -> i32 {
        let mut cumulative = 0.0;
        for (i, probability) in self.probabilities.iter().enumerate() {
            cumulative += probability;
            if cumulative >= q {
                return MIN_TRACKED_ELO + i as i32;
            }
        }
        MAX_TRACKED_ELO
    }

    fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    fn mode(&self) -> i32 {
        let index = self
            .probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap_or(0);
        MIN_TRACKED_ELO + index as i32
    }
}

#[derive(Clone)]
pub struct OpponentModel {
    descriptor: OpponentDescriptor,
    /// Running Elo point estimate retained for compatibility with the
    /// existing calibrated selector. The band posterior supplies uncertainty.
    mean: f64,
    weight: f64,
    posterior: RatingPosterior,
    human_probability: f64,
    engine_probability: f64,
    pub samples: u32,
    pub last_cp_loss: Option<i32>,
    fresh_blunder: bool,
    pub opponent_name: Option<String>,
    pub is_computer: bool,
    pub declared_elo: Option<i32>,
    pub book_depth_plies: u32,
    recent_sample_mean: Option<f64>,
    recent_var: f64,
    trend_ema: f64,
    /// Retained informative per-move Elo samples; lower-tail skill and
    /// upper-tail ceiling are read from this 32-move window.
    strength_samples: Vec<f64>,
    /// Log fractions of remaining clock consumed on informative moves.
    time_log_fractions: Vec<f64>,
    engine_latched: bool,
}

impl Default for OpponentModel {
    fn default() -> Self {
        Self::new()
    }
}

impl OpponentModel {
    pub fn new() -> OpponentModel {
        Self::from_descriptor(OpponentDescriptor::default())
    }

    pub fn from_descriptor(descriptor: OpponentDescriptor) -> OpponentModel {
        let (mean, weight, sigma) = descriptor.initial_strength();
        let (human_probability, engine_probability) = match descriptor.agent {
            DeclaredAgent::Computer => (0.01, 0.99),
            DeclaredAgent::Human => (0.90, 0.10),
            DeclaredAgent::Unknown => (0.55, 0.45),
        };
        let is_computer = descriptor.agent == DeclaredAgent::Computer;
        OpponentModel {
            posterior: RatingPosterior::new(mean, sigma),
            opponent_name: descriptor.name.clone(),
            declared_elo: descriptor.declared_elo,
            descriptor,
            mean,
            weight,
            human_probability,
            engine_probability,
            samples: 0,
            last_cp_loss: None,
            fresh_blunder: false,
            is_computer,
            book_depth_plies: 0,
            recent_sample_mean: None,
            recent_var: 0.0,
            trend_ema: 0.0,
            strength_samples: Vec::with_capacity(32),
            time_log_fractions: Vec::with_capacity(32),
            engine_latched: is_computer,
        }
    }

    pub fn descriptor(&self) -> &OpponentDescriptor {
        &self.descriptor
    }

    pub fn reset_for_new_game(&self) -> OpponentModel {
        Self::from_descriptor(self.descriptor.clone())
    }

    /// Backwards-compatible setter used by tests and non-UCI callers.
    pub fn seed_from_uci_opponent(&mut self, value: &str) -> String {
        let descriptor = OpponentDescriptor::parse(value);
        *self = Self::from_descriptor(descriptor);
        self.descriptor.describe(self.estimate())
    }

    fn elo_sample(cp_loss: f64) -> f64 {
        (2950.0 - 850.0 * (1.0 + cp_loss.max(0.0) / 20.0).ln())
            .clamp(MIN_TRACKED_ELO as f64, MAX_TRACKED_ELO as f64)
    }

    fn normalize_agent_probabilities(&mut self) {
        let sum = self.human_probability + self.engine_probability;
        if sum > 0.0 && sum.is_finite() {
            self.human_probability /= sum;
            self.engine_probability /= sum;
        } else {
            self.human_probability = 0.5;
            self.engine_probability = 0.5;
        }
    }

    fn update_agent_from_move(&mut self, cp_loss: i32, evidence: f64) {
        if self.descriptor.agent == DeclaredAgent::Computer || evidence < 0.5 {
            return;
        }
        // Accuracy by itself is weak type evidence; recognizably human errors
        // are stronger evidence. A trained human-policy likelihood will replace
        // these conservative factors in the next phase.
        if cp_loss >= 180 {
            self.human_probability *= 1.8;
        } else if cp_loss >= 80 {
            self.human_probability *= 1.25;
        } else if cp_loss <= 10 {
            self.engine_probability *= 1.03;
        }
        self.normalize_agent_probabilities();
    }

    fn strength_quantile(&self, q: f64) -> Option<f64> {
        if self.strength_samples.is_empty() {
            return None;
        }
        let mut samples = self.strength_samples.clone();
        samples.sort_by(f64::total_cmp);
        let index = ((samples.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
        samples.get(index).copied()
    }

    pub fn timing_autocorrelation(&self) -> Option<f64> {
        if self.time_log_fractions.len() < 6 {
            return None;
        }
        let left = &self.time_log_fractions[..self.time_log_fractions.len() - 1];
        let right = &self.time_log_fractions[1..];
        let left_mean = left.iter().sum::<f64>() / left.len() as f64;
        let right_mean = right.iter().sum::<f64>() / right.len() as f64;
        let mut covariance = 0.0;
        let mut left_var = 0.0;
        let mut right_var = 0.0;
        for (&a, &b) in left.iter().zip(right) {
            let da = a - left_mean;
            let db = b - right_mean;
            covariance += da * db;
            left_var += da * da;
            right_var += db * db;
        }
        let denominator = (left_var * right_var).sqrt();
        (denominator > 1e-12).then_some(covariance / denominator)
    }

    fn refresh_engine_latch(&mut self) {
        let ceiling = self.strength_quantile(0.75).unwrap_or(self.mean);
        let regular_timing = self.timing_autocorrelation().unwrap_or(-1.0) >= 0.45;
        let enough_evidence = self.weight >= 10.0 || (regular_timing && self.weight >= 6.0);
        let ceiling_signal = ceiling >= 2450.0 && enough_evidence && self.volatility() <= 500;
        if self.is_computer || ceiling_signal {
            // Timing can lower the evidence bar for an already ceiling-level
            // opponent, but can never classify a weak player by itself.
            self.engine_latched = true;
        }
    }

    pub fn observe(&mut self, cp_loss: i32, difficulty_weight: f64) {
        self.last_cp_loss = Some(cp_loss.max(0));
        self.fresh_blunder = cp_loss >= 180;
        let w = difficulty_weight.clamp(0.05, 2.0);
        let sample = Self::elo_sample(cp_loss as f64);

        if let Some(recent) = self.recent_sample_mean {
            let delta = sample - recent;
            let updated = recent + 0.20 * delta;
            let residual = sample - updated;
            self.recent_var = 0.85 * self.recent_var + 0.15 * residual * residual;
            self.trend_ema = 0.80 * self.trend_ema + 0.20 * delta;
            self.recent_sample_mean = Some(updated);
        } else {
            self.recent_sample_mean = Some(sample);
            self.recent_var = 0.0;
            self.trend_ema = 0.0;
        }

        self.mean = (self.mean * self.weight + sample * w) / (self.weight + w);
        self.weight = ((self.weight + w).min(14.0) * 0.985).max(1.0);
        if w >= 0.5 {
            if self.strength_samples.len() == 32 {
                self.strength_samples.remove(0);
            }
            self.strength_samples.push(sample);
        }
        self.posterior.observe(sample, w);
        self.update_agent_from_move(cp_loss, w);
        self.samples += 1;
        self.refresh_engine_latch();
    }

    pub fn observe_time_fraction(
        &mut self,
        used_ms: u64,
        remaining_ms: u64,
        position_had_choice: bool,
    ) {
        if !position_had_choice || used_ms == 0 {
            return;
        }
        let before_move = remaining_ms.saturating_add(used_ms).max(1);
        let fraction = (used_ms as f64 / before_move as f64).clamp(1e-6, 1.0);
        if self.time_log_fractions.len() == 32 {
            self.time_log_fractions.remove(0);
        }
        self.time_log_fractions.push(fraction.ln());

        // Regularity is only a weak type modulator. It never reaches an engine
        // conclusion without independent ceiling-level move quality.
        if self.timing_autocorrelation().unwrap_or(-1.0) >= 0.45
            && matches!(self.last_cp_loss, Some(loss) if loss <= 60)
        {
            self.engine_probability *= 1.10;
            self.normalize_agent_probabilities();
        }
        self.refresh_engine_latch();
    }

    /// Compatibility helper for tests/callers without remaining-clock data.
    pub fn observe_time(&mut self, used_ms: u64, position_had_choice: bool) {
        self.observe_time_fraction(used_ms, used_ms.saturating_mul(20), position_had_choice);
    }

    pub fn engine_suspect(&self) -> bool {
        self.engine_latched
    }

    /// Agent type and playing strength are independent. A known/declared weak
    /// engine remains anti-troll locked but can still be matched at its current
    /// strength; unrestricted or behaviorally detected engines get FULL.
    pub fn requires_full_strength(&self) -> bool {
        if self.descriptor.agent == DeclaredAgent::Computer {
            let unrestricted_known = self.descriptor.declared_elo.is_none()
                && self.descriptor.known_engine_elo.unwrap_or(0) >= 2600;
            return unrestricted_known || self.upper_bound() >= 2400;
        }
        self.engine_latched
    }

    pub fn human_probability(&self) -> f64 {
        self.human_probability
    }

    pub fn engine_probability(&self) -> f64 {
        self.engine_probability
    }

    pub fn classification(&self) -> &'static str {
        if self.descriptor.known_engine.is_some() {
            "known engine"
        } else if self.engine_latched || self.engine_probability >= 0.90 {
            "engine"
        } else if self.human_probability >= 0.90 {
            "human"
        } else {
            "uncertain"
        }
    }

    pub fn anti_troll_lock(&self) -> bool {
        self.descriptor.anti_troll_lock() || self.engine_latched
    }

    pub fn auto_troll_allowed(&self) -> bool {
        !self.anti_troll_lock() && self.samples >= 6 && self.human_probability >= 0.90
    }

    pub fn confident_human(&self) -> bool {
        !self.anti_troll_lock()
            && self.human_probability >= 0.90
            && (self.descriptor.agent == DeclaredAgent::Human || self.samples >= 4)
    }

    pub fn volatility(&self) -> i32 {
        self.recent_var.sqrt().round() as i32
    }

    pub fn observe_book_move(&mut self, plies: u32) {
        self.book_depth_plies = self.book_depth_plies.max(plies);
        self.fresh_blunder = false;
        if plies >= 10 {
            self.observe(10, 0.25);
            self.fresh_blunder = false;
        }
    }

    /// Clear one-move tactical events after persona selection. Historical
    /// loss remains available for logs and statistics.
    pub fn mark_decision_complete(&mut self) {
        self.fresh_blunder = false;
    }

    pub fn estimate(&self) -> i32 {
        self.strength_quantile(0.20)
            .unwrap_or(self.mean)
            .round()
            .clamp(MIN_TRACKED_ELO as f64, MAX_TRACKED_ELO as f64) as i32
    }

    pub fn lower_bound(&self) -> i32 {
        self.posterior.quantile(0.10)
    }

    pub fn upper_bound(&self) -> i32 {
        self.posterior.quantile(0.90)
    }

    /// Probability array indexed by `elo - 100`, one bucket per integer Elo
    /// through 3650 inclusive.
    pub fn rating_probabilities(&self) -> &[f64] {
        self.posterior.probabilities()
    }

    pub fn rating_probability(&self, elo: i32) -> f64 {
        if !(MIN_TRACKED_ELO..=MAX_TRACKED_ELO).contains(&elo) {
            return 0.0;
        }
        self.posterior.probabilities[(elo - MIN_TRACKED_ELO) as usize]
    }

    pub fn most_likely_elo(&self) -> i32 {
        self.posterior.mode()
    }

    pub fn match_offset(&self) -> i32 {
        let amplitude = self.confidence().min(100);
        match self.samples % 4 {
            1 => amplitude,
            3 => -amplitude,
            _ => 0,
        }
    }

    pub fn confidence(&self) -> i32 {
        let posterior_half_width = (self.upper_bound() - self.lower_bound()) / 2;
        // A single stationary-Elo posterior can become artificially narrow on
        // alternating strong/blunder samples. Recent volatility explicitly
        // widens the reported interval for sandbagging/changing profiles.
        posterior_half_width
            .max(self.volatility() / 2)
            .clamp(50, 1200)
    }

    pub fn trend(&self) -> &'static str {
        if !self.is_computer && self.engine_suspect() {
            return if self.timing_autocorrelation().unwrap_or(-1.0) >= 0.45 {
                "ceiling play with regular clock allocation — engine suspected"
            } else {
                "sustained ceiling-level play — engine suspected"
            };
        }
        if self.volatility() > 600 && self.samples >= 6 {
            return "erratic (sandbagging?)";
        }
        if self.trend_ema > 35.0 {
            "trending up"
        } else if self.trend_ema < -35.0 {
            "trending down"
        } else if self.volatility() > 380 && self.samples >= 6 {
            "erratic (sandbagging?)"
        } else {
            "steady"
        }
    }

    pub fn last_was_blunder(&self) -> bool {
        self.fresh_blunder
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GamePhase {
    Opening,
    Middlegame,
    Endgame,
}

#[derive(Clone, Copy, Debug)]
pub struct PersonaContext {
    pub eval_cp: i32,
    pub previous_eval_cp: Option<i32>,
    pub fullmove: u16,
    pub in_check: bool,
    pub legal_moves: usize,
    pub phase: GamePhase,
    pub both_queens: bool,
}

impl PersonaContext {
    pub fn from_position(pos: &Position, eval_cp: i32, previous_eval_cp: Option<i32>) -> Self {
        let non_pawn_non_king = (pos.occ
            & !(pos.bb[0][PAWN] | pos.bb[1][PAWN] | pos.bb[0][KING] | pos.bb[1][KING]))
            .count_ones();
        let both_queens = pos.bb[0][QUEEN] != 0 && pos.bb[1][QUEEN] != 0;
        let phase = if pos.fullmove <= 12 && non_pawn_non_king >= 12 {
            GamePhase::Opening
        } else if !both_queens || non_pawn_non_king <= 6 {
            GamePhase::Endgame
        } else {
            GamePhase::Middlegame
        };
        PersonaContext {
            eval_cp,
            previous_eval_cp,
            fullmove: pos.fullmove,
            in_check: in_check(pos),
            legal_moves: legal(pos).len,
            phase,
            both_queens,
        }
    }

    pub fn eval_swing(self) -> i32 {
        self.previous_eval_cp
            .map(|previous| self.eval_cp - previous)
            .unwrap_or(0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PersonaDecision {
    pub mode: Mode,
    pub reason: &'static str,
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
pub fn decide_persona(
    cfg: &AdaptConfig,
    model: &OpponentModel,
    context: PersonaContext,
    prev: Mode,
) -> PersonaDecision {
    if cfg.limit_strength {
        return PersonaDecision {
            mode: Mode::Match,
            reason: "fixed UCI_Elo has absolute precedence",
        };
    }
    if !cfg.adaptive {
        return PersonaDecision {
            mode: Mode::Full,
            reason: "adaptation disabled",
        };
    }

    let eval = context.eval_cp;
    let swing = context.eval_swing();
    let defend =
        eval < -180 || (prev == Mode::Defend && eval < -80) || (context.in_check && eval < -100);
    if defend {
        return PersonaDecision {
            mode: Mode::Defend,
            reason: if context.in_check {
                "under pressure while in check"
            } else if swing < -120 {
                "position deteriorated sharply"
            } else {
                "clearly worse; maximize resistance"
            },
        };
    }

    // Opponent class determines strength, while the board situation can still
    // choose DEFEND above. This avoids conflating an engine identity threshold
    // with the tactical response required in the current position.
    if model.requires_full_strength() {
        return PersonaDecision {
            mode: Mode::Full,
            reason: "unrestricted or behaviorally detected engine",
        };
    }

    let target = target_elo(cfg, model);
    let found_mate = crate::search::is_mate_score(eval) && eval > 0;
    let fresh_tactical_error = model.last_was_blunder() && eval > 60;
    let live_eval_swing = swing >= 140 && eval > 40;
    let large_skill_gap = target + 500 < ENGINE_CEILING && eval > 250;
    let convert_endgame = context.phase == GamePhase::Endgame && eval > 180;
    let punish_hold = prev == Mode::Punish && eval > 200 && swing > -120;
    if found_mate
        || fresh_tactical_error
        || live_eval_swing
        || large_skill_gap
        || convert_endgame
        || punish_hold
    {
        return PersonaDecision {
            mode: Mode::Punish,
            reason: if found_mate {
                "forced mate available"
            } else if fresh_tactical_error || live_eval_swing {
                "opponent tactical error detected"
            } else if convert_endgame {
                "winning endgame; convert cleanly"
            } else if punish_hold {
                "conversion remains stable"
            } else {
                "large lead against a weaker opponent"
            },
        };
    }

    // Trap-seeking is useful against a confidently human opponent, but not
    // against engines or an unresolved agent type. Queen-rich positions get
    // the full CLINCH treatment; queenless late positions remain MATCH unless
    // there is a concrete winning conversion above.
    let human_opponent = model.confident_human() || model.classification() == "human";
    let clinch_enter = human_opponent
        && context.phase != GamePhase::Opening
        && context.fullmove > 28
        && context.both_queens
        && eval.abs() < 60;
    let clinch_hold = human_opponent
        && prev == Mode::Clinch
        && context.phase != GamePhase::Opening
        && context.both_queens
        && eval.abs() < 100
        && swing > -100;
    if cfg.contempt > 0 && (clinch_enter || clinch_hold) {
        return PersonaDecision {
            mode: Mode::Clinch,
            reason: "late queen-rich human game; seek practical pressure",
        };
    }

    PersonaDecision {
        mode: Mode::Match,
        reason: if model.classification() == "uncertain" {
            "uncertain opponent; conservative strength match"
        } else if context.phase == GamePhase::Opening {
            "normal opening development"
        } else {
            "normal strength-matched play"
        },
    }
}

/// Compatibility wrapper for callers that do not have a full position.
pub fn decide_mode(
    cfg: &AdaptConfig,
    model: &OpponentModel,
    our_eval_cp: i32,
    fullmove: u16,
    prev: Mode,
) -> Mode {
    let phase = if fullmove <= 12 {
        GamePhase::Opening
    } else if fullmove > 40 {
        GamePhase::Endgame
    } else {
        GamePhase::Middlegame
    };
    decide_persona(
        cfg,
        model,
        PersonaContext {
            eval_cp: our_eval_cp,
            previous_eval_cp: None,
            fullmove,
            in_check: false,
            legal_moves: 20,
            phase,
            both_queens: true,
        },
        prev,
    )
    .mode
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
    if cfg.limit_strength {
        return cfg.elo_cap.clamp(MIN_TRACKED_ELO, ENGINE_CEILING);
    }

    // Unknown agent type is handled conservatively: use the upper credible
    // strength bound until human evidence is strong enough. This avoids
    // weakening against a GM/engine merely because the population prior is
    // centered near 1500. Confident humans use the competitive +60 target.
    let target = if model.confident_human() {
        let live = model.estimate() + model.match_offset();
        // A human declaration remains untrusted but protects titled/high-rated
        // players from being grossly underplayed during the cold start.
        model
            .declared_elo
            .map(|declared| live.max(declared - 200))
            .unwrap_or(live)
    } else {
        model.upper_bound().max(model.estimate() + 60)
    };
    target
        .min(cfg_effective_cap(cfg))
        .clamp(MIN_TRACKED_ELO, MAX_TRACKED_ELO)
}

/// Draw score for the search (contempt wiring): when we are chasing a win a
/// draw is mildly bad for us; when we are defending, a draw is a rescue and
/// must not be repelled.
pub fn draw_score_for(cfg: &AdaptConfig, mode: Mode) -> i32 {
    if !cfg.adaptive || cfg.limit_strength || matches!(mode, Mode::Full | Mode::Defend) {
        return 0;
    }
    match mode {
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
            let own_castle = if let Color::White = us {
                WK | WQ
            } else {
                BK | BQ
            };
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

/// Human ACPL target fitted in the research audit. Targets at/above the
/// engine ceiling receive full-strength selection rather than a fake rating.
fn human_target_acpl(target: i32) -> f64 {
    if target >= ENGINE_CEILING {
        0.0
    } else {
        300.0 * (-(target as f64) / 900.0).exp()
    }
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
                let preferred = if far_ahead {
                    is_cap
                } else {
                    is_cap || gives_check
                };
                if preferred {
                    pick = l.mv;
                    why = if is_cap {
                        "forcing capture"
                    } else {
                        "forcing check"
                    };
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
            let both_queens_now = pos.bb[0][QUEEN] != 0 && pos.bb[1][QUEEN] != 0;
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
                let gap = match replies.len() {
                    // A genuinely forced reply is stronger trap/forcing
                    // evidence than any finite MultiPV gap. Keep it bounded
                    // so the root safety/loss budget remains authoritative.
                    1 => 300,
                    n if n >= 2 => (replies[0].score - replies[1].score).max(0),
                    _ => 0,
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
            let target_acpl = human_target_acpl(target);

            // The UCI layer supplies a target-dependent MultiPV root pool in
            // one shared iterative-deepening search. Every loss below is
            // therefore self-consistent and non-negative by construction.
            let viable: Vec<(Move, i32)> = lines
                .iter()
                .map(|line| (line.mv, line.score))
                .filter(|&(_, score)| {
                    let walks_into_mate =
                        crate::search::is_mate_score(score) && score < 0 && best.score > score;
                    !walks_into_mate
                        && !((best.score - score) > 0 && crate::search::is_mate_score(best.score))
                })
                .collect();
            if viable.is_empty() || target_acpl <= 0.0 {
                return Selection {
                    mv: best.mv,
                    reason: if viable.is_empty() {
                        "best move (no viable alternatives)".to_string()
                    } else {
                        "best move (at engine ceiling)".to_string()
                    },
                };
            }

            let candidate_moves: Vec<Move> = viable.iter().map(|&(mv, _)| mv).collect();
            let priors = prior.priors(pos, &candidate_moves, target);

            // Human errors are zero-inflated and heavy-tailed. Draw an intended
            // loss from a lognormal with the fitted mean ACPL, then choose the
            // legal move whose measured loss best matches it. Unlike the old
            // [0.25*max,max] band, this always has a nearest candidate and can
            // never silently fall through to stronger play.
            const LOGNORMAL_SIGMA: f64 = 1.1;
            let mu = target_acpl.ln() - LOGNORMAL_SIGMA * LOGNORMAL_SIGMA / 2.0;
            let intended = (mu + LOGNORMAL_SIGMA * rng.normal()).exp();
            let kernel_sigma = (target_acpl * 0.35).max(12.0);
            let mut weights = Vec::with_capacity(viable.len());
            for (index, &(_, score)) in viable.iter().enumerate() {
                let loss = (best.score - score).max(0) as f64;
                let distance = (loss - intended) / kernel_sigma;
                let kernel = (-0.5 * distance * distance).exp();
                weights.push(kernel * priors.get(index).copied().unwrap_or(1.0).max(0.005));
            }
            let total: f64 = weights.iter().sum();
            if total <= 0.0 || !total.is_finite() {
                let (mv, score) = viable
                    .iter()
                    .min_by_key(|(_, score)| {
                        ((best.score - *score).max(0) as f64 - intended).abs() as i64
                    })
                    .copied()
                    .unwrap_or((best.mv, best.score));
                return Selection {
                    mv,
                    reason: format!(
                        "human error target at ~{} (intended {:.0} cp, realised {} cp)",
                        target,
                        intended,
                        (best.score - score).max(0)
                    ),
                };
            }

            let mut roll = rng.f64() * total;
            for (&(mv, score), weight) in viable.iter().zip(&weights) {
                roll -= weight;
                if roll <= 0.0 {
                    return Selection {
                        mv,
                        reason: format!(
                            "human error target at ~{} (intended {:.0} cp, realised {} cp)",
                            target,
                            intended,
                            (best.score - score).max(0)
                        ),
                    };
                }
            }
            Selection {
                mv: best.mv,
                reason: "best move (sampling fallback)".to_string(),
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
        assert_eq!(target_elo(&AdaptConfig::default(), &m), ENGINE_CEILING);
    }

    #[test]
    fn persona_hysteresis_latches() {
        let cfg = AdaptConfig::default();
        let mut m = OpponentModel::new();
        m.seed_from_uci_opponent("- 1500 human TestPlayer");
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
        // an opponent pinned at our measurement ceiling for many moves is an
        // engine even if we never see their clock
        let mut m = OpponentModel::new();
        for _ in 0..10 {
            m.observe(5, 1.0);
        }
        assert!(m.estimate() >= 2450, "estimate {}", m.estimate());
        assert!(m.engine_suspect());
        // ...but a merely-good erratic human is not flagged
        let mut h = OpponentModel::new();
        for i in 0..10 {
            h.observe(if i % 3 == 0 { 150 } else { 20 }, 1.0);
        }
        assert!(!h.engine_suspect(), "estimate {}", h.estimate());
    }

    #[test]
    fn timing_regularities_only_modulate_ceiling_strength() {
        let mut weak_regular = OpponentModel::new();
        let mut strong_regular = OpponentModel::new();
        let mut strong_irregular = OpponentModel::new();
        for i in 0..7 {
            let smooth = 800 + i * 120;
            weak_regular.observe(220, 1.0);
            weak_regular.observe_time_fraction(smooth, 60_000, true);
            strong_regular.observe(5, 1.0);
            strong_regular.observe_time_fraction(smooth, 60_000, true);
            strong_irregular.observe(5, 1.0);
            strong_irregular.observe_time_fraction(
                if i % 2 == 0 { 80 } else { 5_000 },
                60_000,
                true,
            );
        }
        assert!(weak_regular.timing_autocorrelation().unwrap() >= 0.45);
        assert!(
            !weak_regular.engine_suspect(),
            "timing cannot flag weak play"
        );
        assert!(strong_regular.engine_suspect());
        assert!(
            !strong_irregular.engine_suspect(),
            "premoves alone are not engine evidence"
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
            Line {
                mv: mvs[0],
                score: 20,
                depth: 8,
                pv: vec![mvs[0]],
            },
            Line {
                mv: mvs[1],
                score: -(MATE - 6),
                depth: 8,
                pv: vec![mvs[1]],
            },
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
                &pos,
                &lines,
                Mode::Match,
                &cfg,
                &m,
                &prior,
                &mut rng,
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
    fn contextual_persona_responds_to_board_opponent_and_eval_trajectory() {
        let cfg = AdaptConfig::default();
        let mut human = OpponentModel::new();
        human.seed_from_uci_opponent("GM 2400 human TestGM");
        let base = PersonaContext {
            eval_cp: 20,
            previous_eval_cp: Some(10),
            fullmove: 32,
            in_check: false,
            legal_moves: 25,
            phase: GamePhase::Middlegame,
            both_queens: true,
        };
        assert_eq!(
            decide_persona(&cfg, &human, base, Mode::Match).mode,
            Mode::Clinch
        );
        assert_eq!(
            decide_persona(
                &cfg,
                &human,
                PersonaContext {
                    eval_cp: 180,
                    previous_eval_cp: Some(0),
                    ..base
                },
                Mode::Match,
            )
            .mode,
            Mode::Punish
        );
        assert_eq!(
            decide_persona(
                &cfg,
                &human,
                PersonaContext {
                    eval_cp: -220,
                    in_check: true,
                    ..base
                },
                Mode::Match,
            )
            .mode,
            Mode::Defend
        );
        assert_eq!(
            decide_persona(
                &cfg,
                &human,
                PersonaContext {
                    both_queens: false,
                    ..base
                },
                Mode::Match,
            )
            .mode,
            Mode::Match
        );

        let unknown = OpponentModel::new();
        assert_eq!(
            decide_persona(&cfg, &unknown, base, Mode::Match).mode,
            Mode::Match
        );
    }

    #[test]
    fn contextual_engine_response_keeps_full_strength_but_defends_when_worse() {
        let cfg = AdaptConfig::default();
        let mut engine = OpponentModel::new();
        engine.seed_from_uci_opponent("- - computer Stockfish");
        let neutral = PersonaContext {
            eval_cp: 0,
            previous_eval_cp: Some(10),
            fullmove: 20,
            in_check: false,
            legal_moves: 30,
            phase: GamePhase::Middlegame,
            both_queens: true,
        };
        assert_eq!(
            decide_persona(&cfg, &engine, neutral, Mode::Match).mode,
            Mode::Full
        );
        assert_eq!(
            decide_persona(
                &cfg,
                &engine,
                PersonaContext {
                    eval_cp: -300,
                    in_check: true,
                    ..neutral
                },
                Mode::Full,
            )
            .mode,
            Mode::Defend
        );
    }

    #[test]
    fn adaptive_match_offset_is_zero_mean_over_cycle() {
        let mut model = OpponentModel::new();
        let mut offsets = Vec::new();
        for samples in 0..4 {
            model.samples = samples;
            offsets.push(model.match_offset());
        }
        assert_eq!(offsets.iter().sum::<i32>(), 0, "{:?}", offsets);
        assert!(offsets.iter().any(|offset| *offset < 0));
        assert!(offsets.iter().any(|offset| *offset > 0));
    }

    #[test]
    fn human_acpl_curve_matches_research_reference() {
        for (elo, expected) in [
            (500, 172.1),
            (800, 123.3),
            (1200, 79.1),
            (2000, 32.5),
            (2400, 20.8),
        ] {
            assert!((human_target_acpl(elo) - expected).abs() < 0.2);
        }
        assert_eq!(human_target_acpl(ENGINE_CEILING), 0.0);
        assert_eq!(human_target_acpl(3650), 0.0);
        let above = AdaptConfig {
            adaptive: false,
            limit_strength: true,
            elo_cap: 3650,
            contempt: 0,
        };
        assert_eq!(target_elo(&above, &OpponentModel::new()), ENGINE_CEILING);
    }

    #[test]
    fn lower_tail_estimator_reads_beginner_errors_and_recovers() {
        let mut model = OpponentModel::new();
        for _ in 0..7 {
            model.observe(0, 1.0);
        }
        for _ in 0..3 {
            model.observe(300, 1.0);
        }
        assert!(model.estimate() < 1000, "estimate {}", model.estimate());
        for _ in 0..32 {
            model.observe(5, 1.0);
        }
        assert!(
            model.estimate() > 2400,
            "recovered estimate {}",
            model.estimate()
        );
    }

    #[test]
    fn match_error_magnitude_tracks_target_and_ceiling_is_best() {
        let pos = crate::fen::startpos();
        let moves = legal(&pos);
        let lines: Vec<Line> = moves
            .as_slice()
            .iter()
            .enumerate()
            .map(|(index, &mv)| Line {
                mv,
                score: 500 - index as i32 * 30,
                depth: 8,
                pv: vec![mv],
            })
            .collect();
        let model = OpponentModel::new();
        let mut total_loss = 0i64;
        for seed in 1..=2_000 {
            let cfg = AdaptConfig {
                adaptive: false,
                limit_strength: true,
                elo_cap: 800,
                contempt: 0,
            };
            let mut rng = Rng::new(seed);
            let selected = select_move(
                &pos,
                &lines,
                Mode::Match,
                &cfg,
                &model,
                &HeuristicPrior,
                &mut rng,
                &mut |_| Vec::new(),
            );
            let index = moves
                .as_slice()
                .iter()
                .position(|&mv| mv == selected.mv)
                .unwrap();
            total_loss += (index as i64) * 30;
        }
        let realised = total_loss as f64 / 2_000.0;
        assert!(
            (realised - human_target_acpl(800)).abs() < 30.0,
            "realised {}",
            realised
        );

        let ceiling_cfg = AdaptConfig {
            adaptive: false,
            limit_strength: true,
            elo_cap: ENGINE_CEILING,
            contempt: 0,
        };
        for seed in 1..100 {
            let mut rng = Rng::new(seed);
            let selected = select_move(
                &pos,
                &lines,
                Mode::Match,
                &ceiling_cfg,
                &model,
                &HeuristicPrior,
                &mut rng,
                &mut |_| Vec::new(),
            );
            assert_eq!(selected.mv, lines[0].mv);
        }
    }

    #[test]
    fn fixed_strength_has_absolute_precedence_over_personas() {
        let cfg = AdaptConfig {
            adaptive: true,
            limit_strength: true,
            elo_cap: 2400,
            contempt: 25,
        };
        let mut model = OpponentModel::new();
        model.observe(250, 1.0); // fresh blunder would normally trigger PUNISH
        assert_eq!(target_elo(&cfg, &model), 2400);
        for (eval, move_no) in [(-900, 10), (0, 35), (900, 10)] {
            assert_eq!(
                decide_mode(&cfg, &model, eval, move_no, Mode::Clinch),
                Mode::Match
            );
        }
        assert_eq!(draw_score_for(&cfg, Mode::Clinch), 0);
    }

    #[test]
    fn opponent_identity_survives_new_game_and_locks_auto_troll() {
        let mut model = OpponentModel::new();
        model.seed_from_uci_opponent("GM 1500 computer Stockfish 16");
        assert_eq!(
            model.estimate(),
            1500,
            "declared limited strength is distinct from identity"
        );
        assert_eq!(model.classification(), "known engine");
        assert!(model.anti_troll_lock());
        assert!(!model.auto_troll_allowed());
        let reset = model.reset_for_new_game();
        assert_eq!(reset.estimate(), 1500);
        assert_eq!(reset.classification(), "known engine");
        assert!(reset.engine_suspect());
        assert!(!reset.requires_full_strength());
        assert_eq!(
            decide_mode(&AdaptConfig::default(), &reset, 0, 10, Mode::Match),
            Mode::Match
        );

        let mut unrestricted = OpponentModel::new();
        unrestricted.seed_from_uci_opponent("GM - computer Stockfish 16");
        assert!(unrestricted.requires_full_strength());
        assert_eq!(
            decide_mode(&AdaptConfig::default(), &unrestricted, 0, 10, Mode::Match,),
            Mode::Full
        );
    }

    #[test]
    fn reseeding_rebuilds_evidence_instead_of_leaking_old_opponent() {
        let mut model = OpponentModel::new();
        model.seed_from_uci_opponent("GM - computer Stockfish");
        model.seed_from_uci_opponent("- - human Alice");
        assert_eq!(model.estimate(), 1500);
        assert!(!model.is_computer);
        assert_eq!(model.samples, 0);
        assert!(!model.engine_suspect());
    }

    #[test]
    fn stable_play_is_not_misclassified_as_erratic() {
        let mut model = OpponentModel::new();
        for _ in 0..10 {
            model.observe(10, 1.0);
        }
        assert!(model.volatility() < 50, "volatility {}", model.volatility());
        assert!(
            !model.trend().contains("erratic"),
            "trend {}",
            model.trend()
        );
    }

    #[test]
    fn changing_and_erratic_profiles_remain_distinct() {
        let mut stable = OpponentModel::new();
        let mut erratic = OpponentModel::new();
        let mut improving = OpponentModel::new();
        for i in 0..20 {
            stable.observe(10, 1.0);
            erratic.observe(if i % 2 == 0 { 0 } else { 300 }, 1.0);
            improving.observe(300 - i * 15, 1.0);
        }
        assert!(erratic.trend().contains("erratic"), "{}", erratic.trend());
        assert!(erratic.confidence() > stable.confidence());
        assert_eq!(improving.trend(), "trending up");
    }

    #[test]
    fn engine_classification_latches_for_the_game() {
        let mut model = OpponentModel::new();
        for _ in 0..10 {
            model.observe(5, 1.0);
        }
        assert!(model.engine_suspect());
        for _ in 0..12 {
            model.observe(120, 1.0);
        }
        assert!(
            model.engine_suspect(),
            "classification must not flap within one game"
        );
        assert!(!model.reset_for_new_game().engine_suspect());
    }

    #[test]
    fn blunder_freshness_can_be_consumed() {
        let mut model = OpponentModel::new();
        model.observe(220, 1.0);
        assert!(model.last_was_blunder());
        model.mark_decision_complete();
        assert!(!model.last_was_blunder());
        assert_eq!(model.last_cp_loss, Some(220));
    }

    #[test]
    fn rating_posterior_is_normalized_and_bounded() {
        let mut model = OpponentModel::new();
        for _ in 0..8 {
            model.observe(200, 1.0);
        }
        let sum: f64 = model.rating_probabilities().iter().sum();
        assert_eq!(model.rating_probabilities().len(), 3_551);
        assert!((sum - 1.0).abs() < 1e-9, "posterior sum {}", sum);
        assert_eq!(model.rating_probability(99), 0.0);
        assert_eq!(model.rating_probability(3651), 0.0);
        assert!((100..=3650).contains(&model.most_likely_elo()));
        assert!(model.lower_bound() <= model.estimate());
        assert!(model.estimate() <= model.upper_bound());
        assert!((100..=3650).contains(&model.lower_bound()));
        assert!((100..=3650).contains(&model.upper_bound()));
    }

    #[test]
    fn auto_troll_requires_positive_human_evidence() {
        let mut model = OpponentModel::new();
        assert!(!model.auto_troll_allowed());
        for _ in 0..6 {
            model.observe(240, 1.0);
        }
        assert!(model.human_probability() >= 0.90);
        assert!(model.auto_troll_allowed());
        assert!(target_elo(&AdaptConfig::default(), &model) < 1200);
        assert_eq!(
            decide_mode(&AdaptConfig::default(), &model, 300, 12, Mode::Match),
            Mode::Punish
        );
    }

    #[test]
    fn full_strength_has_neutral_draw_score() {
        let cfg = AdaptConfig::default();
        assert_eq!(draw_score_for(&cfg, Mode::Full), 0);
        assert_eq!(draw_score_for(&cfg, Mode::Defend), 0);
    }

    #[test]
    fn clinch_recognizes_a_single_legal_reply_as_forcing() {
        let pos = crate::fen::startpos();
        let moves = legal(&pos);
        let lines = vec![
            Line {
                mv: moves.moves[0],
                score: 100,
                depth: 8,
                pv: vec![moves.moves[0]],
            },
            Line {
                mv: moves.moves[1],
                score: 95,
                depth: 8,
                pv: vec![moves.moves[1]],
            },
        ];
        let mut calls = 0;
        let mut probe = |after: &Position| {
            calls += 1;
            let replies = legal(after);
            if calls == 2 {
                vec![Line {
                    mv: replies.moves[0],
                    score: 100,
                    depth: 4,
                    pv: vec![],
                }]
            } else {
                vec![
                    Line {
                        mv: replies.moves[0],
                        score: 100,
                        depth: 4,
                        pv: vec![],
                    },
                    Line {
                        mv: replies.moves[1],
                        score: 100,
                        depth: 4,
                        pv: vec![],
                    },
                ]
            }
        };
        let mut rng = Rng::new(7);
        let selected = select_move(
            &pos,
            &lines,
            Mode::Clinch,
            &AdaptConfig::default(),
            &OpponentModel::new(),
            &HeuristicPrior,
            &mut rng,
            &mut probe,
        );
        assert_eq!(selected.mv, moves.moves[1]);
        assert!(selected.reason.contains("300 cp"));
    }

    #[test]
    fn match_uses_only_common_depth_root_pool() {
        let pos = crate::fen::startpos();
        let moves = legal(&pos);
        let lines: Vec<Line> = moves
            .as_slice()
            .iter()
            .take(5)
            .enumerate()
            .map(|(i, &mv)| Line {
                mv,
                score: 100 - i as i32 * 10,
                depth: 8,
                pv: vec![mv],
            })
            .collect();
        let cfg = AdaptConfig {
            adaptive: false,
            limit_strength: true,
            elo_cap: 800,
            contempt: 0,
        };
        let mut probe_calls = 0;
        for seed in 0..100 {
            let mut rng = Rng::new(seed);
            let mut probe = |_after: &Position| {
                probe_calls += 1;
                Vec::new()
            };
            let selected = select_move(
                &pos,
                &lines,
                Mode::Match,
                &cfg,
                &OpponentModel::new(),
                &HeuristicPrior,
                &mut rng,
                &mut probe,
            );
            assert!(!selected.reason.contains("loss -"), "{}", selected.reason);
        }
        assert_eq!(probe_calls, 0, "MATCH must not run mixed-depth side probes");
    }
}
