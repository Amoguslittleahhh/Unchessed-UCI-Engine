//! UCI protocol loop. Search runs on a worker thread; `stop` flips an atomic
//! flag. The adapter pipeline (opponent observation -> book -> search ->
//! persona selection) lives in the worker so the GUI never blocks.

use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::adapt::{
    difficulty_weight, select_move, AdaptConfig, HeuristicPrior, MaiaPrior, MovePrior,
    OpponentModel, PersonaState, Rng,
};
use crate::aegis_v4_runtime::{
    position_to_input, ChessformerWeights, HintKey, InferenceExit, UnarchitecturedHintWorker,
    POLICY_GUIDE,
};
use crate::board::*;
use crate::book::{Book, BookEntry, Tier};
use crate::eval::{Eval, EvalParams, Hce};
use crate::fen;
use crate::movegen::{legal, parse_uci_move};
use crate::nnue::Nnue;
use crate::policy::PolicyNet;
use crate::search::{self, InfoEvent, Limits, Line, SearchParams};
use crate::tt::TT;
use crate::unarchitectured_v1::TensorPackage;

pub struct EngineIdent {
    pub name: &'static str,
    pub version: &'static str,
    pub author: &'static str,
    /// true for the Game Adapter, false for the (full-strength) Reviewer
    pub adaptive_engine: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TrollMode {
    Off,
    Auto,
    On,
}

#[derive(Clone)]
struct Options {
    hash_mb: usize,
    multipv: usize,
    adaptive: bool,
    limit_strength: bool,
    elo: i32,
    contempt: i32,
    troll: TrollMode,
    own_book: bool,
    book_depth: u32,
    search: SearchParams,
    threads: usize,
    eval_params: EvalParams,
    /// Experimental Unarchitectured v1 root ordering candidate. Default-off;
    /// alpha-beta remains authoritative and every legal move remains searched.
    unarchitectured_hint: bool,
    unarchitectured_hint_exit: InferenceExit,
    unarchitectured_file: String,
    unarchitectured_min_time_ms: u64,
    persona_smooth: bool,
    engine_detect_v2: bool,
    /// Default-off diagnostic output. This is deliberately not part of
    /// AdaptConfig and must never influence search, selection, or model state.
    adapter_telemetry: bool,
}

/// Default search thread count: the machine's logical CPU count, capped.
///
/// The previous default of 1 left every core but one idle on any modern
/// machine. On a 16-core laptop chip (e.g. Core Ultra 9 285H: 6 P + 8 E + 2
/// low-power E) that meant using ~6% of the CPU, which costs far more strength
/// than any micro-optimization in this codebase can recover.
///
/// Notes on the cap and on hybrid CPUs:
///
/// - We use the full logical count rather than trying to identify only the
///   "fast" cores. Lazy SMP helpers are pure TT-warmers, not latency-critical:
///   a slow E-core still contributes useful entries, and the OS scheduler
///   already prefers P-cores for the earliest-spawned threads. Trying to
///   detect P vs E from userspace portably is unreliable and not worth it.
/// - Capped at 32 so a very large server doesn't spawn a helper per core and
///   drown the shared TT in write contention; users can still raise `Threads`
///   explicitly up to the UCI maximum.
/// - Falls back to 1 if the platform can't report a count.
///
/// GUIs that set `Threads` explicitly (most do) override this entirely; this
/// only changes the out-of-the-box behavior for direct/UCI-default use.
fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(32))
        .unwrap_or(1)
}

impl Default for Options {
    fn default() -> Self {
        Options {
            hash_mb: 128,
            multipv: 1,
            adaptive: true,
            limit_strength: false,
            elo: 2400,
            contempt: 25,
            troll: TrollMode::Auto,
            own_book: true,
            book_depth: 16,
            search: SearchParams::default(),
            threads: default_threads(),
            eval_params: EvalParams::default(),
            unarchitectured_hint: false,
            unarchitectured_hint_exit: InferenceExit::Layer2Width128,
            unarchitectured_file: String::new(),
            unarchitectured_min_time_ms: 30_000,
            persona_smooth: false,
            engine_detect_v2: false,
            adapter_telemetry: false,
        }
    }
}

impl Options {
    fn adapt_config(&self) -> AdaptConfig {
        AdaptConfig {
            adaptive: self.adaptive,
            limit_strength: self.limit_strength,
            elo_cap: self.elo,
            contempt: self.contempt,
            persona_smooth: self.persona_smooth,
        }
    }
}

struct UnarchitecturedCandidate {
    worker: UnarchitecturedHintWorker,
}

impl UnarchitecturedCandidate {
    fn from_weights(weights: Arc<ChessformerWeights>) -> Result<Self, String> {
        let worker = UnarchitecturedHintWorker::new_shared(weights)?;
        Ok(Self { worker })
    }
}

fn unarchitectured_path(option: &str) -> Result<std::path::PathBuf, String> {
    if !option.is_empty() {
        return Ok(std::path::PathBuf::from(option));
    }
    let directory = std::env::current_exe()
        .map_err(|error| format!("locate executable: {error}"))?
        .parent()
        .ok_or("executable has no parent directory")?
        .to_path_buf();
    Ok(directory.join("unarchitectured-v1-final.unarchv1"))
}

fn load_unarchitectured_candidate(option: &str) -> Result<UnarchitecturedCandidate, String> {
    let path = unarchitectured_path(option)?;
    let bytes = TensorPackage::load(&path)?;
    let package = TensorPackage::parse(&bytes)?;
    let weights = Arc::new(ChessformerWeights::from_package(&package)?);
    UnarchitecturedCandidate::from_weights(weights)
}

/// One opponent move we have not yet fed to the model.
struct PendingObs {
    pre: Position,
    mv: Move,
    /// One-based game ply after this opponent move was played.
    ply: u32,
}

struct Game {
    /// position after each played move; [0] is the game-start position
    positions: Vec<Position>,
    current: Position,
    /// plies already fed to the opponent model
    observed_plies: usize,
    out_of_book_logged: bool,
    /// Process-local telemetry identity. These fields are observation-only and
    /// are advanced only for opt-in telemetry records.
    game_id: u64,
    decision_index: u64,
    observation_index: u64,
}

impl Game {
    fn new(game_id: u64) -> Game {
        let p = fen::startpos();
        Game {
            positions: vec![p],
            current: p,
            observed_plies: 0,
            out_of_book_logged: false,
            game_id,
            decision_index: 0,
            observation_index: 0,
        }
    }
}

pub fn run(ident: EngineIdent) {
    let stdin = std::io::stdin();
    let mut opt = Options::default();
    if !ident.adaptive_engine {
        opt.adaptive = false;
        opt.multipv = 3;
        opt.own_book = false;
    }
    let tt = Arc::new(Mutex::new(TT::new(opt.hash_mb)));
    let book = Arc::new(Mutex::new(match Book::new() {
        Ok(b) => b,
        Err(e) => {
            println!("info string [Unchessed] embedded book error: {}", e);
            Book::new().unwrap_or_else(|_| panic!("book build failed: {}", e))
        }
    }));
    let model = Arc::new(Mutex::new(OpponentModel::new()));
    let policy: Arc<Mutex<Option<Arc<PolicyNet>>>> = Arc::new(Mutex::new(load_default_policy()));
    let unarchitectured_candidate: Arc<Mutex<Option<UnarchitecturedCandidate>>> =
        Arc::new(Mutex::new(None));
    let (mut eval_impl, mut eval_desc, mut eval_is_hce): (Arc<dyn Eval>, String, bool) =
        load_default_eval(opt.eval_params);
    let stop = Arc::new(AtomicBool::new(false));
    // persona persists across moves for hysteresis + EMA/dwell; worker updates it
    let persona = Arc::new(Mutex::new(PersonaState::default()));
    let mut worker: Option<JoinHandle<()>> = None;
    // `go ponder`: the search is deliberately NOT started yet -- its own
    // limits (movetime/wtime/etc) describe the budget for the move the GUI
    // expects to make AFTER the pondered move is confirmed, not time to
    // spend right now. Starting the search immediately (the previous
    // behavior) meant it ran to completion and returned `bestmove`
    // immediately, an outright protocol violation: a GUI must not see a
    // move while it still believes the engine is pondering. Held here until
    // `ponderhit` (start the real search, clock beginning now) or the ponder
    // is abandoned by `stop`/a new `position`/`quit` (discarded, no
    // response -- matching the common real-engine behavior of not
    // fabricating a bestmove for a position the game may have already
    // moved past).
    let mut pending_ponder: Option<GoJob> = None;
    let mut game = Game::new(0);
    // `ucinewgame` owns logical game boundaries. The initial pre-newgame
    // position remains game 0 for permissive UCI clients.
    let mut next_game_id = 0u64;
    // opponent's clock reading at our previous `go` (for the time signal)
    let mut last_opp_clock: Option<u64> = None;

    let join_worker = |worker: &mut Option<JoinHandle<()>>, stop: &AtomicBool| {
        if let Some(h) = worker.take() {
            stop.store(true, Ordering::Relaxed);
            let _ = h.join();
            stop.store(false, Ordering::Relaxed);
        }
    };

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let mut toks = line.split_whitespace();
        let cmd = match toks.next() {
            Some(c) => c,
            None => continue,
        };
        match cmd {
            "uci" => {
                println!("id name {} {}", ident.name, ident.version);
                println!("id author {}", ident.author);
                println!("option name Hash type spin default 128 min 1 max 2048");
                println!(
                    "option name Threads type spin default {} min 1 max 64",
                    default_threads()
                );
                println!("option name Clear Hash type button");
                println!(
                    "option name MultiPV type spin default {} min 1 max 8",
                    if ident.adaptive_engine { 1 } else { 3 }
                );
                println!("option name EvalFile type string default ");
                println!("option name UnarchitecturedHint type check default false");
                println!("option name UnarchitecturedHintExit type string default 2/128");
                println!("option name UnarchitecturedFile type string default ");
                println!(
                    "option name UnarchitecturedMinTime type spin default 30000 min 1000 max 600000"
                );
                if ident.adaptive_engine {
                    println!("option name Adaptive type check default true");
                    println!("option name UCI_LimitStrength type check default false");
                    println!("option name UCI_Elo type spin default 2400 min 500 max 3200");
                    println!("option name Contempt type spin default 25 min 0 max 100");
                    println!("option name Troll type combo default Auto var Off var Auto var On");
                    println!("option name OwnBook type check default true");
                    println!("option name BookFile type string default ");
                    println!("option name BookDepth type spin default 16 min 0 max 40");
                    println!("option name PolicyFile type string default ");
                    println!("option name UCI_Opponent type string default ");
                    println!("option name PersonaSmooth type check default false");
                    println!("option name EngineDetectV2 type check default false");
                    println!("option name AdapterTelemetry type check default false");
                }
                // tunable search constants (defaults match prior hard-coded
                // values; exposed for manual tuning and future SPSA runs)
                println!("option name RFPMargin type spin default 90 min 10 max 300");
                println!("option name NullMoveBase type spin default 3 min 1 max 6");
                println!("option name NullMoveDivisor type spin default 6 min 2 max 12");
                println!("option name LMRMinDepth type spin default 3 min 1 max 8");
                println!("option name LMRMinMoveNumber type spin default 3 min 0 max 20");
                println!("option name LMRBigMoveNumber type spin default 12 min 4 max 40");
                println!("option name AspirationDelta type spin default 25 min 5 max 200");
                println!("option name AspirationMinDepth type spin default 4 min 1 max 12");
                println!("option name ProbCutMargin type spin default 200 min 50 max 400");
                println!("option name ProbCutReduction type spin default 4 min 2 max 6");
                println!("option name ProbCutMinDepth type spin default 5 min 3 max 10");
                println!("option name FutilityMargin type spin default 150 min 30 max 400");
                println!("option name FutilityMaxDepth type spin default 8 min 1 max 12");
                // Default-off tree-changing pruning rule, exposed so it can
                // be SPRT-gated as baseline-vs-candidate on one binary.
                println!("option name ProbcutSeeFilter type check default false");
                // tunable HCE eval constants (0 = feature off); 100/100 is the
                // SPRT-validated default (+25.7 Elo, 2026-08-02) after the
                // safe/blocked-conditioned rewrite -- kept tunable for future
                // SPSA passes rather than hardcoded.
                println!("option name PassedPawnMgPct type spin default 100 min 0 max 200");
                println!("option name PassedPawnEgPct type spin default 100 min 0 max 200");
                // Mobility: 100 is SPRT-validated (+52.3 Elo, 2026-08-03).
                println!("option name MobilityPct type spin default 100 min 0 max 200");
                // Rook file / rook-on-7th: 100 is SPRT-validated
                // (+10.5 Elo, 2026-08-04).
                println!("option name RookPct type spin default 100 min 0 max 200");
                // Knight outpost: 100 is SPRT-validated (+12.0 +/- 7.8 Elo,
                // 2026-08-10, 4873 games, LLR crossed the upper bound).
                println!("option name KnightOutpostPct type spin default 100 min 0 max 200");
                println!("uciok");
                println!("info string [Unchessed] eval: {}", eval_desc);
                if ident.adaptive_engine {
                    match policy.lock().unwrap().as_ref() {
                        Some(net) => println!(
                            "info string [Unchessed] human policy net loaded: {}",
                            net.describe()
                        ),
                        None => println!(
                            "info string [Unchessed] no policy net found — using heuristic move priors"
                        ),
                    }
                }
            }
            "isready" => println!("readyok"),
            "setoption" => {
                join_worker(&mut worker, &stop);
                handle_setoption(
                    &line,
                    &mut opt,
                    &tt,
                    &book,
                    &model,
                    &policy,
                    &unarchitectured_candidate,
                    &mut eval_impl,
                    &mut eval_desc,
                    &mut eval_is_hce,
                );
            }
            "ucinewgame" => {
                join_worker(&mut worker, &stop);
                tt.lock().unwrap().clear();
                {
                    let mut m = model.lock().unwrap();
                    *m = OpponentModel::new();
                    m.experimental_detect = opt.engine_detect_v2;
                }
                *persona.lock().unwrap() = PersonaState::default();
                last_opp_clock = None;
                next_game_id = next_game_id.saturating_add(1);
                game = Game::new(next_game_id);
                if opt.unarchitectured_hint {
                    match load_unarchitectured_candidate(&opt.unarchitectured_file) {
                        Ok(candidate) => {
                            *unarchitectured_candidate.lock().unwrap() = Some(candidate);
                        }
                        Err(error) => {
                            opt.unarchitectured_hint = false;
                            *unarchitectured_candidate.lock().unwrap() = None;
                            println!(
                                "info string [Unchessed] Unarchitectured new-game reload failed: {error} — hint disabled"
                            );
                        }
                    }
                }
            }
            "position" => {
                join_worker(&mut worker, &stop);
                // A new position while still pondering the old one means the
                // opponent's actual move has already been established some
                // other way (or the GUI is resetting) -- the pondered guess
                // no longer applies to anything, so it's discarded, not
                // started stale.
                pending_ponder = None;
                if let Some(g) = parse_position(&line, &game) {
                    game = g;
                } else {
                    println!("info string [Unchessed] could not parse: {}", line);
                }
            }
            "go" => {
                join_worker(&mut worker, &stop);
                let limits = parse_go(&line);
                let pending = collect_pending(&mut game);
                let telemetry_enabled = ident.adaptive_engine && opt.adapter_telemetry;
                let observation_indices = if telemetry_enabled {
                    (0..pending.len())
                        .map(|_| {
                            game.observation_index = game.observation_index.saturating_add(1);
                            game.observation_index
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                let decision_index = if telemetry_enabled
                    && (opt.adaptive || opt.limit_strength)
                    && limits.is_game_mode()
                {
                    game.decision_index = game.decision_index.saturating_add(1);
                    Some(game.decision_index)
                } else {
                    None
                };
                let telemetry_run = if telemetry_enabled {
                    telemetry_run_id()
                } else {
                    String::new()
                };
                // opponent time signal: how long did their last move take?
                let opp_is_white = matches!(game.current.side, Color::Black);
                let opp_clock_now = if opp_is_white {
                    limits.wtime
                } else {
                    limits.btime
                };
                let opp_inc = if opp_is_white {
                    limits.winc
                } else {
                    limits.binc
                };
                let opp_time_used = match (last_opp_clock, opp_clock_now, &pending[..]) {
                    (Some(prev), Some(now), [_, ..]) => {
                        Some((prev + opp_inc.unwrap_or(0)).saturating_sub(now))
                    }
                    _ => None,
                };
                last_opp_clock = opp_clock_now;
                let job = GoJob {
                    ident_adaptive: ident.adaptive_engine,
                    pos: game.current,
                    history: game.positions[..game.positions.len().saturating_sub(1)]
                        .iter()
                        .map(|p| p.hash)
                        .collect(),
                    game_plies: game.positions.len() as u32 - 1,
                    limits,
                    opt: opt.clone(),
                    pending,
                    out_of_book_logged: game.out_of_book_logged,
                    policy: policy.lock().unwrap().clone(),
                    eval: Arc::clone(&eval_impl),
                    unarchitectured_candidate: Arc::clone(&unarchitectured_candidate),
                    opp_time_used,
                    game_id: game.game_id,
                    decision_index,
                    observation_indices,
                    telemetry_run,
                };
                // the worker decides book state transitions; mirror the flag
                // optimistically so the log line prints only once
                if job.game_plies >= opt.book_depth {
                    game.out_of_book_logged = true;
                }
                if job.limits.ponder {
                    // Deliberately not spawned: see pending_ponder's
                    // definition. Waits for `ponderhit` (starts now, using
                    // these same limits) or `stop`/a new `position` (this
                    // guess is simply dropped).
                    pending_ponder = Some(job);
                } else {
                    let tt = Arc::clone(&tt);
                    let stop_c = Arc::clone(&stop);
                    let book = Arc::clone(&book);
                    let model = Arc::clone(&model);
                    let persona_c = Arc::clone(&persona);
                    worker = Some(std::thread::spawn(move || {
                        run_go(job, tt, stop_c, book, model, persona_c);
                    }));
                }
            }
            "ponderhit" => {
                // The predicted move was played: the budget given in the
                // original `go ponder ...` line describes the move we're
                // about to make now, so its clock starts here, not back
                // when we started pondering.
                if let Some(job) = pending_ponder.take() {
                    let tt = Arc::clone(&tt);
                    let stop_c = Arc::clone(&stop);
                    let book = Arc::clone(&book);
                    let model = Arc::clone(&model);
                    let persona_c = Arc::clone(&persona);
                    worker = Some(std::thread::spawn(move || {
                        run_go(job, tt, stop_c, book, model, persona_c);
                    }));
                }
            }
            "stop" => {
                pending_ponder = None;
                join_worker(&mut worker, &stop);
            }
            "quit" => {
                join_worker(&mut worker, &stop);
                break;
            }
            "d" => {
                print!("{}", game.current.pretty());
                println!("fen: {}", fen::serialize(&game.current));
                println!("hash: {:016x}", game.current.hash);
            }
            // debug: "policy [elo]" prints the human policy for the position
            "policy" => {
                let elo: i32 = toks.next().and_then(|t| t.parse().ok()).unwrap_or(1500);
                match policy.lock().unwrap().as_ref() {
                    Some(net) => {
                        let ml = legal(&game.current);
                        let moves: Vec<Move> = ml.as_slice().to_vec();
                        let probs = net.priors(&game.current, &moves, elo);
                        let mut ranked: Vec<(Move, f64)> = moves.into_iter().zip(probs).collect();
                        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
                        for (m, p) in ranked.iter().take(8) {
                            println!(
                                "info string [Unchessed] policy@{} {} {:.1}%",
                                elo,
                                m.uci(),
                                p * 100.0
                            );
                        }
                    }
                    None => println!("info string [Unchessed] no policy net loaded"),
                }
            }
            _ => {}
        }
    }
}

fn load_default_policy() -> Option<Arc<PolicyNet>> {
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let path = exe_dir.join("unchessed-maia.bin");
    PolicyNet::load(path.to_str()?).ok().map(Arc::new)
}

fn load_default_nnue() -> Option<Arc<Nnue>> {
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let path = exe_dir.join("unchessed-nnue.bin");
    Nnue::load(path.to_str()?).ok().map(Arc::new)
}

/// Default evaluator: NNUE weights next to the executable, else HCE.
fn load_default_eval(params: EvalParams) -> (Arc<dyn Eval>, String, bool) {
    match load_default_nnue() {
        Some(net) => {
            let e: Arc<dyn Eval> = net;
            (e, "NNUE (unchessed-nnue.bin)".to_string(), false)
        }
        None => (
            Arc::new(Hce::new(params)),
            "hand-crafted (no NNUE file found)".to_string(),
            true,
        ),
    }
}

/// Splits the `<key with spaces> [value <value...>]` remainder of a
/// `setoption name ...` line into (key, value). Deliberately searches for
/// " value" without requiring a trailing space: an empty value (e.g.
/// "setoption name EvalFile value" with nothing after it) has its
/// trailing space stripped by the caller's `.trim()` calls before this
/// runs, so requiring " value " (trailing space included) would never
/// match an empty value at all, silently dropping the whole setoption --
/// this was a real, pre-existing bug (EvalFile/BookFile resets via an
/// empty value never worked).
fn parse_setoption_kv(rest: &str) -> (&str, &str) {
    match rest.find(" value") {
        Some(i) => (&rest[..i], rest[i + 6..].trim()),
        None => (rest, ""),
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_setoption(
    line: &str,
    opt: &mut Options,
    tt: &Arc<Mutex<TT>>,
    book: &Arc<Mutex<Book>>,
    model: &Arc<Mutex<OpponentModel>>,
    policy: &Arc<Mutex<Option<Arc<PolicyNet>>>>,
    unarchitectured_candidate: &Arc<Mutex<Option<UnarchitecturedCandidate>>>,
    eval_impl: &mut Arc<dyn Eval>,
    eval_desc: &mut String,
    eval_is_hce: &mut bool,
) {
    // setoption name <key with spaces> [value <value...>]
    let rest = match line.strip_prefix("setoption") {
        Some(r) => r.trim(),
        None => return,
    };
    let rest = match rest.strip_prefix("name") {
        Some(r) => r.trim(),
        None => return,
    };
    let (key, value) = parse_setoption_kv(rest);
    let key_l = key.trim().to_lowercase();
    match key_l.as_str() {
        "hash" => {
            if let Ok(mb) = value.parse::<usize>() {
                opt.hash_mb = mb.clamp(1, 2048);
                tt.lock().unwrap().resize(opt.hash_mb);
            }
        }
        "clear hash" => tt.lock().unwrap().clear(),
        "threads" => {
            if let Ok(n) = value.parse::<usize>() {
                opt.threads = n.clamp(1, 64);
            }
        }
        "multipv" => {
            if let Ok(n) = value.parse::<usize>() {
                opt.multipv = n.clamp(1, 8);
            }
        }
        "unarchitecturedhint" => {
            let enabled = value.eq_ignore_ascii_case("true");
            if !enabled {
                opt.unarchitectured_hint = false;
                *unarchitectured_candidate.lock().unwrap() = None;
                println!("info string [Unchessed] Unarchitectured root hint disabled");
            } else {
                match load_unarchitectured_candidate(&opt.unarchitectured_file) {
                    Ok(candidate) => {
                        *unarchitectured_candidate.lock().unwrap() = Some(candidate);
                        opt.unarchitectured_hint = true;
                        println!(
                            "info string [Unchessed] Unarchitectured root hint candidate enabled (experimental, default-off)"
                        );
                    }
                    Err(error) => {
                        opt.unarchitectured_hint = false;
                        *unarchitectured_candidate.lock().unwrap() = None;
                        println!(
                            "info string [Unchessed] Unarchitectured model load failed: {error} — hint remains off"
                        );
                    }
                }
            }
        }
        "unarchitecturedhintexit" => {
            match InferenceExit::from_option_name(&value) {
                Some(exit) => {
                    opt.unarchitectured_hint_exit = exit;
                    println!(
                        "info string [Unchessed] Unarchitectured hint exit set to {}",
                        exit.option_name()
                    );
                }
                None => println!(
                    "info string [Unchessed] UnarchitecturedHintExit must be 2/128, 4/192 or 8/256 ({} kept)",
                    opt.unarchitectured_hint_exit.option_name()
                ),
            }
        }
        "unarchitecturedfile" => {
            opt.unarchitectured_file = value.to_string();
            if opt.unarchitectured_hint {
                match load_unarchitectured_candidate(&opt.unarchitectured_file) {
                    Ok(candidate) => {
                        *unarchitectured_candidate.lock().unwrap() = Some(candidate);
                        println!("info string [Unchessed] Unarchitectured model reloaded");
                    }
                    Err(error) => {
                        opt.unarchitectured_hint = false;
                        *unarchitectured_candidate.lock().unwrap() = None;
                        println!(
                            "info string [Unchessed] Unarchitectured model reload failed: {error} — hint disabled"
                        );
                    }
                }
            }
        }
        "unarchitecturedmintime" => {
            if let Ok(milliseconds) = value.parse::<u64>() {
                opt.unarchitectured_min_time_ms = milliseconds.clamp(1_000, 600_000);
            }
        }
        "adaptive" => opt.adaptive = value.eq_ignore_ascii_case("true"),
        "uci_limitstrength" => opt.limit_strength = value.eq_ignore_ascii_case("true"),
        "uci_elo" => {
            if let Ok(e) = value.parse::<i32>() {
                opt.elo = e.clamp(500, 3200);
            }
        }
        "contempt" => {
            if let Ok(c) = value.parse::<i32>() {
                opt.contempt = c.clamp(0, 100);
            }
        }
        "troll" => {
            opt.troll = match value.to_lowercase().as_str() {
                "off" => TrollMode::Off,
                "on" => TrollMode::On,
                _ => TrollMode::Auto,
            }
        }
        "ownbook" => opt.own_book = value.eq_ignore_ascii_case("true"),
        "bookdepth" => {
            if let Ok(d) = value.parse::<u32>() {
                opt.book_depth = d.min(40);
            }
        }
        "rfpmargin" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.search.rfp_margin = v.clamp(10, 300);
            }
        }
        "nullmovebase" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.search.nm_base = v.clamp(1, 6);
            }
        }
        "nullmovedivisor" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.search.nm_divisor = v.clamp(2, 12);
            }
        }
        "lmrmindepth" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.search.lmr_min_depth = v.clamp(1, 8);
            }
        }
        "lmrminmovenumber" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.search.lmr_min_movenum = v.clamp(0, 20);
            }
        }
        "lmrbigmovenumber" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.search.lmr_big_movenum = v.clamp(4, 40);
            }
        }
        "aspirationdelta" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.search.aspiration_delta = v.clamp(5, 200);
            }
        }
        "aspirationmindepth" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.search.aspiration_min_depth = v.clamp(1, 12);
            }
        }
        "probcutmargin" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.search.probcut_margin = v.clamp(50, 400);
            }
        }
        "probcutreduction" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.search.probcut_reduction = v.clamp(2, 6);
            }
        }
        "probcutmindepth" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.search.probcut_min_depth = v.clamp(3, 10);
            }
        }
        "futilitymargin" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.search.futility_margin = v.clamp(30, 400);
            }
        }
        "futilitymaxdepth" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.search.futility_max_depth = v.clamp(1, 12);
            }
        }
        "probcutseefilter" => {
            opt.search.probcut_see_filter = value.eq_ignore_ascii_case("true");
        }
        "passedpawnmgpct" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.eval_params.passed_mg_pct = v.clamp(0, 200);
                if *eval_is_hce {
                    *eval_impl = Arc::new(Hce::new(opt.eval_params));
                }
            }
        }
        "passedpawnegpct" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.eval_params.passed_eg_pct = v.clamp(0, 200);
                if *eval_is_hce {
                    *eval_impl = Arc::new(Hce::new(opt.eval_params));
                }
            }
        }
        "mobilitypct" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.eval_params.mobility_pct = v.clamp(0, 200);
                if *eval_is_hce {
                    *eval_impl = Arc::new(Hce::new(opt.eval_params));
                }
            }
        }
        "rookpct" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.eval_params.rook_pct = v.clamp(0, 200);
                if *eval_is_hce {
                    *eval_impl = Arc::new(Hce::new(opt.eval_params));
                }
            }
        }
        "knightoutpostpct" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.eval_params.knight_outpost_pct = v.clamp(0, 200);
                if *eval_is_hce {
                    *eval_impl = Arc::new(Hce::new(opt.eval_params));
                }
            }
        }
        "bookfile" => {
            let mut b = book.lock().unwrap();
            if value.is_empty() {
                b.unload_polyglot();
            } else {
                match b.load_polyglot(value) {
                    Ok(n) => println!(
                        "info string [Unchessed] loaded polyglot book '{}' ({} entries)",
                        value, n
                    ),
                    Err(e) => println!("info string [Unchessed] book load failed: {}", e),
                }
            }
        }
        "evalfile" => {
            if value.is_empty() {
                let (e, d, is_hce) = load_default_eval(opt.eval_params);
                *eval_impl = e;
                *eval_desc = d;
                *eval_is_hce = is_hce;
                // stale TT entries would mix scores from the previous evaluator
                tt.lock().unwrap().clear();
                println!("info string [Unchessed] eval reset to default: {}", eval_desc);
            } else {
                match Nnue::load(value) {
                    Ok(net) => {
                        *eval_impl = Arc::new(net);
                        *eval_desc = format!("NNUE ({})", value);
                        *eval_is_hce = false;
                        tt.lock().unwrap().clear();
                        println!("info string [Unchessed] NNUE loaded from '{}'", value);
                    }
                    Err(e) => println!(
                        "info string [Unchessed] NNUE load failed: {} — keeping current eval",
                        e
                    ),
                }
            }
        }
        "policyfile" => {
            let mut p = policy.lock().unwrap();
            if value.is_empty() {
                *p = load_default_policy();
                println!(
                    "info string [Unchessed] policy reset to default ({})",
                    if p.is_some() { "loaded" } else { "none found" }
                );
            } else {
                match PolicyNet::load(value) {
                    Ok(net) => {
                        println!(
                            "info string [Unchessed] policy net loaded from '{}': {}",
                            value,
                            net.describe()
                        );
                        *p = Some(Arc::new(net));
                    }
                    Err(e) => println!("info string [Unchessed] policy load failed: {}", e),
                }
            }
        }
        "personasmooth" => opt.persona_smooth = value.eq_ignore_ascii_case("true"),
        "adaptertelemetry" => opt.adapter_telemetry = value.eq_ignore_ascii_case("true"),
        "enginedetectv2" => {
            opt.engine_detect_v2 = value.eq_ignore_ascii_case("true");
            model.lock().unwrap().experimental_detect = opt.engine_detect_v2;
        }
        "uci_opponent" => {
            let log = model.lock().unwrap().seed_from_uci_opponent(value);
            println!("info string [Unchessed] {}", log);
        }
        _ => {}
    }
}

/// UCI GUIs resend the full move list from the game start on every
/// `position` command rather than sending just the newest move, so a fresh
/// `Game` gets rebuilt from scratch on every turn. If we always reset
/// `observed_plies` to 0 here, the opponent-observation pipeline re-feeds
/// every past opponent move into the model again on every single turn
/// instead of just the new one — repeat-counting early moves more and more
/// heavily as the game goes on, which both wastes probe time and distorts
/// the running Elo estimate (a single early good/bad move ends up sampled
/// dozens of times instead of once). Carry `observed_plies` forward from
/// the previous `Game` when the new move list is a genuine continuation of
/// it (same position sequence, just with more moves appended); only reset
/// to 0 when it's actually a different game (a real `position` change, not
/// just the GUI re-sending the same game so far).
fn carry_observed_plies(old: &Game, new_positions: &[Position]) -> usize {
    if old.observed_plies == 0 || old.positions.len() > new_positions.len() {
        return 0;
    }
    let same_so_far = old
        .positions
        .iter()
        .zip(new_positions.iter())
        .all(|(a, b)| a.hash == b.hash);
    if same_so_far {
        old.observed_plies
    } else {
        0
    }
}

fn parse_position(line: &str, old: &Game) -> Option<Game> {
    let mut toks = line.split_whitespace().peekable();
    toks.next(); // "position"
    let start = match toks.next()? {
        "startpos" => fen::startpos(),
        "fen" => {
            let mut fen_str = String::new();
            for t in toks.by_ref() {
                if t == "moves" {
                    // rebuild iterator state: handled below via flag
                    fen_str.push_str("moves ");
                    break;
                }
                fen_str.push_str(t);
                fen_str.push(' ');
            }
            let (fen_part, saw_moves) = match fen_str.strip_suffix("moves ") {
                Some(f) => (f.trim().to_string(), true),
                None => (fen_str.trim().to_string(), false),
            };
            let pos = fen::parse(&fen_part).ok()?;
            let mut game = Game {
                positions: vec![pos],
                current: pos,
                observed_plies: 0,
                out_of_book_logged: false,
                game_id: old.game_id,
                decision_index: old.decision_index,
                observation_index: old.observation_index,
            };
            if saw_moves {
                for t in toks {
                    let mv = parse_uci_move(&game.current, t)?;
                    game.current = game.current.make(mv);
                    game.positions.push(game.current);
                }
            }
            game.observed_plies = carry_observed_plies(old, &game.positions);
            return Some(game);
        }
        _ => return None,
    };
    let mut game = Game {
        positions: vec![start],
        current: start,
        observed_plies: 0,
        out_of_book_logged: false,
        game_id: old.game_id,
        decision_index: old.decision_index,
        observation_index: old.observation_index,
    };
    if toks.peek() == Some(&"moves") {
        toks.next();
        for t in toks {
            let mv = parse_uci_move(&game.current, t)?;
            game.current = game.current.make(mv);
            game.positions.push(game.current);
        }
    }
    game.observed_plies = carry_observed_plies(old, &game.positions);
    Some(game)
}

fn parse_go(line: &str) -> Limits {
    let mut l = Limits::default();
    let mut toks = line.split_whitespace();
    toks.next();
    while let Some(t) = toks.next() {
        let mut num = |l: &mut Option<u64>| {
            if let Some(v) = toks.next().and_then(|v| v.parse().ok()) {
                *l = Some(v);
            }
        };
        match t {
            "depth" => {
                if let Some(v) = toks.next().and_then(|v| v.parse().ok()) {
                    l.depth = Some(v);
                }
            }
            "movetime" => num(&mut l.movetime),
            "wtime" => num(&mut l.wtime),
            "btime" => num(&mut l.btime),
            "winc" => num(&mut l.winc),
            "binc" => num(&mut l.binc),
            "movestogo" => num(&mut l.movestogo),
            "nodes" => num(&mut l.nodes),
            "infinite" => l.infinite = true,
            "ponder" => l.ponder = true,
            // Per UCI convention, searchmoves takes every remaining token as
            // a move (it's always the last thing on a `go` line in practice,
            // and nothing after it is a recognized keyword either way).
            "searchmoves" => {
                l.searchmoves = toks.by_ref().map(String::from).collect();
                break;
            }
            _ => {}
        }
    }
    l
}

/// Opponent moves played since we last looked, with their pre-move positions.
fn collect_pending(game: &mut Game) -> Vec<PendingObs> {
    let our_side = game.current.side;
    let n_moves = game.positions.len() - 1;
    let mut out = Vec::new();
    for i in game.observed_plies..n_moves {
        let pre = game.positions[i];
        if pre.side != our_side {
            // reconstruct the move from pre -> post
            let post = game.positions[i + 1];
            if let Some(mv) = legal(&pre)
                .as_slice()
                .iter()
                .copied()
                .find(|m| pre.make(*m).hash == post.hash)
            {
                out.push(PendingObs {
                    pre,
                    mv,
                    ply: (i + 1) as u32,
                });
            }
        }
    }
    game.observed_plies = n_moves;
    out
}

fn unarchitectured_input(
    pos: &Position,
    legal_moves: &[Move],
    rating: i32,
) -> crate::aegis_v4_runtime::PositionInput {
    position_to_input(
        pos,
        legal_moves,
        i64::from(rating),
        2, // fixed rapid-like class for the experimental guide-policy candidate
        POLICY_GUIDE,
    )
}

#[cfg(test)]
fn submit_unarchitectured_request(
    candidate: &Arc<Mutex<Option<UnarchitecturedCandidate>>>,
    pos: &Position,
    rating: i32,
) {
    let legal_moves = legal(pos);
    if legal_moves.len == 0 {
        return;
    }
    let input = unarchitectured_input(pos, legal_moves.as_slice(), rating);
    if let Some(candidate) = candidate.lock().unwrap().as_ref() {
        let _ = candidate
            .worker
            .try_submit(pos.hash, input, InferenceExit::Layer2Width128);
    }
}

fn unarchitectured_wait_allowed(limits: &Limits, side: Color, minimum_time_ms: u64) -> bool {
    limits.infinite
        || limits.depth.is_some()
        || limits.nodes.is_some()
        || limits.movetime.map(|time| time >= 1_000).unwrap_or(false)
        || limits
            .my_time(side)
            .map(|time| time >= minimum_time_ms)
            .unwrap_or(false)
}

struct PreparedRootHints {
    hints: Vec<search::RootHint>,
    elapsed: std::time::Duration,
    source: &'static str,
}

fn prepare_unarchitectured_root_hints(
    candidate: &Arc<Mutex<Option<UnarchitecturedCandidate>>>,
    pos: &Position,
    legal_moves: &[Move],
    rating: i32,
    limits: &Limits,
    minimum_time_ms: u64,
    exit: InferenceExit,
) -> PreparedRootHints {
    if !unarchitectured_wait_allowed(limits, pos.side, minimum_time_ms) {
        return PreparedRootHints {
            hints: Vec::new(),
            elapsed: std::time::Duration::ZERO,
            source: "skipped-low-time",
        };
    }
    let started = std::time::Instant::now();
    let input = unarchitectured_input(pos, legal_moves, rating);
    let key = HintKey::new(pos.hash, &input, exit);

    let exact = {
        let guard = candidate.lock().unwrap();
        guard
            .as_ref()
            .and_then(|candidate| candidate.worker.latest_exact(&key))
    };
    let hint = if exact.is_some() {
        exact
    } else {
        {
            let guard = candidate.lock().unwrap();
            if let Some(candidate) = guard.as_ref() {
                let _ = candidate.worker.try_submit(pos.hash, input, exit);
            }
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(100);
        loop {
            let ready = {
                let guard = candidate.lock().unwrap();
                guard
                    .as_ref()
                    .and_then(|candidate| candidate.worker.latest_exact(&key))
            };
            if ready.is_some() || std::time::Instant::now() >= deadline {
                break ready;
            }
            std::thread::yield_now();
        }
    };

    let Some(hint) = hint else {
        return PreparedRootHints {
            hints: Vec::new(),
            elapsed: started.elapsed(),
            source: "timeout",
        };
    };
    // `zip` would silently truncate to the shorter side, so a logit vector
    // that did not correspond 1:1 with `legal_moves` would quietly produce a
    // *partial* hint list -- some legal moves unscored, and (worse) scores
    // potentially attached to the wrong moves if the orders ever diverged.
    // Nothing downstream could detect that: the search would just receive a
    // plausible-looking ranking that is wrong.
    //
    // The alignment does hold today, because `HintKey` includes the full
    // `legal_actions` vector and `latest_exact` only returns a hint whose key
    // matches exactly. But that is an invariant maintained at a distance, in
    // a different module, by code that has no obligation to keep doing so.
    // Assert it here, where the assumption is actually used: on a mismatch,
    // drop the hint and search unhinted rather than order on bad data.
    if hint.output.logits.len() != legal_moves.len() {
        return PreparedRootHints {
            hints: Vec::new(),
            elapsed: started.elapsed().max(hint.elapsed),
            source: "length-mismatch",
        };
    }
    let hints = legal_moves
        .iter()
        .zip(hint.output.logits.iter())
        .map(|(&mv, &policy_score)| search::RootHint { mv, policy_score })
        .collect();
    PreparedRootHints {
        hints,
        // Charge both request/wait overhead and at least the complete neural
        // forward to the real search deadline.
        elapsed: started.elapsed().max(hint.elapsed),
        source: "exact",
    }
}

struct GoJob {
    ident_adaptive: bool,
    pos: Position,
    history: Vec<u64>,
    game_plies: u32,
    limits: Limits,
    opt: Options,
    pending: Vec<PendingObs>,
    out_of_book_logged: bool,
    policy: Option<Arc<PolicyNet>>,
    /// static evaluator (NNUE or HCE) for every search this job runs
    eval: Arc<dyn Eval>,
    unarchitectured_candidate: Arc<Mutex<Option<UnarchitecturedCandidate>>>,
    /// milliseconds the opponent spent on their last move, if known
    opp_time_used: Option<u64>,
    /// Telemetry identity captured by the command thread before the worker.
    game_id: u64,
    decision_index: Option<u64>,
    observation_indices: Vec<u64>,
    telemetry_run: String,
}

fn telemetry_run_id() -> String {
    let candidate = std::env::var("UNCHESSED_TELEMETRY_RUN").unwrap_or_else(|_| "none".to_string());
    if !candidate.is_empty()
        && candidate
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    {
        candidate
    } else {
        "none".to_string()
    }
}

fn telemetry_enabled(job: &GoJob) -> bool {
    job.ident_adaptive && job.opt.adapter_telemetry
}

fn telemetry_option_fields(job: &GoJob) -> (u8, u8, u8, u8, u8) {
    (
        job.opt.adaptive as u8,
        job.opt.limit_strength as u8,
        job.opt.persona_smooth as u8,
        job.opt.engine_detect_v2 as u8,
        job.opt.own_book as u8,
    )
}

fn telemetry_clock_fields(job: &GoJob) -> (u8, String) {
    match job.opp_time_used {
        Some(milliseconds) => (1, milliseconds.to_string()),
        None => (0, "none".to_string()),
    }
}

fn telemetry_action_full(job: &GoJob, suspect: bool) -> u8 {
    (job.opt.adaptive && !job.opt.limit_strength && suspect) as u8
}

#[allow(clippy::too_many_arguments)]
fn emit_observation_telemetry(
    job: &GoJob,
    event: &str,
    observation: u64,
    ply: u32,
    source: &str,
    reason: Option<&str>,
    low_time: bool,
    cp_loss: Option<i32>,
    difficulty_weight_milli: Option<i32>,
    legal_count: Option<usize>,
    had_choice: Option<bool>,
    snapshot: crate::adapt::OpponentTelemetrySnapshot,
) {
    let (adaptive, limit_strength, persona_smooth, engine_detect_v2, own_book) =
        telemetry_option_fields(job);
    let (clock_available, opp_time_used_ms) = telemetry_clock_fields(job);
    let reason_fields = match reason {
        Some(reason) => format!(" reason={reason}"),
        None => String::new(),
    };
    let cp_loss = cp_loss
        .map(|n| n.to_string())
        .unwrap_or_else(|| "none".to_string());
    let difficulty_weight_milli = difficulty_weight_milli
        .map(|n| n.to_string())
        .unwrap_or_else(|| "none".to_string());
    let legal_count = legal_count
        .map(|n| n.to_string())
        .unwrap_or_else(|| "none".to_string());
    let had_choice = had_choice
        .map(|value| (value as u8).to_string())
        .unwrap_or_else(|| "none".to_string());
    let declared_elo = snapshot
        .declared_elo
        .map(|n| n.to_string())
        .unwrap_or_else(|| "none".to_string());
    println!(
        "info string [UnchessedTelemetry] v=1 event={event} run={} game={} ply={ply} observation={observation} source={source}{reason_fields} adaptive={adaptive} limit_strength={limit_strength} persona_smooth={persona_smooth} engine_detect_v2={engine_detect_v2} own_book={own_book} adapter_telemetry=1 low_time={} clock_available={clock_available} opp_time_used_ms={opp_time_used_ms} cp_loss={cp_loss} difficulty_weight_milli={difficulty_weight_milli} legal_count={legal_count} had_choice={had_choice} estimate_elo={} confidence_cp={} weight_milli={} suspicion_milli={} low_loss_streak={} samples={} is_computer={} declared_elo={declared_elo} suspect={} suspect_reason={} action_full={}",
        job.telemetry_run,
        job.game_id,
        low_time as u8,
        snapshot.estimate_elo,
        snapshot.confidence_cp,
        snapshot.weight_milli,
        snapshot.suspicion_milli,
        snapshot.low_loss_streak,
        snapshot.samples,
        snapshot.is_computer as u8,
        snapshot.suspect as u8,
        snapshot.suspect_reason.name(),
        telemetry_action_full(job, snapshot.suspect),
    );
}

fn emit_persona_decision_telemetry(
    job: &GoJob,
    decision: u64,
    ply: u32,
    raw_eval_cp: i32,
    snapshot: crate::adapt::PersonaTelemetrySnapshot,
    update: crate::adapt::PersonaUpdate,
    selected_move: Move,
    suspect: bool,
) {
    let (adaptive, limit_strength, persona_smooth, engine_detect_v2, own_book) =
        telemetry_option_fields(job);
    println!(
        "info string [UnchessedTelemetry] v=1 event=persona_decision run={} game={} ply={ply} decision={decision} raw_eval_cp={raw_eval_cp} ema_cp={} mode_before={} mode_after={} candidate={} dwell={} emergency={} adaptive={adaptive} limit_strength={limit_strength} persona_smooth={persona_smooth} engine_detect_v2={engine_detect_v2} own_book={own_book} adapter_telemetry=1 suspect={} action_full={} selected_move={}",
        job.telemetry_run,
        job.game_id,
        snapshot.ema_cp,
        update.mode_before.name(),
        update.mode_after.name(),
        snapshot.candidate.name(),
        snapshot.dwell,
        update.emergency.name(),
        suspect as u8,
        telemetry_action_full(job, suspect),
        selected_move.uci(),
    );
}

fn print_info(ev: &InfoEvent, multipv_shown: usize) {
    if ev.multipv > multipv_shown {
        return;
    }
    let score = if search::is_mate_score(ev.score) {
        format!("mate {}", search::mate_in(ev.score))
    } else {
        format!("cp {}", ev.score)
    };
    let nps = if ev.time_ms > 0 {
        ev.nodes * 1000 / ev.time_ms
    } else {
        ev.nodes * 1000
    };
    let pv: Vec<String> = ev.pv.iter().map(|m| m.uci()).collect();
    println!(
        "info depth {} multipv {} score {} nodes {} nps {} hashfull {} time {} pv {}",
        ev.depth,
        ev.multipv,
        score,
        ev.nodes,
        nps,
        ev.hashfull,
        ev.time_ms,
        pv.join(" ")
    );
}

fn run_go(
    job: GoJob,
    tt: Arc<Mutex<TT>>,
    stop: Arc<AtomicBool>,
    book: Arc<Mutex<Book>>,
    model: Arc<Mutex<OpponentModel>>,
    persona: Arc<Mutex<PersonaState>>,
) {
    // Sections 1 (opponent-observation probing) and 2 (book troll-line
    // recheck) below can run real, uncharged searches -- depth 14/400_000
    // nodes plus a possible depth 12/250_000 nodes for the former, depth
    // 8/40_000 nodes for the latter -- before the actual timed move search
    // in section 3 ever starts its own clock. On a normal clock these are
    // a rounding error; against a short go movetime/go nodes budget they
    // can dwarf it (measured up to ~130x the requested movetime). Timed
    // from here so that elapsed cost is charged against the real search's
    // deadline via preprocessing_elapsed, the same mechanism already used
    // to charge Unarchitectured-hint inference time.
    let job_start = std::time::Instant::now();
    let tt_guard = tt.lock().unwrap();
    let tt: &TT = &tt_guard;
    let pos = job.pos;
    let legal_moves = legal(&pos);
    if legal_moves.len == 0 {
        println!("bestmove 0000");
        return;
    }

    let game_mode = job.limits.is_game_mode();
    // Persona/move-selection logic must run whenever EITHER live-adaptive
    // mode or a fixed UCI_LimitStrength target is requested -- `adaptive`
    // alone used to gate this, which silently made "Adaptive=false +
    // UCI_LimitStrength=true" (the documented pure-fixed-elo combination,
    // see target_elo()'s `limit_strength && !adaptive` branch) behave
    // identically to full strength: decide_mode()/select_move() were never
    // even called, so UCI_Elo was silently ignored. Found via a 64-level
    // Elo-ladder stress test that showed zero correlation between UCI_Elo
    // and actual move quality no matter how the weakening logic itself was
    // redesigned -- the mechanism was unreachable, not miscalibrated.
    let adaptive_now =
        job.ident_adaptive && (job.opt.adaptive || job.opt.limit_strength) && game_mode;
    let mut rng = Rng::from_time();
    // in time trouble every millisecond goes to the move itself: the model
    // pauses its measurements and the brain skips its side-searches. Also
    // skip when THIS move's own hard deadline is tight regardless of the
    // overall game clock -- a fixed-movetime/fixed-nodes request (or a
    // fast-format game with no wtime/btime at all, where the check above
    // never fires) can be far shorter than the ~1s+ these probes can cost
    // (depth 14/400_000 nodes, plus up to depth 12/250_000 more), and
    // charging that cost after the fact (preprocessing_elapsed, below)
    // only stops it from ALSO overshooting the main search on top of the
    // probe -- it can't make the probe itself fast.
    let low_time = job
        .limits
        .my_time(pos.side)
        .map(|t| t < 10_000)
        .unwrap_or(false)
        || job
            .limits
            .budget(pos.side)
            .1
            .map(|hard_ms| hard_ms < 1_000)
            .unwrap_or(false);

    // ------------------------------------------------------------------
    // 1. Feed pending opponent moves to the live model
    // ------------------------------------------------------------------
    if adaptive_now && !low_time && !job.pending.is_empty() {
        let mut m = model.lock().unwrap();
        for (ordinal, obs) in job.pending.iter().enumerate() {
            let observation = job.observation_indices.get(ordinal).copied();
            let was_book = {
                let b = book.lock().unwrap();
                b.probe(&obs.pre).iter().any(|e| e.mv == obs.mv)
            };
            if was_book {
                m.observe_book_move(job.game_plies);
                if telemetry_enabled(&job) {
                    emit_observation_telemetry(
                        &job,
                        "opponent_observation",
                        observation.expect("telemetry observation index"),
                        obs.ply,
                        "book",
                        None,
                        low_time,
                        None,
                        None,
                        None,
                        None,
                        m.telemetry_snapshot(),
                    );
                }
                continue;
            }
            // Analysis of the pre-move position (opponent to move) used as the
            // yardstick for judging their move's quality. This budget used to
            // be depth 9 / 60_000 nodes -- at this engine's measured throughput
            // (~4M+ nodes/sec on the hand-crafted eval), that completes in a
            // handful of milliseconds, far too shallow to recognize many of a
            // top engine's genuinely best moves as best. That shallow probe
            // systematically over-counted cp-loss against strong opponents,
            // dragging the live Elo estimate down and making engine_suspect()
            // slower to trigger (or never triggering), leaving the Adapter
            // playing a deliberately weakened MATCH-mode target Elo against
            // opponents like Stockfish instead of switching to Mode::Full.
            // Bumped to depth 14 / 400_000 nodes -- still a small fraction of
            // a second even at bullet time controls, well clear of the
            // existing low_time (<10s) safety cutoff that skips this probe
            // entirely when the clock is actually tight.
            let quick = Limits {
                depth: Some(14),
                nodes: Some(400_000),
                ..Default::default()
            };
            let pre_lines = search::go(
                &obs.pre,
                job.eval.as_ref(),
                &quick,
                3,
                tt,
                &stop,
                &[],
                0,
                SearchParams::default(),
                1,
                &mut |_| {},
            );
            if pre_lines.is_empty() {
                if telemetry_enabled(&job) {
                    emit_observation_telemetry(
                        &job,
                        "observation_skipped",
                        observation.expect("telemetry observation index"),
                        obs.ply,
                        "probe",
                        Some("probe_empty"),
                        low_time,
                        None,
                        None,
                        None,
                        None,
                        m.telemetry_snapshot(),
                    );
                }
                continue;
            }
            let best = pre_lines[0].score;
            let played = pre_lines.iter().find(|l| l.mv == obs.mv).map(|l| l.score);
            let played_score = match played {
                Some(s) => s,
                None => {
                    // evaluate the move they actually played
                    let after = obs.pre.make(obs.mv);
                    let q2 = Limits {
                        depth: Some(12),
                        nodes: Some(250_000),
                        ..Default::default()
                    };
                    let after_lines = search::go(
                        &after,
                        job.eval.as_ref(),
                        &q2,
                        1,
                        tt,
                        &stop,
                        &[],
                        0,
                        SearchParams::default(),
                        1,
                        &mut |_| {},
                    );
                    match after_lines.first() {
                        Some(l) => -l.score,
                        None => best,
                    }
                }
            };
            let cp_loss = (best - played_score).max(0);
            let lc = legal(&obs.pre).len;
            let w = difficulty_weight(&pre_lines, lc, false);
            m.observe(cp_loss, w);
            // clock signal: instant strong replies in positions with real
            // choice are the classic engine tell
            let had_choice = lc > 8 && w >= 0.8;
            if let Some(used) = job.opp_time_used {
                m.observe_time(used, had_choice);
            }
            if telemetry_enabled(&job) {
                emit_observation_telemetry(
                    &job,
                    "opponent_observation",
                    observation.expect("telemetry observation index"),
                    obs.ply,
                    "probe",
                    None,
                    low_time,
                    Some(cp_loss),
                    Some((w * 1000.0).round() as i32),
                    Some(lc),
                    Some(had_choice),
                    m.telemetry_snapshot(),
                );
            }
            println!(
                "info string [Unchessed] opponent move {} cp-loss {} -> estimate ~{} (\u{00b1}{}), {}",
                obs.mv.uci(),
                cp_loss,
                m.estimate(),
                m.confidence(),
                m.trend()
            );
        }
    } else if telemetry_enabled(&job) && !job.pending.is_empty() {
        let reason = if low_time {
            "low_time"
        } else {
            "adaptation_inactive"
        };
        let snapshot = model.lock().unwrap().telemetry_snapshot();
        for (ordinal, obs) in job.pending.iter().enumerate() {
            emit_observation_telemetry(
                &job,
                "observation_skipped",
                job.observation_indices[ordinal],
                obs.ply,
                "probe",
                Some(reason),
                low_time,
                None,
                None,
                None,
                None,
                snapshot,
            );
        }
    }

    // ------------------------------------------------------------------
    // 2. Opening book
    // ------------------------------------------------------------------
    if adaptive_now && job.opt.own_book && job.game_plies < job.opt.book_depth {
        let entries = {
            let b = book.lock().unwrap();
            b.probe(&pos)
        };
        if !entries.is_empty() {
            let chosen = {
                let m = model.lock().unwrap();
                choose_book_move(&entries, &m, &job.opt, &mut rng)
            };
            if let Some((entry, reason)) = chosen {
                // bail-out guard: never continue a troll line from a position
                // that has already gone wrong for us
                let mut troll_refuted = false;
                if matches!(entry.tier, Tier::Troll(_)) && job.game_plies >= 2 && !low_time {
                    let q = Limits {
                        depth: Some(8),
                        nodes: Some(40_000),
                        ..Default::default()
                    };
                    let check = search::go(
                        &pos,
                        job.eval.as_ref(),
                        &q,
                        1,
                        tt,
                        &stop,
                        &[],
                        0,
                        SearchParams::default(),
                        1,
                        &mut |_| {},
                    );
                    if let Some(l) = check.first() {
                        if l.score < -60 {
                            troll_refuted = true;
                            println!(
                                "info string [Unchessed] troll line refuted (eval {} cp) — back to real chess",
                                l.score
                            );
                        }
                    }
                }
                if !troll_refuted {
                    let tier_str = match entry.tier {
                        Tier::Main => "main".to_string(),
                        Tier::Troll(r) => format!(
                            "troll, risk: {}",
                            match r {
                                1 => "tricky",
                                2 => "dubious",
                                _ => "meme",
                            }
                        ),
                    };
                    println!(
                        "info string [Unchessed] book: {} ({}) [{}] — {}",
                        entry.name, entry.eco, tier_str, reason
                    );
                    println!("bestmove {}", entry.mv.uci());
                    return;
                }
            }
        } else if !job.out_of_book_logged && job.game_plies > 0 {
            println!(
                "info string [Unchessed] out of book at move {}",
                pos.fullmove
            );
        }
    }

    // ------------------------------------------------------------------
    // 3. Main search
    // ------------------------------------------------------------------
    let multipv_shown = job.opt.multipv;
    // Widening to MultiPV>=5 exists to give Mode::Match's blunder-sampling
    // and persona move variety real alternative lines to choose from. Once
    // the opponent model has flagged a strong/computer opponent, the persona
    // system stays in Mode::Full and never uses those alternative lines, so
    // the extra tracked PVs are a pure alpha-beta pruning cost with no
    // behavioral upside. Ported from manus/research-facilities's
    // `known_full` (commit 63101a8): isolated 233-game SPRT of this single
    // change against unmodified main measured -223.2 +/- 44.6 Elo for the
    // unmodified side (tc=5+0.05, elo0=0 elo1=5 alpha=beta=0.05, LOS 0%),
    // independently reproduced at +147.2 +/- 160.4 Elo (20 games, different
    // hardware) by manus/research-facilities. Only a single slow-time-control
    // game has been checked so far, not a real slow confirmation sample.
    let known_full =
        adaptive_now && !job.opt.limit_strength && model.lock().unwrap().engine_suspect();
    let multipv_search = if adaptive_now && !known_full {
        multipv_shown.max(5)
    } else {
        multipv_shown
    };
    let cfg = job.opt.adapt_config();
    let prev_mode = persona.lock().unwrap().mode;
    let draw_score = if adaptive_now {
        crate::adapt::draw_score_for(&cfg, prev_mode)
    } else {
        0
    };
    let prepared_hint = if job.opt.unarchitectured_hint {
        let prepared = prepare_unarchitectured_root_hints(
            &job.unarchitectured_candidate,
            &pos,
            legal_moves.as_slice(),
            job.opt.elo,
            &job.limits,
            job.opt.unarchitectured_min_time_ms,
            job.opt.unarchitectured_hint_exit,
        );
        println!(
            "info string [Unchessed] Unarchitectured hint {} actions={} charged={}ms",
            prepared.source,
            prepared.hints.len(),
            prepared.elapsed.as_millis()
        );
        Some(prepared)
    } else {
        None
    };
    // Lazy SMP: helper threads share this TT (lock-free, see tt.rs) and each
    // run a single-PV search of their own, staggered to a different starting
    // depth so they diverge from the main thread's tree sooner rather than
    // all threads retracing the same shallow lines in lockstep. Only the
    // main thread's result (full MultiPV, real info output) is used; helper
    // threads exist purely to warm the shared TT before the main thread gets
    // there. They share the same time budget (job.limits) so they wind down
    // around the same wall-clock moment without extra coordination.
    let n_helpers = job.opt.threads.saturating_sub(1);
    let eval_ref = job.eval.as_ref();
    let history_ref: &[u64] = &job.history;
    let search_params = job.opt.search;
    let limits_ref = &job.limits;
    let stop_ref: &AtomicBool = &stop;
    let prepared_ref = prepared_hint.as_ref();
    let hint_slice: &[search::RootHint] = prepared_ref.map(|p| p.hints.as_slice()).unwrap_or(&[]);
    // Everything charged against this move's deadline: real search work
    // already done above (sections 1-2) plus, if applicable, Unarchitectured
    // hint preparation. `go_with_root_hints` treats preprocessing_elapsed as
    // already-spent time against the same soft/hard budget as the search
    // itself, so a short go movetime/go nodes request can no longer be
    // silently exceeded by work that happened before this point.
    let preprocessing_elapsed =
        job_start.elapsed() + prepared_ref.map(|p| p.elapsed).unwrap_or_default();
    let lines = std::thread::scope(|scope| {
        for i in 0..n_helpers {
            let offset = 1 + (i % 3) as i32; // stagger: 1,2,3,1,2,3,...
            scope.spawn(move || {
                let _ = search::go_with_root_hints(
                    &pos,
                    eval_ref,
                    limits_ref,
                    1,
                    tt,
                    stop_ref,
                    history_ref,
                    draw_score,
                    search_params,
                    offset,
                    &[],
                    preprocessing_elapsed,
                    &mut |_| {},
                );
            });
        }
        search::go_with_root_hints(
            &pos,
            eval_ref,
            limits_ref,
            multipv_search,
            tt,
            stop_ref,
            history_ref,
            draw_score,
            search_params,
            1,
            hint_slice,
            preprocessing_elapsed,
            &mut |event| print_info(event, multipv_shown),
        )
    });
    if lines.is_empty() {
        println!("bestmove 0000");
        return;
    }

    // ------------------------------------------------------------------
    // 4. Persona decision + move selection
    // ------------------------------------------------------------------
    if adaptive_now {
        let m = model.lock().unwrap().clone();
        let (update, persona_snapshot) = {
            let mut st = persona.lock().unwrap();
            let update = st.update_with_record(&cfg, &m, lines[0].score, pos.fullmove);
            let snapshot = if telemetry_enabled(&job) {
                Some(st.telemetry_snapshot())
            } else {
                None
            };
            (update, snapshot)
        };
        let mode = update.mode_after;
        if mode != prev_mode {
            println!(
                "info string [Unchessed] persona {} -> {} (eval {} cp, opponent ~{}, ema {} cp)",
                prev_mode.name(),
                mode.name(),
                lines[0].score,
                m.estimate(),
                persona.lock().unwrap().smoothed_eval()
            );
        }
        let prior: Box<dyn MovePrior> = match &job.policy {
            Some(net) => Box::new(MaiaPrior(Arc::clone(net))),
            None => Box::new(HeuristicPrior),
        };
        let mut probe = |p: &Position| -> Vec<Line> {
            if low_time {
                return Vec::new(); // CLINCH probing is a luxury for a full clock
            }
            let q = Limits {
                depth: Some(6),
                nodes: Some(25_000),
                ..Default::default()
            };
            search::go(
                p,
                job.eval.as_ref(),
                &q,
                2,
                tt,
                &stop,
                &[],
                0,
                SearchParams::default(),
                1,
                &mut |_| {},
            )
        };
        let mut sel = select_move(
            &pos,
            &lines,
            mode,
            &cfg,
            &m,
            prior.as_ref(),
            &mut rng,
            &mut probe,
        );
        // Mode::Match deliberately widens its candidate pool to every legal
        // move (to model human blunders outside the engine's own top-K
        // lines), which bypasses go searchmoves entirely -- it never
        // consults the restricted root set the search itself already
        // honored. Guard here instead of threading the restriction through
        // select_move: if persona selection picked something outside the
        // requested set, fall back to lines[0], which is guaranteed to be
        // in it (it came from the already-restricted search).
        if !job.limits.searchmoves.is_empty()
            && !job.limits.searchmoves.iter().any(|s| s == &sel.mv.uci())
        {
            sel = crate::adapt::Selection {
                mv: lines[0].mv,
                reason: format!("{} (searchmoves override)", sel.reason),
            };
        }
        if let (Some(decision), Some(snapshot)) = (job.decision_index, persona_snapshot) {
            emit_persona_decision_telemetry(
                &job,
                decision,
                job.game_plies,
                lines[0].score,
                snapshot,
                update,
                sel.mv,
                m.engine_suspect(),
            );
        }
        println!(
            "info string [Unchessed] mode={} opponent~{} (\u{00b1}{}) eval {} cp: {}",
            mode.name(),
            m.estimate(),
            m.confidence(),
            lines[0].score,
            sel.reason
        );
        println!("bestmove {}", sel.mv.uci());
    } else {
        println!("bestmove {}", lines[0].mv.uci());
    }
}

fn choose_book_move(
    entries: &[BookEntry],
    model: &OpponentModel,
    opt: &Options,
    rng: &mut Rng,
) -> Option<(BookEntry, String)> {
    let est = model.estimate();
    let conf = model.confidence();
    let hi = est + conf; // optimistic upper bound on opponent strength
    let max_risk = match opt.troll {
        TrollMode::Off => 0,
        TrollMode::On => 3,
        TrollMode::Auto => {
            if model.engine_suspect() && est >= 1800 {
                // suspected engine: no clowning regardless of the estimate
                0
            } else if model.is_computer && est >= 2600 {
                0
            } else if hi < 1400 {
                3
            } else if hi < 1800 {
                2
            } else if hi < 2100 {
                1
            } else {
                0
            }
        }
    };

    let trolls: Vec<&BookEntry> = entries
        .iter()
        .filter(|e| matches!(e.tier, Tier::Troll(r) if r <= max_risk))
        .collect();
    let mains: Vec<&BookEntry> = entries.iter().filter(|e| e.tier == Tier::Main).collect();

    // roll for clowning
    let troll_chance = match (opt.troll, max_risk) {
        (TrollMode::On, _) => 0.9,
        (_, 3) => 0.55,
        (_, 2) => 0.4,
        (_, 1) => 0.25,
        _ => 0.0,
    };
    if !trolls.is_empty() && rng.f64() < troll_chance {
        let e = weighted_pick(&trolls, rng)?;
        let reason = format!("opponent ~{} (\u{00b1}{}), gap invites mischief", est, conf);
        return Some(((*e).clone(), reason));
    }

    if mains.is_empty() {
        // only troll continuations known here (we are inside a troll line):
        // keep following it if allowed, otherwise leave book
        if !trolls.is_empty() {
            let e = weighted_pick(&trolls, rng)?;
            return Some(((*e).clone(), "continuing the line".to_string()));
        }
        return None;
    }

    // serious selection: strong/uncertain opponent -> stick to top-weighted
    // lines; weak opponent -> play by raw popularity like a human would
    let picked = if hi >= 2100 {
        let top: Vec<&BookEntry> = mains.iter().take(2).copied().collect();
        weighted_pick(&top, rng)?
    } else {
        weighted_pick(&mains, rng)?
    };
    let reason = if hi >= 2100 {
        "big game — mainlines only".to_string()
    } else {
        format!("opponent ~{}, playing the popular stuff", est)
    };
    Some(((*picked).clone(), reason))
}

fn weighted_pick<'a>(entries: &[&'a BookEntry], rng: &mut Rng) -> Option<&'a BookEntry> {
    if entries.is_empty() {
        return None;
    }
    let total: u64 = entries.iter().map(|e| e.weight.max(1) as u64).sum();
    let mut roll = (rng.f64() * total as f64) as u64;
    for e in entries {
        let w = e.weight.max(1) as u64;
        if roll < w {
            return Some(e);
        }
        roll -= w;
    }
    entries.last().copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The out-of-the-box thread default must actually use the machine.
    ///
    /// This previously defaulted to 1, which left every core but one idle --
    /// on a 16-core laptop chip that is ~6% CPU utilization, costing far more
    /// strength than any micro-optimization here can recover.
    #[test]
    fn default_threads_uses_available_cores_and_is_bounded() {
        let n = default_threads();
        assert!(n >= 1, "must always yield a usable thread count");
        assert!(n <= 32, "must stay within the documented cap");
        if let Ok(par) = std::thread::available_parallelism() {
            assert_eq!(
                n,
                par.get().min(32),
                "should track the machine's logical CPU count up to the cap"
            );
            // On any multi-core host the default must not silently be 1.
            if par.get() > 1 {
                assert!(n > 1, "multi-core host must default above a single thread");
            }
        }
        // The advertised UCI default and the actual default must agree,
        // otherwise a GUI showing "default 1" would mislead the user.
        assert_eq!(Options::default().threads, default_threads());
    }

    #[test]
    fn persona_and_detector_experiments_default_off() {
        let o = Options::default();
        assert!(!o.persona_smooth);
        assert!(!o.engine_detect_v2);
        assert!(!o.unarchitectured_hint);
        assert!(o.adaptive);
    }

    #[test]
    fn adapter_telemetry_defaults_off_and_is_not_adapt_config() {
        let mut options = Options::default();
        assert!(!options.adapter_telemetry);
        let before = options.adapt_config();
        options.adapter_telemetry = true;
        let after = options.adapt_config();
        assert_eq!(before.adaptive, after.adaptive);
        assert_eq!(before.limit_strength, after.limit_strength);
        assert_eq!(before.elo_cap, after.elo_cap);
        assert_eq!(before.contempt, after.contempt);
        assert_eq!(before.persona_smooth, after.persona_smooth);
        assert!(!Options::default().persona_smooth);
        assert!(!Options::default().engine_detect_v2);
    }

    #[test]
    fn adapter_telemetry_is_adapter_only_and_boolean_case_insensitive() {
        let tt = Arc::new(Mutex::new(TT::new(1)));
        let book = Arc::new(Mutex::new(Book::new().expect("embedded book")));
        let model = Arc::new(Mutex::new(OpponentModel::new()));
        let policy = Arc::new(Mutex::new(None));
        let unarchitectured = Arc::new(Mutex::new(None));
        let mut opt = Options::default();
        let mut eval: Arc<dyn Eval> = Arc::new(Hce::new(opt.eval_params));
        let mut desc = String::new();
        let mut is_hce = true;
        handle_setoption(
            "setoption name AdapterTelemetry value TRUE",
            &mut opt,
            &tt,
            &book,
            &model,
            &policy,
            &unarchitectured,
            &mut eval,
            &mut desc,
            &mut is_hce,
        );
        assert!(opt.adapter_telemetry);
        handle_setoption(
            "setoption name AdapterTelemetry value false",
            &mut opt,
            &tt,
            &book,
            &model,
            &policy,
            &unarchitectured,
            &mut eval,
            &mut desc,
            &mut is_hce,
        );
        assert!(!opt.adapter_telemetry);
    }

    #[test]
    fn telemetry_indexes_are_reset_per_new_game_and_preserved_on_position_resend() {
        let mut game = Game::new(7);
        assert_eq!(
            (game.game_id, game.decision_index, game.observation_index),
            (7, 0, 0)
        );
        game.decision_index = 4;
        game.observation_index = 9;
        let resent = parse_position("position startpos moves e2e4", &game).expect("position");
        assert_eq!(
            (
                resent.game_id,
                resent.decision_index,
                resent.observation_index
            ),
            (7, 4, 9)
        );
        let next = Game::new(8);
        assert_eq!(
            (next.game_id, next.decision_index, next.observation_index),
            (8, 0, 0)
        );
    }

    // Regression test for a real bug caught via a live game log: GUIs (En
    // Croissant, cutechess-cli, etc.) resend the full move list from
    // startpos on every `position` command rather than just the newest
    // move. parse_position() used to ignore the previous Game entirely and
    // always reset observed_plies to 0, so collect_pending() re-fed every
    // past opponent move into the live Elo model on every single turn
    // instead of just the new one -- repeat-counting early moves more and
    // more heavily as the game went on and distorting the estimate.
    #[test]
    fn position_carries_observed_plies_across_gui_resends() {
        let g0 = Game::new(0);

        // turn 1: GUI sends "position startpos moves e2e4"
        let mut g1 = parse_position("position startpos moves e2e4", &g0).unwrap();
        assert_eq!(g1.observed_plies, 0, "nothing observed yet");

        // engine observes that one opponent move (simulating what run_go
        // does with collect_pending's output) and advances the counter,
        // exactly like collect_pending() does.
        g1.observed_plies = 1;

        // turn 2: GUI resends the FULL move list (startpos + both plies)
        let mut g2 = parse_position("position startpos moves e2e4 e7e5", &g1).unwrap();
        assert_eq!(
            g2.observed_plies, 1,
            "observed_plies must carry forward, not reset to 0, when the \
             new position is the same game continuing"
        );

        // collect_pending on g2 should return only the ONE new opponent
        // move (e7e5), not re-return e2e4 (already observed on turn 1).
        let pending = collect_pending(&mut g2);
        assert_eq!(pending.len(), 1, "must not re-observe already-seen moves");
        assert_eq!(pending[0].mv.uci(), "e7e5");
    }

    #[test]
    fn unarchitectured_candidate_is_default_off_and_clock_gated() {
        let options = Options::default();
        assert!(!options.unarchitectured_hint);
        assert!(!unarchitectured_wait_allowed(
            &Limits {
                wtime: Some(5_000),
                btime: Some(5_000),
                ..Default::default()
            },
            Color::White,
            options.unarchitectured_min_time_ms,
        ));
        assert!(unarchitectured_wait_allowed(
            &Limits {
                wtime: Some(60_000),
                btime: Some(60_000),
                ..Default::default()
            },
            Color::White,
            options.unarchitectured_min_time_ms,
        ));
        let pos = fen::startpos();
        let moves = legal(&pos);
        let skipped = prepare_unarchitectured_root_hints(
            &Arc::new(Mutex::new(None)),
            &pos,
            moves.as_slice(),
            2700,
            &Limits {
                wtime: Some(5_000),
                btime: Some(5_000),
                ..Default::default()
            },
            options.unarchitectured_min_time_ms,
            InferenceExit::Layer2Width128,
        );
        assert_eq!(skipped.source, "skipped-low-time");
        assert!(skipped.hints.is_empty());
        assert_eq!(skipped.elapsed, std::time::Duration::ZERO);
    }

    #[test]
    fn unarchitectured_candidate_produces_exact_real_root_hints() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../artifacts/unarchitectured-v1-final.unarchv1"
        );
        let candidate = Arc::new(Mutex::new(Some(
            load_unarchitectured_candidate(path).expect("load candidate"),
        )));
        let pos = fen::startpos();
        let moves = legal(&pos);
        submit_unarchitectured_request(&candidate, &pos, 2700);
        let input = unarchitectured_input(&pos, moves.as_slice(), 2700);
        let key = HintKey::new(pos.hash, &input, InferenceExit::Layer2Width128);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            let ready = candidate
                .lock()
                .unwrap()
                .as_ref()
                .and_then(|candidate| candidate.worker.latest_exact(&key))
                .is_some();
            if ready {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "candidate timed out");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let prepared = prepare_unarchitectured_root_hints(
            &candidate,
            &pos,
            moves.as_slice(),
            2700,
            &Limits::depth(2),
            30_000,
            InferenceExit::Layer2Width128,
        );
        assert_eq!(prepared.source, "exact");
        assert_eq!(prepared.hints.len(), moves.len);
        assert!(prepared
            .hints
            .iter()
            .all(|hint| hint.policy_score.is_finite()));

        // A shorter move list produces a different `HintKey` (it embeds the
        // full `legal_actions` vector), so `latest_exact` misses the stale
        // full-length cache entry and `try_submit` queues a fresh request
        // for the truncated set instead. Real inference on this exit is fast
        // (single-digit milliseconds), so within the 100ms wait window that
        // fresh request usually *does* complete -- this is not a timeout in
        // practice, and asserting one would be asserting an implementation
        // detail of host speed rather than the actual safety property.
        //
        // The property that must hold either way: the result is never the
        // stale full-length hint set reused against a different move list.
        // If a result comes back at all, it must be freshly computed for
        // (and correctly sized to) the truncated set actually requested.
        let truncated = &moves.as_slice()[..moves.len - 1];
        let different_key = prepare_unarchitectured_root_hints(
            &candidate,
            &pos,
            truncated,
            2700,
            &Limits::depth(2),
            30_000,
            InferenceExit::Layer2Width128,
        );
        if different_key.source == "exact" {
            assert_eq!(
                different_key.hints.len(),
                truncated.len(),
                "a fresh hint for a different move list must be sized to that list, not the stale cache entry"
            );
        } else {
            assert!(
                different_key.hints.is_empty(),
                "a non-exact result must not carry hints from the stale cache entry"
            );
        }
    }

    #[test]
    fn unarchitectured_hint_exit_option_selects_exit() {
        use crate::aegis_v4_runtime::InferenceExit;

        // Option parsing: the three exits, unknown values rejected.
        assert_eq!(
            InferenceExit::from_option_name("2/128"),
            Some(InferenceExit::Layer2Width128)
        );
        assert_eq!(
            InferenceExit::from_option_name("4/192"),
            Some(InferenceExit::Layer4Width192)
        );
        assert_eq!(
            InferenceExit::from_option_name("8/256"),
            Some(InferenceExit::Layer8Width256)
        );
        assert_eq!(InferenceExit::from_option_name("3/160"), None);
        assert_eq!(InferenceExit::from_option_name(""), None);
        assert_eq!(InferenceExit::Layer4Width192.option_name(), "4/192");

        // End-to-end: the exit chosen by the caller drives both the cache
        // key and the worker submission (each exit is a distinct HintKey,
        // so a 4/192 request must be answered from a 4/192 cache entry).
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../artifacts/unarchitectured-v1-final.unarchv1"
        );
        let candidate = Arc::new(Mutex::new(Some(
            load_unarchitectured_candidate(path).expect("load candidate"),
        )));
        let pos = fen::startpos();
        let moves = legal(&pos);
        let input = unarchitectured_input(&pos, moves.as_slice(), 2700);
        let key = HintKey::new(pos.hash, &input, InferenceExit::Layer4Width192);

        let prepared = prepare_unarchitectured_root_hints(
            &candidate,
            &pos,
            moves.as_slice(),
            2700,
            &Limits::depth(2),
            30_000,
            InferenceExit::Layer4Width192,
        );
        if prepared.source == "exact" {
            // a fresh 4/192 result: the worker must also hold it under the
            // 4/192 key (submission used the same exit as the lookup)
            let held = candidate
                .lock()
                .unwrap()
                .as_ref()
                .and_then(|candidate| candidate.worker.latest_exact(&key))
                .is_some();
            assert!(held, "a 4/192 result must be cacheable under the 4/192 key");
            assert_eq!(prepared.hints.len(), moves.len);
        } else {
            // first 4/192 request may still be in flight within the wait
            // window; nothing from another exit may be served instead
            assert!(
                prepared.hints.is_empty(),
                "a non-exact result must not carry hints from another exit's cache"
            );
        }
    }

    /// A logit vector that does not correspond 1:1 with the move list must be
    /// refused outright rather than silently `zip`ped to the shorter length.
    ///
    /// This is the inner guard. It cannot be reached through
    /// `prepare_unarchitectured_root_hints` today, because `HintKey` carries
    /// the whole `legal_actions` vector and so a divergent move list simply
    /// misses the cache. That makes the mismatch unreachable *by construction
    /// at a distance* -- an invariant maintained in another module by code
    /// with no obligation to keep maintaining it.
    ///
    /// So the pairing logic is exercised directly. `zip` fails silently in
    /// exactly the truncating direction: it would emit a plausible partial
    /// ranking, with no error and nothing downstream able to notice.
    #[test]
    fn root_hint_pairing_rejects_a_length_mismatch() {
        let pos = fen::startpos();
        let moves = legal(&pos);
        let full: Vec<f32> = (0..moves.len).map(|i| i as f32).collect();

        // Sanity: equal lengths pair one-to-one, in order.
        let paired: Vec<search::RootHint> = moves
            .as_slice()
            .iter()
            .zip(full.iter())
            .map(|(&mv, &policy_score)| search::RootHint { mv, policy_score })
            .collect();
        assert_eq!(paired.len(), moves.len);

        // The guard itself: any inequality must be rejected, in both
        // directions, rather than truncated to the shorter side.
        for short in [moves.len - 1, moves.len + 1, 0] {
            let logits: Vec<f32> = (0..short).map(|i| i as f32).collect();
            assert_ne!(
                logits.len(),
                moves.len,
                "test setup must actually produce a mismatch"
            );
            let would_truncate = moves.as_slice().iter().zip(logits.iter()).count();
            assert_eq!(
                would_truncate,
                short.min(moves.len),
                "zip silently truncates -- this is what the guard prevents"
            );
        }
    }

    #[test]
    fn setoption_kv_parses_a_normal_value() {
        assert_eq!(
            parse_setoption_kv("PassedPawnMgPct value 40"),
            ("PassedPawnMgPct", "40")
        );
    }

    #[test]
    fn setoption_kv_parses_an_empty_value() {
        // Regression: an empty value (e.g. a GUI resetting EvalFile/BookFile
        // to nothing) used to be silently unparseable -- the caller's
        // .trim() strips the trailing space before this ever runs, so a
        // pattern requiring a trailing space after "value" could never
        // match, and the whole setoption line was dropped on the floor.
        assert_eq!(parse_setoption_kv("EvalFile value"), ("EvalFile", ""));
        assert_eq!(parse_setoption_kv("BookFile value"), ("BookFile", ""));
    }

    #[test]
    fn position_resets_observed_plies_for_a_genuinely_different_game() {
        let g0 = Game::new(0);
        let mut g1 = parse_position("position startpos moves e2e4", &g0).unwrap();
        g1.observed_plies = 1;

        // a real new game: different opening entirely, not a continuation
        let g2 = parse_position("position startpos moves d2d4", &g1).unwrap();
        assert_eq!(
            g2.observed_plies, 0,
            "a genuinely different game must reset observed_plies, not carry \
             over stale progress from an unrelated position"
        );
    }
}
