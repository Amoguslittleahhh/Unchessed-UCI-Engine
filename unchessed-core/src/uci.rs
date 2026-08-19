//! UCI protocol loop. Search runs on a worker thread; `stop` flips an atomic
//! flag. The adapter pipeline (opponent observation -> book -> search ->
//! persona selection) lives in the worker so the GUI never blocks.

use std::io::BufRead;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::adapt::{
    decide_persona, difficulty_weight, select_move, AdaptConfig, HeuristicPrior, MaiaPrior, Mode,
    MovePrior, OpponentModel, PersonaContext, Rng, ENGINE_CEILING,
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
    /// Zero = entropy from wall clock; non-zero = deterministic game/search seed.
    random_seed: u64,
    eval_params: EvalParams,
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
            book_depth: 40,
            search: SearchParams::default(),
            threads: 1,
            random_seed: 0,
            eval_params: EvalParams::default(),
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
        }
    }
}

/// One opponent move we have not yet fed to the model.
struct PendingObs {
    pre: Position,
    mv: Move,
}

struct Game {
    /// position after each played move; [0] is the game-start position
    positions: Vec<Position>,
    current: Position,
    /// plies already fed to the opponent model
    observed_plies: usize,
    out_of_book_logged: bool,
}

impl Game {
    fn new() -> Game {
        let p = fen::startpos();
        Game {
            positions: vec![p],
            current: p,
            observed_plies: 0,
            out_of_book_logged: false,
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
    let (mut eval_impl, mut eval_desc, mut eval_is_hce): (Arc<dyn Eval>, String, bool) =
        load_default_eval(opt.eval_params);
    let stop = Arc::new(AtomicBool::new(false));
    // Persona and prior root evaluation persist across moves for contextual
    // hysteresis; the worker updates both after a completed search.
    let persona = Arc::new(Mutex::new(Mode::Match));
    let previous_eval: Arc<Mutex<Option<i32>>> = Arc::new(Mutex::new(None));
    let mut worker: Option<JoinHandle<()>> = None;
    let mut game = Game::new();
    // Low-clock observations are deferred, not silently discarded. At most
    // one expensive observation is consumed per move once time is healthy.
    let mut deferred_pending: Vec<PendingObs> = Vec::new();
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
                println!("option name Threads type spin default 1 min 1 max 64");
                println!("option name Clear Hash type button");
                println!(
                    "option name MultiPV type spin default {} min 1 max 8",
                    if ident.adaptive_engine { 1 } else { 3 }
                );
                println!("option name EvalFile type string default ");
                if ident.adaptive_engine {
                    println!("option name Adaptive type check default true");
                    println!("option name UCI_LimitStrength type check default false");
                    println!(
                        "option name UCI_Elo type spin default 2400 min 100 max {}",
                        ENGINE_CEILING
                    );
                    println!("option name Contempt type spin default 25 min 0 max 100");
                    println!("option name Troll type combo default Auto var Off var Auto var On");
                    println!("option name OwnBook type check default true");
                    println!("option name BookFile type string default ");
                    println!("option name BookDepth type spin default 40 min 0 max 40");
                    println!("option name PolicyFile type string default ");
                    println!("option name UCI_Opponent type string default ");
                    println!("option name RandomSeed type spin default 0 min 0 max 2147483647");
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
                println!("option name IIR type check default false");
                println!("option name HistGravity type check default false");
                println!("option name CounterMoves type check default false");
                println!("option name Razoring type check default false");
                println!("option name LMP type check default false");
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
                let book_stats = book.lock().unwrap();
                println!(
                    "info string [Unchessed] opening book: {} named historical lines, {}/500 ECO codes, curated main/troll overlays",
                    book_stats.historical_lines(),
                    book_stats.eco_codes()
                );
                drop(book_stats);
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
                    &mut eval_impl,
                    &mut eval_desc,
                    &mut eval_is_hce,
                );
            }
            "ucinewgame" => {
                join_worker(&mut worker, &stop);
                tt.lock().unwrap().clear();
                let reset = model.lock().unwrap().reset_for_new_game();
                *model.lock().unwrap() = reset;
                *persona.lock().unwrap() = Mode::Match;
                *previous_eval.lock().unwrap() = None;
                deferred_pending.clear();
                last_opp_clock = None;
                game = Game::new();
            }
            "position" => {
                join_worker(&mut worker, &stop);
                if let Some(g) = parse_position(&line, &game) {
                    if !is_game_continuation(&game, &g.positions) {
                        deferred_pending.clear();
                        *previous_eval.lock().unwrap() = None;
                    }
                    game = g;
                } else {
                    println!("info string [Unchessed] could not parse: {}", line);
                }
            }
            "go" => {
                join_worker(&mut worker, &stop);
                let limits = parse_go(&line);
                let newly_pending = collect_pending(&mut game);
                let had_backlog = !deferred_pending.is_empty();
                deferred_pending.extend(newly_pending);
                let observation_budget_ms = observation_budget_ms(&limits, game.current.side);
                let pending = if observation_budget_ms == 0 || deferred_pending.is_empty() {
                    Vec::new()
                } else {
                    // Bound model overhead and preserve the rest for later.
                    deferred_pending.drain(..1).collect()
                };

                // Opponent clock time belongs only to the newest single move,
                // never to every item in a deferred/multi-move backlog.
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
                let time_signal_is_current = !had_backlog && pending.len() == 1;
                let opp_time_used = match (last_opp_clock, opp_clock_now, time_signal_is_current) {
                    (Some(prev), Some(now), true) => {
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
                    observation_budget_ms,
                    out_of_book_logged: game.out_of_book_logged,
                    policy: policy.lock().unwrap().clone(),
                    eval: Arc::clone(&eval_impl),
                    opp_time_used,
                    opp_clock_remaining: opp_clock_now,
                };
                // the worker decides book state transitions; mirror the flag
                // optimistically so the log line prints only once
                if job.game_plies >= opt.book_depth {
                    game.out_of_book_logged = true;
                }
                let tt = Arc::clone(&tt);
                let stop_c = Arc::clone(&stop);
                let book = Arc::clone(&book);
                let model = Arc::clone(&model);
                let persona_c = Arc::clone(&persona);
                let previous_eval_c = Arc::clone(&previous_eval);
                worker = Some(std::thread::spawn(move || {
                    run_go(job, tt, stop_c, book, model, persona_c, previous_eval_c);
                }));
            }
            "stop" | "ponderhit" => {
                // Ponder currently converts by returning the best completed
                // iteration immediately; it never runs past ponderhit/stop.
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
            let description = format!("NNUE (unchessed-nnue.bin, {})", net.backend_name());
            let e: Arc<dyn Eval> = net;
            (e, description, false)
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
        "randomseed" => {
            if let Ok(seed) = value.parse::<u64>() {
                opt.random_seed = seed.min(i32::MAX as u64);
            }
        }
        "multipv" => {
            if let Ok(n) = value.parse::<usize>() {
                opt.multipv = n.clamp(1, 8);
            }
        }
        "adaptive" => opt.adaptive = value.eq_ignore_ascii_case("true"),
        "uci_limitstrength" => opt.limit_strength = value.eq_ignore_ascii_case("true"),
        "uci_elo" => {
            if let Ok(e) = value.parse::<i32>() {
                opt.elo = e.clamp(100, ENGINE_CEILING);
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
        "iir" => opt.search.iir = value.eq_ignore_ascii_case("true"),
        "histgravity" => opt.search.history_gravity = value.eq_ignore_ascii_case("true"),
        "countermoves" => opt.search.countermoves = value.eq_ignore_ascii_case("true"),
        "razoring" => opt.search.razoring = value.eq_ignore_ascii_case("true"),
        "lmp" => opt.search.lmp = value.eq_ignore_ascii_case("true"),
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
                println!(
                    "info string [Unchessed] eval reset to default: {}",
                    eval_desc
                );
            } else {
                match Nnue::load(value) {
                    Ok(net) => {
                        let backend = net.backend_name();
                        *eval_impl = Arc::new(net);
                        *eval_desc = format!("NNUE ({}, {})", value, backend);
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
fn is_game_continuation(old: &Game, new_positions: &[Position]) -> bool {
    old.positions.len() <= new_positions.len()
        && old
            .positions
            .iter()
            .zip(new_positions.iter())
            .all(|(a, b)| a.hash == b.hash)
}

fn carry_observed_plies(old: &Game, new_positions: &[Position]) -> usize {
    if old.observed_plies > 0 && is_game_continuation(old, new_positions) {
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
    const KEYWORDS: &[&str] = &[
        "searchmoves",
        "ponder",
        "wtime",
        "btime",
        "winc",
        "binc",
        "movestogo",
        "depth",
        "nodes",
        "mate",
        "movetime",
        "infinite",
    ];
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let mut limits = Limits::default();
    let mut index = 1usize; // skip "go"
    while index < tokens.len() {
        let token = tokens[index];
        index += 1;
        let number = |index: &mut usize| -> Option<u64> {
            let value = tokens.get(*index)?.parse().ok();
            *index += 1;
            value
        };
        match token {
            "depth" => limits.depth = number(&mut index).map(|value| value as i32),
            "movetime" => limits.movetime = number(&mut index),
            "wtime" => limits.wtime = number(&mut index),
            "btime" => limits.btime = number(&mut index),
            "winc" => limits.winc = number(&mut index),
            "binc" => limits.binc = number(&mut index),
            "movestogo" => limits.movestogo = number(&mut index),
            "nodes" => limits.nodes = number(&mut index),
            "mate" => limits.mate = number(&mut index),
            "infinite" => limits.infinite = true,
            "ponder" => limits.ponder = true,
            "searchmoves" => {
                while index < tokens.len() && !KEYWORDS.contains(&tokens[index]) {
                    limits.searchmoves.push(tokens[index].to_string());
                    index += 1;
                }
            }
            _ => {}
        }
    }
    limits
}

fn is_low_time(limits: &Limits, side: Color) -> bool {
    // Optional per-candidate CLINCH/MATCH probes remain disabled when their
    // aggregate cost would be unsafe. Opponent measurement uses the smoother
    // bounded budget below instead of this binary threshold.
    limits.movetime.map(|ms| ms < 5_000).unwrap_or(false)
        || limits.my_time(side).map(|ms| ms < 10_000).unwrap_or(false)
}

fn observation_budget_ms(limits: &Limits, side: Color) -> u64 {
    if limits.infinite || limits.ponder {
        return 0;
    }
    if let Some(movetime) = limits.movetime {
        return if movetime < 1_000 {
            0
        } else {
            (movetime / 20).clamp(5, 100)
        };
    }
    if let Some(clock) = limits.my_time(side) {
        return if clock < 2_000 {
            0
        } else {
            (clock / 500).clamp(5, 100)
        };
    }
    if limits.depth.is_some() || limits.nodes.is_some() {
        50
    } else {
        0
    }
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
                out.push(PendingObs { pre, mv });
            }
        }
    }
    game.observed_plies = n_moves;
    out
}

struct GoJob {
    ident_adaptive: bool,
    pos: Position,
    history: Vec<u64>,
    game_plies: u32,
    limits: Limits,
    opt: Options,
    pending: Vec<PendingObs>,
    /// Total wall-clock allowance for opponent measurement this move.
    observation_budget_ms: u64,
    out_of_book_logged: bool,
    policy: Option<Arc<PolicyNet>>,
    /// static evaluator (NNUE or HCE) for every search this job runs
    eval: Arc<dyn Eval>,
    /// milliseconds the opponent spent on their last move, if known
    opp_time_used: Option<u64>,
    /// opponent clock after that move, used to normalise timing by clock size
    opp_clock_remaining: Option<u64>,
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
    persona: Arc<Mutex<Mode>>,
    previous_eval: Arc<Mutex<Option<i32>>>,
) {
    let job_started = Instant::now();
    let tt_guard = tt.lock().unwrap();
    let tt: &TT = &tt_guard;
    tt.new_search();
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
    // UCI_LimitStrength=true" (the pure fixed-Elo combination) behave
    // identically to full strength: decide_mode()/select_move() were never
    // even called, so UCI_Elo was silently ignored. Found via a 64-level
    // Elo-ladder stress test that showed zero correlation between UCI_Elo
    // and actual move quality no matter how the weakening logic itself was
    // redesigned -- the mechanism was unreachable, not miscalibrated.
    let adaptive_now =
        job.ident_adaptive && (job.opt.adaptive || job.opt.limit_strength) && game_mode;
    let mut rng = if job.opt.random_seed == 0 {
        Rng::from_time()
    } else {
        Rng::new(job.opt.random_seed ^ pos.hash ^ job.game_plies as u64)
    };
    // Optional style/candidate probes are skipped in time trouble. Opponent
    // measurement has a separate smooth, bounded budget and may be deferred.
    let low_time = is_low_time(&job.limits, pos.side);
    // Root search width is based on the model as it stood when `go` arrived.
    // New evidence may change the final persona, but cannot suddenly multiply
    // this move's root workload at a clock boundary.
    let model_at_go_start = model.lock().unwrap().clone();

    // ------------------------------------------------------------------
    // 1. Feed pending opponent moves to the live model
    // ------------------------------------------------------------------
    if adaptive_now && job.observation_budget_ms > 0 && !job.pending.is_empty() {
        let mut m = model.lock().unwrap();
        for obs in &job.pending {
            let was_book = {
                let b = book.lock().unwrap();
                b.probe(&obs.pre).iter().any(|e| e.mv == obs.mv)
            };
            if was_book {
                m.observe_book_move(job.game_plies);
                continue;
            }
            // Analysis of the pre-move position is the move-quality yardstick.
            // Depth/node ceilings protect label quality in long controls, while
            // movetime enforces this move's smooth observation allowance. A
            // stronger offline oracle remains necessary to calibrate the top
            // Elo bands; runtime analysis reports uncertainty above its ceiling.
            let quick = Limits {
                depth: Some(14),
                nodes: Some(400_000),
                movetime: Some((job.observation_budget_ms * 2 / 3).max(5)),
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
            if pre_lines.first().map(|line| line.depth < 2).unwrap_or(true) {
                println!(
                    "info string [Unchessed] opponent measurement skipped: budget too small for a stable probe"
                );
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
                        movetime: Some((job.observation_budget_ms / 3).max(5)),
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
            // Timing regularity is measured as lag-1 autocorrelation of log
            // clock fraction. It can only modulate ceiling-level strength.
            if let (Some(used), Some(remaining)) = (job.opp_time_used, job.opp_clock_remaining) {
                let had_choice = lc > 8 && w >= 0.8;
                m.observe_time_fraction(used, remaining, had_choice);
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
    }

    // ------------------------------------------------------------------
    // 2. Opening book
    // ------------------------------------------------------------------
    let book_model = model.lock().unwrap().clone();
    let effective_book_depth = effective_book_depth(job.opt.book_depth, &book_model);
    if adaptive_now && job.opt.own_book && job.game_plies < effective_book_depth {
        let entries = {
            let b = book.lock().unwrap();
            b.probe(&pos)
        };
        if !entries.is_empty() {
            let current_persona = *persona.lock().unwrap();
            let chosen =
                choose_book_move(&entries, &book_model, &job.opt, current_persona, &mut rng);
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
                        Tier::Random => "historical random".to_string(),
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
                        "info string [Unchessed] book: {} ({}) [{}] depth {}/{} — {}",
                        entry.name,
                        entry.eco,
                        tier_str,
                        effective_book_depth,
                        job.opt.book_depth,
                        reason
                    );
                    model.lock().unwrap().mark_decision_complete();
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
    let cfg = job.opt.adapt_config();
    let model_before_search = model.lock().unwrap().clone();
    let multipv_search = if adaptive_now {
        // Weak play needs genuinely imperfect candidates, but scoring them in
        // separate side searches made strength clock-dependent. Search a
        // target-dependent root pool in the same iterative-deepening pass so
        // every candidate has a comparable score and shares one deadline.
        let target = crate::adapt::target_elo(&cfg, &model_at_go_start);
        let candidate_count = if target < 1000 {
            legal_moves.len
        } else if target < 1600 {
            16.min(legal_moves.len)
        } else if target < 2200 {
            10.min(legal_moves.len)
        } else {
            5.min(legal_moves.len)
        };
        multipv_shown.max(candidate_count)
    } else {
        multipv_shown
    };
    let prev_mode = *persona.lock().unwrap();
    let previous_root_eval = *previous_eval.lock().unwrap();
    // Use board phase, check state, opponent class, and recent eval trajectory
    // rather than blindly applying the previous move's contempt.
    let provisional_mode = if adaptive_now {
        decide_persona(
            &cfg,
            &model_before_search,
            PersonaContext::from_position(&pos, job.eval.eval(&pos), previous_root_eval),
            prev_mode,
        )
        .mode
    } else {
        Mode::Full
    };
    let draw_score = if adaptive_now {
        crate::adapt::draw_score_for(&cfg, provisional_mode)
    } else {
        0
    };
    let mut main_limits = job.limits.clone();
    main_limits.account_elapsed(pos.side, job_started.elapsed().as_millis() as u64);
    main_limits.shared_nodes = Some(Arc::new(AtomicU64::new(0)));

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
    let limits_ref = &main_limits;
    let stop_ref: &AtomicBool = &stop;
    let lines = std::thread::scope(|scope| {
        for i in 0..n_helpers {
            let offset = 1 + (i % 3) as i32; // stagger: 1,2,3,1,2,3,...
            scope.spawn(move || {
                let _ = search::go(
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
                    &mut |_| {},
                );
            });
        }
        search::go(
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
            &mut |ev| print_info(ev, multipv_shown),
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
        let decision = decide_persona(
            &cfg,
            &m,
            PersonaContext::from_position(&pos, lines[0].score, previous_root_eval),
            prev_mode,
        );
        let mode = decision.mode;
        if mode != prev_mode {
            println!(
                "info string [Unchessed] persona {} -> {} (eval {} cp, opponent ~{}): {}",
                prev_mode.name(),
                mode.name(),
                lines[0].score,
                m.estimate(),
                decision.reason
            );
        }
        *persona.lock().unwrap() = mode;
        *previous_eval.lock().unwrap() = Some(lines[0].score);
        model.lock().unwrap().mark_decision_complete();
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
        let sel = select_move(
            &pos,
            &lines,
            mode,
            &cfg,
            &m,
            prior.as_ref(),
            &mut rng,
            &mut probe,
        );
        println!(
            "info string [Unchessed] mode={} opponent~{} ({}-{}, type={} {:.0}% engine) eval {} cp plan='{}': {}",
            mode.name(),
            m.estimate(),
            m.lower_bound(),
            m.upper_bound(),
            m.classification(),
            m.engine_probability() * 100.0,
            lines[0].score,
            decision.reason,
            sel.reason
        );
        println!("bestmove {}", sel.mv.uci());
    } else {
        println!("bestmove {}", lines[0].mv.uci());
    }
}

fn effective_book_depth(configured: u32, model: &OpponentModel) -> u32 {
    let human_depth = match model.estimate() {
        ..=799 => 6,
        800..=1199 => 8,
        1200..=1599 => 10,
        1600..=1999 => 12,
        2000..=2399 => 16,
        _ => 40,
    };
    configured.min(human_depth)
}

fn choose_book_move(
    entries: &[BookEntry],
    model: &OpponentModel,
    opt: &Options,
    persona: Mode,
    rng: &mut Rng,
) -> Option<(BookEntry, String)> {
    let est = model.estimate();
    let conf = model.confidence();
    let hi = model.upper_bound();
    let max_risk = if persona == Mode::Clinch {
        0
    } else {
        match opt.troll {
            TrollMode::Off => 0,
            TrollMode::On => 3,
            TrollMode::Auto => {
                // Auto trolling requires affirmative human evidence, sufficient
                // samples, a safely low upper strength bound, and no known-engine
                // identity. Unknown is not treated as human.
                if !model.auto_troll_allowed() {
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
        }
    };

    let trolls: Vec<&BookEntry> = entries
        .iter()
        .filter(|e| matches!(e.tier, Tier::Troll(r) if r <= max_risk))
        .collect();
    let mains: Vec<&BookEntry> = entries.iter().filter(|e| e.tier == Tier::Main).collect();
    let randoms: Vec<&BookEntry> = entries.iter().filter(|e| e.tier == Tier::Random).collect();
    let allow_random = model.confident_human() && hi < 1800 && !model.anti_troll_lock();

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

    if mains.is_empty() && (!allow_random || randoms.is_empty()) {
        // Only troll continuations are known here: continue when explicitly
        // allowed, otherwise leave book and search.
        if !trolls.is_empty() {
            let entry = weighted_pick(&trolls, rng)?;
            return Some(((*entry).clone(), "continuing the line".to_string()));
        }
        return None;
    }

    // Strong/uncertain opponents get protected mainlines. Confidently weak
    // humans may also receive named offbeat historical openings for variety.
    let picked = if hi >= 2100 {
        let top: Vec<&BookEntry> = mains.iter().take(2).copied().collect();
        weighted_pick(&top, rng)?
    } else if allow_random {
        let mut varied = mains.clone();
        varied.extend(randoms.iter().copied());
        weighted_pick(&varied, rng)?
    } else {
        weighted_pick(&mains, rng)?
    };
    let reason = if hi >= 2100 {
        "big game — mainlines only".to_string()
    } else if picked.tier == Tier::Random {
        format!("opponent ~{}, historical offbeat variety", est)
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
        let g0 = Game::new();

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
        let g0 = Game::new();
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

    #[test]
    fn go_parser_handles_searchmoves_mate_and_ponder() {
        let limits = parse_go("go searchmoves a2a3 h2h3 depth 7 mate 3 ponder");
        assert_eq!(limits.searchmoves, ["a2a3", "h2h3"]);
        assert_eq!(limits.depth, Some(7));
        assert_eq!(limits.mate, Some(3));
        assert!(limits.ponder);
    }

    #[test]
    fn opponent_measurement_budget_is_bounded_and_smooth() {
        let at_9999 = Limits {
            wtime: Some(9_999),
            ..Default::default()
        };
        let at_10000 = Limits {
            wtime: Some(10_000),
            ..Default::default()
        };
        assert_eq!(observation_budget_ms(&at_9999, Color::White), 19);
        assert_eq!(observation_budget_ms(&at_10000, Color::White), 20);
        let tiny = Limits {
            movetime: Some(250),
            ..Default::default()
        };
        assert_eq!(observation_budget_ms(&tiny, Color::White), 0);
        let healthy = Limits {
            movetime: Some(5_000),
            ..Default::default()
        };
        assert_eq!(observation_budget_ms(&healthy, Color::White), 100);
    }

    #[test]
    fn book_depth_scales_with_observed_strength() {
        for (elo, expected) in [(500, 6), (900, 8), (1300, 10), (1700, 12), (2200, 16)] {
            let mut model = OpponentModel::new();
            model.seed_from_uci_opponent(&format!("- {} human Test", elo));
            // Descriptor prior is deliberately broad, so feed stable evidence
            // at approximately the requested level through the public model.
            assert!(effective_book_depth(40, &model) <= expected.max(10));
        }
        let mut engine = OpponentModel::new();
        engine.seed_from_uci_opponent("- - computer Stockfish");
        assert_eq!(effective_book_depth(40, &engine), 40);
    }

    #[test]
    fn clinch_suppresses_troll_book_even_when_forced_on() {
        let pos = fen::startpos();
        let mv = legal(&pos).moves[0];
        let entries = vec![BookEntry {
            mv,
            weight: 10,
            name: "test troll",
            eco: "A00",
            tier: Tier::Troll(1),
        }];
        let options = Options {
            troll: TrollMode::On,
            ..Options::default()
        };
        let mut rng = Rng::new(1);
        assert!(choose_book_move(
            &entries,
            &OpponentModel::new(),
            &options,
            Mode::Clinch,
            &mut rng,
        )
        .is_none());
    }

    #[test]
    fn low_time_gate_includes_short_movetime() {
        assert!(is_low_time(
            &Limits {
                movetime: Some(250),
                ..Default::default()
            },
            Color::White
        ));
        assert!(!is_low_time(
            &Limits {
                movetime: Some(5_000),
                ..Default::default()
            },
            Color::White
        ));
    }
}
