//! Opening book: complete CC0 named-opening history (all 500 ECO codes),
//! curated main/troll overlays, offbeat historical variety, and optional
//! external Polyglot (.bin) popularity data.

use std::collections::{HashMap, HashSet};
use std::io::Read;

use crate::board::*;
use crate::fen;
use crate::movegen::{legal, parse_uci_move, PAWN_ATT};
use crate::polyglot_keys::POLYGLOT_RANDOM;
use crate::san::parse_san;

/// Troll risk grades: 1 = tricky-but-survivable, 2 = dubious-but-fun, 3 = pure meme.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tier {
    Main,
    /// Named but deliberately offbeat historical openings. These are eligible
    /// for variety against weaker/confidently human opponents, not big games.
    Random,
    Troll(u8),
}

#[derive(Clone, Debug)]
pub struct BookEntry {
    pub mv: Move,
    pub weight: u32,
    pub name: &'static str,
    pub eco: &'static str,
    pub tier: Tier,
}

/// Complete named-opening corpus imported from lichess-org/chess-openings
/// (CC0), pinned in books/lichess-openings/SOURCE.txt. It contains 3,810
/// named lines spanning all 500 ECO codes. Curated lines below overlay weights
/// and troll safety classifications on top of this historical corpus.
const HISTORICAL_TSV: &[&str] = &[
    include_str!("../../books/lichess-openings/a.tsv"),
    include_str!("../../books/lichess-openings/b.tsv"),
    include_str!("../../books/lichess-openings/c.tsv"),
    include_str!("../../books/lichess-openings/d.tsv"),
    include_str!("../../books/lichess-openings/e.tsv"),
];

/// Embedded repertoire. Format per line: `tag;weight;eco;name;uci moves...`
/// tag = main | troll1 | troll2 | troll3.
const EMBEDDED_LINES: &[&str] = &[
    // ---- main theory ----
    "main;80;C50;Italian Game, Giuoco Pianissimo;e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d3 d7d6",
    "main;60;C55;Italian Game, Two Knights;e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8c5",
    "main;85;C84;Ruy Lopez, Closed Morphy;e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6",
    "main;70;C67;Ruy Lopez, Berlin Defence;e2e4 e7e5 g1f3 b8c6 f1b5 g8f6 e1g1 f6e4 d2d4 e4d6 b5c6 d7c6 d4e5 d6f5 d1d8 e8d8",
    "main;90;B92;Sicilian Najdorf, Opocensky;e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 f1e2 e7e5 d4b3 f8e7",
    "main;70;B90;Sicilian Najdorf, English Attack;e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5 d4b3 c8e6",
    "main;55;B76;Sicilian Dragon, Yugoslav;e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6 c1e3 f8g7 f2f3 e8g8",
    "main;60;B33;Sicilian Sveshnikov;e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e5 d4b5 d7d6 c1g5 a7a6 b5a3 b7b5",
    "main;50;B46;Sicilian Taimanov;e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 b8c6 b1c3 d8c7",
    "main;45;B22;Sicilian Alapin;e2e4 c7c5 c2c3 g8f6 e4e5 f6d5 d2d4 c5d4 g1f3 b8c6 c3d4 d7d6",
    "main;45;B31;Sicilian Rossolimo;e2e4 c7c5 g1f3 b8c6 f1b5 g7g6 b5c6 d7c6 d2d3 f8g7 h2h3 g8f6 b1c3",
    "main;50;C02;French Advance;e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3 b8c6 g1f3 d8b6 a2a3",
    "main;45;C09;French Tarrasch Open;e2e4 e7e6 d2d4 d7d5 b1d2 c7c5 e4d5 e6d5 g1f3 b8c6 f1b5",
    "main;45;C18;French Winawer;e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4e5 c7c5 a2a3 b4c3 b2c3 g8e7",
    "main;55;B12;Caro-Kann Advance;e2e4 c7c6 d2d4 d7d5 e4e5 c8f5 g1f3 e7e6 f1e2 c6c5",
    "main;50;B19;Caro-Kann Classical;e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5 e4g3 f5g6 h2h4 h7h6 g1f3 b8d7",
    "main;40;B13;Caro-Kann Exchange;e2e4 c7c6 d2d4 d7d5 e4d5 c6d5 f1d3 b8c6 c2c3 g8f6 c1f4",
    "main;35;B01;Scandinavian, Main Line;e2e4 d7d5 e4d5 d8d5 b1c3 d5a5 d2d4 g8f6 g1f3 c7c6 f1c4 c8f5",
    "main;35;B08;Pirc Classical;e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 g1f3 f8g7 f1e2 e8g8 e1g1",
    "main;25;B09;Modern/Pirc Austrian;e2e4 g7g6 d2d4 f8g7 b1c3 d7d6 f2f4 g8f6 g1f3 e8g8",
    "main;25;B04;Alekhine Modern;e2e4 g8f6 e4e5 f6d5 d2d4 d7d6 g1f3 g7g6 f1c4 d5b6 c4b3 f8g7",
    "main;50;C42;Petroff Classical;e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4 d2d4 d6d5 f1d3 f8d6",
    "main;45;C45;Scotch, Mieses;e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6 d4c6 b7c6 e4e5 d8e7 d1e2 f6d5",
    "main;30;C49;Four Knights Spanish;e2e4 e7e5 g1f3 b8c6 b1c3 g8f6 f1b5 f8b4 e1g1 e8g8 d2d3 d7d6",
    "main;25;C29;Vienna Gambit Declined;e2e4 e7e5 b1c3 g8f6 f2f4 d7d5 f4e5 f6e4 g1f3 f8e7",
    "main;20;C39;King's Gambit Accepted;e2e4 e7e5 f2f4 e5f4 g1f3 g7g5 h2h4 g5g4 f3e5",
    "main;75;D02;London System;d2d4 d7d5 g1f3 g8f6 c1f4 c7c5 e2e3 b8c6 b1d2 e7e6 c2c3 f8d6 f4g3 e8g8",
    "main;40;A48;London vs King's Indian setup;d2d4 g8f6 c1f4 g7g6 g1f3 f8g7 e2e3 e8g8",
    "main;60;D55;Queen's Gambit Declined;d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 g1f3 h7h6 g5h4 b7b6",
    "main;45;D27;Queen's Gambit Accepted;d2d4 d7d5 c2c4 d5c4 g1f3 g8f6 e2e3 e7e6 f1c4 c7c5 e1g1 a7a6",
    "main;55;D17;Slav, Czech;d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4 a2a4 c8f5 e2e3 e7e6 f1c4 f8b4 e1g1",
    "main;50;D47;Semi-Slav Meran;d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 e7e6 e2e3 b8d7 f1d3 d5c4 d3c4 b7b5",
    "main;60;E53;Nimzo-Indian, Rubinstein;d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8 f1d3 d7d5 g1f3 c7c5 e1g1 b8c6",
    "main;45;E15;Queen's Indian;d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 g2g3 c8a6 b2b3 f8b4 c1d2 b4e7 f1g2",
    "main;60;E97;King's Indian, Classical Mar del Plata;d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2 e7e5 e1g1 b8c6 d4d5 c6e7",
    "main;50;D87;Gruenfeld Exchange;d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5 e2e4 d5c3 b2c3 f8g7 f1c4 c7c5 g1e2 b8c6",
    "main;30;A61;Modern Benoni;d2d4 g8f6 c2c4 c7c5 d4d5 e7e6 b1c3 e6d5 c4d5 d7d6 g1f3 g7g6",
    "main;25;A87;Dutch Leningrad;d2d4 f7f5 g2g3 g8f6 f1g2 g7g6 g1f3 f8g7 e1g1 e8g8 c2c4 d7d6",
    "main;50;E04;Catalan Open;d2d4 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2 f8e7 g1f3 e8g8 e1g1 d5c4 d1c2 a7a6 c2c4 b7b5 c4c2 c8b7",
    "main;40;A29;English Four Knights;c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 d7d5 c4d5 f6d5 f1g2 d5b6 e1g1 f8e7",
    "main;35;A37;English Symmetrical;c2c4 c7c5 g1f3 g8f6 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7 e1g1 e8g8",
    "main;35;A12;Reti, Anglo-Slav;g1f3 d7d5 c2c4 c7c6 b2b3 g8f6 g2g3 c8f5 f1g2 e7e6 c1b2 h7h6 e1g1",
    "main;20;A45;Trompowsky;d2d4 g8f6 c1g5 f6e4 g5f4 d7d5 e2e3 c7c5 f1d3 e4f6",
    "main;45;D35;QGD Exchange;d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c4d5 e6d5 c1g5 f8e7 e2e3 c7c6 f1d3 b8d7",
    "main;25;D05;Colle-Zukertort;d2d4 d7d5 g1f3 g8f6 e2e3 e7e6 f1d3 c7c5 b2b3 b8c6 c1b2 f8d6 e1g1 e8g8",
    // ---- troll repertoire ----
    "troll3;5;C20;Bongcloud Attack;e2e4 e7e5 e1e2",
    "troll3;3;C20;Double Bongcloud;e2e4 e7e5 e1e2 b8c6 e2e1",
    "troll2;8;C20;Scholar's Mate Attempt (Qh5);e2e4 e7e5 d1h5 b8c6 f1c4 g7g6 h5f3 g8f6 f3b3",
    "troll2;4;C20;Wayward Queen, Kiddie Countergambit punish;e2e4 e7e5 d1h5 g8f6 h5e5 f8e7",
    "troll3;6;C23;Scholar's Mate Attempt (Qf3);e2e4 e7e5 f1c4 f8c5 d1f3",
    "troll2;5;A00;Grob Attack;g2g4 d7d5 f1g2 c8g4 c2c4",
    "troll2;5;A00;Cow Opening;e2e3 e7e5 g1e2 d7d5 d2d3 b8c6 b1d2",
    "troll2;5;A40;Englund Gambit;d2d4 e7e5 d4e5 b8c6 g1f3 d8e7 c1f4 e7b4 f4d2 b4b2",
    "troll1;10;C42;Stafford Gambit;e2e4 e7e5 g1f3 g8f6 f3e5 b8c6 e5c6 d7c6",
    "troll3;4;C46;Halloween Gambit;e2e4 e7e5 g1f3 b8c6 b1c3 g8f6 f3e5 c6e5 d2d4",
    "troll2;4;C40;Latvian Gambit;e2e4 e7e5 g1f3 f7f5",
    "troll2;4;A02;From's Gambit;f2f4 e7e5 f4e5 d7d6 e5d6 f8d6 g1f3 g7g5",
    "troll3;3;C20;Harry the h-pawn;e2e4 e7e5 h2h4",
    "troll1;12;C57;Fried Liver Attack;e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 f3g5 d7d5 e4d5 f6d5 g5f7 e8f7 d1f3 f7e6 b1c3",
    "troll2;6;C21;Danish Gambit;e2e4 e7e5 d2d4 e5d4 c2c3 d4c3 f1c4 c3b2 c1b2",
];

pub struct Book {
    embedded: HashMap<u64, Vec<BookEntry>>,
    poly: Option<PolyglotBook>,
    historical_lines: usize,
    eco_codes: usize,
}

fn historical_tier(first_move: &str) -> Tier {
    match first_move {
        // The broad, established first-move families remain serious theory.
        "e2e4" | "d2d4" | "c2c4" | "g1f3" | "g2g3" | "b2b3" | "f2f4" | "b1c3" => Tier::Main,
        // Named flank/novelty openings still belong in the historical book,
        // but are offered only as variety rather than big-game mainlines.
        _ => Tier::Random,
    }
}

fn historical_tokens(pgn: &'static str) -> Vec<&'static str> {
    pgn.split_whitespace()
        .filter_map(|token| {
            let san = token.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.');
            (!san.is_empty()).then_some(san)
        })
        .collect()
}

fn add_historical_line(
    map: &mut HashMap<u64, Vec<BookEntry>>,
    eco: &'static str,
    name: &'static str,
    pgn: &'static str,
) -> Result<(), String> {
    let tokens = historical_tokens(pgn);
    let mut pos = fen::startpos();
    let mut first_move = None;
    for (index, san) in tokens.iter().enumerate() {
        let mv = parse_san(&pos, san)
            .ok_or_else(|| format!("illegal historical SAN '{}' in '{}': {}", san, name, pgn))?;
        let first = first_move.get_or_insert_with(|| mv.uci());
        let tier = historical_tier(first);
        let is_named_position = index + 1 == tokens.len();
        let entry_name = if is_named_position {
            name
        } else {
            "Opening database"
        };
        let entries = map.entry(pos.hash).or_default();
        if let Some(entry) = entries.iter_mut().find(|entry| entry.mv == mv) {
            entry.weight = entry.weight.saturating_add(1);
            if tier == Tier::Main && entry.tier == Tier::Random {
                entry.tier = Tier::Main;
            }
            if is_named_position {
                entry.name = name;
                entry.eco = eco;
            }
        } else {
            entries.push(BookEntry {
                mv,
                weight: 1,
                name: entry_name,
                eco,
                tier,
            });
        }
        pos = pos.make(mv);
    }
    Ok(())
}

fn parse_curated(
    line: &'static str,
) -> Result<(&'static str, u32, &'static str, &'static str, &'static str), String> {
    let mut parts = line.splitn(5, ';');
    let tag = parts.next().ok_or("missing tag")?;
    let weight = parts
        .next()
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| format!("bad weight in: {}", line))?;
    let eco = parts.next().ok_or("missing eco")?;
    let name = parts.next().ok_or("missing name")?;
    let moves = parts.next().ok_or("missing moves")?;
    Ok((tag, weight, eco, name, moves))
}

impl Book {
    /// Build the complete CC0 historical corpus, then overlay curated serious
    /// weights and explicitly safety-graded troll lines.
    pub fn new() -> Result<Book, String> {
        let mut map: HashMap<u64, Vec<BookEntry>> = HashMap::new();
        let mut historical_lines = 0usize;
        let mut eco_codes: HashSet<&'static str> = HashSet::new();
        for tsv in HISTORICAL_TSV {
            for row in tsv.lines().skip(1).filter(|line| !line.trim().is_empty()) {
                let mut fields = row.splitn(3, '\t');
                let eco = fields.next().ok_or("historical row missing ECO")?;
                let name = fields.next().ok_or("historical row missing name")?;
                let pgn = fields.next().ok_or("historical row missing PGN")?;
                add_historical_line(&mut map, eco, name, pgn)?;
                historical_lines += 1;
                eco_codes.insert(eco);
            }
        }

        // Serious curated paths establish protected mainline keys first.
        let mut curated_main: HashSet<(u64, u16)> = HashSet::new();
        for line in EMBEDDED_LINES {
            let (tag, weight, eco, name, moves) = parse_curated(line)?;
            if tag != "main" {
                continue;
            }
            let mut pos = fen::startpos();
            for token in moves.split_whitespace() {
                let mv = parse_uci_move(&pos, token)
                    .ok_or_else(|| format!("illegal curated move '{}' in '{}'", token, name))?;
                curated_main.insert((pos.hash, mv.0));
                let entries = map.entry(pos.hash).or_default();
                if let Some(entry) = entries.iter_mut().find(|entry| entry.mv == mv) {
                    entry.tier = Tier::Main;
                    entry.weight = entry.weight.max(weight);
                    entry.name = name;
                    entry.eco = eco;
                } else {
                    entries.push(BookEntry {
                        mv,
                        weight,
                        name,
                        eco,
                        tier: Tier::Main,
                    });
                }
                pos = pos.make(mv);
            }
        }

        // Troll overlays never downgrade a move protected by a curated mainline.
        for line in EMBEDDED_LINES {
            let (tag, weight, eco, name, moves) = parse_curated(line)?;
            let risk = match tag {
                "main" => continue,
                "troll1" => 1,
                "troll2" => 2,
                "troll3" => 3,
                other => return Err(format!("bad tag '{}' in: {}", other, line)),
            };
            let mut pos = fen::startpos();
            for token in moves.split_whitespace() {
                let mv = parse_uci_move(&pos, token)
                    .ok_or_else(|| format!("illegal curated move '{}' in '{}'", token, name))?;
                let protected = curated_main.contains(&(pos.hash, mv.0));
                let entries = map.entry(pos.hash).or_default();
                if let Some(entry) = entries.iter_mut().find(|entry| entry.mv == mv) {
                    let replace_label =
                        !matches!(entry.tier, Tier::Troll(_)) || weight > entry.weight;
                    entry.weight = entry.weight.max(weight);
                    if !protected {
                        entry.tier = Tier::Troll(risk);
                        if replace_label {
                            entry.name = name;
                            entry.eco = eco;
                        }
                    }
                } else {
                    entries.push(BookEntry {
                        mv,
                        weight,
                        name,
                        eco,
                        tier: if protected {
                            Tier::Main
                        } else {
                            Tier::Troll(risk)
                        },
                    });
                }
                pos = pos.make(mv);
            }
        }

        Ok(Book {
            embedded: map,
            poly: None,
            historical_lines,
            eco_codes: eco_codes.len(),
        })
    }

    pub fn historical_lines(&self) -> usize {
        self.historical_lines
    }

    pub fn eco_codes(&self) -> usize {
        self.eco_codes
    }

    pub fn load_polyglot(&mut self, path: &str) -> Result<usize, String> {
        let pb = PolyglotBook::load(path)?;
        let n = pb.entries.len();
        self.poly = Some(pb);
        Ok(n)
    }

    pub fn unload_polyglot(&mut self) {
        self.poly = None;
    }

    pub fn has_polyglot(&self) -> bool {
        self.poly.is_some()
    }

    /// All historical/curated moves plus optional Polyglot popularity data.
    pub fn probe(&self, pos: &Position) -> Vec<BookEntry> {
        let mut out: Vec<BookEntry> = self.embedded.get(&pos.hash).cloned().unwrap_or_default();
        if let Some(pb) = &self.poly {
            for (mv, weight) in pb.probe(pos) {
                if let Some(entry) = out.iter_mut().find(|entry| entry.mv == mv) {
                    entry.weight = entry.weight.max(weight as u32);
                } else {
                    out.push(BookEntry {
                        mv,
                        weight: weight as u32,
                        name: "book file",
                        eco: "",
                        tier: Tier::Main,
                    });
                }
            }
        }
        out.sort_by_key(|entry| std::cmp::Reverse(entry.weight));
        out
    }
}

// ---------------------------------------------------------------------------
// Polyglot format
// ---------------------------------------------------------------------------

pub struct PolyglotBook {
    /// (key, raw move, weight), sorted by key.
    entries: Vec<(u64, u16, u16)>,
}

impl PolyglotBook {
    pub fn load(path: &str) -> Result<PolyglotBook, String> {
        let mut f = std::fs::File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        if buf.len() % 16 != 0 {
            return Err(format!("{}: size not a multiple of 16", path));
        }
        let mut entries = Vec::with_capacity(buf.len() / 16);
        for chunk in buf.chunks_exact(16) {
            let key = u64::from_be_bytes(chunk[0..8].try_into().unwrap());
            let mv = u16::from_be_bytes(chunk[8..10].try_into().unwrap());
            let weight = u16::from_be_bytes(chunk[10..12].try_into().unwrap());
            entries.push((key, mv, weight));
        }
        entries.sort_by_key(|e| e.0);
        Ok(PolyglotBook { entries })
    }

    /// Book moves for this position, decoded against its legal moves.
    pub fn probe(&self, pos: &Position) -> Vec<(Move, u16)> {
        let key = polyglot_key(pos);
        let start = self.entries.partition_point(|e| e.0 < key);
        let legal_moves = legal(pos);
        let mut out = Vec::new();
        for &(k, raw, w) in &self.entries[start..] {
            if k != key {
                break;
            }
            if let Some(mv) = decode_poly_move(pos, raw, &legal_moves) {
                out.push((mv, w.max(1)));
            }
        }
        out
    }
}

fn decode_poly_move(
    pos: &Position,
    raw: u16,
    legal_moves: &crate::movegen::MoveList,
) -> Option<Move> {
    let to_file = (raw & 7) as u8;
    let to_row = ((raw >> 3) & 7) as u8;
    let from_file = ((raw >> 6) & 7) as u8;
    let from_row = ((raw >> 9) & 7) as u8;
    let promo = ((raw >> 12) & 7) as usize;
    let from = sq(from_file, from_row);
    let mut to = sq(to_file, to_row);

    // Polyglot encodes castling as king-takes-rook (e1h1, e1a1, e8h8, e8a8).
    if let Some((c, KING)) = pos.piece_on(from) {
        let is_castle_encoding = match c {
            Color::White => from == 4 && (to == 7 || to == 0),
            Color::Black => from == 60 && (to == 63 || to == 56),
        };
        if is_castle_encoding {
            to = if to > from { from + 2 } else { from - 2 };
            return legal_moves
                .as_slice()
                .iter()
                .copied()
                .find(|m| m.kind() == MK_CASTLE && m.from() == from && m.to() == to);
        }
    }

    if promo > 0 {
        let piece = match promo {
            1 => KNIGHT,
            2 => BISHOP,
            3 => ROOK,
            4 => QUEEN,
            _ => return None,
        };
        return legal_moves.as_slice().iter().copied().find(|m| {
            m.is_promo() && m.from() == from && m.to() == to && m.promo_piece() == piece
        });
    }

    legal_moves
        .as_slice()
        .iter()
        .copied()
        .find(|m| !m.is_promo() && m.from() == from && m.to() == to)
}

/// The Polyglot Zobrist key of a position (distinct from our internal hash).
pub fn polyglot_key(pos: &Position) -> u64 {
    let mut key = 0u64;
    for s in 0..64u8 {
        if let Some((c, p)) = pos.piece_on(s) {
            // kind_of_piece: black pawn 0, white pawn 1, black knight 2, ...
            let kind = p * 2 + if let Color::White = c { 1 } else { 0 };
            key ^= POLYGLOT_RANDOM[64 * kind + s as usize];
        }
    }
    if pos.castling & WK != 0 {
        key ^= POLYGLOT_RANDOM[768];
    }
    if pos.castling & WQ != 0 {
        key ^= POLYGLOT_RANDOM[769];
    }
    if pos.castling & BK != 0 {
        key ^= POLYGLOT_RANDOM[770];
    }
    if pos.castling & BQ != 0 {
        key ^= POLYGLOT_RANDOM[771];
    }
    // ep file hashed only when a side-to-move pawn can actually capture
    if pos.ep != NO_EP {
        let can_capture =
            PAWN_ATT[pos.side.flip().idx()][pos.ep as usize] & pos.pieces(pos.side, PAWN) != 0;
        if can_capture {
            key ^= POLYGLOT_RANDOM[772 + file_of(pos.ep) as usize];
        }
    }
    if let Color::White = pos.side {
        key ^= POLYGLOT_RANDOM[780];
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen;
    use crate::movegen::parse_uci_move;

    #[test]
    fn embedded_book_builds() {
        let book = Book::new().expect("embedded book must be fully legal");
        assert_eq!(book.historical_lines(), 3_810);
        assert_eq!(book.eco_codes(), 500);
        let start = fen::startpos();
        let entries = book.probe(&start);
        assert!(!entries.is_empty());
        // e4 and d4 are protected main theory; Nh3 is available only as
        // offbeat historical variety.
        for m in ["e2e4", "d2d4"] {
            let e = entries.iter().find(|e| e.mv.uci() == m).unwrap();
            assert_eq!(e.tier, Tier::Main, "{} must be Main tier", m);
        }
        let nh3 = entries.iter().find(|e| e.mv.uci() == "g1h3").unwrap();
        assert_eq!(nh3.tier, Tier::Random);
    }

    #[test]
    fn bongcloud_is_meme_tier() {
        let book = Book::new().unwrap();
        let mut pos = fen::startpos();
        for m in ["e2e4", "e7e5"] {
            pos = pos.make(parse_uci_move(&pos, m).unwrap());
        }
        let entries = book.probe(&pos);
        let ke2 = entries
            .iter()
            .find(|e| e.mv.uci() == "e1e2")
            .expect("Bongcloud in book");
        assert_eq!(ke2.tier, Tier::Troll(3));
        assert_eq!(ke2.name, "Bongcloud Attack");
        // ...and the serious mainline move must still be Main tier
        let nf3 = entries.iter().find(|e| e.mv.uci() == "g1f3").unwrap();
        assert_eq!(nf3.tier, Tier::Main);
    }

    /// Known key vectors from the Polyglot book format specification.
    #[test]
    fn polyglot_key_vectors() {
        let cases: &[(&[&str], u64)] = &[
            (&[], 0x463b96181691fc9c),
            (&["e2e4"], 0x823c9b50fd114196),
            (&["e2e4", "d7d5"], 0x0756b94461c50fb0),
            (&["e2e4", "d7d5", "e4e5"], 0x662fafb965db29d4),
            (&["e2e4", "d7d5", "e4e5", "f7f5"], 0x22a48b5a8e47ff78),
            (
                &["e2e4", "d7d5", "e4e5", "f7f5", "e1e2"],
                0x652a607ca3f242c1,
            ),
            (
                &["e2e4", "d7d5", "e4e5", "f7f5", "e1e2", "e8f7"],
                0x00fdd303c946bdd9,
            ),
            (
                &["a2a4", "b7b5", "h2h4", "b5b4", "c2c4"],
                0x3c8123ea7b067637,
            ),
            (
                &["a2a4", "b7b5", "h2h4", "b5b4", "c2c4", "b4c3", "a1a3"],
                0x5c3f9b829b279560,
            ),
        ];
        for (moves, expected) in cases {
            let mut pos = fen::startpos();
            for m in *moves {
                let mv = parse_uci_move(&pos, m).unwrap_or_else(|| panic!("illegal {}", m));
                pos = pos.make(mv);
            }
            assert_eq!(
                polyglot_key(&pos),
                *expected,
                "polyglot key mismatch after {:?}",
                moves
            );
        }
    }
}
