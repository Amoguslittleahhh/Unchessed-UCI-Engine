import sys

search_path = sys.argv[1]
uci_path = sys.argv[2]

with open(search_path, "r", encoding="utf-8") as f:
    src = f.read()


def rreplace(text, old, new, label):
    if old not in text:
        raise AssertionError(f"pattern not found: {label}")
    return text.replace(old, new)


# 1. SearchParams struct fields
src = rreplace(
    src,
    '''    /// plain futility pruning only applies at or below this depth
    pub futility_max_depth: i32,
    /// singular extensions only apply from this depth
    pub singular_min_depth: i32,
    /// TT entry depth must be within this many plies of current depth
    pub singular_depth_margin: i32,
    /// v2 (depth-scaled, per Stockfish/RubiChess): the verification target
    /// window is tt_score - singular_margin_per_depth*depth, replacing the
    /// flat margin from the v1 attempt that failed its SPRT gate — both
    /// reference engines scale this margin by depth, not a fixed constant.
    pub singular_margin_per_depth: i32,
    /// extra margin subtracted at PV nodes (both reference engines widen
    /// the verification window at PV/tt-PV nodes, same spirit as our
    /// existing `is_pv` distinction elsewhere in the search)
    pub singular_pv_margin: i32,
}''',
    '''    /// plain futility pruning only applies at or below this depth
    pub futility_max_depth: i32,
}''',
    "SearchParams fields",
)

# 2. defaults
src = rreplace(
    src,
    '''            futility_margin: 150,
            futility_max_depth: 8,
            singular_min_depth: 6,
            singular_depth_margin: 3,
            // ~24 at depth 8 (the common trigger depth), in the same
            // ballpark as the flat margin (17) the failed v1 attempt's SPSA
            // run converged to at that depth, but now scaling with depth
            // instead of staying fixed as depth grows.
            singular_margin_per_depth: 3,
            singular_pv_margin: 20,
        }''',
    '''            futility_margin: 150,
            futility_max_depth: 8,
        }''',
    "defaults",
)

# 3. Searcher struct field
src = rreplace(
    src,
    '''    killers: [[Move; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    /// static eval at each ply of the current line, used to compute the
    /// "improving" flag (are we doing better now than 2 plies ago, i.e.
    /// before the opponent's reply) -- cheap, and used by the depth-scaled
    /// singular-extension margin below.
    static_eval_stack: [i32; MAX_PLY],
    /// hashes of game positions + current search path (ancestors of the node)
    path: Vec<u64>,
    pv_table: [[Move; MAX_PLY]; MAX_PLY],
    pv_len: [usize; MAX_PLY],
}''',
    '''    killers: [[Move; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    /// hashes of game positions + current search path (ancestors of the node)
    path: Vec<u64>,
    pv_table: [[Move; MAX_PLY]; MAX_PLY],
    pv_len: [usize; MAX_PLY],
}''',
    "Searcher struct field",
)

# 4. Searcher construction
src = rreplace(
    src,
    '''        killers: [[Move::NONE; 2]; MAX_PLY],
        history: [[[0; 64]; 64]; 2],
        static_eval_stack: [0; MAX_PLY],
        path: history.to_vec(),''',
    '''        killers: [[Move::NONE; 2]; MAX_PLY],
        history: [[[0; 64]; 64]; 2],
        path: history.to_vec(),''',
    "Searcher construction",
)

# 5. TT probe: revert tt_entry retention
src = rreplace(
    src,
    '''        // TT probe
        let mut tt_mv = Move::NONE;
        let tt_entry = self.tt.probe(pos.hash);
        if let Some(e) = tt_entry {
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

        let static_eval = self.eval.eval(pos);
        self.static_eval_stack[ply] = static_eval;
        // are we doing better now than 2 plies ago (before the opponent's
        // last reply)? Cheap signal, used below by the singular-extension
        // verification window -- per Stockfish/RubiChess, both widen or
        // tighten several margins based on this, we only wire it into
        // singular extensions here to keep this pass narrowly scoped.
        let improving = ply >= 2 && static_eval > self.static_eval_stack[ply - 2];

        // reverse futility pruning''',
    '''        // TT probe
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

        let static_eval = self.eval.eval(pos);

        // reverse futility pruning''',
    "TT probe / static_eval / improving",
)

# 6. remove the entire singular extension block
start_marker = "        // Singular extensions v2:"
end_marker = "        // ProbCut: before committing to the full-depth search, check whether"
start_idx = src.index(start_marker)
end_idx = src.index(end_marker)
src = src[:start_idx] + src[end_idx:]

# 7. main loop ext computation
src = rreplace(
    src,
    '''            let ext = if gives_check { 1 } else { 0 }
                + if m == tt_mv { singular_ext } else { 0 };
            let nd = depth - 1 + ext;''',
    '''            let ext = if gives_check { 1 } else { 0 };
            let nd = depth - 1 + ext;''',
    "main loop ext",
)

with open(search_path, "w", encoding="utf-8") as f:
    f.write(src)

# --- uci.rs ---
with open(uci_path, "r", encoding="utf-8") as f:
    usrc = f.read()

usrc = rreplace(
    usrc,
    '''                println!("option name SingularMinDepth type spin default 6 min 3 max 12");
                println!("option name SingularDepthMargin type spin default 3 min 0 max 6");
                println!("option name SingularMarginPerDepth type spin default 3 min 1 max 20");
                println!("option name SingularPvMargin type spin default 20 min 0 max 100");
                println!("uciok");''',
    '''                println!("uciok");''',
    "uci option printouts",
)

usrc = rreplace(
    usrc,
    '''        "singularmindepth" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.search.singular_min_depth = v.clamp(3, 12);
            }
        }
        "singulardepthmargin" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.search.singular_depth_margin = v.clamp(0, 6);
            }
        }
        "singularmarginperdepth" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.search.singular_margin_per_depth = v.clamp(1, 20);
            }
        }
        "singularpvmargin" => {
            if let Ok(v) = value.parse::<i32>() {
                opt.search.singular_pv_margin = v.clamp(0, 100);
            }
        }''',
    "",
    "uci setoption handlers",
)

with open(uci_path, "w", encoding="utf-8") as f:
    f.write(usrc)

print("reverted OK")
