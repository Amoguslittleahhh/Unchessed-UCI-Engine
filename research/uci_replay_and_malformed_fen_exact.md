# Exact UCI Replay and Malformed-FEN Evidence

Engine: `/home/ubuntu/unchessed_audit/main_repo/target/debug/unchessed-adapter`

Each section was run in an isolated process; the extractor waited for `bestmove` before sending `quit`.
## en_passant_replay

### Exact command stream
```text
uci
setoption name OwnBook value false
setoption name Adaptive value false
setoption name Troll value Off
isready
position startpos moves e2e4 a7a6 e4e5 d7d5
go depth 1
```

### Exact engine output
```text
id name Unchessed Game Adapter 0.2.3
id author Unchessed AI project
option name Hash type spin default 128 min 1 max 2048
option name Threads type spin default 6 min 1 max 64
option name Clear Hash type button
option name MultiPV type spin default 1 min 1 max 8
option name EvalFile type string default 
option name UnarchitecturedHint type check default false
option name UnarchitecturedHintExit type string default 2/128
option name UnarchitecturedFile type string default 
option name UnarchitecturedMinTime type spin default 30000 min 1000 max 600000
option name Adaptive type check default true
option name UCI_LimitStrength type check default false
option name UCI_Elo type spin default 2400 min 500 max 3200
option name Contempt type spin default 25 min 0 max 100
option name Troll type combo default Auto var Off var Auto var On
option name OwnBook type check default true
option name BookFile type string default 
option name BookDepth type spin default 16 min 0 max 40
option name PolicyFile type string default 
option name UCI_Opponent type string default 
option name PersonaSmooth type check default false
option name EngineDetectV2 type check default false
option name AdapterTelemetry type check default false
option name RFPMargin type spin default 90 min 10 max 300
option name NullMoveBase type spin default 3 min 1 max 6
option name NullMoveDivisor type spin default 6 min 2 max 12
option name LMRMinDepth type spin default 3 min 1 max 8
option name LMRMinMoveNumber type spin default 3 min 0 max 20
option name LMRBigMoveNumber type spin default 12 min 4 max 40
option name AspirationDelta type spin default 25 min 5 max 200
option name AspirationMinDepth type spin default 4 min 1 max 12
option name ProbCutMargin type spin default 200 min 50 max 400
option name ProbCutReduction type spin default 4 min 2 max 6
option name ProbCutMinDepth type spin default 5 min 3 max 10
option name FutilityMargin type spin default 150 min 30 max 400
option name FutilityMaxDepth type spin default 8 min 1 max 12
option name ProbcutSeeFilter type check default false
option name PassedPawnMgPct type spin default 100 min 0 max 200
option name PassedPawnEgPct type spin default 100 min 0 max 200
option name MobilityPct type spin default 100 min 0 max 200
option name RookPct type spin default 100 min 0 max 200
option name KnightOutpostPct type spin default 100 min 0 max 200
uciok
info string [Unchessed] eval: hand-crafted (no NNUE file found)
info string [Unchessed] no policy net found — using heuristic move priors
readyok
info depth 1 multipv 1 score cp 80 nodes 35 nps 35000 hashfull 0 time 0 pv d2d4
bestmove d2d4
```

## castle_replay

### Exact command stream
```text
uci
setoption name OwnBook value false
setoption name Adaptive value false
setoption name Troll value Off
isready
position startpos moves e2e4 e7e5 g1f3 b8c6 f1e2 g8f6 e1g1
go depth 1
```

### Exact engine output
```text
id name Unchessed Game Adapter 0.2.3
id author Unchessed AI project
option name Hash type spin default 128 min 1 max 2048
option name Threads type spin default 6 min 1 max 64
option name Clear Hash type button
option name MultiPV type spin default 1 min 1 max 8
option name EvalFile type string default 
option name UnarchitecturedHint type check default false
option name UnarchitecturedHintExit type string default 2/128
option name UnarchitecturedFile type string default 
option name UnarchitecturedMinTime type spin default 30000 min 1000 max 600000
option name Adaptive type check default true
option name UCI_LimitStrength type check default false
option name UCI_Elo type spin default 2400 min 500 max 3200
option name Contempt type spin default 25 min 0 max 100
option name Troll type combo default Auto var Off var Auto var On
option name OwnBook type check default true
option name BookFile type string default 
option name BookDepth type spin default 16 min 0 max 40
option name PolicyFile type string default 
option name UCI_Opponent type string default 
option name PersonaSmooth type check default false
option name EngineDetectV2 type check default false
option name AdapterTelemetry type check default false
option name RFPMargin type spin default 90 min 10 max 300
option name NullMoveBase type spin default 3 min 1 max 6
option name NullMoveDivisor type spin default 6 min 2 max 12
option name LMRMinDepth type spin default 3 min 1 max 8
option name LMRMinMoveNumber type spin default 3 min 0 max 20
option name LMRBigMoveNumber type spin default 12 min 4 max 40
option name AspirationDelta type spin default 25 min 5 max 200
option name AspirationMinDepth type spin default 4 min 1 max 12
option name ProbCutMargin type spin default 200 min 50 max 400
option name ProbCutReduction type spin default 4 min 2 max 6
option name ProbCutMinDepth type spin default 5 min 3 max 10
option name FutilityMargin type spin default 150 min 30 max 400
option name FutilityMaxDepth type spin default 8 min 1 max 12
option name ProbcutSeeFilter type check default false
option name PassedPawnMgPct type spin default 100 min 0 max 200
option name PassedPawnEgPct type spin default 100 min 0 max 200
option name MobilityPct type spin default 100 min 0 max 200
option name RookPct type spin default 100 min 0 max 200
option name KnightOutpostPct type spin default 100 min 0 max 200
uciok
info string [Unchessed] eval: hand-crafted (no NNUE file found)
info string [Unchessed] no policy net found — using heuristic move priors
readyok
info depth 1 multipv 1 score cp 109 nodes 41 nps 41000 hashfull 0 time 0 pv f6e4
bestmove f6e4
```

## Malformed-FEN probes

## missing_black_king

### Exact command stream
```text
uci
setoption name OwnBook value false
setoption name Adaptive value false
setoption name Troll value Off
isready
position startpos
go depth 1
position fen 4k3/8/8/8/8/8/8/4K3 w - - 0 1
go depth 1
```

### Exact engine output
```text
id name Unchessed Game Adapter 0.2.3
id author Unchessed AI project
option name Hash type spin default 128 min 1 max 2048
option name Threads type spin default 6 min 1 max 64
option name Clear Hash type button
option name MultiPV type spin default 1 min 1 max 8
option name EvalFile type string default 
option name UnarchitecturedHint type check default false
option name UnarchitecturedHintExit type string default 2/128
option name UnarchitecturedFile type string default 
option name UnarchitecturedMinTime type spin default 30000 min 1000 max 600000
option name Adaptive type check default true
option name UCI_LimitStrength type check default false
option name UCI_Elo type spin default 2400 min 500 max 3200
option name Contempt type spin default 25 min 0 max 100
option name Troll type combo default Auto var Off var Auto var On
option name OwnBook type check default true
option name BookFile type string default 
option name BookDepth type spin default 16 min 0 max 40
option name PolicyFile type string default 
option name UCI_Opponent type string default 
option name PersonaSmooth type check default false
option name EngineDetectV2 type check default false
option name AdapterTelemetry type check default false
option name RFPMargin type spin default 90 min 10 max 300
option name NullMoveBase type spin default 3 min 1 max 6
option name NullMoveDivisor type spin default 6 min 2 max 12
option name LMRMinDepth type spin default 3 min 1 max 8
option name LMRMinMoveNumber type spin default 3 min 0 max 20
option name LMRBigMoveNumber type spin default 12 min 4 max 40
option name AspirationDelta type spin default 25 min 5 max 200
option name AspirationMinDepth type spin default 4 min 1 max 12
option name ProbCutMargin type spin default 200 min 50 max 400
option name ProbCutReduction type spin default 4 min 2 max 6
option name ProbCutMinDepth type spin default 5 min 3 max 10
option name FutilityMargin type spin default 150 min 30 max 400
option name FutilityMaxDepth type spin default 8 min 1 max 12
option name ProbcutSeeFilter type check default false
option name PassedPawnMgPct type spin default 100 min 0 max 200
option name PassedPawnEgPct type spin default 100 min 0 max 200
option name MobilityPct type spin default 100 min 0 max 200
option name RookPct type spin default 100 min 0 max 200
option name KnightOutpostPct type spin default 100 min 0 max 200
uciok
info string [Unchessed] eval: hand-crafted (no NNUE file found)
info string [Unchessed] no policy net found — using heuristic move priors
readyok
bestmove a2a3
```

## two_white_kings

### Exact command stream
```text
uci
setoption name OwnBook value false
setoption name Adaptive value false
setoption name Troll value Off
isready
position startpos
go depth 1
position fen 4k3/8/8/8/8/8/4K3/4K3 w - - 0 1
go depth 1
```

### Exact engine output
```text
id name Unchessed Game Adapter 0.2.3
id author Unchessed AI project
option name Hash type spin default 128 min 1 max 2048
option name Threads type spin default 6 min 1 max 64
option name Clear Hash type button
option name MultiPV type spin default 1 min 1 max 8
option name EvalFile type string default 
option name UnarchitecturedHint type check default false
option name UnarchitecturedHintExit type string default 2/128
option name UnarchitecturedFile type string default 
option name UnarchitecturedMinTime type spin default 30000 min 1000 max 600000
option name Adaptive type check default true
option name UCI_LimitStrength type check default false
option name UCI_Elo type spin default 2400 min 500 max 3200
option name Contempt type spin default 25 min 0 max 100
option name Troll type combo default Auto var Off var Auto var On
option name OwnBook type check default true
option name BookFile type string default 
option name BookDepth type spin default 16 min 0 max 40
option name PolicyFile type string default 
option name UCI_Opponent type string default 
option name PersonaSmooth type check default false
option name EngineDetectV2 type check default false
option name AdapterTelemetry type check default false
option name RFPMargin type spin default 90 min 10 max 300
option name NullMoveBase type spin default 3 min 1 max 6
option name NullMoveDivisor type spin default 6 min 2 max 12
option name LMRMinDepth type spin default 3 min 1 max 8
option name LMRMinMoveNumber type spin default 3 min 0 max 20
option name LMRBigMoveNumber type spin default 12 min 4 max 40
option name AspirationDelta type spin default 25 min 5 max 200
option name AspirationMinDepth type spin default 4 min 1 max 12
option name ProbCutMargin type spin default 200 min 50 max 400
option name ProbCutReduction type spin default 4 min 2 max 6
option name ProbCutMinDepth type spin default 5 min 3 max 10
option name FutilityMargin type spin default 150 min 30 max 400
option name FutilityMaxDepth type spin default 8 min 1 max 12
option name ProbcutSeeFilter type check default false
option name PassedPawnMgPct type spin default 100 min 0 max 200
option name PassedPawnEgPct type spin default 100 min 0 max 200
option name MobilityPct type spin default 100 min 0 max 200
option name RookPct type spin default 100 min 0 max 200
option name KnightOutpostPct type spin default 100 min 0 max 200
uciok
info string [Unchessed] eval: hand-crafted (no NNUE file found)
info string [Unchessed] no policy net found — using heuristic move priors
readyok
bestmove a2a3
```

## bad_castling_rights

### Exact command stream
```text
uci
setoption name OwnBook value false
setoption name Adaptive value false
setoption name Troll value Off
isready
position startpos
go depth 1
position fen 4k3/8/8/8/8/8/8/4K3 w K - 0 1
go depth 1
```

### Exact engine output
```text
id name Unchessed Game Adapter 0.2.3
id author Unchessed AI project
option name Hash type spin default 128 min 1 max 2048
option name Threads type spin default 6 min 1 max 64
option name Clear Hash type button
option name MultiPV type spin default 1 min 1 max 8
option name EvalFile type string default 
option name UnarchitecturedHint type check default false
option name UnarchitecturedHintExit type string default 2/128
option name UnarchitecturedFile type string default 
option name UnarchitecturedMinTime type spin default 30000 min 1000 max 600000
option name Adaptive type check default true
option name UCI_LimitStrength type check default false
option name UCI_Elo type spin default 2400 min 500 max 3200
option name Contempt type spin default 25 min 0 max 100
option name Troll type combo default Auto var Off var Auto var On
option name OwnBook type check default true
option name BookFile type string default 
option name BookDepth type spin default 16 min 0 max 40
option name PolicyFile type string default 
option name UCI_Opponent type string default 
option name PersonaSmooth type check default false
option name EngineDetectV2 type check default false
option name AdapterTelemetry type check default false
option name RFPMargin type spin default 90 min 10 max 300
option name NullMoveBase type spin default 3 min 1 max 6
option name NullMoveDivisor type spin default 6 min 2 max 12
option name LMRMinDepth type spin default 3 min 1 max 8
option name LMRMinMoveNumber type spin default 3 min 0 max 20
option name LMRBigMoveNumber type spin default 12 min 4 max 40
option name AspirationDelta type spin default 25 min 5 max 200
option name AspirationMinDepth type spin default 4 min 1 max 12
option name ProbCutMargin type spin default 200 min 50 max 400
option name ProbCutReduction type spin default 4 min 2 max 6
option name ProbCutMinDepth type spin default 5 min 3 max 10
option name FutilityMargin type spin default 150 min 30 max 400
option name FutilityMaxDepth type spin default 8 min 1 max 12
option name ProbcutSeeFilter type check default false
option name PassedPawnMgPct type spin default 100 min 0 max 200
option name PassedPawnEgPct type spin default 100 min 0 max 200
option name MobilityPct type spin default 100 min 0 max 200
option name RookPct type spin default 100 min 0 max 200
option name KnightOutpostPct type spin default 100 min 0 max 200
uciok
info string [Unchessed] eval: hand-crafted (no NNUE file found)
info string [Unchessed] no policy net found — using heuristic move priors
readyok
bestmove a2a3
```

## occupied_en_passant

### Exact command stream
```text
uci
setoption name OwnBook value false
setoption name Adaptive value false
setoption name Troll value Off
isready
position startpos
go depth 1
position fen 4k3/8/8/4p3/4P3/8/8/4K3 w - e5 0 1
go depth 1
```

### Exact engine output
```text
id name Unchessed Game Adapter 0.2.3
id author Unchessed AI project
option name Hash type spin default 128 min 1 max 2048
option name Threads type spin default 6 min 1 max 64
option name Clear Hash type button
option name MultiPV type spin default 1 min 1 max 8
option name EvalFile type string default 
option name UnarchitecturedHint type check default false
option name UnarchitecturedHintExit type string default 2/128
option name UnarchitecturedFile type string default 
option name UnarchitecturedMinTime type spin default 30000 min 1000 max 600000
option name Adaptive type check default true
option name UCI_LimitStrength type check default false
option name UCI_Elo type spin default 2400 min 500 max 3200
option name Contempt type spin default 25 min 0 max 100
option name Troll type combo default Auto var Off var Auto var On
option name OwnBook type check default true
option name BookFile type string default 
option name BookDepth type spin default 16 min 0 max 40
option name PolicyFile type string default 
option name UCI_Opponent type string default 
option name PersonaSmooth type check default false
option name EngineDetectV2 type check default false
option name AdapterTelemetry type check default false
option name RFPMargin type spin default 90 min 10 max 300
option name NullMoveBase type spin default 3 min 1 max 6
option name NullMoveDivisor type spin default 6 min 2 max 12
option name LMRMinDepth type spin default 3 min 1 max 8
option name LMRMinMoveNumber type spin default 3 min 0 max 20
option name LMRBigMoveNumber type spin default 12 min 4 max 40
option name AspirationDelta type spin default 25 min 5 max 200
option name AspirationMinDepth type spin default 4 min 1 max 12
option name ProbCutMargin type spin default 200 min 50 max 400
option name ProbCutReduction type spin default 4 min 2 max 6
option name ProbCutMinDepth type spin default 5 min 3 max 10
option name FutilityMargin type spin default 150 min 30 max 400
option name FutilityMaxDepth type spin default 8 min 1 max 12
option name ProbcutSeeFilter type check default false
option name PassedPawnMgPct type spin default 100 min 0 max 200
option name PassedPawnEgPct type spin default 100 min 0 max 200
option name MobilityPct type spin default 100 min 0 max 200
option name RookPct type spin default 100 min 0 max 200
option name KnightOutpostPct type spin default 100 min 0 max 200
uciok
info string [Unchessed] eval: hand-crafted (no NNUE file found)
info string [Unchessed] no policy net found — using heuristic move priors
readyok
bestmove a2a3
```

## back_rank_pawn

### Exact command stream
```text
uci
setoption name OwnBook value false
setoption name Adaptive value false
setoption name Troll value Off
isready
position startpos
go depth 1
position fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1
go depth 1
```

### Exact engine output
```text
id name Unchessed Game Adapter 0.2.3
id author Unchessed AI project
option name Hash type spin default 128 min 1 max 2048
option name Threads type spin default 6 min 1 max 64
option name Clear Hash type button
option name MultiPV type spin default 1 min 1 max 8
option name EvalFile type string default 
option name UnarchitecturedHint type check default false
option name UnarchitecturedHintExit type string default 2/128
option name UnarchitecturedFile type string default 
option name UnarchitecturedMinTime type spin default 30000 min 1000 max 600000
option name Adaptive type check default true
option name UCI_LimitStrength type check default false
option name UCI_Elo type spin default 2400 min 500 max 3200
option name Contempt type spin default 25 min 0 max 100
option name Troll type combo default Auto var Off var Auto var On
option name OwnBook type check default true
option name BookFile type string default 
option name BookDepth type spin default 16 min 0 max 40
option name PolicyFile type string default 
option name UCI_Opponent type string default 
option name PersonaSmooth type check default false
option name EngineDetectV2 type check default false
option name AdapterTelemetry type check default false
option name RFPMargin type spin default 90 min 10 max 300
option name NullMoveBase type spin default 3 min 1 max 6
option name NullMoveDivisor type spin default 6 min 2 max 12
option name LMRMinDepth type spin default 3 min 1 max 8
option name LMRMinMoveNumber type spin default 3 min 0 max 20
option name LMRBigMoveNumber type spin default 12 min 4 max 40
option name AspirationDelta type spin default 25 min 5 max 200
option name AspirationMinDepth type spin default 4 min 1 max 12
option name ProbCutMargin type spin default 200 min 50 max 400
option name ProbCutReduction type spin default 4 min 2 max 6
option name ProbCutMinDepth type spin default 5 min 3 max 10
option name FutilityMargin type spin default 150 min 30 max 400
option name FutilityMaxDepth type spin default 8 min 1 max 12
option name ProbcutSeeFilter type check default false
option name PassedPawnMgPct type spin default 100 min 0 max 200
option name PassedPawnEgPct type spin default 100 min 0 max 200
option name MobilityPct type spin default 100 min 0 max 200
option name RookPct type spin default 100 min 0 max 200
option name KnightOutpostPct type spin default 100 min 0 max 200
uciok
info string [Unchessed] eval: hand-crafted (no NNUE file found)
info string [Unchessed] no policy net found — using heuristic move priors
readyok
bestmove a2a3
```

