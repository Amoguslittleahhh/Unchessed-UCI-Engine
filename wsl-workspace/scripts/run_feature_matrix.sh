#!/bin/bash
set -e
CUTECHESS=/home/amogusontheterminal/unchessed-ai/data/cutechess/build/cutechess-cli
BOOK=/home/amogusontheterminal/unchessed-ai/data/maia-data/sprt_book.pgn
ADAPTER=/home/amogusontheterminal/unchessed-ai/builds/unchessed-target-elofix3/release/unchessed-adapter
SF=/home/amogusontheterminal/unchessed-ai/data/stockfish_bin/stockfish/stockfish-ubuntu-x86-64-avx2
OUTDIR=/home/amogusontheterminal/unchessed-ai/results/feature_matrix
mkdir -p "$OUTDIR"

# Scenario 1: Adaptive=true, no UCI_Opponent hint, vs weak Stockfish (~1400).
# Should detect a weak opponent from moves alone and play MATCH mode.
"$CUTECHESS" \
  -engine cmd="$ADAPTER" name=AdaptVsWeak option.Threads=1 option.Adaptive=true option.OwnBook=true option.Hash=64 \
  -engine cmd="$SF" name=SF1400 option.Threads=1 option.Hash=64 option.UCI_LimitStrength=true option.UCI_Elo=1400 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=10 \
  -rounds 8 -games 2 -concurrency 8 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=1000 \
  -pgnout "$OUTDIR/s1_adaptive_vs_weak.pgn" \
  > "$OUTDIR/log_s1.txt" 2>&1
echo "s1 done"

# Scenario 2: Adaptive=true vs strong Stockfish (~2800). Should detect via
# ceiling-tell and switch to FULL.
"$CUTECHESS" \
  -engine cmd="$ADAPTER" name=AdaptVsStrong option.Threads=1 option.Adaptive=true option.OwnBook=true option.Hash=64 \
  -engine cmd="$SF" name=SF2800 option.Threads=1 option.Hash=64 option.UCI_LimitStrength=true option.UCI_Elo=2800 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=10 \
  -rounds 8 -games 2 -concurrency 8 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=1000 \
  -pgnout "$OUTDIR/s2_adaptive_vs_strong.pgn" \
  > "$OUTDIR/log_s2.txt" 2>&1
echo "s2 done"

# Scenario 3: Adaptive=true vs full-strength Stockfish (no limit). Sanity/
# crash check + should flag engine_suspect quickly.
"$CUTECHESS" \
  -engine cmd="$ADAPTER" name=AdaptVsFull option.Threads=1 option.Adaptive=true option.OwnBook=true option.Hash=64 \
  -engine cmd="$SF" name=SFFull option.Threads=1 option.Hash=64 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=10 \
  -rounds 6 -games 2 -concurrency 8 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=1000 \
  -pgnout "$OUTDIR/s3_adaptive_vs_full.pgn" \
  > "$OUTDIR/log_s3.txt" 2>&1
echo "s3 done"

# Scenario 4: Troll=On vs weak Stockfish -- exercise troll book lines +
# bail-out guard, watch for crashes/hangs.
"$CUTECHESS" \
  -engine cmd="$ADAPTER" name=TrollOn option.Threads=1 option.Adaptive=true option.Troll=On option.OwnBook=true option.Hash=64 \
  -engine cmd="$SF" name=SF1400b option.Threads=1 option.Hash=64 option.UCI_LimitStrength=true option.UCI_Elo=1400 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=6 \
  -rounds 8 -games 2 -concurrency 8 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=1000 \
  -pgnout "$OUTDIR/s4_troll_on.pgn" \
  > "$OUTDIR/log_s4.txt" 2>&1
echo "s4 done"

# Scenario 5: the newly-fixed Adaptive=true + UCI_LimitStrength=true +
# UCI_Elo=1200 combo (adapt live but never exceed the declared ceiling) vs
# a stronger Stockfish. Should stay capped, not fully catch up to SF2000.
"$CUTECHESS" \
  -engine cmd="$ADAPTER" name=AdaptCapped1200 option.Threads=1 option.Adaptive=true option.UCI_LimitStrength=true option.UCI_Elo=1200 option.OwnBook=true option.Hash=64 \
  -engine cmd="$SF" name=SF2000 option.Threads=1 option.Hash=64 option.UCI_LimitStrength=true option.UCI_Elo=2000 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=10 \
  -rounds 8 -games 2 -concurrency 8 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=1000 \
  -pgnout "$OUTDIR/s5_capped_adaptive.pgn" \
  > "$OUTDIR/log_s5.txt" 2>&1
echo "s5 done"

# Scenario 6: high contempt (draw-averse) vs low contempt, both vs weak SF,
# just checking behavior/no-crash and draw-rate difference.
"$CUTECHESS" \
  -engine cmd="$ADAPTER" name=HighContempt option.Threads=1 option.Adaptive=true option.Contempt=80 option.OwnBook=true option.Hash=64 \
  -engine cmd="$SF" name=SF1400c option.Threads=1 option.Hash=64 option.UCI_LimitStrength=true option.UCI_Elo=1400 \
  -each proto=uci tc=5+0.05 \
  -openings file="$BOOK" format=pgn order=random plies=10 \
  -rounds 8 -games 2 -concurrency 8 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=1000 \
  -pgnout "$OUTDIR/s6_high_contempt.pgn" \
  > "$OUTDIR/log_s6.txt" 2>&1
echo "s6 done"

echo "ALL DONE"
