#!/bin/bash
ENGINE=/home/amogusontheterminal/unchessed-ai/builds/unchessed-seefutility-src/target/release/unchessed-adapter

rm -f ~/uci_in ~/uci_out.log ~/uci_commands.txt
mkfifo ~/uci_in
touch ~/uci_commands.txt

nohup "$ENGINE" < ~/uci_in > ~/uci_out.log 2>&1 &
ENGINE_PID=$!
echo "engine pid: $ENGINE_PID"

nohup tail -n +1 -f ~/uci_commands.txt > ~/uci_in &
TAIL_PID=$!
echo "tail pid: $TAIL_PID"

sleep 1
echo "--- alive check ---"
if kill -0 "$ENGINE_PID" 2>/dev/null; then echo "engine alive"; else echo "engine DEAD"; fi
if kill -0 "$TAIL_PID" 2>/dev/null; then echo "tail alive"; else echo "tail DEAD"; fi

echo "$ENGINE_PID" > ~/uci_engine.pid
echo "$TAIL_PID" > ~/uci_tail.pid
disown -a
