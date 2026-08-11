#!/bin/bash
# Usage: uci_send.sh "<uci commands, one per line via printf %b>"
printf '%b' "$1" >> ~/uci_commands.txt
sleep "${2:-1}"
tail -n 30 ~/uci_out.log
