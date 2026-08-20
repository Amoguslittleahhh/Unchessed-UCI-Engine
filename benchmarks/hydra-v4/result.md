# Unchessed Hydra Aegis v4 calculated budget

| Quantity | Value |
|---|---:|
| XT total runtime | 12.83 MiB |
| XT per-ply state | 1,280 bytes |
| Chessformer parameters | 4,222,905 |
| Chessformer int8 target | 4.03 MiB |
| Legal regret head | 16,610 parameters |
| Legal-only policy dot reduction | 18.79x |
| v4 record | 1,088 bytes + one 64-byte shard header |
| Records per GiB | 986,895 |
| Candidate-set cap | 16 moves |
| Noncandidate pruning | forbidden |
| layer_2_width_128 forward | 47.0 MFLOP |
| layer_4_width_192 forward | 171.6 MFLOP |
| layer_8_width_256 forward | 551.1 MFLOP |

These are deterministic architecture calculations, not measured latency, accuracy, NPS, Elo, or SPRT evidence.
