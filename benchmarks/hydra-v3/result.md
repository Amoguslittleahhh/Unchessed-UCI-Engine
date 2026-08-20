# Unchessed Hydra Aegis v3 calculated budget

| Quantity | Value |
|---|---:|
| XT total runtime | 12.83 MiB |
| XT per-ply state | 1,280 bytes |
| XT conformal calibration | 512 bytes |
| Chessformer parameters | 4,668,583 |
| Chessformer int8 target | 4.45 MiB |
| Private policy adapters | 49,152 parameters |
| History adapter | 13,120 parameters |
| Legal-only policy dot reduction | 18.79x |
| v3 record | 160 bytes + one 64-byte shard header |
| Records per GiB | 6,710,886 |
| layer_2_width_128 forward | 45.9 MFLOP |
| layer_4_width_192 forward | 170.0 MFLOP |
| layer_8_width_256 forward | 549.0 MFLOP |

These are deterministic architecture calculations, not measured latency, accuracy, or Elo.
