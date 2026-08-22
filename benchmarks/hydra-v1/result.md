# Unchessed Hydra v1 architecture budget

| Quantity | Value |
|---|---:|
| XT positional table | 11.00 MiB |
| XT materialized int8 threat table | 0.99 MiB |
| XT total runtime parameters | 12.13 MiB |
| XT state per search ply | 1,152 bytes |
| Chessformer parameters | 4,188,744 |
| Chessformer int8 runtime target | 3.99 MiB |
| Chessformer root forward | 547.8 MFLOP |

These are architecture calculations, not measured latency, accuracy, or Elo.
