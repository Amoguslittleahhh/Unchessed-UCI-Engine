# Direct SFNNv16 black-box comparison

The Stockfish value is obtained only through UCI from the official Stockfish 19 binary and published network; no Stockfish source or weights are imported into Unchessed. The homemade value is obtained through the selected Unchessed UCI mode.

positions=7
stockfish_depth=10
mae_cp=29544.429
median_abs_error_cp=29546.000
max_abs_error_cp=30280

| FEN | SFNNv16/Stockfish white cp | Homemade white cp | Delta cp |
|---|---:|---:|---:|
| `6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1` | 30001 | 228 | -29773 |
| `r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1` | -30001 | -179 | +29822 |
| `6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1` | 30001 | 455 | -29546 |
| `6rk/6pp/8/6N1/8/8/8/6K1 w - - 0 1` | 30001 | -279 | -30280 |
| `7k/5K2/Q7/8/8/8/8/8 w - - 0 1` | 30001 | 852 | -29149 |
| `k7/8/2K5/8/8/8/8/1Q6 w - - 0 1` | 30001 | 848 | -29153 |
| `7k/R7/1R6/8/8/8/8/6K1 w - - 0 1` | 30001 | 913 | -29088 |
