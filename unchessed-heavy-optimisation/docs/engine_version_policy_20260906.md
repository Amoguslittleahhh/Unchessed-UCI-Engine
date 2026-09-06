# Engine-version policy and available runtimes

The official Stockfish download page currently identifies **Stockfish 19** as the current release and provides the Linux x86-64 universal binary at the official GitHub release endpoint for tag `sf_19`.

The official CSSLab Maia repository identifies **Maia-3** as the latest open-source Maia model. Maia-3 uses the Chessformer architecture and is intended to run through the `lc0` runtime with Maia weights, with the repository documenting model weights such as `maia-1100.pb.gz`, `maia-1500.pb.gz`, and `maia-1900.pb.gz`.

The sandbox initially contained an apt-installed Stockfish 16 binary and no `lc0` executable. Follow-up experiments must replace Stockfish 16 with the official Stockfish 19 universal binary. Any Maia experiment must record the exact `lc0` build and Maia-3 weight file; if a runnable current lc0/weight pair cannot be obtained, the result must be reported as unavailable rather than substituted with an older engine or simulation.

Sources:

1. Official Stockfish download page: https://stockfishchess.org/download/
2. Official CSSLab Maia repository: https://github.com/CSSLab/maia-chess
