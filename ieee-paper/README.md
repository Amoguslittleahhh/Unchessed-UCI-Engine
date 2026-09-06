# Opponent-Detection Latency Paper

This directory contains the IEEEtran-style research paper **When a Strong Opponent Looks Erratic: Measuring Engine-Detection Latency in an Adaptive UCI Chess Persona**. The paper documents a reviewer-supplied real-game detector trace from Unchessed Game Adapter versus Dragon by Komodo and proposes a controlled, confidence-aware confirmation-path ablation.

The source is `opponent_detection_latency.tex`; the compiled deliverable is `opponent_detection_latency.pdf`. The figure is generated reproducibly by `make_latency_figure.py` from the move-level values reproduced in the script and is embedded as `detection_latency.pdf`.

To rebuild the paper from a clean checkout, run:

```sh
python3 ieee-paper/make_latency_figure.py
cd ieee-paper
pdflatex -interaction=nonstopmode opponent_detection_latency.tex
pdflatex -interaction=nonstopmode opponent_detection_latency.tex
```

The paper now also records the default-off `AcceleratedDetection` implementation and the 1,000-game real paired-game SPRT. It treats the mirror SPRT as an operational-safety and no-strength-cost result, not as a direct detection-latency measurement, and specifies the asymmetric telemetry experiment needed to measure moves-to-Full.

The supplied raw PGN and CSV for the original Dragon game were not present in this checkout. The paper therefore labels that detector trace as a reviewer-supplied private artifact and explicitly limits causal claims to the evidence available in the trace.
