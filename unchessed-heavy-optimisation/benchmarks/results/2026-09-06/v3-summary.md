# Dispatch benchmark summary

| Hash MiB | Position | Scalar median NPS | Dispatch median NPS | Speedup |
|---:|---|---:|---:|---:|
| 4 | endgame | 934,477 | 1,498,190 | +60.32% |
| 4 | kiwipete | 724,246 | 1,233,288 | +70.29% |
| 4 | middlegame | 765,126 | 1,360,556 | +77.82% |
| 4 | startpos | 844,671 | 1,463,916 | +73.31% |
| 16 | endgame | 931,706 | 1,509,210 | +61.98% |
| 16 | kiwipete | 720,862 | 1,196,752 | +66.02% |
| 16 | middlegame | 738,893 | 1,313,279 | +77.74% |
| 16 | startpos | 842,330 | 1,507,027 | +78.91% |
| 64 | endgame | 924,852 | 1,504,323 | +62.66% |
| 64 | kiwipete | 715,076 | 1,185,209 | +65.75% |
| 64 | middlegame | 755,769 | 1,316,871 | +74.24% |
| 64 | startpos | 798,195 | 1,442,490 | +80.72% |

Aggregate median across all rows: scalar **777,416 NPS**, dispatch **1,357,775 NPS**, speedup **+74.65%**.
