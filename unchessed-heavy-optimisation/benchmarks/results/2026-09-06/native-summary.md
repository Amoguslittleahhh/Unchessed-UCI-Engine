# Dispatch benchmark summary

| Hash MiB | Position | Scalar median NPS | Dispatch median NPS | Speedup |
|---:|---|---:|---:|---:|
| 4 | endgame | 900,372 | 1,588,210 | +76.39% |
| 4 | kiwipete | 686,249 | 1,243,951 | +81.27% |
| 4 | middlegame | 743,963 | 1,398,652 | +88.00% |
| 4 | startpos | 804,378 | 1,497,625 | +86.18% |
| 16 | endgame | 901,800 | 1,554,165 | +72.34% |
| 16 | kiwipete | 683,042 | 1,224,028 | +79.20% |
| 16 | middlegame | 734,217 | 1,313,279 | +78.87% |
| 16 | startpos | 801,204 | 1,467,105 | +83.11% |
| 64 | endgame | 892,960 | 1,485,827 | +66.39% |
| 64 | kiwipete | 666,220 | 1,175,468 | +76.44% |
| 64 | middlegame | 722,773 | 1,306,969 | +80.83% |
| 64 | startpos | 782,627 | 1,466,451 | +87.38% |

Aggregate median across all rows: scalar **742,908 NPS**, dispatch **1,404,292 NPS**, speedup **+89.03%**.
