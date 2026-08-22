# Unchessed Hydra Apex v5 calculated budget

| Quantity | Value |
|---|---:|
| Offline oracle parameters | 58,412,431 |
| Oracle BF16 weights | 111.41 MiB |
| Oracle FP32 AdamW persistent state | 0.87 GiB |
| Oracle maximum legal-set forward | 11.76 GFLOP |
| Runtime student parameters | 4,222,905 |
| Oracle/student parameter ratio | 13.83x |
| CPU teacher workers | 176 x 1 thread on 180 vCPUs |
| Reserved CPU service vCPUs | 4 |
| Aggregate configured teacher hash | 11.00 GiB |
| Exact teacher nodes per legal action | 5,000 |
| Base-profile target VRAM occupancy | 92% |
| Base-profile reserved free VRAM | 8% |

| Resolved Verda profile | Minimum VRAM | Precision | Oracle parameters | Minimum records |
|---|---:|---:|---:|---:|
| blackwell-xxl | 170,000 MiB | bfloat16 | 878,114,575 | 16,384,000 |
| hopper-xl | 135,000 MiB | bfloat16 | 501,835,855 | 10,240,000 |
| 80gb-large | 78,000 MiB | bfloat16 | 230,537,295 | 4,096,000 |
| 40-48gb-base | 39,000 MiB | bfloat16 | 58,412,431 | 1,024,000 |
| v100-compat | 15,000 MiB | float16 | 29,144,367 | 409,600 |

The oracle is training-only. These are calculations, not measured hardware throughput, model accuracy, NPS, Elo, or SPRT evidence.
