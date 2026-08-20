# Aegis v3 feature microbenchmark

Measured on 2026-08-20 with Rust 1.97.1 release optimization on the sandbox's
Intel Xeon 2.60 GHz CPU (2 visible logical CPUs):

```text
Aegis v3 full feature refresh + synthetic accumulation: 2317.9 ns/call
mean active direct/xray/topology: 38.6 / 11.2 / 15.1
```

The frozen suite contains eight positions, both perspectives, and 10,000
iterations (160,000 calls). Each call extracts all direct relations, x-ray
hyperedges, and pawn/king topology hashes, then accumulates synthetic int16
rows of widths 32/16/16. The checksum was `-12112`.

Reproduce:

```bash
cargo test -p unchessed-core --release \
  aegis_v3_feature_microbench -- --ignored --nocapture
```

This is a **standalone full-refresh microbenchmark on a shared sandbox CPU**.
It is not dirty-update cost, integrated engine NPS, trained-model latency,
holdout accuracy, Elo, or SPRT evidence. Run-to-run and deployment-host
variance must be measured separately.
