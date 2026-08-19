# IEEE-styled Unchessed research guide

Main deliverables:

- `unchessed-research-guide.tex` — complete LaTeX source;
- `unchessed-research-guide.pdf` — compiled paper;
- `IEEEtranSSP.cls` — renamed IEEEtran V1.8b derivative used for portable
  compilation with Source Sans Pro in the available minimal TeX distribution.

Build from this directory:

```bash
./build.sh
```

The class retains the original IEEEtran/LPPL notices and was renamed because
its default font families were changed. The section, title, author, abstract,
two-column, table, caption, and bibliography layout remains IEEEtran-derived.
For a formal IEEE submission, use the current unmodified class and the exact
venue template supplied by IEEE.

The paper distinguishes implemented code, local measurements, historical
repository statements that were not rerun, default-off research features, and
external requirements. Numerical tables are sourced from the committed JSON
reports and PGNs under `data/` and `benchmarks/real-engines/`.
