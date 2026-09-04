//! Unchessed Game Reviewer — full-strength UCI analysis engine.

use unchessed_core::uci::{run, EngineIdent};

fn main() {
    run(EngineIdent {
        name: "Unchessed Game Reviewer",
        version: env!("CARGO_PKG_VERSION"),
        author: "Unchessed AI project",
        adaptive_engine: false,
    });
}
