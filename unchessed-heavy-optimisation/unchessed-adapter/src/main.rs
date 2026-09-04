//! Unchessed Game Adapter — adaptive UCI chess engine.

use unchessed_core::uci::{run, EngineIdent};

fn main() {
    run(EngineIdent {
        name: "Unchessed Game Adapter",
        version: env!("CARGO_PKG_VERSION"),
        author: "Unchessed AI project",
        adaptive_engine: true,
    });
}
