//! Unchessed AI core engine library.
//!
//! Board representation, move generation, search, evaluation, opening book,
//! opponent modelling / adaptation, and the UCI protocol loop.

pub mod board;
pub mod movegen;
pub mod see;
pub mod fen;
pub mod perft;
pub mod san;
pub mod eval;
pub mod nnue;
pub mod tt;
pub mod search;
pub mod book;
pub mod policy;
pub mod unarchitectured_v1;
mod polyglot_keys;
pub mod adapt;
pub mod uci;
