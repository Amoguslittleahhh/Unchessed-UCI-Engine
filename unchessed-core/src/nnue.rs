//! NNUE evaluation: a (768 -> 256)x2 perspective feature transformer with
//! SCReLU activation and a single linear output layer. Inference is f32 with
//! a full accumulator recompute per eval (no incremental updates yet).
//!
//! Weights file format "UNCHNNUE" (little-endian):
//!   magic "UNCHNNUE" (8 bytes)
//!   u32 version (1), u32 ft_in (768), u32 acc (256)
//!   feature-transformer weights: [768 rows][256 f32] (one row per feature)
//!   feature-transformer bias: [256 f32]
//!   output weights: [512 f32] (first 256 = STM half, second 256 = non-STM)
//!   output bias: [1 f32]
//!
//! Feature index, from perspective P: a piece of color c, type p, on square s
//! maps to (own/opp)*384 + p*64 + sq, where own/opp is 0 if c == P else 1,
//! and sq is s for a white perspective, s^56 (vertical flip) for black.

use crate::board::*;
use crate::eval::Eval;

pub const FT_IN: usize = 768;
pub const ACC: usize = 256;

const MAGIC: &[u8; 8] = b"UNCHNNUE";
const SCALE: f32 = 400.0;
const EVAL_CLAMP: i32 = 3000;

pub struct Nnue {
    /// [FT_IN][ACC], row-major: one row of ACC weights per feature index
    ft_w: Vec<f32>,
    /// [ACC]
    ft_b: Vec<f32>,
    /// [2 * ACC]: first ACC for the STM accumulator, second for the non-STM
    out_w: Vec<f32>,
    out_b: f32,
}

fn read_u32(buf: &[u8], off: &mut usize) -> Result<u32, String> {
    if *off + 4 > buf.len() {
        return Err("truncated UNCHNNUE file".into());
    }
    let v = u32::from_le_bytes(buf[*off..*off + 4].try_into().unwrap());
    *off += 4;
    Ok(v)
}

fn read_f32s(buf: &[u8], off: &mut usize, n: usize) -> Result<Vec<f32>, String> {
    if *off + n * 4 > buf.len() {
        return Err("truncated UNCHNNUE file".into());
    }
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let s = *off + i * 4;
        v.push(f32::from_le_bytes(buf[s..s + 4].try_into().unwrap()));
    }
    *off += n * 4;
    Ok(v)
}

impl Nnue {
    pub fn load(path: &str) -> Result<Nnue, String> {
        let buf = std::fs::read(path).map_err(|e| format!("open {}: {}", path, e))?;
        Nnue::from_bytes(&buf)
    }

    fn from_bytes(buf: &[u8]) -> Result<Nnue, String> {
        if buf.len() < 20 || &buf[0..8] != MAGIC {
            return Err("not an UNCHNNUE weights file".into());
        }
        let mut off = 8usize;
        let version = read_u32(buf, &mut off)?;
        if version != 1 {
            return Err(format!("unsupported UNCHNNUE version {}", version));
        }
        let ft_in = read_u32(buf, &mut off)? as usize;
        let acc = read_u32(buf, &mut off)? as usize;
        if ft_in != FT_IN || acc != ACC {
            return Err(format!(
                "unsupported dims {}x{} (expected {}x{})",
                ft_in, acc, FT_IN, ACC
            ));
        }
        let expected = 20 + 4 * (FT_IN * ACC + ACC + 2 * ACC + 1);
        if buf.len() != expected {
            return Err(format!(
                "bad UNCHNNUE file size {} (expected {})",
                buf.len(),
                expected
            ));
        }
        let ft_w = read_f32s(buf, &mut off, FT_IN * ACC)?;
        let ft_b = read_f32s(buf, &mut off, ACC)?;
        let out_w = read_f32s(buf, &mut off, 2 * ACC)?;
        let out_b = read_f32s(buf, &mut off, 1)?[0];
        Ok(Nnue {
            ft_w,
            ft_b,
            out_w,
            out_b,
        })
    }

    /// Accumulator for one perspective: ft bias + sum of active feature rows.
    fn accumulate(&self, pos: &Position, persp: Color) -> Vec<f32> {
        let mut acc = self.ft_b.clone();
        let white_persp = matches!(persp, Color::White);
        for c in 0..2 {
            let own = if c == persp.idx() { 0 } else { 1 };
            for p in 0..6 {
                let mut bb = pos.bb[c][p];
                while bb != 0 {
                    let s = bb.trailing_zeros() as usize;
                    bb &= bb - 1;
                    let sq = if white_persp { s } else { s ^ 56 };
                    let idx = own * 384 + p * 64 + sq;
                    let row = &self.ft_w[idx * ACC..(idx + 1) * ACC];
                    for (a, w) in acc.iter_mut().zip(row) {
                        *a += *w;
                    }
                }
            }
        }
        acc
    }
}

/// SCReLU: clamp to [0, 1] then square.
#[inline]
fn screlu(x: f32) -> f32 {
    let v = x.clamp(0.0, 1.0);
    v * v
}

impl Eval for Nnue {
    /// Centipawns from the side-to-move's perspective (negamax-ready).
    fn eval(&self, pos: &Position) -> i32 {
        let acc_stm = self.accumulate(pos, pos.side);
        let acc_nstm = self.accumulate(pos, pos.side.flip());
        let mut out = self.out_b;
        for i in 0..ACC {
            out += screlu(acc_stm[i]) * self.out_w[i];
            out += screlu(acc_nstm[i]) * self.out_w[ACC + i];
        }
        ((out * SCALE) as i32).clamp(-EVAL_CLAMP, EVAL_CLAMP)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen;

    /// Build a deterministic dummy network file image (xorshift weights).
    fn dummy_net_bytes() -> Vec<u8> {
        let n_weights = FT_IN * ACC + ACC + 2 * ACC + 1;
        let mut buf = Vec::with_capacity(20 + 4 * n_weights);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(FT_IN as u32).to_le_bytes());
        buf.extend_from_slice(&(ACC as u32).to_le_bytes());
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        for _ in 0..n_weights {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            let w = (unit as f32) * 0.2 - 0.1; // uniform in [-0.1, 0.1)
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf
    }

    /// Write the dummy net to a uniquely named temp file and Nnue::load it.
    fn load_dummy(tag: &str) -> Nnue {
        let path = std::env::temp_dir().join(format!(
            "unchessed_nnue_test_{}_{}.bin",
            std::process::id(),
            tag
        ));
        std::fs::write(&path, dummy_net_bytes()).unwrap();
        let net = Nnue::load(path.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        net
    }

    /// Color-mirror a FEN: piece colors swapped, board flipped vertically
    /// (square ^56), side flipped, castling rights swapped, ep mirrored.
    fn color_mirror_fen(f: &str) -> String {
        let parts: Vec<&str> = f.split_whitespace().collect();
        let swap = |c: char| -> char {
            if c.is_ascii_uppercase() {
                c.to_ascii_lowercase()
            } else if c.is_ascii_lowercase() {
                c.to_ascii_uppercase()
            } else {
                c
            }
        };
        let placement: Vec<String> = parts[0]
            .split('/')
            .rev()
            .map(|rank| rank.chars().map(swap).collect())
            .collect();
        let side = if parts[1] == "w" { "b" } else { "w" };
        let castling = if parts[2] == "-" {
            "-".to_string()
        } else {
            let swapped: Vec<char> = parts[2].chars().map(swap).collect();
            let mut out = String::new();
            for want in ['K', 'Q', 'k', 'q'] {
                if swapped.contains(&want) {
                    out.push(want);
                }
            }
            out
        };
        let ep = if parts[3] == "-" {
            "-".to_string()
        } else {
            let b = parts[3].as_bytes();
            let rank = b[1] - b'0';
            format!("{}{}", b[0] as char, 9 - rank)
        };
        format!(
            "{} {} {} {} {} {}",
            placement.join("/"),
            side,
            castling,
            ep,
            parts.get(4).unwrap_or(&"0"),
            parts.get(5).unwrap_or(&"1"),
        )
    }

    /// CRITICAL correctness property. The eval is STM-relative, and the
    /// color-mirror swaps both the pieces AND the side to move, so the mover
    /// faces the identical relative position: the two evals must be EQUAL.
    /// (They would be negations of each other only for a white-relative
    /// eval, or for a mirror that did not flip the side to move.) This is
    /// the standard `eval(pos) == eval(colorflip(pos))` NNUE feature-index
    /// sanity check; a tolerance of 1 cp absorbs f32 summation-order noise.
    #[test]
    fn color_mirror_symmetry() {
        let net = load_dummy("mirror");
        let fens = [
            // startpos (symmetric baseline)
            fen::START_FEN,
            // asymmetric middlegame, black castled short, white king in the center
            "r2q1rk1/pp2ppbp/2np1np1/8/3NP3/2N1B3/PPPQ1PPP/R3KB1R w KQ - 5 9",
            // both castled, opposite sides (white short, black long)
            "2kr3r/ppp2ppp/2nqbn2/3pp3/8/2N1PN2/PPPPBPPP/R1BQ1RK1 w - - 6 9",
            // en-passant square set, black to move next after mirror
            "rnbqkb1r/pp1p1ppp/5n2/2pPp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 4",
            // lopsided material, black to move
            "r4rk1/1pp2ppp/p1np1n2/8/2B1P3/2N5/PPP2PPP/2KR3R b - - 0 12",
        ];
        for f in fens {
            let pos = fen::parse(f).unwrap();
            let mir = fen::parse(&color_mirror_fen(f)).unwrap();
            let e1 = net.eval(&pos);
            let e2 = net.eval(&mir);
            assert!(
                (e1 - e2).abs() <= 1,
                "mirror mismatch for {}: {} vs {}",
                f,
                e1,
                e2
            );
        }
    }

    #[test]
    fn eval_is_not_degenerate() {
        let net = Nnue::from_bytes(&dummy_net_bytes()).unwrap();
        let a = net
            .eval(&fen::parse("r2q1rk1/pp2ppbp/2np1np1/8/3NP3/2N1B3/PPPQ1PPP/R3KB1R w KQ - 5 9").unwrap());
        let b = net.eval(&fen::startpos());
        // a random net should not collapse to a constant on different positions
        assert_ne!(a, b, "dummy net evals should differ across positions");
        assert!(a.abs() <= EVAL_CLAMP && b.abs() <= EVAL_CLAMP);
    }

    #[test]
    fn load_rejects_garbage() {
        assert!(Nnue::from_bytes(b"NOTANNUE").is_err());
        assert!(Nnue::from_bytes(b"UNCHNNUE").is_err()); // truncated header
        // right magic, wrong dims
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&512u32.to_le_bytes());
        buf.extend_from_slice(&(ACC as u32).to_le_bytes());
        assert!(Nnue::from_bytes(&buf).is_err());
        // right header, wrong body length
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(FT_IN as u32).to_le_bytes());
        buf.extend_from_slice(&(ACC as u32).to_le_bytes());
        buf.extend_from_slice(&[0u8; 64]);
        assert!(Nnue::from_bytes(&buf).is_err());
    }
}
