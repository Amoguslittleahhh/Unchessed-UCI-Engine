//! Maia-style human policy net inference (pure Rust, no ML framework).
//!
//! Weights file format (little-endian), produced by tools/train_policy.py:
//!   magic "UNCHMAIA"
//!   u32 version (1), u32 n_buckets, u32 input (768), u32 hidden, u32 output (4096)
//!   per bucket:
//!     u32 elo_lo, u32 elo_hi
//!     w1: [hidden][input] f32   (transposed to [input][hidden] at load)
//!     b1: [hidden] f32
//!     w2: [output][hidden] f32
//!     b2: [output] f32
//!
//! One net per rating bucket (Maia's approach); inference blends the two
//! buckets nearest the target Elo. Inputs are side-to-move normalized:
//! planes 0-5 = mover P,N,B,R,Q,K, 6-11 = opponent, board flipped vertically
//! when black is to move. Output index = from | to<<6 (normalized squares).
//!
//! v1 nets use 768 inputs (piece planes only). v2 nets add 4 castling-rights
//! bits (mover K/Q, opponent K/Q) and 8 en-passant-file one-hots (780 total)
//! so the net can actually see the special-rule state.

use crate::board::*;

pub const INPUT_V1: usize = 768;
pub const INPUT_V2: usize = 780;
pub const OUTPUT: usize = 4096;

struct Bucket {
    elo_lo: i32,
    elo_hi: i32,
    /// transposed: [input][hidden]
    w1t: Vec<f32>,
    b1: Vec<f32>,
    /// [output][hidden]
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl Bucket {
    fn center(&self) -> i32 {
        (self.elo_lo.max(900) + self.elo_hi.min(2400)) / 2
    }
}

pub struct PolicyNet {
    input: usize,
    hidden: usize,
    buckets: Vec<Bucket>,
}

fn read_u32(buf: &[u8], off: &mut usize) -> Result<u32, String> {
    if *off + 4 > buf.len() {
        return Err("truncated weights file".into());
    }
    let v = u32::from_le_bytes(buf[*off..*off + 4].try_into().unwrap());
    *off += 4;
    Ok(v)
}

fn read_f32s(buf: &[u8], off: &mut usize, n: usize) -> Result<Vec<f32>, String> {
    if *off + n * 4 > buf.len() {
        return Err("truncated weights file".into());
    }
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let s = *off + i * 4;
        v.push(f32::from_le_bytes(buf[s..s + 4].try_into().unwrap()));
    }
    *off += n * 4;
    Ok(v)
}

impl PolicyNet {
    pub fn load(path: &str) -> Result<PolicyNet, String> {
        let buf = std::fs::read(path).map_err(|e| format!("open {}: {}", path, e))?;
        if buf.len() < 28 || &buf[0..8] != b"UNCHMAIA" {
            return Err("not an UNCHMAIA weights file".into());
        }
        let mut off = 8;
        let version = read_u32(&buf, &mut off)?;
        if version != 1 && version != 2 {
            return Err(format!("unsupported version {}", version));
        }
        let n_buckets = read_u32(&buf, &mut off)? as usize;
        let input = read_u32(&buf, &mut off)? as usize;
        let hidden = read_u32(&buf, &mut off)? as usize;
        let output = read_u32(&buf, &mut off)? as usize;
        if (input != INPUT_V1 && input != INPUT_V2)
            || output != OUTPUT
            || n_buckets == 0
            || n_buckets > 16
        {
            return Err("unexpected net dimensions".into());
        }
        let mut buckets = Vec::with_capacity(n_buckets);
        for _ in 0..n_buckets {
            let elo_lo = read_u32(&buf, &mut off)? as i32;
            let elo_hi = read_u32(&buf, &mut off)? as i32;
            let w1 = read_f32s(&buf, &mut off, hidden * input)?; // [hidden][input]
            let b1 = read_f32s(&buf, &mut off, hidden)?;
            let w2 = read_f32s(&buf, &mut off, output * hidden)?;
            let b2 = read_f32s(&buf, &mut off, output)?;
            // transpose w1 to [input][hidden] for sparse activation
            let mut w1t = vec![0f32; input * hidden];
            for h in 0..hidden {
                for i in 0..input {
                    w1t[i * hidden + h] = w1[h * input + i];
                }
            }
            buckets.push(Bucket {
                elo_lo,
                elo_hi,
                w1t,
                b1,
                w2,
                b2,
            });
        }
        Ok(PolicyNet {
            input,
            hidden,
            buckets,
        })
    }

    pub fn describe(&self) -> String {
        format!(
            "{} rating buckets ({}), hidden {}{}",
            self.buckets.len(),
            self.buckets
                .iter()
                .map(|b| format!("{}-{}", b.elo_lo, b.elo_hi))
                .collect::<Vec<_>>()
                .join(", "),
            self.hidden,
            if self.input == INPUT_V2 {
                ", v2 (castle/ep aware)"
            } else {
                ""
            }
        )
    }

    /// Normalized (mover-as-white) feature planes, extra scalar feature
    /// indices (castling rights / ep file, v2 nets only), and the flip flag.
    fn features(&self, pos: &Position) -> ([u64; 12], Vec<usize>, bool) {
        let flip = matches!(pos.side, Color::Black);
        let us = pos.side.idx();
        let them = pos.side.flip().idx();
        let mut planes = [0u64; 12];
        for p in 0..6 {
            planes[p] = if flip {
                pos.bb[us][p].swap_bytes()
            } else {
                pos.bb[us][p]
            };
            planes[6 + p] = if flip {
                pos.bb[them][p].swap_bytes()
            } else {
                pos.bb[them][p]
            };
        }
        let mut extra = Vec::new();
        if self.input == INPUT_V2 {
            let (mk, mq, ok, oq) = if flip { (BK, BQ, WK, WQ) } else { (WK, WQ, BK, BQ) };
            if pos.castling & mk != 0 {
                extra.push(768);
            }
            if pos.castling & mq != 0 {
                extra.push(769);
            }
            if pos.castling & ok != 0 {
                extra.push(770);
            }
            if pos.castling & oq != 0 {
                extra.push(771);
            }
            if pos.ep != NO_EP {
                extra.push(772 + file_of(pos.ep) as usize);
            }
        }
        (planes, extra, flip)
    }

    fn hidden_activations(&self, b: &Bucket, planes: &[u64; 12], extra: &[usize]) -> Vec<f32> {
        let mut acc = b.b1.clone();
        for (pi, &plane) in planes.iter().enumerate() {
            let mut bits = plane;
            while bits != 0 {
                let s = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let feat = pi * 64 + s;
                let row = &b.w1t[feat * self.hidden..(feat + 1) * self.hidden];
                for (a, w) in acc.iter_mut().zip(row) {
                    *a += w;
                }
            }
        }
        for &feat in extra {
            let row = &b.w1t[feat * self.hidden..(feat + 1) * self.hidden];
            for (a, w) in acc.iter_mut().zip(row) {
                *a += w;
            }
        }
        for a in acc.iter_mut() {
            if *a < 0.0 {
                *a = 0.0;
            }
        }
        acc
    }

    /// Per-bucket probabilities of the candidate moves (softmax restricted to
    /// the candidates).
    fn bucket_probs(
        &self,
        b: &Bucket,
        planes: &[u64; 12],
        extra: &[usize],
        flip: bool,
        moves: &[Move],
    ) -> Vec<f64> {
        let h = self.hidden_activations(b, planes, extra);
        let mut logits = Vec::with_capacity(moves.len());
        for m in moves {
            let (from, to) = if flip {
                ((m.from() ^ 56) as usize, (m.to() ^ 56) as usize)
            } else {
                (m.from() as usize, m.to() as usize)
            };
            let idx = from | (to << 6);
            let row = &b.w2[idx * self.hidden..(idx + 1) * self.hidden];
            let mut dot = b.b2[idx];
            for (a, w) in h.iter().zip(row) {
                dot += a * w;
            }
            let mut logit = dot as f64;
            // the net's from-to output cannot see promotion piece choice:
            // underpromotions are rare in human play
            if m.is_promo() && m.promo_piece() != QUEEN {
                logit -= 3.0;
            }
            logits.push(logit);
        }
        let max = logits.iter().cloned().fold(f64::MIN, f64::max);
        let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum.max(1e-12)).collect()
    }

    /// Human-move probabilities for the candidates at a target rating,
    /// blending the two nearest bucket nets.
    pub fn priors(&self, pos: &Position, moves: &[Move], target_elo: i32) -> Vec<f64> {
        if moves.is_empty() {
            return Vec::new();
        }
        let (planes, extra, flip) = self.features(pos);
        // pick surrounding buckets by center rating
        let mut lower: Option<usize> = None;
        let mut upper: Option<usize> = None;
        for (i, b) in self.buckets.iter().enumerate() {
            if b.center() <= target_elo {
                if lower.map(|l| self.buckets[l].center() < b.center()).unwrap_or(true) {
                    lower = Some(i);
                }
            }
            if b.center() >= target_elo {
                if upper.map(|u| self.buckets[u].center() > b.center()).unwrap_or(true) {
                    upper = Some(i);
                }
            }
        }
        match (lower, upper) {
            (Some(l), Some(u)) if l != u => {
                let cl = self.buckets[l].center() as f64;
                let cu = self.buckets[u].center() as f64;
                let t = ((target_elo as f64 - cl) / (cu - cl)).clamp(0.0, 1.0);
                let pl = self.bucket_probs(&self.buckets[l], &planes, &extra, flip, moves);
                let pu = self.bucket_probs(&self.buckets[u], &planes, &extra, flip, moves);
                pl.iter()
                    .zip(&pu)
                    .map(|(a, b)| a * (1.0 - t) + b * t)
                    .collect()
            }
            (Some(i), _) | (_, Some(i)) => {
                self.bucket_probs(&self.buckets[i], &planes, &extra, flip, moves)
            }
            _ => vec![1.0 / moves.len() as f64; moves.len()],
        }
    }
}
