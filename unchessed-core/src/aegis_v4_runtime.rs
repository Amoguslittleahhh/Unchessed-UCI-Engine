//! Pure-Rust inference forward pass for the AegisV4Chessformer architecture
//! exported to a UNARCHV1 package (see `unarchitectured_v1.rs` for the
//! container format). This runs the *full* exit only (8 layers, width 256) --
//! the shallow matryoshka exits exist purely to supervise distillation and
//! are not needed at inference time.
//!
//! Ported line-for-line from `tools/train_chessformer_v4_a100.py`'s
//! `AegisV4Chessformer.forward_path` / `history_context`, and cross-checked
//! against a Python reference run on the same exported weights.

use crate::board::{file_of, Move, Position, BISHOP, BQ, BK, KNIGHT, NO_EP, QUEEN, ROOK, WK, WQ};
use crate::unarchitectured_v1::TensorPackage;
use std::collections::HashMap;

pub const D_MODEL: usize = 256;
pub const HEADS: usize = 8;
pub const HEAD_DIM: usize = D_MODEL / HEADS;
pub const LAYERS: usize = 8;
pub const HISTORY_WIDTH: usize = 32;
pub const POLICY_ADAPTER_RANK: usize = 16;
pub const REGRET_WIDTH: usize = 32;
pub const GAB_TEMPLATES: usize = 32;
pub const GAB_HIDDEN: usize = 32;
pub const GAB_TOKEN_PROJECTION: usize = 8;
pub const MAX_LEGAL_ACTIONS: usize = 218;

pub const POLICY_HUMAN: usize = 0;
pub const POLICY_GUIDE: usize = 1;

/// One dequantized tensor, row-major, with its logical shape for bounds
/// documentation (not enforced beyond total length).
#[derive(Clone)]
struct Tensor {
    data: Vec<f32>,
}

impl Tensor {
    fn get(&self, i: usize) -> f32 {
        self.data[i]
    }
}

pub struct ChessformerWeights {
    tensors: HashMap<String, Tensor>,
}

fn dequantize(section: &crate::unarchitectured_v1::TensorSection) -> Vec<f32> {
    if section.is_quantized() {
        section
            .data
            .iter()
            .map(|&b| (b as i8) as f32 * section.scale)
            .collect()
    } else {
        section
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

impl ChessformerWeights {
    pub fn from_package(pkg: &TensorPackage) -> Result<Self, String> {
        let mut tensors = HashMap::new();
        for section in &pkg.sections {
            if section.name == "__metadata__" {
                continue;
            }
            tensors.insert(section.name.to_string(), Tensor { data: dequantize(section) });
        }
        let weights = ChessformerWeights { tensors };
        weights.require("piece_embedding.weight")?;
        Ok(weights)
    }

    fn require(&self, name: &str) -> Result<&Tensor, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("UNARCHV1 package missing tensor {name:?}"))
    }

    fn t(&self, name: &str) -> &Tensor {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("UNARCHV1 package missing tensor {name:?} (checked in from_package)"))
    }
}

/// Runtime input: a single position from the mover's perspective (already
/// vertically flipped to White-at-bottom when the mover is Black -- callers
/// are responsible for that transform, matching `unchessed-datagen`'s
/// `write_sample`), plus its legal move list encoded the same way as
/// `aegis_v4_data.encode_action` (source | target<<6 | promotion<<12, with
/// promotion in 0..=4 for none/N/B/R/Q).
pub struct PositionInput {
    /// Piece value per square 0..63: 0=empty, 1..6=own P/N/B/R/Q/K,
    /// 7..12=opponent P/N/B/R/Q/K.
    pub pieces: [u8; 64],
    /// bit0=mover kingside, bit1=mover queenside, bit2=opp kingside, bit3=opp queenside.
    pub castling: u8,
    /// 0..7 file, or 8 if no en-passant square.
    pub ep_file: u8,
    pub halfmove_clock: u8,
    pub rating: i64,
    pub time_class: usize,
    pub policy_kind: usize,
    /// Encoded legal actions, in the same coordinate frame as `pieces`.
    pub legal_actions: Vec<u16>,
}

pub struct ForwardOutput {
    /// One logit per entry in `PositionInput::legal_actions`.
    pub logits: Vec<f32>,
    /// One predicted regret (in `regret_target` units, i.e. centipawns/400) per legal action.
    pub regret_mean: Vec<f32>,
    pub regret_log_scale: Vec<f32>,
    /// Evidential Dirichlet WDL parameters (win, draw, loss), each >= 0; add 1 and
    /// normalize to get win/draw/loss probabilities.
    pub evidence: [f32; 3],
    pub representation: [f32; D_MODEL],
}

fn embed(table: &Tensor, index: usize, width: usize, out: &mut [f32]) {
    let base = index * width_of_table(table, width);
    for i in 0..width {
        out[i] += table.get(base + i);
    }
}

// Embedding tables are stored at their full row width (D_MODEL or
// HISTORY_WIDTH); `width` here is only ever D_MODEL for board embeddings.
fn width_of_table(_table: &Tensor, width: usize) -> usize {
    width
}

fn rmsnorm(values: &[f32], scale: &Tensor, width: usize, out: &mut [f32]) {
    let mean_sq: f32 = values.iter().map(|v| v * v).sum::<f32>() / width as f32;
    let inv = 1.0 / (mean_sq + 1e-6).sqrt();
    for i in 0..width {
        out[i] = values[i] * inv * scale.get(i);
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm_erf(x / std::f32::consts::SQRT_2))
}

fn libm_erf(x: f32) -> f32 {
    // Abramowitz-Stegun 7.1.26 approximation, accurate to ~1.5e-7.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// values: 64 x width, weight: (out_dim x in_dim) row-major over the first
/// `width` columns of a (D_MODEL x D_MODEL)-shaped tensor, taking only the
/// first `width` rows too. bias: width-length or None.
fn linear_full(
    values: &[f32],
    tokens: usize,
    in_width: usize,
    weight: &Tensor,
    weight_stride: usize,
    out_width: usize,
    bias: Option<&Tensor>,
    out: &mut [f32],
) {
    for tok in 0..tokens {
        let in_row = &values[tok * in_width..tok * in_width + in_width];
        for o in 0..out_width {
            let mut acc = bias.map(|b| b.get(o)).unwrap_or(0.0);
            let w_row = o * weight_stride;
            for i in 0..in_width {
                acc += in_row[i] * weight.get(w_row + i);
            }
            out[tok * out_width + o] = acc;
        }
    }
}

struct ElasticBlockWeights<'a> {
    norm_attention: &'a Tensor,
    qkv: &'a Tensor,
    project: &'a Tensor,
    project_bias: &'a Tensor,
    norm_ffn: &'a Tensor,
    up: &'a Tensor,
    up_bias: &'a Tensor,
    down: &'a Tensor,
    down_bias: &'a Tensor,
}

fn elastic_block(
    values: &mut [f32; 64 * D_MODEL],
    geometric_bias: &[f32; HEADS * 64 * 64],
    width: usize,
    w: &ElasticBlockWeights,
) {
    let tokens = 64;
    let mut normalized = [0f32; 64 * D_MODEL];
    for tok in 0..tokens {
        rmsnorm(
            &values[tok * D_MODEL..tok * D_MODEL + width],
            w.norm_attention,
            width,
            &mut normalized[tok * D_MODEL..tok * D_MODEL + width],
        );
    }

    // qkv: full tensor shape (3, D_MODEL, D_MODEL); slice [:, :width, :width]
    // per PyTorch reshape(3*width, width) -- row r of the reshaped matrix for
    // component c (0=q,1=k,2=v) and output index o is qkv[c, o, :width].
    let mut q = vec![0f32; tokens * width];
    let mut k = vec![0f32; tokens * width];
    let mut v = vec![0f32; tokens * width];
    for tok in 0..tokens {
        let in_row = &normalized[tok * D_MODEL..tok * D_MODEL + width];
        for o in 0..width {
            let mut acc_q = 0f32;
            let mut acc_k = 0f32;
            let mut acc_v = 0f32;
            let base_q = (0 * D_MODEL + o) * D_MODEL;
            let base_k = (1 * D_MODEL + o) * D_MODEL;
            let base_v = (2 * D_MODEL + o) * D_MODEL;
            for i in 0..width {
                let x = in_row[i];
                acc_q += x * w.qkv.get(base_q + i);
                acc_k += x * w.qkv.get(base_k + i);
                acc_v += x * w.qkv.get(base_v + i);
            }
            q[tok * width + o] = acc_q;
            k[tok * width + o] = acc_k;
            v[tok * width + o] = acc_v;
        }
    }

    // Scaled dot-product attention per head, with the geometric bias added
    // to the raw scores (matches `attn_mask=geometric_bias` as an additive bias).
    let head_dim = width / HEADS;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attended = vec![0f32; tokens * width];
    let mut scores = vec![0f32; tokens];
    for h in 0..HEADS {
        let bias_base = h * 64 * 64;
        for i in 0..tokens {
            let qi = &q[i * width + h * head_dim..i * width + h * head_dim + head_dim];
            let mut max_score = f32::NEG_INFINITY;
            for j in 0..tokens {
                let kj = &k[j * width + h * head_dim..j * width + h * head_dim + head_dim];
                let mut dot = 0f32;
                for d in 0..head_dim {
                    dot += qi[d] * kj[d];
                }
                let s = dot * scale + geometric_bias[bias_base + i * 64 + j];
                scores[j] = s;
                if s > max_score {
                    max_score = s;
                }
            }
            let mut sum = 0f32;
            for j in 0..tokens {
                let e = (scores[j] - max_score).exp();
                scores[j] = e;
                sum += e;
            }
            let out_base = i * width + h * head_dim;
            for d in 0..head_dim {
                let mut acc = 0f32;
                for j in 0..tokens {
                    acc += scores[j] * v[j * width + h * head_dim + d];
                }
                attended[out_base + d] = acc / sum;
            }
        }
    }

    // project (width x width slice of D_MODEL x D_MODEL) + residual
    let mut delta = vec![0f32; tokens * width];
    linear_full(&attended, tokens, width, w.project, D_MODEL, width, Some(w.project_bias), &mut delta);
    for tok in 0..tokens {
        for i in 0..width {
            values[tok * D_MODEL + i] += delta[tok * width + i];
        }
    }

    // FFN
    let mut normalized2 = vec![0f32; tokens * width];
    for tok in 0..tokens {
        rmsnorm(
            &values[tok * D_MODEL..tok * D_MODEL + width],
            w.norm_ffn,
            width,
            &mut normalized2[tok * width..tok * width + width],
        );
    }
    // up: shape (2, D_MODEL, D_MODEL) sliced to (2, width, width), reshaped (2*width, width)
    let mut hidden = vec![0f32; tokens * width];
    let mut gate = vec![0f32; tokens * width];
    for tok in 0..tokens {
        let in_row = &normalized2[tok * width..tok * width + width];
        for o in 0..width {
            let mut acc_h = w.up_bias.get(o);
            let mut acc_g = w.up_bias.get(width + o);
            let base_h = (0 * D_MODEL + o) * D_MODEL;
            let base_g = (1 * D_MODEL + o) * D_MODEL;
            for i in 0..width {
                acc_h += in_row[i] * w.up.get(base_h + i);
                acc_g += in_row[i] * w.up.get(base_g + i);
            }
            hidden[tok * width + o] = acc_h;
            gate[tok * width + o] = acc_g;
        }
    }
    let mut ffn_input = vec![0f32; tokens * width];
    for i in 0..tokens * width {
        ffn_input[i] = silu(gate[i]) * hidden[i];
    }
    let mut ffn_out = vec![0f32; tokens * width];
    linear_full(&ffn_input, tokens, width, w.down, D_MODEL, width, Some(w.down_bias), &mut ffn_out);
    for tok in 0..tokens {
        for i in 0..width {
            values[tok * D_MODEL + i] += ffn_out[tok * width + i];
        }
    }
}

pub fn forward(weights: &ChessformerWeights, input: &PositionInput) -> ForwardOutput {
    let width = D_MODEL;
    let tokens = 64;

    // --- Embeddings ---
    let mut values = [0f32; 64 * D_MODEL];
    let piece_table = weights.t("piece_embedding.weight");
    let square_table = weights.t("square_embedding.weight");
    for sq in 0..64 {
        let out = &mut values[sq * D_MODEL..sq * D_MODEL + D_MODEL];
        embed(piece_table, input.pieces[sq] as usize, D_MODEL, out);
        embed(square_table, sq, D_MODEL, out);
    }
    let mut global_state = [0f32; D_MODEL];
    embed(weights.t("castling_embedding.weight"), input.castling as usize, D_MODEL, &mut global_state);
    embed(weights.t("ep_embedding.weight"), input.ep_file as usize, D_MODEL, &mut global_state);
    let halfmove_bucket = ((input.halfmove_clock / 8) as usize).min(15);
    embed(weights.t("halfmove_embedding.weight"), halfmove_bucket, D_MODEL, &mut global_state);
    for sq in 0..64 {
        for i in 0..D_MODEL {
            values[sq * D_MODEL + i] += global_state[i];
        }
    }

    // --- GAB context (per-position, independent of layer) ---
    let token_projection = weights.t("gab.token_projection"); // (GAB_TOKEN_PROJECTION, D_MODEL)
    let mut projected = vec![0f32; 64 * GAB_TOKEN_PROJECTION];
    for sq in 0..64 {
        for o in 0..GAB_TOKEN_PROJECTION {
            let mut acc = 0f32;
            let row = o * D_MODEL;
            for i in 0..width {
                acc += values[sq * D_MODEL + i] * token_projection.get(row + i);
            }
            projected[sq * GAB_TOKEN_PROJECTION + o] = acc;
        }
    }
    // flatten(1): (64*GAB_TOKEN_PROJECTION) -> compress: Linear(64*d1, hidden)
    let compress_weight = weights.t("gab.compress.weight");
    let compress_bias = weights.t("gab.compress.bias");
    let flat_width = 64 * GAB_TOKEN_PROJECTION;
    let mut compressed = [0f32; GAB_HIDDEN];
    for o in 0..GAB_HIDDEN {
        let mut acc = compress_bias.get(o);
        let row = o * flat_width;
        for i in 0..flat_width {
            acc += projected[i] * compress_weight.get(row + i);
        }
        compressed[o] = gelu(acc);
    }
    let mut context = [0f32; GAB_HIDDEN];
    let gab_norm = weights.t("gab.norm");
    {
        let mean_sq: f32 = compressed.iter().map(|v| v * v).sum::<f32>() / GAB_HIDDEN as f32;
        let inv = 1.0 / (mean_sq + 1e-6).sqrt();
        for i in 0..GAB_HIDDEN {
            context[i] = compressed[i] * inv * gab_norm.get(i);
        }
    }

    let templates = weights.t("gab.templates"); // (GAB_TEMPLATES, 64, 64)

    for layer in 0..LAYERS {
        let coeff_weight = weights.t(&format!("gab.coefficients.{layer}.weight")); // (HEADS*GAB_TEMPLATES, GAB_HIDDEN)
        let mut coefficients = [0f32; HEADS * GAB_TEMPLATES];
        for o in 0..HEADS * GAB_TEMPLATES {
            let mut acc = 0f32;
            let row = o * GAB_HIDDEN;
            for i in 0..GAB_HIDDEN {
                acc += context[i] * coeff_weight.get(row + i);
            }
            coefficients[o] = acc;
        }
        let mut geometric_bias = vec![0f32; HEADS * 64 * 64].into_boxed_slice();
        for h in 0..HEADS {
            for t in 0..GAB_TEMPLATES {
                let c = coefficients[h * GAB_TEMPLATES + t];
                if c == 0.0 {
                    continue;
                }
                let tmpl_base = t * 64 * 64;
                let out_base = h * 64 * 64;
                for ij in 0..64 * 64 {
                    geometric_bias[out_base + ij] += c * templates.get(tmpl_base + ij);
                }
            }
        }
        let mut geometric_bias_fixed = [0f32; HEADS * 64 * 64];
        geometric_bias_fixed.copy_from_slice(&geometric_bias);

        let block = ElasticBlockWeights {
            norm_attention: weights.t(&format!("blocks.{layer}.norm_attention.scale")),
            qkv: weights.t(&format!("blocks.{layer}.qkv")),
            project: weights.t(&format!("blocks.{layer}.project")),
            project_bias: weights.t(&format!("blocks.{layer}.project_bias")),
            norm_ffn: weights.t(&format!("blocks.{layer}.norm_ffn.scale")),
            up: weights.t(&format!("blocks.{layer}.up")),
            up_bias: weights.t(&format!("blocks.{layer}.up_bias")),
            down: weights.t(&format!("blocks.{layer}.down")),
            down_bias: weights.t(&format!("blocks.{layer}.down_bias")),
        };
        elastic_block(&mut values, &geometric_bias_fixed, width, &block);
    }

    // --- Final norm + pooling ---
    let mut normalized = vec![0f32; tokens * width];
    for tok in 0..tokens {
        rmsnorm(
            &values[tok * D_MODEL..tok * D_MODEL + width],
            weights.t("final_norm.scale"),
            width,
            &mut normalized[tok * width..tok * width + width],
        );
    }
    let mut pooled = [0f32; D_MODEL];
    for tok in 0..tokens {
        for i in 0..width {
            pooled[i] += normalized[tok * width + i];
        }
    }
    for i in 0..width {
        pooled[i] /= tokens as f32;
    }

    // --- Value head (evidential WDL) ---
    let value_weight = weights.t("value_weight"); // (3, D_MODEL)
    let value_bias = weights.t("value_bias");
    let mut evidence = [0f32; 3];
    for o in 0..3 {
        let mut acc = value_bias.get(o);
        let row = o * D_MODEL;
        for i in 0..width {
            acc += pooled[i] * value_weight.get(row + i);
        }
        evidence[o] = softplus(acc);
    }

    // --- History context (no move history at inference: history_len=0) ---
    let normalized_rating = (((input.rating as f32) - 100.0) / 3550.0).clamp(0.0, 1.0);
    let mut history_vec = [0f32; HISTORY_WIDTH];
    embed(weights.t("time_embedding.weight"), input.time_class, HISTORY_WIDTH, &mut history_vec);
    let rating_weight = weights.t("rating_weight");
    let rating_bias = weights.t("rating_bias");
    for i in 0..HISTORY_WIDTH {
        history_vec[i] += normalized_rating * rating_weight.get(i) + rating_bias.get(i);
    }
    let history_project_w = weights.t("history_project.weight"); // (D_MODEL, HISTORY_WIDTH)
    let history_project_b = weights.t("history_project.bias");
    let mut history_full = [0f32; D_MODEL];
    for o in 0..D_MODEL {
        let mut acc = history_project_b.get(o);
        let row = o * HISTORY_WIDTH;
        for i in 0..HISTORY_WIDTH {
            acc += history_vec[i] * history_project_w.get(row + i);
        }
        history_full[o] = acc;
    }

    // --- Policy heads ---
    let mut body = vec![0f32; tokens * width];
    persona_full(
        &normalized, tokens, width, input.policy_kind, &history_full,
        weights.t("policy_body.weight"), Some(weights.t("policy_body.bias")),
        weights.t("policy_body.adapter_a"), weights.t("policy_body.adapter_b"),
        &mut body,
    );
    for v in body.iter_mut() {
        *v = gelu(*v);
    }
    let mut source_values = vec![0f32; tokens * width];
    persona_full(
        &body, tokens, width, input.policy_kind, &history_full,
        weights.t("policy_source.weight"), None,
        weights.t("policy_source.adapter_a"), weights.t("policy_source.adapter_b"),
        &mut source_values,
    );
    let mut target_values = vec![0f32; tokens * width];
    persona_full(
        &body, tokens, width, input.policy_kind, &history_full,
        weights.t("policy_target.weight"), None,
        weights.t("policy_target.adapter_a"), weights.t("policy_target.adapter_b"),
        &mut target_values,
    );

    let legal_count = input.legal_actions.len();
    let mut logits = vec![0f32; legal_count];
    let promotion_bias = weights.t("promotion_bias.weight"); // (5, 1)
    let scale = 1.0 / (width as f32).sqrt();
    for (idx, &action) in input.legal_actions.iter().enumerate() {
        let source_sq = (action & 63) as usize;
        let target_sq = ((action >> 6) & 63) as usize;
        let promotion = ((action >> 12) as usize).min(4);
        let mut dot = 0f32;
        for i in 0..width {
            dot += source_values[source_sq * width + i] * target_values[target_sq * width + i];
        }
        logits[idx] = dot * scale + promotion_bias.get(promotion);
    }

    // --- Regret head ---
    let regret_from = weights.t("regret_from"); // (REGRET_WIDTH, D_MODEL)
    let regret_to = weights.t("regret_to");
    let regret_promotion = weights.t("regret_promotion.weight"); // (5, REGRET_WIDTH)
    let regret_output_w = weights.t("regret_output.weight"); // (2, REGRET_WIDTH)
    let regret_output_b = weights.t("regret_output.bias");

    let mut regret_source_all = vec![0f32; 64 * REGRET_WIDTH];
    let mut regret_target_all = vec![0f32; 64 * REGRET_WIDTH];
    for sq in 0..64 {
        for o in 0..REGRET_WIDTH {
            let mut acc_s = 0f32;
            let mut acc_t = 0f32;
            let row = o * D_MODEL;
            for i in 0..width {
                let x = normalized[sq * width + i];
                acc_s += x * regret_from.get(row + i);
                acc_t += x * regret_to.get(row + i);
            }
            regret_source_all[sq * REGRET_WIDTH + o] = acc_s;
            regret_target_all[sq * REGRET_WIDTH + o] = acc_t;
        }
    }
    let mut regret_mean = vec![0f32; legal_count];
    let mut regret_log_scale = vec![0f32; legal_count];
    for (idx, &action) in input.legal_actions.iter().enumerate() {
        let source_sq = (action & 63) as usize;
        let target_sq = ((action >> 6) & 63) as usize;
        let promotion = ((action >> 12) as usize).min(4);
        let mut hidden = [0f32; REGRET_WIDTH];
        for i in 0..REGRET_WIDTH {
            hidden[i] = (regret_source_all[source_sq * REGRET_WIDTH + i]
                + regret_target_all[target_sq * REGRET_WIDTH + i]
                + regret_promotion.get(promotion * REGRET_WIDTH + i))
            .tanh();
        }
        let mut raw0 = regret_output_b.get(0);
        let mut raw1 = regret_output_b.get(1);
        for i in 0..REGRET_WIDTH {
            raw0 += hidden[i] * regret_output_w.get(i);
            raw1 += hidden[i] * regret_output_w.get(REGRET_WIDTH + i);
        }
        regret_mean[idx] = softplus(raw0);
        regret_log_scale[idx] = raw1.clamp(-8.0, 4.0);
    }

    ForwardOutput { logits, regret_mean, regret_log_scale, evidence, representation: pooled }
}

/// Same as `persona` but takes the full D_MODEL-wide history vector (the
/// reference model's `history[:, None, :width]` broadcasts the full-width
/// history projection, truncated to `width` -- not to HISTORY_WIDTH).
fn persona_full(
    values: &[f32],
    tokens: usize,
    width: usize,
    policy_kind: usize,
    history: &[f32; D_MODEL],
    weight: &Tensor,
    bias: Option<&Tensor>,
    adapter_a: &Tensor,
    adapter_b: &Tensor,
    out: &mut [f32],
) {
    linear_full(values, tokens, width, weight, D_MODEL, width, bias, out);
    let a_base = policy_kind * POLICY_ADAPTER_RANK * D_MODEL;
    let b_base = policy_kind * D_MODEL * POLICY_ADAPTER_RANK;
    let mut low = vec![0f32; tokens * POLICY_ADAPTER_RANK];
    for tok in 0..tokens {
        for r in 0..POLICY_ADAPTER_RANK {
            let mut acc = 0f32;
            let row = a_base + r * D_MODEL;
            for i in 0..width {
                let x = values[tok * width + i] + history[i];
                acc += x * adapter_a.get(row + i);
            }
            low[tok * POLICY_ADAPTER_RANK + r] = acc;
        }
    }
    for tok in 0..tokens {
        for o in 0..width {
            let mut acc = 0f32;
            let row = b_base + o * POLICY_ADAPTER_RANK;
            for r in 0..POLICY_ADAPTER_RANK {
                acc += low[tok * POLICY_ADAPTER_RANK + r] * adapter_b.get(row + r);
            }
            out[tok * width + o] += acc / POLICY_ADAPTER_RANK as f32;
        }
    }
}

/// Convert a live position + its legal moves into the mover-perspective
/// input this runtime expects, matching `unchessed-datagen`'s `write_sample`
/// exactly: vertical flip (rank XOR 56, square XOR 56) whenever Black is to
/// move, so the model always sees "the mover's own pieces at the bottom".
/// `legal_actions[i]` corresponds to `legal_moves[i]` -- callers use that
/// index correspondence to map the output back onto real `Move`s.
pub fn position_to_input(
    pos: &Position,
    legal_moves: &[Move],
    rating: i64,
    time_class: usize,
    policy_kind: usize,
) -> PositionInput {
    let flip = pos.side == crate::board::Color::Black;
    let us = pos.side.idx();
    let them = pos.side.flip().idx();

    let mut pieces = [0u8; 64];
    for p in 0..6usize {
        let mover_bb = if flip { pos.bb[us][p].swap_bytes() } else { pos.bb[us][p] };
        let opp_bb = if flip { pos.bb[them][p].swap_bytes() } else { pos.bb[them][p] };
        let mut bits = mover_bb;
        while bits != 0 {
            let sq = bits.trailing_zeros() as usize;
            pieces[sq] = p as u8 + 1;
            bits &= bits - 1;
        }
        let mut bits = opp_bb;
        while bits != 0 {
            let sq = bits.trailing_zeros() as usize;
            pieces[sq] = 6 + p as u8 + 1;
            bits &= bits - 1;
        }
    }

    let (mk, mq, ok, oq) = if flip { (BK, BQ, WK, WQ) } else { (WK, WQ, BK, BQ) };
    let mut castling = 0u8;
    if pos.castling & mk != 0 {
        castling |= 1;
    }
    if pos.castling & mq != 0 {
        castling |= 2;
    }
    if pos.castling & ok != 0 {
        castling |= 4;
    }
    if pos.castling & oq != 0 {
        castling |= 8;
    }

    let ep_file = if pos.ep == NO_EP { 8 } else { file_of(pos.ep) };
    let halfmove_clock = pos.halfmove.min(255) as u8;

    let legal_actions: Vec<u16> = legal_moves
        .iter()
        .map(|&m| {
            let (from, to) = if flip { (m.from() ^ 56, m.to() ^ 56) } else { (m.from(), m.to()) };
            let promotion: u16 = if m.is_promo() {
                match m.promo_piece() {
                    p if p == KNIGHT => 1,
                    p if p == BISHOP => 2,
                    p if p == ROOK => 3,
                    p if p == QUEEN => 4,
                    _ => 0,
                }
            } else {
                0
            };
            from as u16 | (to as u16) << 6 | (promotion << 12)
        })
        .collect();

    PositionInput {
        pieces,
        castling,
        ep_file,
        halfmove_clock,
        rating,
        time_class,
        policy_kind,
        legal_actions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unarchitectured_v1::TensorPackage;

    fn start_position_input() -> PositionInput {
        let mut pieces = [0u8; 64];
        const ROOK: u8 = 3;
        const KNIGHT: u8 = 1;
        const BISHOP: u8 = 2;
        const QUEEN: u8 = 4;
        const KING: u8 = 5;
        let back = [ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK];
        for f in 0..8usize {
            pieces[f] = back[f] + 1;
            pieces[8 + f] = 1; // own pawn
            pieces[48 + f] = 6 + 1; // opp pawn
            pieces[56 + f] = 6 + back[f] + 1;
        }

        let mut actions: Vec<u16> = Vec::new();
        for f in 0..8u16 {
            let src = 8 + f;
            actions.push(src | ((src + 8) << 6));
            actions.push(src | ((src + 16) << 6));
        }
        actions.push(1 | (16 << 6));
        actions.push(1 | (18 << 6));
        actions.push(6 | (21 << 6));
        actions.push(6 | (23 << 6));
        actions.sort_unstable();

        PositionInput {
            pieces,
            castling: 15,
            ep_file: 8,
            halfmove_clock: 0,
            rating: 2700,
            time_class: 2,
            policy_kind: POLICY_GUIDE,
            legal_actions: actions,
        }
    }

    fn load_reference_weights() -> ChessformerWeights {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../artifacts/unarchitectured-v1-final.unarchv1"
        );
        let bytes = TensorPackage::load(path).expect("read exported package");
        let pkg = TensorPackage::parse(&bytes).expect("parse exported package");
        ChessformerWeights::from_package(&pkg).expect("build weights")
    }

    /// Cross-checked against tools/_scratch_reference_forward.py run on the
    /// same real exported checkpoint (artifacts/unarchitectured-v1-final.unarchv1):
    ///   legal_count 20
    ///   logits[:20] = [-1.94899, -1.53363, -1.711539, -0.696044, -1.576939,
    ///     -0.456242, -0.652165, -0.287219, -1.707068, -1.615397, -1.915703,
    ///     -1.426473, -1.761397, -1.96128, -1.868046, -1.067433, -1.09751,
    ///     -2.014229, -1.843139, -1.567619]
    ///   evidence = [3.3165156841278076, 0.04533608630299568, 3.893404245376587]
    ///   representation[:8] = [0.160754, -0.092479, 0.932122, -0.786012,
    ///     0.049596, -0.239851, -0.521234, -1.072829]
    ///   best_action_index 7, action 1350
    #[test]
    fn start_position_matches_python_reference() {
        let weights = load_reference_weights();
        let input = start_position_input();
        assert_eq!(input.legal_actions.len(), 20);
        let output = forward(&weights, &input);

        let expected_logits = [
            -1.94899, -1.53363, -1.711539, -0.696044, -1.576939, -0.456242, -0.652165,
            -0.287219, -1.707068, -1.615397, -1.915703, -1.426473, -1.761397, -1.96128,
            -1.868046, -1.067433, -1.09751, -2.014229, -1.843139, -1.567619,
        ];
        for (i, (&got, &want)) in output.logits.iter().zip(expected_logits.iter()).enumerate() {
            assert!(
                (got - want).abs() < 5e-3,
                "logit[{i}]: got {got}, want {want}"
            );
        }

        let expected_evidence = [3.3165156841278076f32, 0.04533608630299568, 3.893404245376587];
        for i in 0..3 {
            assert!(
                (output.evidence[i] - expected_evidence[i]).abs() < 5e-3,
                "evidence[{i}]: got {}, want {}",
                output.evidence[i],
                expected_evidence[i]
            );
        }

        let expected_representation = [
            0.160754f32, -0.092479, 0.932122, -0.786012, 0.049596, -0.239851, -0.521234,
            -1.072829,
        ];
        for i in 0..8 {
            assert!(
                (output.representation[i] - expected_representation[i]).abs() < 5e-3,
                "representation[{i}]: got {}, want {}",
                output.representation[i],
                expected_representation[i]
            );
        }

        let best = (0..input.legal_actions.len())
            .max_by(|&a, &b| output.logits[a].partial_cmp(&output.logits[b]).unwrap())
            .unwrap();
        assert_eq!(best, 7, "best action index");
        assert_eq!(input.legal_actions[best], 1350, "best action encoding");
    }

    /// Second, independently-generated position (after 1.e4 e5, White to
    /// move) cross-checked against the same Python reference, to rule out
    /// the first test being a start-position-specific coincidence.
    #[test]
    fn midgame_position_matches_python_reference() {
        let mut pieces = [0u8; 64];
        const ROOK: u8 = 3;
        const KNIGHT: u8 = 1;
        const BISHOP: u8 = 2;
        const QUEEN: u8 = 4;
        const KING: u8 = 5;
        let back = [ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK];
        for f in 0..8usize {
            pieces[f] = back[f] + 1;
            pieces[56 + f] = 6 + back[f] + 1;
            if f != 4 {
                pieces[8 + f] = 1;
                pieces[48 + f] = 6 + 1;
            }
        }
        pieces[28] = 1; // white pawn e4
        pieces[36] = 6 + 1; // black pawn e5

        let actions: Vec<u16> = vec![
            771, 773, 1025, 1153, 1221, 1347, 1350, 1478, 1669, 1923, 2117, 2499, 2565,
        ];

        let input = PositionInput {
            pieces,
            castling: 15,
            ep_file: 8,
            halfmove_clock: 0,
            rating: 2700,
            time_class: 2,
            policy_kind: POLICY_GUIDE,
            legal_actions: actions,
        };

        let weights = load_reference_weights();
        let output = forward(&weights, &input);

        let expected_logits = [
            -1.473108, -1.365497, -1.90742, -0.724959, -1.290341, -1.222262, -0.451867,
            -1.899517, -1.401405, -1.672434, -1.534469, -1.091158, -2.957112,
        ];
        for (i, (&got, &want)) in output.logits.iter().zip(expected_logits.iter()).enumerate() {
            assert!(
                (got - want).abs() < 5e-3,
                "logit[{i}]: got {got}, want {want}"
            );
        }

        let expected_evidence = [3.3614559173583984f32, 0.04310621693730354, 3.917607545852661];
        for i in 0..3 {
            assert!(
                (output.evidence[i] - expected_evidence[i]).abs() < 5e-3,
                "evidence[{i}]: got {}, want {}",
                output.evidence[i],
                expected_evidence[i]
            );
        }

        let best = (0..input.legal_actions.len())
            .max_by(|&a, &b| output.logits[a].partial_cmp(&output.logits[b]).unwrap())
            .unwrap();
        assert_eq!(best, 6, "best action index");
        assert_eq!(input.legal_actions[best], 1350, "best action encoding");
    }

    #[test]
    #[ignore]
    fn benchmark_forward_pass() {
        let weights = load_reference_weights();
        let input = start_position_input();
        for _ in 0..5 {
            forward(&weights, &input);
        }
        let n = 200;
        let started = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(forward(&weights, &input));
        }
        let elapsed = started.elapsed();
        println!("{} calls in {:?} -> {:?}/call", n, elapsed, elapsed / n);
    }

    /// Confirms `position_to_input` (real movegen + real board state) agrees
    /// with the hand-built `start_position_input` fixture already validated
    /// against the Python reference above -- i.e. the live conversion path
    /// (not just the hand-crafted test input) produces a correct model input.
    #[test]
    fn position_to_input_matches_hand_built_start_position() {
        let pos = crate::fen::startpos();
        let moves = crate::movegen::legal(&pos);
        let live_input = position_to_input(&pos, moves.as_slice(), 2700, 2, POLICY_GUIDE);
        let fixture_input = start_position_input();

        assert_eq!(live_input.pieces, fixture_input.pieces, "pieces");
        assert_eq!(live_input.castling, fixture_input.castling, "castling");
        assert_eq!(live_input.ep_file, fixture_input.ep_file, "ep_file");
        let mut live_actions = live_input.legal_actions.clone();
        let mut fixture_actions = fixture_input.legal_actions.clone();
        live_actions.sort_unstable();
        fixture_actions.sort_unstable();
        assert_eq!(live_actions, fixture_actions, "legal_actions (as a set)");

        let weights = load_reference_weights();
        let live_output = forward(&weights, &live_input);
        let best = (0..live_input.legal_actions.len())
            .max_by(|&a, &b| live_output.logits[a].partial_cmp(&live_output.logits[b]).unwrap())
            .unwrap();
        assert_eq!(
            live_input.legal_actions[best], 1350,
            "live conversion should still pick g1-f3 as the best move"
        );
    }
}
