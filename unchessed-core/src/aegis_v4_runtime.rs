//! Pure-Rust inference forward pass for the canonical Unarchitectured v1
//! compact student (training class `AegisV4Chessformer`), exported to a
//! UNARCHV1 package (see `unarchitectured_v1.rs` for the container format).
//! It executes all three trained Matryoshka exits (2/128, 4/192, and 8/256).
//! Dominant package-int8 matrices use dynamically quantized int16 activations
//! and i32 accumulation; non-x86 targets retain a scalar integer fallback.
//!
//! Ported line-for-line from `tools/train_chessformer_v4_a100.py`'s
//! `AegisV4Chessformer.forward_path` / `history_context`, and cross-checked
//! against a Python reference run on the same exported weights.

#![allow(
    clippy::excessive_precision,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use crate::board::{file_of, Move, Position, BISHOP, BK, BQ, KNIGHT, NO_EP, QUEEN, ROOK, WK, WQ};
use crate::unarchitectured_v1::TensorPackage;
use std::collections::HashMap;
use std::sync::OnceLock;

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

type DotKernel = fn(&[f32], &[f32]) -> f32;
type Dot4Kernel = fn(&[f32], usize, &[f32]) -> [f32; 4];
type Dot4x2I16I8Kernel = fn(&[i16], usize, &[i8], &[i8]) -> [[i32; 4]; 2];
type Dot4x3I16I8Kernel = fn(&[i16], usize, &[i8], &[i8], &[i8]) -> [[i32; 4]; 3];
type AxpyKernel = fn(f32, &[f32], &mut [f32]);
type QuantizeI16Kernel = fn(&[f32], &mut [i16]) -> f32;

static DOT_KERNEL: OnceLock<DotKernel> = OnceLock::new();
static DOT4_KERNEL: OnceLock<Dot4Kernel> = OnceLock::new();
static DOT4X2_I16_I8_KERNEL: OnceLock<Dot4x2I16I8Kernel> = OnceLock::new();
static DOT4X3_I16_I8_KERNEL: OnceLock<Dot4x3I16I8Kernel> = OnceLock::new();
static AXPY_KERNEL: OnceLock<AxpyKernel> = OnceLock::new();
static QUANTIZE_I16_KERNEL: OnceLock<QuantizeI16Kernel> = OnceLock::new();
static INFERENCE_THREADS: OnceLock<usize> = OnceLock::new();

fn inference_threads() -> usize {
    *INFERENCE_THREADS.get_or_init(|| {
        std::env::var("UNCHESSED_INFERENCE_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1)
                    .min(4)
            })
    })
}

#[inline]
fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    DOT_KERNEL.get_or_init(select_dot_kernel)(left, right)
}

#[inline]
fn dot_four(rows: &[f32], row_stride: usize, weights: &[f32]) -> [f32; 4] {
    debug_assert!(rows.len() >= 3 * row_stride + weights.len());
    DOT4_KERNEL.get_or_init(select_dot4_kernel)(rows, row_stride, weights)
}

#[cfg(test)]
#[inline]
fn dot_four_two_i16_i8(
    rows: &[i16],
    row_stride: usize,
    weights_0: &[i8],
    weights_1: &[i8],
) -> [[i32; 4]; 2] {
    debug_assert_eq!(weights_0.len(), weights_1.len());
    debug_assert!(rows.len() >= 3 * row_stride + weights_0.len());
    DOT4X2_I16_I8_KERNEL.get_or_init(select_dot4x2_i16_i8_kernel)(
        rows, row_stride, weights_0, weights_1,
    )
}

#[cfg(test)]
#[inline]
fn dot_four_three_i16_i8(
    rows: &[i16],
    row_stride: usize,
    weights_0: &[i8],
    weights_1: &[i8],
    weights_2: &[i8],
) -> [[i32; 4]; 3] {
    debug_assert_eq!(weights_0.len(), weights_1.len());
    debug_assert_eq!(weights_0.len(), weights_2.len());
    debug_assert!(rows.len() >= 3 * row_stride + weights_0.len());
    DOT4X3_I16_I8_KERNEL.get_or_init(select_dot4x3_i16_i8_kernel)(
        rows, row_stride, weights_0, weights_1, weights_2,
    )
}

fn select_dot_kernel() -> DotKernel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return |left, right| unsafe { dot_avx2_fma(left, right) };
    }
    dot_scalar
}

fn select_dot4_kernel() -> Dot4Kernel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return |rows, stride, weights| unsafe { dot4_avx2_fma(rows, stride, weights) };
    }
    dot4_scalar
}

fn select_dot4x2_i16_i8_kernel() -> Dot4x2I16I8Kernel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") {
        return |rows, stride, weights_0, weights_1| unsafe {
            dot4x2_i16_i8_avx2(rows, stride, weights_0, weights_1)
        };
    }
    dot4x2_i16_i8_scalar
}

fn select_dot4x3_i16_i8_kernel() -> Dot4x3I16I8Kernel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") {
        return |rows, stride, weights_0, weights_1, weights_2| unsafe {
            dot4x3_i16_i8_avx2(rows, stride, weights_0, weights_1, weights_2)
        };
    }
    dot4x3_i16_i8_scalar
}

fn select_axpy_kernel() -> AxpyKernel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return |scale, source, destination| unsafe { axpy_avx2_fma(scale, source, destination) };
    }
    axpy_scalar
}

fn select_quantize_i16_kernel() -> QuantizeI16Kernel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") {
        return |source, destination| unsafe { quantize_i16_avx2(source, destination) };
    }
    quantize_i16_scalar
}

#[inline]
fn quantize_i16(source: &[f32], destination: &mut [i16]) -> f32 {
    debug_assert_eq!(source.len(), destination.len());
    QUANTIZE_I16_KERNEL.get_or_init(select_quantize_i16_kernel)(source, destination)
}

#[inline]
fn quantize_i16_scalar(source: &[f32], destination: &mut [i16]) -> f32 {
    const QMAX: f32 = i16::MAX as f32;
    let max_abs = source
        .iter()
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
    if max_abs == 0.0 {
        return 0.0;
    }
    let inverse_scale = QMAX / max_abs;
    for (output, &value) in destination.iter_mut().zip(source) {
        *output = (value * inverse_scale).round().clamp(-QMAX, QMAX) as i16;
    }
    max_abs / QMAX
}

#[inline]
fn dot_scalar(left: &[f32], right: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 4];
    let mut chunks = left.chunks_exact(4);
    let mut right_chunks = right.chunks_exact(4);
    for (a, b) in chunks.by_ref().zip(right_chunks.by_ref()) {
        lanes[0] += a[0] * b[0];
        lanes[1] += a[1] * b[1];
        lanes[2] += a[2] * b[2];
        lanes[3] += a[3] * b[3];
    }
    let mut sum = lanes.into_iter().sum::<f32>();
    for (&a, &b) in chunks.remainder().iter().zip(right_chunks.remainder()) {
        sum += a * b;
    }
    sum
}

#[inline]
fn dot4_scalar(rows: &[f32], row_stride: usize, weights: &[f32]) -> [f32; 4] {
    [
        dot_scalar(&rows[..weights.len()], weights),
        dot_scalar(&rows[row_stride..row_stride + weights.len()], weights),
        dot_scalar(
            &rows[2 * row_stride..2 * row_stride + weights.len()],
            weights,
        ),
        dot_scalar(
            &rows[3 * row_stride..3 * row_stride + weights.len()],
            weights,
        ),
    ]
}

#[inline]
fn dot4_i16_i8_scalar(rows: &[i16], row_stride: usize, weights: &[i8]) -> [i32; 4] {
    let mut output = [0i32; 4];
    for (index, &weight) in weights.iter().enumerate() {
        let weight = i32::from(weight);
        for row in 0..4 {
            output[row] += i32::from(rows[row * row_stride + index]) * weight;
        }
    }
    output
}

#[inline]
fn dot4x2_i16_i8_scalar(
    rows: &[i16],
    row_stride: usize,
    weights_0: &[i8],
    weights_1: &[i8],
) -> [[i32; 4]; 2] {
    [
        dot4_i16_i8_scalar(rows, row_stride, weights_0),
        dot4_i16_i8_scalar(rows, row_stride, weights_1),
    ]
}

#[inline]
fn dot4x3_i16_i8_scalar(
    rows: &[i16],
    row_stride: usize,
    weights_0: &[i8],
    weights_1: &[i8],
    weights_2: &[i8],
) -> [[i32; 4]; 3] {
    [
        dot4_i16_i8_scalar(rows, row_stride, weights_0),
        dot4_i16_i8_scalar(rows, row_stride, weights_1),
        dot4_i16_i8_scalar(rows, row_stride, weights_2),
    ]
}

#[inline]
fn axpy_scalar(scale: f32, source: &[f32], destination: &mut [f32]) {
    for (value, &input) in destination.iter_mut().zip(source) {
        *value += scale * input;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_i32x8(value: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let low = _mm256_castsi256_si128(value);
    let high = _mm256_extracti128_si256(value, 1);
    let sum = _mm_add_epi32(low, high);
    let sum = _mm_hadd_epi32(sum, sum);
    let sum = _mm_hadd_epi32(sum, sum);
    _mm_cvtsi128_si32(sum)
}

#[cfg(target_arch = "x86")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_i32x8(value: std::arch::x86::__m256i) -> i32 {
    use std::arch::x86::*;
    let low = _mm256_castsi256_si128(value);
    let high = _mm256_extracti128_si256(value, 1);
    let sum = _mm_add_epi32(low, high);
    let sum = _mm_hadd_epi32(sum, sum);
    let sum = _mm_hadd_epi32(sum, sum);
    _mm_cvtsi128_si32(sum)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot4x3_i16_i8_avx2(
    rows: &[i16],
    row_stride: usize,
    weights_0: &[i8],
    weights_1: &[i8],
    weights_2: &[i8],
) -> [[i32; 4]; 3] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut accumulators = [[_mm256_setzero_si256(); 4]; 3];
    let vector_end = weights_0.len() / 16 * 16;
    let mut index = 0usize;
    while index < vector_end {
        let packed_0 = _mm_loadu_si128(weights_0.as_ptr().add(index).cast());
        let packed_1 = _mm_loadu_si128(weights_1.as_ptr().add(index).cast());
        let packed_2 = _mm_loadu_si128(weights_2.as_ptr().add(index).cast());
        let widened_0 = _mm256_cvtepi8_epi16(packed_0);
        let widened_1 = _mm256_cvtepi8_epi16(packed_1);
        let widened_2 = _mm256_cvtepi8_epi16(packed_2);
        for row in 0..4 {
            let input = _mm256_loadu_si256(rows.as_ptr().add(row * row_stride + index).cast());
            accumulators[0][row] =
                _mm256_add_epi32(accumulators[0][row], _mm256_madd_epi16(input, widened_0));
            accumulators[1][row] =
                _mm256_add_epi32(accumulators[1][row], _mm256_madd_epi16(input, widened_1));
            accumulators[2][row] =
                _mm256_add_epi32(accumulators[2][row], _mm256_madd_epi16(input, widened_2));
        }
        index += 16;
    }

    let mut output = [[0i32; 4]; 3];
    for output_index in 0..3 {
        let weights = match output_index {
            0 => weights_0,
            1 => weights_1,
            _ => weights_2,
        };
        for row in 0..4 {
            output[output_index][row] = horizontal_sum_i32x8(accumulators[output_index][row]);
            for tail in index..weights.len() {
                output[output_index][row] +=
                    i32::from(rows[row * row_stride + tail]) * i32::from(weights[tail]);
            }
        }
    }
    output
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot4x2_i16_i8_avx2(
    rows: &[i16],
    row_stride: usize,
    weights_0: &[i8],
    weights_1: &[i8],
) -> [[i32; 4]; 2] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut accumulators = [[_mm256_setzero_si256(); 4]; 2];
    let vector_end = weights_0.len() / 16 * 16;
    let mut index = 0usize;
    while index < vector_end {
        let packed_0 = _mm_loadu_si128(weights_0.as_ptr().add(index).cast());
        let packed_1 = _mm_loadu_si128(weights_1.as_ptr().add(index).cast());
        let widened_0 = _mm256_cvtepi8_epi16(packed_0);
        let widened_1 = _mm256_cvtepi8_epi16(packed_1);
        for row in 0..4 {
            let input = _mm256_loadu_si256(rows.as_ptr().add(row * row_stride + index).cast());
            accumulators[0][row] =
                _mm256_add_epi32(accumulators[0][row], _mm256_madd_epi16(input, widened_0));
            accumulators[1][row] =
                _mm256_add_epi32(accumulators[1][row], _mm256_madd_epi16(input, widened_1));
        }
        index += 16;
    }

    let mut output = [[0i32; 4]; 2];
    for output_index in 0..2 {
        let weights = if output_index == 0 {
            weights_0
        } else {
            weights_1
        };
        for row in 0..4 {
            output[output_index][row] = horizontal_sum_i32x8(accumulators[output_index][row]);
            for tail in index..weights.len() {
                output[output_index][row] +=
                    i32::from(rows[row * row_stride + tail]) * i32::from(weights[tail]);
            }
        }
    }
    output
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn quantize_i16_avx2(source: &[f32], destination: &mut [i16]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const QMAX: f32 = i16::MAX as f32;
    let sign_mask = _mm256_set1_ps(-0.0);
    let mut maximum = _mm256_setzero_ps();
    let vector_end = source.len() / 8 * 8;
    let mut index = 0usize;
    while index < vector_end {
        let values = _mm256_loadu_ps(source.as_ptr().add(index));
        maximum = _mm256_max_ps(maximum, _mm256_andnot_ps(sign_mask, values));
        index += 8;
    }
    let mut maximum_lanes = [0.0f32; 8];
    _mm256_storeu_ps(maximum_lanes.as_mut_ptr(), maximum);
    let mut max_abs = maximum_lanes.into_iter().fold(0.0f32, f32::max);
    for &value in &source[vector_end..] {
        max_abs = max_abs.max(value.abs());
    }
    if max_abs == 0.0 {
        destination.fill(0);
        return 0.0;
    }

    let inverse_scale = _mm256_set1_ps(QMAX / max_abs);
    index = 0;
    while index < vector_end {
        let values = _mm256_loadu_ps(source.as_ptr().add(index));
        let quantized = _mm256_cvtps_epi32(_mm256_mul_ps(values, inverse_scale));
        let low = _mm256_castsi256_si128(quantized);
        let high = _mm256_extracti128_si256(quantized, 1);
        let packed = _mm_packs_epi32(low, high);
        _mm_storeu_si128(destination.as_mut_ptr().add(index).cast(), packed);
        index += 8;
    }
    let inverse_scale = QMAX / max_abs;
    while index < source.len() {
        destination[index] = (source[index] * inverse_scale).round().clamp(-QMAX, QMAX) as i16;
        index += 1;
    }
    max_abs / QMAX
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_f32x8(value: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let low = _mm256_castps256_ps128(value);
    let high = _mm256_extractf128_ps(value, 1);
    let sum = _mm_add_ps(low, high);
    let sum = _mm_hadd_ps(sum, sum);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}

#[cfg(target_arch = "x86")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_f32x8(value: std::arch::x86::__m256) -> f32 {
    use std::arch::x86::*;
    let low = _mm256_castps256_ps128(value);
    let high = _mm256_extractf128_ps(value, 1);
    let sum = _mm_add_ps(low, high);
    let sum = _mm_hadd_ps(sum, sum);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot4_avx2_fma(rows: &[f32], row_stride: usize, weights: &[f32]) -> [f32; 4] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let mut accumulators = [_mm256_setzero_ps(); 4];
    let vector_end = weights.len() / 8 * 8;
    let mut index = 0usize;
    while index < vector_end {
        let weight = _mm256_loadu_ps(weights.as_ptr().add(index));
        for row in 0..4 {
            let input = _mm256_loadu_ps(rows.as_ptr().add(row * row_stride + index));
            accumulators[row] = _mm256_fmadd_ps(input, weight, accumulators[row]);
        }
        index += 8;
    }
    let mut output = [0.0f32; 4];
    for row in 0..4 {
        output[row] = horizontal_sum_f32x8(accumulators[row]);
        let mut tail = index;
        while tail < weights.len() {
            output[row] += rows[row * row_stride + tail] * weights[tail];
            tail += 1;
        }
    }
    output
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(left: &[f32], right: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut accumulator = _mm256_setzero_ps();
    let vector_end = left.len() / 8 * 8;
    let mut index = 0usize;
    while index < vector_end {
        let a = _mm256_loadu_ps(left.as_ptr().add(index));
        let b = _mm256_loadu_ps(right.as_ptr().add(index));
        accumulator = _mm256_fmadd_ps(a, b, accumulator);
        index += 8;
    }
    let mut sum = horizontal_sum_f32x8(accumulator);
    while index < left.len() {
        sum += left[index] * right[index];
        index += 1;
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2_fma(scale: f32, source: &[f32], destination: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let factor = _mm256_set1_ps(scale);
    let vector_end = source.len() / 8 * 8;
    let mut index = 0usize;
    while index < vector_end {
        let input = _mm256_loadu_ps(source.as_ptr().add(index));
        let output = _mm256_loadu_ps(destination.as_ptr().add(index));
        _mm256_storeu_ps(
            destination.as_mut_ptr().add(index),
            _mm256_fmadd_ps(factor, input, output),
        );
        index += 8;
    }
    while index < source.len() {
        destination[index] += scale * source[index];
        index += 1;
    }
}

/// Retained symmetric int8 data from the deployment package. Matrix-heavy
/// paths dynamically quantize each activation row to int16 and accumulate
/// exact i16 x i8 products into i32, then apply both scales once per output.
#[derive(Clone)]
struct QuantizedTensor {
    data: Vec<i8>,
    scale: f32,
}

/// A tensor's dequantized values remain available for embeddings, norms,
/// small heads and the independently validated f32 fallback. Quantized package
/// bytes are retained as well so dominant 64-token matrix projections do not
/// expand weights to f32 in their hot loops.
#[derive(Clone)]
struct Tensor {
    data: Vec<f32>,
    quantized: Option<QuantizedTensor>,
}

impl Tensor {
    fn get(&self, i: usize) -> f32 {
        self.data[i]
    }
}

struct QuantizedActivations {
    data: Vec<i16>,
    scales: Vec<f32>,
}

fn quantize_activation_rows(
    values: &[f32],
    tokens: usize,
    row_stride: usize,
    width: usize,
) -> QuantizedActivations {
    let mut data = vec![0i16; tokens * width];
    let mut scales = vec![0.0f32; tokens];
    for token in 0..tokens {
        let source = &values[token * row_stride..token * row_stride + width];
        scales[token] = quantize_i16(source, &mut data[token * width..(token + 1) * width]);
    }
    QuantizedActivations { data, scales }
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
            if section.is_quantized()
                && (section.dtype != crate::unarchitectured_v1::DTYPE_I8 || section.zero_point != 0)
            {
                return Err(format!(
                    "UNARCHV1 tensor {:?} uses unsupported non-symmetric int8 quantization",
                    section.name
                ));
            }
            if !section.is_quantized() && section.dtype != crate::unarchitectured_v1::DTYPE_F32 {
                return Err(format!(
                    "UNARCHV1 tensor {:?} uses an unsupported runtime dtype",
                    section.name
                ));
            }
            let quantized = section.is_quantized().then(|| QuantizedTensor {
                data: section.data.iter().map(|&value| value as i8).collect(),
                scale: section.scale,
            });
            tensors.insert(
                section.name.to_string(),
                Tensor {
                    data: dequantize(section),
                    quantized,
                },
            );
        }
        let weights = ChessformerWeights { tensors };
        weights.validate_schema()?;
        Ok(weights)
    }

    fn require_len(&self, name: &str, expected: usize) -> Result<(), String> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("UNARCHV1 package missing tensor {name:?}"))?;
        if tensor.data.len() != expected {
            return Err(format!(
                "UNARCHV1 tensor {name:?} has {} values, expected {expected}",
                tensor.data.len()
            ));
        }
        Ok(())
    }

    fn validate_schema(&self) -> Result<(), String> {
        for (name, length) in [
            ("piece_embedding.weight", 13 * D_MODEL),
            ("square_embedding.weight", 64 * D_MODEL),
            ("castling_embedding.weight", 16 * D_MODEL),
            ("ep_embedding.weight", 9 * D_MODEL),
            ("halfmove_embedding.weight", 16 * D_MODEL),
            ("gab.token_projection", GAB_TOKEN_PROJECTION * D_MODEL),
            (
                "gab.compress.weight",
                GAB_HIDDEN * 64 * GAB_TOKEN_PROJECTION,
            ),
            ("gab.compress.bias", GAB_HIDDEN),
            ("gab.norm", GAB_HIDDEN),
            ("gab.templates", GAB_TEMPLATES * 64 * 64),
            ("final_norm.scale", D_MODEL),
            ("value_weight", 3 * D_MODEL),
            ("value_bias", 3),
            ("time_embedding.weight", 5 * HISTORY_WIDTH),
            ("rating_weight", HISTORY_WIDTH),
            ("rating_bias", HISTORY_WIDTH),
            ("history_project.weight", D_MODEL * HISTORY_WIDTH),
            ("history_project.bias", D_MODEL),
            ("promotion_bias.weight", 5),
            ("regret_from", REGRET_WIDTH * D_MODEL),
            ("regret_to", REGRET_WIDTH * D_MODEL),
            ("regret_promotion.weight", 5 * REGRET_WIDTH),
            ("regret_output.weight", 2 * REGRET_WIDTH),
            ("regret_output.bias", 2),
        ] {
            self.require_len(name, length)?;
        }
        for prefix in ["policy_body", "policy_source", "policy_target"] {
            self.require_len(&format!("{prefix}.weight"), D_MODEL * D_MODEL)?;
            self.require_len(
                &format!("{prefix}.adapter_a"),
                2 * POLICY_ADAPTER_RANK * D_MODEL,
            )?;
            self.require_len(
                &format!("{prefix}.adapter_b"),
                2 * D_MODEL * POLICY_ADAPTER_RANK,
            )?;
        }
        self.require_len("policy_body.bias", D_MODEL)?;
        for layer in 0..LAYERS {
            self.require_len(
                &format!("gab.coefficients.{layer}.weight"),
                HEADS * GAB_TEMPLATES * GAB_HIDDEN,
            )?;
            let prefix = format!("blocks.{layer}");
            self.require_len(&format!("{prefix}.norm_attention.scale"), D_MODEL)?;
            self.require_len(&format!("{prefix}.qkv"), 3 * D_MODEL * D_MODEL)?;
            self.require_len(&format!("{prefix}.project"), D_MODEL * D_MODEL)?;
            self.require_len(&format!("{prefix}.project_bias"), D_MODEL)?;
            self.require_len(&format!("{prefix}.norm_ffn.scale"), D_MODEL)?;
            self.require_len(&format!("{prefix}.up"), 2 * D_MODEL * D_MODEL)?;
            self.require_len(&format!("{prefix}.up_bias"), 2 * D_MODEL)?;
            self.require_len(&format!("{prefix}.down"), D_MODEL * D_MODEL)?;
            self.require_len(&format!("{prefix}.down_bias"), D_MODEL)?;
        }
        Ok(())
    }

    fn t(&self, name: &str) -> &Tensor {
        self.tensors.get(name).unwrap_or_else(|| {
            panic!("UNARCHV1 package missing tensor {name:?} (checked in from_package)")
        })
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceExit {
    Layer2Width128,
    Layer4Width192,
    Layer8Width256,
}

impl InferenceExit {
    pub const fn layers(self) -> usize {
        match self {
            Self::Layer2Width128 => 2,
            Self::Layer4Width192 => 4,
            Self::Layer8Width256 => 8,
        }
    }

    pub const fn width(self) -> usize {
        match self {
            Self::Layer2Width128 => 128,
            Self::Layer4Width192 => 192,
            Self::Layer8Width256 => 256,
        }
    }
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

fn rmsnorm(values: &[f32], scale: &Tensor, width: usize, out: &mut [f32], dot_kernel: DotKernel) {
    let mean_sq = dot_kernel(&values[..width], &values[..width]) / width as f32;
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
    let threads = inference_threads().min(tokens);
    let work = tokens.saturating_mul(in_width).saturating_mul(out_width);
    if let Some(quantized_weight) = &weight.quantized {
        let activations = quantize_activation_rows(values, tokens, in_width, in_width);
        if threads <= 1 || work < 1_000_000 {
            linear_quantized_sequential(
                &activations.data,
                &activations.scales,
                tokens,
                in_width,
                quantized_weight,
                weight_stride,
                out_width,
                bias,
                out,
            );
            return;
        }
        let tokens_per_thread = tokens.div_ceil(threads);
        std::thread::scope(|scope| {
            for (chunk_index, out_chunk) in
                out.chunks_mut(tokens_per_thread * out_width).enumerate()
            {
                let token_start = chunk_index * tokens_per_thread;
                let token_count = out_chunk.len() / out_width;
                let input = &activations.data
                    [token_start * in_width..(token_start + token_count) * in_width];
                let scales = &activations.scales[token_start..token_start + token_count];
                scope.spawn(move || {
                    linear_quantized_sequential(
                        input,
                        scales,
                        token_count,
                        in_width,
                        quantized_weight,
                        weight_stride,
                        out_width,
                        bias,
                        out_chunk,
                    );
                });
            }
        });
        return;
    }
    if threads <= 1 || work < 1_000_000 {
        linear_full_sequential(
            values,
            tokens,
            in_width,
            weight,
            weight_stride,
            out_width,
            bias,
            out,
        );
        return;
    }
    let tokens_per_thread = tokens.div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk_index, out_chunk) in out.chunks_mut(tokens_per_thread * out_width).enumerate() {
            let token_start = chunk_index * tokens_per_thread;
            let token_count = out_chunk.len() / out_width;
            let input = &values[token_start * in_width..(token_start + token_count) * in_width];
            scope.spawn(move || {
                linear_full_sequential(
                    input,
                    token_count,
                    in_width,
                    weight,
                    weight_stride,
                    out_width,
                    bias,
                    out_chunk,
                );
            });
        }
    });
}

fn linear_quantized_sequential(
    values: &[i16],
    activation_scales: &[f32],
    tokens: usize,
    in_width: usize,
    weight: &QuantizedTensor,
    weight_stride: usize,
    out_width: usize,
    bias: Option<&Tensor>,
    out: &mut [f32],
) {
    let dot_four_two = *DOT4X2_I16_I8_KERNEL.get_or_init(select_dot4x2_i16_i8_kernel);

    const TOKEN_BLOCK: usize = 4;
    for output_start in (0..out_width).step_by(2) {
        let output_end = (output_start + 2).min(out_width);
        let first_weight_start = output_start * weight_stride;
        let first_weight = &weight.data[first_weight_start..first_weight_start + in_width];
        if output_end - output_start == 2 {
            let second_weight_start = (output_start + 1) * weight_stride;
            let second_weight = &weight.data[second_weight_start..second_weight_start + in_width];
            let initial_0 = bias.map(|values| values.get(output_start)).unwrap_or(0.0);
            let initial_1 = bias
                .map(|values| values.get(output_start + 1))
                .unwrap_or(0.0);
            for token_start in (0..tokens).step_by(TOKEN_BLOCK) {
                let token_end = (token_start + TOKEN_BLOCK).min(tokens);
                if token_end - token_start == TOKEN_BLOCK {
                    let sums = dot_four_two(
                        &values[token_start * in_width..],
                        in_width,
                        first_weight,
                        second_weight,
                    );
                    for token_lane in 0..TOKEN_BLOCK {
                        let scale = activation_scales[token_start + token_lane] * weight.scale;
                        out[(token_start + token_lane) * out_width + output_start] =
                            initial_0 + sums[0][token_lane] as f32 * scale;
                        out[(token_start + token_lane) * out_width + output_start + 1] =
                            initial_1 + sums[1][token_lane] as f32 * scale;
                    }
                } else {
                    for token in token_start..token_end {
                        let input = &values[token * in_width..(token + 1) * in_width];
                        for (output_lane, weight_row) in
                            [first_weight, second_weight].into_iter().enumerate()
                        {
                            let sum = input
                                .iter()
                                .zip(weight_row)
                                .map(|(&activation, &weight)| {
                                    i32::from(activation) * i32::from(weight)
                                })
                                .sum::<i32>();
                            let initial = if output_lane == 0 {
                                initial_0
                            } else {
                                initial_1
                            };
                            out[token * out_width + output_start + output_lane] =
                                initial + sum as f32 * activation_scales[token] * weight.scale;
                        }
                    }
                }
            }
        } else {
            let initial = bias.map(|values| values.get(output_start)).unwrap_or(0.0);
            for token in 0..tokens {
                let input = &values[token * in_width..(token + 1) * in_width];
                let sum = input
                    .iter()
                    .zip(first_weight)
                    .map(|(&activation, &weight)| i32::from(activation) * i32::from(weight))
                    .sum::<i32>();
                out[token * out_width + output_start] =
                    initial + sum as f32 * activation_scales[token] * weight.scale;
            }
        }
    }
}

fn linear_full_sequential(
    values: &[f32],
    tokens: usize,
    in_width: usize,
    weight: &Tensor,
    weight_stride: usize,
    out_width: usize,
    bias: Option<&Tensor>,
    out: &mut [f32],
) {
    const TOKEN_BLOCK: usize = 4;
    for token_start in (0..tokens).step_by(TOKEN_BLOCK) {
        let token_end = (token_start + TOKEN_BLOCK).min(tokens);
        for o in 0..out_width {
            let w_row = o * weight_stride;
            let weights = &weight.data[w_row..w_row + in_width];
            let initial = bias.map(|b| b.get(o)).unwrap_or(0.0);
            if token_end - token_start == 4 {
                let results = dot_four(&values[token_start * in_width..], in_width, weights);
                for lane in 0..4 {
                    out[(token_start + lane) * out_width + o] = initial + results[lane];
                }
            } else {
                for tok in token_start..token_end {
                    let in_row = &values[tok * in_width..tok * in_width + in_width];
                    out[tok * out_width + o] = initial + dot_product(in_row, weights);
                }
            }
        }
    }
}

fn project_qkv(
    normalized: &[f32],
    tokens: usize,
    width: usize,
    weights: &Tensor,
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) {
    if let Some(quantized_weights) = &weights.quantized {
        let activations = quantize_activation_rows(normalized, tokens, D_MODEL, width);
        project_qkv_quantized(&activations, tokens, width, quantized_weights, q, k, v);
        return;
    }
    const TOKEN_BLOCK: usize = 4;
    for token_start in (0..tokens).step_by(TOKEN_BLOCK) {
        let token_end = (token_start + TOKEN_BLOCK).min(tokens);
        for o in 0..width {
            let base_q = o * D_MODEL;
            let base_k = (D_MODEL + o) * D_MODEL;
            let base_v = (2 * D_MODEL + o) * D_MODEL;
            let weight_q = &weights.data[base_q..base_q + width];
            let weight_k = &weights.data[base_k..base_k + width];
            let weight_v = &weights.data[base_v..base_v + width];
            if token_end - token_start == 4 {
                let rows = &normalized[token_start * D_MODEL..];
                let result_q = dot_four(rows, D_MODEL, weight_q);
                let result_k = dot_four(rows, D_MODEL, weight_k);
                let result_v = dot_four(rows, D_MODEL, weight_v);
                for lane in 0..4 {
                    q[(token_start + lane) * width + o] = result_q[lane];
                    k[(token_start + lane) * width + o] = result_k[lane];
                    v[(token_start + lane) * width + o] = result_v[lane];
                }
            } else {
                for tok in token_start..token_end {
                    let input = &normalized[tok * D_MODEL..tok * D_MODEL + width];
                    q[tok * width + o] = dot_product(input, weight_q);
                    k[tok * width + o] = dot_product(input, weight_k);
                    v[tok * width + o] = dot_product(input, weight_v);
                }
            }
        }
    }
}

fn project_qkv_quantized(
    activations: &QuantizedActivations,
    tokens: usize,
    width: usize,
    weights: &QuantizedTensor,
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) {
    let dot_four_three = *DOT4X3_I16_I8_KERNEL.get_or_init(select_dot4x3_i16_i8_kernel);

    const TOKEN_BLOCK: usize = 4;
    for token_start in (0..tokens).step_by(TOKEN_BLOCK) {
        let token_end = (token_start + TOKEN_BLOCK).min(tokens);
        for output_index in 0..width {
            let q_start = output_index * D_MODEL;
            let k_start = (D_MODEL + output_index) * D_MODEL;
            let v_start = (2 * D_MODEL + output_index) * D_MODEL;
            let weight_q = &weights.data[q_start..q_start + width];
            let weight_k = &weights.data[k_start..k_start + width];
            let weight_v = &weights.data[v_start..v_start + width];
            if token_end - token_start == TOKEN_BLOCK {
                let rows = &activations.data[token_start * width..];
                let results = dot_four_three(rows, width, weight_q, weight_k, weight_v);
                for lane in 0..TOKEN_BLOCK {
                    let scale = activations.scales[token_start + lane] * weights.scale;
                    q[(token_start + lane) * width + output_index] =
                        results[0][lane] as f32 * scale;
                    k[(token_start + lane) * width + output_index] =
                        results[1][lane] as f32 * scale;
                    v[(token_start + lane) * width + output_index] =
                        results[2][lane] as f32 * scale;
                }
            } else {
                for token in token_start..token_end {
                    let input = &activations.data[token * width..(token + 1) * width];
                    let dot = |weight_row: &[i8]| {
                        input
                            .iter()
                            .zip(weight_row)
                            .map(|(&activation, &weight)| i32::from(activation) * i32::from(weight))
                            .sum::<i32>() as f32
                            * activations.scales[token]
                            * weights.scale
                    };
                    q[token * width + output_index] = dot(weight_q);
                    k[token * width + output_index] = dot(weight_k);
                    v[token * width + output_index] = dot(weight_v);
                }
            }
        }
    }
}

fn project_up(
    normalized: &[f32],
    tokens: usize,
    width: usize,
    weights: &Tensor,
    bias: &Tensor,
    hidden: &mut [f32],
    gate: &mut [f32],
) {
    if let Some(quantized_weights) = &weights.quantized {
        let activations = quantize_activation_rows(normalized, tokens, width, width);
        project_up_quantized(
            &activations,
            tokens,
            width,
            quantized_weights,
            bias,
            hidden,
            gate,
        );
        return;
    }
    const TOKEN_BLOCK: usize = 4;
    for token_start in (0..tokens).step_by(TOKEN_BLOCK) {
        let token_end = (token_start + TOKEN_BLOCK).min(tokens);
        for o in 0..width {
            let base_h = o * D_MODEL;
            let base_g = (D_MODEL + o) * D_MODEL;
            let weight_h = &weights.data[base_h..base_h + width];
            let weight_g = &weights.data[base_g..base_g + width];
            if token_end - token_start == 4 {
                let rows = &normalized[token_start * width..];
                let result_h = dot_four(rows, width, weight_h);
                let result_g = dot_four(rows, width, weight_g);
                for lane in 0..4 {
                    hidden[(token_start + lane) * width + o] = bias.get(o) + result_h[lane];
                    gate[(token_start + lane) * width + o] = bias.get(width + o) + result_g[lane];
                }
            } else {
                for tok in token_start..token_end {
                    let input = &normalized[tok * width..(tok + 1) * width];
                    hidden[tok * width + o] = bias.get(o) + dot_product(input, weight_h);
                    gate[tok * width + o] = bias.get(width + o) + dot_product(input, weight_g);
                }
            }
        }
    }
}

fn project_up_quantized(
    activations: &QuantizedActivations,
    tokens: usize,
    width: usize,
    weights: &QuantizedTensor,
    bias: &Tensor,
    hidden: &mut [f32],
    gate: &mut [f32],
) {
    let dot_four_two = *DOT4X2_I16_I8_KERNEL.get_or_init(select_dot4x2_i16_i8_kernel);

    const TOKEN_BLOCK: usize = 4;
    for token_start in (0..tokens).step_by(TOKEN_BLOCK) {
        let token_end = (token_start + TOKEN_BLOCK).min(tokens);
        for output_index in 0..width {
            let hidden_start = output_index * D_MODEL;
            let gate_start = (D_MODEL + output_index) * D_MODEL;
            let weight_hidden = &weights.data[hidden_start..hidden_start + width];
            let weight_gate = &weights.data[gate_start..gate_start + width];
            if token_end - token_start == TOKEN_BLOCK {
                let rows = &activations.data[token_start * width..];
                let results = dot_four_two(rows, width, weight_hidden, weight_gate);
                for lane in 0..TOKEN_BLOCK {
                    let scale = activations.scales[token_start + lane] * weights.scale;
                    hidden[(token_start + lane) * width + output_index] =
                        bias.get(output_index) + results[0][lane] as f32 * scale;
                    gate[(token_start + lane) * width + output_index] =
                        bias.get(width + output_index) + results[1][lane] as f32 * scale;
                }
            } else {
                for token in token_start..token_end {
                    let input = &activations.data[token * width..(token + 1) * width];
                    let dot = |weight_row: &[i8]| {
                        input
                            .iter()
                            .zip(weight_row)
                            .map(|(&activation, &weight)| i32::from(activation) * i32::from(weight))
                            .sum::<i32>() as f32
                            * activations.scales[token]
                            * weights.scale
                    };
                    hidden[token * width + output_index] =
                        bias.get(output_index) + dot(weight_hidden);
                    gate[token * width + output_index] =
                        bias.get(width + output_index) + dot(weight_gate);
                }
            }
        }
    }
}

fn project_regret_quantized(
    normalized: &[f32],
    tokens: usize,
    width: usize,
    from: &QuantizedTensor,
    to: &QuantizedTensor,
    source_out: &mut [f32],
    target_out: &mut [f32],
) {
    const TOKEN_BLOCK: usize = 4;
    let activations = quantize_activation_rows(normalized, tokens, width, width);
    let dot_four_two = *DOT4X2_I16_I8_KERNEL.get_or_init(select_dot4x2_i16_i8_kernel);
    for token_start in (0..tokens).step_by(TOKEN_BLOCK) {
        let token_end = (token_start + TOKEN_BLOCK).min(tokens);
        for output_index in 0..REGRET_WIDTH {
            let row_start = output_index * D_MODEL;
            let weight_from = &from.data[row_start..row_start + width];
            let weight_to = &to.data[row_start..row_start + width];
            if token_end - token_start == TOKEN_BLOCK {
                let sums = dot_four_two(
                    &activations.data[token_start * width..],
                    width,
                    weight_from,
                    weight_to,
                );
                for lane in 0..TOKEN_BLOCK {
                    let activation_scale = activations.scales[token_start + lane];
                    source_out[(token_start + lane) * REGRET_WIDTH + output_index] =
                        sums[0][lane] as f32 * activation_scale * from.scale;
                    target_out[(token_start + lane) * REGRET_WIDTH + output_index] =
                        sums[1][lane] as f32 * activation_scale * to.scale;
                }
            } else {
                for token in token_start..token_end {
                    let input = &activations.data[token * width..(token + 1) * width];
                    let dot = |weight_row: &[i8], weight_scale: f32| {
                        input
                            .iter()
                            .zip(weight_row)
                            .map(|(&activation, &weight)| i32::from(activation) * i32::from(weight))
                            .sum::<i32>() as f32
                            * activations.scales[token]
                            * weight_scale
                    };
                    source_out[token * REGRET_WIDTH + output_index] = dot(weight_from, from.scale);
                    target_out[token * REGRET_WIDTH + output_index] = dot(weight_to, to.scale);
                }
            }
        }
    }
}

fn attention_heads(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    geometric_bias: &[f32],
    width: usize,
    head_start: usize,
    head_end: usize,
) -> Vec<f32> {
    let tokens = 64;
    let head_dim = width / HEADS;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; (head_end - head_start) * tokens * head_dim];
    let mut scores = [0.0f32; 64];
    let dot = *DOT_KERNEL.get_or_init(select_dot_kernel);
    let axpy = *AXPY_KERNEL.get_or_init(select_axpy_kernel);
    for h in head_start..head_end {
        let bias_base = h * 64 * 64;
        let local_head = h - head_start;
        for i in 0..tokens {
            let qi = &q[i * width + h * head_dim..i * width + h * head_dim + head_dim];
            let mut max_score = f32::NEG_INFINITY;
            for j in 0..tokens {
                let kj = &k[j * width + h * head_dim..j * width + h * head_dim + head_dim];
                let score = dot(qi, kj) * scale + geometric_bias[bias_base + i * 64 + j];
                scores[j] = score;
                max_score = max_score.max(score);
            }
            let mut sum = 0.0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            let output_base = (local_head * tokens + i) * head_dim;
            let row = &mut output[output_base..output_base + head_dim];
            let inverse_sum = 1.0 / sum;
            for j in 0..tokens {
                let value_base = j * width + h * head_dim;
                axpy(
                    scores[j] * inverse_sum,
                    &v[value_base..value_base + head_dim],
                    row,
                );
            }
        }
    }
    output
}

fn copy_attention_heads(
    source: &[f32],
    destination: &mut [f32],
    width: usize,
    head_start: usize,
    head_end: usize,
) {
    let head_dim = width / HEADS;
    for h in head_start..head_end {
        let local_head = h - head_start;
        for token in 0..64 {
            let source_base = (local_head * 64 + token) * head_dim;
            let destination_base = token * width + h * head_dim;
            destination[destination_base..destination_base + head_dim]
                .copy_from_slice(&source[source_base..source_base + head_dim]);
        }
    }
}

struct LayerTensorNames {
    coefficients: &'static str,
    norm_attention: &'static str,
    qkv: &'static str,
    project: &'static str,
    project_bias: &'static str,
    norm_ffn: &'static str,
    up: &'static str,
    up_bias: &'static str,
    down: &'static str,
    down_bias: &'static str,
}

const LAYER_TENSOR_NAMES: [LayerTensorNames; LAYERS] = [
    LayerTensorNames {
        coefficients: "gab.coefficients.0.weight",
        norm_attention: "blocks.0.norm_attention.scale",
        qkv: "blocks.0.qkv",
        project: "blocks.0.project",
        project_bias: "blocks.0.project_bias",
        norm_ffn: "blocks.0.norm_ffn.scale",
        up: "blocks.0.up",
        up_bias: "blocks.0.up_bias",
        down: "blocks.0.down",
        down_bias: "blocks.0.down_bias",
    },
    LayerTensorNames {
        coefficients: "gab.coefficients.1.weight",
        norm_attention: "blocks.1.norm_attention.scale",
        qkv: "blocks.1.qkv",
        project: "blocks.1.project",
        project_bias: "blocks.1.project_bias",
        norm_ffn: "blocks.1.norm_ffn.scale",
        up: "blocks.1.up",
        up_bias: "blocks.1.up_bias",
        down: "blocks.1.down",
        down_bias: "blocks.1.down_bias",
    },
    LayerTensorNames {
        coefficients: "gab.coefficients.2.weight",
        norm_attention: "blocks.2.norm_attention.scale",
        qkv: "blocks.2.qkv",
        project: "blocks.2.project",
        project_bias: "blocks.2.project_bias",
        norm_ffn: "blocks.2.norm_ffn.scale",
        up: "blocks.2.up",
        up_bias: "blocks.2.up_bias",
        down: "blocks.2.down",
        down_bias: "blocks.2.down_bias",
    },
    LayerTensorNames {
        coefficients: "gab.coefficients.3.weight",
        norm_attention: "blocks.3.norm_attention.scale",
        qkv: "blocks.3.qkv",
        project: "blocks.3.project",
        project_bias: "blocks.3.project_bias",
        norm_ffn: "blocks.3.norm_ffn.scale",
        up: "blocks.3.up",
        up_bias: "blocks.3.up_bias",
        down: "blocks.3.down",
        down_bias: "blocks.3.down_bias",
    },
    LayerTensorNames {
        coefficients: "gab.coefficients.4.weight",
        norm_attention: "blocks.4.norm_attention.scale",
        qkv: "blocks.4.qkv",
        project: "blocks.4.project",
        project_bias: "blocks.4.project_bias",
        norm_ffn: "blocks.4.norm_ffn.scale",
        up: "blocks.4.up",
        up_bias: "blocks.4.up_bias",
        down: "blocks.4.down",
        down_bias: "blocks.4.down_bias",
    },
    LayerTensorNames {
        coefficients: "gab.coefficients.5.weight",
        norm_attention: "blocks.5.norm_attention.scale",
        qkv: "blocks.5.qkv",
        project: "blocks.5.project",
        project_bias: "blocks.5.project_bias",
        norm_ffn: "blocks.5.norm_ffn.scale",
        up: "blocks.5.up",
        up_bias: "blocks.5.up_bias",
        down: "blocks.5.down",
        down_bias: "blocks.5.down_bias",
    },
    LayerTensorNames {
        coefficients: "gab.coefficients.6.weight",
        norm_attention: "blocks.6.norm_attention.scale",
        qkv: "blocks.6.qkv",
        project: "blocks.6.project",
        project_bias: "blocks.6.project_bias",
        norm_ffn: "blocks.6.norm_ffn.scale",
        up: "blocks.6.up",
        up_bias: "blocks.6.up_bias",
        down: "blocks.6.down",
        down_bias: "blocks.6.down_bias",
    },
    LayerTensorNames {
        coefficients: "gab.coefficients.7.weight",
        norm_attention: "blocks.7.norm_attention.scale",
        qkv: "blocks.7.qkv",
        project: "blocks.7.project",
        project_bias: "blocks.7.project_bias",
        norm_ffn: "blocks.7.norm_ffn.scale",
        up: "blocks.7.up",
        up_bias: "blocks.7.up_bias",
        down: "blocks.7.down",
        down_bias: "blocks.7.down_bias",
    },
];

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
    geometric_bias: &[f32],
    width: usize,
    w: &ElasticBlockWeights,
) {
    let tokens = 64;
    let dot_kernel = *DOT_KERNEL.get_or_init(select_dot_kernel);
    let axpy_kernel = *AXPY_KERNEL.get_or_init(select_axpy_kernel);
    let mut normalized = [0f32; 64 * D_MODEL];
    for tok in 0..tokens {
        rmsnorm(
            &values[tok * D_MODEL..tok * D_MODEL + width],
            w.norm_attention,
            width,
            &mut normalized[tok * D_MODEL..tok * D_MODEL + width],
            dot_kernel,
        );
    }

    // qkv: full tensor shape (3, D_MODEL, D_MODEL); slice [:, :width, :width]
    // per PyTorch reshape(3*width, width) -- row r of the reshaped matrix for
    // component c (0=q,1=k,2=v) and output index o is qkv[c, o, :width].
    let mut q = vec![0f32; tokens * width];
    let mut k = vec![0f32; tokens * width];
    let mut v = vec![0f32; tokens * width];
    if inference_threads() > 1 {
        let split = tokens / 2;
        let (q_left, q_right) = q.split_at_mut(split * width);
        let (k_left, k_right) = k.split_at_mut(split * width);
        let (v_left, v_right) = v.split_at_mut(split * width);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                project_qkv(
                    &normalized[..split * D_MODEL],
                    split,
                    width,
                    w.qkv,
                    q_left,
                    k_left,
                    v_left,
                );
            });
            project_qkv(
                &normalized[split * D_MODEL..],
                tokens - split,
                width,
                w.qkv,
                q_right,
                k_right,
                v_right,
            );
        });
    } else {
        project_qkv(&normalized, tokens, width, w.qkv, &mut q, &mut k, &mut v);
    }

    // Scaled dot-product attention. Heads are independent, so split them
    // across two scoped workers without changing per-head reduction order.
    let mut attended = vec![0.0f32; tokens * width];
    if inference_threads() > 1 {
        let split = HEADS / 2;
        std::thread::scope(|scope| {
            let left = scope.spawn(|| attention_heads(&q, &k, &v, geometric_bias, width, 0, split));
            let right = attention_heads(&q, &k, &v, geometric_bias, width, split, HEADS);
            let left = left.join().expect("attention worker panicked");
            copy_attention_heads(&left, &mut attended, width, 0, split);
            copy_attention_heads(&right, &mut attended, width, split, HEADS);
        });
    } else {
        let all = attention_heads(&q, &k, &v, geometric_bias, width, 0, HEADS);
        copy_attention_heads(&all, &mut attended, width, 0, HEADS);
    }

    // project (width x width slice of D_MODEL x D_MODEL) + residual
    let mut delta = vec![0f32; tokens * width];
    linear_full(
        &attended,
        tokens,
        width,
        w.project,
        D_MODEL,
        width,
        Some(w.project_bias),
        &mut delta,
    );
    for tok in 0..tokens {
        axpy_kernel(
            1.0,
            &delta[tok * width..(tok + 1) * width],
            &mut values[tok * D_MODEL..tok * D_MODEL + width],
        );
    }

    // FFN
    let mut normalized2 = vec![0f32; tokens * width];
    for tok in 0..tokens {
        rmsnorm(
            &values[tok * D_MODEL..tok * D_MODEL + width],
            w.norm_ffn,
            width,
            &mut normalized2[tok * width..tok * width + width],
            dot_kernel,
        );
    }
    // up: shape (2, D_MODEL, D_MODEL) sliced to (2, width, width), reshaped (2*width, width)
    let mut hidden = vec![0f32; tokens * width];
    let mut gate = vec![0f32; tokens * width];
    if inference_threads() > 1 {
        let split = tokens / 2;
        let (hidden_left, hidden_right) = hidden.split_at_mut(split * width);
        let (gate_left, gate_right) = gate.split_at_mut(split * width);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                project_up(
                    &normalized2[..split * width],
                    split,
                    width,
                    w.up,
                    w.up_bias,
                    hidden_left,
                    gate_left,
                );
            });
            project_up(
                &normalized2[split * width..],
                tokens - split,
                width,
                w.up,
                w.up_bias,
                hidden_right,
                gate_right,
            );
        });
    } else {
        project_up(
            &normalized2,
            tokens,
            width,
            w.up,
            w.up_bias,
            &mut hidden,
            &mut gate,
        );
    }
    let mut ffn_input = vec![0f32; tokens * width];
    for i in 0..tokens * width {
        ffn_input[i] = silu(gate[i]) * hidden[i];
    }
    let mut ffn_out = vec![0f32; tokens * width];
    linear_full(
        &ffn_input,
        tokens,
        width,
        w.down,
        D_MODEL,
        width,
        Some(w.down_bias),
        &mut ffn_out,
    );
    for tok in 0..tokens {
        axpy_kernel(
            1.0,
            &ffn_out[tok * width..(tok + 1) * width],
            &mut values[tok * D_MODEL..tok * D_MODEL + width],
        );
    }
}

pub fn validate_input(input: &PositionInput) -> Result<(), String> {
    if input.pieces.iter().any(|&piece| piece > 12) {
        return Err("Chessformer input contains an invalid piece token".into());
    }
    if input.castling >= 16 || input.ep_file > 8 || input.time_class >= 5 {
        return Err("Chessformer input contains invalid global state".into());
    }
    if input.policy_kind > POLICY_GUIDE {
        return Err("Chessformer input contains invalid policy kind".into());
    }
    if input.legal_actions.is_empty() || input.legal_actions.len() > MAX_LEGAL_ACTIONS {
        return Err("Chessformer legal action count outside 1..=218".into());
    }
    if input
        .legal_actions
        .iter()
        .any(|&action| action as usize >= 64 * 64 * 5)
    {
        return Err("Chessformer input contains invalid action encoding".into());
    }
    Ok(())
}

pub fn forward(weights: &ChessformerWeights, input: &PositionInput) -> ForwardOutput {
    forward_at_exit(weights, input, InferenceExit::Layer8Width256)
}

pub fn forward_at_exit(
    weights: &ChessformerWeights,
    input: &PositionInput,
    exit: InferenceExit,
) -> ForwardOutput {
    validate_input(input).expect("invalid Chessformer runtime input");
    let dot_kernel = *DOT_KERNEL.get_or_init(select_dot_kernel);
    let axpy_kernel = *AXPY_KERNEL.get_or_init(select_axpy_kernel);
    let width = exit.width();
    let layers = exit.layers();
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
    embed(
        weights.t("castling_embedding.weight"),
        input.castling as usize,
        D_MODEL,
        &mut global_state,
    );
    embed(
        weights.t("ep_embedding.weight"),
        input.ep_file as usize,
        D_MODEL,
        &mut global_state,
    );
    let halfmove_bucket = ((input.halfmove_clock / 8) as usize).min(15);
    embed(
        weights.t("halfmove_embedding.weight"),
        halfmove_bucket,
        D_MODEL,
        &mut global_state,
    );
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
            let row = o * D_MODEL;
            projected[sq * GAB_TOKEN_PROJECTION + o] = dot_kernel(
                &values[sq * D_MODEL..sq * D_MODEL + width],
                &token_projection.data[row..row + width],
            );
        }
    }
    // flatten(1): (64*GAB_TOKEN_PROJECTION) -> compress: Linear(64*d1, hidden)
    let compress_weight = weights.t("gab.compress.weight");
    let compress_bias = weights.t("gab.compress.bias");
    let flat_width = 64 * GAB_TOKEN_PROJECTION;
    let mut compressed = [0f32; GAB_HIDDEN];
    for o in 0..GAB_HIDDEN {
        let row = o * flat_width;
        compressed[o] = gelu(
            compress_bias.get(o)
                + dot_kernel(&projected, &compress_weight.data[row..row + flat_width]),
        );
    }
    let mut context = [0f32; GAB_HIDDEN];
    let gab_norm = weights.t("gab.norm");
    {
        let mean_sq = dot_kernel(&compressed, &compressed) / GAB_HIDDEN as f32;
        let inv = 1.0 / (mean_sq + 1e-6).sqrt();
        for i in 0..GAB_HIDDEN {
            context[i] = compressed[i] * inv * gab_norm.get(i);
        }
    }

    let templates = weights.t("gab.templates"); // (GAB_TEMPLATES, 64, 64)

    for layer_names in LAYER_TENSOR_NAMES.iter().take(layers) {
        let coeff_weight = weights.t(layer_names.coefficients); // (HEADS*GAB_TEMPLATES, GAB_HIDDEN)
        let mut coefficients = [0f32; HEADS * GAB_TEMPLATES];
        for o in 0..HEADS * GAB_TEMPLATES {
            let row = o * GAB_HIDDEN;
            coefficients[o] = dot_kernel(&context, &coeff_weight.data[row..row + GAB_HIDDEN]);
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
                axpy_kernel(
                    c,
                    &templates.data[tmpl_base..tmpl_base + 64 * 64],
                    &mut geometric_bias[out_base..out_base + 64 * 64],
                );
            }
        }
        let block = ElasticBlockWeights {
            norm_attention: weights.t(layer_names.norm_attention),
            qkv: weights.t(layer_names.qkv),
            project: weights.t(layer_names.project),
            project_bias: weights.t(layer_names.project_bias),
            norm_ffn: weights.t(layer_names.norm_ffn),
            up: weights.t(layer_names.up),
            up_bias: weights.t(layer_names.up_bias),
            down: weights.t(layer_names.down),
            down_bias: weights.t(layer_names.down_bias),
        };
        elastic_block(&mut values, &geometric_bias, width, &block);
    }

    // --- Final norm + pooling ---
    let mut normalized = vec![0f32; tokens * width];
    for tok in 0..tokens {
        rmsnorm(
            &values[tok * D_MODEL..tok * D_MODEL + width],
            weights.t("final_norm.scale"),
            width,
            &mut normalized[tok * width..tok * width + width],
            dot_kernel,
        );
    }
    let mut pooled = [0f32; D_MODEL];
    for tok in 0..tokens {
        axpy_kernel(
            1.0,
            &normalized[tok * width..(tok + 1) * width],
            &mut pooled[..width],
        );
    }
    for i in 0..width {
        pooled[i] /= tokens as f32;
    }

    // --- Value head (evidential WDL) ---
    let value_weight = weights.t("value_weight"); // (3, D_MODEL)
    let value_bias = weights.t("value_bias");
    let mut evidence = [0f32; 3];
    for o in 0..3 {
        let row = o * D_MODEL;
        evidence[o] = softplus(
            value_bias.get(o) + dot_kernel(&pooled[..width], &value_weight.data[row..row + width]),
        );
    }

    // --- History context (no move history at inference: history_len=0) ---
    let normalized_rating = (((input.rating as f32) - 100.0) / 3550.0).clamp(0.0, 1.0);
    let mut history_vec = [0f32; HISTORY_WIDTH];
    embed(
        weights.t("time_embedding.weight"),
        input.time_class,
        HISTORY_WIDTH,
        &mut history_vec,
    );
    let rating_weight = weights.t("rating_weight");
    let rating_bias = weights.t("rating_bias");
    for i in 0..HISTORY_WIDTH {
        history_vec[i] += normalized_rating * rating_weight.get(i) + rating_bias.get(i);
    }
    let history_project_w = weights.t("history_project.weight"); // (D_MODEL, HISTORY_WIDTH)
    let history_project_b = weights.t("history_project.bias");
    let mut history_full = [0f32; D_MODEL];
    for o in 0..D_MODEL {
        let row = o * HISTORY_WIDTH;
        history_full[o] = history_project_b.get(o)
            + dot_kernel(
                &history_vec,
                &history_project_w.data[row..row + HISTORY_WIDTH],
            );
    }

    // --- Policy heads ---
    let mut body = vec![0f32; tokens * width];
    persona_full(
        &normalized,
        tokens,
        width,
        input.policy_kind,
        &history_full,
        weights.t("policy_body.weight"),
        Some(weights.t("policy_body.bias")),
        weights.t("policy_body.adapter_a"),
        weights.t("policy_body.adapter_b"),
        &mut body,
    );
    for v in body.iter_mut() {
        *v = gelu(*v);
    }
    let mut source_values = vec![0f32; tokens * width];
    persona_full(
        &body,
        tokens,
        width,
        input.policy_kind,
        &history_full,
        weights.t("policy_source.weight"),
        None,
        weights.t("policy_source.adapter_a"),
        weights.t("policy_source.adapter_b"),
        &mut source_values,
    );
    let mut target_values = vec![0f32; tokens * width];
    persona_full(
        &body,
        tokens,
        width,
        input.policy_kind,
        &history_full,
        weights.t("policy_target.weight"),
        None,
        weights.t("policy_target.adapter_a"),
        weights.t("policy_target.adapter_b"),
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
        let source = &source_values[source_sq * width..(source_sq + 1) * width];
        let target = &target_values[target_sq * width..(target_sq + 1) * width];
        logits[idx] = dot_kernel(source, target) * scale + promotion_bias.get(promotion);
    }

    // --- Regret head ---
    let regret_from = weights.t("regret_from"); // (REGRET_WIDTH, D_MODEL)
    let regret_to = weights.t("regret_to");
    let regret_promotion = weights.t("regret_promotion.weight"); // (5, REGRET_WIDTH)
    let regret_output_w = weights.t("regret_output.weight"); // (2, REGRET_WIDTH)
    let regret_output_b = weights.t("regret_output.bias");

    let mut regret_source_all = vec![0f32; 64 * REGRET_WIDTH];
    let mut regret_target_all = vec![0f32; 64 * REGRET_WIDTH];
    if let (Some(quantized_from), Some(quantized_to)) =
        (&regret_from.quantized, &regret_to.quantized)
    {
        project_regret_quantized(
            &normalized,
            64,
            width,
            quantized_from,
            quantized_to,
            &mut regret_source_all,
            &mut regret_target_all,
        );
    } else {
        for o in 0..REGRET_WIDTH {
            let row = o * D_MODEL;
            let weight_from = &regret_from.data[row..row + width];
            let weight_to = &regret_to.data[row..row + width];
            for sq in 0..64 {
                let input = &normalized[sq * width..(sq + 1) * width];
                regret_source_all[sq * REGRET_WIDTH + o] = dot_kernel(input, weight_from);
                regret_target_all[sq * REGRET_WIDTH + o] = dot_kernel(input, weight_to);
            }
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
        let raw0 =
            regret_output_b.get(0) + dot_kernel(&hidden, &regret_output_w.data[..REGRET_WIDTH]);
        let raw1 = regret_output_b.get(1)
            + dot_kernel(
                &hidden,
                &regret_output_w.data[REGRET_WIDTH..2 * REGRET_WIDTH],
            );
        regret_mean[idx] = softplus(raw0);
        regret_log_scale[idx] = raw1.clamp(-8.0, 4.0);
    }

    ForwardOutput {
        logits,
        regret_mean,
        regret_log_scale,
        evidence,
        representation: pooled,
    }
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
    let dot_kernel = *DOT_KERNEL.get_or_init(select_dot_kernel);
    let dot_four_kernel = *DOT4_KERNEL.get_or_init(select_dot4_kernel);
    let axpy_kernel = *AXPY_KERNEL.get_or_init(select_axpy_kernel);
    linear_full(values, tokens, width, weight, D_MODEL, width, bias, out);
    let a_base = policy_kind * POLICY_ADAPTER_RANK * D_MODEL;
    let b_base = policy_kind * D_MODEL * POLICY_ADAPTER_RANK;
    let mut adapter_input = values.to_vec();
    for tok in 0..tokens {
        axpy_kernel(
            1.0,
            &history[..width],
            &mut adapter_input[tok * width..(tok + 1) * width],
        );
    }
    let mut low = vec![0f32; tokens * POLICY_ADAPTER_RANK];
    const TOKEN_BLOCK: usize = 4;
    for token_start in (0..tokens).step_by(TOKEN_BLOCK) {
        let token_end = (token_start + TOKEN_BLOCK).min(tokens);
        for r in 0..POLICY_ADAPTER_RANK {
            let row = a_base + r * D_MODEL;
            let adapter = &adapter_a.data[row..row + width];
            if token_end - token_start == 4 {
                let result = dot_four_kernel(&adapter_input[token_start * width..], width, adapter);
                for lane in 0..4 {
                    low[(token_start + lane) * POLICY_ADAPTER_RANK + r] = result[lane];
                }
            } else {
                for tok in token_start..token_end {
                    low[tok * POLICY_ADAPTER_RANK + r] =
                        dot_kernel(&adapter_input[tok * width..(tok + 1) * width], adapter);
                }
            }
        }
    }
    for tok in 0..tokens {
        let low_row = &low[tok * POLICY_ADAPTER_RANK..(tok + 1) * POLICY_ADAPTER_RANK];
        for o in 0..width {
            let row = b_base + o * POLICY_ADAPTER_RANK;
            out[tok * width + o] +=
                dot_kernel(low_row, &adapter_b.data[row..row + POLICY_ADAPTER_RANK])
                    / POLICY_ADAPTER_RANK as f32;
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
        let mover_bb = if flip {
            pos.bb[us][p].swap_bytes()
        } else {
            pos.bb[us][p]
        };
        let opp_bb = if flip {
            pos.bb[them][p].swap_bytes()
        } else {
            pos.bb[them][p]
        };
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

    let (mk, mq, ok, oq) = if flip {
        (BK, BQ, WK, WQ)
    } else {
        (WK, WQ, BK, BQ)
    };
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
            let (from, to) = if flip {
                (m.from() ^ 56, m.to() ^ 56)
            } else {
                (m.from(), m.to())
            };
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
    use std::sync::Mutex;

    static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

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

    /// Cross-checked against tools/reference_forward_aegis_v4.py run on the
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
    fn integer_microkernels_match_scalar() {
        let width = D_MODEL;
        let activations = (0..4 * width)
            .map(|index| ((index * 7919 + 17) % 65_535) as i32 - 32_767)
            .map(|value| value as i16)
            .collect::<Vec<_>>();
        let weights = (0..3 * width)
            .map(|index| ((index * 97 + 11) % 255) as i16 - 127)
            .map(|value| value as i8)
            .collect::<Vec<_>>();
        let weight_0 = &weights[..width];
        let weight_1 = &weights[width..2 * width];
        let weight_2 = &weights[2 * width..];

        assert_eq!(
            dot_four_two_i16_i8(&activations, width, weight_0, weight_1),
            dot4x2_i16_i8_scalar(&activations, width, weight_0, weight_1),
        );
        assert_eq!(
            dot_four_three_i16_i8(&activations, width, weight_0, weight_1, weight_2),
            dot4x3_i16_i8_scalar(&activations, width, weight_0, weight_1, weight_2),
        );
    }

    #[test]
    fn activation_quantizer_matches_scalar() {
        let source = (0..D_MODEL)
            .map(|index| ((index as f32 * 0.123_456_7).sin() * 3.25) + index as f32 * 1e-4)
            .collect::<Vec<_>>();
        let mut selected = [0i16; D_MODEL];
        let mut scalar = [0i16; D_MODEL];
        let selected_scale = quantize_i16(&source, &mut selected);
        let scalar_scale = quantize_i16_scalar(&source, &mut scalar);
        assert_eq!(selected_scale, scalar_scale);
        for (index, (&got, &expected)) in selected.iter().zip(&scalar).enumerate() {
            assert!(
                (i32::from(got) - i32::from(expected)).abs() <= 1,
                "quantized activation {index}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn rejects_malformed_runtime_input() {
        let mut input = start_position_input();
        input.pieces[0] = 13;
        assert!(validate_input(&input).is_err());
        input.pieces[0] = 4;
        input.legal_actions.clear();
        assert!(validate_input(&input).is_err());
    }

    #[test]
    fn rejects_incomplete_runtime_tensor_schema() {
        let weights = ChessformerWeights {
            tensors: HashMap::new(),
        };
        assert!(weights.validate_schema().is_err());
    }

    #[test]
    fn start_position_matches_python_reference() {
        let weights = load_reference_weights();
        let input = start_position_input();
        assert_eq!(input.legal_actions.len(), 20);
        let output = forward(&weights, &input);

        let expected_logits = [
            -1.94899, -1.53363, -1.711539, -0.696044, -1.576939, -0.456242, -0.652165, -0.287219,
            -1.707068, -1.615397, -1.915703, -1.426473, -1.761397, -1.96128, -1.868046, -1.067433,
            -1.09751, -2.014229, -1.843139, -1.567619,
        ];
        for (i, (&got, &want)) in output.logits.iter().zip(expected_logits.iter()).enumerate() {
            assert!(
                (got - want).abs() < 5e-3,
                "logit[{i}]: got {got}, want {want}"
            );
        }

        let expected_evidence = [
            3.3165156841278076f32,
            0.04533608630299568,
            3.893404245376587,
        ];
        for i in 0..3 {
            assert!(
                (output.evidence[i] - expected_evidence[i]).abs() < 5e-3,
                "evidence[{i}]: got {}, want {}",
                output.evidence[i],
                expected_evidence[i]
            );
        }

        let expected_representation = [
            0.160754f32,
            -0.092479,
            0.932122,
            -0.786012,
            0.049596,
            -0.239851,
            -0.521234,
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
            -1.473108, -1.365497, -1.90742, -0.724959, -1.290341, -1.222262, -0.451867, -1.899517,
            -1.401405, -1.672434, -1.534469, -1.091158, -2.957112,
        ];
        for (i, (&got, &want)) in output.logits.iter().zip(expected_logits.iter()).enumerate() {
            assert!(
                (got - want).abs() < 5e-3,
                "logit[{i}]: got {got}, want {want}"
            );
        }

        let expected_evidence = [
            3.3614559173583984f32,
            0.04310621693730354,
            3.917607545852661,
        ];
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
    fn elastic_exit_shapes_and_finiteness() {
        let weights = load_reference_weights();
        let input = start_position_input();
        for exit in [
            InferenceExit::Layer2Width128,
            InferenceExit::Layer4Width192,
            InferenceExit::Layer8Width256,
        ] {
            let output = forward_at_exit(&weights, &input, exit);
            assert_eq!(output.logits.len(), input.legal_actions.len());
            assert!(output.logits.iter().all(|value| value.is_finite()));
            assert!(output.regret_mean.iter().all(|value| value.is_finite()));
            assert!(output.evidence.iter().all(|value| value.is_finite()));
            assert!(output.representation[..exit.width()]
                .iter()
                .all(|value| value.is_finite()));
            assert!(output.representation[exit.width()..]
                .iter()
                .all(|&value| value == 0.0));
        }
    }

    #[test]
    fn integer_matrix_path_stays_close_to_dequantized_path() {
        let integer_weights = load_reference_weights();
        let mut dequantized_weights = load_reference_weights();
        for tensor in dequantized_weights.tensors.values_mut() {
            tensor.quantized = None;
        }
        let input = start_position_input();

        for exit in [
            InferenceExit::Layer2Width128,
            InferenceExit::Layer4Width192,
            InferenceExit::Layer8Width256,
        ] {
            let integer = forward_at_exit(&integer_weights, &input, exit);
            let dequantized = forward_at_exit(&dequantized_weights, &input, exit);
            let maximum_delta = |left: &[f32], right: &[f32]| {
                left.iter()
                    .zip(right)
                    .map(|(&lhs, &rhs)| (lhs - rhs).abs())
                    .fold(0.0f32, f32::max)
            };
            let logit_delta = maximum_delta(&integer.logits, &dequantized.logits);
            let regret_mean_delta = maximum_delta(&integer.regret_mean, &dequantized.regret_mean);
            let regret_scale_delta =
                maximum_delta(&integer.regret_log_scale, &dequantized.regret_log_scale);
            let evidence_delta = maximum_delta(&integer.evidence, &dequantized.evidence);
            let representation_delta = maximum_delta(
                &integer.representation[..exit.width()],
                &dequantized.representation[..exit.width()],
            );
            eprintln!(
                "{exit:?} integer-vs-f32 max deltas: logits={logit_delta:.8}, \
                 regret_mean={regret_mean_delta:.8}, regret_log_scale={regret_scale_delta:.8}, \
                 evidence={evidence_delta:.8}, representation={representation_delta:.8}"
            );
            for (name, delta) in [
                ("logits", logit_delta),
                ("regret mean", regret_mean_delta),
                ("regret log scale", regret_scale_delta),
                ("evidence", evidence_delta),
                ("representation", representation_delta),
            ] {
                assert!(
                    delta < 5e-4,
                    "{exit:?} {name} drift {delta} exceeds the retained-int8 matrix gate"
                );
            }
        }
    }

    /// Cross-checked against `tools/reference_forward_aegis_v4.py
    /// artifacts/unarchitectured-v1-final.unarchv1 --all-exits` on the same
    /// real checkpoint. `elastic_exit_shapes_and_finiteness` above only
    /// checked shape/finiteness and would not have caught the real bug this
    /// closes: the shared `pooled` accumulator was fixed at D_MODEL length
    /// and fed straight into `scaled_add`/`dot_product` calls sized to the
    /// narrower exit's `width`, corrupting every value derived from pooling
    /// (evidence, representation) for any exit narrower than the full one.
    #[test]
    fn narrow_exits_match_python_reference() {
        let weights = load_reference_weights();
        let input = start_position_input();

        let expected_logits_128 = [
            -1.03965724,
            -0.70582396,
            -1.05701911,
            -0.38986212,
            -0.7868678,
            -0.45352447,
            -0.53338712,
            -0.33900601,
            -0.82796401,
            -0.99180812,
            -0.9507128,
            -0.7821846,
            -0.85881901,
            -1.325477,
            -1.05770719,
            -0.72489899,
            -0.79163575,
            -1.16803908,
            -1.35581696,
            -0.71138501,
        ];
        let expected_evidence_128 = [3.27648902f32, 0.0294219, 3.85723782];
        let expected_representation_128 = [
            -0.51477349f32,
            0.00801361,
            0.37547749,
            -0.60401458,
            0.53481448,
            -0.34796605,
            -0.51862764,
            -1.44890022,
        ];

        let output_128 = forward_at_exit(&weights, &input, InferenceExit::Layer2Width128);
        for (i, (&got, &want)) in output_128
            .logits
            .iter()
            .zip(expected_logits_128.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 5e-3,
                "exit 2/128 logit[{i}]: got {got}, want {want}"
            );
        }
        // evidence and representation both derive from the same pooled
        // 64-term sequential f32 sum, whose accumulation order differs from
        // PyTorch's internal mean reduction -- see the comment below on
        // representation for why these two get a slightly looser bound than
        // logits/best-move, which don't depend on pooled and hold to 5e-3.
        for i in 0..3 {
            assert!(
                (output_128.evidence[i] - expected_evidence_128[i]).abs() < 2e-2,
                "exit 2/128 evidence[{i}]: got {}, want {}",
                output_128.evidence[i],
                expected_evidence_128[i]
            );
        }
        // Looser than the 5e-3 used everywhere else (2e-2): this is a
        // 64-term sequential f32 sum (board-square pooling) whose accumulation
        // order differs from PyTorch's internal mean reduction, and the two
        // orders can legitimately disagree by a bit more in the last couple
        // of digits. logits/evidence/best-move above -- the values that
        // actually decide anything -- all hold at the tighter 5e-3.
        for i in 0..8 {
            assert!(
                (output_128.representation[i] - expected_representation_128[i]).abs() < 2e-2,
                "exit 2/128 representation[{i}]: got {}, want {}",
                output_128.representation[i],
                expected_representation_128[i]
            );
        }
        let best_128 = (0..input.legal_actions.len())
            .max_by(|&a, &b| {
                output_128.logits[a]
                    .partial_cmp(&output_128.logits[b])
                    .unwrap()
            })
            .unwrap();
        assert_eq!(best_128, 7, "exit 2/128 best action index");
        assert_eq!(
            input.legal_actions[best_128], 1350,
            "exit 2/128 best action encoding"
        );

        let expected_logits_192 = [
            -1.12425447,
            -0.75749695,
            -1.15513003,
            -0.43399757,
            -0.82629299,
            -0.42284912,
            -0.55280173,
            -0.44800025,
            -0.87793761,
            -1.07046652,
            -1.14484096,
            -0.76420325,
            -0.97380733,
            -1.37777829,
            -1.10875309,
            -0.6861124,
            -0.66685504,
            -1.18792248,
            -1.37111151,
            -0.83386445,
        ];
        let expected_evidence_192 = [3.53003931f32, 0.02421277, 4.04606676];
        let expected_representation_192 = [
            0.2943553f32,
            -0.58349973,
            0.63125348,
            -0.44087225,
            0.27256548,
            -0.8262307,
            -0.84874725,
            -1.63658905,
        ];

        let output_192 = forward_at_exit(&weights, &input, InferenceExit::Layer4Width192);
        for (i, (&got, &want)) in output_192
            .logits
            .iter()
            .zip(expected_logits_192.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 5e-3,
                "exit 4/192 logit[{i}]: got {got}, want {want}"
            );
        }
        for i in 0..3 {
            assert!(
                (output_192.evidence[i] - expected_evidence_192[i]).abs() < 2e-2,
                "exit 4/192 evidence[{i}]: got {}, want {}",
                output_192.evidence[i],
                expected_evidence_192[i]
            );
        }
        for i in 0..8 {
            assert!(
                (output_192.representation[i] - expected_representation_192[i]).abs() < 2e-2,
                "exit 4/192 representation[{i}]: got {}, want {}",
                output_192.representation[i],
                expected_representation_192[i]
            );
        }
        let best_192 = (0..input.legal_actions.len())
            .max_by(|&a, &b| {
                output_192.logits[a]
                    .partial_cmp(&output_192.logits[b])
                    .unwrap()
            })
            .unwrap();
        assert_eq!(best_192, 5, "exit 4/192 best action index");
        assert_eq!(
            input.legal_actions[best_192], 1227,
            "exit 4/192 best action encoding"
        );
    }

    #[test]
    #[ignore]
    fn benchmark_forward_pass() {
        let _guard = BENCHMARK_LOCK.lock().unwrap();
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

    #[test]
    #[ignore]
    fn benchmark_integer_matrix_speedup() {
        let _guard = BENCHMARK_LOCK.lock().unwrap();
        let integer_weights = load_reference_weights();
        let mut dequantized_weights = load_reference_weights();
        for tensor in dequantized_weights.tensors.values_mut() {
            tensor.quantized = None;
        }
        let input = start_position_input();
        for _ in 0..5 {
            std::hint::black_box(forward(&integer_weights, &input));
            std::hint::black_box(forward(&dequantized_weights, &input));
        }

        let rounds = 4;
        let calls_per_round = 50;
        let mut integer_elapsed = std::time::Duration::ZERO;
        let mut dequantized_elapsed = std::time::Duration::ZERO;
        for round in 0..rounds {
            let measure = |weights: &ChessformerWeights| {
                let started = std::time::Instant::now();
                for _ in 0..calls_per_round {
                    std::hint::black_box(forward(weights, &input));
                }
                started.elapsed()
            };
            if round % 2 == 0 {
                dequantized_elapsed += measure(&dequantized_weights);
                integer_elapsed += measure(&integer_weights);
            } else {
                integer_elapsed += measure(&integer_weights);
                dequantized_elapsed += measure(&dequantized_weights);
            }
        }
        let calls = rounds * calls_per_round;
        println!(
            "{calls} calls/path: dequantized={:?}/call retained-int8={:?}/call speedup={:.4}x",
            dequantized_elapsed / calls,
            integer_elapsed / calls,
            dequantized_elapsed.as_secs_f64() / integer_elapsed.as_secs_f64(),
        );
    }

    #[test]
    #[ignore]
    fn benchmark_exit_ladder() {
        let _guard = BENCHMARK_LOCK.lock().unwrap();
        let weights = load_reference_weights();
        let input = start_position_input();
        for exit in [
            InferenceExit::Layer2Width128,
            InferenceExit::Layer4Width192,
            InferenceExit::Layer8Width256,
        ] {
            for _ in 0..3 {
                std::hint::black_box(forward_at_exit(&weights, &input, exit));
            }
            let calls = match exit {
                InferenceExit::Layer2Width128 => 500,
                InferenceExit::Layer4Width192 => 300,
                InferenceExit::Layer8Width256 => 100,
            };
            let started = std::time::Instant::now();
            for _ in 0..calls {
                std::hint::black_box(forward_at_exit(&weights, &input, exit));
            }
            let elapsed = started.elapsed();
            println!(
                "{:?}: {} calls in {:?} -> {:?}/call",
                exit,
                calls,
                elapsed,
                elapsed / calls,
            );
        }
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
            .max_by(|&a, &b| {
                live_output.logits[a]
                    .partial_cmp(&live_output.logits[b])
                    .unwrap()
            })
            .unwrap();
        assert_eq!(
            live_input.legal_actions[best], 1350,
            "live conversion should still pick g1-f3 as the best move"
        );
    }
}
