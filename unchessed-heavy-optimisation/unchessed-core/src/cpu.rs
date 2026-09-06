//! Runtime CPU capability detection for the portable universal binary.
//!
//! The public predicates are resolved once through `OnceLock`. Callers use
//! them only to select already-verified `#[target_feature]` kernels; every
//! operation must retain a scalar reference implementation. This keeps the
//! default binary runnable on CPUs older than the host used to compile it.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Capabilities {
    pub avx2: bool,
    pub fma: bool,
    pub sse41: bool,
    /// AArch64 Advanced SIMD (NEON). NEON is mandatory on the supported
    /// AArch64 baseline, so this is a compile-target capability rather than
    /// a per-process probe.
    pub neon: bool,
    /// True when compiling for macOS on Apple Silicon (AArch64).
    pub apple_silicon: bool,
}

static CAPABILITIES: OnceLock<Capabilities> = OnceLock::new();

#[inline]
pub(crate) fn capabilities() -> Capabilities {
    *CAPABILITIES.get_or_init(detect)
}

#[allow(dead_code)]
#[inline]
pub(crate) fn has_avx2() -> bool {
    capabilities().avx2
}

#[allow(dead_code)]
#[inline]
pub(crate) fn has_avx2_fma() -> bool {
    let caps = capabilities();
    caps.avx2 && caps.fma
}

#[allow(dead_code)]
#[inline]
pub(crate) fn has_avx2_sse41() -> bool {
    let caps = capabilities();
    caps.avx2 && caps.sse41
}

#[allow(dead_code)]
#[inline]
pub(crate) fn has_neon() -> bool {
    capabilities().neon
}

#[allow(dead_code)]
#[inline]
pub(crate) fn is_apple_silicon() -> bool {
    capabilities().apple_silicon
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn detect() -> Capabilities {
    if std::env::var_os("UNCHESSED_DISABLE_SIMD").is_some() {
        return Capabilities {
            avx2: false,
            fma: false,
            sse41: false,
            neon: false,
            apple_silicon: false,
        };
    }
    Capabilities {
        avx2: std::is_x86_feature_detected!("avx2"),
        fma: std::is_x86_feature_detected!("fma"),
        sse41: std::is_x86_feature_detected!("sse4.1"),
        neon: false,
        apple_silicon: false,
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
fn detect() -> Capabilities {
    if std::env::var_os("UNCHESSED_DISABLE_SIMD").is_some() {
        return Capabilities {
            avx2: false,
            fma: false,
            sse41: false,
            neon: false,
            apple_silicon: false,
        };
    }
    Capabilities {
        avx2: false,
        fma: false,
        sse41: false,
        neon: cfg!(target_arch = "aarch64"),
        apple_silicon: cfg!(all(target_arch = "aarch64", target_os = "macos")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_implications_are_consistent() {
        let caps = capabilities();
        assert!(!has_avx2_fma() || (caps.avx2 && caps.fma));
        assert!(!has_avx2_sse41() || (caps.avx2 && caps.sse41));
        assert_eq!(has_avx2(), caps.avx2);
        assert_eq!(has_neon(), caps.neon);
        assert_eq!(is_apple_silicon(), caps.apple_silicon);
        assert!(!is_apple_silicon() || has_neon());
    }
}
