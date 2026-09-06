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
}

static CAPABILITIES: OnceLock<Capabilities> = OnceLock::new();

#[inline]
pub(crate) fn capabilities() -> Capabilities {
    *CAPABILITIES.get_or_init(detect)
}

#[inline]
pub(crate) fn has_avx2() -> bool {
    capabilities().avx2
}

#[inline]
pub(crate) fn has_avx2_fma() -> bool {
    let caps = capabilities();
    caps.avx2 && caps.fma
}

#[inline]
pub(crate) fn has_avx2_sse41() -> bool {
    let caps = capabilities();
    caps.avx2 && caps.sse41
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn detect() -> Capabilities {
    if std::env::var_os("UNCHESSED_DISABLE_SIMD").is_some() {
        return Capabilities {
            avx2: false,
            fma: false,
            sse41: false,
        };
    }
    Capabilities {
        avx2: std::is_x86_feature_detected!("avx2"),
        fma: std::is_x86_feature_detected!("fma"),
        sse41: std::is_x86_feature_detected!("sse4.1"),
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
fn detect() -> Capabilities {
    Capabilities {
        avx2: false,
        fma: false,
        sse41: false,
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
    }
}
