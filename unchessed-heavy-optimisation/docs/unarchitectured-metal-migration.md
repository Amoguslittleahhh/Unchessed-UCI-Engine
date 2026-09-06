# Unarchitectured Metal codename migration

## Canonical identity

The experimental architecture previously named **Unarchitectured v1** is now canonically named **Unarchitectured Metal**. Active Rust modules, runtime files, training/export tools, configuration files, artifacts, benchmark paths, and architecture research documents use the `unarchitectured_metal` or `unarchitectured-metal` naming convention according to the filesystem’s existing identifier style.

The canonical Rust modules are:

```text
unchessed-core/src/unarchitectured_metal.rs
unchessed-core/src/unarchitectured_metal_runtime.rs
```

The canonical Python package and runtime tooling use the same codename, for example:

```text
tools/unarchitectured_metal_package.py
tools/export_unarchitectured_metal.py
tools/unarchitectured_metal_runtime_readiness.py
```

## Wire-format migration

Newly exported tensor packages use the eight-byte `UNMETAL1` magic. The loader accepts both `UNMETAL1` and the historical `UNARCHV1` magic, so existing checkpoints remain usable. The version, header layout, tensor-table layout, alignment, checksums, quantization fields, and runtime tensor schema are unchanged.

| Interface | Canonical name | Compatibility behavior |
|---|---|---|
| Rust package parser | `unarchitectured_metal::TensorPackage` | Accepts `UNMETAL1` and legacy `UNARCHV1`. |
| Python package writer | `unarchitectured_metal_package` | Emits `UNMETAL1`; reads both formats. |
| Runtime module | `unarchitectured_metal_runtime` | Legacy `aegis_v4_runtime` re-export remains available. |
| Package parser module | `unarchitectured_metal` | Legacy `unarchitectured_v1` re-export remains available. |
| UCI options | `UnarchitecturedMetal*` | Old `Unarchitectured*` names remain accepted and advertised. |

## UCI compatibility

The canonical options are `UnarchitecturedMetalHint`, `UnarchitecturedMetalHintExit`, `UnarchitecturedMetalFile`, and `UnarchitecturedMetalMinTime`. The existing `UnarchitecturedHint`, `UnarchitecturedHintExit`, `UnarchitecturedFile`, and `UnarchitecturedMinTime` names remain accepted and continue to address the same internal option state.

## Validation

The core Rust suite passed 132 tests with no failures and six ignored performance tests. The workspace release build passed. The migrated Metal Python architecture suite passed 33 tests, with five skips caused by optional PyTorch calibration dependencies. A direct compatibility check confirmed that a package emitted as `UNMETAL1` can be transformed to the legacy `UNARCHV1` header and still be parsed successfully.

The two old Rust shim filenames remain deliberately present because external clients may import them directly. They contain no independent implementation and simply re-export the canonical Metal modules.
