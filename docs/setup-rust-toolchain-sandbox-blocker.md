# `scripts/setup-rust-toolchain.sh` run in this sandbox

Host: Debian 12 KVM (`e2b.local`, `systemd-detect-virt` = kvm). Script
is idempotent: PATH rustc, then `~/.cargo/env`, then apt, then rustup.

## Actual stdout (this round)

```
[setup-rust-toolchain] apt path: trying rustc + cargo...
W: Failed to fetch http://deb.debian.org/debian/dists/bookworm/InRelease  Connection failed [IP: 151.101.194.132 80]
W: Failed to fetch http://deb.debian.org/debian/dists/bookworm-updates/InRelease  Connection failed [IP: 151.101.2.132 80]
W: Failed to fetch http://deb.debian.org/debian-security/dists/bookworm-security/InRelease  Connection failed [IP: 151.101.194.132 80]
W: Some index files failed to download. They have been ignored, or old ones used instead.
E: Unable to locate package rustc
E: Unable to locate package cargo
[setup-rust-toolchain] apt path: FAILED (update/install error; often a filtered Debian CDN)
[setup-rust-toolchain] rustup path: curl https://sh.rustup.rs ...
curl: (35) OpenSSL SSL_connect: SSL_ERROR_SYSCALL in connection to sh.rustup.rs:443
[setup-rust-toolchain] rustup path: FAILED status=35 (often TLS to sh.rustup.rs / static.rust-lang.org)
[setup-rust-toolchain] BLOCKER: apt FAILED and rustup FAILED. No rustc in this environment.
[setup-rust-toolchain] Do not push unbuilt .rs changes as verified. Paste this log in the round doc.
```

Exit code **1**. Both install paths failed. This is the environment
blocker, not a skipped check.

`cargo test` was **not** run. The three Unarchitectured parity gates
were **not** executed here. `UnarchitecturedHint` stays default-off.
