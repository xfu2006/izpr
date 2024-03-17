RUSTFLAGS="-C target-feature=+bmi2,+adx" cargo +nightly build --features asm
