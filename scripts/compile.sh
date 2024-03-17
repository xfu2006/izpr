RUSTFLAGS=-Awarnings
#RUSTFLAGS="$RUSTFLAGS -A dead_code" cargo build --release 
#RUSTFLAGS="$RUSTFLAGS -A dead_code" cargo +nightly build --release
#RUSTFLAGS="$RUSTFLAGS -A dead_code -C target-cpu=native" cargo +nightly build --release
#RUSTFLAGS="$RUSTFLAGS -A dead_code -C target-feature=+bmi2,+adx" cargo +nightly build 
RUSTFLAGS="$RUSTFLAGS -A dead_code -C target-feature=+bmi2,+adx" cargo +nightly build --release

