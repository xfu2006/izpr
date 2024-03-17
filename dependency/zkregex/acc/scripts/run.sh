export RUSTFLAGS="$RUSTFLAGS -A dead_code"
export RUST_BACKTRACE=full
mpirun -np 4 --hostfile scripts/host1 /home/xiang/Desktop/ProofCarryExec/Code/zkregex/acc/target/release/acc exp
