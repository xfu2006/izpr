export RUSTFLAGS="$RUSTFLAGS -A dead_code -Awarnings"
export RUST_BACKTRACE=1 
export RUST_TEST_THREADS=1
#mpirun --hostfile scripts/host1 -n 2 cargo test dis_poly -- --nocapture 
#mpirun --hostfile scripts/host1 -n 8 cargo test -- --nocapture 
#mpirun --hostfile scripts/host1 -n 4 cargo test dis_poly::tests::quick -- --nocapture 
#mpirun --hostfile scripts/host1 -n 4 cargo test disfft::tests -- --nocapture 
#mpirun --hostfile scripts/host1 -n 8 cargo test dis_poly::tests -- --nocapture 
#mpirun --hostfile scripts/host1 -n 4 cargo test dis_vec::tests -- --nocapture 
#mpirun --hostfile scripts/host1 -n 8 cargo test dis_key::tests -- --nocapture 
#mpirun --hostfile scripts/host1 -n 1 cargo test serial::tests -- --nocapture 
#mpirun --hostfile scripts/host1 -n 2 cargo test kzg::tests -- --nocapture 
#mpirun --hostfile scripts/host1 -n 4 cargo test r1cs_tests::tests -- --nocapture 
#mpirun --hostfile scripts/host1 -n 4 cargo test groth16_test::tests -- --nocapture 

#cargo test serial_logical_poly_utils::tests -- --nocapture 
#cargo test serial_poly_utils::tests -- --nocapture 
#cargo test serial_group_fft2::tests -- --nocapture 
#cargo test serial_cq::tests -- --nocapture 
#cargo test serial_rng::tests -- --nocapture 
#cargo test serial_pn_lookup::tests -- --nocapture 
cargo test serial_asset1_v2::tests -- --nocapture 
