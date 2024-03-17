#/home/xiang/Desktop/ProofCarryExec/Code/zkregex/acc/target/release/acc profile
export RUST_BACKTRACE=1
#mpirun -mca btl_tcp_if_exclude br-7406b8232d20,docker0,lo,wlp61s0 -v --hostfile ./scripts/host1 -np 8 /home/xiang/Desktop/ProofCarryExec/Code/zkregex/acc/target/debug/acc profile
mpirun  --map-by node -mca btl self,vader,tcp -mca btl_tcp_if_exclude br-7406b8232d20,docker0,lo,wlp61s0 -v --hostfile ./scripts/host1 -np 4 /home/xiang/Desktop/ProofCarryExec/Code/zkregex/acc/target/release/acc profile 
