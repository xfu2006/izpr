RUST_BACKTRACE=1
mpirun -mca btl_tcp_if_exclude br-7406b8232d20,docker0,lo,wlp61s0 -v --hostfile /tmp/tmp_nodelist.txt -np 4 /home/xiang/Desktop/ProofCarryExec/Code/zkregex/main/../acc/target/release/acc rust_prove /home/xiang/Desktop/ProofCarryExec/Code/zkregex/main/../DATA/anti_virus_output/clamav_5/ ../DATA/case103 /tmp/sigma_002i/ Bls381 20 /tmp/tmp_nodelist.txt 15990
