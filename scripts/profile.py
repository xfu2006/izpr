# Creator: Dr. Xiang Fu
# 08/07/2023

# run profile from 1 thread to 2^k threads

import subprocess;
def profile(log2_threads):
	subprocess.run(["target/release/izpr", "profile", str(log2_threads)]);

# MAIN 
log2_num_threads = 3;
for k in range(log2_num_threads + 1):
	profile(2**k);
