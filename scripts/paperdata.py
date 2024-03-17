# Creator: Dr. Xiang Fu
# 09/11/2023

# run profile from 1 thread to 2^k threads

import subprocess;
def paperdata(log2_threads):
	subprocess.run(["target/release/izpr", "paperdata", str(log2_threads)]);

# MAIN 
log2_num_threads = 3;
for k in range(log2_num_threads, 2, -1):
	paperdata(2**k);
