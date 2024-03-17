set term postscript eps color size 16in,10in font 'Verdenta,30'
#set term pdf size 7in,3in 
#set terminal wxt size 1300,600
#set term eps size 1300,600
set output 'output/fig_cq.eps' 
#clear
#reset
set autoscale 
set multiplot layout 2, 2 title ""

#1. cq
set title "cq"
set key bottom right 
set border 3
#set output "output/fig_fft.tex"
set xlabel 'Query Table Size'
set ylabel 'Time (ms)' rotate by 90
ntics = 2
set yzeroaxis
set logscale x 2; 
set logscale y 2; 
set format x '2^{%L}'
set format y '2^{%L}'
set grid

plot "raw_data/cq.dat" using 1:($2) title '2-threads' with linespoints,\
	"raw_data/cq.dat" using 1:($3) title '4-threads' with linespoints,\
	 "raw_data/cq.dat" using 1:($4) title '8-threads' with linespoints,\
	 "raw_data/cq.dat" using 1:($5) title '16-threads' with linespoints,\
	 "raw_data/cq.dat" using 1:($6) title '32-threads' with linespoints,\

#2. range 
set title "zk-range"
set key left 
set border 3
set xlabel 'Number of Elements'
set ylabel 'Time (ms)' rotate by 90
ntics = 1
set yzeroaxis
set logscale x 2; 
set logscale y 2; 
set format x '2^{%L}'
set format y '2^{%L}'
set grid

plot "raw_data/range.dat" using 1:($2)   notitle with linespoints,\
	 "raw_data/range.dat" using 1:($3)   notitle with linespoints,\
	 "raw_data/range.dat" using 1:($4)   notitle with linespoints,\
	 "raw_data/range.dat" using 1:($5)   notitle with linespoints,\
	 "raw_data/range.dat" using 1:($6)   notitle with linespoints

#2. pn-looup 
set title "pn-lookup"
set key left 
set border 3
set xlabel 'Query Table Size'
set ylabel 'Time (ms)' rotate by 90
ntics = 1
set yzeroaxis
set logscale x 2; 
set logscale y 2; 
set format x '2^{%L}'
set format y '2^{%L}'
set grid

plot "raw_data/pn_lookup.dat" using 1:($2)   notitle with linespoints,\
	 "raw_data/pn_lookup.dat" using 1:($3)   notitle with linespoints,\
	 "raw_data/pn_lookup.dat" using 1:($4)   notitle with linespoints,\
	 "raw_data/pn_lookup.dat" using 1:($5)   notitle with linespoints,\
	 "raw_data/pn_lookup.dat" using 1:($6)   notitle with linespoints

#2. asset1 
set title "IZPR"
set key left 
set border 3
set xlabel 'Platform Throughput'
set ylabel 'Time (ms)' rotate by 90
ntics = 1
set yzeroaxis
set logscale x 2; 
set logscale y 2; 
set format x '2^{%L}'
set format y '2^{%L}'
set grid

plot "raw_data/asset1.dat" using 1:($2)   notitle with linespoints,\
	 "raw_data/asset1.dat" using 1:($3)   notitle with linespoints,\
	 "raw_data/asset1.dat" using 1:($4)   notitle with linespoints,\
	 "raw_data/asset1.dat" using 1:($5)   notitle with linespoints,\
	 "raw_data/asset1.dat" using 1:($6)   notitle with linespoints


unset multiplot
