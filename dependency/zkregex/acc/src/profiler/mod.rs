/** 
	Copyright Dr. Xiang Fu

	Author: Dr. Xiang Fu
	All Rights Reserved.
	Created: 05/12/2022
*/

/*
pub mod profile_fft;
pub mod profile_poly;
pub mod profile_proto;
pub mod profile_pairing;
pub mod profile_r1cs;
pub mod profile_groth16;
pub mod profile_group;
*/
pub mod config;

use profiler::config::*;

/// write a report line string: look like
/// op: mul, size: 1000, time: 1024, np: 8 
pub fn report(opname: &str, n: usize, time_ms: usize){
	if RUN_CONFIG.my_rank==0{
		println!("REPORT_op: {}, size: {}, time: {} ms, np: {}", opname, n, time_ms, RUN_CONFIG.n_proc); 
	}
}
pub fn single_report(opname: &str, n: usize, time_ms: usize){
		println!("REPORT_op: {}, size: {}, time: {} ms", opname, n, time_ms);
}
