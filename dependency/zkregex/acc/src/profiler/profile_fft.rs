/** 
	Copyright Dr. Xiang Fu

	Author: Dr. Xiang Fu
	All Rights Reserved.
	Created: 05/12/2022
	This represent the data structure of the running environment
*/
extern crate ark_ff;
extern crate ark_ec;
extern crate ark_poly;
extern crate ark_std;
extern crate mpi;
extern crate ark_serialize;

use profiler::config::*;
use profiler::report;
use super::config::RunConfig;
use super::super::tools::*;
use super::super::poly::disfft::*;
use super::super::poly::dis_vec::*;
use super::super::poly::dis_poly::*;
use super::super::poly::serial::*;
//use super::super::poly::common::*;
use self::ark_ec::{PairingEngine};
//use self::mpi::collective::CommunicatorCollectives;
use self::mpi::topology::Communicator;
type PE=ark_bls12_381::Bls12_381;
type Fr=<PE as PairingEngine>::Fr;
//type PE=ark_bn254::Bn254;
//type Fr=ark_bn254::Fr;

/** profile fft and ifft performance */
pub fn profile_serial_fft(size: usize, cfg: &RunConfig){
  if cfg.univ.world().rank()==0{
	println!("profileFFT: size: {}, cfg: {}", size, cfg);
	let mut t = Timer::new();
	let mut t2 = Timer::new();
	let mut vec = rand_arr_field_ele::<Fr>(size, get_time());
	t.start();
	serial_fft(&mut vec);
	t.stop();
	println!("profile Serial FFT time: {} ms", t.time_us/1000);
	t2.start();
	serial_ifft(&mut vec);
	t2.stop();
	println!("profile Serial IFFT time: {} ms", t.time_us/1000);
  }
}

pub fn profile_serial_dizk_fft(size: usize, cfg: &RunConfig){
  if cfg.univ.world().rank()==0{
	println!("profileSerialDizkFFT: size: {}, cfg: {}", size, cfg);
	let mut t = Timer::new();
	let mut vec = rand_arr_field_ele::<Fr>(size, get_time());
	t.start();
	serial_dizk_fft(&mut vec);
	t.stop();
	println!("profileSerialDizkFFT time: {} ms", t.time_us/1000);
  }
}

pub fn profile_dis_vec(size: usize, cfg: &RunConfig){
	let mut t = Timer::new();
	let v1 = rand_arr_field_ele::<Fr>(size, get_time());
	let mut dv1 = DisVec::new_dis_vec_with_id(0, 0, v1.len(), v1);
	t.start();
	dv1.to_partitions(&&cfg.univ);
	t.stop();
	if cfg.univ.world().rank()==0{
		println!("profileDisVec : size: {}, cfg: {}", size, cfg);
		println!("profileDisVec time: {} ms", t.time_us/1000);
	}
}

pub fn profile_dis_dizk_fft(size: usize, cfg: &RunConfig){
	let mut t = Timer::new();
	let my_rank = cfg.univ.world().rank() as u64;
	let main_rank = 0u64; //ONLY PROCESS the dv1 generated for main rank
	let vecid = 101u64;
	//let univ = &cfg.univ;
	if my_rank==main_rank{ 
 			let vec = rand_arr_field_ele::<Fr>(size, get_time());
		let mut vec2 = vec.clone();
		serial_dizk_fft(&mut vec2);
		let mut dv1 = DisVec::new_dis_vec_with_id(vecid, main_rank,vec.len(), vec);
		dv1.to_partitions(&cfg.univ);
		t.start();
		distributed_dizk_fft(&mut dv1, &cfg.univ);
		RUN_CONFIG.better_barrier("profile_dis_dizk_fft");
		t.stop();
		println!("profileDisDistFFT: size: {}, cfg: {}", size, cfg);
		println!("profileDistDizkFFT time: {} ms", t.time_us/1000);
	}else{
		let mut dv1:DisVec<Fr> = DisVec::new_dis_vec_with_id(vecid, main_rank, size, vec![]); //just pass the vecid, content is fake
		dv1.to_partitions(&cfg.univ);
		distributed_dizk_fft(&mut dv1, &cfg.univ);
		RUN_CONFIG.better_barrier("gen_circ_multi_set2");
	}
}

//do a distirbuted-fft of 1M size to establish TCP connections
pub fn warm_up(){
	let cfg = &RUN_CONFIG;
	let seed = 234243234324u128;
	let p = DisPoly::<Fr>::gen_dp_from_seed(1024*1024, seed);
	let mut dv1= p.dvec;	
	distributed_dizk_fft(&mut dv1, &cfg.univ);
}

pub fn data_profile_fft(size: usize){
	let cfg = &RUN_CONFIG;
	let mut t = Timer::new();
	let seed = 234243234324u128;
	let p = DisPoly::<Fr>::gen_dp_from_seed(size, seed);
	let mut dv1= p.dvec;	
	t.start();
	distributed_dizk_fft(&mut dv1, &cfg.univ);
	t.stop();
	if RUN_CONFIG.my_rank==0{
		report("fft", size, t.time_us/1000);
	}
}

/// do the fft data for trials from log_min_size to log_max_size
/// timeout in seconds 
pub fn data_fft(log_min_size: usize, log_max_size: usize, log_size_step: usize, trials: usize, timeout: usize){
	let me = RUN_CONFIG.my_rank;
	if me==0{
		println!("===== DATA for fft: log_min_size: {} -> log_max_size: {}", 
			log_min_size, log_max_size);
	}
	let cfg = &RUN_CONFIG;
	let mut t = Timer::new();
	let seed = 234243234324u128;
	let min_size = 1<<log_min_size;
	let max_size = 1<<log_max_size;
	let mut size = min_size;
	let step = 1<<log_size_step;
	while size<=max_size{
		let p = DisPoly::<Fr>::gen_dp_from_seed(size, seed);
		let mut dv1= p.dvec;	
		let mut min_time = 9999999999999999;
		if me==0 {println!("---- size: {} -----", size);}
		for _i in 0..trials{ 
			t.clear_start();
			distributed_dizk_fft(&mut dv1, &cfg.univ);
			t.stop();
			let time_ms = t.time_us/1000;
			if time_ms<min_time {min_time = time_ms;}
			report("fft", size, t.time_us/1000);
		}
		report("fft_min", size, min_time);
		if min_time>timeout*1000 {
			if RUN_CONFIG.my_rank==0{
				println!("TIMEOUT for op: fft size: {}", size);
			}
			break;
		}
		size *= step;
	}
	
}
