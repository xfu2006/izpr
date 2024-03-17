/** 
	Copyright Dr. Xiang Fu

	Author: Dr. Xiang Fu
	All Rights Reserved.
	Created: 07/27/2022
*/

/// This is the profiler function for serial and distributed R1CS

extern crate ark_poly;
extern crate ark_ec;
extern crate ark_ff;
extern crate mpi;
extern crate ark_bn254;
extern crate ark_bls12_381;

/*
use std::borrow::Borrow;
use std::rc::Rc;
use crate::proto::proto_tests::*;
use crate::proto::*;
use crate::poly::dis_key::*;
//use self::mpi::traits::*;
use self::ark_ec::{PairingEngine};
*/
use crate::tools::*;
use crate::poly::common::*;
use crate::poly::dis_vec::*;
use crate::r1cs::dis_r1cs::*;
use crate::r1cs::serial_r1cs::*;
use self::ark_ff::UniformRand;
use crate::profiler::config::*;

use std::marker::PhantomData;
use self::ark_bn254::Bn254;
type Fr = ark_bn254::Fr;
type PE= Bn254;
//use self::ark_bls12_381::Bls12_381;
//type Fr = ark_bls12_381::Fr;
//type PE= Bls12_381;

/// just to remove some warning
fn phantom_func(){
	let _d1: PhantomData<Fr>;
	let _d2: PhantomData<PE>;
}

/// profile to serial and from serial
pub fn profile_r1cs_serialization(n: usize){
	phantom_func();
	println!("DEBUG USE 101: generating rand_inst");
	let (r1cs,_) = R1CS::<Fr>::rand_inst(1122u128, n, n, false);
	let mut t1 = Timer::new();
	let mut t2 = Timer::new();
	t1.start();
	println!("DEBUG USE 102: from_serial"); 
	let dr1cs = DisR1CS::<Fr>::from_serial(&r1cs);
	t1.stop();
	t2.start();
	println!("DEBUG USE 103: to_serial"); 
	let _r1cs2 = dr1cs.to_serial();
	t2.stop();
	println!("R1CS size: {}, from_serial: {} us, to_serial: {} us",
		n, t1.time_us, t2.time_us);
}

/// profile to matrix to qap 
pub fn profile_matrix_to_qap(n: usize){
	if RUN_CONFIG.my_rank!=0 { 
		RUN_CONFIG.better_barrier("profile_matrix_to_qap");
		return;
	}
	let mut rng = gen_rng(); 
	let matrix = rand_matrix(n, n, 3, &mut rng);
	let t = Fr::rand(&mut rng);
	let mut timer2 = Timer::new();
	timer2.start();
	let _v2 = matrix_to_qap_poly_eval(&matrix, n, n, t);
	timer2.stop();
	println!("matrix to qap: size: {}, fast (Lagrange): {} us", n,  timer2.time_us);
	RUN_CONFIG.better_barrier("profile_matrix_to_qap");
}

/// profile to compute h() withness 
/// Performance: 9.2 seconds
pub fn profile_compute_witness_h(n: usize){
	if RUN_CONFIG.my_rank!=0 { 
		RUN_CONFIG.better_barrier("profile_matrix_to_qap");
		return;
	}
	//let rng = gen_rng(); 
	let (r1cs,vars) = R1CS::<Fr>::rand_inst(1122u128, n, n, true);
	let n = closest_pow2(r1cs.a.len()); 
	let mut timer1 = Timer::new();
	timer1.start();
	let _h1 = slow_compute_witness_h(&r1cs.a, &r1cs.b, &r1cs.c, &vars);
	timer1.stop();
	println!("SLOW compute h(): size: {}, {} us", n,  timer1.time_us);
	let mut timer2 = Timer::new();
	timer2.start();
	let _h2 = compute_witness_h(&r1cs.a, &r1cs.b, &r1cs.c, &vars);
	timer2.stop();
	println!("IFFT compute h(): size: {}, {} us", n,  timer2.time_us);
	RUN_CONFIG.better_barrier("profile_compute_h()");
}

pub fn profile_make_even(n: usize){
	let seed = 17773171u128;
	let new_n = closest_pow2(2*n);
	let (mut dis_r1cs, _dis_vars) = DisR1CS::<Fr>::rand_inst(seed, n, n, false);
	let mut timer = Timer::new();
	timer.start();
	dis_r1cs.make_even(new_n);
	timer.stop();
	if RUN_CONFIG.my_rank == 0{
		println!("DisR1CS size: {}, make_even time: {} us", n, timer.time_us);
	}	
}

pub fn profile_dis_eval_matrix(n: usize){
	let seed = 17773171u128;
	let new_n = closest_pow2(n);
	let (mut dr1cs, mut dis_vars) = DisR1CS::<Fr>::rand_inst(seed, n, n, false);
	dr1cs.make_even(new_n);
	dis_vars.to_partitions(&RUN_CONFIG.univ);
	let mut timer = Timer::new();
	timer.start();
	let _dvec2 = dis_eval_matrix::<Fr>(&dr1cs.a_share, 
			dr1cs.num_vars, dr1cs.num_constraints, &dis_vars);
	timer.stop();
	if RUN_CONFIG.my_rank == 0{
		println!("DisR1CS size: {}, eval_matrix time: {} us", n, timer.time_us);
	}	
}

/// Performance: 1k: 4ms, 32k: 73ms, 1M: 3.2s
pub fn profile_dis_compute_h(n: usize){
	let n = closest_pow2(n);
	let (r1cs,vars) = R1CS::<Fr>::rand_inst(1122u128, n, n, true);
	let dr1cs = DisR1CS::from_serial(&r1cs);
	let mut dvars = DisVec::new_dis_vec_with_id(0, 0, vars.len(), vars.clone());
	dvars.to_partitions(&RUN_CONFIG.univ);
	let mut timer = Timer::new();
	timer.start();
	let _dvec_h = dis_compute_witness_h(n, n, &dr1cs.a_share, &dr1cs.b_share, &dr1cs.c_share, &dvars);
	timer.stop();
	if RUN_CONFIG.my_rank == 0{
		println!("DisR1CS size: {}, dis_compute_h time: {} us", n, timer.time_us);
	}	
	timer.clear_start();
	let _dvec_h2 = DisR1CS::<Fr>::dis_compute_witness_h_2(n, n, &dr1cs.a_share, &dr1cs.b_share, &dr1cs.c_share, &dvars);
	timer.stop();
	if RUN_CONFIG.my_rank == 0{
		println!("ExchangeVar version compute h2(): size: {}, {} us", n,  timer.time_us);
	}	
}

/// Performance: 1M: 0.8sec (almost linear)
pub fn profile_dis_gen_r1cs(n: usize){
	let n = closest_pow2(n);
	let mut timer = Timer::new();
	timer.start();
	let (_dr1cs,_dvars) = DisR1CS::<Fr>::rand_inst(1122u128, n, n, true);
	timer.stop();
	if RUN_CONFIG.my_rank == 0{
		println!("DisR1CS size: {}, gen time: {} us", n, timer.time_us);
	}	
}

/// Performance: 1M -> 6.2 sec  
pub fn profile_dis_to_qap(n: usize){
	let n = closest_pow2(n);
	let mut timer = Timer::new();
	let mut t2= Timer::new();
	let (dr1cs,dvars) = DisR1CS::<Fr>::rand_inst(1122u128, n, n, true);
	timer.start();
	let t = Fr::from(1322342u64);
	let _d_qap = dr1cs.to_qap(&t);
	timer.stop();
	t2.start();
	let _d_wit = dr1cs.to_qap_witness(dvars);
	t2.stop();
	if RUN_CONFIG.my_rank == 0{
		println!("DisR1CS size: {}, to_qap time: {} us, to_qap_witness: {} us", n, timer.time_us, t2.time_us);
	}	
}
