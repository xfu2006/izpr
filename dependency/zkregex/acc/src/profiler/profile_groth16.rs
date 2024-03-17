/** 
	Copyright Dr. Xiang Fu

	Author: Dr. Xiang Fu
	All Rights Reserved.
	Created: 10/17/2022
*/

/// This is the profiler function for serial and distributed QAP based
/// 2-stage Groth'16 scheme 

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
//use crate::poly::common::*;
//use crate::poly::dis_vec::*;
use crate::poly::dis_key::*;
//use crate::r1cs::dis_r1cs::*;
//use crate::r1cs::serial_r1cs::*;
use crate::groth16::serial_qap::*;
use crate::groth16::new_dis_qap::*;
use crate::groth16::serial_prover::*;
use crate::groth16::dis_prover::*;
use crate::groth16::verifier::*;
use crate::groth16::serial_prove_key::*;
use crate::groth16::dis_prove_key::*;
//use self::ark_ff::UniformRand;
use crate::profiler::config::*;

use std::marker::PhantomData;
use self::ark_bn254::Bn254;
type Fr = ark_bn254::Fr;
type PE= Bn254;
//use self::ark_bls12_381::Bls12_381;
//type Fr = ark_bls12_381::Fr;
//type PE= Bls12_381;

/// return the minimum size required for the cluster size
pub fn get_min_test_size()->usize{
	let np = RUN_CONFIG.n_proc as usize;
	return np*2;
}

/// just to remove some warning
fn phantom_func(){
	let _d1: PhantomData<Fr>;
	let _d2: PhantomData<PE>;
}

/// profile to serial and from serial
pub fn profile_groth16(inp_n: usize){
	phantom_func();
	let mut t1 = Timer::new();
	let mut t2 = Timer::new();
	let mut t3 = Timer::new();
	let mut n = get_min_test_size().next_power_of_two()*4;
	n = if n<inp_n {inp_n.next_power_of_two()} else {n};
	let degree = n - 2; //degree+2 must be power of 2	
	let num_inputs = 2;
	let num_vars = n;
	let seed = 1122u128;
	let (qap, qw) = QAP::<Fr>::rand_inst(seed, num_inputs, num_vars,degree, true);
	let num_segs = qap.num_segs;
	t1.start();
	let prover = SerialProver::<PE>::new(num_segs,11123123u128);
	let diskey = DisKey::<PE>::gen_key1(get_min_test_size()); 	
	let (skey, vkey) = serial_setup(198123123u128, &qap, &diskey);
	t1.stop();
	t2.start();
	let p1 = prover.prove_stage1(&skey, &qw);
	let p2 = prover.prove_stage2(&skey, &qw);
	t2.stop();
	t3.start();
	let bres = verify::<PE>(&p1, &p2, &vkey);
	t3.stop();
	if RUN_CONFIG.my_rank==0{
		assert!(bres==true, "verification failed");
		println!("serial verification passed!");
	};
	println!("Groth16 size: {}, setup time: {} ms, prove time: {}, verify time: {}", n, t1.time_us/1000, t2.time_us/1000, t3.time_us/1000);
}

/// profile distributed serial
pub fn profile_dis_groth16(inp_n: usize){
	phantom_func();
	let mut t1 = Timer::new();
	let mut t2 = Timer::new();
	let mut t3 = Timer::new();
	let mut n = get_min_test_size().next_power_of_two()*4;
	n = if n<inp_n {inp_n.next_power_of_two()} else {n};
	let degree = n - 2; //degree+2 must be power of 2	
	let num_inputs = 2;
	let num_vars = n;
	let seed = 1122u128;
	let (qap, qw) = DisQAP::<Fr>::rand_inst(seed, num_inputs, num_vars,degree, true);
	let num_segs = qap.num_segs;
	t1.start();
	let prover = DisProver::<PE>::new(num_segs,11123123u128, qap.seg_size.clone());
	let diskey = DisKey::<PE>::gen_key1(get_min_test_size()); 	
	let (skey, vkey) = dis_setup(198123123u128, &qap, &diskey);
	t1.stop();
	t2.start();
	let p1 = prover.prove_stage1(&skey, &qw, 1);
	let p2 = prover.prove_stage2(&skey, &qw, 1);
	t2.stop();
	t3.start();
	let bres = verify::<PE>(&p1, &p2, &vkey);
	t3.stop();
	if RUN_CONFIG.my_rank==0{
		assert!(bres==true, "verification failed");
		println!("serial verification passed!");
	};
	println!("Groth16 size: {}, setup time: {} ms, prove time: {}, verify time: {}", n, t1.time_us/1000, t2.time_us/1000, t3.time_us/1000);
}

