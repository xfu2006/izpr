/** 
	Copyright Dr. Xiang Fu

	Author: Dr. Xiang Fu
	All Rights Reserved.
	Created: 07/14/2022
	Completed: 07/16/2022
*/

/**! This is the profiler function.
Modify the get_all_protocols function in proto/proto_tests.rs
when new zk protocols are added!	
*/
extern crate ark_poly;
extern crate ark_ec;
extern crate ark_ff;
extern crate mpi;
extern crate ark_bn254;
extern crate ark_bls12_381;

use std::borrow::Borrow;
use std::rc::Rc;
use crate::profiler::config::*;
use crate::proto::proto_tests::*;
use crate::proto::*;
use crate::tools::*;
use crate::poly::dis_key::*;
//use self::mpi::traits::*;
use self::ark_ec::{PairingEngine};

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

/// profile a protocol (n is the desired test case size)
/// it CANNOT be larger than the prover key of the included instance
fn profile_proto<E:PairingEngine>(inst: Box<dyn Protocol<E>>, 
	key: Rc<DisKey<E>>, n: usize){
	phantom_func();
	if n>get_max_test_size_for_key(&key) {panic!("profile proto err: n: {} >key size: {}", n, get_max_test_size_for_key(&key));}
	let seed = 177737;
	let (proto,mut inp,claim, _prf)= inst.rand_inst(n,seed,false, key.clone()); 
	{
	//1. generate a random instance
	let mut t1 = Timer::new();
	let mut prf = proto.prove(&mut *inp);
	t1.stop();

	let mut t2 = Timer::new();
	t2.start();
	let _bres = proto.verify(claim.borrow(), prf.borrow());
	t2.stop();

	let mut t3 = Timer::new();
	t3.start();
	let bytes = prf.to_bytes();
	let prf_size = bytes.len();
	t3.stop();

	let mut t4 = Timer::new();
	t4.start();
	prf.from_bytes(&bytes);
	t4.stop();

	if RUN_CONFIG.my_rank==0{
		println!("Protocol {}, size: {}, prover: {}ms, verifier: {} us, prf_size: {} bytes, tobytes time: {} us, frombytes time: {} us", inst.name(), n, t1.time_us/1000, t2.time_us, prf_size, t3.time_us, t4.time_us);
	}
	}	


	//RUN_CONFIG.better_barrier("profile_proto");

}

pub fn profile_all_protos(size: usize, _cfg: &RunConfig){
	let key_size = get_adjusted_key_size(size*2+32);  
	let (arr_proto, key) = get_all_protocols::<PE>(key_size);
	for p in arr_proto{
		profile_proto(p, key.clone(), size);
	}
}
