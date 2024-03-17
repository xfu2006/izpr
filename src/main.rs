/** 
    Copyright Dr. Xiang Fu

    Author: Dr. Xiang Fu, Trevor Conley, Diego ESpada, Alvine, Nilso, ...
    All Rights Reserved.
    Created: 07/28/2023
    Revised: ...

    This is the main entry of the project.
*/

extern crate ark_ec;
extern crate izpr;
extern crate ark_bls12_381;
extern crate rayon;
extern crate ark_std;


//use self::ark_std::cmp::max;
//use rayon::prelude::*;


use std::env;
use self::ark_bls12_381::Bls12_381;
//use self::ark_ec::{AffineCurve};
use self::ark_ec::{AffineCurve};
use izpr::izpr::serial_profiler::{
//	profile_basic_ops, 
	profile_cq,
	profile_zk_rng,
	profile_pn_lookup,
//	profile_fft,
	profile_asset1,
};
use izpr::izpr::serial_poly_utils::{vmsm, setup_kzg, KzgTrapDoor};

type Fr381=ark_bls12_381::Fr;
type PE381=Bls12_381;

fn debug(){
	println!("debugging ...");
	let trapdoor = KzgTrapDoor{s: Fr381::from(171u32)};
	let (pk, vk) = setup_kzg::<PE381>(128, 32, &trapdoor);			
	let mut arr_e = vec![];
	let mut sum = Fr381::from(0u64);
	let mut exp = Fr381::from(1u64);
	for i in 0..130{
		arr_e.push(Fr381::from(i as u64));
		sum += exp * Fr381::from(i as u64);
		exp = exp * trapdoor.s;
	}
	let res = vmsm(&pk.vec_g1, &arr_e);
	let res2 = vk.g1.mul(sum);
	println!("res: {}\nres2: {}\n", res, res2);
}

fn run(){
	println!("running ....");
}

fn profile(num_threads: usize){
	rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().unwrap();
	let num_avai= rayon::current_num_threads();
	let (bits, unit_bits) = (32, 8);
	let k = bits/unit_bits;
	let query_size = 16;
	let lookup_size = 4*query_size;
	//let fft_size = 1024 * 16; 
	assert!(2<<unit_bits >= 2 * k * query_size, "Reset config so that: 2<<bits > 2*k*query_size, bits: {}, k: {}, query_size: {}", unit_bits, k, query_size);
	
	println!("profiling ... with threads: {}, lookup_size: {}, query_size: {}", num_avai, lookup_size, query_size);
//	profile_fft::<PE381>(fft_size);
//	profile_basic_ops::<PE381>(1024);
//	profile_cq::<PE381>(1024*2, 1024);
//	profile_zk_rng::<PE381>(64, 16, 1024);
//	profile_pn_lookup::<PE381>(bits, unit_bits, lookup_size, query_size);
	profile_asset1::<PE381>(bits, unit_bits, lookup_size, query_size);
}

fn paper_data(num_threads: usize){
	rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().unwrap();
	let num_avai= rayon::current_num_threads();
	let (bits, unit_bits) = (32, 16);
	let k = bits/unit_bits;
	let mut query_size = 1024;
	let times = 1;
	assert!(2<<unit_bits >= 2 * k * query_size, "Reset config so that: 2<<bits > 2*k*query_size, bits: {}, k: {}, query_size: {}", unit_bits, k, query_size);

	for _i in 1..times+1{
        let lookup_size = query_size * 4;
        println!("Paper Data ... with threads: {}, lookup_size: {}, query_size: {}", num_avai, lookup_size, query_size);
        //profile_cq::<PE381>(lookup_size, query_size);
        //profile_zk_rng::<PE381>(bits, unit_bits, query_size);
        //profile_pn_lookup::<PE381>(bits, unit_bits, lookup_size, query_size);
        profile_asset1::<PE381>(bits, unit_bits, lookup_size, query_size);
        query_size *= 2;
    }

}

fn main(){
	let args: Vec<String> = env::args().collect();
	if args[1]=="debug"{
		debug();
	}else if args[1]=="profile"{
		let num_threads = args[2].parse::<usize>().unwrap();
		profile(num_threads);
	}else if args[1]=="paperdata"{
		let num_threads = args[2].parse::<usize>().unwrap();
		paper_data(num_threads);
	}else if args[1]=="run"{
		run();
	}else{
		panic!("UNKNOWN arg: {}", args[1]);
	}
}

