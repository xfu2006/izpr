/*
    Copyright Dr. Xiang Fu

    Author: Xiang Fu
    All Rights Reserved.
    Created: 08/03/2023

	This files encodes the zero knowledge version of the cq protocol.
	See appendix H.4 of our paper IZPR.
*/
extern crate rayon;
extern crate acc;
extern crate ark_ec;
extern crate ark_ff;
extern crate ark_serialize;

use self::rayon::prelude::*; 
use self::acc::poly::serial::serial_fft;
use self::acc::profiler::config::{LOG1};
use self::acc::tools::{log_perf,Timer,rand_arr_field_ele};
use self::ark_ff::{One};
use self::ark_ec::{AffineCurve, PairingEngine,ProjectiveCurve};
use self::ark_ec::msm::{VariableBaseMSM};
use izpr::serial_poly_utils::{KzgTrapDoor, setup_kzg, default_trapdoor};
use izpr::serial_group_fft2::{serial_group_fft};
use izpr::serial_cq::{prove_cq, slow_verify_cq, verify_cq, preprocess_cq, gen_seed};
use izpr::serial_rng::{setup_rng, prove_rng, verify_rng,produce_rng_commits,rand_arr_fe, rand_arr_fe_with_seed};
use izpr::serial_pn_lookup::{setup_pn_lookup, prove_pn_lookup, verify_pn_lookup};
use izpr::serial_asset1_v2::{setup_asset_one, prove_asset_one, 
verify_asset_one, AssetOneProof}; 
use self::ark_serialize::{CanonicalSerialize, CanonicalDeserialize};

/// report the operation
pub fn report(op: &str, size: usize, timer: &Timer){
	let num_th = rayon::current_num_threads(); 
	let time_us = timer.get_time();
	println!("REPORT_op: {}, size: {}, time: {} ms, np: {}", op, size, time_us/1000, num_th);
}

/// profile fft and group fft
pub fn profile_fft<PE:PairingEngine>(n: usize)  where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField>{
	let mut arr_f = rand_arr_field_ele::<PE::Fr>(n, 73123u128);
	let mut t = Timer::new();
	let g1 = PE::G1Affine::prime_subgroup_generator();
	let mut arr_g = arr_f.clone().into_par_iter()
		.map(|x| g1.mul(x).into_affine())
		.collect::<Vec<PE::G1Affine>>();
	log_perf(LOG1, &format!("gen_rand_group_arr: {}", n), &mut t);
	serial_fft(&mut arr_f);
	log_perf(LOG1, &format!("FFT field: {}", n), &mut t);
	serial_group_fft(&mut arr_g);
	log_perf(LOG1, &format!("FFT Group: {}", n), &mut t);
	
}


/// profile basic operations such as FFT and key set up
pub fn profile_basic_ops<PE:PairingEngine>(n: usize)  where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField>{
	let mut arr_f = rand_arr_field_ele::<PE::Fr>(n*128, 73123u128);
	let s = PE::Fr::from(311231237u64);
	let ktrap = KzgTrapDoor{s: s};
	let mut t = Timer::new();
	setup_kzg::<PE>(n, n/4, &ktrap);
	log_perf(LOG1, &format!("gen_kzg_key: {}", n), &mut t);
	serial_fft(&mut arr_f);
	log_perf(LOG1, &format!("fft: {}", n), &mut t);
}

/// profile cq
pub fn profile_cq<PE:PairingEngine>(n: usize, n2: usize)  where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField>{
	let trapdoor = default_trapdoor::<PE>();
	let seed = gen_seed();
	let lookup_table = rand_arr_field_ele::<PE::Fr>(n, seed); 
	let mut query_table = vec![];
	for i in 0..n2{query_table.push(lookup_table[i*371 % n]); }
	let mut t = Timer::new();
	let (pkey, vkey) = setup_kzg::<PE>(n, query_table.len(), &trapdoor);
	log_perf(LOG1, &format!("gen_kzg_key: {}", n), &mut t);
	let cq_aux = preprocess_cq::<PE>(&pkey, &lookup_table);
	log_perf(LOG1, &format!("preprocess: {}", n), &mut t);
	let (prf, _r_query_table, commit_query_table) = prove_cq(&pkey, 
		&cq_aux, &lookup_table, &query_table);
	report("cq", n2, &t); 
	log_perf(LOG1, &format!("prove_cq: {}", n), &mut t);
	let bres2 = slow_verify_cq(&vkey, cq_aux.commit_t2, commit_query_table, &prf);
	log_perf(LOG1, &format!("slow_verify_cq {}", n), &mut t);
	let bres = verify_cq(&vkey, cq_aux.commit_t2, commit_query_table, &prf);
	log_perf(LOG1, &format!("fast_verify_cq{}", n), &mut t);
	assert!(bres && bres2, "cq failed"); 
}

/// profile zk-range 
/// log_bound: element bound [2, 2^log_bound)
/// log_unit_bound: e.g., 20 bits units for 160 bits element
/// num_eleents: number of elements
pub fn profile_zk_rng<PE:PairingEngine>(
	log_bound: usize,
	log_unit_bound: usize,
	num_ele: usize
)  where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField>{
	let mut t = Timer::new();
	let (bits, unit_bits) = (log_bound, log_unit_bound);
	let t1 = rand_arr_fe::<PE::Fr>(num_ele, bits);
	let t2 = rand_arr_fe::<PE::Fr>(num_ele, bits);
	let arr_r = rand_arr_fe::<PE::Fr>(3, bits);
	log_perf(LOG1, &format!("gen_rand_eles: {}", num_ele), &mut t);

	let (pk, vk) = setup_rng::<PE>(bits, unit_bits, num_ele);
	log_perf(LOG1, &format!("setup range proof keys: {}", num_ele), &mut t);

	let (com_t1, com_t2) = produce_rng_commits::<PE>(&pk, 
			&t1, arr_r[0], &t2, arr_r[1], arr_r[2]); 
	log_perf(LOG1, &format!("produce two commits: {}", num_ele), &mut t);

	let prf = prove_rng(&pk, &t1, arr_r[0], &t2, arr_r[1]);
	report("range", num_ele, &t); 
	log_perf(LOG1, &format!("prove: {}", num_ele), &mut t);

	let bres = verify_rng(&vk, com_t1, com_t2, &prf);
	log_perf(LOG1, &format!("verify: {}", num_ele), &mut t);
	assert!(bres, "range proof failed");	
}

pub fn profile_pn_lookup<PE:PairingEngine>(
	log_bound: usize,
	log_unit_bound: usize,
	lookup_size: usize,
	query_table_size: usize
)  where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField>{
		assert!(lookup_size.is_power_of_two(), "lookup size not power of 2");
		assert!(query_table_size.is_power_of_two(), "query size not pow of 2");
		assert!(lookup_size>query_table_size, "lookup should be >2 times of query table");
		println!("Profile PN Lookup: look size: {}, query size: {}",
			lookup_size, query_table_size);
		let (bits, unit_bits, n2)=(log_bound, log_unit_bound, query_table_size); 
		println!("---- Profile PN-Lookup ----");
		let mut timer = Timer::new();
		let mut t = rand_arr_fe::<PE::Fr>(lookup_size-1, bits);
		t.sort();
		let mut query_table = rand_arr_fe_with_seed::<PE::Fr>(n2, bits, 
			gen_seed()+1);
		for i in 0..query_table.len(){ query_table[i] = t[i*2]; }
		log_perf(LOG1, "-- gen random query tables", &mut timer);

		let (pk, vk) = setup_pn_lookup::<PE>(bits, 
			unit_bits, lookup_size, query_table_size, &t);
		log_perf(LOG1, "-- setup pnlookup", &mut timer);
		let arr_r = rand_arr_fe::<PE::Fr>(3, bits);
		let (com_t, com_o, prf, _) = prove_pn_lookup(&pk, &query_table, 
			arr_r[0], arr_r[1]);
		report("pn_lookup", n2, &timer); 
		log_perf(LOG1, "-- prove pnlookup",  &mut timer);
		let bres = verify_pn_lookup(&vk, com_t, com_o, &prf);
		log_perf(LOG1, "-- verify pnlookup", &mut timer);
		assert!(bres, "pn_lookup failed");	
}

pub fn profile_asset1<PE:PairingEngine>(
	log_bound: usize,
	log_unit_bound: usize,
	lookup_size: usize,
	query_table_size: usize
)  where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField>{
	let bits = log_bound; 
	let unit_bits = log_unit_bound;
	assert!(bits%unit_bits==0, "bits%unit_bits!=0!");
	let num_ele = lookup_size;
	let n2 = query_table_size;
	let mut timer = Timer::new();

	println!("===== Profile Asset Protocol: bits: {}, unit_bits: {}, lookup_size: {}, query_size: {}", bits, unit_bits, num_ele, n2); 
	let mut t = rand_arr_fe::<PE::Fr>(num_ele-1, bits);
	t.sort();
	let mut query_table = rand_arr_fe_with_seed::<PE::Fr>(n2-1, 
		bits, gen_seed()+1);
	query_table[1] = t[3];
	let one = PE::Fr::one();
	query_table.append(&mut vec![one]);
	log_perf(LOG1, "-- generate random data", &mut timer);
	let (pk, vk) = setup_asset_one::<PE>(bits,unit_bits, num_ele, n2, &t);
	let arr_r = rand_arr_fe::<PE::Fr>(3, bits);
	let r_total = arr_r[0];
	let mut arr_changes = rand_arr_fe::<PE::Fr>(n2-1, bits);
	arr_changes.append(&mut vec![one]);
	log_perf(LOG1, "-- set up keys", &mut timer);
	let (com_v, com_acc, com_ch, prf) = prove_asset_one(&pk, &query_table, 
		&arr_changes, r_total);
	report("asset1", n2, &timer); 
	log_perf(LOG1, "-- prove work", &mut timer);
	let bres = verify_asset_one(&vk, com_acc, com_ch,
		com_v, &prf);
	assert!(bres, "asset1 failed");	
	log_perf(LOG1, "-- verifier work", &mut timer);
	let mut bs: Vec<u8> = vec![];
	AssetOneProof::<PE>::serialize(&prf, &mut bs);
	let prf_size = bs.len();
	println!("REPORT_op: proof_size, size: {}", prf_size);
}
