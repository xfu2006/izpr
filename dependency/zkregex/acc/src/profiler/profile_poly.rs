/** 
	Copyright Dr. Xiang Fu

	Author: Dr. Xiang Fu
	All Rights Reserved.
	Created: 06/03/2022
	Modified: 08/31/2022 added profiling for sub(), mul() ...
	Modified: 01/03/2023 -> added profiling dis_key functions
	Modified: 01/20/2023 -> added profiling dis_r1cs
	This represent profile operations related to distributed polynomial
*/
extern crate ark_ff;
extern crate ark_ec;
extern crate ark_poly;
extern crate ark_std;
extern crate ark_bls12_381;
extern crate mpi;
extern crate ark_serialize;

use super::config::*;
use self::ark_poly::{DenseUVPolynomial,univariate::DensePolynomial};
use self::ark_ec::{AffineCurve, PairingEngine};
use super::super::tools::*;
use profiler::*;
use super::super::poly::common::*;
//use super::super::poly::disfft::*;
use super::super::poly::dis_vec::*;
use super::super::poly::dis_poly::*;
use super::super::poly::dis_key::*;
use super::super::poly::serial::*;
//use self::ark_bls12_381::Bls12_381;
//use self::mpi::collective::CommunicatorCollectives;
use self::mpi::topology::Communicator;
use poly::disfft::*;
use groth16::dis_prover::*;
use groth16::common::*;
use groth16::verifier::*;
use groth16::dis_prove_key::*;
use groth16::new_dis_qap::*;
use self::ark_serialize::CanonicalSerialize;


type Fr=ark_bls12_381::Fr;
type PE=ark_bls12_381::Bls12_381;
//type PE=ark_bn254::Bn254;
//type Fr=ark_bn254::Fr;

/** profile single_from_vec.  in old_rust_acc: 1000 -18 ms, 2000 -40 ms,
4000 -83 ms, 1M - 31 s, 10M - 548 s
	Note: the default setting is that all other processes are busy waiting-taking 100% cpu. this slows things done.
	1000 -  115ms (6 times more expensive)
	2000 - 225 ms (5 times more)
	4000 - 492 ms (5 times)
	128k - 21 s (still around 5 times slow) -> run with just 1 node (13 s)
	1M - 120 s (this is about 4 times slower than old_rust! why? 
If using --mca mpi_yield_when_idle (not helping)
*/
pub fn profile_serial_build_poly_from_roots(size: usize, cfg: &RunConfig){
  if cfg.univ.world().rank()==0{
	println!("profile serial_poly_from_roots: size: {}, cfg: {}", size, cfg);
	let arru64 = rand_arr_unique_u64(size, 20, get_time());
	println!("arru64 generated");
	let v= arru64_to_arrft::<Fr>(&arru64);
	println!("v generated");
	let mut t = Timer::new();
	t.start();
	let _dp = DisPoly::single_from_vec(0, cfg.n_proc as u64, &v);
	t.stop();
	println!("profile Serial Poly From Roots time: {} ms", t.time_us/1000);
  }
	RUN_CONFIG.better_barrier("profile_serial_build_poly_from_roots");
}


/**
	4 nodes.
	128 k - serial (21 sec) - dist (16 sec)
	1M - serial (200 sec) - dist 144 sec
*/
pub fn profile_dist_build_poly_from_roots(n: usize, cfg: &RunConfig){
	println!("profile distributed : size: {}, cfg: {}", n, cfg);
	let my_rank = RUN_CONFIG.my_rank;
	let fname = format!("/tmp/t_{}.dat", n);
	let main_processor = 0;
	let log_2_round_size = 1;
	let dist_bar = 1<<15;

	if my_rank==main_processor as usize{
		//1. generate a random u64 array and write to tmp file
		let mut arru64 = rand_arr_unique_u64(n, 20, get_time());
		let mut a1 = vec![arru64.len() as u64];
		//let arr_f = arru64_to_arrft::<Fr>(&arru64);
		a1.append(&mut arru64);
		write_arr(&a1, &fname);
		RUN_CONFIG.better_barrier("profile_dist_build_poly_from_roots1");

		//3. construct discrete one 
		let mut t = Timer::new();
		t.start();
		let _d_p = DisPoly::<Fr>::dispoly_from_roots_in_file_worker(101, main_processor, &fname, log_2_round_size, dist_bar);
		t.stop();
		report("PolyFromRoot", n, t.time_us/1000);
		println!("profile Dist-Poly From Roots time: {} ms", t.time_us/1000);
	}else{
		//3. CO-ordinate to build the distriuted vec. Check will be done
		//by main processor
		RUN_CONFIG.better_barrier("profile_dist_build_poly_from_roots2");
		let mut _d_p = DisPoly::<Fr>::dispoly_from_roots_in_file_worker(101, main_processor, &fname, log_2_round_size, dist_bar);
	}
}

/** profile key generation and polynomial eval 
	KeyGen: 1k - 20ms,  1M - 4s 
	PolyEval: 1k -80ms, 1M - 90s
*/
pub fn profile_key_and_eval(n: usize, cfg: &RunConfig){
	//1. generate key
	let g = DisKey::<PE>::gen_g(13572); 
	let h = DisKey::<PE>::gen_g(3723483917); 
	let h2 = DisKey::<PE>::gen_g2(3723483917); 
	let g2 = DisKey::<PE>::gen_g2(13572); 
	let alpha = Fr::from(223423492347u64);
	let beta= Fr::from(8293234237u64);
	let theta = Fr::from(72342342342u64);
	let s2= Fr::from(12312312387u64);
	let id = 11;
	let len = n+2;
	let mut t1 = Timer::new();
	t1.start();
	let key = DisKey::<PE>::gen(g, id, len, alpha, g2, beta, h, h2, theta, s2);
	t1.stop();
	if cfg.my_rank==0{
		println!("profile key gen size: {}, time: {} ms", n, t1.time_us/1000);
	}
	

	//2. generate polynomial
	let vec_f = rand_arr_field_ele::<Fr>(n, 17234098234209384u128);
	let mut p = DisVec::new_dis_vec_with_id(100, 1, n, vec_f);
	p.to_partitions(&RUN_CONFIG.univ);

	//3. evaluate
	let mut t = Timer::new();
	t.start();
	let _res1 = key.eval_poly(&mut p, false);	
	t.stop();
	if cfg.my_rank==0{
		//println!("profile poly eval size: {}, time: {} ms", n, t.time_us/1000);
		println!("profile poly eval size: {}, time: {} us", n, t.time_us);
	}
}

/** check the serial poly div and mul performance 
*/
pub fn profile_serial_div(n: usize, cfg: &RunConfig){
	if cfg.my_rank==0 {
		let mut rng = gen_rng();
		let f = DensePolynomial::<Fr>::rand(n, & mut rng);
		let g = DensePolynomial::<Fr>::rand(n/2, & mut rng);
	
		let mut t = Timer::new();
		t.start();
		let _ = &f * &f;
		t.stop();
		println!("profile poly mul: size: {}, time: {} us", n, t.time_us);
		//println!("profile poly mul: size: {}, time: {} ms", n, t.time_us/1000);

		t = Timer::new();
		t.start();
		let (_,_) = old_divide_with_q_and_r(&f, &g);
		t.stop();
		println!("profile old divide_with_q: size: {}, time: {} us", n, t.time_us);

		t = Timer::new();
		t.start();
		let (_,_) = new_divide_with_q_and_r(&f, &g);
		t.stop();
		println!("profile NEW divide_with_q: size: {}, time: {} us", n, t.time_us);
	}
	RUN_CONFIG.better_barrier("profile_serial_div");
}

/** compare small mul with typical DensePolynomial mul */
pub fn profile_small_mul(logn: usize, cfg: &RunConfig){
	if cfg.my_rank==0 {
		let mut n = 1;
		let mut rng = gen_rng();
		for _i in 0..logn{	
			let p1 = DensePolynomial::<Fr>::rand(n, & mut rng);
			let p2 = DensePolynomial::<Fr>::rand(n, & mut rng);
			let mut t1 = Timer::new();
			t1.start();
			let _r1 = &p1 * &p2;
			t1.stop();

			let mut t2 = Timer::new();
			t2.start();
			let _r2 = small_mul_poly(&p1, &p2);
			t2.stop();
			println!("n: {}, FFT mul: {} us, small_mul: {} us", n, t1.time_us, t2.time_us);

			n *= 2;
		}
	}
	RUN_CONFIG.better_barrier("profile_small_mul");
}

/** profile gcd 
1M: mul 4.2 sec, div: 12s, hgcd 600sec (almost 100 times tmies of mul)
*/
pub fn profile_hgcd(n: usize, cfg: &RunConfig){
	if cfg.my_rank==0 {
		let mut rng = gen_rng();
		let p1 = DensePolynomial::<Fr>::rand(n, & mut rng);
		let p2 = DensePolynomial::<Fr>::rand(n, & mut rng);

		let mut t3 = Timer::new();
		t3.start();
		let _res = feea(&p1, &p2);
		t3.stop();

		println!("profile size: {}, feea: {} ms", n, t3.time_us/1000);

	}
	RUN_CONFIG.better_barrier("profile hgcd");
}

/** profile sub, add, mul, division, and new_hgcd algorithm 
*/
pub fn profile_poly_ops(n: usize, _cfg: &RunConfig){
	let seed= 12328234u128;
	let mut dp1= DisPoly::<Fr>::gen_dp_from_seed(n, seed);
//	let mut dq1= DisPoly::<Fr>::gen_dp_from_seed(n, seed);
	let mut dp2= DisPoly::<Fr>::gen_dp_from_seed(n/2, seed+10123);
	dp1.to_partitions();
//	dq1.to_partitions();
	dp2.to_partitions();
//	let p1 = dp1.to_serial();
//	let p2 = dp2.to_serial();
	let np = RUN_CONFIG.n_proc;
	let me = RUN_CONFIG.my_rank;
	if me==0{
		println!(" ==== n: {}, np: {} ===", n, np);
	}

	let mut t2 = Timer::new();
	let mut t1 = Timer::new();
	t1.start();
	t2.start();

/*
	//1. sub
	if me==0{
		t1.clear_start();
		let _p3 = &p1 - &p2;
		t1.stop();
		println!("Serial PolySub: size: {}, time: {} us", n, t1.time_us/1000);
	}

	t2.clear_start();
	let _p3 = DisPoly::<Fr>::sub(&mut dp1, &mut dq1);
	t2.stop();
	if me==0{
		report("PolySub", n, t2.time_us/1000);
	}
	

	//2. mul 
	if me==0{
		t1.clear_start();
		let _p3 = &p1 * &p2;
		t1.stop();
		println!("Serial PolyMul : size: {}, time: {} us", n, t1.time_us/1000);
	}

	t2.clear_start();
	let _p3 = DisPoly::<Fr>::mul(&mut dp1, &mut dq1);
	t2.stop();
	if me==0{
		report("PolyMul", n, t2.time_us/1000);
	}
*/

	//2. div 
/*
	if me==0{
		t1.clear_start();
		let (_sp3, _pr) = new_divide_with_q_and_r(&p1, &p2);
		t1.stop();
		//print_poly("sp3:", &sp3);
		println!("Serial PolyDiv: size: {}, time: {} ms", n, t1.time_us/1000);
	}
*/

	t2.clear_start();
	let (_dp3, _pr) = DisPoly::<Fr>::divide_with_q_and_r(&mut dp1, &mut dp2);
	t2.stop();
	if me==0{
		report("PolyDiv", n, t2.time_us/1000);
		//print_poly("dp3:", &dp3.to_serial());
	}
}

/** profile distributed hgcd_worker
Serial version: 1M: mul 4.2 sec, div: 12s,
Distributed 8 nodes 1 computer: 335 s (similar to serial)
*/
pub fn profile_dis_hgcd_worker(n: usize, cfg: &RunConfig){
	let d = n;
	let seed = 192834u128;
	let mut a = DisPoly::<Fr>::gen_dp_from_seed(d, seed);
	let mut b = DisPoly::<Fr>::gen_dp_from_seed(d, seed+10123);
	let sa = a.to_serial();
	let sb = b.to_serial();
	let mut t1 = Timer::new();
	let mut t2 = Timer::new();
	t1.start();
	let _res1 = DisPoly::<Fr>::hgcd_worker(&mut a, &mut b, 1);
	t1.stop();
	t2.start();
	let _res2 = hgcd_worker(&sa, &sb);
	t2.stop();
	println!("Profile size: {}, distributed: {} ms, serial: {} ms, node: {}. NOTE: serial runs the best with 1 node!", n, t1.time_us/1000, t2.time_us/1000, cfg.n_proc);
	RUN_CONFIG.better_barrier("profile distributed hgcd");
}

// performance: (4 nodes 1 computer) - 8 nodes the same
// 1k: 2sec
// 16k: 6sec 
// 128k: 63sec
// 1M: 680 sec
pub fn profile_dis_feea(n: usize, _cfg: &RunConfig){
	let d = n;
	let seed = 192834u128;
	let mut a = DisPoly::<Fr>::gen_dp_from_seed(d, seed);
	let mut b = DisPoly::<Fr>::gen_dp_from_seed(d, seed+10123);
	let mut t1 = Timer::new();
	t1.start();
	let _res1 = DisPoly::<Fr>::feea(&mut a, &mut b);
	t1.stop();
	if RUN_CONFIG.my_rank==0{
		report("gcd", n, t1.time_us/1000);
	}
	RUN_CONFIG.better_barrier("profile distributed hgcd");
}

/** check the serial poly div and mul performance 
*/
pub fn profile_serial_ops(n: usize, cfg: &RunConfig){
	if cfg.my_rank==0 {
		let mut rng = gen_rng();
		let f = DensePolynomial::<Fr>::rand(n, & mut rng);
		let f2 = DensePolynomial::<Fr>::rand(n, & mut rng);
		let g = DensePolynomial::<Fr>::rand(n/2, & mut rng);
	
		let mut t1 = Timer::new();
		t1.clear_start();
		let _ = &f * &f2;
		t1.stop();
		report("serial_mul", n, t1.time_us/1000);

		t1.clear_start();
		let (_,_) = new_divide_with_q_and_r(&f, &g);
		t1.stop();
		report("serial_div", n, t1.time_us/1000);

		t1.clear_start();
		let (_,_,_) = feea(&f, &f2);
		t1.stop();
		report("serial_gcd", n, t1.time_us/1000);
	}
	RUN_CONFIG.better_barrier("profile_serial_ops");
}

/** collect all data */
pub fn data_profile_poly(n: usize){
	let cfg = &RUN_CONFIG;
	profile_poly_ops(n, cfg);
//	profile_dist_build_poly_from_roots(n, cfg);
//	profile_dis_feea(n, cfg);
}

pub fn data_profile_poly_mul(n: usize){
	let seed= 12328234u128;
	//let np = RUN_CONFIG.n_proc;
	let me = RUN_CONFIG.my_rank;
	let mut dp1= DisPoly::<Fr>::gen_dp_from_seed(n, seed);
	let mut dq1= DisPoly::<Fr>::gen_dp_from_seed(n, seed);
	dp1.to_partitions();
	dq1.to_partitions();
	let mut t2 = Timer::new();

	t2.clear_start();
	let _p3 = DisPoly::<Fr>::mul(&mut dp1, &mut dq1);
	t2.stop();
	if me==0{
		report("PolyMul", n, t2.time_us/1000);
	}
}

/// collect the data for the given operation
/// log_min_size, log_max_size, log_size_step controls the range and step
/// timeout in esconds
pub fn collect_poly_data(op: &str, log_min_size: usize, log_max_size: usize, log_size_step: usize, trials: usize, timeout: usize){
	let me = RUN_CONFIG.my_rank;
	let b_mem = true;
	let b_perf= true;
	let b_test = true;

	if trials<1 {panic!("collect_poly_data: trials should be >=1");}
	if me==0{
		println!("===== DATA for {}: log_min_size: {} -> log_max_size: {}, NODES: {} ========", op, log_min_size, log_max_size, RUN_CONFIG.n_proc);
		let g = <PE as PairingEngine>::G1Affine::prime_subgroup_generator();
		let mut b1:Vec<u8> = vec![];
		<PE as PairingEngine>::G1Affine::serialize(&g, &mut b1).unwrap();
		println!(" === G1 element size: {}", b1.len());
	}
	let cfg = &RUN_CONFIG;
	let nodes_file = "/tmp/tmp_nodelist.txt";
	let netarch = &get_network_arch(&nodes_file.to_string());
	let mut t = Timer::new();
	let seed = 234243234324u128;
	let min_size = 1<<log_min_size;
	let max_size = 1<<log_max_size;
	let mut size = min_size;
	let step = 1<<log_size_step;
	let np = RUN_CONFIG.n_proc;
	let abnormal_bar = 4;
	let mut prev_time = 10000000000000000000;
	while size<=max_size{
		let bskip_1st = size==min_size;
		log(LOG1, &format!("size: {} -----", size));
		let mut total_time = 0;
		let mut round_trials = if size==min_size {trials+1} else {trials};
		let mut actual_trials = 0;
		let mut i = 0;
		while i<round_trials{//note round_trials might be increased
			if op=="fft"{
				let arr = rand_arr_field_ele::<Fr>(size/np, seed + me as u128);
				let mut dv = DisVec::<Fr>::new_from_each_node(0, 0, size/np*np,arr);
				t.clear_start();
				distributed_dizk_fft(&mut dv, &cfg.univ);
				RUN_CONFIG.better_barrier("serial_div");
			}else if op=="mul"{
				let mut p = DisPoly::<Fr>::gen_dp_from_seed(size, seed);
				let mut p2 = DisPoly::<Fr>::gen_dp_from_seed(size, seed+1017);
				t.clear_start();
				let _p4 = DisPoly::<Fr>::mul(&mut p, &mut p2);
			}else if op=="div"{
				let mut p = DisPoly::<Fr>::gen_dp_from_seed(size, seed);
				let mut p3 = DisPoly::<Fr>::gen_dp_from_seed(size/2, seed+1017);
				t.clear_start();
				let _p5 = DisPoly::<Fr>::divide_with_q_and_r(&mut p, &mut p3);
			}else if op=="serial_div"{
				let sp = rand_poly::<Fr>(size, seed);
				let sp2 =rand_poly::<Fr>(size/2, seed+117);
				t.clear_start();	
				if me==0{
					let (_, _) = adapt_divide_with_q_and_r(&sp, &sp2);
				}
				RUN_CONFIG.better_barrier("serial_div");
			}else if op=="serial_binacc"{
				//size -1 to avoid overflow power of 2 boundary because
				//cof needs to be +1
				let arr = rand_arr_field_ele::<Fr>(size-1, seed + me as u128);
				RUN_CONFIG.better_barrier("wait writing is done");
				t.clear_start();
				let _p5 = DisPoly::<Fr>::binacc_poly(&arr);
			}else if op=="binacc"{
				let tmp_file = format!("/tmp/arr.dat");
				if me==0{
					//make size-np for avoiding overflow power of 2 
					//due to (coefs +1)
					let arr = rand_arr_unique_u64(size-np, 62, seed as u128);
					write_arr_with_size(&arr, &tmp_file);
				}
				RUN_CONFIG.better_barrier("wait writing is done");
				dump_mem_usage("BEFORE returning p5");
				t.clear_start();
				let _p5 = DisPoly::<Fr>::dispoly_from_roots_in_file_from_mainnode(0, 0, &tmp_file, netarch);
				//DisPoly::<Fr>::test_mem_leak(size);
				dump_mem_usage("AFTER returning p5");
				if me==0{
					remove_file(&tmp_file);
				}
				t.stop();
				RUN_CONFIG.better_barrier("wait remove is done");
			}else if op=="serial_gcd"{
				let sp = rand_poly::<Fr>(size, seed);
				let sp2 =rand_poly::<Fr>(size, seed+117);
				t.clear_start();	
				if me==0{
					let (_, _, _) = feea(&sp, &sp2);
				}
				RUN_CONFIG.better_barrier("serial_feea");
			}else if op=="gcd"{
				let mut p = DisPoly::<Fr>::gen_dp_from_seed(size, seed);
				let mut p2 = DisPoly::<Fr>::gen_dp_from_seed(size, seed+1799);
				//let mut p = DisPoly::<Fr>::gen_binacc_from_seed(size, seed);
				//let mut p2 = DisPoly::<Fr>::gen_binacc_from_seed(size, seed+22);
				t.clear_start();
				let (gcd, _, _) = DisPoly::<Fr>::feea(&mut p, &mut p2);
				if b_test{
					let pone = get_poly(vec![Fr::from(1u64)]); 
					let sgcd = gcd.to_serial();
					let sp1 = p.to_serial();
					let sp2 = p2.to_serial();
					if me==0{
						if sgcd!=pone{
							print_poly("sgcd:", &sgcd);
							print_poly("p1:", &sp1);
							print_poly("p2:", &sp2);
						}
						assert!(sgcd==pone, "gcd is not 1");
					}
				}
			}else if op=="gcd_new"{
				let arr = rand_arr_field_ele::<Fr>(size/np, seed + me as u128);
				let mut dv = DisVec::<Fr>::new_from_each_node(0, 0, size/np*np,arr);
				let mut p2 = DisPoly::<Fr>::gen_dp_from_seed(size, seed+1799);
				t.clear_start();
				let (gcd, _s, _t) = DisPoly::<Fr>::feea_new(&mut dv, &mut p2);
				if b_test{
					let pone = get_poly(vec![Fr::from(1u64)]); 
					let sgcd = gcd.to_serial();
					if me==0{
						if sgcd!=pone{
							print_poly("sgcd:", &sgcd);
						}
						assert!(sgcd==pone, "gcd is not 1");
					}
				}
			}else if op=="net"{//test speed of nonblock_broadcast
				let mut vsend = vec![vec![]; np];
				let unit_size = size/np;
				for i in 0..np{
					vsend[i] = rand_arr_field_ele::<Fr>(
						unit_size, seed + (me*np) as u128 +i as u128);
				}
				t.clear_start();
				let _res = nonblock_broadcast_old(&vsend, np as u64, 
						&RUN_CONFIG.univ, Fr::from(0u64));
			}else if op=="gen_key"{
				if b_mem {dump_mem_usage("BEFORE_GEN_KEY");}
				t.clear_start();	
				let _diskey = DisKey::<PE>::gen_key1(size);
				if b_mem {dump_mem_usage("AFTER_GEN_KEY");}
				RUN_CONFIG.better_barrier("dis_key");
			}else if op=="sub_key"{
				let diskey = DisKey::<PE>::gen_key1(size);
				if b_mem {dump_mem_usage("BEFORE_SUB_KEY");}
				t.clear_start();	
				let _skey2 = diskey.sub_key(size-1, 0u64);
				if b_mem {dump_mem_usage("AFTER_SUB_KEY");}
				RUN_CONFIG.better_barrier("dis_key");
			}else if op=="bubble_sort"{
				let mut arr = vec![0; size];
				for i in 0..size {arr[i] = size - i;}
				t.clear_start();	
				for i in 0..size{
					for j in 0..size-1{
						if arr[i]>arr[j]{
							let tmp = arr[i];
							arr[i] = arr[j];
							arr[j] = tmp;
						}
					}
				}	
				RUN_CONFIG.better_barrier("bubble_sort");
			}else if op=="vmsm"{//just to check scalability
				let g = DisKey::<PE>::gen_g(123u64); 
				let g1_proj = g.clone();
				let exp = rand_arr_field_ele(size, 123123u128);
				let bases = msm_g1::<PE>(g1_proj, &exp);
				t.clear_start();	
				let _res = vmsm_g1::<PE>(&bases, &exp);	
				
			}else if op=="groth16prove"{
				if b_mem {dump_mem_usage("-- BEFORE GenQAP");}
				let mut timer2 = Timer::new();
				timer2.start();
        		let n = size.next_power_of_two();
				let degree = n - 2; //degree+2 must be power of 2	
				let num_inputs = 2;
				let num_vars = n;
				let seed = 1122u128;
        		let (qap, qw) = DisQAP::<Fr>::rand_inst(seed, num_inputs, num_vars,degree, true);
				if b_perf {log_perf(LOG1, &format!(" -- Generating QAP of size: {}", n), &mut timer2);}
				if b_mem {dump_mem_usage("-- BEFORE Gen ProverKey");}
				let num_segs = qap.num_segs;
				let prover = DisProver::<PE>::new(num_segs, seed, qap.seg_size.clone());
				let diskey = DisKey::<PE>::gen_key1(np*2); 	
				let (dkey, vkey) = dis_setup(234234234u128, &qap, &diskey);
				if b_perf {log_perf(LOG1, &format!(" -- Set up ProverKey of size: {}", n), &mut timer2);}
				if b_mem {dump_mem_usage("-- BEFORE Prove");}
				if b_perf {log(LOG1, &format!("qw.len: {}, num_inputs: {}, num_vars: {}, degree: {}", qw.num_vars, num_inputs, num_vars, degree));}
				t.clear_start();	
				let p1 = prover.prove_stage1(&dkey, &qw, 1);
				let p2 = prover.prove_stage2(&dkey, &qw, 1);
				let bres = verify::<PE>(&p1, &p2, &vkey);
				if RUN_CONFIG.my_rank==0{
					assert!(bres==true, "verification failed");
					println!("DisProver verification passed!");
				};
				
				RUN_CONFIG.better_barrier("groth16");
				if b_mem {dump_mem_usage("-- ProveStage1and2");}
				if b_perf {log_perf(LOG1, &format!("ProveStage1 and 2: {}", n), &mut timer2);}
			}else{
				panic!("collect_poly_data do not support OP: {}", op);
			}
			t.stop();
			let time_ms = t.time_us/1000;
			log(LOG1, &format!("\t Round: {}, Time: {} ms", i, time_ms));

			//broadcast to all so that all get the same time
			let vecd = broadcast_small_arr(&vec![time_ms], 0);
			let time_ms = vecd[0];

			if !bskip_1st || i>0{
				//1. check if abnormal
				if time_ms>abnormal_bar * prev_time && 
					round_trials<trials+1{
					log(LOG1, &format!("\t Abnormal Data. Likely TCP reconn. Add 1 run"));
					round_trials += 1;
				}else{//if dropped one already, count it anyway
					total_time += time_ms;
					actual_trials += 1;
					prev_time = time_ms;
				}
			}else{
				log(LOG1, &format!("SKIP 1st record for discounting TCP connection establish time"));
			}
			i += 1;
			RUN_CONFIG.better_barrier("synch_round");
		}
		let avg_time = total_time/actual_trials;
		report(op, size, avg_time);
		if avg_time>timeout*1000 {
			if RUN_CONFIG.my_rank==0{
				println!("TIMEOUT for op: fft size: {}", size);
			}
			break;
		}
		size *= step;
	}
}
