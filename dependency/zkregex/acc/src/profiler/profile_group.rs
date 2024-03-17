/** 
	Copyright Dr. Xiang Fu

	Author: Dr. Xiang Fu
	All Rights Reserved.
	Created: 01/01/2023
	This represent profile operations related to basic field and group operations
*/

extern crate ark_ff;
extern crate ark_ec;
extern crate ark_serialize;

use self::ark_ec::{AffineCurve, PairingEngine, ProjectiveCurve};
use self::ark_ec::msm::{VariableBaseMSM,FixedBase};
use self::ark_ff::{UniformRand,PrimeField};
//use self::ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use profiler::*;
use tools::*;
use poly::dis_key::*;

#[cfg(feature = "parallel")]
use ark_std::cmp::max;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/**
	DATA: (compared with ref1: https://hackmd.io/@gnark/eccbench 
		BN254: 
			field_mul: 1M 27ms -> 27ns
			g1_mul: 1M 5.4 sec (two group elements add) -> 5.4 us 
				!!! (380ns in ref1. Seems 14 times slower!)
			g1_mul_proj: 1M 412ms (412 ns. Now matching!)
			 -> g1_exp 256bit -> 412ns * 256 = 100 us
			g2_mul: 1M 6.6 sec  -> 6.6us
				!!! (1.2 us in ref1. Seems 5 times slower!)
			g2_mul_proj: 1M 1.3 ms -> 1.3 us now matching!
			pairing: 1k 998ms-> 1msms 
			g1_msm: 1M 6 sec (16x FASTER than 100us * 1M = 100 sec)
			g2_msm: 1M 21 sec (15x FASTER due to Pippener)

		BLS381:
			field_mul: 1M 27ms -> 27ns (SINGLE thread)
			g1_mul: 1M 12 sec -> 12 us
				!!! (670 ns seems 18 times slower!)
		 	g1_mul_proj: 1M 707ms (707ns ok!)
			g2_mul: 1M 13.8 sec -> 13.8us
				!!! (2.4us in ref1 --->  5.75 times slower!)
			g2_mul_proj: 1M 2438ms (2.4 usOK!)
			pairing: 1k 1228ms -> 1.28ms 
				(ref1: 1.2ms, ok)
			g1_msm: 1M 11 sec (almost 1x slower than BN254)
			g2_msm: 1M 35 sec.
			g1_fixed_msm: 1M 15 sec
			g2_fixed_msm: 1M 40 sec
			g1_msm from 2^10 to 2^20
			23, 42, 81, 131, 241		452, 845, 1528, 2971, 5770, --> 11sec  

			gen_key: 1M -> 400MB RAM -> peak memory consumption 650M
			HOWEVER, why it's NOT rounded to powers of 2,
				the peak memory consumption could be 650x2 = 1300x!!!

*/
pub fn collect_group_data<PE:PairingEngine>(op: &str, log_min_size: usize, 
	log_max_size: usize, log_size_step: usize, 
	trials: usize, _timeout: usize)
  where
 <<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField>{
	if trials<1 {panic!("collect_groups: trials should be >=1");}
	let mut t = Timer::new();
	t.start();
	let seed = 234243234324u128;
	let min_size = 1<<log_min_size;
	let max_size = 1<<log_max_size;
	let mut size = min_size;
	let step = 1<<log_size_step;
	let mut rng = gen_rng_from_seed(seed);
	let g = PE::G1Affine::rand(&mut rng).into_projective();
	let g2 = PE::G2Affine::rand(&mut rng).into_projective();
	let alpha = PE::Fr::rand(&mut rng);
	let one = PE::Fr::from(1u64);
	let base = 7098123123123u64;

	while size<=max_size{
		let bskip_1st = size==min_size;
		println!("---op: {},  size: {} -----", op, size);
		let mut total_time = 0;
		let round_trials = if size==min_size {trials+1} else {trials};
		for i in 0..round_trials{ 
			if op=="field_mul"{
				let arr = rand_arr_field_ele::<PE::Fr>(2, seed);
				let mut res = arr[0];
				let v2 = arr[1]; 
				t.clear_start();
				for _x in 0..size{
					res = res * v2;
				}
			}else if op=="serialize_field"{
				let arr = rand_arr_field_ele::<PE::Fr>(2, seed);
				t.clear_start();
				let _res = to_vecu8(&arr);
			}else if op=="deserialize_field"{
				let arr = rand_arr_field_ele::<PE::Fr>(2, seed);
				let res = to_vecu8(&arr);
				t.clear_start();
				let _res = from_vecu8(&res, arr[0]);
			}else if op=="serialize_g1"{
				let arr = vec![g.into_affine(); size];
				t.clear_start();
				let _res = to_vecu8(&arr);
			}else if op=="deserialize_g1"{
				let arr = vec![g.into_affine(); size];
				let res = to_vecu8(&arr);
				t.clear_start();
				let arr2 = from_vecu8(&res, arr[0]);
				assert!(arr2[0]==arr[0], "arr2[0]!=arr[0]");
			}else if op=="msm_g1"{//multi-scalar-multiplication (exponentiation)
				let arr_e = rand_arr_field_ele::<PE::Fr>(size, seed);
				let arr_g1:Vec<PE::G1Affine> = gen_powers::<PE>(g, alpha, one, base, size);
				t.clear_start();
				let _res: _ = multi_scalar_mul::<PE::G1Affine>(&arr_g1, &arr_e); 	
			}else if op=="msm_g2"{//multi-scalar-multiplication (exponentiation)
				let arr_e = rand_arr_field_ele::<PE::Fr>(size, seed);
				let arr_g = gen_powers_g2::<PE>(g2, alpha, one, base, size);
				t.clear_start();
				let _res = multi_scalar_mul::<PE::G2Affine>(&arr_g, &arr_e); 	
			}else if op=="fsm_g1"{//multi-scalar-multiplication (exponentiation)
				let arr_alpha= rand_arr_field_ele::<PE::Fr>(size, seed);
				t.clear_start();
				let window_size = FixedBase::get_mul_window_size(size+1);
				let scalar_bits = PE::Fr::MODULUS_BIT_SIZE as usize;
				let g_table = FixedBase::get_window_table(scalar_bits, window_size, g);
				let _vec1 = FixedBase::msm::<PE::G1Projective>(
            		scalar_bits,
            		window_size,
            		&g_table,
            		&arr_alpha,
				);
			}else if op=="fsm_g2"{//multi-scalar-multiplication (exponentiation)
				t.clear_start();
				let _arr_g = gen_powers_g2::<PE>(g2, alpha, one, base, size);
			}else if op=="g1_mul"{
				let mut g1 = PE::G1Affine::rand(&mut rng);
				let g2 = PE::G1Affine::rand(&mut rng);
				t.clear_start();
				for _x in 0..size{
					g1 = g1 + g2;
				}
			}else if op=="g1_mul_proj"{
				let mut g1 = PE::G1Affine::rand(&mut rng).into_projective();
				let g2 = PE::G1Affine::rand(&mut rng).into_projective();
				t.clear_start();
				for _x in 0..size{
					g1 = g1 + g2;
				}
			}else if op=="g2_mul"{
				let mut g1 = PE::G2Affine::rand(&mut rng);
				let g2 = PE::G2Affine::rand(&mut rng);
				t.clear_start();
				for _x in 0..size{
					g1 = g1 + g2;
				}
			}else if op=="g2_mul_proj"{
				let mut g1 = PE::G2Affine::rand(&mut rng).into_projective();
				let g2 = PE::G2Affine::rand(&mut rng).into_projective();
				t.clear_start();
				for _x in 0..size{
					g1 = g1 + g2;
				}
			}else if op=="pair"{
				let g1 = PE::G1Affine::rand(&mut rng);
				let g2 = PE::G2Affine::rand(&mut rng);
				t.clear_start();
				for _x in 0..size{
					let _res = PE::pairing(g1, g2);
				}
			}else{
				panic!("collect_groups do not support OP: {}", op);
			}
			t.stop();
			let time_ms = t.time_us/1000;
			println!("\t Round: {}, Time: {} ms", i, time_ms);
			if !bskip_1st || i>0{
				total_time += time_ms
			}
		}
		let avg_time = if bskip_1st {total_time/(round_trials-1)} else
				{total_time/round_trials};
		single_report(op, size, avg_time);
		size *= step;
	}//end while
}
