/** 
	Copyright Dr. Xiang Fu

	Author: Dr. Xiang Fu
	All Rights Reserved.
	Created: 07/19/2022
	Modified: 09/18/2022 -> added MSM profileing
*/

/**! This is the profiler function for profiling pairing/field/group
operations.
*/
extern crate ark_poly;
extern crate ark_ec;
extern crate ark_ff;
extern crate mpi;
extern crate ark_bn254;
extern crate ark_bls12_381;

//use std::borrow::Borrow;
//use std::rc::Rc;
use crate::profiler::config::*;
use self::ark_ec::{AffineCurve, PairingEngine, ProjectiveCurve};
//use self::ark_ff::{Field,PrimeField,FftField,Zero,biginteger::BigInteger};
use self::ark_ff::UniformRand;
use self::ark_ec::msm::{VariableBaseMSM};
use tools::{Timer,gen_rng};
use std::marker::PhantomData;

use self::ark_bn254::Bn254;
type Fr254 = ark_bn254::Fr;
type PE254= Bn254;
use self::ark_bls12_381::Bls12_381;
type Fr381 = ark_bls12_381::Fr;
type PE381= Bls12_381;

/// just to remove some warning
fn phantom_func(){
	let _d1: PhantomData<Fr254>;
	let _d2: PhantomData<PE254>;
	let _d3: PhantomData<Fr381>;
	let _d4: PhantomData<PE381>;
}

/// profile a group
pub fn profile_pairing<E:PairingEngine>(name: &str, n: usize, cfg: &RunConfig)
where
<<E as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=E::G1Affine, Scalar=<<E  as PairingEngine>::G1Affine as AffineCurve>::ScalarField> {
	phantom_func();
	if cfg.my_rank!=0 {cfg.better_barrier("profile_pair"); return;}
	println!("===========\n profiling pairing: {}\n=============", name);
	let mut rng = gen_rng();
	let g1 = E::G1Affine::rand(&mut rng);
	let g1_2 = E::G1Affine::rand(&mut rng);
	let g1p = g1.into_projective();
	let g1_2p = g1_2.into_projective();
	let g2 = E::G2Affine::rand(&mut rng);
	let g2_2 = E::G2Affine::rand(&mut rng);
	let g2p = g2.into_projective();
	let g2_2p = g2_2.into_projective();
	let gt = E::Fqk::rand(&mut rng);
	let gt_2 = E::Fqk::rand(&mut rng);
	let r: E::Fr= E::Fr::rand(&mut rng);
	let fr0 = E::Fr::from(0u64);
	let bn = 128;	
	let mut arr_base = vec![g1; bn];
	let mut arr_coefs = vec![fr0; bn];
	for i in 0..bn{
		let r1: E::Fr= E::Fr::rand(&mut rng);
		arr_base[i] = g1.mul(r1).into_affine();
		arr_coefs[i] = E::Fr::rand(&mut rng); 
	}

	let mut t = Timer::new();
	t.start();
	for _i in 0..n{
		let _g1_3 = g1 + g1_2;	
	}
	t.stop();
	println!("G1AffineAdd avg: {} us, n: {}, total: {}us", t.time_us/n, n, t.time_us);

	let mut t = Timer::new();
	t.start();
	for _i in 0..n{
		let _g1_3 = g1p + g1_2p;	
	}
	t.stop();
	println!("G1Proj avg: {} us, n: {}, total: {}us", t.time_us/n, n, t.time_us);
	let mut t = Timer::new();
	t.start();
	for _i in 0..n{
		let _g1_3 = g1.mul(r);
	}
	t.stop();
	println!("G1Affine mul: {} us, n: {}, total: {}us", t.time_us/n, n, t.time_us);

	let mut t = Timer::new();
	t.start();
	let mut res = g1p + g1p;
	for _i in 0..n{
		let tres: _= <E::G1Projective as VariableBaseMSM>::msm(
            &arr_base[0..bn],
            &arr_coefs[0..bn]
		);
		res = tres;
	}
	t.stop();
	let mut res1 = g1p;
	for i in 0..bn{
		let tres = arr_base[i].mul(arr_coefs[i]);
		res1 = if i==0 {tres} else {res1 + tres};
	}	
	assert!(res==res1, "MUL error: res!=res1");
	println!("G1MSM  mul avg: {} us, n: {}, total: {}us, base_len: {}", t.time_us/(n*bn), n, t.time_us, bn);

	let mut t = Timer::new();
	t.start();
	for _i in 0..n{
		let _g2_3 = g2 + g2_2;	
	}
	t.stop();
	println!("G2AffineAdd avg: {} us, n: {}, total: {}us", t.time_us/n, n, t.time_us);

	let mut t = Timer::new();
	t.start();
	for _i in 0..n{
		let _g2_3 = g2p + g2_2p;	
	}
	t.stop();
	println!("G2Proj avg: {} us, n: {}, total: {}us", t.time_us/n, n, t.time_us);
	let mut t = Timer::new();
	t.start();
	for _i in 0..n{
		let _g2_3 = g2.mul(r);
	}
	t.stop();
	println!("G2Affine mul: {} us, n: {}, total: {}us", t.time_us/n, n, t.time_us);

	let mut t = Timer::new();
	t.start();
	for _i in 0..n{
		let _gt_3 = gt + gt_2;	
	}
	t.stop();
	println!("GtAffineAdd avg: {} us, n: {}, total: {}us", t.time_us/n, n, t.time_us);

	let mut t = Timer::new();
	t.start();
	for _i in 0..n{
		let _gt_3 = E::pairing(g1, g2);
	}
	t.stop();
	println!("Pairing: {} us, n: {}, total: {}us", t.time_us/n, n, t.time_us);

	cfg.better_barrier("profile_pairing");
} 
