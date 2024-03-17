/*
    Copyright Dr. Xiang Fu

    Author: Xiang Fu
    All Rights Reserved.
    Created: 08/07/2023
	Completed: 08/09/2023

	This files provides logical/slow version of the functions in
	serial_poly_utils.rs (for unit testing purpose)
*/
extern crate ark_ff;
extern crate ark_ec;
extern crate acc;
use self::ark_ec::{AffineCurve, ProjectiveCurve};
use self::ark_ec::msm::{VariableBaseMSM};
use self::ark_ff::{Field, FftField, One, Zero};
//use self::acc::profiler::config::{LOG1};
//use self::acc::tools::{log_perf,Timer};


/// generate [\omega^0, .., \omega^{n-1}]
pub fn gen_root_unity_arr<F:FftField>(n: u64)->Vec<F>{
	let mut v = vec![];
	let omega =  F::get_root_of_unity(n).unwrap();
	let mut value = F::one();
	for _i in 0..n{
		v.push(value);
		value = value * omega;
	}
	return v;
}

/// evaluate the i'th Lagrange basis poly defined over arr
/// L_i(X) = \prod_{j\neq i} (X-a_j)  / \prod{j\neq i} (a_i-a_j)
/// evaluate L_i(x)
/// NOTE: i is in range [0,n-1]
/// slow version, only use it for debugging or testing
pub fn eval_lag_poly<F:Field>(i: usize, arr: &Vec<F>, x: F)->F{
	let mut prod1 = F::one();
	let mut prod2 = F::one();
	let a_i = arr[i];
	for j in 0..arr.len(){
		if j!=i{
			prod1 *= x - arr[j];
			prod2 *= a_i - arr[j];
		}
	}
	return prod1/prod2;
}

/// check kzg_key is generated using the trapdoor s
/// i.e., for each i: kzg_key[i] = [s^i]
pub fn check_kzg_key<G:AffineCurve>(kzg_key: &Vec<G>, s: G::ScalarField){
	let n = kzg_key.len()-1;  
	let g = G::prime_subgroup_generator();
	let mut cur_g = g.mul(s).into_affine();
	for i in 0..n+1{
		assert!(cur_g==kzg_key[i], "kzg_key err at idx {}", i);
		cur_g = cur_g + g;
	}
}

/// Treat vector as co-efficient array of a poly
/// More exactly T(X) = \sum_{i=0}^{n-1} t_i L_i(X)
/// evaluate T(s)
pub fn logical_eval_as_poly<F: FftField>(t: &Vec<F>, s: F, n: usize) -> F{
	assert!(n.is_power_of_two(), "n is not power of 2");
	let mut t_s = F::zero();
	let arr_roots = gen_root_unity_arr(n as u64);
	let mut pow_s = F::one();
	for i in 0..t.len(){
		t_s += t[i] * eval_lag_poly(i, &arr_roots, s);
		pow_s *= s;	
	}
	t_s
}

/// Let T(X) = \sum_{i=0}^{n-1} t_i L_i(X) + r*z_V(X)
/// evaluate T(s)
pub fn logical_eval_as_poly_ext<F:FftField>(t: &Vec<F>, r: F, s: F, n: usize) -> F{
	let part1 = logical_eval_as_poly(t, s, n);
	let part2 = r*(s.pow(&[n as u64]) - F::one());
	part1 + part2
}
/// logical version of precompute_kzg_proof:
/// for each i in [0, n-1]: compute
/// [ (T(s) - t_i))/(s - \omega^i)] 
/// n has to be a power of 2
/// define T(X) = \sum_{i=0}^{n-1} t_i L_i(X)
pub fn logical_precompute_kzg_proof<G:AffineCurve>(
	t: &Vec<G::ScalarField>, 
	n: usize, 
	s: G::ScalarField) 
	-> Vec<G>{
	assert!(n.is_power_of_two(), "n is not power of 2");
	assert!(t.len()<=n+1, "t.len() > n+1");

	let t_s = logical_eval_as_poly(t, s, n);
	let g = G::prime_subgroup_generator();
	let mut res:Vec<G> = vec![];
	let omega =  G::ScalarField::get_root_of_unity(n as u64).unwrap();
	let mut cur_omega = G::ScalarField::one(); //omega^0
	for i in 0..n{
		let ts_v = t_s - t[i];
		let item = ts_v / (s - cur_omega);
		cur_omega *= omega;
		let g_item = g.mul(item).into_affine();
		res.push(g_item);
	}

	res
}

/// given n is fft domain size, compute for all i in [0, n)
/// L_i(s) where L_i(x) is the Lagrange poly for the FFT domain [0,n)
pub fn logical_precompute_lag_group<G:AffineCurve>(
	n: usize, s: G::ScalarField) -> Vec<G>{
	assert!(n.is_power_of_two(), "n is not power of 2");
	let mut res = vec![];
	let arr = gen_root_unity_arr(n as u64);
	let g = G::prime_subgroup_generator();
	for i in 0..n{
		let item = eval_lag_poly(i, &arr, s); 
		let g_item = g.mul(item).into_affine();
		res.push(g_item);
	}
	res
}

/// copute all L_i(0) for i in [0, n)
pub fn logical_precompute_lag0_fe<F:FftField>(n: usize)->Vec<F>{
	assert!(n.is_power_of_two(), "n is power of 2");
	let arr = gen_root_unity_arr(n as u64);
	let zero = F::zero();
	let mut res = vec![zero; n];
	for i in 0..n{
		res[i] = eval_lag_poly(i, &arr, zero); 
	}

	res
}

/// compute all [(L_i(s)-L_i(0))/s] for each i, logical version for testing
pub fn logical_precompute_lag_diff<G:AffineCurve>(
	n: usize, 
	s: G::ScalarField)
	->Vec<G> where <G as AffineCurve>::Projective: VariableBaseMSM<MSMBase=G, Scalar=<G as AffineCurve>::ScalarField>{
	let mut res = vec![];
	let arr = gen_root_unity_arr(n as u64);
	let g = G::prime_subgroup_generator();
	for i in 0..n{
		let li_s = eval_lag_poly(i, &arr, s); 
		let li_0 = eval_lag_poly(i, &arr, G::ScalarField::zero()); 
		let item = (li_s - li_0)/s;
		let g_item = g.mul(item).into_affine();
		res.push(g_item);
	}
	res
}

/// compute the quotient polys used in CQ paper
/// Q_i(x) = Z_v'(\omega^i)^-1 kzg_prf_i(x)
pub fn logical_precompute_quotient_poly<G:AffineCurve>(
	t: &Vec<G::ScalarField>,
	n: usize,
	s: G::ScalarField) -> Vec<G>{
	//1. compute all kzg_proof for table t
	assert!(n.is_power_of_two(), "n is not power of 2");
	let kzg_prfs: Vec<G> = logical_precompute_kzg_proof(t, n, s);

	//2. compute z_v'(\omega^i)^-1
	//note that z_v(X) = X^n-1 and z_v'(X) = nx^{n-1} 
	let mut res = vec![];
	let omega =  G::ScalarField::get_root_of_unity(n as u64).unwrap();
	let mut omega_i = G::ScalarField::one();
	let one = G::ScalarField::one();
	let fe_n = G::ScalarField::from(n as u64);
	for i in 0..n{
		let z_der_omega_i = fe_n * omega_i.pow(&[(n-1) as u64]);
		let inv_z_der_omega_i = one/z_der_omega_i;
		let qi_s = kzg_prfs[i].mul(inv_z_der_omega_i).into_affine();
		res.push(qi_s);
		omega_i *= omega;
	}
	res
}

/// logical version of precompute_kzg_proof in field elements:
/// for each i in [0, n-1]: compute
///  (T(s) - t_i))/(s - \omega^i) 
/// n has to be a power of 2
/// define T(X) = \sum_{i=0}^{n-1} t_i L_i(X)
pub fn logical_precompute_kzg_proof_fe<F:FftField>(
	t: &Vec<F>, 
	n: usize, 
	s: F) 
	-> Vec<F>{
	assert!(n.is_power_of_two(), "n is not power of 2");
	assert!(t.len()<=n+1, "t.len() > n+1");

	let t_s = logical_eval_as_poly(t, s, n);
	let mut res:Vec<F> = vec![];
	let omega =  F::get_root_of_unity(n as u64).unwrap();
	let mut cur_omega = F::one(); //omega^0
	for i in 0..n{
		let ts_v = t_s - t[i];
		let item = ts_v / (s - cur_omega);
		cur_omega *= omega;
		res.push(item);
	}

	res
}

/// precompute the all quotient poly in the field elements
pub fn logical_precompute_quotient_poly_fe<F:FftField>(
	t: &Vec<F>,
	n: usize,
	s: F) -> Vec<F>{
	//1. compute all kzg_proof for table t
	assert!(n.is_power_of_two(), "n is not power of 2");
	let kzg_prfs: Vec<F> = logical_precompute_kzg_proof_fe(t, n, s);

	//2. compute z_v'(\omega^i)^-1
	//note that z_v(X) = X^n-1 and z_v'(X) = nx^{n-1} 
	let mut res = vec![];
	let omega =  F::get_root_of_unity(n as u64).unwrap();
	let mut omega_i = F::one();
	let one = F::one();
	let fe_n = F::from(n as u64);
	for i in 0..n{
		let z_der_omega_i = fe_n * omega_i.pow(&[(n-1) as u64]);
		let inv_z_der_omega_i = one/z_der_omega_i;
		let qi_s = kzg_prfs[i] * (inv_z_der_omega_i);
		res.push(qi_s);
		omega_i *= omega;
	}
	res
}


#[cfg(test)]
mod tests{
	extern crate ark_bls12_381;
	extern crate acc;
	use self::ark_bls12_381::Bls12_381;
	type Fr381 = ark_bls12_381::Fr;
	type PE381 = Bls12_381;
	type G1_381= ark_bls12_381::G1Affine;

	use izpr::serial_logical_poly_utils::*;
	use self::ark_ff::{FftField, PrimeField, One, Zero};
	use self::acc::tools::{rand_arr_field_ele};
	use izpr::serial_poly_utils::GLOBAL_TRAPDOOR_S;
	use izpr::serial_cq::{gen_seed};
	

	#[test]
	pub fn test_logical_eval_lag_poly(){
		let n = 32usize;
		let roots= gen_root_unity_arr::<Fr381>(n as u64);
		for i in 0..n{
			let a_i = roots[i];
			for j in 0..n{
				let res = eval_lag_poly(j, &roots, a_i);
				if i==j{assert!(res==Fr381::one(), "res==1 failed");}
				else{assert!(res==Fr381::zero(), "res==0 failed");}
			}
		}
	}

	#[test]
	pub fn test_logical_precompute_all_kzg(){
		let n = 32usize;
		let seed = gen_seed();
		let t = rand_arr_field_ele::<Fr381>(n, seed);
		let s = Fr381::from(GLOBAL_TRAPDOOR_S);
		let all_kzg_prfs = logical_precompute_kzg_proof::<G1_381>(&t, n, s);
		let omega =  Fr381::get_root_of_unity(n as u64).unwrap();
		let mut cur_omega = Fr381::one();
		let g = G1_381::prime_subgroup_generator();
		for i in 0..n{
			let prf_i = all_kzg_prfs[i];
			let f_val = logical_eval_as_poly(&t, s, n);
			let o_val = t[i];
			let exp_val = (f_val - o_val)/(s - cur_omega);
			cur_omega *= omega;
		
			let g_exp_eval = g.mul(exp_val).into_affine();
			assert!(g_exp_eval==prf_i, "failed at index: {}", i);	
		}
	}

	#[test]
	pub fn test_logical_precompute_quotient_poly(){
		//idea: test Li(s) T(s) = Ti Li(s) + z_V(s) Q_i(s)
		let n = 32usize;
		let seed = gen_seed();
		//let t = rand_arr_field_ele::<Fr381>(n, seed);
		let mut t = vec![Fr381::zero(); n];
		t[0] = Fr381::one();
		let s = Fr381::from(GLOBAL_TRAPDOOR_S);
		let arr_qi = logical_precompute_quotient_poly::<G1_381>(&t, n, s);
		let arr_lag = logical_precompute_lag_group::<G1_381>(n, s);
		let g = G1_381::prime_subgroup_generator();
		let t_s = logical_eval_as_poly(&t, s, n);
		for i in 0..n{
			let g1_li_s: G1_381 = arr_lag[i];
			let lhs = g1_li_s.mul(t_s).into_affine();

			let ti_li_s = g1_li_s.mul(t[i]).into_affine();
			//z_V(s) = s^n - 1
			let zvs = s.pow(&[n as u64]) - Fr381::one();
			let qi_s = arr_qi[i];
			let z_vs_qi_s = qi_s.mul(zvs).into_affine();
			let rhs = ti_li_s + z_vs_qi_s;
			assert!(lhs==rhs, "test logical_quotient_poly failed at {}", i);
		}	
	}
}
