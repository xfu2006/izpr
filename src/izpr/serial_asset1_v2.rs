/*
    Copyright Dr. Xiang Fu

    Author: Xiang Fu
    All Rights Reserved.
    Created: 09/05/2023
	Completed: 09/10/2023

	This files implements the zk-asset1 protocol
*/
extern crate ark_ff;
extern crate ark_ec;
extern crate ark_poly;
extern crate ark_serialize;
extern crate ark_std;
extern crate ark_bls12_381;
extern crate acc;
extern crate rayon;

use self::rayon::prelude::*; 
use self::acc::profiler::config::{LOG1};
use self::ark_ff::{Field,UniformRand,Zero,One,FftField};
use self::acc::poly::serial::{get_poly,adapt_divide_with_q_and_r};
use self::ark_poly::{Polynomial};
use izpr::serial_cq::{gen_seed,ped_commit,compute_coefs_of_lag_points};
use self::ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Read, SerializationError, Write};
use self::ark_ec::{AffineCurve, PairingEngine, ProjectiveCurve,msm::VariableBaseMSM};
use self::acc::tools::{gen_rng_from_seed, to_vecu8, hash, log_perf, Timer};
use izpr::serial_poly_utils::{compute_powers,vmsm};
use izpr::serial_pn_lookup::{PnLookupProverKey,PnLookupVerifierKey, setup_pn_lookup, prove_pn_lookup, verify_pn_lookup,PnLookupProof};


/// The Prover Key of Asset Protoocl 1
#[derive(Clone)]
pub struct AssetOneProverKey<PE:PairingEngine>{
	/// size of the asset table (lookup table, the N)
	pub lookup_size: usize,
	/// number of transactions in a cycle (query table, the n) 
	pub query_size: usize,
	/// prover key of pn_lookup argument
	pub pk_pn: PnLookupProverKey<PE>,
}

/// The Verifier Key
#[derive(Clone)]
pub struct AssetOneVerifierKey<PE:PairingEngine>{
	pub vk_pn: PnLookupVerifierKey<PE>,
}

/// Proof
#[derive(Clone,CanonicalSerialize,CanonicalDeserialize)]
pub struct AssetOneProof<PE:PairingEngine>{
	/// proof of pn_lookup
	pub prf_pn: PnLookupProof<PE>,
	/// commit to boolean array o
	pub com_o: PE::G1Affine,
	/// commit to polynomial q
	pub com_q: PE::G1Affine,
	/// commit to polynoimal u
	pub com_u: PE::G1Affine,
	/// proof for u(1) = 0
	pub kzg_u0: PE::G1Affine,
	/// u(t*omega_n)
	pub t_u2: PE::Fr,
	/// u(t)
	pub t_u: PE::Fr,
	/// o(t)
	pub t_o: PE::Fr,
	/// q(t)
	pub t_q: PE::Fr, 
	/// Delta(t)
	pub t_delta: PE::Fr, 
	/// Proof of batch kzg
	pub prf_batch_kzg: PE::G1Affine,
	/// proof for t_u2
	pub kzg_u2: PE::G1Affine,
	/// proof for evaluation of u(X) at omega^{n2-1} to v
	pub prf_v: PE::G1Affine,
	/// balance term for prf_v
	pub bal_prf_v: PE::G1Affine,
	/// message for DLOG proof for bal_prf_v
	pub msg1_q_v: PE::G1Affine,
	/// DLOG proof for bal_prf_v
	pub res1_q_v: PE::Fr,
	/// DLOG proof part 2 for bal_prf_v
	pub res2_q_v: PE::Fr
	
}


/// generate the prover and verifier keys
/// log_bound: the log of bound for elements, log_unit_bound: for
/// range proof based on lookup, lookup_size: the size of
/// lookup table, query_size: the size of query table,
/// sorted_elements: the set of sorted and distinct elements
pub fn setup_asset_one<PE:PairingEngine>(
	log_bound: usize,
	log_unit_bound: usize,
	lookup_size: usize,
	query_size: usize,
	sorted_eles: &Vec<PE::Fr>) 
	-> (AssetOneProverKey<PE>, AssetOneVerifierKey<PE>) where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField> {
	let (pk_pn, vk_pn) = setup_pn_lookup(log_bound, log_unit_bound, lookup_size,
		query_size, sorted_eles);
	let pk = AssetOneProverKey{lookup_size: lookup_size, query_size: query_size,
		pk_pn: pk_pn};
	let vk = AssetOneVerifierKey{vk_pn: vk_pn};

	(pk,vk)
}

/// prove the asset change is the claimed commit_total
/// note that the asset table of the organization is already
/// cotained in prover key.
/// r_a, r_c are the randoms used for generating commitments to
///  accounts and changes. r_total is the random nonce for 
///  the commitment to total
/// return (commit_total, proof)
/// ASSUMPTION: the accounts/changes are of length n2
/// and the LAST elements are both 1, assuming account ID 1
/// is never used.
/// return (commit_total_changes, commit_accounts, commit_changes, prf) 
/// commit_total_changes is the commitment to the total changes of
///   accounts owned by the prover.
/// commit_accounts and commit_changes are actually the data to be
/// released by the blockchain platform (it is a compact representation
/// of the array of accounts and changes in the entire transaction cycle. 
pub fn prove_asset_one<PE:PairingEngine>(
	pk: &AssetOneProverKey<PE>, 
	accounts: &Vec<PE::Fr>, 
	changes: &Vec<PE::Fr>, 
	r_total: PE::Fr)
	-> (PE::G1Affine, PE::G1Affine, PE::G1Affine, AssetOneProof<PE>) where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField> {
	//0. data check
	let (b_test, b_perf) = (true, true);
	let n2 = pk.query_size;
	let one = PE::Fr::one();
	let zero = PE::Fr::zero();
	assert!(accounts.len()==n2, "accounts.len: {} != n2: {}", 
		accounts.len(), n2);
	assert!(accounts.len()==changes.len(), "accounts.len!=changes.len ");
	assert!(accounts[accounts.len()-1]==one, "last ele of accounts!=1");
	assert!(changes[changes.len()-1]==one, "last ele of changes!=1");
	if b_perf {println!("=== asset1 prove ===");}
	let mut timer = Timer::new();
	


	//1. run the pn-setup proof
	let mut rng = gen_rng_from_seed(gen_seed());
	let r_o = PE::Fr::rand(&mut rng);
	let r_a = zero;
	let (com_acc, com_o, prf_pn, vec_o) = prove_pn_lookup(&pk.pk_pn,
		&accounts, r_a, r_o);
	if b_perf {log_perf(LOG1, "---- pn_lookup proof", &mut timer);}
	let mut v = zero;
	for i in 0..n2-1{
		v = v + vec_o[i] * changes[i];
	}
	let g1 = PE::G1Affine::prime_subgroup_generator();
	let com_v = (g1.mul(v) + pk.pk_pn.pk_kzg.zv_s1_n2.mul(r_total)).into_affine();
	if b_perf {log_perf(LOG1, "---- compute com_v", &mut timer);}

	//2. compute vector u 
	let mut vec_u = vec![PE::Fr::zero(); n2];
	for i in 0..n2-1{
		vec_u[i+1] = vec_u[i] + vec_o[i] * changes[i];
	}

	//3. compute u(X) and commit_u
	let (_, _, com_ch) = ped_commit(&changes, n2, &pk.pk_pn.pk_kzg.lag_all_n2, pk.pk_pn.pk_kzg.zv_s1_n2); 
	let mut coef_u = compute_coefs_of_lag_points(&vec_u, zero); 
	//make it 2-leakly
	let r_u1 = PE::Fr::rand(&mut rng);
	let r_u0 = PE::Fr::rand(&mut rng);
	let poly_u_old = get_poly(coef_u.clone());
	//append items (r_u1 * x  + r_u0)(x^n-1)
	// x^n+1: r_u1, x^n: r_u0, x^1: -r_u1, x^0: -r_u0
	assert!(coef_u.len()==n2+1, "coef_u.len() ! = n2+1");
	coef_u.append(&mut vec![r_u1]);
	coef_u[n2] += r_u0;
	coef_u[1] -= r_u1;
	coef_u[0] -= r_u0;
	assert!(coef_u.len()==n2+2, "coef_u.len !=n2+2");
	if b_perf {log_perf(LOG1, "---- compute com_u", &mut timer);}
	
	

	//4. compute q(X) which is degree n2 + 2 
	let omega = PE::Fr::get_root_of_unity(n2 as u64).unwrap();
	let omega_n_1 = omega.pow(&[(n2-1) as u64]);
	let coef_o = compute_coefs_of_lag_points(&vec_o, r_o);
	let mut arr_omega = compute_powers(n2, omega);
	arr_omega.append(&mut vec![one, omega]);
	assert!(coef_u.len()==arr_omega.len(), "len not equal");
	let coef_u2 = coef_u.clone().into_par_iter().zip(arr_omega).map(|(x,y)| x*y)
		.collect::<Vec<PE::Fr>>();
	let coef_delta = compute_coefs_of_lag_points(&changes, zero);

	let poly_u2 = get_poly(coef_u2);
	let poly_u = get_poly(coef_u.clone());
	let poly_o = get_poly(coef_o);
	let poly_delta = get_poly(coef_delta);
	let poly_1 = get_poly(vec![zero-one, one]);
	let poly_2 = get_poly(vec![zero-omega_n_1, one]);
	let poly_3 = &poly_1 * &poly_2;
	let lhs1 = &poly_u2 - &poly_u;
	let lhs2 = &poly_o * &poly_delta;
	let lhs = &(&lhs1-&lhs2) * &poly_3;
	let mut vec_zh = vec![zero; n2+1];
	vec_zh[0] = zero - one;
	vec_zh[n2] = one;
	let poly_zh = get_poly(vec_zh);
	let (poly_q, rem_q) = adapt_divide_with_q_and_r(&lhs, &poly_zh);  
	if b_test{ assert!(rem_q.is_zero(), "poly q failed!");}
	if b_perf {log_perf(LOG1, "---- compute q", &mut timer);}
	
	//5. compute com_q
	let coef_q = poly_q.coeffs.clone();
	let com_q = vmsm(&pk.pk_pn.pk_kzg.vec_g1, &coef_q);
	if b_perf {log_perf(LOG1, "---- commit q", &mut timer);}

	//5. compute random point t using Fiat-Shamir
	let com_u = vmsm(&pk.pk_pn.pk_kzg.vec_g1, &coef_u);
	let t = hash::<PE::Fr>(&to_vecu8(
		&vec![com_v, com_u, com_q, com_o]));

	//6. verify u(1) = 0
	let u0 = poly_u.evaluate(&one);
	if b_test{assert!(u0.is_zero(), "u0 != 0");}
	let (w_u0, rw_u0) = adapt_divide_with_q_and_r(&poly_u, &get_poly(vec![zero-one, one])); 
	let kzg_u0 = vmsm(&pk.pk_pn.pk_kzg.vec_g1, &w_u0.coeffs);
	if b_test{assert!(rw_u0.is_zero(), "error in gen kzg for u0");}
	if b_perf {log_perf(LOG1, "---- commit u", &mut timer);}
	

	//5. compute t_u2, t_u, t_o, t_delta, t_q
	let t_u2 = poly_u.evaluate(&(omega * t));
	let t_u = poly_u.evaluate(&t);
	let t_o = poly_o.evaluate(&t);
	let t_q = poly_q.evaluate(&t);
	let t_delta = poly_delta.evaluate(&t);

	//8. batch kzg proof for t_u, t_o, t_delta, t_q
	let eta = hash::<PE::Fr>(&to_vecu8(
		&vec![t, t_u2, t_u, t_o, t_delta]));
	let poly_lhs1 = poly_u.clone() + &poly_o * &get_poly(vec![eta]) + 
		&poly_delta * &get_poly(vec![eta*eta]) +
		&poly_q * &get_poly(vec![eta*eta*eta]);
	let eta_v = t_u + t_o * eta + t_delta * eta * eta + t_q*eta*eta*eta;
	let poly_lhs = &poly_lhs1 - &get_poly(vec![eta_v]);
	let poly_rhs = get_poly(vec![zero-t, one]); //x-t
	let (h_x, rem_hx) = adapt_divide_with_q_and_r(&poly_lhs, &poly_rhs);  
	if b_test {assert!(rem_hx.is_zero(), "rem_hx is zero()");}
	let prf_batch_kzg = vmsm(&pk.pk_pn.pk_kzg.vec_g1, &h_x.coeffs);

	//9. kzg proof for t_u2
	let (w_u2, rw_u2) = adapt_divide_with_q_and_r(&(&poly_u - &get_poly(vec![t_u2])), &get_poly(vec![zero-t*omega, one]));
	if b_test {assert!(rw_u2.is_zero(), "kzg_prf_u2 error");}
	let kzg_u2 = vmsm(&pk.pk_pn.pk_kzg.vec_g1, &w_u2.coeffs);
	if b_perf {log_perf(LOG1, "---- batch kzg", &mut timer);}

	//10. zk-proof for u evaluates to v at omega_{n2-1}
	let pk_kzg= &pk.pk_pn.pk_kzg;
	let poly_q_v_lhs = &poly_u_old - &get_poly(vec![v]);
	let poly_q_v_rhs = get_poly(vec![zero-omega_n_1, one]);
	let (poly_q_v, rem_q_v) = adapt_divide_with_q_and_r(&poly_q_v_lhs,
		&poly_q_v_rhs);
	if b_test {assert!(rem_q_v.is_zero(), "rem_q_v != 0");}	
	let r_q_v = PE::Fr::rand(&mut rng);
	let prf_v = vmsm(&pk.pk_pn.pk_kzg.vec_g1, &poly_q_v.coeffs) + pk_kzg.zv_s1_n2.mul(r_q_v).into_affine();
	let bal_v = pk_kzg.zv_s1_n2.mul(r_u0 - r_total).into_affine()
		+ pk_kzg.zv_s_s1_n2.mul(r_u1 - r_q_v).into_affine()
		+ pk_kzg.zv_s1_n2.mul(omega_n_1*r_q_v).into_affine();
	
	let ddlog_rand_v1= PE::Fr::rand(&mut rng);
	let ddlog_rand_v2= PE::Fr::rand(&mut rng);
	let msg1_q_v = pk_kzg.zv_s1_n2.mul(ddlog_rand_v1).into_affine() + 
		pk_kzg.zv_s_s1_n2.mul(ddlog_rand_v2).into_affine();
	let ver_ch_v= hash::<PE::Fr>(&to_vecu8(&vec![bal_v, msg1_q_v])); 
	let res1_q_v= ddlog_rand_v1- ver_ch_v*(r_u0-r_total+omega_n_1*r_q_v);
	let res2_q_v= ddlog_rand_v2- ver_ch_v*(r_u1 - r_q_v);
	if b_perf {log_perf(LOG1, "---- dlog ", &mut timer);}

	//11. generate the proof for y_t,2
	(com_v, com_acc, com_ch,
		AssetOneProof{prf_pn: prf_pn, com_o: com_o, com_q: com_q, com_u: com_u,
		kzg_u0: kzg_u0, t_u2: t_u2, t_u: t_u, t_o: t_o, t_q: t_q, t_delta: t_delta, prf_batch_kzg: prf_batch_kzg, kzg_u2: kzg_u2, prf_v: prf_v, 
bal_prf_v: bal_v, msg1_q_v: msg1_q_v, res1_q_v: res1_q_v, res2_q_v: res2_q_v
		})
} 

/// verify the proof (the total change is the sum of
///  those changes in accounts that belong to assets table
/// commit_assets: ALREADY in key as the commit of lookup table.
/// commit_accounts: commitment to the accounts in the cycle. the
///   random opening is 0 (it's not hiding).
/// commit-changes: commitment to the changes in the transaction cycle,
///   random opening is 0 (it's not hiding). commit_accounts and
///   commit_changes are usually released by blockchain at each cycle.
/// commit_total_change: Pedersen commitment to the total changes of
///   the accounts of the prover (that is accmulating the changes
///   of those accounts in arr_accounts that belong to arr_assets
pub fn verify_asset_one<PE:PairingEngine>(vk: &AssetOneVerifierKey<PE>,
	commit_accounts: PE::G1Affine, 
	commit_changes: PE::G1Affine, 
	commit_total_change: PE::G1Affine, 
	prf: &AssetOneProof<PE>)-> bool where
<<PE as PairingEngine>::G1Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G1Affine, Scalar=<<PE as PairingEngine>::G1Affine as AffineCurve>::ScalarField>,
 <<PE as PairingEngine>::G2Affine as AffineCurve>::Projective: VariableBaseMSM<MSMBase=PE::G2Affine, Scalar=<<PE as PairingEngine>::G2Affine as AffineCurve>::ScalarField> {
	//1. verify pn-lookup
	let one = PE::Fr::one();
	let b1 = verify_pn_lookup(&vk.vk_pn, commit_accounts, prf.com_o, &prf.prf_pn);

	//2. verify kzg_u0
	let s2 = vk.vk_pn.vk_kzg.s2;
	let g2 = PE::G2Affine::prime_subgroup_generator();
	let g1 = PE::G1Affine::prime_subgroup_generator();
	let lhs = PE::pairing(prf.com_u, g2);
	let rhs = PE::pairing(prf.kzg_u0, s2.into_projective() - g2.into_projective()); 
	let b2 = lhs == rhs;

	//3. verify the equation on u
	let t = hash::<PE::Fr>(&to_vecu8(
		&vec![commit_total_change, prf.com_u, prf.com_q, prf.com_o]));
	let n2 = vk.vk_pn.n2;
	let omega = PE::Fr::get_root_of_unity(n2 as u64).unwrap();
	let lhs = prf.t_u2 - prf.t_u - prf.t_o * prf.t_delta;
	let rhs = prf.t_q * (t.pow(&[n2 as u64]) - one)/((t-one)*(t-omega.pow(&[(n2-1) as u64])) );
	let b3 = lhs == rhs;

	//4. verify the batched proof for t_u, t_o, t_q, t_delta
	let eta = hash::<PE::Fr>(&to_vecu8(
		&vec![t, prf.t_u2, prf.t_u, prf.t_o, prf.t_delta]));
	let eta_v = prf.t_u + prf.t_o * eta + prf.t_delta * eta * eta + prf.t_q*eta*eta*eta;
	let lhs = prf.com_u + prf.com_o.mul(eta).into_affine()
		+ commit_changes.mul(eta*eta).into_affine() + 
		prf.com_q.mul(eta*eta*eta).into_affine();
	let gt_lhs  = PE::pairing(lhs.into_projective() - g1.mul(eta_v), g2);
	let gt_rhs = PE::pairing(prf.prf_batch_kzg, s2.into_projective() - 
		g2.mul(t));
	let b4 = gt_lhs ==gt_rhs;
	
	//5. verify kzg_u2
	let lhs = PE::pairing(prf.com_u.into_projective() - g1.mul(prf.t_u2), g2);
	let rhs = PE::pairing(prf.kzg_u2, s2.into_projective() - g2.mul(t*omega)); 
	let b5 = lhs == rhs;

	//6. verify the zk proof for v
	let omega_n_1 = omega.pow(&[(n2-1) as u64]);
	let lhs = PE::pairing(prf.com_u.into_projective()-commit_total_change.into_projective(), g2);
	let rhs = PE::pairing(prf.prf_v, s2.into_projective() - g2.mul(omega_n_1)) * PE::pairing(prf.bal_prf_v, g2); 
	let b6 = lhs == rhs;
	
	let vk_kzg = &vk.vk_pn.vk_kzg;
	let ver_ch_v= hash::<PE::Fr>(&to_vecu8(&vec![prf.bal_prf_v,prf.msg1_q_v])); 
	let b7 = prf.msg1_q_v ==
		(vk_kzg.zv_s1_n2.mul(prf.res1_q_v).into_affine() 
			+ vk_kzg.zv_s_s1_n2.mul(prf.res2_q_v).into_affine())
		+ prf.bal_prf_v.mul(ver_ch_v).into_affine();
	
	//println!("DEBUG USE 202: b1: {}, b2: {}, b3: {}, b4: {}, b5: {}, b6: {}, b7: {}", b1, b2, b3, b4, b5, b6, b7);

	b1 && b2 && b3 && b4 && b5 && b6 && b7
}

#[cfg(test)]
mod tests{
	extern crate ark_bls12_381;
	extern crate acc;
	extern crate ark_ec;
	extern crate ark_ff;
	use self::ark_bls12_381::Bls12_381;
	type Fr381 = ark_bls12_381::Fr;
	type PE381 = Bls12_381;
	type G1_381= ark_bls12_381::G1Affine;

	//use izpr::serial_poly_utils::{default_trapdoor, setup_kzg};
	use self::acc::tools::{rand_arr_field_ele,gen_rng_from_seed};
	use izpr::serial_cq::{ped_commit,gen_seed};
	use izpr::serial_rng::{rand_arr_fe,rand_arr_fe_with_seed};
	use izpr::serial_asset1_v2::{setup_asset_one, prove_asset_one, verify_asset_one};
	use izpr::serial_poly_utils::{vmsm,setup_kzg, default_trapdoor,precompute_lag_group};
	use self::ark_ec::{AffineCurve,ProjectiveCurve};
	use self::ark_ff::{Field,UniformRand,Zero,One};

	#[test]
	pub fn test_asset1(){
		let bits = 32; 
		let unit_bits = 8;
		let num_ele = 16;
		let n2 = 4;
		let trap = default_trapdoor::<PE381>();
		let seed = gen_seed();
		let mut t = rand_arr_fe::<Fr381>(num_ele-1, bits);
		t.sort();
		let mut query_table = rand_arr_fe_with_seed::<Fr381>(n2-1, 
			bits, gen_seed()+1);
		query_table[1] = t[3];
		let one = Fr381::one();
		query_table.append(&mut vec![one]);
		let (pk, vk) = setup_asset_one::<PE381>(bits,unit_bits, num_ele, n2, &t);
		let arr_r = rand_arr_fe::<Fr381>(3, bits);
		let r_total = arr_r[0];
		let mut arr_changes = rand_arr_fe::<Fr381>(n2-1, bits);
		arr_changes.append(&mut vec![one]);
		let (com_v, com_acc, com_ch, prf) = prove_asset_one(&pk, &query_table, 
			&arr_changes, r_total);
		let bres = verify_asset_one(&vk, com_acc, com_ch,
			com_v, &prf);
		assert!(bres, "asset1 failed");	
	}
}
